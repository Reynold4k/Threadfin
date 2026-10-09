"""Clone-level statistics with permutation nulls that respect the sampling design.

Two principles run through this module:

1. **The clone is the unit of replication.** Cells of one clone share their
   ancestry, so testing at cell level counts one expanded clone thousands of
   times (pseudoreplication) and makes p-values meaningless. Every test here
   counts clones.
2. **Permute within strata.** Clonally related cells are usually sampled
   together (same donor, sample, tissue, timepoint or sort gate). A null that
   ignores this attributes sample composition to clonality. Labels are
   therefore shuffled only among cells (or clones) that share a stratum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import bh_fdr, codes, get_uns, log, perm_pvalue, permute_within, require_obs, require_complete_groups, require_positive_int
from .profiles import get_profiles, model_from_adata, variance_components


# --------------------------------------------------------------------------- clonal coherence


def clonal_coherence(
    adata,
    *,
    strata_key: str | None = None,
    n_perm: int = 200,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """Test whether clone identity explains transcriptional state.

    The statistic is the clonal ICC (fraction of embedding variance explained
    by clone, see :mod:`threadfin.profiles`). The null distribution comes from
    shuffling clone labels among BCR+ cells *within* ``strata_key`` (for
    example sample, or donor x tissue x timepoint), which keeps each
    stratum's cell-state composition and clone-size distribution intact and
    destroys only the link between a cell's clone and its state.

    Parameters
    ----------
    strata_key
        ``obs`` column defining permutation strata. Defaults to the
        ``context_key`` used for :func:`threadfin.tl.clone_profiles`; with
        neither, labels are shuffled globally (not recommended).
    n_perm
        Number of permutations.

    Returns
    -------
    dict with the observed ``icc``, null mean/sd/95th percentile,
    ``excess`` (observed - null mean), ``p_value`` and bookkeeping; also stored
    in ``adata.uns['threadfin']['coherence']``.
    """
    require_positive_int(n_perm, "n_perm")
    prof = get_profiles(adata)
    # the ICC is always measured on unsmoothed cells in the embedding itself, so it reads
    # as "share of transcriptional variance explained by clone" whatever the profile type
    model = model_from_adata(adata, unsmoothed=True, representation="mean")
    strata_key = strata_key or prof["params"].get("context_key")
    strata = None
    if strata_key is not None:
        require_complete_groups(adata, strata_key, context="coherence permutations")
        strata, _ = codes(adata.obs[strata_key])

    observed = model.vc.icc
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = permute_within(model.clone_codes, strata, rng)
        null[b] = variance_components(model.residuals, perm).icc
    res = {
        "icc": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if n_perm > 1 else float("nan"),
        "null_q95": float(np.quantile(null, 0.95)),
        "excess": float(observed - null.mean()),
        "p_value": perm_pvalue(observed, null, "greater"),
        "n_perm": int(n_perm),
        "strata_key": strata_key,
        "n_clones": int(model.vc.n_clones),
        "n_cells": int(model.vc.n_cells),
        "icc_per_feature": model.vc.icc_per_feature,
        "null": null,
    }
    get_uns(adata)["coherence"] = res
    log(
        f"clonal coherence: ICC {observed:.3f} vs {res['null_mean']:.3f} under within-"
        f"'{strata_key}' permutation (p = {res['p_value']:.3g}, {n_perm} permutations).",
        verbose,
    )
    return res


# --------------------------------------------------------------------------- clone-level labels


def clone_labels(
    adata,
    label_key: str,
    *,
    clone_key: str | None = None,
    how: str = "auto",
    min_frac: float = 0.5,
) -> pd.Series:
    """Collapse a cell-level ``obs`` column to one value per clone.

    Parameters
    ----------
    how
        ``"majority"``: modal value, kept only if it covers at least
        ``min_frac`` of the clone's labelled cells (else NaN);
        ``"mean"`` / ``"median"``: for numeric columns;
        ``"fraction:<level>"``: fraction of the clone's labelled cells equal
        to ``<level>`` (numeric output);
        ``"auto"``: ``"mean"`` for numeric columns, else ``"majority"``.
    """
    clone_key = clone_key or get_profiles(adata)["params"]["clone_key"]
    require_obs(adata, clone_key, label_key)
    df = adata.obs[[clone_key, label_key]].dropna()
    df = df.assign(_clone=df[clone_key].astype(str))
    numeric = pd.api.types.is_numeric_dtype(df[label_key]) and not pd.api.types.is_bool_dtype(df[label_key])
    if how == "auto":
        how = "mean" if numeric else "majority"
    grouped = df.groupby("_clone", observed=True)[label_key]
    if how in ("mean", "median"):
        values = pd.to_numeric(df[label_key], errors="coerce")
        grouped = values.groupby(df["_clone"])
        out = grouped.mean() if how == "mean" else grouped.median()
    elif how.startswith("fraction:"):
        level = how.split(":", 1)[1]
        out = df[label_key].astype(str).eq(level).groupby(df["_clone"]).mean()
    elif how == "majority":
        def _maj(s):
            vc = s.astype(str).value_counts(normalize=True)
            tied = len(vc) > 1 and vc.iloc[1] == vc.iloc[0]
            return None if tied or vc.iloc[0] < min_frac else vc.index[0]
        out = grouped.agg(_maj)
    else:
        raise ValueError(f"unknown how={how!r}")
    out.index.name = clone_key
    return out.rename(label_key)


# --------------------------------------------------------------------------- association tests


def _mantel_haenszel(a, b, c, d):
    """MH pooled odds ratio and 95% CI (Robins-Breslow-Greenland) over strata."""
    n = a + b + c + d
    ok = n > 0
    a, b, c, d, n = a[ok], b[ok], c[ok], d[ok], n[ok]
    r = a * d / n
    s = b * c / n
    R, S = r.sum(), s.sum()
    if R == 0 or S == 0:
        or_mh = np.inf if S == 0 and R > 0 else (0.0 if R == 0 and S > 0 else np.nan)
        return or_mh, np.nan, np.nan
    p_ = (a + d) / n
    q_ = (b + c) / n
    var = (np.sum(p_ * r) / (2 * R**2) + np.sum(p_ * s + q_ * r) / (2 * R * S)
           + np.sum(q_ * s) / (2 * S**2))
    lor = np.log(R / S)
    half = 1.959964 * np.sqrt(var)
    return float(np.exp(lor)), float(np.exp(lor - half)), float(np.exp(lor + half))


def _assoc_table(adata, label_key, programme_key, clone_key, strata_key, how, min_frac):
    prof = get_profiles(adata)
    table = prof["clone_table"]
    if programme_key not in table.columns:
        raise KeyError(f"'{programme_key}' not in the clone table; run find_programmes() first.")
    lab = clone_labels(adata, label_key, clone_key=clone_key, how=how, min_frac=min_frac)
    df = pd.DataFrame({
        "programme": table[programme_key].astype("object"),
        "label": lab.reindex(table.index),
        "n_cells": table["n_cells"],
    })
    if strata_key is not None:
        if strata_key in table.columns:
            df["stratum"] = table[strata_key].astype(str)
        else:
            df["stratum"] = clone_labels(adata, strata_key, clone_key=clone_key, how="majority",
                                         min_frac=0.0).reindex(table.index).astype(str)
    else:
        df["stratum"] = "all"
    return df.dropna(subset=["programme", "label"])


def association_test(
    adata,
    label_key: str,
    *,
    programme_key: str = "clone_programme",
    clone_key: str | None = None,
    strata_key: str | None = "auto",
    how: str = "auto",
    min_frac: float = 0.5,
    n_perm: int = 2000,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clone-level association between programmes and a label.

    Each clone contributes once. Cell-level labels are first collapsed per
    clone (:func:`clone_labels`). P-values come from permuting clone labels
    within strata (default: donor), which keeps every donor's label
    frequencies fixed; Benjamini-Hochberg FDR is computed across all rows.

    *Categorical labels* (e.g. antigen specificity, isotype, infection
    status) give one row per (programme, level) with the observed and
    expected number of clones, the Mantel-Haenszel odds ratio across strata
    with a 95% confidence interval, and a two-sided permutation p-value.

    *Numeric labels* (e.g. mutation frequency, fraction of a clone's cells in
    a sort gate) give one row per programme with the median in vs out of the
    programme, a rank-based effect (difference in mean ranks scaled to
    [-1, 1]) and a two-sided permutation p-value; an omnibus Kruskal-Wallis
    test with the same permutation scheme is stored in ``.attrs``.

    Parameters
    ----------
    label_key
        ``obs`` column to test.
    strata_key
        Permutation strata. ``"auto"`` uses the donor recorded by
        :func:`threadfin.tl.clone_profiles` (if any); ``None`` permutes
        globally.
    how, min_frac
        Passed to :func:`clone_labels`.
    """
    require_positive_int(n_perm, "n_perm")
    prof = get_profiles(adata)
    clone_key = clone_key or prof["params"]["clone_key"]
    if strata_key == "auto":
        strata_key = "donor" if "donor" in prof["clone_table"].columns else None
    df = _assoc_table(adata, label_key, programme_key, clone_key, strata_key, how, min_frac)
    if df.empty:
        raise ValueError(f"No clones with both a programme and a '{label_key}' value.")
    rng = np.random.default_rng(random_state)
    prog_codes, progs = codes(df["programme"])
    if len(progs) < 2:
        raise ValueError("Only one programme: clone-level tests compare programmes, so at least two are needed.")
    strata = codes(df["stratum"])[0]
    numeric = pd.api.types.is_numeric_dtype(df["label"]) and not pd.api.types.is_bool_dtype(df["label"])

    if numeric:
        out = _numeric_association(df, prog_codes, progs, strata, n_perm, rng)
    else:
        out = _categorical_association(df, prog_codes, progs, strata, n_perm, rng)
    out.insert(0, "label", label_key)
    out["fdr"] = bh_fdr(out["pvalue"].to_numpy())
    out.attrs["strata_key"] = strata_key
    out.attrs["n_clones"] = int(len(df))
    get_uns(adata).setdefault("associations", {})[label_key] = out
    n_sig = int((out["fdr"] < 0.05).sum())
    log(f"association '{label_key}': {len(df)} clones, {n_sig} programme effects at FDR < 0.05 "
        f"(within-'{strata_key}' permutation).", verbose)
    return out


# --------------------------------------------------------------------------- label vs clone profiles


def profile_association(
    adata,
    label_key: str,
    *,
    clone_key: str | None = None,
    strata_key: str | None = "auto",
    how: str = "auto",
    min_frac: float = 0.5,
    min_reliability: float = 0.5,
    n_perm: int = 2000,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """Does a clone-level label explain how clones differ? (no programmes needed)

    Programme tests (:func:`association_test`) need distinct programmes. When
    clones vary along a continuum there are none, but a label (division
    history, antigen binding, isotype) can still track where a clone sits.
    This test asks how much of the variation between clone profiles is
    explained by the label: the share of the profile sum of squares between
    label groups (categorical labels, as in PERMANOVA; Anderson 2001) or
    captured by a linear trend in the label (numeric labels, as in
    distance-based redundancy analysis; McArdle & Anderson 2001). The
    p-value comes from permuting labels among clones within strata (donors).

    Parameters
    ----------
    label_key, how, min_frac
        Cell-level label and how it is collapsed per clone (see
        :func:`clone_labels`).
    strata_key
        Clone-table column whose levels define permutation strata;
        ``"auto"`` uses the donor when available.
    min_reliability
        Only clones whose profile reliability reaches this value are used.

    Returns
    -------
    dict with ``r2`` (fraction of profile variance explained by the label),
    the permutation ``null_mean``, ``excess_r2`` (``r2 - null_mean``),
    ``p_value``, ``n_clones``, ``kind`` and, for every level of a categorical
    label (or for the numeric label), Spearman correlations with the first
    three principal axes of the profiles (``axis_correlation``). Stored in
    ``adata.uns['threadfin']['profile_associations'][label_key]``.
    """
    from scipy.stats import spearmanr

    require_positive_int(n_perm, "n_perm")
    prof = get_profiles(adata)
    table = prof["clone_table"]
    clone_key = clone_key or prof["params"]["clone_key"]
    if strata_key == "auto":
        strata_key = "donor" if "donor" in table.columns else None
    labels = clone_labels(adata, label_key, clone_key=clone_key, how=how, min_frac=min_frac)
    use = table.index[(table["reliability"] >= min_reliability).to_numpy()]
    use = use[labels.reindex(use).notna().to_numpy()]
    if use.size < 10:
        raise ValueError(f"Only {use.size} reliable clones have a '{label_key}' value (need >= 10).")
    x = prof["features"].loc[use].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    total = float(np.einsum("ij,ij->", x, x))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Clone profiles have no finite between-clone variation to explain.")
    y = labels.loc[use]
    strata = (codes(table.loc[use, strata_key].astype(str))[0] if strata_key is not None
              else np.zeros(use.size, dtype=np.int64))
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    rng = np.random.default_rng(random_state)

    if numeric:
        yv = y.to_numpy(dtype=float)

        def stat(v):
            v = v - v.mean()
            ss = float(v @ v)
            return 0.0 if ss <= 0 else float(np.sum((x.T @ v) ** 2) / ss / total)
    else:
        yv, levels = codes(y.astype(str))
        if len(levels) < 2:
            raise ValueError(f"'{label_key}' has a single level among the reliable clones.")

        def stat(v):
            sums = np.zeros((len(levels), x.shape[1]))
            np.add.at(sums, v, x)
            n = np.bincount(v, minlength=len(levels)).astype(float)
            ok = n > 0
            return float(np.sum(sums[ok] ** 2 / n[ok, None]) / total)

    observed = stat(yv)
    null = np.array([stat(permute_within(yv, strata, rng)) for _ in range(n_perm)])
    p_value = perm_pvalue(observed, null)

    # where along the main axes of clone variation does the label sit?
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    axes = x @ vt[: min(3, vt.shape[0])].T
    if numeric:
        axis_corr = {"label": [float(spearmanr(axes[:, k], yv).statistic) for k in range(axes.shape[1])]}
    else:
        axis_corr = {str(lv): [float(spearmanr(axes[:, k], (yv == i).astype(float)).statistic)
                               for k in range(axes.shape[1])] for i, lv in enumerate(levels)}
    res = {
        "label": label_key, "kind": "numeric" if numeric else "categorical", "r2": observed,
        "null_mean": float(null.mean()), "excess_r2": float(observed - null.mean()), "p_value": float(p_value),
        "n_clones": int(use.size), "n_levels": None if numeric else int(len(levels)),
        "strata_key": strata_key, "n_perm": int(n_perm), "axis_correlation": axis_corr,
    }
    get_uns(adata).setdefault("profile_associations", {})[label_key] = res
    log(f"profile association '{label_key}': explains {100 * observed:.1f}% of clone-profile variance vs "
        f"{100 * res['null_mean']:.1f}% for permuted labels (p = {p_value:.3g}, {use.size} clones, "
        f"within-'{strata_key}' permutation).", verbose)
    return res


def _categorical_association(df, prog_codes, progs, strata, n_perm, rng):
    lab_codes, levels = codes(df["label"].astype(str))
    n_p, n_l = len(progs), len(levels)
    n_s = int(strata.max()) + 1

    def counts(labs):
        return np.bincount(prog_codes * n_l + labs, minlength=n_p * n_l).reshape(n_p, n_l)

    observed = counts(lab_codes)
    # hypergeometric expectation under within-stratum permutation
    expected = np.zeros((n_p, n_l))
    for s in range(n_s):
        m = strata == s
        if not m.any():
            continue
        cp = np.bincount(prog_codes[m], minlength=n_p).astype(float)
        cl = np.bincount(lab_codes[m], minlength=n_l).astype(float)
        expected += np.outer(cp, cl) / m.sum()
    null_dev = np.zeros((n_p, n_l))
    obs_dev = np.abs(observed - expected)
    for _ in range(n_perm):
        perm = permute_within(lab_codes, strata, rng)
        null_dev += np.abs(counts(perm) - expected) >= obs_dev - 1e-9
    pvals = (1 + null_dev) / (n_perm + 1)

    rows = []
    for i, prog in enumerate(progs):
        in_p = prog_codes == i
        for j, level in enumerate(levels):
            has = lab_codes == j
            a = np.bincount(strata[in_p & has], minlength=n_s).astype(float)
            b = np.bincount(strata[in_p & ~has], minlength=n_s).astype(float)
            c = np.bincount(strata[~in_p & has], minlength=n_s).astype(float)
            d = np.bincount(strata[~in_p & ~has], minlength=n_s).astype(float)
            or_mh, lo, hi = _mantel_haenszel(a, b, c, d)
            n_in = int(in_p.sum())
            rows.append({
                "programme": prog, "level": level,
                "n_clones_programme": n_in, "n_clones_level": int(has.sum()),
                "observed": int(observed[i, j]), "expected": float(expected[i, j]),
                "frac_in_programme": observed[i, j] / n_in if n_in else np.nan,
                "frac_elsewhere": float(c.sum() / max((~in_p).sum(), 1)),
                "odds_ratio": or_mh, "ci_low": lo, "ci_high": hi,
                "pvalue": float(pvals[i, j]),
            })
    return pd.DataFrame(rows)


def _numeric_association(df, prog_codes, progs, strata, n_perm, rng):
    values = df["label"].to_numpy(dtype=float)
    ranks = pd.Series(values).rank().to_numpy()
    n = ranks.size
    n_p = len(progs)
    sizes = np.bincount(prog_codes, minlength=n_p).astype(float)

    def rank_effects(r):
        sums = np.bincount(prog_codes, weights=r, minlength=n_p)
        mean_in = sums / np.maximum(sizes, 1)
        mean_out = (r.sum() - sums) / np.maximum(n - sizes, 1)
        return (mean_in - mean_out) / (n / 2.0)  # scaled to roughly [-1, 1]

    def kruskal_h(r):
        sums = np.bincount(prog_codes, weights=r, minlength=n_p)
        return 12.0 / (n * (n + 1)) * np.sum(sums**2 / np.maximum(sizes, 1)) - 3 * (n + 1)

    obs_eff = rank_effects(ranks)
    obs_h = kruskal_h(ranks)
    exceed = np.zeros(n_p)
    exceed_h = 0
    for _ in range(n_perm):
        r = permute_within(np.arange(n), strata, rng)
        pr = ranks[r]
        exceed += np.abs(rank_effects(pr)) >= np.abs(obs_eff) - 1e-12
        exceed_h += kruskal_h(pr) >= obs_h - 1e-12
    rows = []
    for i, prog in enumerate(progs):
        in_p = prog_codes == i
        rows.append({
            "programme": prog, "n_clones_programme": int(in_p.sum()),
            "median_in_programme": float(np.median(values[in_p])) if in_p.any() else np.nan,
            "median_elsewhere": float(np.median(values[~in_p])) if (~in_p).any() else np.nan,
            "rank_effect": float(obs_eff[i]),
            "pvalue": float((1 + exceed[i]) / (n_perm + 1)),
        })
    out = pd.DataFrame(rows)
    out.attrs["kruskal_h"] = float(obs_h)
    out.attrs["kruskal_pvalue"] = float((1 + exceed_h) / (n_perm + 1))
    return out
