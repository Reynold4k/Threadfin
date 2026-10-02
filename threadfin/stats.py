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

from ._utils import bh_fdr, codes, get_uns, log, perm_pvalue, permute_within, require_obs
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
    prof = get_profiles(adata)
    # the ICC is always measured on unsmoothed cells in the embedding itself, so it reads
    # as "share of transcriptional variance explained by clone" whatever the profile type
    model = model_from_adata(adata, unsmoothed=True, representation="mean")
    strata_key = strata_key or prof["params"].get("context_key")
    strata = None
    if strata_key is not None:
        require_obs(adata, strata_key)
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
    prof = get_profiles(adata)
    clone_key = clone_key or prof["params"]["clone_key"]
    if strata_key == "auto":
        strata_key = "donor" if "donor" in prof["clone_table"].columns else None
    df = _assoc_table(adata, label_key, programme_key, clone_key, strata_key, how, min_frac)
    if df.empty:
        raise ValueError(f"No clones with both a programme and a '{label_key}' value.")
    rng = np.random.default_rng(random_state)
    prog_codes, progs = codes(df["programme"])
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
