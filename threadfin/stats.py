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
   Clone size can be a stratum too: small clones' profiles are shrunk towards
   the average profile, so a label that follows clone size can look like a
   difference in state (``size_strata`` in the label tests).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import bh_fdr, codes, get_uns, h5ad_safe, log, nested_strata, perm_pvalue, permute_within, require_obs, require_complete_groups, require_positive_int
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
        neither, labels are shuffled globally (not recommended). Strata
        stay within the ``context_key`` groups: if a stratum holds cells of
        several of them (a library pooling donors), the strata are the
        ``context_key`` x ``strata_key`` combinations, with a warning.
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
    context_key = prof["params"].get("context_key")
    strata_key = strata_key or context_key
    if strata_key is not None:
        require_complete_groups(adata, strata_key, context="coherence permutations")
    strata, strata_key = nested_strata(adata.obs, strata_key, context_key)

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
    get_uns(adata)["coherence"] = h5ad_safe(res)
    log(
        f"clonal coherence: ICC {observed:.3f} vs {res['null_mean']:.3f} under within-"
        f"'{strata_key}' permutation (p = {res['p_value']:.3g}, {n_perm} permutations).",
        verbose,
    )
    return res


# --------------------------------------------------------------------------- clone-level labels

#: Text that stands for a missing value when it is a level of a label column.
_MISSING_LIKE = ("nan", "NaN", "None", "none", "NA", "<NA>", "", ".")


def _warn_missing_like_levels(adata, label_key: str) -> None:
    """Warn about label levels such as 'nan' or '.' that look like missing values (shown once per label)."""
    counts = adata.obs[label_key].value_counts(dropna=True)
    counts = counts[counts.to_numpy() > 0]
    text = pd.Index(counts.index.astype(str))
    hit = np.asarray(text.str.strip().isin(_MISSING_LIKE))
    if not hit.any():
        return
    found = ", ".join(f"{lv!r} ({int(n):,} cells)" for lv, n in zip(text[hit], counts.to_numpy()[hit]))
    one = int(hit.sum()) == 1
    # stacklevel=1: both label tests call this for the same label, and the warning is shown once
    warnings.warn(
        f"adata.obs['{label_key}'] has {'a level that looks' if one else 'levels that look'} like a missing value: "
        f"{found}. {'It is' if one else 'They are'} treated as {'a level' if one else 'levels'} of their own; if "
        f"these cells have no {label_key}, set their adata.obs['{label_key}'] to NaN.", UserWarning, stacklevel=1)


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

    Levels written as text that stand for a missing value (``"nan"``,
    ``"None"``, ``"NA"``, ``"."`` ...) are kept as levels, with a warning; set
    those cells to NaN to leave them out.
    """
    clone_key = clone_key or get_profiles(adata)["params"]["clone_key"]
    require_obs(adata, clone_key, label_key)
    df = adata.obs[[clone_key, label_key]].dropna()
    df = df.assign(_clone=df[clone_key].astype(str))
    numeric = pd.api.types.is_numeric_dtype(df[label_key]) and not pd.api.types.is_bool_dtype(df[label_key])
    if how == "auto":
        how = "mean" if numeric else "majority"
    if how == "majority" or how.startswith("fraction:"):
        _warn_missing_like_levels(adata, label_key)
    grouped = df.groupby("_clone", observed=True)[label_key]
    if how in ("mean", "median"):
        values = pd.to_numeric(df[label_key], errors="coerce")
        grouped = values.groupby(df["_clone"])
        out = grouped.mean() if how == "mean" else grouped.median()
    elif how.startswith("fraction:"):
        level = how.split(":", 1)[1]
        values = df[label_key].astype(str)
        if not values.eq(level).any():
            similar = sorted(v for v in values.unique() if v.startswith(level))[:6]
            hint = (f" Levels starting with '{level}': {', '.join(similar)}; to pool them, add a column such as "
                    f"adata.obs['{label_key}'].str[:{len(level)}] and use that.") if similar else ""
            warnings.warn(f"'{level}' never occurs in adata.obs['{label_key}'], so its fraction is 0 for every "
                          f"clone.{hint}", UserWarning, stacklevel=2)
        out = values.eq(level).groupby(df["_clone"]).mean()
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

#: Lower bounds, in cells, of the clone-size bins used by ``size_strata=True``: clones of 1, 2, 3-4, 5-9 and
#: >= 10 cells (profiles need two cells by default, so the first bin is usually empty).
SIZE_BINS = (1, 2, 3, 5, 10)

#: What ``size_strata=None`` (the default) means in each label test: shuffle the label among clones of the
#: same donor and clone-size bin (True), or of the same donor only (False). In simulations where programmes
#: differ in clone size as much as in the RBD data, a label that follows only clone size was called by 30-42 %
#: of programme tests and 29-54 % of profile tests within donors, and by 3-6 % within donor x size bins, while
#: labels that follow the programme kept their power (losses of 0-7 points).
SIZE_STRATA_DEFAULT = {"association_test": True, "profile_association": True}


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


def _size_bins(size_strata, test: str) -> tuple[int, ...] | None:
    """Lower bounds (cells) of the clone-size bins that ``size_strata`` asks for, or None for no size bins."""
    if size_strata is None:
        size_strata = SIZE_STRATA_DEFAULT[test]
    if isinstance(size_strata, (bool, np.bool_)):
        return tuple(SIZE_BINS) if size_strata else None
    edges = [] if isinstance(size_strata, (str, bytes)) or not np.iterable(size_strata) else list(size_strata)
    if (not edges or any(b <= a for a, b in zip(edges, edges[1:]))
            or not all(isinstance(e, (int, np.integer)) and not isinstance(e, (bool, np.bool_)) and e >= 1
                       for e in edges)):
        raise ValueError("size_strata must be True, False, None, or increasing clone sizes (in cells) at which "
                         "bins start, e.g. (2, 3, 5, 10).")
    return tuple(int(e) for e in ([1] if edges[0] > 1 else []) + edges)


def _size_bin_labels(n_cells, edges: tuple[int, ...]) -> np.ndarray:
    """The clone-size bin of each clone, named like '2', '3-4' or '>=10'."""
    names = [f">={lo}" if i == len(edges) - 1 else str(lo) if edges[i + 1] == lo + 1 else f"{lo}-{edges[i + 1] - 1}"
             for i, lo in enumerate(edges)]
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(n_cells, dtype=float), side="right") - 1
    return np.asarray(names, dtype=object)[np.clip(idx, 0, None)]


def _bins_text(size_bin) -> str:
    """The bins that occur, in order of size, e.g. '2, 3-4, 5-9 and >=10 cells'."""
    names = sorted(pd.unique(np.asarray(size_bin, dtype=object)),
                   key=lambda name: int(str(name).lstrip(">=").split("-")[0]))
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + " cells"


def _perm_text(strata_key: str | None, size_bin) -> str:
    """How the labels were shuffled, for the log."""
    if strata_key and size_bin is not None:
        return f"permutation within '{strata_key}' x clone-size bins"
    if strata_key:
        return f"within-'{strata_key}' permutation"
    return "permutation within clone-size bins" if size_bin is not None else "permutation among all clones"


def _require_variation_within_strata(labels: pd.Series, strata, label_key: str, strata_key: str | None,
                                     size_bin=None) -> None:
    """Refuse a label that never varies among the clones of one stratum: shuffling within strata cannot test it."""
    df = pd.DataFrame({"label": np.asarray(labels), "stratum": np.asarray(strata)})
    if df["label"].nunique() < 2:   # one value overall is reported by the caller
        return
    if strata_key is not None and not df.groupby("stratum")["label"].nunique().gt(1).any():
        raise ValueError(
            f"'{label_key}' does not vary among the clones of any single '{strata_key}', so shuffling it within "
            f"'{strata_key}' cannot test it: it is a property of each {strata_key}. Compare the {strata_key}s "
            f"instead, with one value per {strata_key}.")
    if size_bin is not None:
        df["size"] = np.asarray(size_bin)
        if not df.groupby(["stratum", "size"])["label"].nunique().gt(1).any():
            within = f" of any single '{strata_key}'" if strata_key is not None else ""
            raise ValueError(
                f"'{label_key}' does not vary among the clones of similar size{within} (clones of "
                f"{_bins_text(size_bin)}), so shuffling it within these clone-size bins cannot test it: it follows "
                f"clone size. Test it with size_strata=False, knowing that profiles and programmes also depend "
                f"on clone size.")


def _sample_level(adata, label_key: str, clone_key: str, clones, sample_key, context_key) -> dict:
    """Is the label the same for every cell of each sample? The test then compares samples, not clones."""
    sample_key = context_key if sample_key == "auto" else sample_key
    out = {"sample_key": sample_key, "n_samples": None, "sample_level": False, "samples_per_level": None}
    if sample_key is None:
        return out
    require_obs(adata, sample_key, context="the sample-level check of a label test")
    obs = adata.obs
    keep = (obs[clone_key].notna() & obs[label_key].notna() & obs[sample_key].notna()).to_numpy()
    keep &= obs[clone_key].astype(str).isin(pd.Index(clones).astype(str)).to_numpy()
    if not keep.any():
        return out
    lab = pd.Series(obs.loc[keep, label_key].astype(str).to_numpy())
    smp = obs.loc[keep, sample_key].astype(str).to_numpy()
    per_sample = lab.groupby(smp).nunique()
    out["n_samples"] = int(per_sample.size)
    if per_sample.max() == 1 and lab.nunique() > 1:
        out["sample_level"] = True
        cell_numeric = (pd.api.types.is_numeric_dtype(obs[label_key])
                        and not pd.api.types.is_bool_dtype(obs[label_key]))
        if not cell_numeric:
            n_per = lab.groupby(smp).first().value_counts().sort_index()
            out["samples_per_level"] = [{"level": str(k), "n_samples": int(v)} for k, v in n_per.items()]
    return out


def _warn_sample_level(label_key: str, info: dict) -> None:
    """Say that a sample-level label is tested with clones as replicates (``tf.run`` says it once per label)."""
    if info["sample_level"]:
        warnings.warn(
            f"'{label_key}' is the same for every cell of each '{info['sample_key']}', so this test compares "
            f"{info['n_samples']} samples, not independent clones: whatever differs between samples (library, sort, "
            f"processing) is read as an effect of '{label_key}', and the p-value counts clones as replicates. Read "
            f"it as descriptive, or compare the samples with one value per sample.", UserWarning, stacklevel=1)


def _mirror_rows(level: pd.Series, labels: pd.Series) -> np.ndarray:
    """Rows of the more common level of a two-level label (the later level in sort order on a tie): they
    mirror the rows of the other level."""
    counts = labels.astype(str).value_counts()
    if len(counts) != 2:
        return np.zeros(len(level), dtype=bool)
    a, b = sorted(counts.index)
    common = a if counts[a] > counts[b] else b
    return (level.astype(str) == common).to_numpy()


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
    size_strata: bool | tuple[int, ...] | None = None,
    sample_key: str | None = "auto",
    how: str = "auto",
    min_frac: float = 0.5,
    n_perm: int = 2000,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clone-level association between programmes and a label.

    Each clone contributes once. Cell-level labels are first collapsed per
    clone (:func:`clone_labels`). P-values come from permuting clone labels
    within strata: clones of the same donor and, by default, of similar size
    (``size_strata``), which keeps each donor's label frequencies and the
    label's link with clone size as observed. Benjamini-Hochberg FDR is
    computed across all rows. A label that never varies within a stratum (a
    donor's age or sex) cannot be tested this way and raises ``ValueError``.

    Clone size matters because profiles are shrunk: a small clone is pulled
    towards the average profile, so programmes differ in clone size, and a
    label that follows clone size (a majority call over few cells, detection
    in a rare sort gate, expansion) would otherwise show up as a programme
    effect. Each row gives the programme's median clone size
    (``median_cells_programme``) and share of two-cell clones
    (``frac_2cell_programme``).

    A label that is the same for every cell of each sample (library, sort
    gate, tissue, visit) is tested with a warning and
    ``.attrs['sample_level'] = True``: it then compares samples
    (``.attrs['n_samples']``), and clones of one sample are not independent
    replicates.

    *Categorical labels* (e.g. antigen specificity, isotype, infection
    status) give one row per (programme, level) with the observed number of
    clones, the number expected from the strata alone, the Mantel-Haenszel
    odds ratio across strata with a 95% confidence interval, and a two-sided
    permutation p-value. For a two-level label the rows of the more common
    level mirror those of the rarer level (same p-value, inverse odds ratio):
    they are marked ``mirror``, are not counted as separate effects, and leave
    the FDR unchanged (Benjamini-Hochberg gives the same values with or
    without exact duplicates).

    *Numeric labels* (e.g. mutation frequency, fraction of a clone's cells in
    a sort gate) give one row per programme with the median in vs out of the
    programme, a rank-based effect (difference in mean ranks scaled to
    [-1, 1]), the effect expected from the strata alone
    (``expected_rank_effect``) and a two-sided permutation p-value for the
    difference between the two; an omnibus Kruskal-Wallis test with the same
    permutation scheme (rank sums against their expectation within strata) is
    stored in ``.attrs``.

    Parameters
    ----------
    label_key
        ``obs`` column to test.
    strata_key
        Permutation strata. ``"auto"`` uses the donor recorded by
        :func:`threadfin.tl.clone_profiles` (if any); ``None`` permutes
        among all clones (within clone-size bins if ``size_strata``).
    size_strata
        Also permute within clone-size bins (``SIZE_BINS``: 2, 3-4, 5-9 and
        >= 10 cells). ``None`` uses ``SIZE_STRATA_DEFAULT`` (on for this
        test); a tuple of clone sizes sets where bins start, e.g. ``(2, 5, 20)``.
    sample_key
        ``obs`` column of samples for the sample-level check. ``"auto"`` uses
        the ``context_key`` of :func:`threadfin.tl.clone_profiles` (the
        ``sample_key`` of :func:`threadfin.run`); ``None`` skips the check.
    how, min_frac
        Passed to :func:`clone_labels`.
    """
    require_positive_int(n_perm, "n_perm")
    prof = get_profiles(adata)
    clone_key = clone_key or prof["params"]["clone_key"]
    if strata_key == "auto":
        strata_key = "donor" if "donor" in prof["clone_table"].columns else None
    bins = _size_bins(size_strata, "association_test")
    df = _assoc_table(adata, label_key, programme_key, clone_key, strata_key, how, min_frac)
    if df.empty:
        raise ValueError(f"No clones with both a programme and a '{label_key}' value.")
    size_bin = None if bins is None else _size_bin_labels(df["n_cells"], bins)
    _require_variation_within_strata(df["label"], df["stratum"], label_key, strata_key, size_bin)
    rng = np.random.default_rng(random_state)
    prog_codes, progs = codes(df["programme"])
    if len(progs) < 2:
        raise ValueError("Only one programme: clone-level tests compare programmes, so at least two are needed.")
    strata = codes(df["stratum"] if size_bin is None else df["stratum"].astype(str) + "|" + size_bin)[0]
    numeric = pd.api.types.is_numeric_dtype(df["label"]) and not pd.api.types.is_bool_dtype(df["label"])

    if numeric:
        out = _numeric_association(df, prog_codes, progs, strata, n_perm, rng)
    else:
        out = _categorical_association(df, prog_codes, progs, strata, n_perm, rng)
    n_cells = df["n_cells"].to_numpy(dtype=float)
    at = out.columns.get_loc("n_clones_programme") + 1
    median = {p: float(np.median(n_cells[prog_codes == i])) for i, p in enumerate(progs)}
    two_cell = {p: float(np.mean(n_cells[prog_codes == i] == 2)) for i, p in enumerate(progs)}
    out.insert(at, "median_cells_programme", out["programme"].map(median))
    out.insert(at + 1, "frac_2cell_programme", out["programme"].map(two_cell))
    out.insert(0, "label", label_key)
    out["fdr"] = bh_fdr(out["pvalue"].to_numpy())
    if "level" in out.columns:
        out["mirror"] = _mirror_rows(out["level"], df["label"])
    sample = _sample_level(adata, label_key, clone_key, df.index, sample_key, prof["params"].get("context_key"))
    _warn_sample_level(label_key, sample)
    out.attrs.update(strata_key=strata_key, size_bins=None if bins is None else list(bins), n_clones=int(len(df)),
                     **sample)
    get_uns(adata).setdefault("associations", {})[label_key] = h5ad_safe(out)
    mirror = out["mirror"].to_numpy() if "mirror" in out.columns else np.zeros(len(out), dtype=bool)
    n_sig = int(((out["fdr"] < 0.05).to_numpy() & ~mirror).sum())
    log(f"association '{label_key}': {len(df)} clones, {n_sig} programme effects at FDR < 0.05 "
        f"({_perm_text(strata_key, size_bin)}).", verbose)
    return out


# --------------------------------------------------------------------------- label vs clone profiles


def profile_association(
    adata,
    label_key: str,
    *,
    clone_key: str | None = None,
    strata_key: str | None = "auto",
    size_strata: bool | tuple[int, ...] | None = None,
    sample_key: str | None = "auto",
    how: str = "auto",
    min_frac: float = 0.5,
    min_reliability: float = 0.0,
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
    p-value comes from permuting labels among clones within strata (donors,
    and clone-size bins). A label that never varies within a stratum (a
    donor's age or sex) cannot be tested this way and raises ``ValueError``.

    Profiles of small clones are shrunk towards the average profile, so they
    are less spread out, and clone size can go with clone state; a label that
    follows clone size can then explain profile variance by itself. The label
    is therefore shuffled, by default, among clones of the same donor and
    similar size (``size_strata``), and the result gives the clone sizes
    behind the label (``clone_size``).

    A label that is the same for every cell of each sample (library, sort
    gate, tissue, visit) is tested with a warning and ``sample_level`` set to
    True: it then compares ``n_samples`` samples, and clones of one sample are
    not independent replicates.

    Parameters
    ----------
    label_key, how, min_frac
        Cell-level label and how it is collapsed per clone (see
        :func:`clone_labels`).
    strata_key
        Clone-table column whose levels define permutation strata;
        ``"auto"`` uses the donor when available.
    size_strata
        Also permute within clone-size bins (``SIZE_BINS``: 2, 3-4, 5-9 and
        >= 10 cells). ``None`` uses ``SIZE_STRATA_DEFAULT`` (on for this
        test); a tuple of clone sizes sets where bins start, e.g. ``(2, 5, 20)``.
    sample_key
        ``obs`` column of samples for the sample-level check. ``"auto"`` uses
        the ``context_key`` of :func:`threadfin.tl.clone_profiles` (the
        ``sample_key`` of :func:`threadfin.run`); ``None`` skips the check.
    min_reliability
        Reliability floor; ``0.0`` (the default) uses every profiled clone.
        Threadfin does not filter clones on reliability. Profiles are already shrunk: a clone seen in
        few cells is pulled towards the average profile, so its influence is down-weighted in
        proportion to how little was seen. A reliability floor on top of that discards the same
        information twice; in the germinal-centre datasets of the manuscript a floor of 0.5 removes
        75-100 % of the profiled clones.

    Returns
    -------
    dict with ``r2`` (fraction of profile variance explained by the label),
    the permutation ``null_mean``, ``excess_r2`` (``r2 - null_mean``),
    ``p_value``, ``n_clones``, ``kind`` and, for every level of a categorical
    label (or for the numeric label), Spearman correlations with the first
    three principal axes of the profiles (``axis_correlation``); the clone
    sizes behind the label (``clone_size``: number of clones, median cells and
    share of two-cell clones per level, and for a numeric label its Spearman
    correlation with clone size); ``size_bins`` (None without size strata);
    and the sample-level check (``sample_level``, ``sample_key``,
    ``n_samples``, and ``samples_per_level`` for a categorical label). Stored
    in ``adata.uns['threadfin']['profile_associations'][label_key]``.
    """
    from scipy.stats import spearmanr

    require_positive_int(n_perm, "n_perm")
    prof = get_profiles(adata)
    table = prof["clone_table"]
    clone_key = clone_key or prof["params"]["clone_key"]
    if strata_key == "auto":
        strata_key = "donor" if "donor" in table.columns else None
    bins = _size_bins(size_strata, "profile_association")
    labels = clone_labels(adata, label_key, clone_key=clone_key, how=how, min_frac=min_frac)
    candidates = table.index[(table["reliability"] >= min_reliability).to_numpy()]
    use = candidates[labels.reindex(candidates).notna().to_numpy()]
    # clones whose cells carry the label but no level covers min_frac of them (e.g. a 1:1 two-cell clone)
    labelled = adata.obs.loc[adata.obs[label_key].notna(), clone_key].dropna().astype(str).unique()
    n_without_majority = int(np.isin(candidates.astype(str), labelled).sum()) - int(use.size)
    if use.size < 10:
        raise ValueError(f"Only {use.size} reliable clones have a '{label_key}' value (need >= 10).")
    x = prof["features"].loc[use].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    total = float(np.einsum("ij,ij->", x, x))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Clone profiles have no finite between-clone variation to explain.")
    y = labels.loc[use]
    n_cells = table.loc[use, "n_cells"].to_numpy(dtype=float)
    base = (table.loc[use, strata_key].astype(str) if strata_key is not None
            else pd.Series("all", index=use)).to_numpy(dtype=object)
    size_bin = None if bins is None else _size_bin_labels(n_cells, bins)
    _require_variation_within_strata(y, base, label_key, strata_key, size_bin)
    strata = codes(base if size_bin is None else base + "|" + size_bin)[0]
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
    # the clone sizes behind the label (profiles, and how spread out they are, depend on clone size)
    groups = [("all", np.ones(yv.size, dtype=bool))] if numeric else [(str(lv), yv == i) for i, lv in enumerate(levels)]
    clone_size = [{"level": name, "n_clones": int(m.sum()),
                   "median_cells": float(np.median(n_cells[m])) if m.any() else np.nan,
                   "frac_2cell": float(np.mean(n_cells[m] == 2)) if m.any() else np.nan} for name, m in groups]
    if numeric:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # constant sizes or labels give NaN
            clone_size[0]["spearman_with_clone_size"] = float(spearmanr(yv, n_cells).statistic)
    sample = _sample_level(adata, label_key, clone_key, use, sample_key, prof["params"].get("context_key"))
    _warn_sample_level(label_key, sample)
    res = {
        "label": label_key, "kind": "numeric" if numeric else "categorical", "r2": observed,
        "null_mean": float(null.mean()), "excess_r2": float(observed - null.mean()), "p_value": float(p_value),
        "n_clones": int(use.size), "n_levels": None if numeric else int(len(levels)),
        "strata_key": strata_key, "size_bins": None if bins is None else list(bins), "n_perm": int(n_perm),
        "axis_correlation": axis_corr, "clone_size": clone_size, "n_without_majority": n_without_majority,
        **sample,
    }
    get_uns(adata).setdefault("profile_associations", {})[label_key] = h5ad_safe(res)
    log(f"profile association '{label_key}': explains {100 * observed:.1f}% of clone-profile variance vs "
        f"{100 * res['null_mean']:.1f}% for permuted labels (p = {p_value:.3g}, {use.size} clones, "
        f"{_perm_text(strata_key, size_bin)}).", verbose)
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
    # rank sum each programme is expected to have when ranks are shuffled within strata, i.e. from its donor
    # (and clone-size) make-up alone; with a single stratum this is size * (n + 1) / 2
    n_s = int(strata.max()) + 1
    stratum_mean = (np.bincount(strata, weights=ranks, minlength=n_s)
                    / np.maximum(np.bincount(strata, minlength=n_s), 1))
    expected_sums = np.bincount(prog_codes, weights=stratum_mean[strata], minlength=n_p)

    def rank_sums(r):
        return np.bincount(prog_codes, weights=r, minlength=n_p)

    def rank_effects(sums):
        mean_in = sums / np.maximum(sizes, 1)
        mean_out = (ranks.sum() - sums) / np.maximum(n - sizes, 1)
        return (mean_in - mean_out) / (n / 2.0)  # scaled to roughly [-1, 1]

    def kruskal_h(sums):  # rank sums against their expectation within strata; the usual H with one stratum
        return 12.0 / (n * (n + 1)) * np.sum((sums - expected_sums) ** 2 / np.maximum(sizes, 1))

    obs_sums = rank_sums(ranks)
    obs_eff = rank_effects(obs_sums)
    exp_eff = rank_effects(expected_sums)
    obs_h = kruskal_h(obs_sums)
    exceed = np.zeros(n_p)
    exceed_h = 0
    for _ in range(n_perm):
        r = permute_within(np.arange(n), strata, rng)
        sums = rank_sums(ranks[r])
        exceed += np.abs(rank_effects(sums) - exp_eff) >= np.abs(obs_eff - exp_eff) - 1e-12
        exceed_h += kruskal_h(sums) >= obs_h - 1e-12
    rows = []
    for i, prog in enumerate(progs):
        in_p = prog_codes == i
        rows.append({
            "programme": prog, "n_clones_programme": int(in_p.sum()),
            "median_in_programme": float(np.median(values[in_p])) if in_p.any() else np.nan,
            "median_elsewhere": float(np.median(values[~in_p])) if (~in_p).any() else np.nan,
            "rank_effect": float(obs_eff[i]),
            "expected_rank_effect": float(exp_eff[i]),
            "pvalue": float((1 + exceed[i]) / (n_perm + 1)),
        })
    out = pd.DataFrame(rows)
    out.attrs["kruskal_h"] = float(obs_h)
    out.attrs["kruskal_pvalue"] = float((1 + exceed_h) / (n_perm + 1))
    return out
