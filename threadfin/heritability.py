"""Gene-level clonal ICC: which genes are shared within clones?

For every gene, the same one-way random-effects model as :mod:`threadfin.profiles` is fitted to its
context-centred log expression: the gene's clonal ICC is the fraction of its (within-context) variance that
is explained by clone identity. A high ICC says that cells of one clone resemble each other in that gene more
than unrelated cells do; it does not say why. A state fixed within the clone, a shared niche, sampling time or
antigen, and sister cells sampled together all raise it. Programmes that every clone cycles through (for
example dark-zone / light-zone germinal-centre programmes, or the cell cycle) are expected to have a low ICC
even when they are highly variable. Significance comes from shuffling clone labels among cells within strata,
exactly as in :func:`threadfin.tl.clonal_coherence`.

:func:`geneset_heritability` compares named gene sets with genes of similar expression. Genes of one set are
usually co-expressed, so their ICCs are not independent; the standard error of that comparison comes from
resampling clones, which keeps the correlation between genes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import (
    bh_fdr,
    codes,
    get_uns,
    h5ad_safe,
    log,
    nested_strata,
    permute_within,
    require_complete_groups,
    require_obs,
    require_positive_int,
)
from .profiles import variance_components

BOOTSTRAP_ATTR = "clone_bootstrap"


def _dense_chunk(x, cols):
    sub = x[:, cols]
    return sub.toarray() if sp.issparse(sub) else np.asarray(sub)


class _Replicates:
    """Holds the clone-bootstrap table in ``DataFrame.attrs``.

    pandas deep-copies ``attrs`` on every operation; this keeps that a reference instead of a copy of a
    genes x replicates table. The table is read-only by convention.
    """

    __slots__ = ("table",)

    def __init__(self, table: pd.DataFrame):
        self.table = table

    def __deepcopy__(self, memo):
        return self

    def __repr__(self) -> str:
        return f"<clone bootstrap: {self.table.shape[0]} genes x {self.table.shape[1]} replicates>"


def _clone_bootstrap_weights(clone_idx: np.ndarray, strata: np.ndarray | None, n_boot: int,
                             rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n_clones)`` multiplicities for clone indices ``0..C-1`` (one per cell).

    Each clone is drawn with replacement among the clones of its stratum (the stratum holding most of its
    cells), so every replicate keeps the number of clones per stratum.
    """
    n_clones = int(clone_idx.max()) + 1
    if strata is None:
        home = np.zeros(n_clones, dtype=np.int64)
    else:
        levels, s_idx = np.unique(np.asarray(strata), return_inverse=True)
        counts = sp.csr_matrix((np.ones(clone_idx.size), (clone_idx, s_idx)), shape=(n_clones, levels.size))
        home = np.asarray(counts.argmax(axis=1)).ravel()
    weights = np.zeros((n_boot, n_clones))
    for s in np.unique(home):
        members = np.flatnonzero(home == s)
        weights[:, members] = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size),
                                              size=n_boot)
    return weights


def _mean_squares(resid: np.ndarray, clone_idx: np.ndarray, weights: np.ndarray):
    """Between- and within-clone mean squares, and ``n0``, for each row of clone weights.

    The unbalanced one-way ANOVA of :func:`threadfin.profiles.variance_components`, written in per-clone sums
    so that a whole set of bootstrap replicates costs a few matrix products. Each row of ``weights`` gives
    clone multiplicities; a row of ones is the observed data. Returns ``msb`` and ``msw`` of shape
    ``(rows, columns of resid)`` and ``n0`` of shape ``(rows,)``.
    """
    resid = np.asarray(resid, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    n_clones = weights.shape[1]
    member = sp.csr_matrix((np.ones(clone_idx.size), (clone_idx, np.arange(clone_idx.size))),
                           shape=(n_clones, clone_idx.size))
    n_c = np.asarray(member.sum(axis=1)).ravel()
    sums = np.asarray(member @ resid)
    between = sums**2 / n_c[:, None]
    within = np.asarray(member @ (resid * resid)) - between
    n_tot = weights @ n_c
    k = weights.sum(axis=1)
    total = weights @ sums
    msw = np.maximum(weights @ within, 0.0) / (n_tot - k)[:, None]
    msb = np.maximum(weights @ between - total**2 / n_tot[:, None], 0.0) / (k - 1)[:, None]
    n0 = (n_tot - (weights @ n_c**2) / n_tot) / (k - 1)
    return msb, msw, n0


def _shuffled_mean_squares(resid: np.ndarray, clone_idx: np.ndarray, strata: np.ndarray | None):
    """Exact expected mean squares when clone labels are shuffled among cells within strata.

    Shuffling keeps each clone's number of cells in every stratum, so a clone's cells in stratum ``s`` are a
    simple random sample of that stratum's cells: their sum has mean ``n_cs * mean_s`` and variance
    ``n_cs (N_s - n_cs) / (N_s - 1) * var_s``. No Monte Carlo error.
    """
    resid = np.asarray(resid, dtype=float)
    n_clones = int(clone_idx.max()) + 1
    s_idx = np.zeros(clone_idx.size, dtype=np.int64) if strata is None else np.unique(strata, return_inverse=True)[1]
    n_strata = int(s_idx.max()) + 1
    n_cs = sp.csr_matrix((np.ones(clone_idx.size), (clone_idx, s_idx)), shape=(n_clones, n_strata)).toarray()
    n_s = n_cs.sum(axis=0)
    smember = sp.csr_matrix((np.ones(s_idx.size), (s_idx, np.arange(s_idx.size))), shape=(n_strata, s_idx.size))
    mean_s = np.asarray(smember @ resid) / n_s[:, None]
    var_s = np.maximum(np.asarray(smember @ (resid * resid)) / n_s[:, None] - mean_s**2, 0.0)
    fpc = np.divide(n_cs * (n_s - n_cs), n_s - 1, out=np.zeros_like(n_cs), where=n_s > 1)
    expected_sq = fpc @ var_s + (n_cs @ mean_s) ** 2
    between = (expected_sq / n_cs.sum(axis=1)[:, None]).sum(axis=0)
    n_tot = float(clone_idx.size)
    msb = (between - resid.sum(axis=0) ** 2 / n_tot) / (n_clones - 1)
    msw = ((resid * resid).sum(axis=0) - between) / (n_tot - n_clones)
    return msb, msw


def _half_log_ratio(msb, msw):
    """Fisher's z of the ICC, ``0.5 * log(msb / msw)``; nan where either mean square is zero."""
    msb, msw = np.asarray(msb, dtype=float), np.asarray(msw, dtype=float)
    ok = (msb > 0) & (msw > 0)
    return np.where(ok, 0.5 * (np.log(np.where(ok, msb, 1.0)) - np.log(np.where(ok, msw, 1.0))), np.nan)


def gene_heritability(
    adata,
    *,
    clone_key: str | None = None,
    context_key: str | None = "auto",
    strata_key: str | None = None,
    layer: str | None = "log_norm",
    genes=None,
    min_frac_expressed: float = 0.05,
    exclude_receptor_genes: bool = True,
    n_perm: int = 100,
    n_boot: int = 200,
    chunk_size: int = 1000,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clonal ICC of every gene, with a within-stratum permutation p-value.

    Parameters
    ----------
    clone_key, context_key
        Default to the values used by :func:`threadfin.tl.clone_profiles`
        (``context_key="auto"``); pass ``None`` to disable context centring.
    strata_key
        Permutation strata (defaults to the context), kept within donors as in
        :func:`threadfin.tl.clonal_coherence`.
    layer
        Log-normalised expression layer (``None`` uses ``adata.X``).
    genes
        Genes to test (default: genes expressed in at least
        ``min_frac_expressed`` of the analysed cells).
    exclude_receptor_genes
        Leave immunoglobulin/TCR genes out of the scan. They encode the
        receptor itself, so they are clonal by definition; their median ICC is
        still reported (``uns[...]['receptor_gene_icc']``) as a positive
        control for the statistic.
    n_perm
        Permutations (shared across genes, so the cost is ``n_perm`` matrix
        products per chunk of genes).
    n_boot
        Clone-bootstrap replicates of every gene's ``excess_fisher_z``: clones
        drawn with replacement within strata (each clone in the stratum
        holding most of its cells). :func:`geneset_heritability` takes its
        standard errors from them. ``0`` skips them.

    Returns
    -------
    DataFrame indexed by gene: ``icc``, ``tau2``, ``sigma2``, ``mean``,
    ``frac_expressed``, ``null_mean``, ``pvalue``, ``fdr`` and ``excess_icc``
    (``icc - null_mean``, the effect size), sorted by ``excess_icc``. In large
    datasets nearly every gene passes FDR < 0.05; interpret ``excess_icc``.
    ``excess_fisher_z`` is the same comparison on Fisher's z scale of the ICC,
    ``0.5 * log(MSB / MSW)`` (between- and within-clone mean squares, not
    truncated at zero), minus the same quantity for the mean squares expected
    under the shuffles (exact, no Monte Carlo error); its sampling variance
    hardly depends on the ICC, and :func:`geneset_heritability` tests on it.
    Also stored in ``adata.uns['threadfin']['gene_heritability']``. The
    bootstrap replicates of ``excess_fisher_z`` (genes x replicates) are
    attached to the table as ``table.attrs['clone_bootstrap']`` and stored in
    ``adata.uns['threadfin']['gene_heritability_bootstrap']``.
    """
    require_positive_int(n_perm, "n_perm")
    require_positive_int(chunk_size, "chunk_size")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError("n_boot must be a non-negative integer.")
    if not 0 <= min_frac_expressed <= 1:
        raise ValueError("min_frac_expressed must be between 0 and 1.")
    prof = get_uns(adata).get("profiles", {})
    params = prof.get("params", {})
    clone_key = clone_key or params.get("clone_key", "clone_id")
    if context_key == "auto":
        context_key = params.get("context_key")
    strata_key = strata_key or context_key
    require_obs(adata, clone_key, context_key, strata_key, context="gene heritability")

    require_complete_groups(adata, context_key, strata_key, context="gene heritability")
    x = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(x):
        x = x.tocsc()
    clone_codes, _ = codes(adata.obs[clone_key])
    if not (clone_codes >= 0).any():
        raise ValueError("Too few cells in clones with >= 2 cells.")
    sizes = np.bincount(clone_codes[clone_codes >= 0])
    in_multi = (clone_codes >= 0) & (sizes[np.maximum(clone_codes, 0)] >= 2)
    cells = np.flatnonzero(in_multi)
    if cells.size < 20:
        raise ValueError("Too few cells in clones with >= 2 cells.")

    # context means come from all cells of each context (like the profile model)
    if context_key is not None:
        ctx, ctx_index = codes(adata.obs[context_key])
    else:
        ctx, ctx_index = np.zeros(adata.n_obs, dtype=np.int64), pd.Index(["all"])
    ctx_member = sp.csr_matrix(
        (np.ones((ctx >= 0).sum()), (ctx[ctx >= 0], np.flatnonzero(ctx >= 0))),
        shape=(len(ctx_index), adata.n_obs),
    )
    ctx_n = np.asarray(ctx_member.sum(axis=1)).ravel()

    var_names = pd.Index(adata.var_names)
    if genes is None:
        sub = x[cells] if not sp.issparse(x) else x.tocsr()[cells]
        nnz = np.asarray((sub > 0).sum(axis=0)).ravel() if sp.issparse(sub) else (sub > 0).sum(axis=0)
        gene_idx = np.flatnonzero(nnz / cells.size >= min_frac_expressed)
    else:
        gene_idx = var_names.get_indexer(pd.Index(genes))
        if (gene_idx < 0).any():
            missing = list(pd.Index(genes)[gene_idx < 0][:5])
            raise KeyError(f"genes not in var_names, e.g. {missing}")

    from .pp import ig_gene_mask

    receptor = ig_gene_mask(var_names[gene_idx], constant=True, tr_genes=True)
    control_idx = gene_idx[receptor]
    if exclude_receptor_genes:
        gene_idx = gene_idx[~receptor]

    strata = nested_strata(adata.obs, strata_key, params.get("donor_key"))[0]
    if strata is not None:
        strata = strata[cells]
    cl = clone_codes[cells]
    rng = np.random.default_rng(random_state)
    perms = [permute_within(cl, strata, rng) for _ in range(n_perm)]
    clone_idx = np.unique(cl, return_inverse=True)[1].ravel()
    ones = np.ones((1, int(clone_idx.max()) + 1))
    if n_boot:
        # its own generator, so that the permutations (and p-values) do not depend on n_boot
        boot_rng = np.random.default_rng(None if random_state is None else [int(random_state), 1])
        boot_weights = _clone_bootstrap_weights(clone_idx, strata, n_boot, boot_rng)

    rows, boot_rows = [], []
    for start in range(0, gene_idx.size, chunk_size):
        cols = gene_idx[start:start + chunk_size]
        full = _dense_chunk(x, cols)
        mu = np.asarray(ctx_member @ full) / np.maximum(ctx_n, 1)[:, None]
        resid = full[cells] - mu[np.maximum(ctx[cells], 0)]
        vc = variance_components(resid, cl)
        obs_icc = vc.icc_per_feature
        exceed = np.zeros(cols.size)
        null_sum = np.zeros(cols.size)
        for perm in perms:
            null_icc = variance_components(resid, perm).icc_per_feature
            exceed += null_icc >= obs_icc - 1e-12
            null_sum += null_icc
        # Fisher's z scale: centred on the exactly expected mean squares under the same shuffles
        centre = _half_log_ratio(*_shuffled_mean_squares(resid, clone_idx, strata))
        msb, msw, _ = _mean_squares(resid, clone_idx, ones)
        if n_boot:
            bmsb, bmsw, _ = _mean_squares(resid, clone_idx, boot_weights)
            boot_rows.append((_half_log_ratio(bmsb, bmsw) - centre).T.astype(np.float32))
        sub = full[cells]
        rows.append(pd.DataFrame({
            "icc": obs_icc, "tau2": vc.tau2, "sigma2": vc.sigma2,
            "mean": sub.mean(axis=0), "frac_expressed": (sub > 0).mean(axis=0),
            "null_mean": null_sum / max(n_perm, 1),
            "pvalue": (1 + exceed) / (n_perm + 1),
            "excess_fisher_z": _half_log_ratio(msb[0], msw[0]) - centre,
        }, index=var_names[cols]))
    if not rows:
        raise ValueError("No genes remain after expression and receptor filtering.")
    out = pd.concat(rows)
    out["fdr"] = bh_fdr(out["pvalue"].to_numpy())
    # effect size: clonal ICC beyond what sample structure alone produces; with tens of
    # thousands of cells almost every gene is "significant", so rank by this instead
    out["excess_icc"] = out["icc"] - out["null_mean"]
    out = out[[c for c in out.columns if c != "excess_fisher_z"] + ["excess_fisher_z"]]
    control_icc = float("nan")
    if exclude_receptor_genes and control_idx.size:
        full = _dense_chunk(x, control_idx[:200])
        mu = np.asarray(ctx_member @ full) / np.maximum(ctx_n, 1)[:, None]
        control_icc = float(np.median(variance_components(full[cells] - mu[np.maximum(ctx[cells], 0)], cl)
                                      .icc_per_feature))
    out = out.sort_values("excess_icc", ascending=False)
    out.index.name = "gene"
    stored = h5ad_safe(out)
    stored.attrs = {}
    get_uns(adata)["gene_heritability"] = stored
    get_uns(adata)["gene_heritability_receptor_icc"] = control_icc
    if n_boot:
        boot_table = pd.DataFrame(np.vstack(boot_rows), index=var_names[gene_idx],
                                  columns=[f"b{i}" for i in range(n_boot)]).reindex(out.index)
        boot_table.index.name = "gene"
        get_uns(adata)["gene_heritability_bootstrap"] = boot_table
        out.attrs[BOOTSTRAP_ATTR] = _Replicates(boot_table)
    log(
        f"gene heritability: {out.shape[0]} genes, {cells.size} cells in "
        f"{int((sizes >= 2).sum())} clones; {int((out['fdr'] < 0.05).sum())} genes shared within clones "
        f"(clonal ICC above shuffled clones) at FDR < 0.05 (median ICC {out['icc'].median():.3f}; "
        f"receptor-gene positive control {control_icc:.2f}).",
        verbose,
    )
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def _bootstrap_table(table: pd.DataFrame, bootstrap) -> pd.DataFrame:
    if bootstrap is None:
        bootstrap = table.attrs.get(BOOTSTRAP_ATTR)
    if isinstance(bootstrap, _Replicates):
        bootstrap = bootstrap.table
    if bootstrap is None or "excess_fisher_z" not in table.columns:
        raise ValueError(
            "geneset_heritability needs the clone bootstrap of gene_heritability (threadfin >= 4.6, n_boot > 0). "
            "It travels with the table that gene_heritability returns; for a table read back from disk pass "
            "bootstrap=adata.uns['threadfin']['gene_heritability_bootstrap'].")
    bootstrap = pd.DataFrame(bootstrap)
    if bootstrap.shape[1] < 2:
        raise ValueError("At least two bootstrap replicates are needed.")
    if bootstrap.shape[1] < 100:
        warnings.warn(f"Only {bootstrap.shape[1]} bootstrap replicates; standard errors are imprecise "
                      "(use n_boot >= 100 in gene_heritability).", UserWarning, stacklevel=3)
    return bootstrap


def geneset_heritability(table: pd.DataFrame, gene_sets: dict[str, list[str]], *, n_bins: int = 10,
                         seed: int = 0, bootstrap=None) -> pd.DataFrame:
    """Test whether named gene sets are more clonal than genes of similar expression.

    Highly expressed genes are measured with less noise and therefore have
    higher ICCs, so each set is compared with an **expression-matched**
    background: all other tested genes in the same mean-expression bin
    (``n_bins`` quantile bins), each bin weighted by its share of the set.

    The test is competitive: is the set's mean ``excess_fisher_z`` (the excess
    clonal ICC on Fisher's z scale, whose sampling variance hardly depends on
    the ICC) different from that of its matched background? Under the null
    hypothesis the set's genes are like other genes of the same expression
    level, so the variance of the difference has two parts:

    * ``se_genes``: genes differ in how clonal they truly are. The spread of
      true values between genes of one bin is the variance of
      ``excess_fisher_z`` across the bin's genes minus their mean bootstrap
      variance.
    * ``se_clones``: the error from sampling clones, from the clone bootstrap
      of :func:`gene_heritability` (clones resampled with replacement within
      strata, every gene from the same resampled clones). It carries the
      correlation between genes, whether it comes from co-expression within
      cells or from programmes shared within clones.

    Genes of one set are usually co-expressed, so their estimates rise and fall
    together; the Mann-Whitney test used up to Threadfin 4.5.3 treated them as
    independent and rejected 15 % of co-expressed null sets at alpha = 0.05.
    This is the competitive-test problem described by Goeman & Buehlmann (2007)
    and Wu & Smyth (2012, camera); here the correlation is measured by
    resampling clones instead of being estimated from residuals. The p-value is
    two-sided, from ``difference / se`` and the normal distribution. Like the
    permutation test of each gene, the inference is conditional on the sampled
    strata (samples or donors). In simulations with co-expressed and clonal
    gene modules the test rejected 4.4-6.3 % of null sets at alpha = 0.05,
    and 3.8-5.1 % of random gene sets of the gut case study. Shuffling clone
    labels is no reference for this comparison: it removes every gene's clonal
    component, and with it the variation that comes from sampling clones.

    Parameters
    ----------
    table
        Output of :func:`gene_heritability`.
    gene_sets
        ``{name: [genes]}``; sets with fewer than 3 tested genes are skipped.
    n_bins
        Mean-expression bins used to match the background.
    seed
        Random draw of background genes for ``pvalue_mw_genes_independent``
        only.
    bootstrap
        Clone-bootstrap replicates (genes x replicates). Taken from
        ``table.attrs`` when the table comes straight from
        :func:`gene_heritability`; for a table read back from disk, pass
        ``adata.uns['threadfin']['gene_heritability_bootstrap']``.

    Returns
    -------
    DataFrame, one row per set: ``median_icc`` and
    ``matched_background_median_icc``, the median ``excess_icc`` of the set
    and of its matched background (descriptive); ``difference``, the set's
    mean ``excess_fisher_z`` minus the background's, with its standard error
    ``se`` (from ``se_genes`` and ``se_clones``) and 95 % interval
    (``ci_low``, ``ci_high``); ``pvalue_vs_matched`` (the test above) and its
    Benjamini-Hochberg ``fdr`` across sets; ``frac_significant`` (set genes at
    gene-level FDR < 0.05); ``pvalue_mw_genes_independent``, the previous
    Mann-Whitney test against a random matched draw, which ignores the
    correlation between genes and is kept for comparison only; ``method``;
    ``genes``.
    """
    from scipy.stats import mannwhitneyu, norm

    boot = _bootstrap_table(table, bootstrap)
    reps = boot.reindex(table.index).to_numpy(float)
    col = "excess_icc" if "excess_icc" in table.columns else "icc"
    values = table[col].to_numpy(float)
    fz = table["excess_fisher_z"].to_numpy(float)
    testable = np.isfinite(fz) & np.isfinite(reps).all(axis=1)
    rng = np.random.default_rng(seed)
    bins = pd.Series(pd.qcut(table["mean"].rank(method="first"), n_bins, labels=False), index=table.index)
    bin_codes = bins.to_numpy()
    # spread of true values between genes of one bin: variance across genes minus mean sampling variance
    boot_var = np.full(fz.size, np.nan)
    boot_var[testable] = reps[testable].var(axis=1, ddof=1)
    spread = np.zeros(n_bins)
    for b in range(n_bins):
        members = testable & (bin_codes == b)
        if members.sum() > 2:
            spread[b] = max(float(np.var(fz[members], ddof=1) - np.mean(boot_var[members])), 0.0)
    position = pd.Series(np.arange(table.shape[0]), index=table.index)
    z975 = norm.ppf(0.975)
    method = (f"excess Fisher z, set minus expression-matched background; SE from the spread between genes "
              f"and a clone bootstrap ({boot.shape[1]} replicates)")
    rows = []
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in table.index]
        if len(present) < 3:
            continue
        others = table.index.difference(present)
        # the previous test, kept for comparison: a random draw of up to 20 matched genes per set gene
        background = []
        for b, k in bins.loc[present].value_counts().items():
            pool = others[bins.loc[others].to_numpy() == b]
            if pool.size:
                background.extend(rng.choice(pool, size=min(pool.size, 20 * k), replace=False))
        inside = table.loc[present, col].to_numpy()
        matched = table.loc[background, col].to_numpy()
        p_mw = mannwhitneyu(inside, matched, alternative="two-sided").pvalue if matched.size else np.nan

        # the test: set mean minus the bin-weighted mean of every matched background gene
        in_set = np.zeros(table.shape[0], dtype=bool)
        in_set[position.loc[present].to_numpy()] = True
        pool_ok = testable & ~in_set
        usable = [g for g in present if testable[position[g]] and (pool_ok & (bin_codes == bins.loc[g])).any()]
        m = len(usable)
        weights = np.zeros(table.shape[0])
        diff = se = se_clones = se_genes = p = med_bg = np.nan
        if m:
            weights[position.loc[usable].to_numpy()] = 1.0 / m
            for b, k in bins.loc[usable].value_counts().items():
                pool = np.flatnonzero(pool_ok & (bin_codes == b))
                weights[pool] = -(k / m) / pool.size
            involved = np.flatnonzero(weights != 0)
            w = weights[involved]
            diff = float(w @ fz[involved])
            se_clones = float(np.std(w @ reps[involved], ddof=1))
            se_genes = float(np.sqrt(np.sum(w**2 * spread[bin_codes[involved]])))
            se = float(np.hypot(se_clones, se_genes))
            p = float(2 * norm.sf(abs(diff) / se)) if se > 0 else np.nan
            bg = involved[w < 0]
            med_bg = _weighted_median(values[bg], -weights[bg])
        rows.append({
            "gene_set": name, "n_genes": len(present), "statistic": col,
            "median_icc": float(np.median(inside)),
            "matched_background_median_icc": med_bg,
            "difference": diff, "se": se, "ci_low": diff - z975 * se, "ci_high": diff + z975 * se,
            "se_genes": se_genes, "se_clones": se_clones,
            "frac_significant": float((table.loc[present, "fdr"] < 0.05).mean()),
            "pvalue_vs_matched": p,
            "pvalue_mw_genes_independent": float(p_mw),
            "method": method,
            "genes": ",".join(present),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["pvalue_vs_matched"].to_numpy())
    return out
