"""Gene-level clonal ICC: which genes vary consistently with clone identity?

For every gene, the same one-way random-effects model as
:mod:`threadfin.profiles` is fitted to its context-centred log expression:
the gene's clonal ICC is the fraction of its (within-context) variance that
is explained by clone identity. Genes that define a stable, inherited fate
(for example a plasma-cell commitment programme) are expected to have high
ICC; this is a hypothesis, not evidence of inherited fate. Genes that every clone cycles through (for example dark-zone / light-zone
germinal-centre programmes, or the cell cycle) are expected to have low ICC
even when they are highly variable. Significance comes from shuffling clone
labels among cells within strata, exactly as in
:func:`threadfin.tl.clonal_coherence`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import bh_fdr, codes, get_uns, log, permute_within, require_obs, require_complete_groups, require_positive_int
from .profiles import variance_components


def _dense_chunk(x, cols):
    sub = x[:, cols]
    return sub.toarray() if sp.issparse(sub) else np.asarray(sub)


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
        Permutation strata (defaults to the context).
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

    Returns
    -------
    DataFrame indexed by gene: ``icc``, ``tau2``, ``sigma2``, ``mean``,
    ``frac_expressed``, ``null_mean``, ``pvalue``, ``fdr`` and ``excess_icc``
    (``icc - null_mean``, the effect size), sorted by ``excess_icc``. In large
    datasets nearly every gene passes FDR < 0.05; interpret ``excess_icc``.
    Also stored in ``adata.uns['threadfin']['gene_heritability']``.
    """
    require_positive_int(n_perm, "n_perm")
    require_positive_int(chunk_size, "chunk_size")
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

    strata = codes(adata.obs[strata_key])[0][cells] if strata_key is not None else None
    cl = clone_codes[cells]
    rng = np.random.default_rng(random_state)
    perms = [permute_within(cl, strata, rng) for _ in range(n_perm)]

    rows = []
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
        sub = full[cells]
        rows.append(pd.DataFrame({
            "icc": obs_icc, "tau2": vc.tau2, "sigma2": vc.sigma2,
            "mean": sub.mean(axis=0), "frac_expressed": (sub > 0).mean(axis=0),
            "null_mean": null_sum / max(n_perm, 1),
            "pvalue": (1 + exceed) / (n_perm + 1),
        }, index=var_names[cols]))
    if not rows:
        raise ValueError("No genes remain after expression and receptor filtering.")
    out = pd.concat(rows)
    out["fdr"] = bh_fdr(out["pvalue"].to_numpy())
    # effect size: clonal ICC beyond what sample structure alone produces; with tens of
    # thousands of cells almost every gene is "significant", so rank by this instead
    out["excess_icc"] = out["icc"] - out["null_mean"]
    control_icc = float("nan")
    if exclude_receptor_genes and control_idx.size:
        full = _dense_chunk(x, control_idx[:200])
        mu = np.asarray(ctx_member @ full) / np.maximum(ctx_n, 1)[:, None]
        control_icc = float(np.median(variance_components(full[cells] - mu[np.maximum(ctx[cells], 0)], cl)
                                      .icc_per_feature))
    out = out.sort_values("excess_icc", ascending=False)
    out.index.name = "gene"
    get_uns(adata)["gene_heritability"] = out
    get_uns(adata)["gene_heritability_receptor_icc"] = control_icc
    log(
        f"gene heritability: {out.shape[0]} genes, {cells.size} cells in "
        f"{int((sizes >= 2).sum())} clones; {int((out['fdr'] < 0.05).sum())} genes clonally "
        f"inherited at FDR < 0.05 (median ICC {out['icc'].median():.3f}; receptor-gene positive "
        f"control {control_icc:.2f}).",
        verbose,
    )
    return out


def geneset_heritability(table: pd.DataFrame, gene_sets: dict[str, list[str]], *, n_bins: int = 10,
                         seed: int = 0) -> pd.DataFrame:
    """Summarise :func:`gene_heritability` over named gene sets.

    Highly expressed genes are measured with less noise and therefore have
    higher ICCs, so each set is compared with an **expression-matched**
    background: other tested genes drawn from the same mean-expression deciles
    as the set's genes (up to 20 background genes per set gene). Reports the
    set's median ICC, the matched background median, the fraction of set genes
    significant at FDR < 0.05 and a Mann-Whitney p-value (set vs matched
    background).
    """
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(seed)
    bins = pd.Series(pd.qcut(table["mean"].rank(method="first"), n_bins, labels=False), index=table.index)
    rows = []
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in table.index]
        if len(present) < 3:
            continue
        others = table.index.difference(present)
        background = []
        for b, k in bins.loc[present].value_counts().items():
            pool = others[bins.loc[others].to_numpy() == b]
            if pool.size:
                background.extend(rng.choice(pool, size=min(pool.size, 20 * k), replace=False))
        col = "excess_icc" if "excess_icc" in table.columns else "icc"
        inside = table.loc[present, col].to_numpy()
        matched = table.loc[background, col].to_numpy()
        p = mannwhitneyu(inside, matched, alternative="two-sided").pvalue if matched.size else np.nan
        rows.append({
            "gene_set": name, "n_genes": len(present), "statistic": col,
            "median_icc": float(np.median(inside)),
            "matched_background_median_icc": float(np.median(matched)) if matched.size else np.nan,
            "frac_significant": float((table.loc[present, "fdr"] < 0.05).mean()),
            "pvalue_vs_matched": float(p), "genes": ",".join(present),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["pvalue_vs_matched"].to_numpy())
    return out
