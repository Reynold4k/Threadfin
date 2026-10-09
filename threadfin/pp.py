"""Preprocessing for clone-level analysis.

Why a dedicated step: in B cells, immunoglobulin (IG) transcripts are the
receptor itself. Every cell of a clone expresses the same IGHV/IGKV/IGLV
genes and, after class switching, the same constant-region gene. If those
transcripts enter the highly-variable-gene set, clonally related cells look
similar *because they share a receptor*, not because they share a state —
which inflates every downstream clone-state statistic, and makes isotype
useless as an independent validation label. :func:`ig_gene_mask` and
:func:`prepare_embedding` exclude them by default.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import log, require_complete_groups, require_positive_int
from .expression import FrozenExpressionModel

# V/D/J segments (incl. orphons such as IGKV1OR1-1 and Roman-numeral
# pseudogenes such as IGHVII-1-1) for human (IGHV1-2) and mouse (Ighv1-72).
_IG_SEGMENT = re.compile(r"^IG[HKL][VDJ](\d|I|V|X|OR)", re.IGNORECASE)
# constant regions: IGHM, IGHD, IGHE(P1), IGHG1-4/GP, mouse Ighg2b/Ighg2c,
# IGHA1/2 (mouse Igha), IGKC, IGLC1-7, surrogate light chain IGLL1/5.
_IG_CONSTANT = re.compile(
    r"^(IGH(M|D|E|EP\d*|G\d*[A-Z]?|GP|A\d*)|IGKC|IGLC\d*|IGLL\d*)$", re.IGNORECASE
)
_TR_SEGMENT = re.compile(r"^TR[ABDG][VDJC]", re.IGNORECASE)


def ig_gene_mask(
    var_names,
    *,
    constant: bool = True,
    tr_genes: bool = False,
) -> np.ndarray:
    """Boolean mask of immunoglobulin (and optionally TCR) genes.

    Parameters
    ----------
    var_names
        Gene symbols (``adata.var_names``); human and mouse casing supported.
    constant
        Also flag constant-region genes (IGHM, IGHG1, IGKC, ...). Keep
        ``True`` when isotype is used as an external validation label.
    tr_genes
        Also flag TCR segment genes (TRAV, TRBV, ...), useful when T cells
        contaminate a B-cell dataset.

    Notes
    -----
    Real genes with IG-like prefixes (e.g. ``IGHMBP2``, ``IGLON5``, ``JCHAIN``)
    are *not* flagged.
    """
    names = pd.Index(var_names).astype(str)
    mask = names.str.match(_IG_SEGMENT)
    if constant:
        mask = mask | names.str.match(_IG_CONSTANT)
    if tr_genes:
        mask = mask | names.str.match(_TR_SEGMENT)
    return np.asarray(mask, dtype=bool)


def exclude_ig_genes(
    adata,
    *,
    hvg_key: str = "highly_variable",
    constant: bool = True,
    tr_genes: bool = False,
    verbose: bool = True,
):
    """Remove IG (and optionally TR) genes from an existing HVG selection.

    Sets ``adata.var[hvg_key]`` to ``False`` for receptor genes and records
    the mask in ``adata.var['threadfin_receptor_gene']``. Re-run PCA after
    calling this.
    """
    mask = ig_gene_mask(adata.var_names, constant=constant, tr_genes=tr_genes)
    adata.var["threadfin_receptor_gene"] = mask
    if hvg_key in adata.var.columns:
        before = int(adata.var[hvg_key].sum())
        adata.var[hvg_key] = adata.var[hvg_key].astype(bool) & ~mask
        log(
            f"excluded {before - int(adata.var[hvg_key].sum())} receptor genes "
            f"from '{hvg_key}' ({int(mask.sum())} receptor genes in var).",
            verbose,
        )
    return adata


def prepare_embedding(
    adata,
    *,
    counts_layer: str | None = None,
    batch_key: str | None = None,
    n_top_genes: int = 3000,
    n_comps: int = 30,
    exclude_receptor_genes: bool = True,
    integrate: str | None = "harmony",
    key_added: str = "X_threadfin",
    random_state: int = 0,
    verbose: bool = True,
):
    """Standard expression embedding for clone-level analysis.

    normalize_total (1e4) -> log1p -> HVG (``flavor='seurat'``, batch-aware,
    receptor genes excluded) -> scale -> PCA -> optional Harmony on
    ``batch_key``. The log-normalised matrix is kept in
    ``adata.layers['log_norm']`` and ``adata.X`` is left untouched.

    Parameters
    ----------
    counts_layer
        Layer holding raw counts; ``None`` uses ``adata.X``.
    batch_key
        ``obs`` column for batch-aware HVG selection and Harmony
        integration. Integrate over *technical* batches (donor, library),
        not over biology you want to study (tissue, timepoint, sort gate).
    integrate
        ``"harmony"`` (requires ``harmonypy``) or ``None``.
    key_added
        ``obsm`` key for the final embedding (the unintegrated PCA is also
        stored as ``X_pca``).

    Returns
    -------
    ``adata`` (modified in place).
    """
    import scanpy as sc

    require_complete_groups(adata, batch_key, context="prepare_embedding")
    require_positive_int(n_comps, "n_comps")
    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(
            f"counts_layer '{counts_layer}' is not in adata.layers. Store raw counts there or omit "
            "counts_layer to use adata.X."
        )
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError(
            "prepare_embedding needs at least 2 cells and 2 genes. Supply a larger expression matrix "
            "or pass a precomputed embedding to threadfin.run(basis=...)."
        )

    counts = adata.layers[counts_layer] if counts_layer else adata.X
    values = counts.data if sp.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Counts contain NaN, infinite, or negative values. Use finite non-negative raw counts, "
            "or pass a precomputed embedding with threadfin.run(basis=...)."
        )
    tmp = sc.AnnData(X=counts.copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    if batch_key is not None:
        tmp.obs[batch_key] = adata.obs[batch_key].astype(str).values
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    adata.layers["log_norm"] = tmp.X.copy()

    receptor = ig_gene_mask(tmp.var_names, constant=True, tr_genes=True)
    adata.var["threadfin_receptor_gene"] = receptor
    pool = tmp[:, ~receptor] if exclude_receptor_genes else tmp
    if pool.n_vars == 0:
        raise ValueError(
            "No genes remain after receptor-gene exclusion. Provide non-receptor expression genes, "
            "set exclude_receptor_genes=False, or pass a precomputed embedding."
        )
    hv = sc.pp.highly_variable_genes(
        pool, n_top_genes=n_top_genes, batch_key=batch_key, flavor="seurat", inplace=False
    )
    hvg = pd.Series(False, index=tmp.var_names)
    hvg.loc[pool.var_names[np.asarray(hv["highly_variable"], dtype=bool)]] = True
    adata.var["highly_variable"] = hvg.values
    log(
        f"HVG: {int(hvg.sum())} genes (receptor genes "
        f"{'excluded' if exclude_receptor_genes else 'kept'}: "
        f"{int(receptor.sum())} in var).",
        verbose,
    )

    sub = tmp[:, hvg.values].copy()
    if sub.n_vars == 0:
        raise ValueError(
            "No highly variable genes were selected. Check that counts vary across cells, or pass a "
            "precomputed embedding to threadfin.run(basis=...)."
        )
    sc.pp.scale(sub, max_value=10)
    # ARPACK requires fewer components than both axes.  Capping here lets a
    # legitimately small pilot dataset use the same workflow without claiming
    # nonexistent PCA dimensions.
    max_comps = min(sub.n_obs - 1, sub.n_vars - 1)
    if max_comps < 1:
        raise ValueError(
            "The selected expression matrix has fewer than two independent dimensions. "
            "Provide more variable genes/cells or pass a precomputed embedding."
        )
    n_comps_used = min(int(n_comps), max_comps)
    sc.tl.pca(sub, n_comps=n_comps_used, random_state=random_state)
    adata.obsm["X_pca"] = sub.obsm["X_pca"]
    adata.uns["threadfin_pca_variance_ratio"] = sub.uns["pca"]["variance_ratio"]

    emb = sub.obsm["X_pca"]
    if integrate == "harmony" and batch_key is not None:
        try:
            import harmonypy
        except ImportError as e:  # pragma: no cover
            raise ImportError("integrate='harmony' requires harmonypy (pip install harmonypy).") from e
        ho = harmonypy.run_harmony(
            emb, adata.obs[[batch_key]].astype(str), batch_key, random_state=random_state, verbose=False
        )
        z = np.asarray(ho.Z_corr)
        emb = z.T if z.shape[0] != emb.shape[0] else z
        log(f"Harmony integration over '{batch_key}' -> obsm['{key_added}'].", verbose)
    elif integrate not in (None, "harmony"):
        raise ValueError("integrate must be 'harmony' or None.")
    adata.obsm[key_added] = np.ascontiguousarray(emb, dtype=np.float32)
    adata.uns.setdefault("threadfin", {})["prepare_embedding"] = {
        "batch_key": batch_key,
        "n_top_genes": int(n_top_genes),
        "n_comps": int(n_comps_used),
        "exclude_receptor_genes": bool(exclude_receptor_genes),
        "integrate": integrate if batch_key is not None else None,
        "key_added": key_added,
    }
    return adata
