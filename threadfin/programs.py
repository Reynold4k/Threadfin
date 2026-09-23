"""Community-level gene programmes: markers and signature scores."""

from __future__ import annotations

import warnings

import pandas as pd
import scanpy as sc


def _require_obs(adata, col: str) -> None:
    if col not in adata.obs.columns:
        raise KeyError(f"'{col}' not found in adata.obs.")


def _rank_genes_df(ad) -> pd.DataFrame:
    """rank_genes_groups results as a tidy DataFrame across scanpy versions."""
    if hasattr(sc.get, "rank_genes_groups_df"):
        df = sc.get.rank_genes_groups_df(ad, group=None)
        df = df.rename(
            columns={
                "group": "community",
                "names": "gene",
                "scores": "score",
                "logfoldchanges": "logfoldchange",
                "pvals_adj": "pval_adj",
            }
        )
    else:
        rgg = ad.uns["rank_genes_groups"]
        rows = []
        for group in rgg["names"].dtype.names:
            n = len(rgg["names"][group])
            rows.append(
                pd.DataFrame(
                    {
                        "community": group,
                        "gene": rgg["names"][group],
                        "score": rgg["scores"][group],
                        "logfoldchange": rgg.get("logfoldchanges", [None] * n)[group],
                        "pval_adj": rgg.get("pvals_adj", [None] * n)[group],
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
    return df[["community", "gene", "score", "logfoldchange", "pval_adj"]]


def community_markers(
    adata,
    *,
    cluster_key: str = "clone_cluster",
    method: str = "wilcoxon",
    n_genes: int = 25,
    layer: str | None = None,
) -> pd.DataFrame:
    """Differentially expressed genes per clone community.

    Cells without a ``cluster_key`` label are excluded. Runs scanpy
    ``rank_genes_groups`` (each community vs. the rest) on a copy of the
    labelled cells.

    Returns
    -------
    Tidy DataFrame with columns ``[community, gene, score, logfoldchange,
    pval_adj]``, top ``n_genes`` per community.
    """
    _require_obs(adata, cluster_key)
    ad = adata[adata.obs[cluster_key].notna()].copy()
    if isinstance(ad.obs[cluster_key].dtype, pd.CategoricalDtype):
        ad.obs[cluster_key] = ad.obs[cluster_key].cat.remove_unused_categories()
    if ad.obs[cluster_key].nunique() < 2:
        raise ValueError(f"Need at least two communities in '{cluster_key}'.")
    sc.tl.rank_genes_groups(
        ad,
        groupby=cluster_key,
        method=method,
        layer=layer,
        n_genes=min(n_genes, ad.n_vars),
    )
    return _rank_genes_df(ad).reset_index(drop=True)


def community_score(
    adata,
    signatures: dict[str, list[str]],
    *,
    cluster_key: str = "clone_cluster",
    score_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Mean per-community score of named gene programmes.

    Each signature is scored per cell with scanpy ``score_genes`` over the
    genes present in ``var_names`` (missing genes are dropped); signatures
    with no gene present are skipped with a warning. Cells without a
    ``cluster_key`` label are excluded from the per-community means.

    Returns
    -------
    Tidy DataFrame with columns ``[community, signature, score, n_cells]``.
    """
    _require_obs(adata, cluster_key)
    score_kwargs = dict(score_kwargs or {})
    clusters = adata.obs[cluster_key]

    rows = []
    skipped = []
    for name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            skipped.append(name)
            continue
        col = f"_threadfin_score_{name}"
        sc.tl.score_genes(adata, gene_list=present, score_name=col, **score_kwargs)
        df = pd.DataFrame({"community": clusters, "score": adata.obs[col]}).dropna()
        del adata.obs[col]
        for community, grp in df.groupby("community", observed=True):
            rows.append(
                {
                    "community": community,
                    "signature": name,
                    "score": float(grp["score"].mean()),
                    "n_cells": int(len(grp)),
                }
            )
    if skipped:
        warnings.warn(
            f"Signatures skipped, no genes found in var_names: {skipped}",
            UserWarning,
            stacklevel=2,
        )
    out = pd.DataFrame(rows, columns=["community", "signature", "score", "n_cells"])
    return out.sort_values(["signature", "community"]).reset_index(drop=True)
