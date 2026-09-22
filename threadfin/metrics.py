"""Evaluation metrics for clonotype--transcriptional-state integration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def state_concordance(
    adata,
    cluster_key: str = "clone_cluster",
    state_key: str = "leiden",
) -> dict:
    """Agreement between clone clusters and transcriptional cell states.

    Parameters
    ----------
    adata
        AnnData with both ``obs[cluster_key]`` and ``obs[state_key]``.
    cluster_key
        Clone cluster labels (from :func:`threadfin.clonotype_recluster`).
    state_key
        Reference transcriptional state labels (e.g. Leiden cell clusters or
        curated cell-type annotations).

    Returns
    -------
    dict with ``nmi`` (normalized mutual information), ``ari`` (adjusted Rand
    index) and ``n_cells_used``.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    obs = adata.obs[[cluster_key, state_key]].dropna()
    if len(obs) == 0:
        raise ValueError("No cells with both labels present.")
    ari = adjusted_rand_score(obs[state_key], obs[cluster_key])
    nmi = normalized_mutual_info_score(obs[state_key], obs[cluster_key])
    return {"nmi": float(nmi), "ari": float(ari), "n_cells_used": int(len(obs))}


def clone_state_purity(
    adata,
    clone_key: str = "clone_id",
    state_key: str = "leiden",
    min_clone_size: int = 3,
) -> pd.DataFrame:
    """Per-clone transcriptional-state purity.

    For every clone with at least ``min_clone_size`` cells, computes the
    fraction of cells in the clone's dominant state and the Shannon entropy
    of the state distribution (0 = all cells in one state).

    Returns
    -------
    DataFrame indexed by clone id with columns ``n_cells``,
    ``dominant_state``, ``purity``, ``entropy``.
    """
    obs = adata.obs[[clone_key, state_key]].dropna()
    rows = []
    for clone, grp in obs.groupby(clone_key, observed=True):
        if len(grp) < min_clone_size:
            continue
        freq = grp[state_key].value_counts(normalize=True)
        entropy = float(-(freq * np.log2(freq + 1e-12)).sum())
        rows.append(
            {
                clone_key: clone,
                "n_cells": int(len(grp)),
                "dominant_state": freq.index[0],
                "purity": float(freq.iloc[0]),
                "entropy": entropy,
            }
        )
    if not rows:
        raise ValueError("No clones pass min_clone_size.")
    return pd.DataFrame(rows).set_index(clone_key)


def clone_cluster_summary(adata, cluster_key: str = "clone_cluster") -> pd.DataFrame:
    """Cell and clone counts per clone cluster."""
    clone_map = adata.uns.get("threadfin", {}).get("clone_map")
    if clone_map is None:
        raise ValueError("Run threadfin.clonotype_recluster() first.")
    out = (
        clone_map.groupby(cluster_key, observed=True)
        .agg(n_clones=(cluster_key, "size"), n_cells=("n_cells", "sum"))
        .sort_values("n_cells", ascending=False)
    )
    return out


def state_enrichment(
    adata,
    cluster_key: str = "clone_cluster",
    state_key: str = "state",
) -> pd.DataFrame:
    """Fisher-exact enrichment of transcriptional states in clone clusters.

    For every (clone cluster, state) pair, tests whether cells of that state
    are over-represented in the cluster. This is the quantitative evidence
    that clone clusters correspond to biologically meaningful cell states,
    and is more informative than global agreement scores when states are
    imbalanced (e.g. a small plasmablast state among many B-cell cells).

    Returns
    -------
    DataFrame with one row per (cluster, state) pair: ``n_cells``,
    ``odds_ratio``, ``pvalue`` and BH-adjusted ``fdr``, sorted by p-value.
    """
    from scipy.stats import fisher_exact

    obs = adata.obs[[cluster_key, state_key]].dropna()
    clusters = pd.Categorical(obs[cluster_key]).categories
    states = pd.Categorical(obs[state_key]).categories

    rows = []
    n = len(obs)
    for cl in clusters:
        in_cl = (obs[cluster_key] == cl).to_numpy()
        n_cl = int(in_cl.sum())
        for st in states:
            in_st = (obs[state_key] == st).to_numpy()
            a = int((in_cl & in_st).sum())
            b = n_cl - a
            c = int(in_st.sum()) - a
            d = n - a - b - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({cluster_key: cl, state_key: st, "n_cells": a,
                         "odds_ratio": float(odds), "pvalue": float(p)})

    out = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    # Benjamini-Hochberg FDR
    m = len(out)
    ranks = np.arange(1, m + 1)
    fdr = out["pvalue"] * m / ranks
    out["fdr"] = np.minimum.accumulate(fdr.to_numpy()[::-1])[::-1].clip(max=1.0)
    return out
