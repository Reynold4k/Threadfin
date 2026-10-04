#!/usr/bin/env python
"""Does a clone's mutation lineage predict what its cells are doing?

Inside one clone, some cells are closer relatives than others: they share more
somatic mutations and sit nearer each other in the clone's lineage tree. If
clonal history carries transcriptional state, then cells that are close in the
tree should also be close in expression - and this should hold *within* a
clone, where differences between clones cannot produce it.

This is the finest-grained test of clonal inheritance available, and it needs
no clonal expansion: three cells from one clone are enough. That makes it the
right test for responses in which clones stay small, such as the first weeks of
a Plasmodium infection, where the largest clone holds eight cells.

The statistic is a within-clone Mantel correlation between the matrix of
mutation distances and the matrix of expression distances, with cells permuted
inside the clone. Clone-level correlations are then combined.

Usage:
    python lineage_vs_state.py <dataset>      # after lineage_trees.py prepare + run_gctree.sh
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def mutation_distances(fasta: Path) -> tuple[list[str], np.ndarray]:
    """Pairwise mutation distances between the cells of one clone, from its aligned sequences."""
    names, seqs, name = [], [], None
    buf: list[str] = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                names.append(name)
                seqs.append("".join(buf))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        names.append(name)
        seqs.append("".join(buf))
    keep = [i for i, n in enumerate(names) if n != "naive"]
    names = [names[i] for i in keep]
    arr = np.array([list(seqs[i]) for i in keep])
    if arr.size == 0:
        return names, np.zeros((0, 0))
    valid = ~np.isin(arr, ["N", "-", "."]).any(axis=0)
    arr = arr[:, valid]
    d = (arr[:, None, :] != arr[None, :, :]).sum(axis=2).astype(float)
    return names, d


def mantel(d1: np.ndarray, d2: np.ndarray, n_perm: int = 2000, seed: int = 0):
    """Correlation between two distance matrices, with the usual within-matrix permutation."""
    from scipy.stats import spearmanr

    n = d1.shape[0]
    iu = np.triu_indices(n, 1)
    a, b = d1[iu], d2[iu]
    if n < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan
    obs = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = spearmanr(d1[np.ix_(p, p)][iu], b).statistic
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))


def main(dataset: str, min_cells: int = 3):
    import anndata as ad  # noqa: F401
    from scipy.spatial.distance import squareform, pdist
    from scipy.stats import combine_pvalues, wilcoxon

    from run_case_study import prepare

    out = HERE / "results" / dataset / "lineage"
    cells = pd.read_csv(out / "cells.csv", index_col=0)
    adata, *_ = prepare(dataset, stamp=lambda m: None)
    emb = pd.DataFrame(adata.obsm["X_threadfin"], index=adata.obs_names)

    rows = []
    for name, grp in cells.groupby("file"):
        fasta = out / f"{name}.fasta"
        if not fasta.exists():
            continue
        ids, dmut = mutation_distances(fasta)
        fasta_to_cell = grp.reset_index().set_index("fasta_id")["index"] if "fasta_id" in grp.columns else None
        if fasta_to_cell is None:
            continue
        keep = [i for i, f in enumerate(ids) if f in fasta_to_cell.index
                and fasta_to_cell[f] in emb.index]
        if len(keep) < min_cells:
            continue
        ids = [ids[i] for i in keep]
        dmut = dmut[np.ix_(keep, keep)]
        cell_names = [fasta_to_cell[f] for f in ids]
        dexp = squareform(pdist(emb.loc[cell_names].to_numpy()))
        rho, p = mantel(dmut, dexp)
        if np.isfinite(rho):
            rows.append({"clone": grp["clone"].iloc[0], "n_cells": len(ids),
                         "mutation_spread": float(dmut[np.triu_indices(len(ids), 1)].mean()),
                         "rho": rho, "p_value": p,
                         "group": grp["group"].mode().iloc[0] if "group" in grp.columns else ""})
    if not rows:
        raise SystemExit("no clone had enough cells with both a tree and an embedding")
    res = pd.DataFrame(rows)
    res.to_csv(out / "lineage_vs_state.csv", index=False)

    informative = res[res["mutation_spread"] > 0]
    stat, p_comb = combine_pvalues(informative["p_value"].clip(1e-10, 1), method="fisher")
    w = wilcoxon(informative["rho"]).pvalue if (informative["rho"] != 0).sum() >= 6 else np.nan
    print(f"{dataset}: {len(res)} clones tested, {len(informative)} of them carrying any mutation difference")
    print(f"  median within-clone correlation between mutation distance and expression distance: "
          f"{informative['rho'].median():.3f}")
    print(f"  positive in {100 * (informative['rho'] > 0).mean():.0f}% of clones; "
          f"Wilcoxon p = {w:.3g}; combined p = {p_comb:.3g}")
    print(f"  clones with at least {min_cells} cells, median {informative['n_cells'].median():.0f} cells per clone")


if __name__ == "__main__":
    main(sys.argv[1])
