#!/usr/bin/env python
"""WATERMELON: can single-cell RNA see what the clone embedding sees?

The clone map built from day 0/3/7 cells alone carries the clone's later day-14 mCherry gate
(R^2 = 0.053). Does the same information show on a plain single-cell UMAP? Cells of the early
days are coloured by their clone's *measured future* gate fraction; neighbourhood agreement of
that label is quantified on the cell UMAP and on the clone map.

Outputs: case_studies/results/watermelon/cells_umap.csv.gz, cell_vs_clone_visibility.json
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import threadfin as tf

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parents[3] / "case_studies" / "data")) / "gse150949_watermelon"
OUT = HERE / "results" / "watermelon"


def knn_corr(xy: np.ndarray, v: np.ndarray, k: int = 15) -> float:
    """Pearson r between a value and its k-NN mean — a continuous neighbourhood agreement."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    nm = v[idx].mean(axis=1)
    return float(np.corrcoef(v, nm)[0, 1])


def main():
    t0 = time.time()
    def stamp(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    cells = pd.read_csv(OUT / "cells.csv.gz", index_col=0)
    cells.index = cells.index.astype(str)
    early_tab = pd.read_csv(OUT / "early_clone_table.csv")
    gate = early_tab.set_index(early_tab["clone_id"].astype(str))["gate14:mCherry-high"]
    cells["future_gate"] = cells["clone_id"].map(gate)   # each cell carries its clone's future

    stamp("loading counts and building the cell UMAP (X_threadfin basis)")
    meta = cells.copy()
    x = sp.load_npz(DATA / "counts_sparse.npz")
    genes = pd.read_csv(DATA / "gene_names.txt", header=None)[0].astype(str).to_numpy()
    import anndata as ad
    a = ad.AnnData(X=x, obs=meta, var=pd.DataFrame(index=pd.Index(genes)))
    a.var_names_make_unique()
    tf.pp.prepare_embedding(a, batch_key=None, key_added="X_threadfin", verbose=False)
    sc.pp.neighbors(a, use_rep="X_threadfin", n_neighbors=30)
    sc.tl.umap(a, random_state=0)
    a.obs["umap_1"], a.obs["umap_2"] = a.obsm["X_umap"][:, 0], a.obsm["X_umap"][:, 1]
    a.obs[["umap_1", "umap_2", "future_gate"]].to_csv(OUT / "cells_umap.csv.gz")
    stamp("cell UMAP done")

    # early cells only (day 0/3/7), labelled by the clone's measured day-14 gate fraction
    e = a.obs[(a.obs["day"] < 14) & a.obs["future_gate"].notna()]
    v = e["future_gate"].to_numpy()
    cell_r = knn_corr(e[["umap_1", "umap_2"]].to_numpy(), v)
    rng = np.random.default_rng(0)
    null_cell = [knn_corr(e[["umap_1", "umap_2"]].to_numpy(), rng.permutation(v))
                 for _ in range(20)]
    # the same label on the early-only clone map
    et = early_tab.dropna(subset=["gate14:mCherry-high"])
    clone_r = knn_corr(et[["x", "y"]].to_numpy(), et["gate14:mCherry-high"].to_numpy())
    null_clone = [knn_corr(et[["x", "y"]].to_numpy(),
                           rng.permutation(et["gate14:mCherry-high"].to_numpy()))
                  for _ in range(20)]
    out = {"early_cells_with_future_label": int(len(e)),
           "cell_umap_knn_r": cell_r, "cell_umap_null_mean": float(np.mean(null_cell)),
           "cell_umap_null_max": float(np.max(null_cell)),
           "clone_map_knn_r": clone_r, "clone_map_null_mean": float(np.mean(null_clone)),
           "clone_map_null_max": float(np.max(null_clone)),
           "n_clones_early_map_with_label": int(len(et))}
    (OUT / "cell_vs_clone_visibility.json").write_text(json.dumps(out, indent=1) + "\n")
    stamp(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
