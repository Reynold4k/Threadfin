#!/usr/bin/env python
"""clone2vec (Isaev et al. 2026, kharchenkolab/clone2vec), run with its documented defaults.

1. ``pp.clones_adata``: the clones with at least ``min_size`` cells (package default 3);
2. ``tl.clonal_nn``: k = 15 nearest neighbours of every cell of those clones in the embedding,
   aggregated into a clone-by-clone co-occurrence graph;
3. ``tl.clone2vec``: skip-gram embedding of the graph, 10 dimensions, SVD initialisation;
4. neighbours + UMAP of the clone embedding (timing of the end-to-end workflow only).

clone2vec builds its graph from the cells of eligible clones only, and has no notion of donor or sample;
its ``split_by`` option builds separate graphs per category, which for donor-private clones would leave
the clone graph disconnected between donors, so it is not used.

Usage: python run_clone2vec.py <dataset> <variant> <min_size> <method_id>
"""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bench_io


def main(dataset, variant, min_size, method):
    import clone2vec as c2v
    import scanpy as sc
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    out = bench_io.out_dir(dataset, variant, method)
    x, cells = bench_io.load(dataset, variant)
    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32), obs=cells[["donor", "sample"]].copy())
    a.obs["clone"] = cells["clone"].astype(object).where(cells["clone"].notna(), "NA").astype(str).values
    a.obsm["X_pca"] = np.asarray(x, dtype=np.float32)
    timer = bench_io.Timer()
    try:
        with timer.stage("clones_adata"):
            clones = c2v.pp.clones_adata(a, obs_name="clone", min_size=min_size, na_value="NA")
        with timer.stage("clonal_nn"):
            c2v.tl.clonal_nn(a, clones, obs_name="clone", k=15, use_rep="X_pca", random_state=4)
        with timer.stage("clone2vec"):
            c2v.tl.clone2vec(clones, z_dim=10, progress_bar=False, random_state=4)
        with timer.stage("umap"):
            sc.pp.neighbors(clones, use_rep="clone2vec", n_neighbors=min(15, clones.n_obs - 1), random_state=0)
            sc.tl.umap(clones, random_state=0)
    except Exception as exc:
        bench_io.fail(out, f"{type(exc).__name__}: {exc}", {"method": method, "min_size": min_size})
        print(f"{dataset} {variant} {method}: FAILED {exc}", flush=True)
        return
    emb = np.asarray(clones.obsm["clone2vec"])
    info = {"method": method, "software": f"clone2vec {c2v.__version__}", "dataset": dataset, "variant": variant,
            "min_size": min_size, "n_cells_input": int(len(cells)), "n_clones": int(clones.n_obs),
            "n_cells_in_graph": int(clones.obs["n_cells"].sum()), "seconds": timer.stages,
            "epochs": int(len(clones.uns["clone2vec"]["loss_history"]))}
    bench_io.write(out, clones.obs_names.to_numpy(), clones.obs["n_cells"].to_numpy(), features=emb, info=info)
    pd.DataFrame(clones.obsm["X_umap"], index=clones.obs_names, columns=["umap1", "umap2"]).to_csv(out / "umap.tsv", sep="\t")
    print(f"{dataset} {variant} {method}: {clones.n_obs} clones, {timer.total:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
