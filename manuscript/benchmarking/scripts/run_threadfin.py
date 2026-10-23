#!/usr/bin/env python
"""Threadfin clone representations, its ablations and the simple baselines.

Methods (all receive the same embedding, clone ids, donors and samples):

``threadfin_kernel``           default Threadfin 4.3.0 profile: kernel mean embedding (256 random Fourier
                               features, median-heuristic bandwidth), cells centred on their donor, each
                               cell averaged with its 15 nearest cells, best linear unbiased predictor
                               (shrinkage by the number of cells), 30 principal components; clones >= 2 cells
``threadfin_mean``             the same model on the embedding coordinates themselves (shrunken centroid)
``threadfin_kernel_nocentre``  ablation: no donor centring
``threadfin_kernel_nosmooth``  ablation: no neighbour smoothing
``threadfin_kernel_noshrink``  ablation: raw clone means of the kernel features (no shrinkage)
``mean_pca``                   baseline: clone mean of the embedding (no centring, no shrinkage)
``composition``                baseline: fraction of the clone's cells in each Leiden cluster of the cells
                               (Hellinger coordinates, i.e. square roots of the fractions)
``conga_medoid``               CoNGA's reduction rule (Schattgen et al. 2022): each clone is represented by
                               the cell with the smallest summed distance to the other cells of the clone

Sensitivity of the default profile to its settings (one setting changed at a time):

``threadfin_kernel_smooth5`` / ``_smooth30``      neighbour smoothing over 5 / 30 cells (default 15; 0 is
                                                  ``threadfin_kernel_nosmooth``)
``threadfin_kernel_rff128`` / ``_rff512`` / ``_rff1024``   number of random Fourier features (default 256)
``threadfin_kernel_bw0.5`` / ``_bw2``             kernel bandwidth 0.5 / 2 times the median heuristic
``threadfin_kernel_pc10`` / ``_pc50``             principal components kept (default 30)
``threadfin_kernel_seed1`` / ``_seed2``           another draw of the random features

Usage: python run_threadfin.py <dataset> <variant> <method> [<method> ...]
"""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bench_io

MIN_CELLS = 2


def _adata(x, cells):
    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32), obs=cells[["clone", "donor", "sample"]].copy())
    a.obs["clone"] = a.obs["clone"].astype(object).where(a.obs["clone"].notna(), None)
    a.obsm["X_emb"] = np.asarray(x, dtype=float)
    return a


def median_heuristic(x, cells, context="donor"):
    """The bandwidth the default profile uses: median distance between context-centred cells."""
    from threadfin._utils import codes
    from threadfin.profiles import _center_by, median_bandwidth

    finite = np.isfinite(x).all(axis=1)
    ctx = codes(cells[context])[0] if context else np.zeros(len(cells), dtype=np.int64)
    resid = _center_by(np.where(finite[:, None], x, 0.0), ctx, finite)
    return median_bandwidth(resid[finite], rng=np.random.default_rng(0))


def threadfin(x, cells, *, representation="kernel", context="donor", smooth=None, shrink=True, n_features=256,
              n_components=30, bandwidth="median", random_state=0):
    import threadfin as tf
    from threadfin.profiles import build_model

    a = _adata(x, cells)
    if isinstance(bandwidth, tuple):                  # ("scale", factor): a multiple of the median heuristic
        bandwidth = bandwidth[1] * median_heuristic(np.asarray(x, dtype=float), cells, context)
    if shrink:
        tab = tf.tl.clone_profiles(a, clone_key="clone", basis="X_emb", context_key=context, donor_key="donor",
                                   representation=representation, smooth=smooth, min_cells=MIN_CELLS,
                                   n_features=n_features, n_components=n_components, bandwidth=bandwidth,
                                   random_state=random_state, verbose=False)
        feats = a.uns["threadfin"]["profiles"]["features"]
        vc = a.uns["threadfin"]["profiles"]["variance_components"]
        return feats.index.to_numpy(), tab["n_cells"].to_numpy(), feats.to_numpy(), {"variance_share_between_clones_of_representation": vc["icc"]}
    # unshrunk clone means of the same context-centred (and smoothed) kernel features
    model = build_model(a, clone_key="clone", basis="X_emb", context_key=context, representation=representation,
                        smooth=15 if smooth is None else smooth)
    g = model.clone_codes
    ok = np.flatnonzero(g >= 0)
    member = sp.csr_matrix((np.ones(ok.size), (g[ok], ok)), shape=(len(model.clone_index), g.size))
    n = np.asarray(member.sum(axis=1)).ravel()
    means = np.asarray(member @ model.residuals) / np.maximum(n, 1)[:, None] - model.vc.grand_mean[None, :]
    keep = n >= MIN_CELLS
    feats = means[keep]
    u, s, _ = np.linalg.svd(feats - feats.mean(axis=0), full_matrices=False)
    k = int(min(30, (s > 1e-12).sum()))
    return model.clone_index[keep].astype(str), n[keep], u[:, :k] * s[:k], {}


def clone_groups(cells, min_size=MIN_CELLS):
    sizes = bench_io.eligible(cells, min_size)
    lab = cells["clone"].to_numpy(dtype=object)
    idx = {c: i for i, c in enumerate(sizes.index)}
    code = np.array([idx.get(c, -1) if isinstance(c, str) else -1 for c in lab])
    return sizes, code


def mean_pca(x, cells):
    sizes, code = clone_groups(cells)
    ok = code >= 0
    m = sp.csr_matrix((np.ones(ok.sum()), (code[ok], np.flatnonzero(ok))), shape=(len(sizes), len(cells)))
    feats = np.asarray(m @ x) / sizes.to_numpy()[:, None]
    return sizes.index.to_numpy(), sizes.to_numpy(), feats, {}


def composition(x, cells, resolution=1.0, seed=0):
    import scanpy as sc

    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32))
    a.obsm["X"] = np.asarray(x, dtype=float)
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    cl = a.obs["leiden"].astype(int).to_numpy()
    sizes, code = clone_groups(cells)
    ok = code >= 0
    counts = sp.csr_matrix((np.ones(ok.sum()), (code[ok], cl[ok])), shape=(len(sizes), cl.max() + 1)).toarray()
    frac = counts / counts.sum(axis=1, keepdims=True)
    return sizes.index.to_numpy(), sizes.to_numpy(), np.sqrt(frac), {"n_cell_clusters": int(cl.max() + 1),
                                                                      "leiden_resolution": resolution}


def conga_medoid(x, cells):
    from sklearn.metrics import pairwise_distances

    sizes, code = clone_groups(cells)
    feats = np.zeros((len(sizes), x.shape[1]))
    order = np.argsort(code, kind="stable")
    bounds = np.searchsorted(code[order], np.arange(len(sizes) + 1))
    for c in range(len(sizes)):
        rows = order[bounds[c]:bounds[c + 1]]
        if rows.size == 1:
            feats[c] = x[rows[0]]
            continue
        d = pairwise_distances(x[rows])
        feats[c] = x[rows[np.argmin(d.sum(axis=1))]]
    return sizes.index.to_numpy(), sizes.to_numpy(), feats, {}


METHODS = {
    "threadfin_kernel": lambda x, c: threadfin(x, c),
    "threadfin_mean": lambda x, c: threadfin(x, c, representation="mean", smooth=0),
    "threadfin_kernel_nocentre": lambda x, c: threadfin(x, c, context=None),
    "threadfin_kernel_nosmooth": lambda x, c: threadfin(x, c, smooth=0),
    "threadfin_kernel_noshrink": lambda x, c: threadfin(x, c, shrink=False),
    "threadfin_kernel_smooth5": lambda x, c: threadfin(x, c, smooth=5),
    "threadfin_kernel_smooth30": lambda x, c: threadfin(x, c, smooth=30),
    "threadfin_kernel_rff128": lambda x, c: threadfin(x, c, n_features=128),
    "threadfin_kernel_rff512": lambda x, c: threadfin(x, c, n_features=512),
    "threadfin_kernel_rff1024": lambda x, c: threadfin(x, c, n_features=1024),
    "threadfin_kernel_bw0.5": lambda x, c: threadfin(x, c, bandwidth=("scale", 0.5)),
    "threadfin_kernel_bw2": lambda x, c: threadfin(x, c, bandwidth=("scale", 2.0)),
    "threadfin_kernel_pc10": lambda x, c: threadfin(x, c, n_components=10),
    "threadfin_kernel_pc50": lambda x, c: threadfin(x, c, n_components=50),
    "threadfin_kernel_seed1": lambda x, c: threadfin(x, c, random_state=1),
    "threadfin_kernel_seed2": lambda x, c: threadfin(x, c, random_state=2),
    "mean_pca": mean_pca,
    "composition": composition,
    "conga_medoid": conga_medoid,
}


SENSITIVITY = ["threadfin_kernel_smooth5", "threadfin_kernel_smooth30", "threadfin_kernel_rff128",
               "threadfin_kernel_rff512", "threadfin_kernel_rff1024", "threadfin_kernel_bw0.5", "threadfin_kernel_bw2",
               "threadfin_kernel_pc10", "threadfin_kernel_pc50", "threadfin_kernel_seed1", "threadfin_kernel_seed2"]
DEFAULT_METHODS = [m for m in METHODS if m not in SENSITIVITY]


def main(dataset, variant, methods):
    import threadfin

    x, cells = bench_io.load(dataset, variant)
    for method in methods:
        out = bench_io.out_dir(dataset, variant, method)
        timer = bench_io.Timer()
        try:
            with timer.stage("representation"):
                clones, n, feats, extra = METHODS[method](x, cells)
        except Exception as exc:  # recorded, never silently dropped
            bench_io.fail(out, f"{type(exc).__name__}: {exc}", {"method": method})
            print(f"{dataset} {variant} {method}: FAILED {exc}", flush=True)
            continue
        info = {"method": method, "software": f"threadfin {threadfin.__version__}", "dataset": dataset,
                "variant": variant, "n_cells_input": int(len(cells)), "n_clones": int(len(clones)),
                "min_cells": MIN_CELLS, "seconds": timer.stages, **extra}
        bench_io.write(out, clones, n, features=feats, info=info)
        print(f"{dataset} {variant} {method}: {len(clones)} clones, {timer.total:.1f} s", flush=True)


if __name__ == "__main__":
    chosen = sys.argv[3:] or DEFAULT_METHODS
    main(sys.argv[1], sys.argv[2], SENSITIVITY if chosen == ["sensitivity"] else chosen)
