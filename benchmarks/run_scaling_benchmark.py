#!/usr/bin/env python
"""Scaling benchmark: synthetic paired GEX+BCR data, 2k -> 200k cells.

Times each Threadfin stage and records peak RSS. Writes
benchmarks/results/scaling_report.json and scaling.png.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from anndata import AnnData

import threadfin as tf

OUTDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results")
OUTDIR.mkdir(parents=True, exist_ok=True)
proc = psutil.Process()
_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_data(n_cells, clone_ratio, rng):
    n_clones = max(10, n_cells // clone_ratio)
    d, n_states = 10, 4
    centers = rng.normal(0, 6.0, size=(n_states, d))
    clone_home = rng.integers(0, n_states, size=n_clones)
    cell_clone = rng.integers(0, n_clones, size=n_cells)
    states = np.where(rng.random(n_cells) < 0.85,
                      clone_home[cell_clone], rng.integers(0, n_states, size=n_cells))
    emb = centers[states] + rng.normal(0, 1.0, size=(n_cells, d))

    obs = pd.DataFrame({"state": [f"s{s}" for s in states]},
                       index=[f"c{i}" for i in range(n_cells)])
    adata = AnnData(X=np.zeros((n_cells, 50), dtype=np.float32), obs=obs)
    adata.obsm["X_umap"] = emb[:, :2]

    # per-clone constant VDJ + CDR3
    v = np.array([f"IGHV{h + 1}-{c % 40}*01" for c, h in enumerate(clone_home)])
    cdr3 = ["".join(rng.choice(list(_AA), size=14)) for _ in range(n_clones)]
    bcr = pd.DataFrame(
        {"clone_id": [f"CL{c:06d}" for c in cell_clone],
         "v_call": v[cell_clone], "d_call": "IGHD1-1*01",
         "j_call": "IGHJ4*01", "cdr3": np.array(cdr3, dtype=object)[cell_clone]},
        index=adata.obs_names,
    )
    return adata, bcr


rows = []
for n_cells, clone_ratio in [(2_000, 10), (10_000, 12), (50_000, 15), (200_000, 20)]:
    rng = np.random.default_rng(0)
    print(f"[scaling] {n_cells} cells ...", flush=True)
    adata, bcr = make_data(n_cells, clone_ratio, rng)
    row = {"n_cells": n_cells, "n_clones": int(bcr["clone_id"].nunique())}

    t0 = time.perf_counter()
    tf.attach_bcr(adata, bcr)
    row["attach_bcr_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    tf.clone_centroids(adata, min_clone_size=3)
    row["centroids_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    m0 = proc.memory_info().rss
    tf.clonotype_recluster(adata, n_neighbors=15, resolution=0.5, random_state=0)
    row["recluster_s"] = round(time.perf_counter() - t0, 3)
    row["recluster_peak_rss_gb"] = round(proc.memory_info().rss / 1e9, 2)

    t0 = time.perf_counter()
    tf.clonal_pseudotime(adata)
    row["pseudotime_s"] = round(time.perf_counter() - t0, 3)

    print(f"[scaling] {n_cells} cells done: {row}", flush=True)
    rows.append(row)
    del adata, bcr

df = pd.DataFrame(rows)
df.to_json(OUTDIR / "scaling_report.json", orient="records", indent=2)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(df["n_cells"], df["recluster_s"], "o-", label="clonotype_recluster (full)")
ax.plot(df["n_cells"], df["centroids_s"], "s-", label="clone_centroids")
ax.plot(df["n_cells"], df["attach_bcr_s"], "^-", label="attach_bcr")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of cells")
ax.set_ylabel("wall time (s)")
ax.set_title("Threadfin scaling (synthetic data)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "scaling.png", dpi=300)
print("[done]", OUTDIR / "scaling_report.json", flush=True)
