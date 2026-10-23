#!/usr/bin/env python
"""Does the analysis scale to atlas-size data? A synthetic repertoire of up to a million cells.

Cells get a 30-dimensional embedding (as from PCA) and clone ids with a heavy-tailed size distribution like a real
repertoire (most clones are seen once); clones belong to one of ten donors and cells to one of 40 samples. Every
step of `threadfin.run` after the embedding is timed with its peak memory: clone profiles, the coherence test
(200 permutations), the clone map with automatically chosen settings, programmes with 30 bootstrap replicates,
and one clone-level label test.

Usage: python atlas_scale.py <n_cells> <n_jobs> <out.json>
       python atlas_scale.py --plot <results_dir>
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

import threadfin as tf


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def synthetic(n_cells, dim=30, seed=0):
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n_cells:                      # Zipf-like clone sizes, truncated at 2,000 cells
        sizes.append(int(min(rng.zipf(2.0), 2000)))
    sizes = np.array(sizes)
    sizes[-1] -= sizes.sum() - n_cells
    sizes = sizes[sizes > 0]
    clone = np.repeat(np.arange(sizes.size), sizes)
    donor = (np.arange(sizes.size) % 10)[clone]
    programme = rng.integers(0, 6, sizes.size)
    centres = rng.normal(0, 3, size=(6, dim))
    x = centres[programme[clone]] + rng.normal(0, 1, size=(n_cells, dim)) + rng.normal(0, 0.5, (10, dim))[donor]
    sample = donor * 4 + rng.integers(0, 4, n_cells)
    obs = pd.DataFrame({"clone_id": [f"d{d}|c{c}" for d, c in zip(donor, clone)], "donor": [f"d{d}" for d in donor],
                        "sample": [f"s{s}" for s in sample], "label": (programme[clone] % 2).astype(str)},
                       index=[f"cell{i}" for i in range(n_cells)])
    adata = AnnData(X=sp.csr_matrix((n_cells, 0), dtype=np.float32), obs=obs)
    adata.obsm["X_emb"] = x.astype(np.float32)
    return adata


def main(n_cells, n_jobs, out):
    t0 = time.perf_counter()
    adata = synthetic(n_cells)
    sizes = adata.obs["clone_id"].value_counts()
    rec = {"n_cells": n_cells, "n_jobs": n_jobs, "threadfin": tf.__version__, "clones": int(sizes.size),
           "clones_ge2": int((sizes >= 2).sum()), "steps": {}}
    print(json.dumps(rec), flush=True)

    def step(name, fn):
        t = time.perf_counter()
        fn()
        rec["steps"][name] = {"seconds": round(time.perf_counter() - t, 1), "peak_gb": round(peak_gb(), 2)}
        print(name, rec["steps"][name], flush=True)

    step("clone_profiles", lambda: tf.tl.clone_profiles(adata, basis="X_emb", context_key="sample", donor_key="donor",
                                                        representation="kernel", verbose=False))
    step("clonal_coherence", lambda: tf.tl.clonal_coherence(adata, strata_key="sample", n_perm=200, verbose=False))
    step("clone_map", lambda: tf.tl.embed_clones(adata, n_jobs=n_jobs, verbose=False))
    step("programmes", lambda: tf.tl.find_programmes(adata, n_boot=30, n_jobs=n_jobs, verbose=False))
    step("profile_association", lambda: tf.tl.profile_association(adata, "label", verbose=False))
    rec["total_seconds"] = round(time.perf_counter() - t0, 1)
    rec["peak_gb"] = round(peak_gb(), 2)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)


STEPS = {"clone_profiles": "clone profiles", "clonal_coherence": "coherence test (200 permutations)",
         "clone_map": "clone map (settings chosen automatically)", "programmes": "programmes (30 bootstraps)",
         "profile_association": "label test (2,000 permutations)"}


def plot(results: Path):
    import matplotlib.pyplot as plt
    from threadfin.plotting import PALETTE

    tf.pl.set_style()
    recs = sorted((json.loads(f.read_text()) for f in results.glob("atlas_*.json")), key=lambda r: r["n_cells"])
    cells = [r["n_cells"] for r in recs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.9), constrained_layout=True, width_ratios=(1.4, 1))
    for colour, (key, label) in zip(PALETTE, STEPS.items()):
        a.plot(cells, [r["steps"][key]["seconds"] / 60 for r in recs], "-o", ms=4, lw=1.2, color=colour, label=label)
    a.plot(cells, [r["total_seconds"] / 60 for r in recs], "-o", ms=4, lw=1.6, color="#0b0b0b", label="all steps")
    b.plot(cells, [r["peak_gb"] for r in recs], "-o", ms=4, lw=1.6, color="#0b0b0b")
    for ax, ticks in ((a, [0.1, 0.3, 1, 3, 10, 30, 60]), (b, [1, 2, 4, 8])):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(cells, [f"{c // 1000:,}k" if c < 10 ** 6 else f"{c // 10 ** 6}M" for c in cells])
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("cells (8 CPU cores)")
    a.set_ylabel("minutes")
    a.set_title("time per step", loc="left")
    b.set_ylabel("peak memory, GB")
    b.set_title("memory", loc="left")
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=5.5)
    (results / "figures").mkdir(exist_ok=True)
    fig.savefig(results / "figures" / "atlas_scale.png")
    fig.savefig(results / "figures" / "atlas_scale.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(Path(sys.argv[2]))
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
