#!/usr/bin/env python
"""Time and memory of every step of the complete Threadfin analysis (``threadfin.run``) on a benchmark input.

The benchmark's representation readouts time only the clone profiles. A user runs the whole analysis, so this
script runs the same steps as ``threadfin.run`` one by one on the shared embedding and clone ids (clone
profiles, the coherence test with 200 permutations, the automatically tuned clone map, programmes with 30
bootstrap replicates, and one clone-level label test when the dataset has a sorting gate), and records the
wall time and the process's peak memory after each step.

Usage: python profile_threadfin_run.py <dataset> <n_jobs> <out.json>
"""
from __future__ import annotations

import inspect
import json
import resource
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bench_io

LABEL = {"mouse_rbd": "division_gate", "mouse_np": "division_gate", "gut_gc_reporter": "gc_sort",
         "gc_np_pc": "fate", "ln_vaccine": "tissue", "covid_bcell": "isotype", "spont_gc": "isotype",
         "malaria_late": "cell_state", "malaria": "cell_state"}


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main(dataset, n_jobs, out):
    import threadfin as tf

    x, cells = bench_io.load(dataset, "full")
    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32),
                   obs=cells.drop(columns=[c for c in cells.columns if c == "clone"]).copy())
    a.obs["clone_id"] = cells["clone"].astype(object).where(cells["clone"].notna(), None).to_numpy()
    a.obsm["X_emb"] = np.asarray(x, dtype=np.float32)
    kw = {"n_jobs": n_jobs} if n_jobs is not None else {}
    steps, t_all = {}, time.perf_counter()

    def step(name, fn, **extra):
        t0 = time.perf_counter()
        fn(**extra)
        steps[name] = {"seconds": round(time.perf_counter() - t0, 2), "peak_rss_gb": round(peak_gb(), 2)}
        print(f"{dataset}: {name} {steps[name]}", flush=True)

    step("clone_profiles", lambda: tf.tl.clone_profiles(a, clone_key="clone_id", basis="X_emb", context_key="sample",
                                                        donor_key="donor", representation="kernel", verbose=False))
    step("clonal_coherence", lambda: tf.tl.clonal_coherence(a, strata_key="sample", n_perm=200, verbose=False))
    step("clone_map", lambda: tf.tl.embed_clones(a, verbose=False, **kw))
    prog_kw = kw if "n_jobs" in inspect.signature(tf.tl.find_programmes).parameters else {}
    step("programmes", lambda: tf.tl.find_programmes(a, n_boot=30, verbose=False, **prog_kw))
    label = LABEL.get(dataset.split("_f0.")[0])          # the size subsets keep their dataset's label
    if label and label in a.obs:
        step("profile_association", lambda: tf.tl.profile_association(a, label, verbose=False))
    sizes = a.obs["clone_id"].value_counts()
    rec = {"dataset": dataset, "n_jobs": n_jobs, "threadfin": tf.__version__, "n_cells": int(a.n_obs),
           "n_clones_ge2": int((sizes >= 2).sum()), "steps": steps,
           "total_seconds": round(time.perf_counter() - t_all, 2), "peak_rss_gb": round(peak_gb(), 2)}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    nj = None if sys.argv[2] in ("none", "0") else int(sys.argv[2])
    main(sys.argv[1], nj, sys.argv[3])
