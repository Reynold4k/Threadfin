"""Isolated-process representation timing on identical synthetic inputs.

Includes reference fitting and transform for density; includes default
smoothing/RFF/PCA for kernel. These outputs solve different representation
tasks, so speed is not evidence of biological accuracy or Clonotrace parity.
"""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results" / "algorithm_revision"


def worker(method, n_cells):
    import anndata as ad
    import threadfin as tf
    rng = np.random.default_rng(17)
    n_clones = n_cells//10
    clone_codes = np.repeat(np.arange(n_clones), 10)
    x = rng.normal(size=(n_cells, 30)) + rng.normal(size=(n_clones, 30))[clone_codes]
    ids = np.array([f"c{i}" for i in clone_codes])
    a = ad.AnnData(np.zeros((n_cells, 1)))
    a.obs["clone_id"] = ids
    a.obsm["X_pca"] = x
    start = time.perf_counter()
    if method == "density":
        model = tf.StateDensityModel.fit(x, ids, n_states=16)
        result = model.transform_batches((x[i:i+2048], ids[i:i+2048], None)
                                         for i in range(0, len(x), 2048))
        dimension = result.proportions.shape[1]
    else:
        tf.tl.clone_profiles(a, representation=method, verbose=False)
        dimension = a.uns["threadfin"]["profiles"]["features"].shape[1]
    return {"method": method, "n_cells": n_cells, "n_clones": n_clones,
            "seconds": time.perf_counter()-start,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "dimensions": dimension, "seed": 17}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(worker(sys.argv[1], int(sys.argv[2]))))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in (3000, 15000, 50000):
            for method in ("mean", "kernel", "density"):
                output = subprocess.check_output([sys.executable, __file__, method, str(n)], text=True)
                row = json.loads(output.strip().splitlines()[-1])
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv(OUT / "representation_runtime.csv", index=False)
