#!/usr/bin/env python
"""Build the slim expression-embedding cache for one dataset (no counts, obs + X_threadfin).

Usage: python cache_embedding.py <dataset>
Output: $THREADFIN_EMBEDDING_CACHE/<dataset>_embedding.h5ad (default, as in the benchmark:
internal_validation/algorithm_revision/embedding_cache next to the repository clone)
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

import run_case_study as case

name = sys.argv[1]
cache = Path(os.environ.get("THREADFIN_EMBEDDING_CACHE", Path(__file__).resolve().parents[2].parent
                            / "internal_validation" / "algorithm_revision" / "embedding_cache"))
cache.mkdir(parents=True, exist_ok=True)

adata, bcr = case.LOADERS[name]()
print(f"loaded {adata.n_obs} cells x {adata.n_vars} genes; {len(bcr)} with a heavy chain", flush=True)
import threadfin as tf
tf.pp.prepare_embedding(adata, batch_key=case.CONFIGS[name]["batch_key"], key_added="X_threadfin",
                        verbose=True)
slim = ad.AnnData(X=sp.csr_matrix((adata.n_obs, 0)), obs=adata.obs.copy())
slim.obsm["X_threadfin"] = np.asarray(adata.obsm["X_threadfin"])
slim.write_h5ad(cache / f"{name}_embedding.h5ad", compression="gzip")
print(f"wrote {cache / (name + '_embedding.h5ad')}", flush=True)
