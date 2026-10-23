#!/usr/bin/env python
"""Build the slim expression-embedding cache for one dataset (no counts, obs + X_threadfin).

Usage: python cache_embedding.py <dataset>
Output: internal_validation/algorithm_revision/embedding_cache/<dataset>_embedding.h5ad
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

import run_case_study as case

name = sys.argv[1]
cache = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/"
             "algorithm_revision/embedding_cache")
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
