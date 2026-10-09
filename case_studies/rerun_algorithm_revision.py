"""Recompute affected analyses into a separate result root and cache their fixed cell embedding."""
from pathlib import Path
import os
import sys
import anndata as ad
import numpy as np
import scipy.sparse as sp
import run_case_study as case

cache = Path(os.environ["THREADFIN_REVISION_CACHE"])
cache.mkdir(parents=True, exist_ok=True)
original_prepare = case.prepare
def prepare(*args, **kwargs):
    a, cfg, state, summary = original_prepare(*args, **kwargs)
    # No expression counts are copied into the repository.
    slim = ad.AnnData(X=sp.csr_matrix((a.n_obs, 0)), obs=a.obs.copy())
    slim.obsm["X_threadfin"] = np.asarray(a.obsm["X_threadfin"])
    slim.write_h5ad(cache / f"{args[0]}_embedding.h5ad", compression="gzip")
    return a, cfg, state, summary
case.prepare = prepare
names = ["ln_vaccine", "bone_marrow_pc", "ebv", "flu", "mouse_rbd", "mouse_np",
         "malaria", "malaria_late", "tonsil", "stephenson", "flu_lung", "gc_np_pc"]
case.main(names[int(sys.argv[1])])
