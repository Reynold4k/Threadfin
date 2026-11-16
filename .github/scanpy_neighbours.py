"""Scanpy's own neighbour search, UMAP and Leiden on 12,000 cells, with the pynndescent and umap-learn installed.

Users call these themselves (Tutorial 5, Step 8). Scanpy finds neighbours with pynndescent from 8,192 cells on and
reads the sparse arrays of pynndescent 0.7 only from scanpy 1.12.5 on, so this fails when the installed pair does not
fit together; pyproject.toml is written so that pip never installs such a pair.
"""
from importlib.metadata import version

import anndata as ad
import numpy as np
import scanpy as sc

rng = np.random.default_rng(0)
centres = rng.normal(0, 4, size=(10, 30))
adata = ad.AnnData((centres[rng.integers(0, 10, 12_000)] + rng.normal(size=(12_000, 30))).astype(np.float32))
sc.pp.neighbors(adata, n_neighbors=15)          # 30 variables: scanpy uses X itself (no use_rep, renamed in 1.13)
sc.tl.umap(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
print(f"scanpy {version('scanpy')}, pynndescent {version('pynndescent')}, umap-learn {version('umap-learn')}: "
      f"neighbours ({type(adata.obsp['distances']).__name__}), UMAP and {adata.obs['leiden'].nunique()} Leiden "
      "clusters on 12,000 cells")
