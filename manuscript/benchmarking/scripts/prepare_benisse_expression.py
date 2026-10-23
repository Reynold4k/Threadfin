#!/usr/bin/env python
"""Log-normalised expression of the highly variable genes, for Benisse.

Benisse's README allows a dimension reduction in place of the expression matrix, but its
initialisation keeps the 10 % most variable rows and takes 10 principal components of them, so it needs
at least 100 rows; the 30-dimensional shared embedding fails there ("subscript out of bounds"). Benisse
therefore receives what its README recommends: log(1 + x) of library-size-normalised counts, here for
the same 3,000 highly variable genes (immunoglobulin and T-cell-receptor genes excluded) from which the
shared embedding was computed. Only the cells of clones with at least two cells are stored, because
Benisse is run on those cells.

Usage: python prepare_benisse_expression.py [dataset ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bench_paths import INPUTS, case_studies

DATASETS = ["gc_np_pc", "gut_gc_reporter", "mouse_np", "mouse_rbd", "spont_gc", "malaria", "malaria_late",
            "ln_vaccine", "covid_bcell"]
BATCH = {"ln_vaccine": "donor"}


def main(dataset):
    import scanpy as sc
    import threadfin as tf

    sys.path.insert(0, str(case_studies()))
    import datasets as loaders  # noqa: E402

    cells = pd.read_csv(INPUTS / dataset / "cells.tsv.gz", sep="\t", index_col=0, usecols=["cell", "clone", "donor"])
    cells.index = cells.index.astype(str)
    adata, _ = loaders.LOADERS[dataset]()
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[adata.obs_names.isin(cells.index)].copy()
    missing = cells.index.difference(adata.obs_names)
    if len(missing):
        raise ValueError(f"{dataset}: {len(missing)} benchmark cells are missing from the loader output")
    adata = adata[cells.index].copy()
    tmp = sc.AnnData(X=adata.X.copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    receptor = tf.pp.ig_gene_mask(tmp.var_names, constant=True, tr_genes=True)
    pool = tmp[:, ~receptor].copy()
    key = BATCH.get(dataset)
    if key:
        pool.obs[key] = cells.loc[pool.obs_names, key].astype(str).values
    sc.pp.highly_variable_genes(pool, n_top_genes=3000, flavor="seurat", batch_key=key)
    genes = pool.var_names[pool.var["highly_variable"].to_numpy()]
    sizes = cells["clone"].value_counts()
    keep = cells["clone"].isin(sizes.index[sizes >= 2]).to_numpy()
    x = sp.csr_matrix(pool[keep, genes].X, dtype=np.float32)
    out = INPUTS / dataset
    sp.save_npz(out / "expr_hvg.npz", x)
    pd.Series(pool.obs_names[keep]).to_csv(out / "expr_hvg_cells.txt", index=False, header=False)
    pd.Series(genes).to_csv(out / "expr_hvg_genes.txt", index=False, header=False)
    print(f"{dataset}: {x.shape[0]} cells x {x.shape[1]} genes (of {int((~receptor).sum())} non-receptor genes)", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        main(name)
