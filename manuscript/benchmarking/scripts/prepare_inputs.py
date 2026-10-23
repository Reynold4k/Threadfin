#!/usr/bin/env python
"""Write one self-contained benchmark input per dataset.

Every method in the benchmark receives exactly the same three things for a dataset:

* ``X.npy`` - the cell-level expression embedding the manuscript used (30 principal components of
  log-normalised expression, immunoglobulin genes excluded; ``obsm['X_threadfin']``);
* ``cells.tsv.gz`` - one row per cell: clone id (paired heavy + light chain Dandelion clones, defined
  within donors; empty when the cell has no clone), donor, sample and every annotation that can be
  scored afterwards (sorting gates, tissue, isotype, V-region mutation frequency ...);
* ``split_seed<k>.tsv.gz`` - a random division of *all* cells into two halves. Each clone's cells are
  divided as evenly as possible; cells without a clone are divided within each sample. The halves
  are disjoint, so a method run on one half never sees a cell of the other.

``cdr3.tsv.gz`` (heavy-chain V, J and junction amino-acid sequence per cell) is added for the
datasets on which Benisse runs, because Benisse embeds receptor sequences.

Usage: python prepare_inputs.py [dataset ...]
       python prepare_inputs.py --cdr3-only <dataset> ...   (add cdr3.tsv.gz to an existing input)
"""
from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

from bench_paths import CACHE, DATA, INPUTS, case_studies

DATASETS = ["gut_gc_reporter", "mouse_rbd", "mouse_np", "malaria_late", "malaria", "ln_vaccine",
            "spont_gc", "gc_np_pc", "covid_bcell"]
BENISSE = {"gut_gc_reporter", "mouse_rbd", "mouse_np", "malaria_late", "malaria", "spont_gc",
           "gc_np_pc", "ln_vaccine", "covid_bcell"}
DROP = ("clone_programme", "clone_programme_assigned", "umap_1", "umap_2")
SEEDS = (0, 1, 2)


def split_halves(cells: pd.DataFrame, seed: int) -> pd.Series:
    """Half A or B for every cell; clones split evenly, unlabelled cells split within samples."""
    rng = np.random.default_rng(seed)
    half = pd.Series("B", index=cells.index, dtype=object)
    for _, idx in cells.groupby("clone", observed=True).groups.items():
        idx = rng.permutation(np.asarray(idx))
        half.loc[idx[: len(idx) // 2]] = "A"
    free = cells["clone"].isna()
    for _, idx in cells[free].groupby("sample", observed=True).groups.items():
        idx = rng.permutation(np.asarray(idx))
        half.loc[idx[: len(idx) // 2]] = "A"
    return half


def heavy_cdr3(dataset: str, cells: pd.Index) -> pd.DataFrame:
    sys.path.insert(0, str(case_studies()))
    import datasets as loaders  # noqa: E402  (case-study loaders)

    if dataset == "ln_vaccine":
        heavy = pd.read_csv(DATA / "gse195673_ln_vaccine" / "bcr_heavy.tsv.gz", sep="\t", low_memory=False)
        heavy = heavy[heavy["cell_id"].notna()].drop_duplicates("cell_id").set_index("cell_id")
        bcr = heavy[["v_call", "j_call", "junction"]]
    else:
        _, bcr = loaders.LOADERS[dataset]()
        if "junction" not in bcr and "cdr3_nt" in bcr:
            bcr = bcr.rename(columns={"cdr3_nt": "junction"})
        bcr = bcr[["v_call", "j_call", "junction"]]
    bcr.index = bcr.index.astype(str)
    bcr = bcr.loc[bcr.index.intersection(cells)].copy()
    nt = bcr["junction"].astype(str).str.upper()
    ok = nt.str.fullmatch(r"[ACGT]+") & (nt.str.len() % 3 == 0)
    bcr["junction_aa"] = np.where(ok, nt.map(loaders._translate), None)
    bcr["v_call"] = bcr["v_call"].astype(str).str.split(",").str[0].str.split("*").str[0]
    bcr["j_call"] = bcr["j_call"].astype(str).str.split(",").str[0].str.split("*").str[0]
    good = bcr["junction_aa"].notna() & bcr["junction_aa"].astype(str).str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
    out = bcr.loc[good, ["v_call", "j_call", "junction_aa"]]
    out.index.name = "cell"
    return out


def prepare(dataset: str) -> dict:
    out = INPUTS / dataset
    out.mkdir(parents=True, exist_ok=True)
    cache = ad.read_h5ad(CACHE / f"{dataset}_embedding.h5ad")
    cache.obs_names = cache.obs_names.astype(str)
    table = pd.read_csv(case_studies() / "results" / dataset / "cells.csv.gz", index_col=0, low_memory=False)
    table.index = table.index.astype(str)
    missing = cache.obs_names.difference(table.index)
    if len(missing):
        raise ValueError(f"{dataset}: {len(missing)} embedded cells have no row in cells.csv.gz")
    table = table.loc[cache.obs_names]
    extra = [c for c in cache.obs.columns if c not in table.columns and c not in ("donor", "sample")]
    cells = pd.concat([table.drop(columns=[c for c in DROP if c in table]), cache.obs[extra]], axis=1)
    cells = cells.rename(columns={"clone_id": "clone"})
    cells["clone"] = cells["clone"].where(cells["clone"].notna() & cells["clone"].astype(str).ne("nan"))
    for col in ("donor", "sample"):
        cells[col] = cells[col].astype(str)
    # a clone id is private to one donor (Dandelion was run within donors)
    per_donor = cells.dropna(subset=["clone"]).groupby("clone")["donor"].nunique()
    if per_donor.gt(1).any():
        raise ValueError(f"{dataset}: {int(per_donor.gt(1).sum())} clone ids span donors")
    x = np.asarray(cache.obsm["X_threadfin"], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{dataset}: non-finite embedding values")
    np.save(out / "X.npy", x)
    cells.index.name = "cell"
    cells.to_csv(out / "cells.tsv.gz", sep="\t")
    for seed in SEEDS:
        split_halves(cells, seed).rename("half").to_frame().to_csv(out / f"split_seed{seed}.tsv.gz", sep="\t")
    sizes = cells["clone"].value_counts()
    meta = {
        "dataset": dataset, "n_cells": int(len(cells)), "n_dims": int(x.shape[1]),
        "n_cells_with_clone": int(cells["clone"].notna().sum()), "n_clones": int(len(sizes)),
        "n_donors": int(cells["donor"].nunique()), "n_samples": int(cells["sample"].nunique()),
        **{f"n_clones_ge{k}": int((sizes >= k).sum()) for k in (2, 3, 4, 5, 10, 20)},
        "n_clones_eq2": int((sizes == 2).sum()), "largest_clone": int(sizes.max()),
        "columns": [c for c in cells.columns],
        "label_levels": {c: cells[c].astype(str).value_counts().head(12).to_dict()
                         for c in cells.columns if c not in ("clone", "mutation_frequency")
                         and cells[c].dtype == object and cells[c].nunique() <= 40},
    }
    if dataset in BENISSE:
        cdr3 = heavy_cdr3(dataset, cells.index)
        cdr3.to_csv(out / "cdr3.tsv.gz", sep="\t")
        lab = cells.index[cells["clone"].notna()]
        meta["cdr3_cells"] = int(len(cdr3))
        meta["cdr3_coverage_of_clone_cells"] = float(cdr3.index.isin(lab).sum() / max(len(lab), 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("label_levels", "columns")}), flush=True)
    return meta


def add_cdr3(dataset: str) -> None:
    """Write cdr3.tsv.gz for an input that already exists, without rewriting its other files."""
    out = INPUTS / dataset
    cells = pd.read_csv(out / "cells.tsv.gz", sep="\t", index_col=0, usecols=["cell", "clone"])
    cells.index = cells.index.astype(str)
    cdr3 = heavy_cdr3(dataset, cells.index)
    cdr3.to_csv(out / "cdr3.tsv.gz", sep="\t")
    meta = json.loads((out / "meta.json").read_text())
    lab = cells.index[cells["clone"].notna()]
    meta["cdr3_cells"] = int(len(cdr3))
    meta["cdr3_coverage_of_clone_cells"] = float(cdr3.index.isin(lab).sum() / max(len(lab), 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{dataset}: cdr3 for {len(cdr3):,} cells ({meta['cdr3_coverage_of_clone_cells']:.1%} of clone cells)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cdr3-only"]:
        for name in sys.argv[2:]:
            add_cdr3(name)
    else:
        for name in sys.argv[1:] or DATASETS:
            prepare(name)
