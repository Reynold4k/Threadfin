#!/usr/bin/env python
"""Benchmark inputs from LARRY (Weinreb et al. 2020, Science; GSE140802), the dataset of Clonotrace's
own tutorial.

``larry_d2``   all day-2 cells (28,249) with the embedding computed from day-2 cells only; a clone is a
               lineage barcode. Its targets are measured four days later on *other* cells of the same
               barcode: the fractions of day-6 cells annotated as neutrophils and as monocytes, and
               the neutrophil share of the two (``clone_targets.tsv``). Nothing from day 6 enters the
               day-2 representation.
``larry_d46``  day-4 and day-6 barcoded cells; a clone is a barcode at one time point (Clonotrace's
               definition in its vignette). The benchmark asks whether the day-4 representation of a
               barcode identifies the same barcode among all day-6 clones.

Every cell gets the single donor ``pool`` (one culture) and its library as sample.

Usage: python prepare_larry.py
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from bench_paths import INPUTS
from prepare_inputs import SEEDS, split_halves

LARRY = "/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/larry"


def write(name, x, cells, extra=None):
    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", np.asarray(x, dtype=np.float32))
    cells.index.name = "cell"
    cells.to_csv(out / "cells.tsv.gz", sep="\t")
    for seed in SEEDS:
        split_halves(cells, seed).rename("half").to_frame().to_csv(out / f"split_seed{seed}.tsv.gz", sep="\t")
    sizes = cells["clone"].value_counts()
    meta = {"dataset": name, "n_cells": int(len(cells)), "n_cells_with_clone": int(cells["clone"].notna().sum()),
            "n_clones": int(len(sizes)), **{f"n_clones_ge{k}": int((sizes >= k).sum()) for k in (2, 3, 4, 10, 20)},
            "n_clones_eq2": int((sizes == 2).sum()), "largest_clone": int(sizes.max()), **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


def main():
    early = ad.read_h5ad(f"{LARRY}/early_embedding.h5ad")
    every = ad.read_h5ad(f"{LARRY}/all_embedding.h5ad")
    # day 2: all cells; clone = barcode
    o = early.obs
    cells = pd.DataFrame({"clone": o["barcode"].astype("string"), "donor": "pool",
                          "sample": o["library"].astype(str), "state": o["state"].astype(str)},
                         index=early.obs_names.astype(str))
    late = every.obs[every.obs["day"].astype(int) == 6]
    counts = pd.crosstab(late["barcode"].astype(str), late["state"].astype(str))
    n6 = counts.sum(axis=1)
    targets = pd.DataFrame({"n_day6_cells": n6,
                            "day6_neutrophil_fraction": counts.get("Neutrophil", 0) / n6,
                            "day6_monocyte_fraction": counts.get("Monocyte", 0) / n6})
    nm = counts.get("Neutrophil", 0) + counts.get("Monocyte", 0)
    targets["day6_neutrophil_share_of_neu_mono"] = np.where(nm >= 2, counts.get("Neutrophil", 0) / nm.clip(lower=1), np.nan)
    targets.index.name = "clone"
    write("larry_d2", early.obsm["X_threadfin"], cells,
          {"targets": "day-6 fate fractions of the same barcode (clone_targets.tsv); clones need >= 4 day-6 cells"})
    targets[targets["n_day6_cells"] >= 4].to_csv(INPUTS / "larry_d2" / "clone_targets.tsv", sep="\t")
    # days 4 and 6: clone = barcode at one time point
    keep = every.obs["day"].astype(int).isin([4, 6]).to_numpy()
    o = every.obs[keep]
    cells = pd.DataFrame({"clone": (o["barcode"].astype(str) + "|d" + o["day"].astype(int).astype(str)).astype("string"),
                          "donor": "pool", "sample": o["library"].astype(str), "state": o["state"].astype(str),
                          "day": o["day"].astype(int)}, index=every.obs_names[keep].astype(str))
    write("larry_d46", every.obsm["X_threadfin"][keep], cells,
          {"task": "re-identify a barcode's day-6 clone from its day-4 clone"})


if __name__ == "__main__":
    main()
