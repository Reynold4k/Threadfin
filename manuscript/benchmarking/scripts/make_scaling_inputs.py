#!/usr/bin/env python
"""Subsets of the largest dataset (GSE195673) for the scaling readout: how time and memory grow with data size.

Each subset keeps a random fraction of the clones with >= 2 cells (all of their cells) and the same fraction
of the remaining cells (single-cell clones and cells without a clone), so the mix of clone sizes, donors and
samples stays that of the full dataset. The embedding rows, the receptor sequences and the expression input of
Benisse are subset to the same cells. Inputs are written as ``inputs/ln_vaccine_f<fraction>/``.

Usage: python make_scaling_inputs.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bench_paths import INPUTS
from prepare_inputs import SEEDS, split_halves

FRACTIONS = (0.05, 0.1, 0.25, 0.5)
SOURCE = "ln_vaccine"


def main():
    src = INPUTS / SOURCE
    x = np.load(src / "X.npy")
    cells = pd.read_csv(src / "cells.tsv.gz", sep="\t", index_col=0, low_memory=False)
    cells.index = cells.index.astype(str)
    sizes = cells["clone"].value_counts()
    multi = sizes.index[sizes >= 2]
    in_multi = cells["clone"].isin(multi).to_numpy()
    cdr3 = pd.read_csv(src / "cdr3.tsv.gz", sep="\t", index_col=0)
    cdr3.index = cdr3.index.astype(str)
    expr = sp.load_npz(src / "expr_hvg.npz")
    expr_cells = pd.read_csv(src / "expr_hvg_cells.txt", header=None)[0].astype(str)
    expr_pos = pd.Series(np.arange(len(expr_cells)), index=expr_cells)
    rng = np.random.default_rng(20261023)
    for frac in FRACTIONS:
        name = f"{SOURCE}_f{frac:g}"
        out = INPUTS / name
        out.mkdir(parents=True, exist_ok=True)
        keep_clones = set(rng.choice(multi, size=int(round(frac * len(multi))), replace=False))
        other = np.flatnonzero(~in_multi)
        keep_other = rng.choice(other, size=int(round(frac * other.size)), replace=False)
        mask = cells["clone"].isin(keep_clones).to_numpy()
        mask[keep_other] = True
        sub = cells.loc[mask]
        np.save(out / "X.npy", x[mask])
        sub.index.name = "cell"
        sub.to_csv(out / "cells.tsv.gz", sep="\t")
        for seed in SEEDS:
            split_halves(sub, seed).rename("half").to_frame().to_csv(out / f"split_seed{seed}.tsv.gz", sep="\t")
        cdr3.loc[cdr3.index.intersection(sub.index)].to_csv(out / "cdr3.tsv.gz", sep="\t")
        e_cells = expr_cells[expr_cells.isin(sub.index)]
        sp.save_npz(out / "expr_hvg.npz", expr[expr_pos.loc[e_cells].to_numpy()])
        e_cells.to_csv(out / "expr_hvg_cells.txt", index=False, header=False)
        shutil.copy2(src / "expr_hvg_genes.txt", out / "expr_hvg_genes.txt")
        s = sub["clone"].value_counts()
        meta = {"dataset": name, "source": SOURCE, "fraction": frac, "n_cells": int(len(sub)),
                "n_cells_with_clone": int(sub["clone"].notna().sum()), "n_clones": int(len(s)),
                "n_donors": int(sub["donor"].nunique()), "n_samples": int(sub["sample"].nunique()),
                **{f"n_clones_ge{k}": int((s >= k).sum()) for k in (2, 3, 4, 5, 10, 20)},
                "n_clones_eq2": int((s == 2).sum()), "largest_clone": int(s.max())}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
