#!/usr/bin/env python
"""Simulated repertoires with a known answer for the benchmark.

Cells live in a 30-dimensional state space with four states: germinal-centre B cell (GC, at the origin),
plasma cell (PC, 4 within-state standard deviations away along axis 1), memory (MEM, 4 along axis 2) and
an intermediate state halfway between GC and PC. Each clone follows one of five programmes:

=================  =========================================
GC_only            all cells GC
mixture_GC_PC      each cell GC or PC with probability 1/2
intermediate       all cells in the intermediate state
PC_only            all cells PC
memory_GC          each cell MEM (0.7) or GC (0.3)
=================  =========================================

``mixture_GC_PC`` and ``intermediate`` have the same mean, so only a representation of the whole state
distribution of a clone can separate them. Clone sizes follow the shape of real repertoires
(``small``: 2 + Geometric(0.55), about half the clones have two cells; ``large``: 10 + Geometric(0.08)).
Clones are donor-private; every donor's cells are shifted by a random vector of length ``batch``
(in within-state standard deviations), and donors differ in programme composition (Dirichlet(4)
weights). Unlabelled background cells (8,000, from all states) are the context that real data carry.

The two donor-shift levels of one seed share the same clones and cells (the shift is added afterwards), and
the small and large regimes of one seed share the donor shifts and programme weights, so the independent
unit is the seed. Seeds 1-3 were run with every tool; seeds 4-10 were added on 2026-10-28 so that the
simulation readouts can be tested at the level of seeds (every tool except Clonotrace, whose runs on large
simulated clones take hours each).

Usage: python simulate_inputs.py [seed ...] [--force]     (default: seeds 1-10; an existing input is kept
       unless --force is given)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from bench_paths import INPUTS
from prepare_inputs import SEEDS, split_halves

SIM_SEEDS = tuple(range(1, 11))

DIM = 30
SEP = 4.0
CENTRES = {"GC": np.zeros(DIM), "PC": np.eye(DIM)[0] * SEP, "MEM": np.eye(DIM)[1] * SEP,
           "INTER": np.eye(DIM)[0] * SEP / 2}
PROGRAMMES = {"GC_only": {"GC": 1.0}, "mixture_GC_PC": {"GC": 0.5, "PC": 0.5}, "intermediate": {"INTER": 1.0},
              "PC_only": {"PC": 1.0}, "memory_GC": {"MEM": 0.7, "GC": 0.3}}
BACKGROUND = {"GC": 0.35, "PC": 0.25, "MEM": 0.2, "INTER": 0.2}


def draw(rng, mix: dict, n: int, shift: np.ndarray) -> np.ndarray:
    states = list(mix)
    pick = rng.choice(len(states), size=n, p=[mix[s] for s in states])
    centres = np.stack([CENTRES[states[i]] for i in pick])
    return centres + rng.standard_normal((n, DIM)) + shift


def simulate(name: str, *, seed: int, regime: str, batch: float, n_clones=600, n_donors=6, n_background=8000):
    rng = np.random.default_rng(seed)
    shifts = rng.standard_normal((n_donors, DIM))
    shifts = shifts / np.linalg.norm(shifts, axis=1, keepdims=True) * batch
    weights = rng.dirichlet(np.full(len(PROGRAMMES), 4.0), size=n_donors)
    rows, xs = [], []
    names = list(PROGRAMMES)
    for c in range(n_clones):
        d = int(rng.integers(n_donors))
        prog = names[int(rng.choice(len(names), p=weights[d]))]
        n = int(2 + rng.geometric(0.55) - 1) if regime == "small" else int(10 + rng.geometric(0.08) - 1)
        n = min(n, 80)
        xs.append(draw(rng, PROGRAMMES[prog], n, shifts[d]))
        rows += [{"clone": f"d{d}|clone{c}", "donor": f"d{d}", "programme": prog}] * n
    for i in range(n_background):
        d = int(rng.integers(n_donors))
        xs.append(draw(rng, BACKGROUND, 1, shifts[d]))
        rows.append({"clone": None, "donor": f"d{d}", "programme": None})
    x = np.vstack(xs).astype(np.float32)
    cells = pd.DataFrame(rows)
    cells["sample"] = cells["donor"]
    cells["clone"] = cells["clone"].astype("string")
    cells.index = [f"cell{i}" for i in range(len(cells))]
    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", x)
    cells.index.name = "cell"
    cells.to_csv(out / "cells.tsv.gz", sep="\t")
    for s in SEEDS:
        split_halves(cells, s).rename("half").to_frame().to_csv(out / f"split_seed{s}.tsv.gz", sep="\t")
    sizes = cells["clone"].value_counts()
    meta = {"dataset": name, "seed": seed, "regime": regime, "batch": batch, "n_cells": int(len(cells)),
            "n_clones": int(len(sizes)), "n_clones_eq2": int((sizes == 2).sum()),
            "n_clones_ge10": int((sizes >= 10).sum()), "median_clone_size": float(sizes.median()),
            "programmes": cells.dropna(subset=["clone"]).drop_duplicates("clone")["programme"].value_counts().to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


def main(argv):
    force = "--force" in argv
    seeds = [int(a) for a in argv if a.isdigit()] or SIM_SEEDS
    for regime in ("small", "large"):
        for batch in (0.0, 1.5):
            for seed in seeds:
                name = f"sim_{regime}_b{batch:g}_s{seed}"
                if (INPUTS / name / "meta.json").exists() and not force:
                    print(f"{name}: exists, kept", flush=True)
                    continue
                simulate(name, seed=seed, regime=regime, batch=batch)


if __name__ == "__main__":
    main(sys.argv[1:])
