"""Reading benchmark inputs and writing method outputs in one format.

A *variant* says which cells a method sees: ``full`` (all cells) or ``halfA_s<seed>`` /
``halfB_s<seed>`` (one of two disjoint halves of every clone and of the unlabelled cells, see
``prepare_inputs.py``). Every method writes the same files into its run directory:

``clones.tsv``     the clones it represented, in output order, with their number of cells;
``features.npy``   clones x dimensions (methods that output coordinates), or
``distances.npy``  clones x clones (methods that output a distance matrix);
``run.json``       method, software version, parameters, status and wall-clock time of each stage.
"""
from __future__ import annotations

import json
import platform
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from bench_paths import INPUTS, run_dir


def load(dataset: str, variant: str = "full"):
    """Embedding and cell table for one variant (cells of the other half are removed).

    Variants ``half<A|B>_s<seed>`` keep one half of a random split; any other variant name (``full``,
    ``full_w8`` for a run with 8 workers, ...) is the full data.
    """
    d = INPUTS / dataset
    x = np.load(d / "X.npy")
    cells = pd.read_csv(d / "cells.tsv.gz", sep="\t", index_col=0, low_memory=False)
    cells.index = cells.index.astype(str)
    cells["clone"] = cells["clone"].astype("string")
    if variant.startswith("half"):
        half, seed = variant[4], int(variant.split("_s")[1])
        split = pd.read_csv(d / f"split_seed{seed}.tsv.gz", sep="\t", index_col=0)["half"]
        split.index = split.index.astype(str)
        keep = (split.reindex(cells.index) == half).to_numpy()
        x, cells = x[keep], cells.loc[keep]
    return x, cells


def eligible(cells: pd.DataFrame, min_size: int) -> pd.Series:
    """Clone sizes of the clones with at least ``min_size`` cells in this variant."""
    sizes = cells["clone"].dropna().value_counts()
    return sizes[sizes >= min_size]


def write(out: Path, clones, n_cells, *, features=None, distances=None, info=None):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"clone": list(map(str, clones)), "n_cells": np.asarray(n_cells, int)}).to_csv(
        out / "clones.tsv", sep="\t", index=False)
    if features is not None:
        np.save(out / "features.npy", np.asarray(features, dtype=np.float32))
    if distances is not None:
        np.save(out / "distances.npy", np.asarray(distances, dtype=np.float32))
    record = {"status": "completed", "python": platform.python_version(),
              "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    record.update(info or {})
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def fail(out: Path, reason: str, info=None):
    out.mkdir(parents=True, exist_ok=True)
    record = {"status": "failed", "reason": reason}
    record.update(info or {})
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


class Timer:
    """Wall-clock time of named stages."""

    def __init__(self):
        self.stages = {}

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        yield
        self.stages[name] = round(time.perf_counter() - t0, 3)

    @property
    def total(self):
        return round(sum(self.stages.values()), 3)


def out_dir(dataset, variant, method) -> Path:
    return run_dir(dataset, variant, method)
