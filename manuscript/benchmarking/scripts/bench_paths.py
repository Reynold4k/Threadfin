"""Locations shared by the benchmark scripts.

The benchmark reads the per-dataset results of the case studies (clone ids, donors, samples and
annotations, one row per cell) and the cached expression embeddings, and writes everything large to a
working directory outside the repository. Only the scripts, the summary tables and the figures are
kept in ``manuscript/benchmarking/``.

Set ``THREADFIN_BENCH_WORK`` and ``THREADFIN_EMBEDDING_CACHE`` to run it elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent                                   # manuscript/benchmarking
REPO = BENCH.parents[1]                               # repository root

_PROJECT = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表")
WORK = Path(os.environ.get("THREADFIN_BENCH_WORK", _PROJECT / "benchmarking_2026-10-20" / "work"))
CACHE = Path(os.environ.get("THREADFIN_EMBEDDING_CACHE",
                            _PROJECT / "internal_validation" / "algorithm_revision" / "embedding_cache"))
DATA = Path(os.environ.get("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data"))

INPUTS = WORK / "inputs"
RUNS = WORK / "runs"
RESULTS = BENCH / "results"
FIGURES = BENCH / "figures"


def case_studies() -> Path:
    """The case-study folder (moved from the repository root into ``manuscript/`` on 2026-10-20)."""
    for p in (REPO / "manuscript" / "case_studies", REPO / "case_studies"):
        if (p / "results").is_dir():
            return p
    raise FileNotFoundError("case_studies/results not found under the repository")


def run_dir(dataset: str, variant: str, method: str) -> Path:
    return RUNS / dataset / variant / method
