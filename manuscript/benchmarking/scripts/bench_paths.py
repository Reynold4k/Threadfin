"""Locations shared by the benchmark scripts.

The benchmark reads the per-dataset results of the case studies (clone ids, donors, samples and
annotations, one row per cell) and the cached expression embeddings, and writes everything large to a
working directory outside the repository. Only the scripts, the summary tables and the figures are
kept in ``manuscript/benchmarking/``.

Every location is an environment variable; the defaults are folders next to the repository clone (the
authors' layout), so no path is tied to one computer:

``THREADFIN_BENCH_WORK``       inputs/ and runs/ of the benchmark (default ``../benchmarking_2026-10-20/work``)
``THREADFIN_EMBEDDING_CACHE``  the case studies' embedding cache, ``<dataset>_embedding.h5ad``
                               (default ``../internal_validation/algorithm_revision/embedding_cache``)
``THREADFIN_DATA``             the downloaded public data (``case_studies/download.sh``; default
                               ``manuscript/case_studies/data``, as in ``case_studies/datasets.py``)
``THREADFIN_LARRY_CACHE``      the two LARRY embeddings (``early_embedding.h5ad``, ``all_embedding.h5ad``)
                               (default ``../internal_validation/paper_673503/larry``)
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent                                   # manuscript/benchmarking
REPO = BENCH.parents[1]                               # repository root

WORK = Path(os.environ.get("THREADFIN_BENCH_WORK", REPO.parent / "benchmarking_2026-10-20" / "work"))
CACHE = Path(os.environ.get("THREADFIN_EMBEDDING_CACHE",
                            REPO.parent / "internal_validation" / "algorithm_revision" / "embedding_cache"))
DATA = Path(os.environ.get("THREADFIN_DATA", REPO / "manuscript" / "case_studies" / "data"))
LARRY_CACHE = Path(os.environ.get("THREADFIN_LARRY_CACHE",
                                  REPO.parent / "internal_validation" / "paper_673503" / "larry"))

INPUTS = WORK / "inputs"
RUNS = WORK / "runs"
# THREADFIN_BENCH_RESULTS redirects the tables (and, next to them, the figures) for a trial run; the README is
# rewritten only when the tables are written to the repository's results/
RESULTS = Path(os.environ.get("THREADFIN_BENCH_RESULTS", BENCH / "results"))
FIGURES = RESULTS.parent / "figures"


def case_studies() -> Path:
    """The case-study folder (moved from the repository root into ``manuscript/`` on 2026-10-20)."""
    for p in (REPO / "manuscript" / "case_studies", REPO / "case_studies"):
        if (p / "results").is_dir():
            return p
    raise FileNotFoundError("case_studies/results not found under the repository")


def run_dir(dataset: str, variant: str, method: str) -> Path:
    return RUNS / dataset / variant / method
