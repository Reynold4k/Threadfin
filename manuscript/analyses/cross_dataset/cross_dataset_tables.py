#!/usr/bin/env python
"""Collect three per-dataset results of the case studies into cross-dataset tables.

* ``clonal_memory.csv``: the clonal-memory index for every dataset and every axis it was measured over
  (time point, tissue, sort gate): is a clone seen in two places or times still recognisably itself?
* ``label_effects.csv``: every clone-level association test (annotation vs clone profiles) of every dataset.
* ``geneset_heritability.csv``: clonal inheritance of marker gene sets, per dataset.

Nothing is re-computed; the tables are read from ``case_studies/results/<dataset>/``.

Usage: python cross_dataset_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "case_studies" / "results"
DATASETS = ["np_cgg_wt_d14", "gut_gc_reporter", "mouse_rbd", "mouse_np", "gc_np_pc", "malaria_late", "malaria",
            "ln_vaccine", "spont_gc", "covid_bcell", "flu", "flu_lung", "tonsil", "ebv", "bone_marrow_pc"]


def main():
    memory, effects, genesets = [], [], []
    for ds in DATASETS:
        d = RES / ds
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        for axis, m in (s.get("memory") or {}).items():
            if isinstance(m, dict) and "memory_index" in m:
                lo, hi = (m.get("memory_index_ci") or [None, None])[:2]
                memory.append({"dataset": ds, "axis": axis, "n_clones": m.get("n_clones"), "n_pairs": m.get("n_pairs"),
                               "memory_index": m["memory_index"], "ci_low": lo, "ci_high": hi, "p_value": m.get("p_value"),
                               "same_programme": m.get("persistence"), "same_programme_random": m.get("persistence_null"),
                               "levels": ", ".join(map(str, m.get("levels", [])))})
        if (d / "label_effects.csv").exists():
            t = pd.read_csv(d / "label_effects.csv")
            t.insert(0, "dataset", ds)
            effects.append(t)
        if (d / "geneset_heritability.csv").exists():
            t = pd.read_csv(d / "geneset_heritability.csv")
            t.insert(0, "dataset", ds)
            genesets.append(t)
    pd.DataFrame(memory).to_csv(HERE / "clonal_memory.csv", index=False)
    pd.concat(effects, ignore_index=True).to_csv(HERE / "label_effects.csv", index=False)
    pd.concat(genesets, ignore_index=True).to_csv(HERE / "geneset_heritability.csv", index=False)
    print(pd.DataFrame(memory).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
