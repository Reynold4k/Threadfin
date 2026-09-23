#!/usr/bin/env python
"""Collect per-dataset validation summaries into a cross-dataset table.

Usage:
    python collect_results.py

Reads results/<name>/summary.json + joint_diagnostics.json for every dataset
and writes results/cross_dataset_summary.md (+ .json).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def main():
    rows = []
    blob = {}
    for d in sorted(RESULTS.iterdir()):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        s = json.load(open(sj))
        jd = s.get("joint_diagnostics") or {}
        ho = s.get("heldout") or {}
        n1 = s.get("null_permutation") or {}
        joint_run = next((r for r in s.get("runs", []) if r.get("basis") == "joint"), {})
        row = {
            "dataset": s["name"],
            "cells": s["qc"]["n_cells"],
            "clonotypes": s["qc"]["n_clonotypes"],
            "expanded>=3": s["qc"]["expanded_clones_min3"],
            "null1_purity": n1.get("real_mean_purity"),
            "null1_null": n1.get("null_mean"),
            "null1_p": n1.get("p_value"),
            "heldout_cocluster": ho.get("cocluster_rate_mean"),
            "heldout_chance": ho.get("chance_rate_mean"),
            "basis_ARI": (s.get("robustness") or {}).get("clone_level_ari"),
            "joint_clusters": joint_run.get("n_clone_clusters"),
            "n_bcr_edges": joint_run.get("n_bcr_edges"),
            "testcor_gex": jd.get("testcor_gex"),
            "testcor_bcr": jd.get("testcor_bcr"),
            "modality_gex": (jd.get("modality_contribution") or {}).get("gex"),
        }
        rows.append(row)
        blob[s["name"]] = s

    cols = ["dataset", "cells", "clonotypes", "expanded>=3",
            "null1_purity", "null1_null", "null1_p",
            "heldout_cocluster", "heldout_chance", "basis_ARI",
            "joint_clusters", "n_bcr_edges", "testcor_gex", "testcor_bcr",
            "modality_gex"]
    lines = ["# Cross-dataset Threadfin validation summary", "",
             "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines.append("")
    out_md = RESULTS / "cross_dataset_summary.md"
    out_md.write_text("\n".join(lines))
    (RESULTS / "cross_dataset_summary.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines), flush=True)
    print(f"-> {out_md}", flush=True)


if __name__ == "__main__":
    main()
