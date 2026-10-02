#!/usr/bin/env python
"""Collect the case-study results into cross-dataset tables.

Reads results/<dataset>/ written by run_case_study.py and writes
results/cross_dataset_*.csv and results/CROSS_DATASET_SUMMARY.md (used by the
report and by the figure builder).

Usage: python summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import threadfin as tf

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ORDER = ["ln_vaccine", "mouse_np", "mouse_rbd", "flu", "ebv", "tonsil", "stephenson"]
LABELS = {
    "ln_vaccine": "Human lymph node + blood, SARS-CoV-2 mRNA vaccine (Kim 2022)",
    "mouse_np": "Mouse germinal centres, NP-OVA, division reporter (Merkenschlager 2025)",
    "mouse_rbd": "Mouse germinal centres, RBD protein/mRNA, bait and zone sorts (Merkenschlager 2025)",
    "flu": "Human blood, influenza vaccine, day 0 and day 7 (Wang 2023)",
    "ebv": "Human tonsil organoids, EBV infection (Mitul 2026)",
    "tonsil": "Human tonsil (King 2021)",
    "stephenson": "Human blood, COVID-19 (Stephenson 2021, 5,000-cell subset)",
}


def md_table(df: pd.DataFrame) -> str:
    """Plain Markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else (f"{v:.3g}" if abs(v) < 1e4 else f"{v:,.0f}")
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * df.shape[1]
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main():
    coh, progs, labels, assoc, mem, her, sets = ([] for _ in range(7))
    for ds in ORDER:
        path = RES / ds / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        q, c = s["qc"], s["coherence"]
        coh.append({"dataset": ds, "description": LABELS[ds], "cells": q["n_cells"],
                    "cells_with_BCR": q["n_cells_bcr"], "clones": q["n_clones"], "expanded_clones": q["clones_ge2"],
                    "clones_ge10_cells": q["clones_ge10"], "variance_explained_by_clone": c["icc"],
                    "shuffled_within_samples": c["null_mean"], "excess": c["excess"], "p_value": c["p_value"],
                    "cells_per_clone_for_reliable_profile": c["min_cells_reliability_0.5"]})

        p, sp = s.get("programmes", {}), s.get("programme_splits", {})
        base = {"dataset": ds, "n_programmes": p.get("n", 0), "core_clones": p.get("n_clones"),
                "split_test_p_at_root": sp.get("root_pvalue")}
        if p.get("n", 0) >= 2:
            markers, sigs = s.get("programme_markers", {}), s.get("programme_signatures", {})
            for name, stab in p["stability"].items():
                top_sig = max(sigs[name].items(), key=lambda kv: kv[1])[0] if sigs.get(name) else ""
                progs.append({**base, "programme": name, "stability": stab,
                              "top_markers": ", ".join(markers.get(name, [])[:8]), "highest_signature": top_sig})
        else:
            progs.append({**base, "programme": "continuum" if p.get("n") == 1 else "-", "stability": np.nan,
                          "top_markers": p.get("note", ""), "highest_signature": ""})

        for label, r in (s.get("label_effects") or {}).items():
            labels.append({"dataset": ds, "label": label, "kind": r["kind"], "n_clones": r["n_clones"],
                           "explained": r["r2"], "shuffled": r["null_mean"], "excess": r["excess_r2"],
                           "p_value": r["p_value"], "note": r.get("note", "")})
        a = RES / ds / "programme_associations.csv"
        if a.exists() and p.get("n", 0) >= 2:
            t = pd.read_csv(a)
            sig = t[t["fdr"] < 0.05].copy()
            sig.insert(0, "dataset", ds)
            assoc.append(sig)
        for key, m in (s.get("memory") or {}).items():
            mem.append({"dataset": ds, "across": key, "levels": " vs ".join(map(str, m.get("levels", []))),
                        "n_clones": m["n_clones"], "memory_index": m["memory_index"],
                        "ci_low": m["memory_index_ci"][0], "ci_high": m["memory_index_ci"][1],
                        "p_value": m["p_value"]})
        hpath = RES / ds / "gene_heritability.csv"
        if hpath.exists():
            ht = pd.read_csv(hpath, index_col=0).sort_values("excess_icc", ascending=False)
            her.append({"dataset": ds, "genes_tested": len(ht), "median_icc": float(ht["icc"].median()),
                        "most_clonal_genes": ", ".join(map(str, ht.index[:15]))})
            gs = pd.read_csv(RES / ds / "geneset_heritability.csv")
            ribo = [g for g in ht.index if str(g).upper().startswith(("RPS", "RPL")) and "-" not in str(g)]
            extra = tf.tl.geneset_heritability(ht, {"translation (ribosomal)": ribo}) if len(ribo) >= 3 else None
            gs = pd.concat([gs, extra], ignore_index=True) if extra is not None else gs
            gs.insert(0, "dataset", ds)
            sets.append(gs.drop(columns=["genes"], errors="ignore"))

    tables = {
        "coherence": pd.DataFrame(coh), "programmes": pd.DataFrame(progs), "label_effects": pd.DataFrame(labels),
        "programme_associations": pd.concat(assoc, ignore_index=True) if assoc else pd.DataFrame(),
        "memory": pd.DataFrame(mem), "heritability": pd.DataFrame(her),
        "geneset_heritability": pd.concat(sets, ignore_index=True) if sets else pd.DataFrame(),
    }
    md = ["# Cross-dataset summary (generated by summarize.py)", ""]
    for name, tab in tables.items():
        tab.to_csv(RES / f"cross_dataset_{name}.csv", index=False)
        md += [f"## {name}", "", md_table(tab) if not tab.empty else "_none_", ""]
    (RES / "CROSS_DATASET_SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md)[:8000])


if __name__ == "__main__":
    main()
