#!/usr/bin/env python
"""One table of every dataset analysed for the manuscript, built from the saved case-study results.

Reads ``case_studies/results/<dataset>/summary.json`` and writes ``dataset_overview.csv`` and
``dataset_overview.md`` next to this script. Nothing is re-computed.

Usage: python dataset_overview.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "case_studies" / "results"

DATASETS = [
    # key, study, accession, organism, system, where it appears in the manuscript
    ("np_cgg_wt_d14", "Manakkat Vijay et al. 2026 Nat. Immunol.", "GSE305877", "mouse", "NP-CGG immunisation, one WT mouse, day 14", "Fig. 2B; Supp. Fig. 1 (sheet 1)"),
    ("gut_gc_reporter", "Siniscalco et al. 2025 Immunity", "GSE284622", "mouse", "oral OVA + cholera toxin, GC-fate reporter, MLN/PP/LP", "Fig. 3g-j; Supp. Figs. 1, 2, 4, 5, 6"),
    ("mouse_rbd", "Merkenschlager et al. 2025 Nature", "GSE287123", "mouse", "RBD protein / mRNA, H2B-mCherry division reporter", "Fig. 4; Supp. Fig. 1 (sheet 2)"),
    ("mouse_np", "Merkenschlager et al. 2025 Nature", "GSE287123", "mouse", "NP-OVA, H2B-mCherry division reporter", "Supp. Fig. 1 (sheet 2)"),
    ("gc_np_pc", "ElTanbouly et al. 2024 J. Exp. Med.", "GSE246382", "mouse", "NP-OVA day 14, sorted GC zones and plasma cells (Smart-seq2)", "Supp. Fig. 1 (sheet 1)"),
    ("malaria_late", "Skinner et al. 2026 Nat. Immunol.", "GSE286215 (exp. 2)", "mouse", "Plasmodium chabaudi infection, days 10-42", "Fig. 2B-D; Supp. Figs. 2, 6"),
    ("malaria", "Skinner et al. 2026 Nat. Immunol.", "GSE286215 (exp. 1)", "mouse", "Plasmodium chabaudi infection, days 0-14", "Supp. Table 2"),
    ("ln_vaccine", "Kim et al. 2022 Nature", "GSE195673", "human", "SARS-CoV-2 mRNA vaccination, lymph node FNA + blood", "Fig. 2B; Supp. Figs. 2, 6"),
    ("spont_gc", "Castrillon et al. 2023 eLife", "GSE203132", "mouse", "spontaneous GCs, AID-EYFP+ splenic B cells", "Fig. 3a-f; Supp. Fig. 3"),
    ("covid_bcell", "Dugan et al. 2021 Immunity", "GSE171703", "human", "antigen-bait-sorted B cells after SARS-CoV-2 infection", "additional (not in manuscript)"),
    ("flu", "Wang et al. 2023", "GSE175522 / GSE175523", "human", "seasonal influenza vaccination, blood d0 and d7 (heavy chain only)", "additional (not in manuscript)"),
    ("flu_lung", "GEO series", "GSE317692", "mouse", "influenza A infection, lung and mediastinal LN, day 20, HA tetramers", "additional (not in manuscript)"),
    ("tonsil", "King et al. 2021 Sci. Immunol.", "E-MTAB-9005 / 9003", "human", "paediatric tonsil", "additional (not in manuscript)"),
    ("ebv", "Mitul et al. 2026 PNAS", "GSE317492", "human", "EBV infection of tonsil organoids (donors pooled)", "additional (not in manuscript)"),
    ("bone_marrow_pc", "GEO series", "GSE253857", "human", "bone-marrow plasma cells and memory B cells, with blood", "additional (not in manuscript)"),
]


def row(key, study, accession, organism, system, where):
    d = json.loads((RES / key / "summary.json").read_text())
    coh = d.get("coherence", {})
    q = d.get("qc", {})
    if key == "np_cgg_wt_d14":
        q = {"n_cells": d.get("n_cells"), "n_cells_bcr": d.get("n_cells_paired_bcr"), "n_clones": d.get("n_clones_all"),
             "clones_ge2": d.get("n_profiles_ge2")}
    prog = d.get("programmes", {})
    n_prog = prog.get("n") if isinstance(prog, dict) else None
    return {"dataset": key, "study": study, "accession": accession, "organism": organism, "system": system,
            "cells": q.get("n_cells"), "cells_with_BCR": q.get("n_cells_bcr"), "clones": q.get("n_clones"),
            "clones_ge2": q.get("clones_ge2", coh.get("n_clones")), "clones_ge10": q.get("clones_ge10"),
            "clonal_ICC": coh.get("icc"), "ICC_shuffled": coh.get("null_mean"), "ICC_p": coh.get("p_value"),
            "programmes": n_prog, "in_manuscript": where}


def main():
    t = pd.DataFrame([row(*x) for x in DATASETS])
    t.to_csv(HERE / "dataset_overview.csv", index=False)
    show = t.copy()
    for c in ("clonal_ICC", "ICC_shuffled"):
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    show["ICC_p"] = show["ICC_p"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
    for c in ("cells", "cells_with_BCR", "clones", "clones_ge2", "clones_ge10", "programmes"):
        show[c] = show[c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    cols = ["dataset", "accession", "organism", "system", "cells", "cells_with_BCR", "clones_ge2", "clones_ge10",
            "clonal_ICC", "ICC_shuffled", "ICC_p", "programmes", "in_manuscript"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in show.iterrows()]
    (HERE / "dataset_overview.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
