#!/usr/bin/env python
"""Collect the per-dataset records of run_dataset.py into results/tutorial_path.csv and the README table.

Usage: python summarise_results.py <run_dataset.py output dir>
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OVERVIEW = HERE.parents[2] / "dataset_overview.md"
NAMES = {"ln_vaccine": "GSE195673 lymph node, mRNA vaccine", "flu": "GSE175522 blood, influenza vaccine",
         "tonsil": "E-MTAB-9005 paediatric tonsil", "ebv": "GSE317492 EBV tonsil organoids",
         "mouse_np": "GSE287123 NP reporter", "mouse_rbd": "GSE287123 RBD reporter (Tutorial 5)",
         "gc_np_pc": "GSE246382 GC + plasma cells (Smart-seq2, TRUST4)", "flu_lung": "GSE317692 influenza lung",
         "bone_marrow_pc": "GSE253857 bone-marrow plasma cells", "malaria": "GSE286215 Plasmodium d0-14",
         "malaria_late": "GSE286215 Plasmodium d10-42", "gut_gc_reporter": "GSE284622 gut GC-fate reporter",
         "covid_bcell": "GSE171703 SARS-CoV-2 bait-sorted B cells", "spont_gc": "GSE203132 spontaneous GCs",
         "larry_d2": "GSE140802 LARRY day 2 (clone ids given)", "larry_d46": "GSE140802 LARRY days 4/6 (clone ids given)"}
OURS = ("api.py", "dynamics.py", "programmes.py", "profiles.py", "pp.py", "stats.py", "maptune.py", "dandelion.py")


def published_icc():
    """Clonal ICC of each case study, from dataset_overview.md."""
    out = {}
    for line in OVERVIEW.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) > 9 and re.fullmatch(r"[a-z_0-9]+", cells[0] or "-"):
            try:
                out[cells[0]] = (int(cells[6].replace(",", "")), float(cells[8]))
            except ValueError:
                pass
    return out


def main(src: Path):
    pub = published_icc()
    dst = HERE / "results"
    dst.mkdir(exist_ok=True)
    rows = []
    for f in sorted(src.glob("*.json")):
        if f.name == "tutorial_blocks.json":
            continue
        r = json.loads(f.read_text())
        ds = r["dataset"]
        for extra in (f.with_name(f"{ds}_summary.txt"), f.with_name(f"{ds}_methods.txt")):
            if extra.exists():
                shutil.copy(extra, dst / extra.name)
        shutil.copy(f, dst / f.name)
        ours = sorted({w["message"].split(";")[0][:70] for w in r["warnings"]
                       if w["file"] in OURS and w["category"] == "UserWarning"})
        clones_pub, icc_pub = pub.get(ds, (None, None))
        rows.append({"dataset": NAMES.get(ds, ds), "key": ds, "errors": 0 if r["error"] is None else 1,
                     "cells": r.get("cells"), "contigs matching a cell": r.get("contigs_matching_a_cell"),
                     "clones with >= 2 cells": r.get("clones_ge2"), "case study": clones_pub,
                     "clonal ICC": r.get("icc"), "case study ICC": icc_pub, "programmes": r.get("programmes"),
                     "minutes": round(r["seconds_total"] / 60, 1), "peak memory, GB": r["peak_rss_gb"],
                     "Threadfin's own notes": "; ".join(ours), "not run": "; ".join((r.get("skipped") or {}).keys())})
    t = pd.DataFrame(rows)
    t.to_csv(dst / "tutorial_path.csv", index=False)

    def fmt(v, kind):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return {"int": f"{int(v):,}", "pct": f"{100 * v:.0f} %", "icc": f"{v:.3f}"}.get(kind, str(v))

    lines = ["| dataset | errors | cells | contigs matching a cell | clones ≥ 2 cells (case study) | clonal ICC "
             "(case study) | minutes | peak memory |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['dataset']} | {r['errors']} | {fmt(r['cells'], 'int')} | {fmt(r['contigs matching a cell'], 'pct')} | "
                     f"{fmt(r['clones with >= 2 cells'], 'int')} ({fmt(r['case study'], 'int')}) | "
                     f"{fmt(r['clonal ICC'], 'icc')} ({fmt(r['case study ICC'], 'icc')}) | {r['minutes']} | "
                     f"{r['peak memory, GB']} GB |")
    readme = HERE / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"(<!-- BEGIN:table -->).*?(<!-- END:table -->)", lambda m: m.group(1) + "\n\n" + "\n".join(lines)
                  + "\n\n" + m.group(2), text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    print(t[["key", "errors", "clones with >= 2 cells", "case study", "clonal ICC", "case study ICC"]].to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
