#!/usr/bin/env python
"""Diagnose the GEX <-> BCR cell-name mapping for GSE284622 reporter mice.

Prints name formats on both sides and the match rate under candidate rules.
Run after gex_reporter/ export exists.
"""
import os
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parent / "data")) / "gse284622_gut_gc"

meta = pd.read_csv(DATA / "gex_reporter" / "metadata.csv.gz", low_memory=False)
print("GEX metadata columns:", list(meta.columns))
print("GEX n cells:", len(meta))
for c in meta.columns:
    if meta[c].dtype == object and meta[c].nunique() < 40:
        print(f"  {c}: {meta[c].value_counts().to_dict()}")
names = meta["cell"].astype(str) if "cell" in meta.columns else meta.iloc[:, 0].astype(str)
print("GEX cell name examples:", names.head(6).tolist())

frames = []
for m in ("mouse2", "mouse3", "mouse4"):
    t = pd.read_csv(DATA / f"GSE284621_{m}_airr_rearrangement.tsv.gz", sep="\t", low_memory=False,
                    usecols=["sequence_id", "cell_id", "locus", "productive", "v_call", "j_call",
                             "c_call", "junction", "junction_aa", "sequence_alignment",
                             "germline_alignment", "duplicate_count", "subject_id", "sample_id",
                             "tissue", "GC", "clone_id"])
    frames.append(t)
airr = pd.concat(frames, ignore_index=True)
print("\nAIRR rows:", len(airr))
print("AIRR cell_id examples:", airr["cell_id"].head(3).tolist())
print("AIRR sample_id x subject:", airr.groupby(["subject_id", "sample_id"]).size().to_dict())

# candidate rules: how to rebuild the GEX cell name from AIRR fields
parts = names.str.rsplit("_", n=1, expand=True)
print("\nGEX name split at last underscore -> prefix examples:", parts[0].head(3).tolist(),
      "| barcode examples:", parts[1].head(3).tolist())

airr["barcode_core"] = airr["cell_id"].astype(str).str.replace(r"-\d+$", "", regex=True)  # CGTT...-1
rule_a = airr["sample_id"].astype(str) + "_" + airr["barcode_core"]
airr["barcode_plain"] = airr["barcode_core"].str.replace(r"-1$", "", regex=True)
rule_b = airr["sample_id"].astype(str) + "_" + airr["barcode_plain"]
gex_set = set(names)
for label, rule in (("sample_BARCODE-1", rule_a), ("sample_BARCODE", rule_b)):
    hit = rule.isin(gex_set).mean()
    print(f"rule {label}: {hit:.3%} of AIRR rows match a GEX cell")
