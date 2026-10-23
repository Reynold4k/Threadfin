#!/usr/bin/env python3
"""Integrate corrected case studies, retaining the publication's saved display coordinates.

Usage: python integrate_algorithm_revision.py --reruns PATH --backup PATH
Fresh diagnostic PNGs retain the rerun layouts; publication figures read the
coordinate-preserved CSV tables. All biological identifiers and display-eligible
families must agree before any dataset is updated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "case_studies/results"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(reruns, backup):
    audited = []
    plans = []
    # Validate every dataset before modifying any of them.
    for new in sorted(reruns.iterdir()):
        if not (new / "summary.json").exists():
            continue
        old = RESULTS / new.name
        baseline = backup / new.name if (backup / new.name / "summary.json").exists() else old
        a, b = [json.loads((p / "summary.json").read_text()) for p in (baseline, new)]
        assert a["qc"] == b["qc"], new.name
        tables = {}
        for filename, coords in [("clone_table.csv", ["x", "y"]),
                                 ("cells.csv.gz", ["umap_1", "umap_2"])]:
            x, y = [pd.read_csv(p / filename, index_col=0, low_memory=False) for p in (baseline, new)]
            assert x.index.equals(y.index), (new.name, filename, "identities")
            if filename == "clone_table.csv":
                np.testing.assert_array_equal(x.n_cells, y.n_cells)
                np.testing.assert_array_equal(x.reliability.ge(.5), y.reliability.ge(.5))
            else:
                assert x.clone_id.fillna("").equals(y.clone_id.fillna("")), new.name
            # Independent biological annotations added after run_case_study
            # remain valid because identities and captured members are identical.
            for col in x.columns.difference(y.columns):
                y[col] = x[col]
            for col in coords:
                if col in x:
                    y[col] = x[col]
            tables[filename] = y
        audited.append(dict(dataset=new.name, identities_equal=True,
                            capture_counts_equal=True, reliable_family_set_equal=True,
                            preserved_coordinates={"clone_table.csv": ["x", "y"],
                                                   "cells.csv.gz": ["umap_1", "umap_2"]},
                            original_summary_sha256=digest(baseline / "summary.json"),
                            rerun_summary_sha256=digest(new / "summary.json"),
                            old_programmes=a.get("programmes"), new_programmes=b.get("programmes"),
                            old_memory=a.get("memory"), new_memory=b.get("memory"),
                            old_min_cells=a["coherence"]["min_cells_reliability_0.5"],
                            new_min_cells=b["coherence"]["min_cells_reliability_0.5"]))
        plans.append((old, new, tables))
    assert len(plans) == 12, "Expected all twelve completed case studies"
    for old, new, tables in plans:
        dst_backup = backup / new.name
        if not dst_backup.exists():
            shutil.copytree(old, dst_backup, ignore=shutil.ignore_patterns("lineage"))
        for path in new.iterdir():
            if path.is_file() and path.name not in tables:
                shutil.copy2(path, old / path.name)
        for name, table in tables.items():
            compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
            table.to_csv(old / name, compression=compression)
        # Remove obsolete generated summaries, keeping independently produced
        # lineage/sequence audits outside this script's ownership.
        for pattern in ["programme*.csv", "memory_pairs_*.csv", "memory_transitions*.csv"]:
            for path in old.glob(pattern):
                if not (new / path.name).exists():
                    path.unlink()
        figures = old / "figures"
        if (new / "figures").exists():
            figures.mkdir(exist_ok=True)
            for path in figures.glob("*.png"):
                if not (new / "figures" / path.name).exists():
                    path.unlink()
            for path in (new / "figures").glob("*.png"):
                shutil.copy2(path, figures / path.name)
    out = RESULTS / "algorithm_revision/case_integration_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(
        status="passed", datasets=audited,
        note="Published maps retain their previous display coordinates after identity, capture and eligibility checks. Fresh diagnostic PNGs use rerun coordinates. Statistical estimates and programme annotations come from the corrected algorithms."
    ), indent=2) + "\n")
    print("Integrated and audited", len(plans), "datasets:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reruns", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    main(args.reruns, args.backup)
