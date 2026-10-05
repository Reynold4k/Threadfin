#!/usr/bin/env python
"""Audit the Kim et al. spike label before using it in Threadfin figures.

This is deliberately an audit of provenance rather than an affinity analysis.
The public heavy-chain table has a clone-level ``s_pos_clone`` call, an
optional recombinant-mAb ``elisa`` call, and SHM fields; it has no numeric
binding affinity (for example KD/EC50).  Results use inferred Threadfin IGH
families as the unit, never individual cells.

Usage (from repository root)::

    /data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python \
        case_studies/spike_binding_audit.py
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("/data/scratch/projects/punim1236/threadfin_data/gse195673_ln_vaccine")
DEFAULT_RESULT = ROOT / "case_studies/results/ln_vaccine"
DEFAULT_OUT = ROOT / "case_studies/results/spike_binding_audit"


def status(v) -> str:
    """Keep missing author calls distinct from explicit False."""
    if pd.isna(v):
        return "missing"
    if str(v).strip().upper() == "TRUE":
        return "S+"
    if str(v).strip().upper() == "FALSE":
        return "S-"
    return "other"


def one_status(values: pd.Series) -> str:
    vals = {status(v) for v in values}
    return next(iter(vals)) if len(vals) == 1 else "mixed"


def main(raw: Path, results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    use = ["cell_id", "donor", "sample", "timepoint", "tissue", "clone_id",
           "s_pos_clone", "expressed_id", "elisa", "nuc_RS_freq_19_312"]
    heavy = pd.read_csv(raw / "bcr_heavy.tsv.gz", sep="\t", usecols=use, low_memory=False)
    # This matches load_ln_vaccine: one heavy chain per cell, retaining source order.
    heavy = heavy.dropna(subset=["cell_id"]).drop_duplicates("cell_id").set_index("cell_id")
    heavy.index = heavy.index.astype(str)
    heavy["author_binding"] = heavy["s_pos_clone"].map(status)
    heavy["elisa_status"] = heavy["elisa"].map(
        lambda v: "ELISA+" if str(v).upper() == "TRUE" else ("ELISA-" if str(v).upper() == "FALSE" else "missing"))
    heavy["shm_frequency"] = pd.to_numeric(heavy["nuc_RS_freq_19_312"], errors="coerce")

    # Reuse the saved Threadfin run, so this audit neither reclusters receptors
    # nor uses expression to define the crosswalk.
    cells = pd.read_csv(results / "cells.csv.gz", index_col=0)
    cells.index = cells.index.astype(str)
    cells = cells.rename(columns={"clone_id": "threadfin_family", "clone_programme": "programme"})
    keep = ["threadfin_family", "programme", "donor", "sample", "timepoint", "tissue"]
    x = cells[[c for c in keep if c in cells.columns]].join(heavy, how="left", rsuffix="_raw")
    x = x.dropna(subset=["threadfin_family"]).copy()
    # Prefer source donor/sample metadata, while retaining Threadfin columns for an audit trail.
    for c in ("donor", "sample", "timepoint", "tissue"):
        if c + "_raw" in x:
            x[c] = x[c + "_raw"].fillna(x[c])

    raw_status = heavy["author_binding"].value_counts()
    source = pd.DataFrame([
        {"metric": "heavy_rows_after_one_per_cell", "value": len(heavy)},
        {"metric": "raw_author_binding_Splus_cells", "value": int(raw_status.get("S+", 0))},
        {"metric": "raw_author_binding_Sminus_cells", "value": int(raw_status.get("S-", 0))},
        {"metric": "raw_author_binding_missing_cells", "value": int(raw_status.get("missing", 0))},
        {"metric": "Threadfin_family_cells_with_heavy", "value": len(x)},
        {"metric": "Threadfin_families", "value": x.threadfin_family.nunique()},
        {"metric": "author_binding_Splus_cells", "value": int((x.author_binding == "S+").sum())},
        {"metric": "author_binding_Sminus_cells", "value": int((x.author_binding == "S-").sum())},
        {"metric": "author_binding_missing_cells", "value": int((x.author_binding == "missing").sum())},
    ])
    source.to_csv(out / "input_coverage.csv", index=False)

    fam = x.groupby("threadfin_family", observed=True).agg(
        donor=("donor", "first"), n_cells=("threadfin_family", "size"),
        n_author_clones=("clone_id", "nunique"), author_clone_id=("clone_id", "first"),
        programme=("programme", "first"), mean_shm_frequency=("shm_frequency", "mean"),
        n_shm_cells=("shm_frequency", "count"), n_samples=("sample", "nunique"),
        n_timepoints=("timepoint", "nunique"), n_elisa_positive=("elisa_status", lambda z: (z == "ELISA+").sum()),
        n_elisa_negative=("elisa_status", lambda z: (z == "ELISA-").sum()),
    ).reset_index()
    fam["author_binding"] = x.groupby("threadfin_family", observed=True)["s_pos_clone"].agg(one_status).values
    fam.to_csv(out / "threadfin_family_crosswalk.csv", index=False)

    author = x.groupby(["donor", "clone_id"], observed=True).agg(
        n_cells=("threadfin_family", "size"), n_threadfin_families=("threadfin_family", "nunique"),
        threadfin_family=("threadfin_family", "first"), mean_shm_frequency=("shm_frequency", "mean"),
    ).reset_index()
    author["author_binding"] = x.groupby(["donor", "clone_id"], observed=True)["s_pos_clone"].agg(one_status).values
    author.to_csv(out / "author_clone_crosswalk.csv", index=False)

    # One family contributes once in each donor/timepoint; labels remain three-state.
    ft = x.groupby(["threadfin_family", "donor", "timepoint"], observed=True).agg(
        mean_shm_frequency=("shm_frequency", "mean"), n_cells=("threadfin_family", "size"),
        n_shm_cells=("shm_frequency", "count"), n_author_clones=("clone_id", "nunique"),
    ).reset_index()
    ft["author_binding"] = x.groupby(["threadfin_family", "donor", "timepoint"], observed=True)["s_pos_clone"].agg(one_status).values
    ft.to_csv(out / "family_timepoint_source.csv", index=False)
    dt = ft.groupby(["donor", "timepoint", "author_binding"], observed=True).agg(
        n_threadfin_families=("threadfin_family", "nunique"),
        n_families_with_shm=("n_shm_cells", lambda z: int((z > 0).sum())),
        median_family_shm=("mean_shm_frequency", "median"),
        mean_family_shm=("mean_shm_frequency", "mean"),
    ).reset_index()
    dt.to_csv(out / "donor_timepoint_binding_shm.csv", index=False)

    # Programme enrichment is shown as donor-level family proportions; unknown
    # calls are coverage, never silently counted as experimental negatives.
    fp = fam[fam.programme.notna()].copy()
    rows = []
    for (donor, programme), g in fp.groupby(["donor", "programme"], observed=True):
        n_known = int(g.author_binding.isin(["S+", "S-"]).sum())
        rows.append({"donor": donor, "programme": programme, "n_threadfin_families": len(g),
                     "n_Splus": int((g.author_binding == "S+").sum()),
                     "n_Sminus": int((g.author_binding == "S-").sum()),
                     "n_missing_or_ambiguous": int((~g.author_binding.isin(["S+", "S-"])).sum()),
                     "n_known": n_known,
                     "frac_Splus_among_known": ((g.author_binding == "S+").sum() / n_known) if n_known else np.nan})
    enrich = pd.DataFrame(rows)
    enrich.to_csv(out / "donor_programme_binding_coverage.csv", index=False)

    # Compact visual: donor/family summaries only.  The data files above are its source data.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot = dt[dt.author_binding.isin(["S+", "S-"])].copy()
    order = sorted(plot.timepoint.unique(), key=lambda s: (len(str(s)), str(s)))
    for label, colour in [("S+", "#2166ac"), ("S-", "#b2182b")]:
        q = plot[plot.author_binding == label]
        for i, tp in enumerate(order):
            y = q.loc[q.timepoint == tp, "median_family_shm"].dropna()
            axes[0].scatter(np.full(len(y), i + (-.12 if label == "S+" else .12)), y, color=colour, alpha=.75, s=24,
                            label=label if i == 0 else None)
    axes[0].set_xticks(range(len(order)), order, rotation=45, ha="right")
    axes[0].set_ylabel("donor median of family mean SHM frequency")
    axes[0].set_title("SHM is reported separately from affinity")
    axes[0].legend(frameon=False, title="author clone label")
    for programme, g in enrich.groupby("programme", observed=True):
        axes[1].scatter([programme] * len(g), g.frac_Splus_among_known, s=24, color="#2166ac", alpha=.75)
    axes[1].set_ylim(-.03, 1.03)
    axes[1].set_ylabel("S+ Threadfin families / known-label families")
    axes[1].set_xlabel("expression programme")
    axes[1].set_title("donor-level coverage; missing calls excluded")
    fig.savefig(out / "binding_shm_donor_summary.png", dpi=220)
    fig.savefig(out / "binding_shm_donor_summary.pdf")

    print(f"wrote {out}")
    print(source.to_string(index=False))
    print("author clones split across >1 Threadfin family:", int((author.n_threadfin_families > 1).sum()))
    print("Threadfin families with >1 author clone:", int((fam.n_author_clones > 1).sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.raw, a.results, a.out)
