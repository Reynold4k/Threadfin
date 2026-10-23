#!/usr/bin/env python
"""Figure 4 restricted to cells with a complete productive heavy + light chain pair (GSE287123, RBD cohorts).

The clones of Figure 4 are Dandelion's paired-chain clones, but Dandelion also assigns a clone to cells in
which only one of the two chains was recovered (from that chain alone), and one profiled two-cell group has no
heavy chain at all. Here every cell without at least one productive heavy chain and one productive light chain
loses its clone label, the clone profiles are recomputed (clones left with one cell drop out) and the analyses
of Figure 4 are repeated with the code that made the figure (figures/reserve/source/figure4_analysis_2026-10-19.py):
the clone-level association of the division gate in each cohort, of the RBD bait and of the zone gate; the
library crossover (cosine of the division directions of the two cohorts); and the within-mouse correlations
of the first profile component with the recently divided fraction. The full data are analysed the same way
first, which reproduces the saved Figure 4 values.

Usage: python complete_pairs.py            (writes results/ next to this file)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import threadfin as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "manuscript" / "figures" / "reserve" / "source" / "figure4_analysis_2026-10-19.py"
SAVED = ROOT / "manuscript" / "figures" / "reserve" / "figure4_rbd_2026-10-19" / "summary.json"
OUT = HERE / "results"
sys.path.insert(0, str(ROOT / "manuscript" / "case_studies"))

from dandelion_clones import contigs_for  # noqa: E402  (the contig files and cell names of the clone calls)

_spec = importlib.util.spec_from_file_location("figure4_analysis", SRC)
fig4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig4)


def chains() -> pd.DataFrame:
    t = contigs_for("mouse_rbd")
    t = t[t["productive"].astype(str).str.upper().isin(["T", "TRUE", "1"])]
    locus = t["locus"].astype(str) if "locus" in t else t["v_call"].astype(str).str[:3]
    c = t.groupby([t["cell"].astype(str), locus]).size().unstack(fill_value=0)
    return pd.DataFrame({"n_heavy": c.get("IGH", 0), "n_light": c.get("IGK", 0) + c.get("IGL", 0)})


def analyse(a, label: str) -> dict:
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", min_cells=2, verbose=False)
    tf.tl.embed_clones(a, min_reliability=0.0, verbose=False, **fig4.MAP)
    p = a.uns["threadfin"]["profiles"]
    prof = {"table": p["clone_table"].copy(), "features": p["features"].copy()}
    t = fig4.clone_table(a, prof)
    fig4.principal_axes(prof["features"], t)
    assoc = fig4.associations(a)
    cross, _ = fig4.crossover(prof["features"], t)
    corr = fig4.correlations(t)
    mice = {k: v for k, v in corr["mouse"].items() if k.startswith("recently_divided|")}
    out = {"clones": int(len(t)), "cells_in_clones": int(t["n_cells"].sum()),
           "clones_by_cohort": t["cohort"].value_counts().to_dict(),
           "associations": {k: {"excess_r2": v["excess_r2"], "r2": v["r2"], "null_mean": v["null_mean"],
                                "p_value": v["p_value"], "n_clones": v["n_clones"]} for k, v in assoc.items()},
           "crossover_cosine": cross["cosine"], "crossover_null_q99": cross["null_q99"], "crossover_p": cross["p_value"],
           "tube_effect_relative_to_division": cross["tube_effect_relative"],
           "division_rho_by_cohort": {k: v["rho"] for k, v in corr["cohort"].items()},
           "mice_with_positive_division_rho": int(sum(v["rho"] > 0 for v in mice.values())),
           "mice": int(len(mice))}
    fig4.stamp(f"{label}: {json.dumps(out, default=float)}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = fig4.load()
    ch = chains().reindex(a.obs_names.astype(str)).fillna(0).astype(int)
    complete = ((ch["n_heavy"] >= 1) & (ch["n_light"] >= 1)).to_numpy()
    in_clone = a.obs["clone_id"].notna().to_numpy()
    full = analyse(a.copy(), "all cells (Figure 4)")
    b = a.copy()
    b.obs.loc[~complete, "clone_id"] = None
    sub = analyse(b, "complete heavy + light pairs only")
    counts = {"cells_with_clone": int(in_clone.sum()),
              "cells_with_clone_without_complete_pair": int((in_clone & ~complete).sum()),
              "of_which_no_heavy_chain": int((in_clone & (ch["n_heavy"] == 0).to_numpy()).sum()),
              "of_which_no_light_chain": int((in_clone & (ch["n_light"] == 0).to_numpy()).sum()),
              "cells_with_two_or_more_heavy_chains": int((in_clone & (ch["n_heavy"] >= 2).to_numpy()).sum())}
    saved = json.loads(SAVED.read_text())
    check = {"saved_crossover_cosine": saved["crossover"]["cosine"],
             "saved_division_excess_r2": {k: saved["associations"][k]["excess_r2"] for k in saved["associations"]}}
    res = {"counts": counts, "all_cells": full, "complete_pairs": sub, "saved_figure4": check,
           "threadfin": tf.__version__}
    (OUT / "summary.json").write_text(json.dumps(fig4.jsonable(res), indent=1) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
