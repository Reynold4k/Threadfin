#!/usr/bin/env python
"""How much did reused contig ids change the GSE287123 clones, and does Figure 4 depend on it?

Cell Ranger names contigs after the cell barcode (``ACGGGTTTCATTAAGC-1_contig_1``). A mouse of GSE287123 was
sequenced in several libraries, so two of its cells can carry the same barcode and therefore the same contig
ids. Dandelion indexes contigs by this id, and threadfin <= 4.3.1 passed the ids on unchanged, so the chains
of such cells were mixed (18 cells of 21,939 in the RBD data, 8 of 4,855 in the NP data). threadfin 4.4.0
renames duplicated ids after their cell.

This script calls the clones twice in one Python session (``PYTHONHASHSEED=0``), with and without the
renaming, compares the partitions, and repeats the Figure 4 statistics (as complete_pairs.py) on the corrected
RBD clones.

Usage: PYTHONHASHSEED=0 python contig_id_fix.py      (needs sc-dandelion; writes results/contig_id_fix.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import threadfin.dandelion as tfd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "case_studies"))
from dandelion_clones import CACHE, contigs_for  # noqa: E402

import complete_pairs as cp  # noqa: E402  (loads the Figure 4 analysis module)


def call(ds: str, unique: bool) -> pd.Series:
    obs = ad.read_h5ad(CACHE / f"{ds}_embedding.h5ad", backed="r").obs
    obs.index = obs.index.astype(str)
    t = contigs_for(ds)
    t = t[t["cell"].astype(str).isin(obs.index)].copy()
    t["donor"] = t["cell"].astype(str).map(obs["donor"].astype(str))
    if "locus" not in t.columns:
        t["locus"] = t["v_call"].astype(str).str.upper().str[:3]
    original = tfd._unique_sequence_ids
    if not unique:
        tfd._unique_sequence_ids = lambda frame, verbose=True: frame          # threadfin <= 4.3.1 behaviour
    try:
        return tfd.define_clones_dandelion(t, cell_key="cell", donor_key="donor", identity=0.85, verbose=False)
    finally:
        tfd._unique_sequence_ids = original


def compare(old: pd.Series, new: pd.Series, affected: set) -> dict:
    cells = old.index.union(new.index)
    o = old.reindex(cells).fillna("none").astype(str)
    n = new.reindex(cells).fillna("none").astype(str)
    # a cell's membership changed if the set of cells sharing its clone differs
    mates_o = o.groupby(o).groups
    mates_n = n.groupby(n).groups
    changed = [c for c in cells if set(mates_o[o[c]]) != set(mates_n[n[c]])]
    return {"cells": int(len(cells)), "adjusted_rand_index": float(adjusted_rand_score(o, n)),
            "cells_whose_clone_mates_changed": len(changed),
            "of_which_had_a_reused_contig_id": int(sum(c in affected for c in changed)),
            "cells_with_a_reused_contig_id": int(len(affected)),
            "light_chain_only_ids_before": int(o.str.contains("B_VJ_").sum()),
            "light_chain_only_ids_after": int(n.str.contains("B_VJ_").sum())}


def reused(ds: str) -> set:
    """Cells with a contig id that another cell of the same mouse also carries."""
    obs = ad.read_h5ad(CACHE / f"{ds}_embedding.h5ad", backed="r").obs
    obs.index = obs.index.astype(str)
    t = contigs_for(ds)
    t = t[t["cell"].astype(str).isin(obs.index)]
    donor = t["cell"].astype(str).map(obs["donor"].astype(str))
    n_cells = t.assign(donor=donor, sid=t["sequence_id"].astype(str)).groupby(["donor", "sid"])["cell"].transform("nunique")
    return set(t.loc[n_cells > 1, "cell"].astype(str))


def name_like_saved(new: pd.Series, saved: pd.Series) -> pd.Series:
    """Give each new clone the id of the saved clone with which it shares most cells (largest new clones first),
    so the saved clone-level module scores of Figure 4 can be joined; a clone without a match keeps its new id."""
    new, saved = new.astype(object), saved.reindex(new.index).astype(object)
    both = pd.DataFrame({"new": new, "saved": saved}).dropna()
    overlap = both.groupby(["new", "saved"]).size().sort_values(ascending=False)
    rename, taken = {}, set()
    for (n, s_), _ in overlap.items():
        if n not in rename and s_ not in taken:
            rename[n] = s_
            taken.add(s_)
    return new.map(lambda c: rename.get(c, c) if pd.notna(c) else c)


def principal_axes_omit_missing(features, t):
    """figure4_analysis.principal_axes, orienting PC1 on the clones that have a saved module score."""
    import numpy as np
    from scipy import stats
    x = features.loc[t.index].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    _, s_, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:3].T
    sign = np.sign(stats.spearmanr(pcs[:, 0], t["dark zone / cycling"], nan_policy="omit").statistic)
    t["pc1"] = sign * pcs[:, 0]
    t["pc2"], t["pc3"] = pcs[:, 1], pcs[:, 2]
    t["pc1_decile"] = pd.qcut(t["pc1"], 10, labels=False) + 1
    return {"orientation_sign": float(sign)}


def main():
    out = {"python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    calls = {}
    for ds in ("mouse_rbd", "mouse_np"):
        old, new = call(ds, unique=False), call(ds, unique=True)
        calls[ds] = new
        out[ds] = compare(old, new, reused(ds))
        print(ds, json.dumps(out[ds]), flush=True)
    a = cp.fig4.load()
    a.obs["clone_id"] = name_like_saved(calls["mouse_rbd"].reindex(a.obs_names.astype(str)), a.obs["clone_id"])
    cp.fig4.principal_axes = principal_axes_omit_missing
    out["figure4_with_corrected_clones"] = cp.analyse(a, "corrected contig ids")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "contig_id_fix.json").write_text(json.dumps(cp.fig4.jsonable(out), indent=1) + "\n")


if __name__ == "__main__":
    main()
