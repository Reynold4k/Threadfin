#!/usr/bin/env python
"""Spontaneous-GC dataset (GSE203132, Castrillon et al. 2023): robustness of the state-diversity analysis
of Figure 3d-f for the 2026-10-17 revision.

Figure 3d-f relate clone state diversity (Shannon entropy of RNA-annotated states) to the first coordinate
of the displayed clone embedding (UMAP). Here the same clones are re-profiled with the Figure 3 settings and
the analysis is repeated (i) on profile-space axes instead of the display coordinate (first principal
component of the clone profiles; projection on the axis from the GC-programme to the ASC-programme
centroid), (ii) with clone size controlled (rarefied entropy and rarefied probability of containing a
memory B cell, both at five annotated cells; partial correlation on log clone size) and (iii) within each
mouse. Outputs: reserve/revision_2026-10-17/spont_gc/.
"""
from __future__ import annotations

import json
import time
from math import comb
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

import threadfin as tf

ROOT = Path(__file__).resolve().parents[4]
CACHE = (ROOT.parent / "internal_validation" / "algorithm_revision" / "embedding_cache"
         / "spont_gc_embedding.h5ad")
RES = ROOT / "manuscript" / "case_studies" / "results" / "spont_gc"
OUT = ROOT / "manuscript" / "figures" / "reserve" / "revision_2026-10-17" / "spont_gc"
STATES = ["GC", "ASC", "Fcrl5+ MemB", "CD23+ MemB", "other MemB"]
MEM = ["Fcrl5+ MemB", "CD23+ MemB", "other MemB"]
RARE_N, N_DRAW = 5, 1000
T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def rank_partial(x, y, z):
    """Spearman partial correlation of x and y given z (Pearson on rank residuals)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = [r - np.polyval(np.polyfit(rz, r, 1), rz) for r in (rx, ry)]
    r, p = stats.pearsonr(*res)
    return float(r), float(p)


def spear(x, y):
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(CACHE)
    ddl = pd.read_csv(ROOT / "manuscript" / "case_studies" / "results" / "dandelion_clones" / "spont_gc.csv")
    a.obs["clone_id"] = ddl.set_index("cell")["clone_id"].reindex(a.obs_names).values
    cells = pd.read_csv(RES / "cells.csv.gz", index_col=0)
    assert (cells.index == a.obs_names).all()
    fate = pd.read_csv(RES / "spont_gc_fate_cells.csv.gz", index_col=0)["fate"].reindex(a.obs_names)
    a.obs["fate"] = fate.values
    stamp(f"{a.n_obs} cells; {a.obs['clone_id'].notna().sum()} with a clone")

    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor", representation="mean",
                         min_cells=2, verbose=False)
    icc = a.uns["threadfin"]["profiles"]["variance_components"]["icc"]
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", min_cells=2, verbose=False)
    table = tf.tl.embed_clones(a, n_neighbors=30, min_dist=0.3, min_reliability=0.0, verbose=False).copy()
    feats = a.uns["threadfin"]["profiles"]["features"].copy()
    ref = pd.read_csv(RES / "clone_table.csv").set_index("clone_id")
    t = table.join(ref[["x", "y", "clone_programme"]], rsuffix="_saved")
    check = {"n_clones": int(len(t)), "max_abs_dx": float((t.x - t.x_saved).abs().max()),
             "max_abs_dy": float((t.y - t.y_saved).abs().max()), "icc": float(icc),
             "icc_saved": json.loads((RES / "summary.json").read_text())["coherence"]["icc"]}
    stamp(f"reproduction {check}")

    # profile-space axes, oriented so that larger values lie towards the ASC programme (as clone UMAP 1)
    pc1 = feats["kpc0"].copy()
    if stats.spearmanr(pc1, ref.loc[pc1.index, "x"]).statistic < 0:
        pc1 = -pc1
    prog = ref.loc[feats.index, "clone_programme"]
    c1, c2 = feats[prog == "P1"].mean(), feats[prog == "P2"].mean()
    v = (c2 - c1).to_numpy()
    gc_asc = pd.Series(((feats - c1).to_numpy() @ v) / (v @ v), index=feats.index)

    # state counts per clone (annotated cells only)
    obs = a.obs[a.obs["clone_id"].notna() & a.obs["fate"].isin(STATES)].copy()
    obs["clone_id"] = obs["clone_id"].astype(str)
    counts = pd.crosstab(obs["clone_id"], obs["fate"]).reindex(columns=STATES, fill_value=0)
    counts = counts.reindex(feats.index).fillna(0).astype(int)
    n_annot = counts.sum(axis=1)
    d = pd.DataFrame({"n_annotated": n_annot, "x": ref.loc[feats.index, "x"], "pc1": pc1, "gc_asc": gc_asc,
                      "donor": ref.loc[feats.index, "donor"], "programme": prog,
                      "n_mem": counts[MEM].sum(axis=1), "n_GC": counts["GC"], "n_ASC": counts["ASC"]})
    d["memory_containing"] = d["n_mem"] > 0
    sel = d[d["n_annotated"] >= RARE_N].copy()
    rng = np.random.default_rng(20261017)
    ent, rare = [], []
    for cid in sel.index:
        c = counts.loc[cid].to_numpy()
        ent.append(entropy(c))
        pool = np.repeat(np.arange(len(STATES)), c)
        draws = [entropy(np.bincount(rng.choice(pool, RARE_N, replace=False), minlength=len(STATES)))
                 for _ in range(N_DRAW)]
        rare.append(float(np.mean(draws)))
    sel["entropy"], sel["entropy_rarefied"] = ent, rare
    n, m = sel["n_annotated"].to_numpy(), sel["n_mem"].to_numpy()
    sel["p_mem_rarefied"] = [1 - comb(int(ni - mi), RARE_N) / comb(int(ni), RARE_N) for ni, mi in zip(n, m)]
    saved = pd.read_csv(RES / "spont_gc_fate_clones.csv", index_col=0)
    check["n_entropy_clones"] = int(len(sel))
    check["entropy_max_abs_diff_vs_saved"] = float((sel["entropy"] - saved.reindex(sel.index)["entropy"]).abs().max())

    res = {"reproduction": check, "n_clones_all": int(len(d)), "rarefaction_cells": RARE_N,
           "rarefaction_draws": N_DRAW}
    res["axis_agreement_all_clones"] = {"x_vs_pc1": spear(d["x"], d["pc1"]), "x_vs_gc_asc": spear(d["x"], d["gc_asc"]),
                                        "pc1_vs_gc_asc": spear(d["pc1"], d["gc_asc"])}
    logn = np.log(sel["n_annotated"].to_numpy())
    res["entropy"] = {}
    for axis in ("x", "pc1", "gc_asc"):
        r = {"raw": spear(sel["entropy"], sel[axis]), "rarefied": spear(sel["entropy_rarefied"], sel[axis])}
        r["partial_raw_given_log_n"] = dict(zip(("rho", "p"), rank_partial(sel["entropy"], sel[axis], logn)))
        r["within_mouse_raw"] = {mo: spear(g["entropy"], g[axis]) for mo, g in sel.groupby("donor")}
        r["within_mouse_rarefied"] = {mo: spear(g["entropy_rarefied"], g[axis]) for mo, g in sel.groupby("donor")}
        res["entropy"][axis] = r
    res["size_effects"] = {"entropy_vs_log_n": spear(sel["entropy"], logn),
                           "rarefied_entropy_vs_log_n": spear(sel["entropy_rarefied"], logn),
                           "pc1_vs_log_n": spear(sel["pc1"], logn)}
    # memory-containing clones along each axis (quartiles), observed and rarefied
    res["memory"] = {}
    for axis in ("x", "pc1", "gc_asc"):
        q = pd.qcut(sel[axis], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        tab = sel.groupby(q, observed=True).agg(n=("memory_containing", "size"),
                                                observed=("memory_containing", "mean"),
                                                n_mem=("memory_containing", "sum"),
                                                rarefied=("p_mem_rarefied", "mean"),
                                                median_n=("n_annotated", "median"))
        res["memory"][axis] = {"quartiles": tab.round(4).to_dict(orient="index"),
                               "rarefied_vs_axis": spear(sel["p_mem_rarefied"], sel[axis])}
    # which states do memory-containing clones share? (all profiled clones with >= 2 annotated cells)
    allc = d[d["n_annotated"] >= 2]
    memc = allc[allc["memory_containing"]]
    other_states = memc.assign(with_GC=memc["n_GC"] > 0, with_ASC=memc["n_ASC"] > 0)
    res["memory_clone_partners"] = {
        "n_memory_containing": int(len(memc)), "n_clones": int(len(allc)),
        "with_GC": int(other_states["with_GC"].sum()), "with_ASC": int(other_states["with_ASC"].sum()),
        "with_GC_only_partner": int((other_states["with_GC"] & ~other_states["with_ASC"]).sum()),
        "with_ASC_only_partner": int((~other_states["with_GC"] & other_states["with_ASC"]).sum()),
        "memory_only": int((~other_states["with_GC"] & ~other_states["with_ASC"]).sum())}
    # odds of containing GC versus ASC cells among memory-containing clones, versus other multi-cell clones
    nonmem = allc[~allc["memory_containing"]]
    res["memory_clone_partners"]["non_memory_with_GC"] = int((nonmem["n_GC"] > 0).sum())
    res["memory_clone_partners"]["non_memory_with_ASC"] = int((nonmem["n_ASC"] > 0).sum())
    res["memory_clone_partners"]["n_non_memory"] = int(len(nonmem))

    sel.to_csv(OUT / "clone_state_diversity.csv")
    d.to_csv(OUT / "clone_axes.csv")
    (OUT / "summary.json").write_text(json.dumps(jsonable(res), indent=1))
    stamp(f"wrote {OUT}")


if __name__ == "__main__":
    main()
