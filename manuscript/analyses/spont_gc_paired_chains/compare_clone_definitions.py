#!/usr/bin/env python
"""Does Figure 3a-f depend on defining the GSE203132 clones on the heavy chain only?

Figure 3a-f used heavy-chain clones, because the deposited AIRR table holds heavy chains only. Here the light
chains are taken from the deposited contigs (annotate_contigs.sh, IgBLAST) and clones are re-defined on paired
chains (paired_clones.py). The state-diversity analysis of Figure 3d-f and Supplementary Fig. 3 is then run for
both clone definitions with the same code (that of figures/reserve/source/revision_spont_gc_2026-10-17.py):
kernel profiles centred on the mouse, the clone map with the Figure 3 settings, the first principal component
of the profiles (oriented towards antibody-secreting cells), Shannon entropy of the RNA-annotated states of every
clone with >= 5 annotated cells (raw and rarefied to five cells), memory-containing clones by quartile of the
axis, and the same within each mouse. It also reports how much the light chain changes the clones.

Usage: python compare_clone_definitions.py <paired_clones.csv>   (writes results/ and figures/ next to this file)
       python compare_clone_definitions.py --plot                 (redraws the figure from results/)
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import threadfin as tf  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT.parent / "internal_validation" / "algorithm_revision" / "embedding_cache" / "spont_gc_embedding.h5ad"
RES = ROOT / "manuscript" / "case_studies" / "results" / "spont_gc"
HEAVY = ROOT / "manuscript" / "case_studies" / "results" / "dandelion_clones" / "spont_gc.csv"
OUT, FIG = HERE / "results", HERE / "figures"
STATES = ["GC", "ASC", "Fcrl5+ MemB", "CD23+ MemB", "other MemB"]
MEM = ["Fcrl5+ MemB", "CD23+ MemB", "other MemB"]
RARE_N, N_DRAW = 5, 1000


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def spear(x, y):
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def analyse(a, ids, label, fate):
    rng = np.random.default_rng(20261017)          # the seed of the Figure 3 revision analysis
    a = a.copy()
    a.obs["clone_id"] = ids.reindex(a.obs_names).values
    sizes = a.obs["clone_id"].dropna().value_counts()
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor", representation="mean",
                         min_cells=2, verbose=False)
    icc = float(a.uns["threadfin"]["profiles"]["variance_components"]["icc"])
    coh = tf.tl.clonal_coherence(a, strata_key="donor", n_perm=500, verbose=False)
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor", representation="kernel",
                         min_cells=2, verbose=False)
    table = tf.tl.embed_clones(a, n_neighbors=30, min_dist=0.3, min_reliability=0.0, verbose=False).copy()
    feats = a.uns["threadfin"]["profiles"]["features"]
    obs = a.obs[a.obs["clone_id"].notna() & fate.reindex(a.obs_names).isin(STATES).to_numpy()].copy()
    obs["fate"] = fate.reindex(obs.index).values
    obs["clone_id"] = obs["clone_id"].astype(str)
    counts = pd.crosstab(obs["clone_id"], obs["fate"]).reindex(columns=STATES, fill_value=0)
    counts = counts.reindex(feats.index.astype(str)).fillna(0).astype(int)
    asc_frac = counts["ASC"] / counts.sum(axis=1).replace(0, np.nan)
    pc1 = pd.Series(feats["kpc0"].to_numpy(), index=feats.index.astype(str))
    if stats.spearmanr(pc1, asc_frac, nan_policy="omit").statistic < 0:   # oriented towards ASCs, as in Fig. 3
        pc1 = -pc1
    x = pd.Series(table["x"].to_numpy(), index=table.index.astype(str))
    if stats.spearmanr(x, asc_frac, nan_policy="omit").statistic < 0:
        x = -x
    donor = table["donor"].astype(str) if "donor" in table else a.obs.groupby("clone_id")["donor"].first()
    d = pd.DataFrame({"n_annotated": counts.sum(axis=1), "pc1": pc1, "x": x,
                      "donor": donor.reindex(feats.index).astype(str).to_numpy(), "n_mem": counts[MEM].sum(axis=1)})
    sel = d[d["n_annotated"] >= RARE_N].copy()
    ent, rare = [], []
    for cid in sel.index:
        c = counts.loc[cid].to_numpy()
        ent.append(entropy(c))
        pool = np.repeat(np.arange(len(STATES)), c)
        rare.append(float(np.mean([entropy(np.bincount(rng.choice(pool, RARE_N, replace=False),
                                                       minlength=len(STATES))) for _ in range(N_DRAW)])))
    sel["entropy"], sel["entropy_rarefied"] = ent, rare
    n, m = sel["n_annotated"].to_numpy(), sel["n_mem"].to_numpy()
    sel["p_mem_rarefied"] = [1 - comb(int(ni - mi), RARE_N) / comb(int(ni), RARE_N) for ni, mi in zip(n, m)]
    sel["memory_containing"] = sel["n_mem"] > 0
    q = pd.qcut(sel["pc1"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    res = {
        "clone_definition": label, "cells_with_clone": int(sizes.sum()), "clones": int(sizes.size),
        "clones_ge2": int((sizes >= 2).sum()), "largest_clone": int(sizes.max()),
        "icc": icc, "icc_permutation": float(coh["icc"]), "icc_shuffled": float(coh["null_mean"]), "icc_p": float(coh["p_value"]),
        "clones_with_5_annotated_cells": int(len(sel)),
        "entropy_vs_pc1": spear(sel["entropy"], sel["pc1"]),
        "rarefied_entropy_vs_pc1": spear(sel["entropy_rarefied"], sel["pc1"]),
        "entropy_vs_clone_map_x": spear(sel["entropy"], sel["x"]),
        "within_mouse_entropy_vs_pc1": {mo: spear(g["entropy"], g["pc1"]) for mo, g in sel.groupby("donor")
                                       if len(g) >= 5},
        "memory_containing_by_pc1_quartile": sel.groupby(q, observed=True)["memory_containing"].mean().round(4).to_dict(),
        "rarefied_memory_vs_pc1": spear(sel["p_mem_rarefied"], sel["pc1"]),
    }
    res["mice_with_negative_rho"] = int(sum(v["rho"] < 0 for v in res["within_mouse_entropy_vs_pc1"].values()))
    res["mice_tested"] = int(len(res["within_mouse_entropy_vs_pc1"]))
    return res, sel


def partition_change(heavy, paired):
    both = heavy.index.intersection(paired.index)
    h, p = heavy.loc[both].astype(str), paired.loc[both].astype(str)
    tab = pd.crosstab(h, p)
    h_sizes = h.value_counts()
    multi = h_sizes.index[h_sizes >= 2]
    split = int(sum((tab.loc[c] > 0).sum() > 1 for c in multi))
    return {"cells_with_both_definitions": int(len(both)), "cells_heavy_only_definition": int(len(heavy) - len(both)),
            "cells_paired_only_definition": int(len(paired) - len(both)),
            "adjusted_rand_index": float(adjusted_rand_score(h, p)),
            "heavy_chain_clones_ge2": int(len(multi)), "heavy_chain_clones_split_by_light_chain": split,
            "largest_heavy_chain_clone": int(h_sizes.max()), "largest_paired_clone": int(p.value_counts().max())}


def main(paired_csv):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(CACHE)
    a.obs_names = a.obs_names.astype(str)
    fate = pd.read_csv(RES / "spont_gc_fate_cells.csv.gz", index_col=0)["fate"]
    fate.index = fate.index.astype(str)
    heavy = pd.read_csv(HEAVY).set_index("cell")["clone_id"].dropna()
    paired = pd.read_csv(paired_csv).set_index("cell")["clone_id"].dropna()
    heavy.index, paired.index = heavy.index.astype(str), paired.index.astype(str)
    out, sels = {"partition_change": partition_change(heavy, paired)}, {}
    for label, ids in (("heavy chain (Figure 3)", heavy), ("heavy and light chain", paired)):
        out[label], sels[label] = analyse(a, ids, label, fate)
        print(json.dumps({k: v for k, v in out[label].items() if not isinstance(v, dict) or "rho" in v},
                         default=float), flush=True)
    (OUT / "comparison.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for label, sel in sels.items():
        sel.to_csv(OUT / f"clones_{'paired' if 'light' in label else 'heavy'}.csv")
    plot(out, sels)
    print(json.dumps(out["partition_change"]))


def plot(out, sels):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    colours = {"heavy chain (Figure 3)": "#8a8984", "heavy and light chain": "#2a78d6"}
    for label, sel in sels.items():
        r = out[label]["entropy_vs_pc1"]
        axes[0].scatter(sel["pc1"].rank(pct=True), sel["entropy"], s=8, alpha=0.6, color=colours[label], lw=0,
                        label=f"{label}: rho {r['rho']:.2f} (n = {r['n']})")
        qq = out[label]["memory_containing_by_pc1_quartile"]
        axes[1].plot(list(qq), [100 * v for v in qq.values()], "o-", color=colours[label], label=label)
        wm = out[label]["within_mouse_entropy_vs_pc1"]
        axes[2].scatter(list(wm), [v["rho"] for v in wm.values()], color=colours[label], s=26, label=label)
    axes[0].set_xlabel("clone axis (rank of profile PC1, towards ASCs)")
    axes[0].set_ylabel("state entropy of the clone")
    axes[0].set_ylim(-0.05, 1.45)
    axes[1].set_xlabel("quartile of the clone axis")
    axes[1].set_ylabel("clones containing memory B cells (%)")
    axes[1].set_ylim(0, None)
    axes[2].axhline(0, color="#52514e", lw=0.8)
    axes[2].set_ylim(-1, 0.1)
    axes[2].set_ylabel("entropy vs axis, Spearman rho")
    axes[2].set_xlabel("mouse")
    for ax, letter, loc in zip(axes, "abc", ("upper right", "upper right", "lower right")):
        ax.legend(fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none", loc=loc)
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, ha="right", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GSE203132: the analysis of Figure 3d-f with clones on the heavy chain only (as published) and on "
                 "heavy + light chains", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.savefig(FIG / "heavy_vs_paired_clones.png", dpi=200)


def replot():
    out = json.loads((OUT / "comparison.json").read_text())
    sels = {label: pd.read_csv(OUT / f"clones_{'paired' if 'light' in label else 'heavy'}.csv", index_col=0)
            for label in ("heavy chain (Figure 3)", "heavy and light chain")}
    plot(out, sels)


if __name__ == "__main__":
    replot() if sys.argv[1] == "--plot" else main(sys.argv[1])
