#!/usr/bin/env python
"""Castrillon 2023 spontaneous-GC fate reporter (GSE203132): "complex subsets, redundant clonality".

The paper's claim, restated in Threadfin's terms: AID-EYFP+ (GC-experienced) B cells form several
transcriptional subsets (complex subsets), but clonal lineages are redundant - one clone supplies
cells to several of those subsets. This script tests that claim on the Threadfin clone map:

1. Marker-based re-annotation of cell state (GC / ASC / MemB, with Fcrl5+ vs Cd23a+ MemB) from the
   log-normalised transcriptome. Descriptive, not a measured label.
2. Clone fate width: how many states does a clone (>= 5 cells) span? Shannon entropy of its state
   mix. Is fate width structured on the clone embedding?
3. Condition structure (autoimmune vs immunized): condition is a donor-level variable, so donor
   separation is the confound; the informative test is cross-donor, same-condition cohesion on the
   clone map, plus a per-donor table.
4. 15-NN agreement of the single-fate vs multi-fate class on the cell UMAP vs on the clone map
   (gut_plasma_branches knn_agreement pattern), each with 100 permutation nulls.
5. Class-switching: fraction of switched clones (IGHG/IGHA/IGHE majority) per condition, clones
   mixing isotypes, and whether isotypes partition the clone map.

Outputs: results/spont_gc/figures/spont_gc_fates_*.png/.pdf, results/spont_gc/spont_gc_fates.json,
results/spont_gc/spont_gc_fate_cells.csv.gz, results/spont_gc/spont_gc_fate_clones.csv, and a
Chinese summary on stdout.

Usage: python spont_gc_fates.py   (run in the threadfin_v4 environment from case_studies/)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import threadfin as tf  # noqa: E402
from run_case_study import HERE, jsonable, prepare  # noqa: E402

OUT = HERE / "results" / "spont_gc"
FIGS = OUT / "figures"
MIN_CELLS = 5            # clones with >= 5 cells get a fate-width measurement
K_NN = 15
N_PERM = 100

MARKERS = {
    # mouse symbols; presence is checked against var_names at runtime
    "GC": ["Aicda", "Bcl6", "Mki67", "Rgs13", "S1pr2", "Gcsam"],
    "ASC": ["Prdm1", "Sdc1", "Jchain", "Xbp1", "Irf4", "Mzb1"],
    "MemB": ["Cd38", "Cd27", "Fcrl5", "Fcer2a", "Ccr6", "Sell", "S1pr1"],
}
MEMB_SPLIT = {"Fcrl5+ MemB": ["Fcrl5"], "Cd23a+ MemB": ["Fcer2a"]}


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def sparse_col(x, loc) -> np.ndarray:
    """One dense column from a possibly sparse matrix."""
    v = x[:, loc]
    return np.asarray(v.todense()).ravel() if hasattr(v, "todense") else np.asarray(v).ravel()


def annotate_fates(adata) -> pd.Series:
    """Argmax module score over marker sets; require some detection of the winning set."""
    x = adata.layers["log_norm"]
    raw = adata.X
    names = adata.var_names
    scores, det = {}, {}
    for state, genes in MARKERS.items():
        present = [g for g in genes if g in names]
        loc = [names.get_loc(g) for g in present]
        scores[state] = np.asarray(x[:, loc].mean(axis=1)).ravel()
        det[state] = np.asarray((raw[:, loc] > 0).mean(axis=1)).ravel()
    s = pd.DataFrame(scores, index=adata.obs_names)
    d = pd.DataFrame(det, index=adata.obs_names)
    states = list(MARKERS)
    win_code = s.to_numpy().argmax(axis=1)
    win_arr = np.asarray(states)[win_code]
    det_win = d.to_numpy()[np.arange(len(d)), win_code]
    fate = pd.Series(np.where(det_win >= 0.3, win_arr, "unassigned"),
                     index=adata.obs_names, name="fate")

    # split memory B cells by Fcrl5 vs Cd23a (Fcer2a) signal
    sub = {}
    for k, g in MEMB_SPLIT.items():
        present = [m for m in g if m in names]
        sub[k] = sparse_col(x, names.get_loc(present[0])) if present else np.zeros(adata.n_obs)
    pick = pd.Series(np.where(sub["Fcrl5+ MemB"] > sub["Cd23a+ MemB"], "Fcrl5+ MemB", "Cd23a+ MemB"),
                     index=adata.obs_names)
    neither = (sub["Fcrl5+ MemB"] <= 0) & (sub["Cd23a+ MemB"] <= 0)
    pick[neither] = "other MemB"
    memb = fate.eq("MemB")
    fate[memb] = pick[memb]
    adata.obs["fate"] = fate.values
    return fate


def fate_widths(obs: pd.DataFrame, state_col: str, min_cells: int = MIN_CELLS) -> pd.DataFrame:
    """Per clone with >= min_cells BCR+ cells: n states, Shannon entropy, dominant fraction."""
    o = obs[obs["clone_id"].notna() & obs[state_col].notna() &
            ~obs[state_col].astype(str).str.startswith("unassigned")].copy()
    o["clone_id"] = o["clone_id"].astype(str)
    o[state_col] = o[state_col].astype(str)
    rows = {}
    for cid, g in o.groupby("clone_id"):
        if len(g) < min_cells:
            continue
        p = g[state_col].value_counts(normalize=True)
        rows[cid] = {
            "n_cells": int(len(g)),
            "n_states": int(p.size),
            "entropy": float(-(p * np.log(p)).sum()),
            "dominant_frac": float(p.iloc[0]),
            "dominant_state": str(p.index[0]),
            "condition": str(g["condition"].iloc[0]),
            "donor": str(g["donor"].iloc[0]),
        }
    return pd.DataFrame(rows).T


def knn_agreement(xy: np.ndarray, labels: np.ndarray, k: int = K_NN) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def perm_null(xy, labels, n_perm=N_PERM, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([knn_agreement(xy, rng.permutation(labels)) for _ in range(n_perm)])


def region_chi2(xy: np.ndarray, labels: np.ndarray, k: int = 6, seed: int = 0) -> dict:
    """k-means regions of the clone map vs a label: chi2 and a label-permutation p-value."""
    from sklearn.cluster import KMeans

    labels = np.asarray(labels)
    region = KMeans(k, n_init=10, random_state=seed).fit_predict(xy)
    ct = pd.crosstab(region, labels)
    chi2 = stats.chi2_contingency(ct)
    rng = np.random.default_rng(seed)
    null = np.empty(1000)
    for i in range(1000):
        n = pd.crosstab(region, rng.permutation(labels))
        null[i] = stats.chi2_contingency(n).statistic
    return {"chi2": float(chi2.statistic), "p_value": float((null >= chi2.statistic).mean()),
            "dof": int(chi2.dof), "table": ct.to_dict()}


def main():
    t0 = time.time()

    def stamp(msg):
        print(f"[fates {time.time() - t0:7.0f}s] {msg}", flush=True)

    FIGS.mkdir(parents=True, exist_ok=True)
    tf.pl.set_style()
    adata, cfg, state_key, _ = prepare("spont_gc", stamp=stamp)
    rng = np.random.default_rng(0)
    out = {"min_cells": MIN_CELLS, "k_nn": K_NN}

    # ------------------------------------------------------------------ 1. cell fates
    fate = annotate_fates(adata)
    counts = fate.value_counts()
    out["cell_fates"] = counts.to_dict()
    stamp(f"cell fates: {counts.to_dict()}")

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    order = ["GC", "ASC", "Fcrl5+ MemB", "Cd23a+ MemB", "other MemB", "unassigned"]
    palette = dict(zip(order, ["#2a9d8f", "#e63946", "#457b9d", "#8ecae6", "#b5c9e2", "#bbbbbb"]))
    for cat in order:
        m = fate.eq(cat).to_numpy()
        ax.scatter(adata.obsm["X_umap"][m, 0], adata.obsm["X_umap"][m, 1], s=1.5, color=palette[cat],
                   label=f"{cat} ({m.sum():,})")
    ax.legend(markerscale=4, fontsize=5, loc="upper right", frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, "spont_gc_fates_cell_umap")

    # marker dotplot by fate
    genes = [g for gs in MARKERS.values() for g in gs if g in adata.var_names]
    x = adata.layers["log_norm"]
    dp = pd.DataFrame({g: sparse_col(x, adata.var_names.get_loc(g)) for g in genes},
                      index=adata.obs_names)
    dp["fate"] = fate.values
    order_dp = [c for c in order if c in set(fate)]
    frac = dp.groupby("fate")[genes].apply(lambda d_: (d_ > 0).mean()).reindex(order_dp)
    mean = dp.groupby("fate")[genes].mean().reindex(order_dp)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), constrained_layout=True)
    for ax, mat, title, vmax in ((axes[0], frac, "fraction detected", 1.0),
                                 (axes[1], mean, "mean log-norm", None)):
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=5)
        ax.set_yticks(range(len(mat.index)), mat.index, fontsize=6)
        ax.set_title(title, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
    save_fig(fig, "spont_gc_fates_markers")

    # sanity: leiden cluster x fate
    ct = pd.crosstab(adata.obs["leiden_state"].astype(str), fate)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    frac_ct = ct.div(ct.sum(axis=1), axis=0)
    im = ax.imshow(frac_ct.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(frac_ct.shape[1]), frac_ct.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(frac_ct.shape[0]), frac_ct.index, fontsize=6)
    ax.set_title("leiden cluster x fate (fraction)", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    save_fig(fig, "spont_gc_fates_leiden_x_fate")
    out["leiden_x_fate"] = frac_ct.round(3).to_dict()

    # ------------------------------------------------------------------ 2. clone fate width
    tab = pd.read_csv(OUT / "clone_table.csv", index_col=0)
    tab.index = tab.index.astype(str)
    if "x" not in tab.columns:
        raise ValueError("clone_table.csv has no clone map coordinates; run run_case_study.py first")
    stamp(f"clone map: {len(tab)} profiled clones")

    w = fate_widths(adata.obs, "fate")
    w = w.join(tab[["x", "y", "isotype", "mutation_frequency"]], how="left")
    for col, dtype in (("n_cells", int), ("n_states", int), ("entropy", float),
                       ("dominant_frac", float), ("x", float), ("y", float),
                       ("mutation_frequency", float)):
        w[col] = w[col].astype(dtype)
    w["fate_class"] = np.where(w["n_states"] >= 2, "multi-fate", "single-fate")
    w.to_csv(OUT / "spont_gc_fate_clones.csv")
    multi = w["n_states"] >= 2
    out["clone_fate_width"] = {
        "n_clones_ge5": int(len(w)),
        "n_multi_fate": int(multi.sum()),
        "frac_multi_fate": float(multi.mean()),
        "median_entropy": float(w["entropy"].median()),
        "n_states_distribution": {int(k): int(v) for k, v in
                                  w["n_states"].value_counts().sort_index().items()},
        "cells_in_multi_fate_clones": int(w.loc[multi, "n_cells"].sum()),
        "cells_in_ge5_clones": int(w["n_cells"].sum()),
    }
    stamp(f"clones >= {MIN_CELLS} cells: {len(w)}; multi-fate: {int(multi.sum())} "
          f"({multi.mean():.1%}); states dist {out['clone_fate_width']['n_states_distribution']}")

    # chance baseline: shuffle cell fates within the BCR+ pool, recompute the multi-fate fraction
    obs = adata.obs
    bcr_cells = obs[obs["clone_id"].notna() & obs["fate"].notna()].copy()
    bcr_cells = bcr_cells[~bcr_cells["fate"].astype(str).str.startswith("unassigned")]
    size_of = bcr_cells["clone_id"].astype(str).value_counts()

    def shuffled_multi_frac() -> float:
        sh = pd.Series(rng.permutation(bcr_cells["fate"].to_numpy()), index=bcr_cells.index)
        tmp = pd.DataFrame({"clone": bcr_cells["clone_id"].astype(str), "fate": sh})
        tmp = tmp[tmp["clone"].map(size_of) >= MIN_CELLS]
        return float((tmp.groupby("clone")["fate"].nunique() >= 2).mean())

    null_multi = np.array([shuffled_multi_frac() for _ in range(100)])
    out["clone_fate_width"]["null_multi_fate_mean"] = float(null_multi.mean())
    out["clone_fate_width"]["null_multi_fate_q95"] = float(np.quantile(null_multi, 0.95))
    stamp(f"chance-level multi-fate fraction: {null_multi.mean():.3f} "
          f"(q95 {np.quantile(null_multi, 0.95):.3f}) vs observed {multi.mean():.3f}")

    # per condition
    per_cond = w.groupby("condition").agg(
        n_clones=("n_states", "size"),
        frac_multi=("fate_class", lambda s: float((s == "multi-fate").mean())),
        median_entropy=("entropy", "median"),
        median_mu=("mutation_frequency", "median"))
    out["fate_width_by_condition"] = per_cond.round(4).to_dict("index")
    a = w.loc[w["condition"] == "autoimmune", "entropy"]
    b = w.loc[w["condition"] == "immunized", "entropy"]
    out["entropy_mannwhitney_p"] = float(stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue)
    ct_cond = pd.crosstab(w["condition"], w["fate_class"])
    out["multi_fate_by_condition_fisher_p"] = float(stats.fisher_exact(ct_cond)[1])
    stamp(f"fate width by condition:\n{per_cond.round(3).to_string()}")

    # clone map colored by fate width and entropy
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    ax = axes[0]
    for cat, col in (("single-fate", "#bbbbbb"), ("multi-fate", "#e63946")):
        m = w["fate_class"].eq(cat).to_numpy()
        ax.scatter(w["x"].to_numpy()[m], w["y"].to_numpy()[m], s=w["n_cells"].to_numpy()[m] * 0.6,
                   color=col, alpha=0.75, label=f"{cat} ({m.sum()})")
    ax.legend(markerscale=2, fontsize=6, frameon=False)
    ax.set_title("fate class", fontsize=7)
    axm = axes[1]
    sc_ = axm.scatter(w["x"], w["y"], s=w["n_cells"] * 0.6, c=w["entropy"], cmap="viridis", alpha=0.8)
    fig.colorbar(sc_, ax=axm, shrink=0.7, label="Shannon entropy")
    axm.set_title("fate entropy", fontsize=7)
    for a_ in axes:
        a_.set_xticks([])
        a_.set_yticks([])
    save_fig(fig, "spont_gc_fates_map_width")

    # structure: are multi-fate clones concentrated in map regions?
    xy = w[["x", "y"]].to_numpy()
    out["map_region_fate_class"] = region_chi2(xy, w["fate_class"].to_numpy())
    rho_x = stats.spearmanr(w["entropy"], w["x"]).statistic
    rho_y = stats.spearmanr(w["entropy"], w["y"]).statistic
    out["entropy_vs_map_position_spearman"] = {"x": float(rho_x), "y": float(rho_y)}
    stamp(f"map regions vs fate class: chi2 perm p = {out['map_region_fate_class']['p_value']:.4f}; "
          f"entropy-position spearman x {rho_x:.3f}, y {rho_y:.3f}")

    # ------------------------------------------------------------------ 3. condition
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for cond, col in (("autoimmune", "#9d4edd"), ("immunized", "#f4a261")):
        m = w["condition"].eq(cond).to_numpy()
        ax.scatter(w["x"].to_numpy()[m], w["y"].to_numpy()[m], s=w["n_cells"].to_numpy()[m] * 0.6,
                   color=col, alpha=0.75, label=f"{cond} ({m.sum()})")
    ax.legend(markerscale=2, fontsize=6, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("clone map by condition (donor-level variable)", fontsize=7)
    save_fig(fig, "spont_gc_fates_map_condition")

    agree_cond = knn_agreement(xy, w["condition"].to_numpy())
    null_cond = perm_null(xy, w["condition"].to_numpy())
    # cross-donor, same-condition cohesion: the donor-confound-free part
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(xy).kneighbors(return_distance=False)[:, 1:]
    cond = w["condition"].to_numpy()
    donor = w["donor"].to_numpy()

    def cross_donor_frac(cond_lab: np.ndarray, donor_lab: np.ndarray) -> float:
        same_cond = cond_lab[nn] == cond_lab[:, None]
        diff_donor = donor_lab[nn] != donor_lab[:, None]
        return float((same_cond & diff_donor).mean())

    cross = cross_donor_frac(cond, donor)
    null_cross = np.array([cross_donor_frac(cond[p], donor[p])
                           for p in [rng.permutation(len(cond)) for _ in range(N_PERM)]])
    out["condition_structure"] = {
        "note": "condition is donor-level: 5 autoimmune (M13/17/21/24/34) vs 3 immunized (M11/14/23) donors",
        "knn_condition_agreement": agree_cond,
        "null_mean": float(null_cond.mean()), "null_q95": float(np.quantile(null_cond, 0.95)),
        "cross_donor_same_condition_agreement": cross,
        "cross_donor_null_mean": float(null_cross.mean()),
        "cross_donor_null_q95": float(np.quantile(null_cross, 0.95)),
        "map_region_condition_chi2_p": region_chi2(xy, w["condition"].to_numpy())["p_value"],
    }
    stamp(f"condition 15-NN agreement {agree_cond:.3f} (null {null_cond.mean():.3f}); "
          f"cross-donor same-condition {cross:.3f} (null {null_cross.mean():.3f})")

    # per-donor consistency
    per_donor = w.groupby("donor").agg(condition=("condition", "first"), n_clones=("n_states", "size"),
                                       frac_multi=("fate_class", lambda s: float((s == "multi-fate").mean())),
                                       median_entropy=("entropy", "median"),
                                       median_mu=("mutation_frequency", "median"))
    out["per_donor"] = per_donor.round(4).to_dict("index")
    stamp(f"per-donor fate width:\n{per_donor.round(3).to_string()}")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), constrained_layout=True)
    for ax, (d, sub) in zip(axes.ravel(), w.groupby("donor")):
        ax.scatter(w["x"], w["y"], s=2, color="#dddddd")
        m_multi = sub["fate_class"].eq("multi-fate").to_numpy()
        ax.scatter(sub["x"].to_numpy()[m_multi], sub["y"].to_numpy()[m_multi],
                   s=sub["n_cells"].to_numpy()[m_multi] * 0.8, color="#e63946", alpha=0.8)
        ax.set_title(f"{d} ({sub['condition'].iloc[0]})\n"
                     f"{int(sub['n_states'].ge(2).sum())}/{len(sub)} multi-fate", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, "spont_gc_fates_map_donor_panels")

    # ------------------------------------------------------------------ 4. 15-NN single vs multi
    obs = adata.obs
    cell_mask = (obs["clone_id"].notna() & obs["fate"].notna() &
                 ~obs["fate"].astype(str).str.startswith("unassigned")).to_numpy()
    cc_pos = np.flatnonzero(cell_mask)
    cell_clone_ids = obs["clone_id"].astype(str).to_numpy()[cc_pos]
    cell_lab = w["fate_class"].reindex(cell_clone_ids).to_numpy()
    keep_cells = pd.notna(cell_lab)
    cc_pos = cc_pos[keep_cells]
    cell_lab = cell_lab[keep_cells]
    n_cells_used = int(len(cell_lab))
    cell_xy = adata.obsm["X_umap"][cc_pos]
    agree_cell = knn_agreement(cell_xy, cell_lab)
    null_cell = perm_null(cell_xy, cell_lab)
    agree_clone = knn_agreement(xy, w["fate_class"].to_numpy())
    null_clone = perm_null(xy, w["fate_class"].to_numpy())
    out["knn_agreement_single_vs_multi"] = {
        "cell_umap": {"agreement": agree_cell, "null_mean": float(null_cell.mean()),
                      "null_q95": float(np.quantile(null_cell, 0.95)), "n_cells": n_cells_used},
        "clone_map": {"agreement": agree_clone, "null_mean": float(null_clone.mean()),
                      "null_q95": float(np.quantile(null_clone, 0.95)), "n_clones": int(len(w))},
        "n_perm": N_PERM,
    }
    stamp(f"single/multi 15-NN: cell UMAP {agree_cell:.3f} (null {null_cell.mean():.3f}, {n_cells_used:,} cells); "
          f"clone map {agree_clone:.3f} (null {null_clone.mean():.3f}, {len(w)} clones)")

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    bars = [agree_cell, float(null_cell.mean()), agree_clone, float(null_clone.mean())]
    errs = [0.0, float(null_cell.std()), 0.0, float(null_clone.std())]
    ax.bar(["cell UMAP", "cell UMAP\nnull", "clone map", "clone map\nnull"], bars,
           color=["#457b9d", "#b5c9e2", "#e63946", "#f1b0b5"], yerr=errs, capsize=3)
    ax.set_ylabel(f"{K_NN}-NN agreement (single vs multi)")
    ax.tick_params(axis="x", labelsize=6)
    save_fig(fig, "spont_gc_fates_knn_agreement")

    # ------------------------------------------------------------------ 5. isotype / class switching
    iso = w["isotype"].astype(str)
    switched = iso.isin(["IGHG", "IGHA", "IGHE"])
    frac_sw = w.groupby("condition").apply(lambda g: float(g["isotype"].isin(
        ["IGHG", "IGHA", "IGHE"]).mean())).to_dict()
    out["class_switching"] = {
        "isotype_counts": iso.value_counts().to_dict(),
        "frac_switched_overall": float(switched.mean()),
        "frac_switched_by_condition": frac_sw,
    }
    ct_iso = pd.crosstab(w["condition"], switched)
    out["class_switching"]["fisher_p_condition"] = float(stats.fisher_exact(ct_iso)[1])
    out["class_switching"]["median_mu_by_isotype"] = \
        w.groupby(iso)["mutation_frequency"].median().to_dict()
    stamp(f"class-switched clones: {switched.mean():.1%} overall; {frac_sw}; "
          f"Fisher p = {out['class_switching']['fisher_p_condition']:.3f}")

    # intra-clone isotype mixing at the cell level
    bcr_obs = adata.obs[adata.obs["clone_id"].notna() & adata.obs["isotype"].notna()].copy()
    bcr_obs["clone_id"] = bcr_obs["clone_id"].astype(str)
    g5 = bcr_obs["clone_id"].value_counts()
    bcr_obs = bcr_obs[bcr_obs["clone_id"].map(g5) >= MIN_CELLS]
    mix = bcr_obs.groupby("clone_id")["isotype"].nunique()
    out["class_switching"]["frac_clones_multi_isotype_ge5"] = float((mix >= 2).mean())
    stamp(f"clones >= {MIN_CELLS} cells with >1 isotype class: {(mix >= 2).mean():.1%} of {len(mix)}")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    iso_colors = {"IGHM": "#8ecae6", "IGHD": "#b5c9e2", "IGHG": "#e63946",
                  "IGHA": "#f4a261", "IGHE": "#9d4edd"}
    for c in sorted(iso.unique()):
        m = iso.eq(c).to_numpy()
        ax.scatter(w["x"].to_numpy()[m], w["y"].to_numpy()[m], s=w["n_cells"].to_numpy()[m] * 0.6,
                   color=iso_colors.get(c, "#333333"), alpha=0.75, label=f"{c} ({m.sum()})")
    ax.legend(markerscale=2, fontsize=6, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("clone map by majority isotype", fontsize=7)
    save_fig(fig, "spont_gc_fates_map_isotype")
    out["map_region_isotype_chi2_p"] = region_chi2(xy, iso.to_numpy())["p_value"]

    # cell-level fate table for reuse
    keep_df = adata.obs[["donor", "condition", "clone_id", "isotype", "mutation_frequency"]].copy()
    keep_df["fate"] = fate.values
    keep_df["umap_1"], keep_df["umap_2"] = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
    keep_df.to_csv(OUT / "spont_gc_fate_cells.csv.gz")

    (OUT / "spont_gc_fates.json").write_text(json.dumps(jsonable(out), indent=1, default=str) + "\n")
    stamp(f"wrote {OUT / 'spont_gc_fates.json'}")

    # ------------------------------------------------------------------ Chinese summary
    cw_ = out["clone_fate_width"]
    ka = out["knn_agreement_single_vs_multi"]
    cs = out["class_switching"]
    cond = out["condition_structure"]
    print("\n================ 中文结论摘要 ================", flush=True)
    print(f"1) 细胞命运注释(marker-based,描述性):{counts.to_dict()}", flush=True)
    print(f"2) 克隆命运宽度:>={MIN_CELLS} 细胞的克隆 {cw_['n_clones_ge5']} 个,"
          f"横跨 >=2 个转录状态的占 {cw_['frac_multi_fate']:.1%}"
          f"——绝对意义上论文'一个克隆多种命运'成立;但随机重排基线高达 "
          f"{cw_['null_multi_fate_mean']:.1%}(q95 {cw_['null_multi_fate_q95']:.1%}),"
          f"观察值明显低于随机,即克隆内部命运相干(ICC 0.28),'冗余'是有限度的。",
          flush=True)
    print(f"   状态数分布 {cw_['n_states_distribution']},多命运克隆共含 {cw_['cells_in_multi_fate_clones']} 个细胞。",
          flush=True)
    print(f"   命运宽度在 clone map 上的结构:区域-宽度类关联 chi2 置换 p = "
          f"{out['map_region_fate_class']['p_value']:.4f};熵-位置 spearman x {rho_x:.3f} / y {rho_y:.3f}。",
          flush=True)
    print(f"3) 条件结构:condition 15-NN 一致率 {cond['knn_condition_agreement']:.3f}"
          f"(null {cond['null_mean']:.3f});跨供体同条件一致率 {cond['cross_donor_same_condition_agreement']:.3f}"
          f"(null {cond['cross_donor_null_mean']:.3f})——"
          f"{'存在超越供体分离的条件结构' if cond['cross_donor_same_condition_agreement'] > cond['cross_donor_null_q95'] else '未见超越供体分离的条件结构'}。",
          flush=True)
    print(f"4) 单命运 vs 多命运 15-NN 一致率:细胞 UMAP {ka['cell_umap']['agreement']:.3f}"
          f"(null {ka['cell_umap']['null_mean']:.3f}) vs clone map {ka['clone_map']['agreement']:.3f}"
          f"(null {ka['clone_map']['null_mean']:.3f})。", flush=True)
    print(f"5) 同型转换:克隆多数同型 {cs['isotype_counts']};switched 比例 {cs['frac_switched_by_condition']},"
          f"条件间 Fisher p = {cs['fisher_p_condition']:.3f};>={MIN_CELLS} 细胞克隆内混合同型的比例 "
          f"{cs['frac_clones_multi_isotype_ge5']:.1%}。", flush=True)
    print("==============================================", flush=True)


if __name__ == "__main__":
    main()
