#!/usr/bin/env python
"""GSE171703 covid_bcell: is antigen specificity visible on the clone map but not the cell UMAP?

Dugan et al. 2021 measured oligo-tagged SARS-CoV-2 baits (spike / RBD / NP / ORF8).
antigen_call is a heuristic dominant-probe label (raw max >=2 UMIs). Threadfin never sees that label when it builds the
expression-based clone map. This script asks:

1. per-clone antigen repertoire: dominant antigen + entropy / purity of the antigen_call
   distribution (clones with >= 5 cells only);
2. side-by-side: cell UMAP coloured by antigen_call vs clone map coloured by dominant antigen,
   plus the clone map coloured by antigen purity;
3. quantified with 15-NN label agreement + 100 permutation nulls: antigen_call on the cell UMAP,
   the clone's dominant antigen on the clone map, and a control inside memory-like (non-plasma)
   cells only, to rule out "antigen structure = plasma-cell enrichment";
4. HCoV cross-reactivity: the 229E/HKU1 probes were measured in the three acute hashtag libraries
   (R125/R3/R6) only; clones positive there are highlighted on the clone map and tested for
   proximity to SARS-CoV-2-specific clones;
5. acute vs convalescent clones on the clone map (donor-level comparison by construction).

Outputs: results/covid_bcell/figures/antigen_*.png/.pdf and covid_bcell_antigen.json.
Run in the threadfin_v4 environment from case_studies/ (after run_case_study.py covid_bcell).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from storyline_existing import load_existing, neighbor_indices, ORIGINAL  # noqa: E402

RES = HERE / "results" / "covid_bcell"
FIGS = RES / "figures"

ANTIGENS = ["spike", "rbd", "np", "orf8", "none"]
MIN_CELLS = 5
K = 15
N_PERM = 100
PALETTE = {"spike": "#E69F00", "rbd": "#56B4E9", "np": "#009E73",
           "orf8": "#CC79A7", "none": "#B0B0B0"}
HCOV_PROBES = ["probe_spike_229e", "probe_spike_hku1", "probe_np_229e",
               "probe_np_hku1", "probe_rbd_229e", "probe_rbd_hku1"]
PLASMA_GENES = ["PRDM1", "XBP1", "JCHAIN", "MZB1", "SDC1", "TNFRSF17", "IRF4", "DERL3"]


def stamp(msg):
    print(f"[covid_bcell_antigen] {msg}", flush=True)


def knn_agreement(xy: np.ndarray, labels: np.ndarray, k: int = K) -> float:
    idx = neighbor_indices(xy, k)
    return float((labels[idx] == labels[:, None]).mean())


def null_agreement(xy: np.ndarray, labels: np.ndarray, k: int, n_perm: int, rng) -> np.ndarray:
    idx = neighbor_indices(xy, k)
    out = []
    for _ in range(n_perm):
        lab = rng.permutation(labels)
        out.append(float((lab[idx] == lab[:, None]).mean()))
    return np.asarray(out)


def entropy_purity(fracs: np.ndarray) -> tuple[float, float]:
    p = fracs[fracs > 0]
    h = float(-(p * np.log(p)).sum())
    return h, 1.0 - h / np.log(len(ANTIGENS))


def scatter_labels(ax, xy, labels, palette, s=6, alpha=0.7):
    for lev in [l for l in palette if l in set(labels)]:
        m = labels == lev
        sm = s[m] if isinstance(s, (np.ndarray, pd.Series)) else s
        ax.scatter(xy[m, 0], xy[m, 1], s=sm, c=palette[lev], alpha=alpha, linewidths=0, label=lev)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = {"params": {"min_cells": MIN_CELLS, "k": K, "n_perm": N_PERM,
                      "antigen_levels": ANTIGENS},
           "measurement_notes": {
               "antigen_call": "Heuristic dominant raw SARS2 probe, max >=2 UMIs; no background/control normalization. A probe measurement, not a validated specificity call.",
               "hcov_positive": "Raw probe threshold only; almost all measured cells pass. Not evidence of biological cross-reactivity.",
               "plasma_split": "Below/above global median plasma marker score; low score is not a validated MBC annotation.",
               "coordinates": "Frozen cell UMAP and clone map; no pipeline or Dandelion rerun."}}

    tab_file = RES / "clone_table.csv"
    assert tab_file.exists(), f"{tab_file} missing: run run_case_study.py covid_bcell first"
    tab = pd.read_csv(tab_file, dtype={"clone_id": str}).set_index("clone_id")
    stamp(f"clone table: {len(tab)} profiled clones")

    # ---- prepared adata: antigen_call, probes, X_umap -------------------------------
    adata = load_existing("covid_bcell", stamp=stamp)
    obs = adata.obs
    obs["clone_id"] = obs["clone_id"].astype(str)
    stamp(f"prepared adata: {adata.n_obs} cells; states via leiden on X_threadfin")

    analyzed = tab[(tab["n_cells"] >= MIN_CELLS) & tab["x"].notna() & tab["y"].notna()].copy()
    stamp(f"analysed clones (>= {MIN_CELLS} cells, laid out): {len(analyzed)} "
          f"covering {int(analyzed['n_cells'].sum())} cells")

    # ---- 1. per-clone antigen repertoire --------------------------------------------
    cells = obs[obs["clone_id"].isin(analyzed.index)].copy()
    comp = pd.crosstab(cells["clone_id"], cells["antigen_call"]).reindex(columns=ANTIGENS, fill_value=0)
    fracs = comp.div(comp.sum(axis=1), axis=0).fillna(0)
    hp = fracs.apply(lambda r: entropy_purity(r.to_numpy()), axis=1, result_type="expand")
    hp.columns = ["entropy", "purity"]
    analyzed = analyzed.join(hp)
    analyzed["dominant"] = fracs.idxmax(axis=1)
    analyzed["dominant_frac"] = fracs.max(axis=1)
    analyzed.to_csv(RES / "antigen_clone_table.csv")
    out["clone_repertoire"] = {
        "n_clones": int(len(analyzed)),
        "dominant_counts": analyzed["dominant"].value_counts().to_dict(),
        "mean_purity": float(analyzed["purity"].mean()),
        "mean_entropy": float(analyzed["entropy"].mean()),
        "dominant_frac_median": float(analyzed["dominant_frac"].median()),
    }
    stamp(f"dominant antigen counts: {out['clone_repertoire']['dominant_counts']}; "
          f"mean purity {out['clone_repertoire']['mean_purity']:.3f}")

    # ---- 2. figures: UMAP vs clone map; purity map ----------------------------------
    import threadfin as tf
    tf.pl.set_style()

    cell_xy = adata.obsm["X_umap"]
    cell_lab = obs["antigen_call"].astype(str).to_numpy()
    order = rng.permutation(len(cell_lab))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    scatter_labels(axes[0], cell_xy[order], cell_lab[order], PALETTE, s=3, alpha=0.45)
    axes[0].set_title("cell UMAP (X_threadfin) — antigen_call")
    cmap_xy = analyzed[["x", "y"]].to_numpy()
    cmap_lab = analyzed["dominant"].to_numpy()
    sizes = 4 + 26 * (analyzed["n_cells"] / analyzed["n_cells"].max())
    scatter_labels(axes[1], cmap_xy, cmap_lab, PALETTE, s=sizes, alpha=0.85)
    axes[1].set_title(f"clone map — dominant antigen (n={len(analyzed)} clones ≥ {MIN_CELLS} cells)")
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, markerscale=2.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"antigen_umap_vs_clonemap.{ext}", dpi=200)
    plt.close(fig)
    stamp("figure: antigen_umap_vs_clonemap")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    sc = axes[0].scatter(cmap_xy[:, 0], cmap_xy[:, 1], c=analyzed["purity"], cmap="viridis",
                         s=sizes, alpha=0.85, linewidths=0, vmin=0, vmax=1)
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].set_title("clone map — antigen purity (1 − H/log 5)")
    fig.colorbar(sc, ax=axes[0], shrink=0.7, label="purity")
    lev_order = [l for l in ANTIGENS if l in set(analyzed["dominant"])]
    bp = [analyzed.loc[analyzed["dominant"] == l, "purity"] for l in lev_order]
    axes[1].boxplot(bp, tick_labels=lev_order, showfliers=False,
                    medianprops=dict(color="black"))
    for i, l in enumerate(lev_order):
        axes[1].scatter(np.full(len(bp[i]), i + 1) + rng.uniform(-0.12, 0.12, len(bp[i])),
                        bp[i], s=6, c=PALETTE[l], alpha=0.5, linewidths=0)
    axes[1].set_ylabel("antigen purity"), axes[1].set_ylim(0, 1.02)
    axes[1].set_title("purity by dominant antigen")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"antigen_clone_map_purity.{ext}", dpi=200)
    plt.close(fig)
    stamp("figure: antigen_clone_map_purity")

    # ---- 3. 15-NN agreement: cell UMAP vs clone map, with nulls ---------------------
    m = obs["clone_id"].isin(analyzed.index) & obs["antigen_call"].notna()
    xy_c = cell_xy[m]
    lab_call = obs.loc[m, "antigen_call"].astype(str).to_numpy()
    agree_cell = knn_agreement(xy_c, lab_call)
    null_cell = null_agreement(xy_c, lab_call, K, N_PERM, rng)

    xy_m = analyzed[["x", "y"]].to_numpy()
    lab_dom = analyzed["dominant"].to_numpy()
    agree_clone = knn_agreement(xy_m, lab_dom)
    null_clone = null_agreement(xy_m, lab_dom, K, N_PERM, rng)

    # same inherited label, but on the cell UMAP (isolates label smoothing from space)
    inherit = analyzed["dominant"].reindex(obs["clone_id"]).astype(str).to_numpy()
    lab_inh = inherit[m]
    agree_inh = knn_agreement(xy_c, lab_inh)
    null_inh = null_agreement(xy_c, lab_inh, K, N_PERM, rng)

    # 4-class sensitivity: drop 'none' on both layers
    keep4 = lab_call != "none"
    agree_cell4 = knn_agreement(xy_c[keep4], lab_call[keep4])
    null_cell4 = null_agreement(xy_c[keep4], lab_call[keep4], K, N_PERM, rng)
    keep4m = lab_dom != "none"
    agree_clone4 = knn_agreement(xy_m[keep4m], lab_dom[keep4m])
    null_clone4 = null_agreement(xy_m[keep4m], lab_dom[keep4m], K, N_PERM, rng)

    # MBC control: plasma module score, non-plasma half only
    genes = [g for g in PLASMA_GENES if g in adata.var_names]
    x = adata[:, genes].layers["log_norm"]
    x = np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)
    pscore = pd.Series(np.asarray(x.mean(axis=1)).ravel(), index=obs.index)
    med = float(pscore.median())
    prior_json = json.loads((ORIGINAL / "covid_bcell/covid_bcell_antigen.json").read_text())
    prior_med = prior_json["visibility_knn15"]["plasma_score_median"]
    assert np.isclose(med, prior_med, rtol=1e-5), (med, prior_med)
    out["frozen_input_validation"] = {"plasma_score_median_matches_original": True}
    export = obs[["clone_id", "donor", "sample", "disease_state", "leiden_state", "antigen_call"]].copy()
    export["plasma_score"] = pscore.values
    export["umap_1"], export["umap_2"] = cell_xy[:, 0], cell_xy[:, 1]
    export.to_csv(RES / "covid_antigen_cells.csv.gz")
    mbc = m & (pscore < med)
    pc_hi = m & (pscore >= med)

    def cell_layer(mask):
        xy_sub = cell_xy[mask.to_numpy()]
        lab_sub = obs.loc[mask, "antigen_call"].astype(str).to_numpy()
        return knn_agreement(xy_sub, lab_sub), null_agreement(xy_sub, lab_sub, K, N_PERM, rng)

    agree_mbc, null_mbc = cell_layer(mbc)
    agree_pc, null_pc = cell_layer(pc_hi)

    def pack(agree, null, n):
        return {"agreement": float(agree), "null_mean": float(null.mean()),
                "null_std": float(null.std()), "z": float((agree - null.mean()) / null.std()),
                "p_perm": float((np.sum(null >= agree) + 1) / (len(null) + 1)), "n": int(n)}

    out["visibility_knn15"] = {
        "cell_umap_antigen_call": pack(agree_cell, null_cell, m.sum()),
        "cell_umap_inherited_dominant": pack(agree_inh, null_inh, m.sum()),
        "clone_map_dominant": pack(agree_clone, null_clone, len(analyzed)),
        "cell_umap_antigen_call_excl_none": pack(agree_cell4, null_cell4, keep4.sum()),
        "clone_map_dominant_excl_none": pack(agree_clone4, null_clone4, keep4m.sum()),
        "mbc_like_cell_umap": pack(agree_mbc, null_mbc, int(mbc.sum())),
        "plasma_high_cell_umap": pack(agree_pc, null_pc, int(pc_hi.sum())),
        "plasma_score_genes": genes, "plasma_score_median": med,
    }
    v = out["visibility_knn15"]
    stamp(f"15-NN agreement — cell UMAP (measured call): {agree_cell:.3f} vs null "
          f"{null_cell.mean():.3f} (n={m.sum()}); clone map (dominant): {agree_clone:.3f} vs null "
          f"{null_clone.mean():.3f} (n={len(analyzed)}); MBC-like cells: {agree_mbc:.3f} vs null "
          f"{null_mbc.mean():.3f} (n={int(mbc.sum())})")

    # ---- 4. HCoV cross-reactivity ----------------------------------------------------
    hcov = obs[HCOV_PROBES].apply(pd.to_numeric, errors="coerce")
    measured = obs.index[hcov.notna().any(axis=1)]
    stamp(f"HCoV 229E/HKU1 probes measured in {len(measured)} cells "
          f"({obs.loc[measured, 'sample'].nunique()} libraries)")
    pos = hcov.fillna(0).max(axis=1) >= 2
    cl = cells.loc[cells.index.isin(measured)].copy()
    cl["hcov_pos"] = pos.reindex(cl.index).fillna(False).astype(bool)
    g = cl.groupby("clone_id")["hcov_pos"]
    hcov_tab = pd.DataFrame({"n_measured": g.size(), "n_pos": g.sum(),
                             "frac_pos": g.mean()}).join(analyzed[["x", "y", "dominant"]], how="inner")
    hcov_pos_clones = hcov_tab[(hcov_tab["n_pos"] >= 2) & (hcov_tab["frac_pos"] >= 0.5)]
    sars2_clones = set(analyzed.index[analyzed["dominant"].isin(["spike", "rbd", "np", "orf8"])])
    out["hcov"] = {"n_cells_measured": int(len(measured)),
                   "n_cells_positive": int(pos.reindex(measured).sum()),
                   "n_clones_measured": int(len(hcov_tab)),
                   "n_clones_positive": int(len(hcov_pos_clones)),
                   "positive_clones": hcov_pos_clones.index.tolist()[:50]}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    axes[0].scatter(cmap_xy[:, 0], cmap_xy[:, 1], s=4, c="#D9D9D9", linewidths=0)
    if len(hcov_pos_clones):
        axes[0].scatter(hcov_pos_clones["x"], hcov_pos_clones["y"], marker="*", s=140,
                        facecolor="#D62728", edgecolor="black", linewidths=0.5, zorder=5,
                        label=f"Raw HCoV-probe positive (n={len(hcov_pos_clones)})")
        axes[0].legend(frameon=False, loc="best")
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].set_title("clone map — raw HCoV-probe positive clones")
    sub = analyzed[analyzed.index.isin(hcov_tab.index)]
    scatter_labels(axes[1], sub[["x", "y"]].to_numpy(), sub["dominant"].to_numpy(),
                   PALETTE, s=14, alpha=0.8)
    if len(hcov_pos_clones):
        axes[1].scatter(hcov_pos_clones["x"], hcov_pos_clones["y"], marker="*", s=160,
                        facecolor="none", edgecolor="black", linewidths=1.0, zorder=5)
    axes[1].set_title(f"HCoV-measured donors only (n={len(sub)} clones)")
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, markerscale=2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"antigen_hcov_clonemap.{ext}", dpi=200)
    plt.close(fig)
    stamp(f"figure: antigen_hcov_clonemap; HCoV+ clones: {list(hcov_pos_clones.index)}")

    # distance from HCoV+ clones to nearest SARS-CoV-2-specific clone, vs label-permutation null
    if len(hcov_pos_clones) >= 3 and len(sars2_clones) >= 10:
        pool = hcov_tab.index  # clones with >= 1 measured cell
        sxy = analyzed.loc[analyzed.index.isin(sars2_clones), ["x", "y"]].to_numpy()

        target_ids = analyzed.index[analyzed.index.isin(sars2_clones)].to_numpy()

        def mean_min_dist(member_ids):
            members = analyzed.loc[member_ids, ["x", "y"]].to_numpy()
            d = np.sqrt(((members[:, None, :] - sxy[None, :, :]) ** 2).sum(-1))
            d[np.asarray(member_ids)[:, None] == target_ids[None, :]] = np.inf
            return float(d.min(axis=1).mean())

        observed = mean_min_dist(hcov_pos_clones.index)
        n_pos = len(hcov_pos_clones)
        nulls = []
        pool_xy = hcov_tab[["x", "y"]].to_numpy()
        for _ in range(N_PERM):
            sel = rng.choice(len(pool_xy), size=n_pos, replace=False)
            nulls.append(mean_min_dist(pool.to_numpy()[sel]))
        nulls = np.asarray(nulls)
        out["hcov"]["proximity"] = {
            "mean_min_dist_to_sars2_clone": observed,
            "null_mean": float(nulls.mean()), "null_std": float(nulls.std()),
            "z": float((observed - nulls.mean()) / nulls.std()),
            "p_perm_smaller": float((np.sum(nulls <= observed) + 1) / (len(nulls) + 1)),
            "n_pool_clones": int(len(pool)), "n_sars2_clones": int(len(sars2_clones)),
            "self_matches_excluded": True, "interpretation": "Exploratory geometry of raw probe calls; specificity not validated."}
        stamp(f"HCoV proximity: {observed:.2f} vs null {nulls.mean():.2f} "
              f"(z={out['hcov']['proximity']['z']:.2f})")

    # ---- 5. acute vs convalescent (donor-level) --------------------------------------
    state = analyzed["donor"].map(obs.groupby("donor")["disease_state"].first())
    analyzed["disease_state"] = state
    lab_state = state.astype(str).to_numpy()
    agree_state = knn_agreement(xy_m, lab_state)
    null_state = null_agreement(xy_m, lab_state, K, N_PERM, rng)
    out["disease_state_donor_level"] = {
        "acute_clones": int((lab_state == "Acute").sum()),
        "convalescent_clones": int((lab_state == "Convalescent").sum()),
        "clone_map_knn15": pack(agree_state, null_state, len(analyzed)),
        "note": "donor-level label: agreement conflates donor identity with disease state",
    }
    stamp(f"disease state: {out['disease_state_donor_level']['acute_clones']} acute vs "
          f"{out['disease_state_donor_level']['convalescent_clones']} convalescent clones; "
          f"15-NN {agree_state:.3f} vs null {null_state.mean():.3f}")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    for lev, c in (("Acute", "#D62728"), ("Convalescent", "#1F77B4")):
        mm = lab_state == lev
        ax.scatter(xy_m[mm, 0], xy_m[mm, 1], s=5 if lev == "Convalescent" else 22,
                   c=c, alpha=0.65 if lev == "Convalescent" else 0.95,
                   linewidths=0, label=f"{lev} (n={mm.sum()})")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("clone map — donor disease state (donor-level comparison)")
    ax.legend(frameon=False, markerscale=2.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"antigen_disease_state_clonemap.{ext}", dpi=200)
    plt.close(fig)
    stamp("figure: antigen_disease_state_clonemap")

    (RES / "covid_bcell_antigen.json").write_text(
        json.dumps(jsonable(out), indent=1, default=str) + "\n")
    stamp(f"wrote {RES / 'covid_bcell_antigen.json'}")

    # ---- Chinese summary -------------------------------------------------------------
    print("\n================ 中文结论摘要 ================", flush=True)
    c = out["clone_repertoire"]
    print(f"分析克隆(≥{MIN_CELLS} 细胞): {c['n_clones']} 个, dominant antigen 组成 "
          f"{c['dominant_counts']}, 平均纯度 {c['mean_purity']:.3f}", flush=True)
    v = out["visibility_knn15"]
    print("15-NN 一致率 (观察值 vs permutation null 均值):", flush=True)
    print(f"  细胞 UMAP (antigen_call, n={v['cell_umap_antigen_call']['n']}): "
          f"{v['cell_umap_antigen_call']['agreement']:.3f} vs {v['cell_umap_antigen_call']['null_mean']:.3f}", flush=True)
    print(f"  克隆 map (dominant antigen, n={v['clone_map_dominant']['n']}): "
          f"{v['clone_map_dominant']['agreement']:.3f} vs {v['clone_map_dominant']['null_mean']:.3f}", flush=True)
    print(f"  MBC 样细胞 UMAP 对照 (n={v['mbc_like_cell_umap']['n']}): "
          f"{v['mbc_like_cell_umap']['agreement']:.3f} vs {v['mbc_like_cell_umap']['null_mean']:.3f}", flush=True)
    h = out["hcov"]
    print(f"HCoV 229E/HKU1: {h['n_cells_measured']} 细胞有 probe 测量, {h['n_clones_positive']} 个阳性克隆", flush=True)
    d = out["disease_state_donor_level"]
    print(f"急性 vs 恢复期克隆: {d['acute_clones']} vs {d['convalescent_clones']}; "
          f"clone map 15-NN {d['clone_map_knn15']['agreement']:.3f} vs null "
          f"{d['clone_map_knn15']['null_mean']:.3f} (供体间比较,与 donor 身份混杂)", flush=True)
    print("==============================================", flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 200 else f"<array {obj.shape}>"
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    main()
