#!/usr/bin/env python
"""Subsplit the frozen S2 P3 rows and write a new figure and statistics JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper" / "figure_plan"))
import figstyle as fs

OUT = HERE / "results" / "spont_gc"
GROUPS = ["P3a", "P3b"]
COLORS = {"P3a": "#1baf7a", "P3b": "#0b7a5c"}
LABEL_M = ["Fcrl5+ MemB", "CD23+ MemB", "other MemB"]


def indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError(f"Nonunique clone index: {path}")
    return frame


def mw_result(frame: pd.DataFrame, column: str) -> dict:
    arrays = [frame.loc[frame.subgroup.eq(g), column].dropna().to_numpy() for g in GROUPS]
    result = stats.mannwhitneyu(*arrays, alternative="two-sided", method="auto")
    return {"method": "Mann-Whitney U", "alternative": "two-sided", "method_parameter": "auto",
            "group_order": GROUPS, "n": [int(len(a)) for a in arrays],
            "U": float(result.statistic), "pvalue": float(result.pvalue)}


def main() -> None:
    print("[A1] Read frozen tables and verify P3 coverage", flush=True)
    tab = indexed_csv(OUT / "clone_table.csv")
    scores = indexed_csv(OUT / "clone_gene_scores.csv")
    fate_clones = indexed_csv(OUT / "spont_gc_fate_clones.csv")
    cells = pd.read_csv(OUT / "spont_gc_fate_cells.csv.gz", index_col=0)
    cells["clone_id"] = cells["clone_id"].astype("string")
    p3 = tab.loc[tab.clone_programme.eq("P3")].copy()
    if len(p3) != 241:
        raise ValueError(f"Expected 241 P3 rows, found {len(p3)}")
    gene_matrix = scores.loc[p3.index].to_numpy(dtype=float)
    if not np.isfinite(gene_matrix).all() or not np.isfinite(p3[["x", "y"]]).all().all():
        raise ValueError("Nonfinite feature or coordinate values")
    if not p3.index.isin(cells.clone_id).all():
        raise ValueError("Some P3 clones have no cell records; label-M denominator needs review")

    print("[A2] Standardize P3 gene-score rows; fit two P3-only KMeans models", flush=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(gene_matrix)
    gene_labels = KMeans(n_clusters=2, n_init=50, random_state=0).fit_predict(z)
    xy_labels = KMeans(n_clusters=2, n_init=50, random_state=0).fit_predict(p3[["x", "y"]])
    p3["gene_label"] = gene_labels
    low_gene = p3.groupby("gene_label")["y"].median().idxmin()
    p3["subgroup"] = np.where(gene_labels == low_gene, "P3a", "P3b")
    p3["coordinate_label"] = xy_labels
    low_xy = p3.groupby("coordinate_label")["y"].median().idxmin()
    p3["coordinate_subgroup"] = np.where(xy_labels == low_xy, "P3a", "P3b")
    silhouette = float(silhouette_score(z, gene_labels))
    ari = float(adjusted_rand_score(gene_labels, xy_labels))

    print("[A3] Compute subgroup statistics using explicit per-metric denominators", flush=True)
    label_m = cells["fate"].isin(LABEL_M).groupby(cells["clone_id"]).any()
    p3["label_m"] = label_m.reindex(p3.index).to_numpy(dtype=bool)
    joined = p3[["subgroup"]].join(fate_clones, how="left", validate="one_to_one")
    joined["fate_clone_present"] = joined.index.isin(fate_clones.index)
    groups = {}
    quantiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    p2_centroid = tab.loc[tab.clone_programme.eq("P2"), ["x", "y"]].mean().to_numpy()
    for group in GROUPS:
        sub = p3.loc[p3.subgroup.eq(group)]
        fate = joined.loc[joined.subgroup.eq(group) & joined.fate_clone_present]
        centroid = sub[["x", "y"]].mean().to_numpy()
        groups[group] = {
            "n_clones": int(len(sub)),
            "n_clones_with_fate_cells": int(sub.index.isin(cells.clone_id).sum()),
            "n_fate_clones_matched": int(len(fate)),
            "n_fate_clones_unmatched": int(len(sub) - len(fate)),
            "median_n_cells": float(fate.n_cells.median()),
            "median_n_cells_source": "spont_gc_fate_clones.csv matched rows",
            "median_n_cells_all_clone_table_rows": float(sub.n_cells.median()),
            "median_entropy": float(fate.entropy.median()),
            "n_entropy": int(fate.entropy.notna().sum()),
            "label_m_positive_clones": int(sub.label_m.sum()),
            "label_m_negative_clones": int((~sub.label_m).sum()),
            "label_m_denominator": int(len(sub)),
            "label_m_proportion": float(sub.label_m.mean()),
            "dominant_state_counts": {str(k): int(v) for k, v in fate.dominant_state.value_counts().sort_index().items()},
            "dominant_state_proportions": {str(k): float(v) for k, v in fate.dominant_state.value_counts(normalize=True).sort_index().items()},
            "n_dominant_state": int(fate.dominant_state.notna().sum()),
            "median_mutation_frequency": float(fate.mutation_frequency.median()),
            "n_mutation_frequency": int(fate.mutation_frequency.notna().sum()),
            "centroid_xy": [float(v) for v in centroid],
            "distance_to_p2_centroid": float(np.linalg.norm(centroid - p2_centroid)),
            "y_quantiles": {f"q{q:g}": float(sub.y.quantile(q)) for q in quantiles},
        }
    fisher_table = np.array([[groups[g]["label_m_positive_clones"], groups[g]["label_m_negative_clones"]]
                             for g in GROUPS])
    fisher = stats.fisher_exact(fisher_table, alternative="two-sided")
    dominant_table = pd.crosstab(joined.subgroup, joined.dominant_state).reindex(GROUPS, fill_value=0)
    chi2 = stats.chi2_contingency(dominant_table.to_numpy(), correction=False)
    comparison = pd.crosstab(p3.subgroup, p3.coordinate_subgroup).reindex(index=GROUPS, columns=GROUPS, fill_value=0)
    criteria = {"silhouette_at_least_0_1": silhouette >= 0.1,
                "label_m_fisher_p_below_0_05": float(fisher.pvalue) < 0.05,
                "each_subgroup_at_least_30_clones": all(groups[g]["n_clones"] >= 30 for g in GROUPS)}
    result = {
        "dataset": "S2", "input_directory": str(OUT),
        "input_rows": {"clone_table": int(len(tab)), "gene_scores": int(len(scores)),
                       "fate_clones": int(len(fate_clones)), "fate_cells": int(len(cells))},
        "n_p3_clones": int(len(p3)), "n_gene_score_features": int(gene_matrix.shape[1]),
        "gene_score_features": list(scores.columns),
        "standardization": "StandardScaler across the 241 P3 rows, mean=0, population variance=1",
        "kmeans_parameters": {"n_clusters": 2, "n_init": 50, "random_state": 0},
        "coordinate_standardization": "none; raw x/y",
        "silhouette_score": silhouette, "silhouette_metric": "euclidean on standardized gene scores",
        "adjusted_rand_index": ari, "partitions_equivalent_by_ari_0_9": ari >= 0.9,
        "partition_used": "gene_scores", "naming_rule": "P3a has lower median y; P3b is the other gene-score subgroup",
        "partition_comparison": {"rows_gene_scores": GROUPS, "columns_coordinates": GROUPS, "counts": comparison.to_numpy().tolist()},
        "label_m_definition": {"cell_fates": LABEL_M, "positive": "at least one matching cell",
                               "denominator": "all clone_table P3 rows in each subgroup"},
        "fate_clone_join": {"type": "left by clone_id", "matched": int(joined.fate_clone_present.sum()),
                            "unmatched": int((~joined.fate_clone_present).sum()),
                            "handling": "no imputation; entropy, dominant_state, mutation_frequency and primary median_n_cells use matched fate_clones rows"},
        "groups": groups,
        "tests": {
            "entropy_mannwhitney": mw_result(joined, "entropy"),
            "label_m_fisher": {"alternative": "two-sided", "rows": GROUPS, "columns": ["label-M", "not label-M"],
                               "contingency_table": fisher_table.tolist(), "odds_ratio": float(fisher.statistic), "pvalue": float(fisher.pvalue)},
            "dominant_state_chi_square": {"method": "Pearson chi-square; correction=False", "rows": GROUPS,
                                          "columns": [str(v) for v in dominant_table.columns],
                                          "contingency_table": dominant_table.to_numpy().tolist(),
                                          "chi2": float(chi2.statistic), "pvalue": float(chi2.pvalue), "dof": int(chi2.dof),
                                          "expected_frequencies": chi2.expected_freq.tolist(),
                                          "expected_below_5_count": int((chi2.expected_freq < 5).sum()),
                                          "expected_minimum": float(chi2.expected_freq.min())},
            "mutation_frequency_mannwhitney": mw_result(joined, "mutation_frequency"),
        },
        "geometry": {"n_p2_clones": int(tab.clone_programme.eq("P2").sum()),
                     "p2_centroid_xy": [float(v) for v in p2_centroid],
                     "p3b_closer_to_p2_than_p3a": groups["P3b"]["distance_to_p2_centroid"] < groups["P3a"]["distance_to_p2_centroid"]},
        "split_criteria": criteria, "split_accepted": all(criteria.values()),
        "failed_criteria": [key for key, value in criteria.items() if not value],
        "assignments_by_clone_id": p3.subgroup.to_dict(),
        "coordinate_assignments_by_clone_id": p3.coordinate_subgroup.to_dict(),
        "outputs": {"statistics": str(OUT / "p3_subsplit.json"),
                    "png": str(OUT / "figures" / "p3_subsplit.png"),
                    "pdf": str(OUT / "figures" / "p3_subsplit.pdf"),
                    "figure_width_mm": 210, "figure_height_mm": 112, "dpi": 1200},
    }
    with (OUT / "p3_subsplit.json").open("w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if "assignments_by_clone_id" not in k},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)

    print("[A4] Draw two-panel subgroup figure with shared figstyle; export at 1200 dpi", flush=True)
    fig = fs.new_page("Fig. cand.", "Dataset S2 spont_gc (GSE203132)", height=112)
    ax = fs.panel(fig, 4, 10, 78, 62, letter="a")
    ax.set_title("S2 clones — subgroup", pad=4)
    bg = tab.loc[~tab.clone_programme.eq("P3")]
    ax.scatter(bg.x, bg.y, s=np.clip(bg.n_cells * 0.35, 0.6, None), color="#e7e6e0", linewidths=0, alpha=0.8)
    for group in GROUPS:
        sub = p3.loc[p3.subgroup.eq(group)]
        ax.scatter(sub.x, sub.y, s=np.clip(sub.n_cells * 0.65, 2.0, None), color=COLORS[group], linewidths=0, alpha=0.9)
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    handles = [plt.Line2D([], [], color=COLORS[g], marker="o", ls="", markersize=3.5,
                          label=f"{g} (n={groups[g]['n_clones']})") for g in GROUPS]
    handles.append(plt.Line2D([], [], color="#e7e6e0", marker="o", ls="", markersize=3.5, label=f"Other (n={len(bg)})"))
    ax.legend(handles=handles, title="Subgroup", title_fontsize=5, fontsize=4.8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), columnspacing=1, handletextpad=0.3)

    ax = fs.panel(fig, 106, 10, 72, 62, letter="b")
    ax.set_title("S2 clones — subgroup summary", pad=4)
    xx = np.arange(2)
    width = 0.30
    for i, group in enumerate(GROUPS):
        g = groups[group]
        values = [g["label_m_proportion"], g["median_entropy"]]
        positions = xx + (i - 0.5) * width
        ax.bar(positions, values, width=width, color=COLORS[group], linewidth=0)
        labels = [f"{values[0]:.1%}\nn={g['label_m_denominator']}", f"{values[1]:.3f}\nn={g['n_entropy']}"]
        for pos, value, label in zip(positions, values, labels):
            ax.text(pos, value + 0.016, label, ha="center", va="bottom", fontsize=5)
    ax.set_xticks(xx, ["label-M fraction", "median entropy"], fontsize=5.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction / median entropy", fontsize=5.5)
    ax.text(0.02, 0.98, f"Fisher p = {fisher.pvalue:.3g}\nEntropy U p = {result['tests']['entropy_mannwhitney']['pvalue']:.3g}",
            transform=ax.transAxes, va="top", fontsize=5.2)
    ax.legend(handles=handles[:2], title="Subgroup", title_fontsize=5, fontsize=4.8, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3)
    fig.text(fs.LEFT / 210, 7 / 112,
             f"Gene-score silhouette = {silhouette:.3f}; coordinate ARI = {ari:.3f}. "
             "Entropy uses matched fate-clone rows (P3a: 5; P3b: 28).", fontsize=5, color=fs.INK2)
    fs.save(fig, "p3_subsplit", folder=OUT / "figures")
    print("[A5] Complete: new JSON, PNG and PDF written", flush=True)


if __name__ == "__main__":
    main()
