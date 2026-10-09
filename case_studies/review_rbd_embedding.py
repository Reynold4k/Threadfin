#!/usr/bin/env python
"""Create a descriptive sensitivity review for the RBD clone-map embedding.

This script deliberately reads only the completed audit artefacts.  It does
not reprocess RNA, construct new Threadfin features, or choose a replacement
for the previously used Figure 2 configuration.  The two alternative layouts
are geometry-led sensitivity displays, not selected for their label patterns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
AUDIT = HERE / "results" / "mouse_rbd_embedding_audit"
OUT = AUDIT / "review"
CLONES = HERE / "results" / "mouse_rbd" / "clone_table.csv"

# Default was the Figure 2 configuration before this review.  Alternatives
# were selected on unsupervised geometry (not label association): k15/md0.9
# has high sampled distance faithfulness, while k50/md0.5 displays a smoother
# neighbourhood-scale layout.
SETTINGS = [
    {"key": "default", "n_neighbors": 15, "min_dist": 0.1,
     "spread": 1.0, "random_state": 0,
     "decision": "primary_retained"},
    {"key": "geometry_sensitivity", "n_neighbors": 15, "min_dist": 0.9,
     "spread": 1.0, "random_state": 0,
     "decision": "sensitivity_high_geometry_faithfulness"},
    {"key": "continuous_sensitivity", "n_neighbors": 50, "min_dist": 0.5,
     "spread": 1.0, "random_state": 0,
     "decision": "sensitivity_continuous_layout"},
]
VARIABLES = [
    ("division_gate:mCherry-low", "Measured division\n(fraction mCherry-low)", "viridis", 0.0, 1.0),
    ("zone_gate:DZ", "Measured DZ sort fraction\n(mRNA arm; unmeasured = grey)", "plasma", 0.0, 1.0),
    ("mutation_frequency", "SHM\n(mean V-region mutation frequency)", "magma", 0.0, 0.03),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setting_mask(frame: pd.DataFrame, s: dict) -> pd.Series:
    return ((frame.n_neighbors == s["n_neighbors"]) &
            np.isclose(frame.min_dist, s["min_dist"]) &
            np.isclose(frame.spread, s["spread"]) &
            (frame.random_state == s["random_state"]))


def metric_row(sweep: pd.DataFrame, setting: dict, variable: str) -> dict:
    row = sweep.loc[setting_mask(sweep, setting) & sweep.variable.eq(variable)]
    if len(row) != 1:
        raise ValueError(f"Expected one sweep row for {setting['key']} / {variable}, got {len(row)}")
    return row.iloc[0].to_dict()


def seed_robustness(sweep: pd.DataFrame, variable: str) -> dict:
    q = sweep.loc[sweep.variable.eq(variable)]
    wide = q.pivot_table(index=["n_neighbors", "min_dist", "spread"], columns="random_state",
                         values=["lin_r2", "excess"], aggfunc="first")
    out = {}
    for metric in ("lin_r2", "excess"):
        if (metric, 0) not in wide or (metric, 1) not in wide:
            continue
        d = (wide[(metric, 1)] - wide[(metric, 0)]).dropna()
        out[metric] = {"n_paired_settings": int(len(d)),
                       "median_seed1_minus_seed0": float(d.median()),
                       "median_absolute_difference": float(d.abs().median()),
                       "min_difference": float(d.min()), "max_difference": float(d.max())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sweep_path, coords_path = AUDIT / "sweep.csv", AUDIT / "gallery_coords.csv.gz"
    sweep = pd.read_csv(sweep_path)
    coords = pd.read_csv(coords_path).set_index("clone_id")
    clone = pd.read_csv(CLONES, index_col=0)
    clone = clone.reindex(coords.index)
    if (not clone.index.is_unique or clone.index.hasnans or
            clone["reliability"].isna().any() or clone["reliability"].lt(0.5).any()):
        raise ValueError("gallery coordinates are not exactly the reliable clone subset")
    if len(clone) != 381:
        raise ValueError(f"Expected 381 reliable families, got {len(clone)}")

    # Exact reproduction is assessed from the independently stored gallery
    # coordinates for the same setting, preserving clone order by clone_id.
    xcol, ycol = "x_k15_md0.1", "y_k15_md0.1"
    reproduction = {"x_correlation": float(np.corrcoef(clone.x, coords[xcol])[0, 1]),
                    "y_correlation": float(np.corrcoef(clone.y, coords[ycol])[0, 1]),
                    "max_abs_x_difference": float(np.max(np.abs(clone.x - coords[xcol]))),
                    "max_abs_y_difference": float(np.max(np.abs(clone.y - coords[ycol])))}

    rows = []
    for setting in SETTINGS:
        xy = coords[[f"x_k{setting['n_neighbors']}_md{setting['min_dist']}",
                     f"y_k{setting['n_neighbors']}_md{setting['min_dist']}"]].to_numpy()
        for variable, _, _, _, _ in VARIABLES:
            r = metric_row(sweep, setting, variable)
            from scipy.spatial import cKDTree
            values = clone[variable].to_numpy(float)
            keep = np.isfinite(values)
            value = values[keep]
            neighbours = cKDTree(xy[keep]).query(xy[keep], k=11)[1][:, 1:]
            residual = value - value[neighbours].mean(axis=1)
            ev = 1 - residual.var() / value.var()
            # The legacy sweep column named knn_r2 is actually explained
            # variance, not standard predictive R² when mean bias is nonzero.
            if not np.isclose(ev, r['knn_r2'], atol=1e-6):
                raise ValueError(f"Saved-map local metric mismatch: {setting['key']} / {variable}")
            standard_r2 = 1 - np.mean(residual**2) / value.var()
            rows.append({"setting": setting["key"], "decision": setting["decision"],
                         **{k: setting[k] for k in ("n_neighbors", "min_dist", "spread", "random_state")},
                         "variable": variable,
                         **{k: r[k] for k in ("faithfulness", "n", "lin_r2", "p_value")},
                         "knn_explained_variance": ev, "knn_loo_standard_r2": standard_r2,
                         "mean_prediction_residual": float(residual.mean()),
                         "null_mean_explained_variance": r['null_mean'],
                         "excess_explained_variance": r['excess']})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "parameter_comparison.csv", index=False)

    # All-sweep ranges are descriptive.  Nominal permutation p-values are not
    # reinterpreted after examining 85 settings or multiple variables.
    ranges = {}
    for variable, *_ in VARIABLES:
        q = sweep.loc[sweep.variable.eq(variable)]
        ranges[variable] = {
            "n_parameter_sets": int(len(q)),
            "faithfulness": {"min": float(q.faithfulness.min()), "max": float(q.faithfulness.max()),
                               "median": float(q.faithfulness.median())},
            "lin_r2": {"min": float(q.lin_r2.min()), "max": float(q.lin_r2.max()),
                       "median": float(q.lin_r2.median())},
            "excess_explained_variance": {"min": float(q.excess.min()), "max": float(q.excess.max()),
                       "median": float(q.excess.median())},
            "nominal_p_le_0_05": int((q.p_value <= .05).sum()),
            "seed_0_1_robustness": seed_robustness(sweep, variable),
        }

    fig, axs = plt.subplots(3, 3, figsize=(11.4, 10.2), constrained_layout=True)
    sizes = 1 + 5 * np.sqrt(clone.n_cells.to_numpy())
    for col, setting in enumerate(SETTINGS):
        x = coords[f"x_k{setting['n_neighbors']}_md{setting['min_dist']}"]
        y = coords[f"y_k{setting['n_neighbors']}_md{setting['min_dist']}"]
        faith = metric_row(sweep, setting, "division_gate:mCherry-low")["faithfulness"]
        axs[0, col].set_title(
            f"{setting['key'].replace('_', ' ')}\nk={setting['n_neighbors']}, min_dist={setting['min_dist']}, "
            f"spread={setting['spread']}, seed={setting['random_state']}\nfaithfulness={faith:.3f}", fontsize=9)
        for row, (variable, label, cmap, lo, hi) in enumerate(VARIABLES):
            ax = axs[row, col]
            values = clone[variable].to_numpy(float)
            finite = np.isfinite(values)
            if variable == "zone_gate:DZ":
                ax.scatter(x, y, s=sizes, c="#d9dde3", edgecolors="white", linewidths=.14)
            plotted = ax.scatter(x[finite], y[finite], c=np.clip(values[finite], lo, hi), s=sizes[finite],
                                 cmap=cmap, vmin=lo, vmax=hi, edgecolors="white", linewidths=.14)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            metric = metric_row(sweep, setting, variable)
            ax.text(.02, .02, f"linear R²={metric['lin_r2']:.3f}\nlocal ΔEV={metric['excess']:.3f}",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if col == 2:
                cb = fig.colorbar(plotted, ax=ax, fraction=.035, pad=.01)
                cb.ax.tick_params(labelsize=6)
    fig.suptitle("RBD Threadfin family-map sensitivity review (381 reliable same-mouse IGH sequence-defined families)\n"
                 "Previously used configuration retained; alternatives are geometry-led sensitivity displays.\n"
                 "Local ΔEV = leave-one-family-out explained variance minus donor-stratified permutation mean.",
                 fontsize=9.5)
    fig.savefig(OUT / "parameter_comparison.png", dpi=220)
    fig.savefig(OUT / "parameter_comparison.pdf")
    plt.close(fig)

    summary = {
        "scope": "Read-only review of existing sweep, gallery coordinates, and committed clone table; no RNA reprocessing.",
        "family_definition": "Same-donor IGH families defined from heavy-chain V, J and nucleotide junction sequence similarity; not pooled V-D-J combination groups.",
        "n_reliable_families": int(len(clone)),
        "metric_definitions": {
            "lin_r2": "In-sample OLS R² for value ~ 1 + x + y",
            "legacy_knn_r2": "Actually explained variance EV = 1 - Var(y - prediction) / Var(y); legacy field name retained in original sweep.csv",
            "excess": "Observed EV minus mean EV across 200 donor-stratified permutations; not R²",
            "knn_loo_standard_r2": "1 - mean((y - prediction)^2) / Var(y), recomputed for the three displayed settings; no new permutation test",
            "neighbour_readout": "Mean of 10 nearest labelled families, excluding self; family leave-one-out, not held-out-mouse prediction",
            "faithfulness": "Spearman correlation of pair distances in features and 2D, sampled anew for each setting; not local neighbour preservation",
        },
        "parameter_decision": {
            "primary": SETTINGS[0],
            "rationale": "Retain the previously used Figure 2 setting because label conclusions are stable across the scan; geometry-led alternatives are sensitivity displays, not post hoc label optimisation.",
            "sensitivity_settings": SETTINGS[1:],
            "inference_note": "Programme scores reuse RNA used to form family features and are descriptive. Measured labels are external, but gate-specific libraries remain a potential confound. Nominal p-values across this exploratory scan are not selection-adjusted confirmation tests.",
        },
        "exact_default_coordinate_reproduction": reproduction,
        "all_85_setting_ranges_and_seed_robustness": ranges,
        "source_sha256": {str(p.relative_to(HERE)): sha256(p) for p in (sweep_path, coords_path, CLONES)},
        "outputs": ["parameter_comparison.csv", "parameter_comparison.png", "parameter_comparison.pdf", "review_summary.json"],
    }
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote review to {OUT}")


if __name__ == "__main__":
    main()
