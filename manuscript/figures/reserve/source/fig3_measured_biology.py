#!/usr/bin/env python
"""Figure 3 — clone embeddings organise measurements they never saw.

One design, three systems. In each row the measured label was recorded at the bench and never
given to the embedding: an antigen bait, a division reporter and a zone gate (RBD); the later
myeloid fate of the same barcode (LARRY); the later drug-persister gate of the same lineage
barcode (WATERMELON). Single cells carry the labels weakly or diffusely; the clone embedding
concentrates them into coherent structure, and early clone profiles predict the later measured
fate.

Usage: python fig3_measured_biology.py
Output: Figure_3_measured_biology.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import clone_coords as cc
from figstyle import (BLUE, INK, INK2, MUTED, ORANGE, RED, VIOLET, new_page, panel, save)

ROOT = Path(__file__).resolve().parents[5]          # the folder that holds the repository clone
CASE = ROOT / "Threadfin" / "manuscript" / "case_studies" / "results"
SR = ROOT / "story_review_2026-10-12"
# non-B-cell comparison results moved out of the package repository on 2026-10-18
LOCAL = ROOT / "internal_validation/repo_validation_2026-10-18/case_studies/results"


def note(ax, text, y=-0.195, size=5.3):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=size, color=INK2, va="top",
            linespacing=1.45)


def frame(ax, title, xlab, ylab=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)
    ax.set_title(title, fontsize=6.4, color=INK, pad=3)
    ax.set_xlabel(xlab, fontsize=5.9, labelpad=2)
    if ylab:
        ax.set_ylabel(ylab, fontsize=5.9, labelpad=2)


def scatter_label(fig, ax, x, y, v, n, cmap, title, xlab, ylab=None, vmin=0, vmax=1):
    ok = v.notna().to_numpy()
    ax.scatter(x[~ok], y[~ok], s=3.0, c="#ececea", linewidths=0, rasterized=True)
    s = ax.scatter(x[ok], y[ok], s=np.clip(n[ok] * 1.2, 2.5, 26), c=v[ok], cmap=cmap, vmin=vmin,
                   vmax=vmax, linewidths=0, alpha=0.9, rasterized=True)
    frame(ax, title, xlab, ylab)
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.4)
    return ok


# ------------------------------------------------------------------ A: RBD — antigen bait, division, zone

def row_rbd(fig, y0):
    t = pd.read_csv(CASE / "mouse_rbd" / "clone_table.csv")
    xy = cc.coords("mouse_rbd")
    t = t.set_index(t["clone_id"].astype(str)).reindex(xy.index)
    cells = pd.read_csv(CASE / "mouse_rbd" / "cells.csv.gz")

    ax = panel(fig, 2, y0, 40, 42, "A")
    for g, col in (("mCherry-high", ORANGE), ("mCherry-low", BLUE)):
        m = (cells["division_gate"] == g).to_numpy()
        ax.scatter(cells["umap_1"][m], cells["umap_2"][m], s=0.5, c=col, linewidths=0, alpha=0.5,
                   rasterized=True, label=g.replace("mCherry-", ""))
    frame(ax, "RBD cells, 36 h division reporter", "Cell UMAP 1", "Cell UMAP 2")
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", markerscale=6, handletextpad=0.15,
              labelspacing=0.2, borderpad=0.1)
    note(ax, f"{len(cells):,} cells; germinal-centre-only sort.")

    for i, (col, label, cmap) in enumerate((
            ("division_gate:mCherry-low", "clone fraction ≥6 recent divisions", "viridis"),
            ("rbd_bait:RBD+", "clone fraction binding the RBD bait", "cividis"),
            ("zone_gate:DZ", "clone fraction sorted dark zone", "magma_r"))):
        ax = panel(fig, 48 + i * 45, y0, 40, 42, None)
        v = pd.to_numeric(t[col], errors="coerce")
        ok = scatter_label(fig, ax, xy["x"], xy["y"], v, t["n_cells"], cmap, label,
                           "Clone UMAP 1", "Clone UMAP 2" if i == 0 else None)
        note(ax, f"{int(ok.sum()):,} clones measured")


# ------------------------------------------------------------------ B: LARRY — later myeloid fate

def row_larry(fig, y0):
    d = LOCAL / "clonotrace_larry"
    cells = pd.read_csv(d / "larry_day2_cells_fate.csv.gz")
    clones = pd.read_csv(d / "larry_day2_clonemap_fate.csv.gz")
    oof = pd.read_csv(SR / "LARRY_OOF_barcode_bootstrap.csv")
    sw = pd.read_csv(SR / "LARRY_sister_well_points.csv")
    vis = json.loads((d / "larry_cell_vs_clone_visibility.json").read_text())

    ax = panel(fig, 2, y0, 40, 42, "B")
    s = ax.scatter(cells["spring_x"], cells["spring_y"], s=2.2, c=cells["fate"], cmap="cividis",
                   vmin=0, vmax=1, linewidths=0, rasterized=True)
    frame(ax, "LARRY day-2 cells (author SPRING)", "SPRING 1", "SPRING 2")
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.4)
    note(ax, f"{len(cells):,} day-2 cells coloured by the clone's\nday-6 neutrophil fraction; "
             f"15-NN r {vis['cell_spring_knn_r_excluding_same_clone']:.2f}.")

    ax = panel(fig, 48, y0, 40, 42, None)
    s = ax.scatter(clones["x"], clones["y"], s=14, c=clones["fate"], cmap="cividis", vmin=0,
                   vmax=1, linewidths=0, rasterized=True)
    frame(ax, "The same clones, day-2 profiles only", "Clone UMAP 1", "Clone UMAP 2")
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.4)
    note(ax, f"{len(clones):,} clones with ≥2 early cells; the map\nnever saw day 4–6.")

    # held-out prediction, kernel vs single-cell baselines
    ax = panel(fig, 94, y0, 44, 42, None)
    methods = [("Training_mean", "#c9c8bf", "null"), ("Size_only", "#a8a79e", "size"),
               ("RNA_mean", BLUE, "cell mean"), ("RNA_mean_variance", "#7aa7e0", "mean+var"),
               ("Threadfin_kernel", RED, "clone profile")]
    xt, xlabels = [], []
    for bi, target in enumerate(("Neutrophil", "Monocyte")):
        sub = oof[oof["target"] == target].set_index("method")
        for mi, (m, colr, lab) in enumerate(methods):
            if m not in sub.index:
                continue
            x = bi * (len(methods) + 1) + mi
            r = sub.loc[m]
            ax.bar(x, r["repeated_OOF_R2"], width=0.8, color=colr)
            ax.errorbar(x, r["repeated_OOF_R2"],
                        yerr=[[r["repeated_OOF_R2"] - r["boot_low"]],
                              [r["boot_high"] - r["repeated_OOF_R2"]]],
                        color=INK2, lw=0.6, capsize=1.2)
            xt.append(x); xlabels.append(lab)
    ax.set_ylim(top=0.36)
    for bi, target in enumerate(("Neutrophil", "Monocyte")):
        ax.text(bi * (len(methods) + 1) + 2, 0.335, target, fontsize=5.4, ha="center", color=INK)
    ax.set_xticks(xt, xlabels, fontsize=4.6, rotation=38, ha="right")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("out-of-fold R²", fontsize=5.6)
    ax.set_title("Day-2 profile predicts day-6 fate\n(held-out barcodes)", fontsize=6.4, pad=3)
    note(ax, "blue: single-cell RNA baselines; red: Threadfin\nkernel clone profile.")

    # sister-well reproducibility
    ax = panel(fig, 142, y0, 40, 42, None)
    for tgt, colr in (("Neutrophil", RED), ("Monocyte", VIOLET)):
        s_ = sw[sw["target"] == tgt]
        ax.scatter(s_["well1"], s_["well2"], s=7, c=colr, alpha=0.5, linewidths=0,
                   rasterized=True, label=tgt)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=":")
    ax.set_xlabel("fraction in well 1", fontsize=5.6)
    ax.set_ylabel("fraction in well 2", fontsize=5.6)
    ax.set_title("Sister wells reproduce the bias", fontsize=6.4, pad=3)
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", handletextpad=0.15, borderpad=0.1)
    note(ax, "211 barcodes in both day-6 wells: ρ = 0.73\n(Neutrophil), 0.69 (Monocyte), "
             "p = 2×10⁻⁴.")


# ------------------------------------------------------------------ C: WATERMELON — later persister gate

def row_watermelon(fig, y0):
    d = LOCAL / "watermelon"
    cells = pd.read_csv(d / "cells_umap.csv.gz", index_col=0)
    base = pd.read_csv(d / "cells.csv.gz", index_col=0)
    cells["day"] = base["day"].reindex(cells.index)
    et = pd.read_csv(d / "early_clone_table.csv")
    vis = json.loads((d / "cell_vs_clone_visibility.json").read_text())
    summ = json.loads((d / "summary.json").read_text())

    ax = panel(fig, 2, y0, 40, 42, "C")
    e = cells[(cells["day"] < 14)]
    eg = e.dropna(subset=["future_gate"])
    ax.scatter(e["umap_1"], e["umap_2"], s=0.3, c="#e7e6e1", linewidths=0, rasterized=True)
    s = ax.scatter(eg["umap_1"], eg["umap_2"], s=0.7, c=eg["future_gate"], cmap="viridis",
                   vmin=0, vmax=1, linewidths=0, alpha=0.6, rasterized=True)
    frame(ax, "PC9 day 0–7 cells (scRNA UMAP)", "Cell UMAP 1", "Cell UMAP 2")
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.4)
    note(ax, f"{len(eg):,} cells coloured by their clone's later\nday-14 mCherry-high fraction; "
             f"15-NN r {vis['cell_umap_knn_r_excluding_same_clone']:.2f}.")

    ax = panel(fig, 48, y0, 40, 42, None)
    et2 = et.dropna(subset=["gate14:mCherry-high"])
    s = ax.scatter(et2["x"], et2["y"], s=10, c=et2["gate14:mCherry-high"], cmap="viridis",
                   vmin=0, vmax=1, linewidths=0, rasterized=True)
    frame(ax, "The same clones, day 0–7 profiles only", "Clone UMAP 1", "Clone UMAP 2")
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.4)
    note(ax, f"{len(et2):,} clones; map built without day-14 cells.")

    # association strength: kernel vs mean profiles
    ax = panel(fig, 94, y0, 44, 42, None)
    ek = summ["early_to_late"]["label_effects"]
    em = summ["early_to_late"]["label_effects_mean_profiles"]
    rows = [("mCherry-high\n(persister)", ek["gate14:mCherry-high"]["excess_r2"],
             em["gate14:mCherry-high"]["excess_r2"]),
            ("mCherry-low\n(cycling)", ek["gate14:mCherry-low"]["excess_r2"],
             em["gate14:mCherry-low"]["excess_r2"])]
    x = np.arange(2)
    ax.bar(x - 0.18, [r[2] for r in rows], width=0.36, color=BLUE, label="clone mean (scRNA)")
    ax.bar(x + 0.18, [r[1] for r in rows], width=0.36, color=RED, label="clone kernel profile")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=5.2)
    ax.set_ylabel("R² over permuted null", fontsize=5.6)
    ax.set_title("Early clone profile vs later\nday-14 gate (measured)", fontsize=6.4, pad=3)
    ax.legend(frameon=False, fontsize=5.0, loc="upper right", borderpad=0.1)
    note(ax, "n = 1,016 clones with ≥2 early + ≥2 day-14 cells;\nboth p = 5×10⁻⁴.")

    # design summary / proposed validation
    ax = panel(fig, 142, y0, 40, 42, None)
    ax.set_axis_off()
    ax.text(0, 1.0, "Readout", transform=ax.transAxes, fontsize=6.2, color=INK, weight="bold",
            va="top")
    ax.text(0, 0.82, "A clone's day 0–7 state predicts\nits persister propensity a week later\n"
                     "(osimertinib, PC9). Proposed: sort\npredicted high-persister clones early,\n"
                     "challenge, re-score the gate.", transform=ax.transAxes, fontsize=5.3,
            color=INK2, va="top", linespacing=1.5)


def main():
    fig = new_page("Figure 3",
                   "Clone embeddings organise measurements they never saw",
                   "", height=196)
    row_rbd(fig, 6)
    row_larry(fig, 68)
    row_watermelon(fig, 130)
    save(fig, "Figure_3_measured_biology")


if __name__ == "__main__":
    main()
