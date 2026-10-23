#!/usr/bin/env python
"""Figure 3h re-coloured: programme, mouse, nonGC share among plasma cells; plus the same clones uncentred.

Run check_plasma_arms.py first (numbers in allB_threadfin/story/plasma_arms_check.json).
Output: plasma_arms_by_mouse.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIG = ROOT / "case_studies" / "results" / "gut_gc_reporter"
ALLB = HERE / "allB_threadfin"

PROG = {"P1": "#405BD8", "P2": "#E9AE14", "P3": "#E44336"}          # Figure 3 g-h colours
MOUSE = {"mouse2": "#2a78d6", "mouse3": "#eb6834", "mouse4": "#1baf7a"}  # as in allB_centring_check
GREY, INK, MUTED = "#e7e6e0", "#1f1f1f", "#5f5f5f"
RAMP = LinearSegmentedColormap.from_list("nongc", ["#e4e0f7", "#7a6ad8", "#2b1d73"])

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#4a4a4a",
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def size(n):
    return 1.5 + 1.1 * np.minimum(n, 60)


def frame(ax, title, sub):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8, color=INK, pad=21, loc="left")
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6.3, color=MUTED, va="bottom", ha="left")


def legend(ax, items, loc="lower left"):
    h = [Line2D([], [], ls="", marker="o", ms=4.5, mfc=c, mec="white", mew=0.4, label=l) for l, c in items]
    ax.legend(handles=h, loc=loc, frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.2,
              labelcolor=INK)


def main():
    chk = json.loads((ALLB / "story" / "plasma_arms_check.json").read_text())["figure_3_run"]
    tab = pd.read_csv(ORIG / "clone_table.csv")
    cells = pd.read_csv(ORIG / "cells.csv.gz")
    cells["gc_sort"] = cells["sample"].str.split("_").str[-1]
    pc = cells[cells["celltype"] == "plasma"].groupby("clone_id")["gc_sort"]
    tab["pc_nonGC"] = tab["clone_id"].map(pc.apply(lambda s: (s == "nonGC").mean()))
    plasma = (tab["celltype:plasma"] >= 0.5) & tab["clone_programme"].isin(["P2", "P3"])
    unc = pd.read_csv(ALLB / "none" / "clone_table.csv")

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.03, wspace=0.10)
    order = tab.sample(frac=1, random_state=1).index

    ax = axs[0]
    t = tab.loc[order]
    ax.scatter(t.x, t.y, s=size(t.n_cells), c=t.clone_programme.map(PROG).fillna(GREY), lw=0.25, ec="white")
    frame(ax, "Figure 3h as drawn", f"{len(tab)} clones\ndonor-centred, as published")
    legend(ax, list(PROG.items()))

    ax = axs[1]
    ax.scatter(t.x, t.y, s=size(t.n_cells), c=t.donor.map(MOUSE), lw=0.25, ec="white")
    r = chk["region_by_mouse"]
    low, up = r["lower arm (P2)"], r["upper arm (P3)"]
    frame(ax, "Same map, by mouse",
          f"lower arm: mouse3 {low['mouse3']}/{sum(low.values())}\nupper arm: mouse3 {up['mouse3']}/{sum(up.values())}")
    legend(ax, list(MOUSE.items()))

    ax = axs[2]
    ax.scatter(tab.x[~plasma], tab.y[~plasma], s=size(tab.n_cells[~plasma]), c=GREY, lw=0)
    p = tab[plasma].sort_values("pc_nonGC")
    sc = ax.scatter(p.x, p.y, s=size(p.n_cells), c=p.pc_nonGC, cmap=RAMP, vmin=0, vmax=1, lw=0.25, ec="white")
    q = chk["region_nonGC"]
    frame(ax, "nonGC share of plasma cells",
          f"upper arm {q['upper arm (P3)']['observed_nonGC_share']:.1%} (mouse mix: "
          f"{q['upper arm (P3)']['expected_from_mouse_only']:.1%})\nlower arm "
          f"{q['lower arm (P2)']['observed_nonGC_share']:.1%} (mouse mix: {q['lower arm (P2)']['expected_from_mouse_only']:.1%})")
    cax = ax.inset_axes([0.05, 0.08, 0.32, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.tick_params(labelsize=6, length=1.5, colors=INK); cb.outline.set_linewidth(0.4)
    cb.ax.set_title("plasma clones", fontsize=6, color=INK, pad=2)

    ax = axs[3]
    u = unc.sample(frac=1, random_state=1)
    ax.scatter(u.x, u.y, s=size(u.n_cells), c=u.donor.map(MOUSE), lw=0.25, ec="white")
    mix = json.loads((ALLB / "summary.json").read_text())["variants"]["none"]["donor_mixing_plasma_clones"]
    frame(ax, "Same clones, no centring", f"{len(unc)} clones; one plasma lobe\n"
          f"same-mouse neighbours {mix['observed']:.2f} (chance {mix['null_mean']:.2f})")
    legend(ax, list(MOUSE.items()), loc="upper right")

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"plasma_arms_by_mouse.{ext}", dpi=300)
    print("wrote", HERE / "plasma_arms_by_mouse.png")


if __name__ == "__main__":
    main()
