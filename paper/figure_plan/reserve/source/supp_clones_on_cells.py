#!/usr/bin/env python
"""Supplementary — the same information at three resolutions: cells, clones, clone maps.

For each of the three germinal-centre systems of Figure 2, in three columns:

1. the single-cell UMAP coloured by the annotated cell state (what scRNA shows);
2. the same UMAP with every cell that entered the clone embedding, sister cells of one clone
   joined by lines (what scRNA + scBCR shows when the receptor identity is only *overlaid*);
3. the Threadfin clone embedding of the same clones, coloured by programme (what the method
   resolves).

The point: neither the cell map nor the bare clone connections organise the response; the
clone profiles do.

Usage: python supp_clones_on_cells.py
Output: Supplementary_clones_on_cells.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fig2_gc_overview import COL, COMPARTMENT, KEY, ORDER, PROG_COL, state_col
from figstyle import INK, INK2, MUTED, new_page, panel, save

ROOT = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表")
CASE = ROOT / "Threadfin" / "case_studies" / "results"

DATASETS = [
    ("gut_gc_reporter", "Gut oral OVA · GSE284622"),
    ("malaria_late", "Plasmodium d10–42 · GSE286215"),
    ("ln_vaccine", "Human lymph node · GSE195673"),
]


def note(ax, text, y=-0.16, size=5.3):
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


def cell_panel(ax, cells, title, letter_note):
    grp = cells[state_col(cells)].map(COMPARTMENT).fillna("other")
    for k in ORDER:
        m = (grp == k).to_numpy()
        if m.any():
            ax.scatter(cells["umap_1"][m], cells["umap_2"][m], s=0.4, c=COL[k], linewidths=0,
                       alpha=0.55, rasterized=True)
    frame(ax, title, "Cell UMAP 1", "Cell UMAP 2")
    note(ax, letter_note)


def connections_panel(ax, cells, clones_tab, title):
    """Same UMAP; cells used in the clone embedding, sister cells of one clone joined."""
    multi = clones_tab[clones_tab["n_cells"] >= 2].index.astype(str)
    c = cells[cells["clone_id"].astype(str).isin(multi)].copy()
    c["clone_id"] = c["clone_id"].astype(str)
    ax.scatter(cells["umap_1"], cells["umap_2"], s=0.25, c="#ececea", linewidths=0,
               rasterized=True)
    # one line from every cell to its clone's centroid; long lines are the message
    cen = c.groupby("clone_id")[["umap_1", "umap_2"]].mean()
    xs, ys = [], []
    cx = cen["umap_1"]; cy = cen["umap_2"]
    for cid, sub in c.groupby("clone_id"):
        mx, my = cx[cid], cy[cid]
        for x0, y0 in zip(sub["umap_1"], sub["umap_2"]):
            xs += [x0, mx, None]
            ys += [y0, my, None]
    ax.plot(xs, ys, color=INK2, lw=0.25, alpha=0.35, rasterized=True)
    ax.scatter(c["umap_1"], c["umap_2"], s=1.0, c=INK, linewidths=0, alpha=0.6, rasterized=True)
    frame(ax, title, "Cell UMAP 1", "Cell UMAP 2")
    note(ax, f"{len(c):,} cells in {c['clone_id'].nunique():,} clones (≥2 cells);\n"
             "lines join sister cells of one clone.")


def clone_panel(ax, ds, tab, title):
    progs = sorted(tab["clone_programme"].dropna().unique(),
                   key=lambda p: int(str(p).lstrip("P")) if str(p).lstrip("P").isdigit() else 99)
    for p in progs:
        m = (tab["clone_programme"] == p).to_numpy()
        ax.scatter(tab["x"][m], tab["y"][m], s=np.clip(tab["n_cells"][m] * 1.2, 2.5, 26),
                   c=PROG_COL.get(str(p), MUTED), linewidths=0, alpha=0.85, rasterized=True)
    frame(ax, title, "Clone UMAP 1", "Clone UMAP 2")
    icc = json.loads((CASE / ds / "summary.json").read_text())["coherence"]
    note(ax, f"{len(tab):,} clones in {len(progs)} programmes\n"
             f"clonal ICC {icc['icc']:.2f} (p = {icc['p_value']:.3f})")


def main():
    fig = new_page("Supplementary",
                   "Cells, clones and clone maps of the three germinal-centre systems",
                   "", height=240)
    y0, dy, dx, w, h = 4, 76, 61, 58, 58
    for row, (ds, title) in enumerate(DATASETS):
        y = y0 + row * dy
        cells = pd.read_csv(CASE / ds / "cells.csv.gz")
        tab = pd.read_csv(CASE / ds / "clone_table.csv")
        tab = tab.set_index(tab["clone_id"].astype(str))
        cell_panel(panel(fig, 2, y, w, h, chr(ord("A") + row) if row == 0 else None),
                   cells, f"{title} — cells by annotated state",
                   f"{len(cells):,} cells")
        connections_panel(panel(fig, 2 + dx, y, w, h), cells, tab,
                          "same cells, sister cells joined per clone")
        clone_panel(panel(fig, 2 + 2 * dx, y, w, h), ds, tab,
                    "same clones, Threadfin clone embedding")
    save(fig, "Supplementary_clones_on_cells")


if __name__ == "__main__":
    main()
