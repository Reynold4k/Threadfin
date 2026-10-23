#!/usr/bin/env python
"""Figure 2 — clone embeddings resolve germinal-centre structure that scRNA alone does not show.

The figure makes one argument in three steps.

**Capability (B).** For three germinal-centre systems, Threadfin turns each clone into a profile
of where its cells sit in expression space and lays the clones out: genetically distinct clones
organise into sub-clone clusters (programmes, significant splits only).

**Biology (C).** Those regions are arms with biological meaning — germinal centre, plasmablast,
plasma cell, memory — coloured in the figure's shared compartment vocabulary. In the gut model
the plasma arm splits into two programmes with a maturation interpretation.

**Ground truth (D).** In the gut fate-reporter experiment the map organises measurements it never
saw: the tdTomato GC-fate sort. And the two plasma programmes, separated on the clone map at
15-NN agreement 0.99, are invisible on the cell-level UMAP of the same cells (0.55 vs a 0.50
null) — the structure exists between sister cells, not inside any one cell.

Clones throughout are Dandelion's paired heavy + light definition (`threadfin.define_clones_dandelion`).

Usage: python fig2_gc_overview.py
Output: Figure_2_GC_overview.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import clone_coords as cc
import schematics as sk
from figstyle import (AQUA, BLUE, GREEN, INK, INK2, MAGENTA, MUTED, ORANGE, RED, VIOLET, new_page,
                      panel, save)

ROOT = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表")
CASE = ROOT / "Threadfin" / "manuscript" / "case_studies" / "results"

# One vocabulary across the studies, so a colour means the same thing in every panel. Plasmablasts and
# resident plasma cells stay apart: the clone embedding puts them in different places, and collapsing
# them made one compartment look like it had two endpoints.
COMPARTMENT = {
    "GC": "germinal centre", "dark zone": "germinal centre", "light zone": "germinal centre",
    "Myc+ light zone": "germinal centre", "GC_Bcell": "germinal centre",
    "PB": "plasmablast", "PB-like": "plasmablast",
    "LNPC": "plasma cell", "plasma cell": "plasma cell", "plasma": "plasma cell",
    "Memory": "memory", "RMB": "memory", "Memory-like": "memory",
}
COL = {"germinal centre": BLUE, "plasmablast": RED, "plasma cell": VIOLET, "memory": GREEN,
       # no single compartment: either the clone's cells sit outside these four, or it is split
       # evenly between two of them
       "other": "#dcdbd3"}
ORDER = ["other", "memory", "plasma cell", "plasmablast", "germinal centre"]
KEY = ["germinal centre", "plasmablast", "plasma cell", "memory", "other"]


def note(ax, text, y=-0.195, size=5.3):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=size, color=INK2, va="top", linespacing=1.45)


def state_col(c: pd.DataFrame) -> str:
    return next(s for s in ("cell_state", "state", "fate", "celltype") if s in c.columns)


def cells_of(ds: str) -> pd.DataFrame:
    return pd.read_csv(CASE / ds / "cells.csv.gz")


def dominant(ds: str, cells: pd.DataFrame | None = None) -> pd.Series:
    """Compartment holding most of each clone's captured cells, in the shared vocabulary."""
    c = cells_of(ds) if cells is None else cells
    c = c[c["clone_id"].notna()]
    grp = c[state_col(c)].map(COMPARTMENT).fillna("other")
    tab = pd.crosstab(c["clone_id"].astype(str), grp)
    real = [k for k in KEY[:4] if k in tab.columns]
    if not real:
        return pd.Series("other", index=tab.index)
    frac = tab[real].div(tab[real].sum(axis=1).replace(0, np.nan), axis=0)
    best = frac.max(axis=1)
    # A clone split evenly between two compartments has no dominant one. `idxmax` would hand every
    # tie to the alphabetically first column -- "germinal centre" -- and with most clones holding
    # exactly two cells that is not a rounding detail: it silently moved 18 % of GSE246382's clones,
    # 12 % of GSE286215's and 8 % of GSE195673's into the germinal centre.
    decided = best.notna() & frac.eq(best, axis=0).sum(axis=1).eq(1)
    out = pd.Series("other", index=tab.index, dtype=object)
    out[decided] = frac[decided].idxmax(axis=1)
    return out


def frame(ax, title, xlab, ylab=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)
    ax.set_title(title, fontsize=6.4, color=INK, pad=3)
    ax.set_xlabel(xlab, fontsize=5.9, labelpad=2)
    if ylab:
        ax.set_ylabel(ylab, fontsize=5.9, labelpad=2)


def clone_count_label(ax, table, dataset, *, n_clones=None):
    """Place the clone/programme totals inside a clear region of each clone map."""
    total = len(table) if n_clones is None else n_clones
    n_programmes = table["clone_programme"].dropna().nunique()
    # The human map has a clear lower-right region; the other maps have clear upper-left corners.
    if dataset == "ln_vaccine":
        x, y, ha, va = 0.97, 0.035, "right", "bottom"
    else:
        x, y, ha, va = 0.035, 0.975, "left", "top"
    return ax.text(x, y, f"{total:,} clones in {n_programmes} programmes",
                   transform=ax.transAxes, ha=ha, va=va, fontsize=5.3, color=INK2,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
                   zorder=10)


# ------------------------------------------------------------------ A: the designs, drawn

#: name, icon, antigen glyph, design line, kept compartments, size line
DESIGNS = [
    ("model antigen", BLUE, [
        ("oral OVA / NP-OVA + CT · GSE284622", "mouse", "syringe",
         "3 reporter mice · gut GC fate sort (tdTomato) · 16,619 cells · 6,957 clones",
         ("germinal centre", "plasma cell"), "panels B–D"),
        ("NP-OVA · GSE287123", "mouse", "syringe",
         "7 mice · H2B-mCherry, 36 h of division history · 10,109 cells · 2,680 clones",
         ("germinal centre",), "Fig. S"),
        ("RBD protein / mRNA · GSE287123", "mouse", "syringe",
         "10 mice · two arms · bait and LZ/DZ gates recorded · 36,188 cells · 10,302 clones",
         ("germinal centre",), "Figure 3"),
        ("NP-OVA · GSE246382", "mouse", "syringe",
         "11 mice · Smart-seq2 · day 14 · 884 cells · W33L typed · 636 clones",
         ("germinal centre", "plasma cell"), "Fig. S"),
    ]),
    ("Plasmodium infection", RED, [
        ("P. chabaudi AS · experiment 1", "mouse", "parasite",
         "20 mice · days 0, 4, 7, 10, 14 · spleen · 40,623 cells · 20,500 clones",
         ("germinal centre", "plasmablast", "memory"), "Fig. S"),
        ("P. chabaudi AS · experiment 2", "mouse", "parasite",
         "42 mice · days 10–42 · ± artesunate · spleen · 72,359 cells · 58,587 clones",
         ("germinal centre", "plasmablast", "memory"), "panels B, C"),
    ]),
    ("human vaccination", VIOLET, [
        ("BNT162b2 · GSE195673", "person", "syringe",
         "8 participants · days 28–201 · repeat lymph-node FNA + blood · 193,442 cells · 104,322 clones",
         ("germinal centre", "plasmablast", "plasma cell", "memory"), "panels B, C"),
    ]),
]
CHIP = [("GC", "germinal centre"), ("PB", "plasmablast"), ("PC", "plasma cell"), ("Mem", "memory")]
CHIP_X = 112.0


def designs(ax):
    W, H = 183.0, 88.0
    sk.canvas(ax, W, H)
    sk.label(ax, 0, H, "Public datasets used to test Threadfin", 6.8, ha="left", weight="bold")
    sk.label(ax, CHIP_X + 4 * 11.5 / 2, H - 9.5, "Cell population included", 6.2, ha="center",
             color=INK2, weight="bold")
    sk.label(ax, 163, H - 9.5, "shown in", 6.2, ha="left", color=INK2, weight="bold")
    y = H - 9.5
    for header, col, entries in DESIGNS:
        sk.label(ax, 0, y, header, 6.0, ha="left", color=col, weight="bold")
        ax.plot([0, W], [y - 2.9, y - 2.9], color=col, lw=0.8, alpha=0.35)
        y -= 5.4
        for name, who, glyph, design, kept, where in entries:
            # the animal or participant, and what it met
            getattr(sk, who)(ax, 4.0, y - 2.6, s=0.50, color=col)
            if glyph == "syringe":
                sk.syringe(ax, 14.0, y - 2.6, s=0.44, color=col)
            else:
                sk.parasite(ax, 11.6, y - 2.6, s=0.85, color=col)
            sk.label(ax, 18, y, name, 5.5, ha="left", color=INK)
            sk.label(ax, 18, y - 3.3, design, 5.0, ha="left", color=INK2)
            # which compartments survived the sort, in the figure's own colours
            for i, (short, comp) in enumerate(CHIP):
                x = CHIP_X + i * 11.5
                on = comp in kept
                ax.add_patch(plt_rect(x, y - 3.9, 9.4, 3.7, COL[comp] if on else "none",
                                      COL[comp] if on else MUTED, 0.5 if on else 0.45))
                sk.label(ax, x + 4.7, y - 0.9, short, 4.9,
                         color="white" if on else MUTED, weight="bold" if on else None)
            if where:
                sk.label(ax, 163, y - 0.9, where, 4.9, ha="left",
                         color=MUTED if where.startswith("panel") else col)
            y -= 7.6
        y -= 1.2
    sk.label(ax, 0, 3.4, "All rows are paired scRNA-seq + scBCR-seq data. Dandelion defines each "
                         "clone from the paired heavy and light chains.",
             5.8, ha="left", color=INK2, wrap=180)


def plt_rect(x, y, w, h, face, edge, lw):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                          facecolor=face, edgecolor=edge, lw=lw, zorder=3)


# ------------------------------------------------------------------ B: cells, C: clones


def cell_umap(ax, ds: str, title: str):
    c = cells_of(ds)
    if "umap_1" not in c.columns:
        ax.set_axis_off(); return
    grp = c[state_col(c)].map(COMPARTMENT).fillna("other")
    for k in ORDER:
        m = (grp == k).to_numpy()
        if m.any():
            ax.scatter(c["umap_1"][m], c["umap_2"][m], s=0.5, c=COL[k], linewidths=0, alpha=0.5,
                       rasterized=True)
    frame(ax, title, "Cell UMAP 1", "Cell UMAP 2")
    note(ax, f"{len(c):,} cells, {c['clone_id'].notna().sum():,} with a receptor.")


def clone_embedding(ax, ds: str, title: str, with_note: bool = True):
    xy = cc.coords(ds)
    cells = cells_of(ds)
    dom = dominant(ds, cells).reindex(xy.index).fillna("other")
    t = pd.read_csv(CASE / ds / "clone_table.csv")
    size = t.set_index(t["clone_id"].astype(str))["n_cells"].reindex(xy.index).fillna(2).astype(float)
    # a clone captured once is drawn as a triangle. Threadfin filters nothing, so most points on
    # these maps rest on a single cell (75-98 % of clones in these deposits) and a reader is entitled
    # to see which ones at a glance.
    single = (size <= 1).to_numpy()
    for k in ORDER:
        for sel, mk, sc in ((single, "^", 1.0), (~single, "o", 1.5)):
            m = (dom == k).to_numpy() & sel
            if m.any():
                ax.scatter(xy["x"][m], xy["y"][m], s=np.clip(size[m] * sc, 2.5, 30), c=COL[k],
                           marker=mk, linewidths=0, alpha=0.85, rasterized=True)
    frame(ax, title, "Clone UMAP 1", "Clone UMAP 2")
    clone_count_label(ax, t, ds, n_clones=len(xy))
    icc = json.loads((CASE / ds / "summary.json").read_text())["coherence"]
    counts = dom.value_counts()
    top = " · ".join(f"{counts.get(k, 0):,} {s}" for s, k in
                     (("GC", "germinal centre"), ("PB", "plasmablast"),
                      ("PC", "plasma cell"), ("mem", "memory")) if counts.get(k, 0))
    if with_note:
        note(ax, f"{len(xy):,} clones · {int(size.sum()):,} cells\n"
                 f"clonal ICC {icc['icc']:.2f} (p = {icc['p_value']:.3f})\n{top}")
    return dom


def compartment_key(ax, present):
    """The shared compartment colour key, compact; title directly above the items."""
    ax.set_axis_off()
    ax.text(0, 1.0, "compartment (arm)", transform=ax.transAxes, fontsize=6.8, color=INK,
            va="top", fontweight="bold", linespacing=1.25)
    for i, k in enumerate(KEY):
        y = 0.80 - i * 0.16
        ax.scatter([0.06], [y], s=52, c=COL[k], linewidths=0, transform=ax.transAxes, clip_on=False)
        label = k
        ax.text(0.17, y, label, transform=ax.transAxes, fontsize=6.6, va="center",
                color=INK if k in present else MUTED)


PROG_COL = {"P1": BLUE, "P2": RED, "P3": VIOLET, "P4": ORANGE, "P5": AQUA, "P6": GREEN,
            "P7": MAGENTA, "P8": "#7a5c2e"}


def programme_map(ax, ds: str, title: str):
    """Clone map coloured by sub-clone cluster (programme) — the raw Threadfin output."""
    t = pd.read_csv(CASE / ds / "clone_table.csv")
    t = t.set_index(t["clone_id"].astype(str))
    progs = sorted(t["clone_programme"].dropna().unique(),
                   key=lambda p: int(str(p).lstrip("P")) if str(p).lstrip("P").isdigit() else 99)
    for p in progs:
        m = (t["clone_programme"] == p).to_numpy()
        ax.scatter(t["x"][m], t["y"][m], s=np.clip(t["n_cells"][m] * 1.2, 2.5, 26),
                   c=PROG_COL.get(str(p), MUTED), linewidths=0, alpha=0.85, rasterized=True,
                   label=str(p))
    frame(ax, title, "Clone UMAP 1", "Clone UMAP 2")
    clone_count_label(ax, t, ds)
    return progs


def prog_legend(ax, all_progs):
    """Compact programme colour key; title directly above the items."""
    ax.set_axis_off()
    ax.text(0, 1.0, "programme", transform=ax.transAxes, fontsize=6.8, color=INK, va="top",
            fontweight="bold", linespacing=1.25)
    ax.text(0, 0.87, "(sub-clone cluster)", transform=ax.transAxes, fontsize=5.6, color=INK2,
            va="top")
    for i, p in enumerate(all_progs):
        y = 0.66 - i * 0.15
        ax.scatter([0.06], [y], s=52, c=PROG_COL.get(p, MUTED), linewidths=0,
                   transform=ax.transAxes, clip_on=False)
        ax.text(0.17, y, p, transform=ax.transAxes, fontsize=6.6, va="center", color=INK)


# ------------------------------------------------------------------ B: the gut lead — two plasma branches


def gut_programme_cells(cells, clone_table, *, plasma_only=True):
    """Map clone programmes to existing cell coordinates with an explicit display subset."""
    programme = cells["clone_id"].map(clone_table["clone_programme"])
    selected = programme.isin(("P2", "P3"))
    if plasma_only:
        selected &= cells["celltype"].eq("plasma")
    return programme, selected.to_numpy()


def gut_validation(fig, y0):
    """D — ground truth for the gut map: a measured fate sort it never saw, and two plasma
    programmes that the cell-level UMAP of the same cells cannot show."""
    ds = "gut_gc_reporter"
    t = pd.read_csv(CASE / ds / "clone_table.csv")
    t = t.set_index(t["clone_id"].astype(str))
    cells = cells_of(ds)
    br = json.loads((CASE / ds / "plasma_branches.json").read_text())
    pc = {"P2": RED, "P3": VIOLET}

    # D1: measured GC-fate sort on the clone map
    ax = panel(fig, 2, y0, 50, 46, "D")
    v = pd.to_numeric(t["gc_sort:GC"], errors="coerce")
    ok = v.notna().to_numpy()
    ax.scatter(t["x"][~ok], t["y"][~ok], s=3.5, c="#ececea", linewidths=0, rasterized=True)
    s = ax.scatter(t["x"][ok], t["y"][ok], s=np.clip(t["n_cells"][ok] * 1.5, 2.5, 30), c=v[ok],
                   cmap="viridis", vmin=0, vmax=1, linewidths=0, alpha=0.9, rasterized=True)
    frame(ax, "Measured GC-fate sort,\nnever given to the map", "Clone UMAP 1", "Clone UMAP 2")
    clone_count_label(ax, t, ds)
    # A dedicated colourbar column leaves the next y-axis label clear.
    cax = panel(fig, 54, y0 + 1, 2.1, 44, None)
    cb = fig.colorbar(s, cax=cax)
    cb.ax.tick_params(labelsize=4.6, pad=1.5)

    # D2: the same branches on the cell-level UMAP — invisible
    ax = panel(fig, 68, y0, 52, 46, None)
    programme, pl = gut_programme_cells(cells, t, plasma_only=True)
    ax.scatter(cells["umap_1"][~pl], cells["umap_2"][~pl], s=0.4, c="#e7e6e1", linewidths=0,
               rasterized=True)
    for p in ("P2", "P3"):
        m = pl & (programme == p).to_numpy()
        ax.scatter(cells["umap_1"][m], cells["umap_2"][m], s=1.4, c=pc[p], linewidths=0, alpha=0.6,
                   rasterized=True)
    frame(ax, "The two plasma programmes\non the cell UMAP", "Cell UMAP 1", "Cell UMAP 2")
    vis = br["visibility"]

    # D3: what separates the branches — cell-level expression ratio
    ax = panel(fig, 134, y0, 48, 46, None)
    expr = br["branch_gene_expression"]
    genes = [g for g in ("Mki67", "Top2a", "Irf8", "Sdc1", "Ccr10", "Irf4") if g in expr]
    r = [np.log2(expr[g]["ratio_P3_over_P2"]) for g in genes]
    ypos = np.arange(len(genes))[::-1]
    ax.barh(ypos, r, height=0.62, color=[VIOLET if x > 0 else RED for x in r])
    ax.set_yticks(ypos, labels=genes, fontsize=5.4)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlabel("log2 P3 / P2", fontsize=5.6, labelpad=2)
    ax.set_title("Plasma branch markers\n(cell level)", fontsize=6.4, color=INK, pad=3)


def main():
    fig = new_page("Figure 2",
                   "Clone embeddings resolve germinal-centre structure that scRNA alone does not show",
                   "", height=304)

    designs(panel(fig, 0, 4, 183, 88, "A"))

    # B — what Threadfin draws: clone maps coloured by sub-clone cluster (programme)
    all_progs = []
    for ds, title, box in (
            ("gut_gc_reporter", "Gut oral OVA · GSE284622", (2, 98, 58, 52)),
            ("malaria_late", "Plasmodium d10–42 · GSE286215", (63, 98, 58, 52)),
            ("ln_vaccine", "Human lymph node · GSE195673", (124, 98, 58, 52))):
        progs = programme_map(panel(fig, *box, "B" if ds == "gut_gc_reporter" else None), ds, title)
        all_progs = sorted(set(all_progs) | set(progs))

    # C — the same maps annotated by compartment: the arms
    for ds, title, box in (
            ("gut_gc_reporter", "Gut oral OVA · GSE284622", (2, 166, 58, 52)),
            ("malaria_late", "Plasmodium d10–42 · GSE286215", (63, 166, 58, 52)),
            ("ln_vaccine", "Human lymph node · GSE195673", (124, 166, 58, 52))):
        clone_embedding(panel(fig, *box, "C" if ds == "gut_gc_reporter" else None), ds, title,
                        with_note=False)

    # D — ground truth: the gut fate sort, and the branches the cell UMAP cannot show
    gut_validation(fig, 228)

    save(fig, "Figure_2_GC_overview", dpi=1200)


if __name__ == "__main__":
    main()
