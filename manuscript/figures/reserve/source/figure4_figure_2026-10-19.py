#!/usr/bin/env python
"""Figure 4: division history, antigen binding and zone in RBD germinal-centre clones (GSE287123).

Draws the A4 figure from the outputs of ``figure4_analysis_2026-10-19.py`` in the typography of the
current Figures 2-3 (Arial/Liberation Sans; 8.8-pt titles, 8-pt axis labels, 7.5-pt ticks, 11-pt panel
letters). Statistics are read, not recomputed. ``--preview`` writes only the preview and the layout audit.

    python reserve/source/figure4_figure_2026-10-19.py [--preview]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Text

import schematics as sk

HERE = Path(__file__).resolve().parents[2]                       # manuscript/figures
WORK = HERE / "reserve" / "figure4_rbd_2026-10-19"
STEM = "Figure_4_model_antigen_GC"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
MISSING = "#e7e6e0"
LOW, HIGH = "#2a78d6", "#eb6834"                                  # division gates, as in Supplementary Fig. 1
PROTEIN, MRNA = "#4a3aa7", "#008300"                              # cohorts
STRIP = LinearSegmentedColormap.from_list("strip", ["#f3f3f0", "#9fb3c2", "#2f4b62"])
MODULES = [("dark zone / cycling", "DZ / cycling"), ("light zone", "Light zone"),
           ("memory", "Memory"), ("naive", "Naive")]
DECILES = (1, 5, 10)
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
         "font.size": 8, "axes.titlesize": 8.8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK,
         "axes.edgecolor": INK2, "axes.linewidth": .5, "axes.spines.top": False, "axes.spines.right": False,
         "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2, "ytick.major.size": 2,
         "xtick.major.width": .5, "ytick.major.width": .5, "legend.frameon": False,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "svg.hashsalt": "threadfin-figure4-rbd-20261019", "savefig.facecolor": "white"}


# --------------------------------------------------------------------------- page helpers (millimetres)


def ax_mm(fig, x, top, w, h, **kw):
    return fig.add_axes([x / 210, 1 - (top + h) / 297, w / 210, h / 297], **kw)


def text_mm(fig, x, y, s, size=7.5, **kw):
    kw.setdefault("va", "top")
    return fig.text(x / 210, 1 - y / 297, s, fontsize=size, **kw)


def letter(fig, x, y, s):
    text_mm(fig, x, y - .6, s, 11, weight="bold")


def title(fig, x, y, s, sub=None):
    text_mm(fig, x, y, s, 8.8, ha="center")
    if sub:
        text_mm(fig, x, y + 4.3, sub, 7.6, ha="center", color=INK2)


def square_limits(xy, pad=.06):
    lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    c, r = (lo + hi) / 2, np.max(hi - lo) * (.5 + pad)
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def map_frame(ax, limits, kind):
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{kind} UMAP 1", labelpad=2)
    ax.set_ylabel(f"{kind} UMAP 2", labelpad=2)


def clone_area(n):
    return np.clip(2.0 + .5 * np.asarray(n, dtype=float), 2.4, 30)


def fraction_map(ax, clones, column, limits, seed):
    missing = clones[column].isna()
    ax.scatter(clones.loc[missing, "x"], clones.loc[missing, "y"], s=clone_area(clones.loc[missing, "n_cells"]),
               c=MISSING, linewidths=0, rasterized=True)
    shown = clones[~missing].sample(frac=1, random_state=seed)
    sc = ax.scatter(shown["x"], shown["y"], s=clone_area(shown["n_cells"]), c=shown[column], cmap="viridis",
                    vmin=0, vmax=1, linewidths=0, alpha=.92, rasterized=True)
    map_frame(ax, limits, "Clone")
    return sc, int((~missing).sum())


def colourbar(fig, sc, x, y, w, label):
    cax = ax_mm(fig, x, y, w, 2.1)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.ax.set_xticklabels(["0", "0.5", "1"])
    cb.outline.set_linewidth(.45)
    cb.ax.tick_params(labelsize=7.5, pad=1.5, length=1.5)
    text_mm(fig, x + w / 2, y - 1.0, label, 7.6, ha="center", va="bottom")


def gate_colours(series):
    return np.where(series.eq("mCherry-low"), LOW, HIGH)


def decile_lines(ax, dec, specs, *, offset=.09):
    for i, (annotation, cohort, colour) in enumerate(specs):
        d = dec[(dec["annotation"] == annotation) & (dec["cohort"] == cohort)].sort_values("pc1_decile")
        xs = d["pc1_decile"].to_numpy() + (i - (len(specs) - 1) / 2) * 2 * offset
        ax.vlines(xs, d["ci_low"], d["ci_high"], color=colour, lw=.8, alpha=.85, zorder=2)
        ax.plot(xs, d["mean"], color=colour, lw=1.3, zorder=3)
        ax.scatter(xs, d["mean"], s=13, color=colour, edgecolors="white", linewidths=.5, zorder=4)
    ax.set_xlim(.4, 10.6)
    ax.set_xticks(range(1, 11))
    ax.tick_params(axis="x", length=2)
    ax.grid(axis="y", color="#ecebe6", lw=.5, zorder=0)
    ax.set_axisbelow(True)


def line_handles(items):
    return [Line2D([], [], color=c, marker="o", markersize=3.8, lw=1.3, label=s) for s, c in items]


def dot_handles(items, size=3.8):
    return [Line2D([], [], color=c, marker="o", markersize=size, ls="", label=s) for s, c in items]


# --------------------------------------------------------------------------- panel a: design


def design(ax, summary):
    w, h = 80.0, 55.0
    sk.canvas(ax, w, h)
    # H2B-mCherry is diluted by every division during the chase
    ax.text(38.5, 53.6, "H2B–mCherry, 36-h chase", ha="center", va="center", fontsize=7.6)
    xs, y = [21.0, 32.0, 43.0, 54.0], 46.6
    for i, (x, tone) in enumerate(zip(xs, ["#c8384f", "#dc7a8a", "#ecb3bb", "#f8e6e8"])):
        ax.add_patch(Circle((x, y), 2.5, facecolor=tone, edgecolor=INK2, lw=.5, zorder=3))
        if i < 3:
            sk.arrow(ax, x + 3.1, y, xs[i + 1] - 3.1, y, color=MUTED, lw=.6, head=1.5)
    for x, c, s in ((21.0, HIGH, "mCherry-high"), (54.0, LOW, "mCherry-low (≥6 divisions)")):
        ax.add_patch(FancyBboxPatch((x - 6.5, 41.4), 13.0, .9, boxstyle="round,pad=0,rounding_size=.4",
                                    facecolor=c, edgecolor="none", zorder=2))
        ax.text(x, 40.2, s, ha="center", va="top", fontsize=7.4)
    # four libraries shared by the two cohorts, with the division gates crossed
    gx, gw, gh = 27.0, 12.6, 8.6
    ax.text(10.0, 30.6, "Library", ha="left", va="center", fontsize=7.6)
    for j, lib in enumerate("ABCD"):
        ax.text(gx + j * gw + gw / 2, 30.6, lib, ha="center", va="center", fontsize=7.6, weight="bold")
    rows = [("Protein", ["RBD−", "RBD−", "RBD+", "RBD+"], [1, 0, 1, 0], 22.8),
            ("mRNA", ["LZ", "LZ", "DZ", "DZ"], [0, 1, 0, 1], 12.6)]
    for name, gates, low, yc in rows:
        sk.mouse(ax, 4.0, yc + .2, s=.52, color=INK2)
        ax.text(9.2, yc + 1.8, name, ha="left", va="center", fontsize=7.6)
        ax.text(9.2, yc - 1.9, "5 mice", ha="left", va="center", fontsize=7.4, color=INK2)
        for j, (g, is_low) in enumerate(zip(gates, low)):
            ax.add_patch(FancyBboxPatch((gx + j * gw + .45, yc - gh / 2), gw - .9, gh,
                                        boxstyle="round,pad=0,rounding_size=.8",
                                        facecolor="#dbe8f8" if is_low else "#fbe3d8",
                                        edgecolor=LOW if is_low else HIGH, lw=.7, zorder=2))
            ax.text(gx + j * gw + gw / 2, yc + 1.6, g, ha="center", va="center", fontsize=7.4, zorder=3)
            ax.text(gx + j * gw + gw / 2, yc - 1.8, "low" if is_low else "high", ha="center", va="center",
                    fontsize=7.4, zorder=3, color=INK2)
    ax.text(w / 2, 2.6, f"{summary['n_cells']:,} GC B cells · {summary['n_clones']:,} clones with ≥2 cells",
            ha="center", va="center", fontsize=7.6)


# --------------------------------------------------------------------------- figure


def build(summary, clones, dec, cells, null):
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    record = {}
    mice = summary["correlations"]["mouse"]
    climits = square_limits(cells[["umap_1", "umap_2"]].to_numpy())
    mlimits = square_limits(clones[["x", "y"]].to_numpy())

    # row 1: design, cells, clones --------------------------------------------
    letter(fig, 2.5, 6.5, "a")
    title(fig, 45, 6.5, "GSE287123 | Division reporter")
    design(ax_mm(fig, 5, 12, 80, 55), summary)

    letter(fig, 92, 6.5, "b")
    title(fig, 120, 6.5, "Measured division gate")
    ax = ax_mm(fig, 96, 12, 48, 48)
    order = cells.sample(frac=1, random_state=20261019)
    ax.scatter(order["umap_1"], order["umap_2"], s=.3, linewidths=0, alpha=.8, rasterized=True,
               c=gate_colours(order["division_gate"]))
    map_frame(ax, climits, "Cell")
    n_low, n_high = int(cells["division_gate"].eq("mCherry-low").sum()), int(cells["division_gate"].eq("mCherry-high").sum())
    ax.legend(handles=dot_handles([(f"mCherry-low (n = {n_low:,})", LOW), (f"mCherry-high (n = {n_high:,})", HIGH)]),
              loc="upper left", bbox_to_anchor=(0, -.085), borderaxespad=0, handletextpad=.4, labelspacing=.3)
    record["b"] = {"n_cells": len(cells), "n_low": n_low, "n_high": n_high}

    letter(fig, 151, 6.5, "c")
    title(fig, 180, 6.5, "Recently divided fraction")
    ax = ax_mm(fig, 156, 12, 48, 48)
    sc, n = fraction_map(ax, clones, "recently_divided", mlimits, seed=1)
    colourbar(fig, sc, 162, 69.5, 36, "Fraction of the clone's cells, mCherry-low")
    record["c"] = {"n_clones": n}

    # row 2: division along PC1, crossover test, mice ---------------------------
    letter(fig, 6, 80, "d")
    title(fig, 53, 80, "Recent division along PC1")
    ax = ax_mm(fig, 22, 86, 62, 29)
    decile_lines(ax, dec, [("recently_divided", "protein", PROTEIN), ("recently_divided", "mRNA", MRNA)])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Recently divided\nfraction", labelpad=3, linespacing=1.1)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(handles=line_handles([("Protein (mCherry-low in A, C)", PROTEIN), ("mRNA (mCherry-low in B, D)", MRNA)]),
              loc="lower right", borderaxespad=.2, handlelength=1.4, handletextpad=.5, labelspacing=.3, fontsize=7.4)
    strip = ax_mm(fig, 22, 117.2, 62, 11.6)
    mat = []
    for key, _ in MODULES:
        v = dec[dec["annotation"] == key].sort_values("pc1_decile")["mean"].to_numpy()
        mat.append((v - v.min()) / (v.max() - v.min()))
    strip.imshow(np.vstack(mat), aspect="auto", cmap=STRIP, vmin=0, vmax=1, extent=(.5, 10.5, len(MODULES), 0),
                 interpolation="nearest")
    strip.set_xlim(.4, 10.6)
    strip.set_yticks(np.arange(len(MODULES)) + .5)
    strip.set_yticklabels([lab for _, lab in MODULES], fontsize=7.4)
    strip.set_xticks(range(1, 11))
    strip.tick_params(axis="y", length=0, pad=2)
    strip.tick_params(axis="x", length=2)
    for s in strip.spines.values():
        s.set_visible(False)
    strip.set_xlabel("Clone profile PC1 (decile)", labelpad=2)
    text_mm(fig, 86.5, 117.6, "RNA module\n(row-scaled)", 7.2, ha="left", linespacing=1.1, color=INK2)
    key = ax_mm(fig, 87, 124.4, 8, 1.8)
    key.imshow(np.linspace(0, 1, 50)[None, :], aspect="auto", cmap=STRIP)
    key.set_xticks([0, 49])
    key.set_xticklabels(["low", "high"], fontsize=7.2)
    key.set_yticks([])
    key.tick_params(axis="x", length=0, pad=1.2)
    for s in key.spines.values():
        s.set_visible(False)
    co = summary["correlations"]["cohort"]
    record["d"] = {"rho_protein": co["recently_divided|protein"]["rho"], "rho_mrna": co["recently_divided|mRNA"]["rho"]}

    cross = summary["crossover"]
    letter(fig, 104, 80, "e")
    title(fig, 131, 80, "Library crossover test")
    ax = ax_mm(fig, 113, 86, 39, 32)
    counts, _, _ = ax.hist(null, bins=np.linspace(-1, 1, 41), color=GRID, edgecolor="white", linewidth=.3)
    top = counts.max() * 1.14
    ax.set_ylim(0, top)
    ax.axvline(cross["cosine"], color=INK, lw=1.2)
    ax.text(cross["cosine"] - .05, top * .97, f"observed\n{cross['cosine']:.2f}", ha="right", va="top",
            fontsize=7.4, linespacing=1.1)
    ax.plot([-1], [top * .045], marker="v", ms=4.5, color=INK2, clip_on=False, zorder=5)
    ax.plot([-1, -1], [top * .1, top * .5], color=INK2, lw=.5)
    ax.text(-.96, top * .5, "library\neffect", ha="left", va="top", fontsize=7.4, color=INK2, linespacing=1.1)
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "1"])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Cosine similarity of the\ndivision directions", labelpad=2, linespacing=1.1)
    ax.set_ylabel("Permutations", labelpad=2)
    text_mm(fig, 132.5, 130.6, f"P = {cross['p_value']:.4f}; 2,000\npermutations within mice", 7.4,
            ha="center", linespacing=1.15)
    record["e"] = {k: cross[k] for k in ("cosine", "null_mean", "null_q95", "p_value")}

    letter(fig, 157, 80, "f")
    title(fig, 183, 80, "Within-mouse correlations")
    ax = ax_mm(fig, 177, 86, 29, 37)
    rows = [mice[k] for k in mice if k.startswith("recently_divided|")]
    rows.sort(key=lambda r: (r["cohort"] != "protein", int(r["donor"][1:])))
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r["rho"] for r in rows], height=.66,
            color=[PROTEIN if r["cohort"] == "protein" else MRNA for r in rows])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['donor']} (n = {r['n']})" for r in rows], fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.axvline(0, color=INK2, lw=.5)
    ax.set_xlim(0, .52)
    ax.set_xticks([0, .25, .5])
    ax.set_xticklabels(["0", "0.25", "0.5"])
    ax.set_xlabel("Spearman ρ\n(division, PC1)", labelpad=2, linespacing=1.1)
    ax.spines["left"].set_visible(False)
    record["f"] = {r["donor"]: round(r["rho"], 3) for r in rows}

    # row 3: cells of the clones at three positions along PC1 --------------------
    cl = cells.join(clones["pc1_decile"], on="clone_id")
    letter(fig, 6, 146.5, "g")
    record["g"] = {}
    for i, d in enumerate(DECILES):
        x0 = 18 + i * 66
        part = cl[cl["pc1_decile"].eq(d)]
        frac = part["division_gate"].eq("mCherry-low").mean()
        n_cl = int(clones["pc1_decile"].eq(d).sum())
        title(fig, x0 + 24, 146.5, f"Clones in PC1 decile {d}",
              f"{n_cl} clones · {len(part):,} cells · {100 * frac:.0f}% mCherry-low")
        ax = ax_mm(fig, x0, 157, 48, 48)
        ax.scatter(cells["umap_1"], cells["umap_2"], s=.25, c=MISSING, linewidths=0, rasterized=True)
        part = part.sample(frac=1, random_state=d)
        ax.scatter(part["umap_1"], part["umap_2"], s=1.6, linewidths=0, alpha=.9, rasterized=True,
                   c=gate_colours(part["division_gate"]))
        map_frame(ax, climits, "Cell")
        record["g"][f"decile_{d}"] = {"n_clones": n_cl, "n_cells": int(len(part)), "fraction_low": float(frac)}
    fig.legend(handles=dot_handles([("mCherry-low", LOW), ("mCherry-high", HIGH),
                                    ("Other cells", MISSING)], size=4.2),
               loc="upper center", bbox_to_anchor=(.5, 1 - 211.5 / 297), ncol=3, borderaxespad=0,
               handletextpad=.4, columnspacing=1.6)

    # row 4: antigen binding and zone ------------------------------------------
    letter(fig, 6, 221.5, "h")
    sc, n_h = fraction_map(ax_mm(fig, 16, 232, 46, 46), clones, "rbd_binding", mlimits, seed=2)
    title(fig, 39, 221.5, "RBD-binding fraction", f"Protein mice · {n_h:,} clones")
    letter(fig, 70, 221.5, "i")
    sc, n_i = fraction_map(ax_mm(fig, 80, 232, 46, 46), clones, "dark_zone", mlimits, seed=3)
    title(fig, 103, 221.5, "Dark-zone fraction", f"mRNA mice · {n_i:,} clones")
    colourbar(fig, sc, 24, 289.0, 46, "Fraction of the clone's cells in the gate")
    fig.legend(handles=dot_handles([("Not measured", MISSING)], size=4.2), loc="upper left",
               bbox_to_anchor=(80 / 210, 1 - 287.2 / 297), borderaxespad=0, handletextpad=.4)
    record["h_i"] = {"n_clones_binding": n_h, "n_clones_zone": n_i}

    letter(fig, 140, 221.5, "j")
    title(fig, 178, 221.5, "Binding and zone along PC1")
    ax = ax_mm(fig, 152, 230, 54, 37)
    decile_lines(ax, dec, [("rbd_binding", "protein", PROTEIN), ("dark_zone", "mRNA", MRNA)])
    ax.set_ylim(0, .9)
    ax.set_yticks([0, .25, .5, .75])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75"])
    ax.set_ylabel("Fraction of the\nclone's cells", labelpad=3, linespacing=1.1)
    ax.set_xlabel("Clone profile PC1 (decile)", labelpad=2)
    ax.legend(handles=line_handles([("RBD+ (protein)", PROTEIN), ("Dark zone (mRNA)", MRNA)]),
              loc="upper left", borderaxespad=.2, handlelength=1.4, handletextpad=.5, labelspacing=.3, fontsize=7.4)
    rb = [m for k, m in mice.items() if k.startswith("rbd_binding|")]
    dz = [m for k, m in mice.items() if k.startswith("dark_zone|")]
    text_mm(fig, 179, 282.5, f"ρ > 0 within each mouse: binding {sum(m['rho'] > 0 for m in rb)}/{len(rb)},"
                             f" zone {sum(m['rho'] > 0 for m in dz)}/{len(dz)}", 7.4, ha="center")
    record["j"] = {"binding_positive": sum(m["rho"] > 0 for m in rb), "zone_positive": sum(m["rho"] > 0 for m in dz)}
    fig.threadfin_record = record
    return fig


# --------------------------------------------------------------------------- audit and export


def audit(fig):
    """Text outside the page and overlapping text boxes (measured from the rendered extents)."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(ren)) for t in texts]
    boxes = [(t, b) for t, b in boxes if np.isfinite(b.extents).all()]
    outside = [t.get_text() for t, b in boxes if b.x0 < 0 or b.y0 < 0 or b.x1 > w or b.y1 > h]
    overlaps = []
    for i, (t, b) in enumerate(boxes):
        for u, c in boxes[i + 1:]:
            dx = min(b.x1, c.x1) - max(b.x0, c.x0)
            dy = min(b.y1, c.y1) - max(b.y0, c.y0)
            if dx > 1 and dy > 1:
                overlaps.append([t.get_text(), u.get_text()])
    return {"page_mm": [210, 297], "text_outside_page": outside, "text_overlaps": overlaps,
            "font_sizes_pt": sorted({round(t.get_fontsize(), 2) for t, _ in boxes}),
            "minimum_font_pt": min(t.get_fontsize() for t, _ in boxes)}


def render(preview: bool = False, dpi: int = 600) -> None:
    inputs = {"summary": WORK / "summary.json", "clones": WORK / "clones.csv", "deciles": WORK / "pc1_deciles.csv",
              "cells": WORK / "cells.csv.gz", "null": WORK / "crossover_null.npy"}
    summary = json.loads(inputs["summary"].read_text())
    clones = pd.read_csv(inputs["clones"], index_col=0)
    dec = pd.read_csv(inputs["deciles"])
    cells = pd.read_csv(inputs["cells"], index_col=0)
    null = np.load(inputs["null"])
    fig = build(summary, clones, dec, cells, null)
    report = audit(fig)
    (WORK / "layout_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    fig.savefig(WORK / "preview.png", dpi=170)
    print(json.dumps({k: report[k] for k in ("text_outside_page", "text_overlaps", "minimum_font_pt")},
                     ensure_ascii=False))
    if not preview:
        assert not report["text_outside_page"], report["text_outside_page"]
        assert not report["text_overlaps"], report["text_overlaps"]
        for ext in ("png", "pdf", "svg"):
            fig.savefig(HERE / f"{STEM}.{ext}", dpi=dpi)
        fig.savefig(HERE / "reserve" / "current_previews" / f"{STEM}.png", dpi=170)
        prov = {"generator": str(Path(__file__).relative_to(HERE)),
                "analysis": "reserve/source/figure4_analysis_2026-10-19.py",
                "page_mm": [210, 297], "dpi": dpi,
                "inputs": {k: {"path": str(p.relative_to(HERE)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for k, p in inputs.items()},
                "panels": fig.threadfin_record, "layout_audit": report}
        (HERE / "reserve" / "provenance" / f"{STEM}.json").write_text(
            json.dumps(prov, ensure_ascii=False, indent=1, default=float) + "\n")
        print("exported", HERE / STEM)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    render(args.preview, args.dpi)


if __name__ == "__main__":
    main()
