#!/usr/bin/env python
"""Supplementary Figures 3, 4 and 7 of the 2026-10-17 revision (600-dpi PNG).

S3  State diversity of spontaneous-GC clones in profile space, with clone size controlled (GSE203132).
S4  Clone relationships in the gut cohort on the Figure 3h embedding (GSE284622).
S7  Donor composition and centring of the gut clone profiles (GSE284622).

Inputs are the outputs of revision_spont_gc_2026-10-17.py and revision_gut_2026-10-17.py; the saved Figure 3
coordinates and programmes are used throughout. Output: reserve/revision_2026-10-17/figures/.
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
from matplotlib.patches import Patch, Rectangle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[4]
REV = ROOT / "manuscript" / "figures" / "reserve" / "revision_2026-10-17"
OUT = REV / "figures"
RES = ROOT / "manuscript" / "case_studies" / "results"
DPI = 600
MM = 1 / 25.4

INK, INK2, MUTED, GREY, BG = "#111111", "#4d4c49", "#86857f", "#c3c2b7", "#dedcd5"
GUT_PROG = {"P1": "#405BD8", "P2": "#E9AE14", "P3": "#E44336"}
REGION = {"P1": "#405BD8", "P2 bridge": "#F2C94C", "P2 lower": "#B07D0A", "P3": "#E44336"}
REGION_ORDER = ["P1", "P2 bridge", "P2 lower", "P3"]
MOUSE = {"mouse2": "#6a5acd", "mouse3": "#e07fb0", "mouse4": "#1b9e77"}
MOUSE_LABEL = {"mouse2": "Mouse 2", "mouse3": "Mouse 3", "mouse4": "Mouse 4"}
CELLTYPE = {"B_cell": "#91a6bd", "GC_Bcell": "#2a78d6", "plasma": "#4a3aa7", "other": "#c3c2b7"}
CELLTYPE_LABEL = {"B_cell": "B cell", "GC_Bcell": "GC B cell", "plasma": "Plasma cell", "other": "Other"}
TISSUE = {"PP": "#6c9a3f", "MLN": "#c9a227", "LP": "#b5523b"}
SPAN = {"GC B only": "#405BD8", "GC B and plasma": "#2fa4a0", "plasma only": "#E44336", "other": "#c3c2b7"}
ISO = {"IgA": "#b5523b", "IgG1": "#4a3aa7", "IgG2b": "#8f86cf", "IgM": "#91a6bd", "IgD": "#c3c2b7",
       "other": "#e6e4dc"}
HIST = {"all GC-sorted": "#f2c94c", "mixed": "#2fa4a0", "all nonGC": "#2b1d73"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f1e6", "#a9cfc3", "#3f8f9b", "#22406e"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 7.0, "axes.labelsize": 6.5, "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0, "legend.fontsize": 6.0, "axes.linewidth": 0.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "axes.titlepad": 3.0, "savefig.dpi": DPI,
})


# --------------------------------------------------------------------------- layout helpers


class Page:
    """Axes placed in millimetres from the top-left corner of a 210-mm-wide page."""

    def __init__(self, height_mm: float):
        self.h = height_mm
        self.fig = plt.figure(figsize=(210 * MM, height_mm * MM))

    def ax(self, x, y, w, h, letter=None, dx=-5.0):
        a = self.fig.add_axes([x / 210, 1 - (y + h) / self.h, w / 210, h / self.h])
        if letter:
            self.fig.text((x + dx) / 210, 1 - (y - 3.2) / self.h, letter, fontsize=9, fontweight="bold",
                          color=INK, ha="left", va="bottom")
        return a

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / f"{name}.png", dpi=DPI)
        plt.close(self.fig)
        print("wrote", OUT / f"{name}.png")


def embedding_frame(ax, title, xlabel="Clone UMAP 1", ylabel="Clone UMAP 2"):
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=2), ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title)


def size_of(n, scale=0.75, cap=60):
    return 1.5 + scale * np.minimum(np.asarray(n, dtype=float), cap)


def legend_points(ax, items, loc="upper left", ncol=1, bbox=None, title=None, ms=4.2):
    h = [Line2D([], [], ls="", marker="o", ms=ms, mfc=c, mec="none", label=l) for l, c in items]
    kw = dict(loc=loc, ncol=ncol, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2, labelspacing=0.3)
    if bbox is not None:
        kw["bbox_to_anchor"] = bbox
    lg = ax.legend(handles=h, title=title, **kw)
    if title:
        lg.get_title().set_fontsize(6.0)
    return lg


def colourbar(page, ax, sc, label, x, y, w=2.2, h=14):
    cax = page.ax(x, y, w, h)
    cb = page.fig.colorbar(sc, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.set_label(label, fontsize=6.0, labelpad=2)
    return cb


def scatter_clones(ax, x, y, c, n, order=None, **kw):
    order = np.argsort(order) if order is not None else np.random.default_rng(1).permutation(len(x))
    return ax.scatter(np.asarray(x)[order], np.asarray(y)[order], s=size_of(np.asarray(n)[order]),
                      c=np.asarray(c, dtype=object)[order] if kw.get("cmap") is None else np.asarray(c)[order],
                      linewidths=0, rasterized=True, **kw)


def num(v, nd=2):
    """Fixed-point number with a typographic minus sign."""
    return f"{v:.{nd}f}".replace("-", "−")


def legend_below(ax, items, ncol=2, y=-0.13, ms=4.2):
    h = [Line2D([], [], ls="", marker="o", ms=ms, mfc=c, mec="none", label=l) for l, c in items]
    return ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handletextpad=0.2,
                     columnspacing=0.9, borderaxespad=0.0, labelspacing=0.3)


def p_text(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"P = {p / 10 ** e:.1f} × 10$^{{{e}}}$"
    return f"P = {p:.2g}" if p < 0.01 else f"P = {p:.2f}"


# --------------------------------------------------------------------------- Supplementary Figure 3


def figure_s3():
    d = pd.read_csv(REV / "spont_gc" / "clone_axes.csv", index_col=0)
    s = pd.read_csv(REV / "spont_gc" / "clone_state_diversity.csv", index_col=0)
    summ = json.loads((REV / "spont_gc" / "summary.json").read_text())
    ref = pd.read_csv(RES / "spont_gc" / "clone_table.csv").set_index("clone_id").loc[d.index]
    pg = Page(262)

    # a, clone embedding coloured by profile PC1
    ax = pg.ax(14, 12, 50, 46, "a")
    sc = ax.scatter(ref["x"], ref["y"], c=d["pc1"], s=size_of(ref["n_cells"], 0.5), cmap="cividis",
                    linewidths=0, rasterized=True)
    embedding_frame(ax, "Profile PC1 on the clone embedding")
    colourbar(pg, ax, sc, "PC1", 66, 22, h=20)

    # b, display coordinate versus profile PC1
    ax = pg.ax(92, 12, 42, 46, "b")
    ax.scatter(d["x"], d["pc1"], s=2.5, c=INK2, alpha=0.5, linewidths=0, rasterized=True)
    r = summ["axis_agreement_all_clones"]["x_vs_pc1"]
    ax.set_xlabel("Clone UMAP 1"), ax.set_ylabel("Profile PC1")
    ax.set_title("Display axis and profile axis")
    ax.text(0.03, 0.97, f"ρ = {num(r['rho'])}\nn = {r['n']:,} clones", transform=ax.transAxes, va="top")

    # c, entropy versus profile PC1
    ax = pg.ax(146, 12, 50, 46, "c")
    ax.scatter(s["pc1"], s["entropy"], s=size_of(s["n_annotated"], 0.35), c="#2a9d8f", alpha=0.75,
               linewidths=0, rasterized=True)
    r = summ["entropy"]["pc1"]["raw"]
    ax.set_xlabel("Profile PC1"), ax.set_ylabel("Shannon entropy")
    ax.set_title("State diversity along profile PC1")
    ax.text(0.97, 0.97, f"ρ = {num(r['rho'])}\n{p_text(r['p'])}\nn = {r['n']} clones", transform=ax.transAxes,
            va="top", ha="right")

    # d, rarefied entropy
    ax = pg.ax(14, 76, 50, 46, "d")
    ax.scatter(s["pc1"], s["entropy_rarefied"], s=size_of(s["n_annotated"], 0.35), c="#2a9d8f", alpha=0.75,
               linewidths=0, rasterized=True)
    r = summ["entropy"]["pc1"]["rarefied"]
    ax.set_xlabel("Profile PC1"), ax.set_ylabel("Entropy, rarefied to 5 cells")
    ax.set_title("Clone size controlled by rarefaction")
    ax.text(0.97, 0.97, f"ρ = {num(r['rho'])}\n{p_text(r['p'])}", transform=ax.transAxes, va="top", ha="right")

    # e, entropy versus clone size
    ax = pg.ax(80, 76, 44, 46, "e")
    ax.scatter(s["n_annotated"], s["entropy"], s=3, c=INK2, alpha=0.55, linewidths=0, rasterized=True)
    ax.set_xscale("log")
    r = summ["size_effects"]["entropy_vs_log_n"]
    ax.set_xlabel("Annotated cells per clone"), ax.set_ylabel("Shannon entropy")
    ax.set_title("Entropy and clone size")
    ax.text(0.97, 0.97, f"ρ = {num(r['rho'])}\n{p_text(r['p'])}", transform=ax.transAxes, va="top", ha="right")

    # f, within-mouse correlations, raw and rarefied
    ax = pg.ax(148, 76, 48, 46, "f")
    raw = summ["entropy"]["pc1"]["within_mouse_raw"]
    rar = summ["entropy"]["pc1"]["within_mouse_rarefied"]
    mice = sorted(raw, key=lambda m: raw[m]["rho"])
    yy = np.arange(len(mice))
    ax.barh(yy + 0.18, [raw[m]["rho"] for m in mice], height=0.34, color="#2a9d8f", label="Entropy")
    ax.barh(yy - 0.18, [rar[m]["rho"] for m in mice], height=0.34, color="#9cd3cb", label="Rarefied")
    ax.set_yticks(yy, [f"{m} (n = {raw[m]['n']})" for m in mice])
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlim(-1, 0.05)
    ax.set_xlabel("Spearman ρ with profile PC1")
    ax.set_title("Within each mouse")
    ax.legend(loc="lower left", handlelength=1.0, borderaxespad=0.1)
    ax.invert_yaxis()

    # g, memory-containing clones by PC1 quartile
    ax = pg.ax(14, 142, 78, 46, "g")
    q = summ["memory"]["pc1"]["quartiles"]
    keys = ["Q1", "Q2", "Q3", "Q4"]
    xx = np.arange(4)
    obs = [100 * q[k]["observed"] for k in keys]
    rare = [100 * q[k]["rarefied"] for k in keys]
    ax.bar(xx - 0.2, obs, width=0.38, color="#457b9d", label="Observed")
    ax.bar(xx + 0.2, rare, width=0.38, color="#a8c5da", label="Rarefied to 5 cells")
    for i, k in enumerate(keys):
        ax.text(i - 0.2, obs[i] + 0.8, f"{q[k]['n_mem']}/{q[k]['n']}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(xx, keys)
    ax.set_xlabel("Quartile of profile PC1 (GC side → ASC side)")
    ax.set_ylabel("Clones with ≥1 memory B cell (%)")
    ax.set_title("Memory-containing clones along profile PC1")
    r = summ["memory"]["pc1"]["rarefied_vs_axis"]
    ax.text(0.97, 0.97, f"Rarefied vs PC1: ρ = {num(r['rho'])}, {p_text(r['p'])}", transform=ax.transAxes,
            ha="right", va="top")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.0)

    # h, which other states do memory-containing clones contain?
    ax = pg.ax(118, 142, 78, 46, "h")
    mp = summ["memory_clone_partners"]
    groups = ["Memory-containing", "Other clones"]
    with_gc = [mp["with_GC"] / mp["n_memory_containing"], mp["non_memory_with_GC"] / mp["n_non_memory"]]
    with_asc = [mp["with_ASC"] / mp["n_memory_containing"], mp["non_memory_with_ASC"] / mp["n_non_memory"]]
    xx = np.arange(2)
    ax.bar(xx - 0.2, np.array(with_gc) * 100, width=0.38, color="#2a9d8f", label="Contains GC B cells")
    ax.bar(xx + 0.2, np.array(with_asc) * 100, width=0.38, color="#e63946", label="Contains ASCs")
    ns = [mp["n_memory_containing"], mp["n_non_memory"]]
    for i in range(2):
        ax.text(i - 0.2, with_gc[i] * 100 + 1, f"{round(with_gc[i] * ns[i])}", ha="center", fontsize=5.6)
        ax.text(i + 0.2, with_asc[i] * 100 + 1, f"{round(with_asc[i] * ns[i])}", ha="center", fontsize=5.6)
    ax.set_xticks(xx, [f"{g}\n(n = {n})" for g, n in zip(groups, ns)])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Clones (%)")
    ax.set_title("Other states in the same clone")
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, bbox_to_anchor=(0.5, 1.0))

    # i, memory-containing clones on the clone embedding (all profiled clones)
    ax = pg.ax(14, 208, 60, 44, "i")
    mem = d["memory_containing"].to_numpy()
    o = np.argsort(mem)
    ax.scatter(ref["x"].to_numpy()[o], ref["y"].to_numpy()[o], s=size_of(ref["n_cells"].to_numpy()[o], 0.5),
               c=np.where(mem[o], "#1b7a3a", BG), linewidths=0, rasterized=True)
    embedding_frame(ax, "Clones containing a memory B cell")
    legend_points(ax, [("≥1 memory B cell", "#1b7a3a"), ("None", BG)], loc="upper right", ms=3.5)

    # j, memory-containing clones by programme (Figure 3b programmes; P3 and P4 are post hoc)
    ax = pg.ax(100, 208, 60, 44, "j")
    prov = json.loads((ROOT / "manuscript" / "figures" / "reserve" / "provenance" /
                       "Figure_3_memory_annotation.json").read_text())["groups"]
    pal = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#12AD78", "P4": "#9860D1"}
    xs = np.arange(len(prov))
    ax.bar(xs, [100 * g["fraction"] for g in prov], color=[pal[g["programme"]] for g in prov], width=0.62)
    for x_, g in zip(xs, prov):
        ax.text(x_, 100 * g["fraction"] + 1, f"{g['memory_positive']}/{g['n']}", ha="center", fontsize=5.6)
    ax.set_xticks(xs, [g["programme"] for g in prov])
    ax.set_ylim(0, 50), ax.set_ylabel("Clones with ≥1 memory B cell (%)")
    ax.set_title("By programme (Fig. 3b)")
    pg.save("Supplementary_Figure_3_state_diversity")


# --------------------------------------------------------------------------- Supplementary Figure 4


def path_positions(d: pd.DataFrame) -> tuple[pd.Series, str]:
    """Graph distance from the GC programme along a 15-nearest-neighbour graph of the Figure 3 profiles.
    Root: the P1 clone with >= 99% GC B cells closest to the P1 centroid. Scaled to [0, 1]."""
    f = pd.read_csv(REV / "gut" / "features_donor.csv.gz", index_col=0).loc[d.index]
    x = f.to_numpy()
    dist, idx = NearestNeighbors(n_neighbors=16).fit(x).kneighbors(x)
    n = len(x)
    g = csr_matrix((dist[:, 1:].ravel(), (np.repeat(np.arange(n), 15), idx[:, 1:].ravel())), shape=(n, n))
    g = g.maximum(g.T)
    p1 = (d["region"] == "P1").to_numpy()
    cand = np.flatnonzero(p1 & (d["frac_GC"] >= 0.99).to_numpy())
    root = cand[np.argmin(((x[cand] - x[p1].mean(axis=0)) ** 2).sum(axis=1))]
    dd = dijkstra(g, indices=root)
    return pd.Series(dd / dd[np.isfinite(dd)].max(), index=d.index), d.index[root]


PATH_LINES = [("frac_GC", "GC B cells", "#2a78d6", "-"), ("frac_plasma", "Plasma cells", "#4a3aa7", "-"),
              ("frac_LP", "Lamina propria", "#b5523b", "-"), ("frac_IgA", "IgA", "#e0a33a", "-"),
              ("frac_IgG1", "IgG1", "#8f86cf", "-"), ("nonGC", "NonGC-sorted", "#111111", (0, (3, 1.5)))]


def path_panel(ax, d, regions, title):
    s = d[d["region"].isin(regions)].copy()
    s["bin"] = pd.qcut(s["path"], 10, labels=False)
    t = s.groupby("bin").agg(path=("path", "median"), region=("region", lambda v: v.value_counts().index[0]),
                             **{k: (k, "mean") for k, *_ in PATH_LINES})
    for k, lab, col, ls in PATH_LINES:
        ax.plot(t["path"], t[k], color=col, ls=ls, lw=1.1, marker="o", ms=2.4, label=lab)
    for _, r in t.iterrows():
        ax.add_patch(Rectangle((r["path"] - 0.02, -0.11), 0.04, 0.05, color=REGION[r["region"]], clip_on=False,
                               lw=0))
    ax.set_xlim(0.05, 1.0), ax.set_ylim(-0.12, 1.02), ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Position along the path (scaled graph distance)")
    ax.set_ylabel("Mean proportion per clone")
    ax.set_title(title)


def figure_s4():
    d = pd.read_csv(REV / "gut" / "clone_descriptors.csv", index_col=0)
    summ = json.loads((REV / "gut" / "summary.json").read_text())
    cells = pd.read_csv(REV / "gut" / "cells.csv.gz", index_col=0)
    d["nonGC"] = 1 - d["frac_GCsort"]
    d["path"], root = path_positions(d)
    d[["path"]].to_csv(REV / "gut" / "path_positions.csv")
    pg = Page(292)
    rng = np.random.default_rng(1)
    order = rng.permutation(len(d))

    def emb(ax, colours, title, sizes=None):
        ax.scatter(d["x"].to_numpy()[order], d["y"].to_numpy()[order],
                   s=size_of(d["n_cells"].to_numpy()[order] if sizes is None else sizes[order]),
                   c=np.asarray(colours, dtype=object)[order], linewidths=0, rasterized=True)
        embedding_frame(ax, title)

    # a, regions
    ax = pg.ax(14, 12, 52, 44, "a")
    emb(ax, d["region"].map(REGION), "Regions of the Figure 3h embedding")
    legend_below(ax, [(r, REGION[r]) for r in REGION_ORDER], ncol=2)
    # b, state span
    ax = pg.ax(80, 12, 52, 44, "b")
    emb(ax, d["state_span"].map(SPAN), "Cell states within each clone")
    legend_below(ax, [(k, SPAN[k]) for k in ("GC B only", "GC B and plasma", "plasma only", "other")], ncol=2)
    # c, tissue span
    ax = pg.ax(146, 12, 52, 44, "c")
    o2 = np.argsort(d["path"].to_numpy())
    sc = ax.scatter(d["x"].to_numpy()[o2], d["y"].to_numpy()[o2], s=size_of(d["n_cells"].to_numpy()[o2]),
                    c=d["path"].to_numpy()[o2], cmap="cividis", linewidths=0, rasterized=True)
    ax.scatter([d.loc[root, "x"]], [d.loc[root, "y"]], marker="*", s=40, c="#e8352e", linewidths=0, zorder=3)
    embedding_frame(ax, "Path from the GC programme")
    cax = pg.ax(160, 61, 26, 2.0)
    cb = pg.fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4), cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.set_label("Scaled graph distance (star, start)", fontsize=6.0, labelpad=2)

    # d, e, composition along the path to each plasma-cell branch
    ax = pg.ax(14, 84, 58, 44, "d")
    path_panel(ax, d, ["P1", "P2 bridge", "P3"], "Path to the upper branch (P3)")
    ax = pg.ax(84, 84, 58, 44, "e")
    path_panel(ax, d, ["P1", "P2 bridge", "P2 lower"], "Path to the lower branch (P2 lower)")
    lg_ax = pg.ax(14, 141, 128, 6)
    lg_ax.set_axis_off()
    lg_ax.legend(handles=[Line2D([], [], color=c, ls=ls, lw=1.1, marker="o", ms=2.4, label=l)
                          for _, l, c, ls in PATH_LINES] +
                 [Patch(color=REGION[r], label=r) for r in REGION_ORDER],
                 loc="center", ncol=5, handlelength=1.4, columnspacing=0.9)

    # f, isotype composition of GC B cells and plasma cells (cells)
    ax = pg.ax(160, 88, 36, 28, "f", dx=-12)
    cc = cells.dropna(subset=["isotype_fine"])
    cc = cc[cc["celltype"].isin(["GC_Bcell", "plasma"])]
    iso = cc["isotype_fine"].where(cc["isotype_fine"].isin(["IgA", "IgG1", "IgG2b", "IgM", "IgD"]), "other")
    tab = pd.crosstab(cc["celltype"], iso, normalize="index").reindex(columns=list(ISO), fill_value=0)
    left = np.zeros(2)
    for k in ISO:
        ax.barh([1, 0], tab[k].to_numpy() * 100, left=left, color=ISO[k], height=0.6, label=k)
        left += tab[k].to_numpy() * 100
    ax.set_yticks([1, 0], [f"GC B cells\n({int((cc.celltype == 'GC_Bcell').sum()):,})",
                           f"Plasma cells\n({int((cc.celltype == 'plasma').sum()):,})"])
    ax.set_xlabel("Isotyped cells (%)"), ax.set_xlim(0, 100)
    ax.set_title("Isotype by cell state")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, handlelength=0.9, columnspacing=0.8)

    # g, state span by region
    comp = summ["region_composition"]
    ax = pg.ax(14, 160, 52, 40, "g")
    bottom = np.zeros(4)
    for k in ("GC B only", "GC B and plasma", "plasma only", "other"):
        vals = np.array([comp[r]["state_span"].get(k, 0) / comp[r]["n_clones"] for r in REGION_ORDER]) * 100
        ax.bar(np.arange(4), vals, bottom=bottom, color=SPAN[k], width=0.66, label=k)
        bottom += vals
    ax.set_xticks(np.arange(4), [f"{r}\n({comp[r]['n_clones']})" for r in REGION_ORDER])
    ax.set_ylabel("Clones (%)"), ax.set_ylim(0, 100)
    ax.set_title("Cell states within clones, by region")

    # h, tissue span and IgG1 + IgA co-occurrence by region
    ax = pg.ax(80, 160, 52, 40, "h")
    span = [comp[r]["tissue_span"] / comp[r]["n_clones"] * 100 for r in REGION_ORDER]
    both = [comp[r]["iga_igg1"] / comp[r]["n_clones"] * 100 for r in REGION_ORDER]
    xx = np.arange(4)
    ax.bar(xx - 0.2, span, width=0.38, color="#b5523b", label="Spans PP/MLN and LP")
    ax.bar(xx + 0.2, both, width=0.38, color="#4a3aa7", label="Contains IgG1 and IgA")
    ax.set_xticks(xx, REGION_ORDER)
    ax.set_ylabel("Clones (%)"), ax.set_ylim(0, 100)
    ax.set_title("Tissue and isotype relationships")
    ax.legend(loc="upper right", handlelength=1.0)

    # i, size-matched comparison
    ax = pg.ax(146, 160, 52, 40, "i")
    strata = [(2, 2, "2 cells"), (3, 5, "3–5 cells"), (6, 10 ** 6, "≥6 cells")]
    w = 0.2
    for j, r in enumerate(REGION_ORDER):
        vals, ns = [], []
        for lo, hi, _ in strata:
            s = d[(d["region"] == r) & (d["n_cells"] >= lo) & (d["n_cells"] <= hi)]
            vals.append((s["state_span"] == "GC B and plasma").mean() * 100)
            ns.append(len(s))
        ax.bar(np.arange(3) + (j - 1.5) * w, vals, width=w * 0.95, color=REGION[r], label=r)
    ax.set_xticks(np.arange(3), [s[2] for s in strata])
    ax.set_ylabel("Clones with GC B and plasma cells (%)"), ax.set_ylim(0, 105)
    ax.set_title("Comparison at matched clone size")
    ax.legend(loc="upper left", ncol=2, handlelength=0.9, columnspacing=0.6)

    # j, profile associations
    ax = pg.ax(52, 230, 60, 50, "j", dx=-38)
    assoc = summ["associations"]
    kind = {"Plasma-cell proportion": "RNA", "GC B-cell proportion": "RNA",
            "Clone contains GC B and plasma cells": "RNA", "Lamina-propria proportion": "sampling",
            "Peyer's-patch proportion": "sampling", "Clone spans PP/MLN and LP": "sampling",
            "IgA proportion": "sequence", "IgG1 proportion": "sequence", "V-region mutation frequency": "sequence",
            "GC-sorted proportion": "sorted"}
    kcol = {"RNA": "#9a9890", "sampling": "#b5523b", "sequence": "#4a3aa7", "sorted": "#2fa4a0"}
    items = sorted(assoc.items(), key=lambda kv: kv[1]["excess_r2"])
    yy = np.arange(len(items))
    ax.barh(yy, [v["excess_r2"] for _, v in items], color=[kcol[kind[k]] for k, _ in items], height=0.66)
    ax.set_yticks(yy, [k for k, _ in items])
    ax.set_xlabel("Excess R$^2$ (2,000 permutations within mice)")
    ax.set_title("Annotations associated with clone profiles")
    for i, (_, v) in enumerate(items):
        ax.text(v["excess_r2"] + 0.01, i, f"{v['excess_r2']:.3f}", va="center", fontsize=5.6)
    ax.set_xlim(0, 0.72)
    ax.legend(handles=[Patch(color=kcol[k], label=k) for k in ("RNA", "sampling", "sequence", "sorted")],
              loc="lower right", title="Annotation type", handlelength=0.9)

    # k, GC history by region
    ax = pg.ax(140, 230, 56, 50, "k")
    bottom = np.zeros(4)
    for k in ("all GC-sorted", "mixed", "all nonGC"):
        vals = np.array([comp[r]["gc_history"].get(k, 0) / comp[r]["n_clones"] for r in REGION_ORDER]) * 100
        ax.bar(np.arange(4), vals, bottom=bottom, color=HIST[k], width=0.66, label=k)
        bottom += vals
    ax.set_xticks(np.arange(4), REGION_ORDER)
    ax.set_ylabel("Clones (%)"), ax.set_ylim(0, 124), ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title("GC-fate sort of clone members")
    ax.legend(loc="upper center", ncol=3, handlelength=0.9, columnspacing=0.7, borderaxespad=0.2)

    # shared legend for g
    fig_ax = pg.ax(14, 208, 52, 6)
    fig_ax.set_axis_off()
    fig_ax.legend(handles=[Patch(color=SPAN[k], label=k) for k in ("GC B only", "GC B and plasma",
                                                                    "plasma only", "other")],
                  loc="center", ncol=2, handlelength=0.9, columnspacing=0.8)
    pg.save("Supplementary_Figure_4_gut_clone_relationships")


# --------------------------------------------------------------------------- Supplementary Figure 7


def figure_s7():
    d = pd.read_csv(REV / "gut" / "clone_descriptors.csv", index_col=0)
    cells = pd.read_csv(REV / "gut" / "cells.csv.gz", index_col=0)
    summ = json.loads((REV / "gut" / "summary.json").read_text())
    pg = Page(296)
    rng = np.random.default_rng(1)

    # a, cell UMAP by mouse
    ax = pg.ax(14, 12, 60, 56, "a")
    o = rng.permutation(len(cells))
    ax.scatter(cells["cell_umap_1"].to_numpy()[o], cells["cell_umap_2"].to_numpy()[o], s=0.25,
               c=cells["donor"].map(MOUSE).to_numpy()[o], linewidths=0, rasterized=True)
    embedding_frame(ax, "Cells coloured by mouse", "Cell UMAP 1", "Cell UMAP 2")
    (x0, x1), (y0, y1) = (5.1087474434, 16.118153176599996), (-10.470856728600001, 0.5385490045999974)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls=(0, (3, 2)), lw=0.6, ec=INK2))
    legend_points(ax, [(MOUSE_LABEL[m], MOUSE[m]) for m in MOUSE], loc="upper left", ms=3.5)
    ins = pg.ax(76, 12, 26, 26)
    sel = (cells["cell_umap_1"].between(x0, x1) & cells["cell_umap_2"].between(y0, y1)).to_numpy()
    oo = o[sel[o]]
    ins.scatter(cells["cell_umap_1"].to_numpy()[oo], cells["cell_umap_2"].to_numpy()[oo], s=0.6,
                c=cells["donor"].map(MOUSE).to_numpy()[oo], linewidths=0, rasterized=True)
    ins.set_xlim(x0, x1), ins.set_ylim(y0, y1), ins.set_xticks([]), ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True), sp.set_linewidth(0.5)
    ins.set_title("Plasma-cell region", fontsize=6.2)

    # b, c, composition per mouse
    comp = summ["mouse_composition"]
    for k, (key, pal, lab, title) in enumerate((("celltype", CELLTYPE, CELLTYPE_LABEL, "Cell types"),
                                                ("tissue", TISSUE, {"PP": "Peyer's patch", "MLN": "MLN",
                                                                    "LP": "Lamina propria"}, "Tissue"))):
        ax = pg.ax(124 + 38 * k, 12, 26, 52, "bc"[k])
        tab = pd.DataFrame(comp[key]).T.reindex(index=list(MOUSE)).fillna(0)
        tab = tab[[c for c in pal if c in tab.columns]]
        frac = tab.div(tab.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(frac))
        for c in frac.columns:
            ax.bar(np.arange(len(frac)), frac[c], bottom=bottom, color=pal[c], width=0.7, label=lab[c])
            bottom += frac[c].to_numpy()
        ax.set_xticks(np.arange(3), ["M2", "M3", "M4"])
        ax.set_ylim(0, 100), ax.set_ylabel("Cells (%)")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, handlelength=0.9)

    # d, Figure 3h embedding by mouse; e, f, g, other centrings
    o = rng.permutation(len(d))

    def emb(ax, xcol, ycol, colours, title):
        ax.scatter(d[xcol].to_numpy()[o], d[ycol].to_numpy()[o], s=size_of(d["n_cells"].to_numpy()[o]),
                   c=np.asarray(colours, dtype=object)[o], linewidths=0, rasterized=True)
        embedding_frame(ax, title)

    mice_items = [(MOUSE_LABEL[m], MOUSE[m]) for m in MOUSE]
    ax = pg.ax(14, 98, 52, 42, "d")
    emb(ax, "x", "y", d["donor"].map(MOUSE), "Figure 3h (centred on each mouse)")
    legend_below(ax, mice_items, ncol=3)
    ax = pg.ax(80, 98, 52, 42, "e")
    emb(ax, "x_none", "y_none", d["region"].map(REGION), "No centring, by Figure 3h region")
    legend_below(ax, [(r, REGION[r]) for r in REGION_ORDER], ncol=4)
    ax = pg.ax(146, 98, 52, 42, "f")
    emb(ax, "x_none", "y_none", d["donor"].map(MOUSE), "No centring, by mouse")
    legend_below(ax, mice_items, ncol=3)

    ax = pg.ax(14, 168, 52, 42, "g")
    emb(ax, "x_balanced", "y_balanced", d["donor"].map(MOUSE), "Composition-balanced centring")
    legend_below(ax, mice_items, ncol=3)

    # h, programme stability under each centring
    ax = pg.ax(80, 168, 52, 42, "h")
    variants = [("Centred on mouse", summ["programmes_donor"]), ("No centring", summ["programmes_none"]),
                ("Balanced", summ["programmes_balanced"])]
    xpos = 0
    ticks, labels = [], []
    for name, pr in variants:
        st = pr["stability"]
        xs = xpos + np.arange(len(st))
        ax.bar(xs, list(st.values()), width=0.75, color=["#7f7e78"] * len(st))
        for x_, (k, v) in zip(xs, st.items()):
            ax.text(x_, v + 0.02, f"{k}\n{pr['n_clones'][k]}", ha="center", va="bottom", fontsize=5.0)
        ticks.append(xs.mean()), labels.append(name)
        xpos = xs[-1] + 2
    ax.axhline(0.75, ls=(0, (3, 2)), lw=0.6, color=INK2)
    ax.set_xticks(ticks, labels)
    ax.set_ylim(0, 1.18), ax.set_ylabel("Bootstrap Jaccard index")
    ax.set_title("Programmes under each centring")

    # i, same-mouse neighbours: cells versus clones
    ax = pg.ax(146, 168, 52, 42, "i")
    rows = [("Plasma cells\n(cell embedding)", summ["mixing_cells_embedding"]),
            ("Clones, centred\non mouse", summ["mixing_clones_donor"]),
            ("Clones, no\ncentring", summ["mixing_clones_none"]),
            ("Clones, balanced\ncentring", summ["mixing_clones_balanced"])]
    xx = np.arange(len(rows))
    ax.bar(xx, [r["observed"] for _, r in rows], width=0.6, color=["#4a3aa7", "#E9AE14", "#9a9890", "#9a9890"])
    ax.scatter(xx, [r["null_mean"] for _, r in rows], marker="_", s=120, color=INK, linewidths=1.0, zorder=3)
    for i, (_, r) in enumerate(rows):
        ax.text(i, r["observed"] + 0.02, f"{r['observed']:.2f}", ha="center", fontsize=5.6)
    ax.set_xticks(xx, [r[0] for r in rows], fontsize=5.4)
    ax.set_ylim(0, 1), ax.set_ylabel("Same-mouse fraction of 15 neighbours")
    ax.set_title("Separation of mice: cells versus clones")
    ax.legend(handles=[Line2D([], [], ls="", marker="_", ms=8, mec=INK, label="Permutation mean")],
              loc="upper right", handlelength=1.0)

    # j, nonGC share of plasma cells per region: observed versus mouse composition
    ax = pg.ax(14, 238, 52, 40, "j")
    regs = ["P2 bridge", "P2 lower", "P3"]
    sh = summ["nonGC_share_by_region"]
    xx = np.arange(3)
    ax.bar(xx - 0.2, [100 * sh[r]["observed_nonGC_share"] for r in regs], width=0.38, color="#2b1d73",
           label="Observed")
    ax.bar(xx + 0.2, [100 * sh[r]["expected_from_mouse"] for r in regs], width=0.38, color="#a59fd8",
           label="Expected from mouse mix")
    ax.set_xticks(xx, [f"{r}\n({sh[r]['n_clones']})" for r in regs])
    ax.set_ylabel("NonGC-sorted plasma cells (%)"), ax.set_ylim(0, 24)
    ax.set_title("GC-fate sort of plasma cells by region")
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8)

    # k, within-mouse GC-history neighbourhoods under each centring
    ax = pg.ax(80, 238, 70, 40, "k")
    wm = summ["within_mouse_nonGC"]
    w = 0.26
    for j, (v, col, lab) in enumerate((("donor", "#E9AE14", "Centred on mouse"), ("none", "#9a9890", "No centring"),
                                        ("balanced", "#cfcdc4", "Balanced"))):
        for i, m in enumerate(["mouse2", "mouse3", "mouse4"]):
            r = wm[v][m]
            ax.bar(i + (j - 1) * w, r["rho"], width=w * 0.95, color=col, label=lab if i == 0 else None)
            ax.plot([i + (j - 1) * w - w / 2.2, i + (j - 1) * w + w / 2.2], [r["null_q95"]] * 2, color=INK,
                    lw=0.7)
    ax.set_xticks(np.arange(3), [f"{MOUSE_LABEL[m]}\n({wm['donor'][m]['n_with_nonGC_plasma']}/"
                                 f"{wm['donor'][m]['n_clones']} clones)" for m in ["mouse2", "mouse3", "mouse4"]])
    ax.set_ylabel("Spearman ρ, nonGC share vs\nmean of 10 nearest clones")
    ax.set_title("GC-fate sort within each mouse")
    ax.set_ylim(0, 0.98)
    hd, lb = ax.get_legend_handles_labels()
    hd.append(Line2D([], [], color=INK, lw=0.8)), lb.append("95th percentile, 2,000 permutations")
    ax.legend(hd, lb, loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.8)

    # l, plasma-cell-dominated clones per region and mouse
    ax = pg.ax(168, 238, 28, 40, "l")
    tab = pd.DataFrame(summ["region_by_mouse_plasma_dominated"]).T.reindex(["P2 bridge", "P2 lower", "P3"])
    bottom = np.zeros(3)
    for m in MOUSE:
        ax.bar(np.arange(3), tab[m].to_numpy(), bottom=bottom, color=MOUSE[m], width=0.7)
        bottom += tab[m].to_numpy()
    ax.set_xticks(np.arange(3), ["Bridge", "Lower", "P3"], fontsize=5.6)
    ax.set_ylabel("Plasma-cell-dominated clones"), ax.set_ylim(0, 240)
    ax.set_title("Mice per region")
    legend_points(ax, [(MOUSE_LABEL[m], MOUSE[m]) for m in MOUSE], loc="upper left", ms=3.5)
    pg.save("Supplementary_Figure_5_gut_centring")


if __name__ == "__main__":
    figure_s3()
    figure_s4()
    figure_s7()
