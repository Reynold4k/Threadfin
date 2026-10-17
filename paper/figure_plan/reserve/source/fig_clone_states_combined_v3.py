#!/usr/bin/env python
"""Refine panel m with explicit quantile whiskers and offset raw observations."""
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
from scipy import stats

import figstyle as fs
from fig_clone_states_combined import STATE_COLORS, M_LABELS, counts, describe, emit, native, draw_categories

HERE = Path(__file__).resolve().parents[2]
CASE = HERE.parent.parent / "case_studies" / "results"
GRAY = "#e7e6e0"
PROG_COLORS = {
    "S1": {"P1": "#8053B5", "P2": "#EC4776", "P3": "#00A8A5", "P4": "#E2A13B"},
    "S2": {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#12AD78", "P4": "#9860D1"},
    "S3": {"P1": "#405BD8", "P2": "#E9AE14", "P3": "#E44336"},
}
GSE = {"S1": "GSE286215", "S2": "GSE203132", "S3": "GSE284622"}
LAYOUTS = {
    "compact": {"height": 356, "rows": [8, 79, 153, 220, 291]},
    "balanced": {"height": 362, "rows": [9, 81, 155, 225, 296]},
    "open": {"height": 382, "rows": [9, 85, 166, 240, 315]},
    "filled": {"height": 356, "rows": [8, 79, 153, 220, 291]},
}
BOXES = {}
LEGEND_BOXES = {}
LEGEND_TEXTS = []


def at(fig, x, top, w, h, letter=None):
    """Absolute page millimetres, using the read-only figstyle panel helper."""
    return fs.panel(fig, x - fs.LEFT, top - fs.TOP, w, h, letter, letter_dx=-4.0)


def make_ax(fig, x, top, w, h, letter, title="", *, coordinates=None, key=None):
    ax = at(fig, x, top, w, h, letter)
    BOXES[key or letter] = [x, top, w, h]
    if title:
        ax.set_title(title, fontsize=6.6, pad=3.5,
                     fontweight="bold" if title.startswith("GSE") else "normal")
    if coordinates:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(coordinates[0], fontsize=6.0, labelpad=3)
        ax.set_ylabel(coordinates[1], fontsize=6.0, labelpad=3)
    ax.tick_params(labelsize=5.3, pad=2)
    return ax


def square_limits(table, columns):
    a = table[columns].to_numpy(dtype=float)
    lo, hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
    center = (lo + hi) / 2
    radius = np.max(hi - lo) * .555
    return [(float(c - radius), float(c + radius)) for c in center]


def finish_map(ax, limits):
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")


def left_key(fig, key, top, series, palette, title):
    items = [(name, color, int(series.eq(name).sum())) for name, color in palette.items()
             if series.eq(name).any()]
    pitch = 4.45 if len(items) > 8 else 5.5
    total_h = 3.8 + len(items) * pitch
    y = top + (53 - total_h) / 2
    ax = at(fig, 2, y, 23.5, total_h)
    ax.set_axis_off()
    ax.set_xlim(0, 23.5)
    ax.set_ylim(total_h, 0)
    LEGEND_BOXES[key] = [2, y, 23.5, total_h]
    LEGEND_TEXTS.append(ax.text(.1, .1, title, fontsize=5.8, fontweight="bold", va="top"))
    for i, (name, color, n) in enumerate(items):
        yy = 3.4 + pitch * i
        ax.scatter([.9], [yy + .45], s=10, color=color, linewidths=0, clip_on=False)
        LEGEND_TEXTS.append(ax.text(2.4, yy, name, fontsize=5.5, va="top"))
        LEGEND_TEXTS.append(ax.text(2.4, yy + 2.1, f"n = {n:,}", fontsize=4.9,
                                   va="top", color=fs.INK2))


def shared_key(fig, key, top, palette):
    items = [*palette.items(), ("no programme", GRAY)]
    ax = at(fig, 91, top + 60, 113, 7.2)
    ax.set_axis_off()
    ax.set_xlim(0, 113)
    ax.set_ylim(7.2, 0)
    LEGEND_BOXES[key] = [91, top + 60, 113, 7.2]
    LEGEND_TEXTS.append(ax.text(0, .1, "Programme", fontsize=5.6, fontweight="bold", va="top"))
    step = 113 / len(items)
    for i, (label, color) in enumerate(items):
        x = i * step
        ax.scatter([x + .9], [3.7], s=11, c=color, linewidths=0, clip_on=False)
        LEGEND_TEXTS.append(ax.text(x + 2.7, 3.7, label, fontsize=5.6, va="center"))


def load_data():
    print("Stage: read frozen inputs and rename S2 programmes for display", flush=True)
    datasets, paths = {}, []
    for name, ds, fname, state in [("S1", "malaria_late", "cells.csv.gz", "cell_state"),
                                   ("S2", "spont_gc", "spont_gc_fate_cells.csv.gz", "fate"),
                                   ("S3", "gut_gc_reporter", "cells.csv.gz", "celltype")]:
        cell_path, clone_path = CASE / ds / fname, CASE / ds / "clone_table.csv"
        paths.extend([cell_path, clone_path])
        cells, clones = pd.read_csv(cell_path, index_col=0), pd.read_csv(clone_path, index_col=0)
        clones.index = clones.index.astype(str)
        assert clones.index.is_unique
        mapped = cells["clone_id"].map(clones["clone_programme"])
        if "clone_programme" in cells:
            assert cells["clone_programme"].fillna("").eq(mapped.fillna("")).all()
        datasets[name] = {"cells": cells, "clones": clones, "state": state,
                          "cell_limits": square_limits(cells, ["umap_1", "umap_2"]),
                          "clone_limits": square_limits(clones, ["x", "y"])}
    sub_path = CASE / "spont_gc" / "p3_subsplit.json"
    sub = json.loads(sub_path.read_text())
    paths.append(sub_path)
    assert sub["split_accepted"]
    s2 = datasets["S2"]
    assignment = pd.Series(sub["assignments_by_clone_id"]).map({"P3a": "P3", "P3b": "P4"})
    target = s2["clones"]["clone_programme"].eq("P3")
    assert set(assignment.index) == set(s2["clones"].index[target]) and assignment.notna().all()
    s2["clones"].loc[target, "clone_programme"] = assignment.reindex(s2["clones"].index[target]).to_numpy()
    for ds in datasets.values():
        ds["mapped"] = ds["cells"]["clone_id"].map(ds["clones"]["clone_programme"])
    wpath = CASE / "spont_gc" / "spont_gc_fate_clones.csv"
    w = pd.read_csv(wpath, index_col=0)
    w.index = w.index.astype(str)
    paths.append(wpath)
    assert w.index.is_unique and w.index.isin(s2["clones"].index).all()
    assert np.allclose(w[["x", "y"]], s2["clones"].reindex(w.index)[["x", "y"]])
    cell_has_m = s2["cells"]["fate"].isin(M_LABELS).groupby(s2["cells"]["clone_id"]).any()
    assert w.index.isin(cell_has_m.index).all()
    w["label_M"] = cell_has_m.reindex(w.index).astype(bool)
    branches_path = CASE / "gut_gc_reporter" / "plasma_branches.json"
    branches = json.loads(branches_path.read_text())
    paths.append(branches_path)
    return datasets, w, branches, paths, assignment


def validate(fig, height, layout_name):
    print("Stage: check physical dimensions, page bounds and legend text", flush=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    outside, collision = [], []
    for txt in fig.findobj(match=matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text():
            continue
        bb = txt.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > page.x1 + 1 or bb.y1 > page.y1 + 1:
            outside.append(txt.get_text())
    for i, a in enumerate(LEGEND_TEXTS):
        aa = a.get_window_extent(renderer)
        for b in LEGEND_TEXTS[i + 1:]:
            bb = b.get_window_extent(renderer)
            if aa.overlaps(bb):
                collision.append([a.get_text(), b.get_text()])
    assert not outside, outside
    assert not collision, collision
    for key in ["a", "b", "c", "d", "e", "f", "g", "j", "k", "l", "m"]:
        assert BOXES[key][2:] == [53, 53], (key, BOXES[key])
    gaps = []
    for i, (a, aa) in enumerate(BOXES.items()):
        for b, bb in list(BOXES.items())[i + 1:]:
            dx = max(bb[0] - aa[0] - aa[2], aa[0] - bb[0] - bb[2])
            dy = max(bb[1] - aa[1] - aa[3], aa[1] - bb[1] - bb[3])
            assert max(dx, dy) >= 4, (a, b, dx, dy)
            gaps.append(max(dx, dy))
    return {"name": layout_name, "page_mm": [210, height], "dpi": fs.EXPORT_DPI,
            "panel_letters": list("abcdefghijklm"), "axes_boxes_mm": BOXES,
            "legend_boxes_mm": LEGEND_BOXES, "square_scatter_size_mm": [53, 53],
            "scatter_coordinate_aspect": "equal; matched square limits per dataset and coordinate type",
            "minimum_axes_gap_mm": min(gaps), "minimum_legend_font_pt": 4.9,
            "text_outside_page": outside, "legend_text_collisions": collision}



def refined_measurement_boxplot(ax, progs, arrays, palette, pvalue):
    """Offset raw points and boxes; whiskers are exact 5th/95th quantiles."""
    from matplotlib.colors import to_rgba

    rng = np.random.default_rng(0)
    summaries = []
    positions = np.arange(len(progs)) + .15
    for i, (prog, values) in enumerate(zip(progs, arrays)):
        q05, q1, median, q3, q95 = np.quantile(values, [.05, .25, .5, .75, .95], method="linear")
        summaries.append({"programme": prog, "n": len(values), "q05": float(q05),
                          "q25": float(q1), "median": float(median), "q75": float(q3),
                          "q95": float(q95), "n_equal_1": int(np.equal(values, 1).sum()),
                          "fraction_equal_1": float(np.equal(values, 1).mean()),
                          "IQR_zero": bool(q1 == q3)})
        # Deterministic horizontal offsets retain every measured y value exactly.
        offsets = rng.uniform(-.105, .105, len(values))
        ax.scatter(i - .18 + offsets, values, s=6.0, color=palette[prog], alpha=.48,
                   linewidths=.15, edgecolors="white", rasterized=True, zorder=3)
    bxstats = [{"q1": s["q25"], "med": s["median"], "q3": s["q75"],
                "whislo": s["q05"], "whishi": s["q95"], "fliers": []}
               for s in summaries]
    artists = ax.bxp(bxstats, positions=positions, widths=.27, capwidths=.16,
                     patch_artist=True, showfliers=False, manage_ticks=False,
                     boxprops={"linewidth": .85},
                     medianprops={"color": "#252525", "linewidth": 1.2, "solid_capstyle": "butt"},
                     whiskerprops={"linewidth": .85}, capprops={"linewidth": .95}, zorder=4)
    for i, prog in enumerate(progs):
        color = palette[prog]
        artists["boxes"][i].set_facecolor(to_rgba(color, .16))
        artists["boxes"][i].set_edgecolor(color)
        for k in [2*i, 2*i+1]:
            artists["whiskers"][k].set_color(color)
            artists["caps"][k].set_color(color)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e7e9", linewidth=.35)
    ax.xaxis.grid(False)
    ax.spines["left"].set_color("#85898c")
    ax.spines["bottom"].set_color("#85898c")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2, width=.5)
    ax.set_xticks(np.arange(len(progs)), [f"{p}\n(n={len(v)})" for p, v in zip(progs, arrays)])
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlim(-.5, len(progs) - .5)
    ax.set_ylim(-.06, 1.43)
    ax.set_xlabel("Programme", fontsize=6, labelpad=3)
    ax.set_ylabel("gc_sort:GC", fontsize=6, labelpad=3)
    ax.text(.015, .975, "celltype:plasma ≥ 0.5", transform=ax.transAxes,
            fontsize=5.3, ha="left", va="top", color="#44484b")
    ax.text(.985, .975, "Box: Q1–Q3; line: median\nWhiskers: 5th–95th percentiles",
            transform=ax.transAxes, fontsize=4.9, ha="right", va="top", color="#565b5e",
            linespacing=1.3)
    p2x, p3x = positions[progs.index("P2")], positions[progs.index("P3")]
    ax.plot([p2x, p2x, p3x, p3x], [1.10, 1.145, 1.145, 1.10],
            color="#666b6e", lw=.65, clip_on=False, zorder=5)
    exponent = int(np.floor(np.log10(pvalue)))
    mantissa = pvalue / 10**exponent
    label = rf"P2–P3: $p = {mantissa:.2f}\times10^{{{exponent}}}$"
    ax.text((p2x+p3x)/2, 1.17, label, fontsize=5.5, ha="center", va="bottom", color="#303437")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=LAYOUTS, default="filled")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    layout = LAYOUTS[args.layout]
    height, rows = layout["height"], layout["rows"]
    wide_rows = args.layout == "filled"
    datasets, w, branches, paths, assignment = load_data()
    report = {"input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "panels": {}, "programme_colors": PROG_COLORS, "state_colors": STATE_COLORS,
              "S2_display_rename": {"P3a": "P3", "P3b": "P4"},
              "S2_display_assignments_by_clone_id": assignment.to_dict(),
              "label_M_definition": {"fate_values": M_LABELS, "minimum_cells_per_clone": 1},
              "analysis_population": {"S2_g_h_i": "322 existing fate-clone rows",
                                      "S2_e": "1051 existing clone-table rows",
                                      "S3_m_ii": "all clone-table rows with celltype:plasma >= 0.5"},
              "shared_programme_keys": {"b_c": "S1", "e_f": "S2", "k_l": "S3"},
              "legend_counts": "left keys show cell counts; shared programme keys use category names only",
              "test_methods": {"correlations": "two-sided scipy.stats.spearmanr",
                               "S3_P2_P3": "two-sided scipy.stats.mannwhitneyu, method=auto"}}

    print(f"Stage: draw {args.layout} page, 210 x {height} mm", flush=True)
    fig = fs.new_page("", "", height=height)
    for txt in list(fig.texts):
        txt.remove()
    for name, letters, top in [("S1", "abc", rows[0]), ("S2", "def", rows[1]), ("S3", "jkl", rows[3])]:
        ds = datasets[name]
        cells, clones, state, mapped = (ds[k] for k in ("cells", "clones", "state", "mapped"))
        pal = PROG_COLORS[name]
        ax = make_ax(fig, 31, top, 53, 53, letters[0], GSE[name], coordinates=("Cell UMAP 1", "Cell UMAP 2"))
        draw_categories(ax, cells, cells[state], STATE_COLORS[name], ["umap_1", "umap_2"], .35, alpha=.7)
        finish_map(ax, ds["cell_limits"])
        left_key(fig, letters[0], top, cells[state], STATE_COLORS[name], state)
        emit(letters[0], {"n_cells": len(cells), "category_counts": counts(cells[state]),
                          "n_missing_coordinates": int(cells[["umap_1", "umap_2"]].isna().any(axis=1).sum())}, report)
        ax = make_ax(fig, 91, top, 53, 53, letters[1], "Clone map", coordinates=("Clone x", "Clone y"))
        scale = {"S1": 1.5, "S2": .35, "S3": .30}[name]
        draw_categories(ax, clones, clones["clone_programme"], pal, ["x", "y"], clones["n_cells"].to_numpy() * scale, alpha=.92)
        finish_map(ax, ds["clone_limits"])
        emit(letters[1], {"n_clones": len(clones), "category_counts": counts(clones["clone_programme"]),
                          "n_cells_sum": int(clones["n_cells"].sum()), "n_cells": describe(clones["n_cells"]),
                          "marker_area_pt2_per_cell": scale}, report)
        ax = make_ax(fig, 151, top, 53, 53, letters[2], "Map-back", coordinates=("Cell UMAP 1", "Cell UMAP 2"))
        draw_categories(ax, cells, mapped, pal, ["umap_1", "umap_2"], .42, alpha=.88)
        finish_map(ax, ds["cell_limits"])
        shared_key(fig, f"{letters[1]}_{letters[2]}", top, pal)
        emit(letters[2], {"n_cells": len(cells), "category_counts": counts(mapped),
                          "n_labelled": int(mapped.notna().sum()), "n_without_programme": int(mapped.isna().sum()),
                          "n_missing_clone_id": int(cells["clone_id"].isna().sum()),
                          "n_nonmissing_clone_id_not_in_clone_table": int((cells["clone_id"].notna() & ~cells["clone_id"].isin(clones.index)).sum())}, report)

    print("Stage: draw entropy, quartiles and donor correlations", flush=True)
    top = rows[2]
    gx, cx = (7, 62) if wide_rows else (31, 86)
    ax = make_ax(fig, gx, top, 53, 53, "g", "Entropy", coordinates=("Clone x", "Clone y"))
    sc = ax.scatter(w["x"], w["y"], s=w["n_cells"] * .45, c=w["entropy"], cmap="viridis",
                    linewidths=0, alpha=.9, rasterized=True)
    finish_map(ax, datasets["S2"]["clone_limits"])
    rho = stats.spearmanr(w["entropy"], w["x"])
    ax.text(.035, .965, f"Spearman ρ(entropy, x)\n{rho.statistic:.3f}; n = {len(w)}", transform=ax.transAxes,
            va="top", fontsize=5.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1.5})
    cax = at(fig, cx, top + 7, 1.9, 40)
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("entropy", fontsize=5.2, pad=3)
    cb.ax.tick_params(labelsize=5, pad=1.5)
    cb.set_ticks([0, .5, 1])
    emit("g", {"n_clones": len(w), "entropy": describe(w["entropy"]),
               "spearman_rho": rho.statistic, "spearman_p": rho.pvalue}, report)

    hx, hw = (79, 53) if wide_rows else (101, 47)
    ax = make_ax(fig, hx, top, hw, 53, "h", "label-M")
    quartiles, edges = pd.qcut(w["x"], 4, labels=["Q1", "Q2", "Q3", "Q4"], retbins=True)
    qrows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        ww = w[quartiles.eq(q)]
        qrows.append({"quartile": q, "n": len(ww), "label_M_n": int(ww["label_M"].sum()),
                      "label_M_fraction": ww["label_M"].mean(), "x_min": ww["x"].min(), "x_max": ww["x"].max()})
    ax.bar(np.arange(4), [r["label_M_fraction"] for r in qrows], width=.64, color="#58778B", linewidth=0)
    for i, r in enumerate(qrows):
        ax.text(i, r["label_M_fraction"] + .015, f"{r['label_M_fraction']:.1%}\nn={r['n']}",
                ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylim(0, .49)
    ax.set_yticks([0, .2, .4], ["0%", "20%", "40%"])
    ax.set_ylabel("Clones with label-M (%)", fontsize=5.8, labelpad=3)
    ax.set_xlabel("Clone x quartile", fontsize=5.8, labelpad=3)
    emit("h", {"n_clones": len(w), "qcut_edges": edges, "quartiles": qrows,
               "label_M_n": int(w["label_M"].sum())}, report)

    ix, iw = (145, 63) if wide_rows else (158, 50)
    ax = make_ax(fig, ix, top, iw, 53, "i", "Donor correlations")
    donor_rows = []
    for donor, ww in w.groupby("donor"):
        rr = stats.spearmanr(ww["entropy"], ww["x"])
        donor_rows.append({"donor": donor, "n": len(ww), "rho": rr.statistic, "p": rr.pvalue})
    donor_rows.sort(key=lambda d: d["rho"])
    ax.barh(np.arange(len(donor_rows)), [d["rho"] for d in donor_rows], height=.58, color="#58778B", linewidth=0)
    for i, d in enumerate(donor_rows):
        ax.text(d["rho"] - .03, i, f"{d['donor']} (n={d['n']})", ha="right", va="center", fontsize=5.2)
    ax.axvline(0, color=fs.INK2, linewidth=.5)
    ax.set_yticks([])
    ax.set_ylim(-.7, len(donor_rows) - .3)
    ax.invert_yaxis()
    ax.set_xlim(-1.62, .06)
    ax.set_xticks([-1, -.5, 0])
    ax.set_ylabel("Donor", fontsize=5.8, labelpad=3)
    ax.set_xlabel("Spearman ρ(entropy, x)", fontsize=5.8, labelpad=3)
    emit("i", {"n_clones": len(w), "donors_sorted_by_rho": donor_rows}, report)

    print("Stage: draw S3 continuous map and grouped measurement", flush=True)
    t = datasets["S3"]["clones"]
    measured = pd.to_numeric(t["gc_sort:GC"], errors="coerce")
    top = rows[4]
    ax = make_ax(fig, gx, top, 53, 53, "m", "gc_sort:GC", coordinates=("Clone x", "Clone y"))
    good = measured.notna()
    if (~good).any():
        ax.scatter(t.loc[~good, "x"], t.loc[~good, "y"], s=t.loc[~good, "n_cells"] * .30,
                   c=GRAY, linewidths=0, rasterized=True)
    sc = ax.scatter(t.loc[good, "x"], t.loc[good, "y"], s=t.loc[good, "n_cells"] * .30,
                    c=measured[good], cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
    finish_map(ax, datasets["S3"]["clone_limits"])
    cax = at(fig, cx, top + 7, 1.9, 40)
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("gc_sort:GC", fontsize=5.1, pad=3)
    cb.ax.tick_params(labelsize=5, pad=1.5)
    cb.set_ticks([0, .5, 1])

    bx, bw = (95, 109) if wide_rows else (113, 86)
    ax = make_ax(fig, bx, top, bw, 53, None, "gc_sort:GC by programme", key="m_ii")
    selected = t[t["celltype:plasma"].ge(.5)].copy()
    progs = sorted(selected["clone_programme"].dropna().unique())
    arrays = [selected.loc[selected["clone_programme"].eq(p), "gc_sort:GC"].dropna().to_numpy() for p in progs]
    group_stats = {p: describe(selected.loc[selected["clone_programme"].eq(p), "gc_sort:GC"]) for p in progs}
    p2, p3 = arrays[progs.index("P2")], arrays[progs.index("P3")]
    test = stats.mannwhitneyu(p2, p3, alternative="two-sided")
    expected_p = branches["branch_comparison"]["gc_sort"]["p"]
    assert np.isclose(test.pvalue, expected_p, rtol=1e-12, atol=0)
    box_summary = refined_measurement_boxplot(ax, progs, arrays, PROG_COLORS["S3"], test.pvalue)
    emit("m", {"map": {"n_clones": len(t), "gc_sort:GC": describe(measured)},
               "comparison": {"threshold_column": "celltype:plasma", "threshold_minimum": .5,
                              "n_all_programmes": len(selected), "programmes": group_stats,
                              "P2_P3_n": len(p2) + len(p3), "U_P2_vs_P3": test.statistic,
                              "p_two_sided": test.pvalue, "reference_json_p": expected_p,
                              "reference_json_matches": True,
                              "display": "offset raw points and tinted boxes; exact 5th/95th quantile whiskers",
                              "boxplot_summary": box_summary,
                              "quantile_method": "numpy.quantile(method='linear')",
                              "whisker_rule": "exact empirical quantiles at 0.05 and 0.95; box Q1-Q3; median line; no vertical jitter"}}, report)

    report["layout"] = validate(fig, height, args.layout)
    report["panels"] = {letter: report["panels"][letter] for letter in "abcdefghijklm"}
    print("Layout:", json.dumps(native(report["layout"]), ensure_ascii=False), flush=True)
    if args.preview:
        path = Path(f"/tmp/clone_states_v3_{args.layout}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print("Preview:", path, flush=True)
    else:
        print("Stage: export revised PNG/PDF at 1200 dpi", flush=True)
        fs.save(fig, "Figure_clone_states_combined_v3", folder=HERE)
        print("Stage: export panel m boxplot separately at 1200 dpi", flush=True)
        detail = plt.figure(figsize=(124 * fs.MM, 72 * fs.MM))
        detail_ax = detail.add_axes([.11, .19, .87, .735])
        detail_ax.set_title("gc_sort:GC by programme", fontsize=6.6, pad=3.5)
        detail_ax.tick_params(labelsize=5.3, pad=2)
        refined_measurement_boxplot(detail_ax, progs, arrays, PROG_COLORS["S3"], test.pvalue)
        fs.save(detail, "Figure_clone_states_combined_v3_boxplot", folder=HERE)
        out = HERE / "reserve" / "provenance" / "figure_clone_states_stats_v3.json"
        out.write_text(json.dumps(native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print("Statistics:", out, flush=True)


if __name__ == "__main__":
    main()
