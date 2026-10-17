#!/usr/bin/env python
"""Assemble the requested frozen tables into one A4 page; no upstream runs."""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parents[2]
CASE = HERE.parent.parent / "case_studies" / "results"
GRAY = "#e7e6e0"
M_LABELS = ["Fcrl5+ MemB", "CD23+ MemB", "other MemB"]
FATE_ORDER = ["GC", "ASC", *M_LABELS, "unassigned"]
FATE_COLORS = dict(zip(FATE_ORDER, ["#2a9d8f", "#e63946", "#457b9d", "#8ecae6", "#b5c9e2", "#c3c2b7"]))
PROG_COLORS = {
    "S1": {"P1": "#7b3294", "P2": "#c51b7d", "P3": "#8c6d31", "P4": "#6b6b6b"},
    "S2": {"P1": fs.BLUE, "P2": fs.ORANGE, "P3": fs.AQUA, "P3a": "#1baf7a", "P3b": "#0b7a5c"},
    "S3": {"P1": "#343a70", "P2": "#ba9600", "P3": "#bd3b32"},
}
STATE_COLORS = {
    "S1": {"Follicular Naive": "#91a6bd", "Immature": "#b7b39c", "GC": fs.BLUE,
           "PB": fs.RED, "Activated": "#b56576", "Precursor": "#8d6e63", "Bystander": "#d4a373",
           "MZ": "#4e9994", "Memory": fs.GREEN, "Atypical": "#7563aa", "B1": "#ae843e"},
    "S2": FATE_COLORS,
    "S3": {"B_cell": "#91a6bd", "GC_Bcell": fs.BLUE, "plasma": fs.VIOLET, "other": "#c3c2b7"},
}
MAIN_BOXES = {}
LEGEND_BOXES = {}
ARTISTS = []


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [native(x) for x in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def emit(key, value, report):
    report["panels"][key] = native(value)
    print(f"Panel {key}: {json.dumps(report['panels'][key], ensure_ascii=False, allow_nan=False)}", flush=True)


def counts(s):
    return {str(k): int(v) for k, v in s.fillna("no programme").value_counts().items()}


def describe(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return {"n": len(v), "missing": len(s) - len(v), "median": v.median(), "mean": v.mean(),
            "min": v.min(), "q25": v.quantile(.25), "q75": v.quantile(.75), "max": v.max()}


def text_mm(fig, x, y, label, **kwargs):
    art = fig.text((fs.LEFT + x) / 210, 1 - y / 297, label, **kwargs)
    ARTISTS.append(art)
    return art


def section(fig, y, label):
    text_mm(fig, 0, y, label, fontsize=5.6, fontweight="bold", color=fs.INK2, va="top")


def make_ax(fig, x, y, w, h, letter, title, *, map_axis=True, suffix=""):
    ax = fs.panel(fig, x, y, w, h, letter, letter_dx=-2.8)
    MAIN_BOXES[letter or suffix] = [x, y + fs.TOP, w, h]
    ax.set_title(title, fontsize=6.1, pad=3)
    if map_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(labelsize=4.7, pad=1.5)
    return ax


def legend(fig, key, x, y, w, h, title, items, ncols=2, fontsize=4.5):
    """Legends use their own millimetre boxes, directly below the owning panel."""
    ax = fs.panel(fig, x, y, w, h)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    LEGEND_BOXES[key] = [x, y + fs.TOP, w, h]
    ARTISTS.append(ax.text(0, .1, title, va="top", fontsize=4.8, fontweight="bold"))
    for k, (label, color) in enumerate(items):
        row, col = divmod(k, ncols)
        xx, yy = col * w / ncols, 3.1 + row * 2.15
        assert yy + 1 < h, (key, yy, h)
        ax.scatter([xx + .8], [yy], s=8, c=color, linewidths=0, clip_on=False)
        ARTISTS.append(ax.text(xx + 2.4, yy, label, va="center", fontsize=fontsize))
    return ax


def draw_categories(ax, table, labels, colors, columns, size, alpha=.8):
    assert labels.index.equals(table.index)
    unknown = set(labels.dropna().unique()) - set(colors)
    assert not unknown, unknown
    xx, yy = (table[c].to_numpy() for c in columns)
    finite = np.isfinite(xx) & np.isfinite(yy)
    size_array = np.broadcast_to(np.asarray(size), (len(table),))
    ordered = [(None, GRAY), *colors.items()]
    for label, color in ordered:
        mask = labels.isna().to_numpy() if label is None else labels.eq(label).to_numpy()
        mask &= finite
        if mask.any():
            ax.scatter(xx[mask], yy[mask], s=size_array[mask], color=color, linewidths=0,
                       alpha=.75 if label is None else alpha, rasterized=True)
    ax.margins(.055)


def items_for(labels, colors, include_missing=False):
    cc = labels.value_counts()
    items = [(f"{p} ({int(cc[p]):,})", color) for p, color in colors.items() if p in cc]
    if include_missing:
        items += [(f"no programme ({int(labels.isna().sum()):,})", GRAY)]
    return items


def validate_layout(fig):
    """Measure real text extents in addition to checking allocated plot boxes."""
    print("Stage: validate A4 layout and typography", flush=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    outside = []
    for txt in fig.findobj(match=matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text():
            continue
        bb = txt.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > page.x1 + 1 or bb.y1 > page.y1 + 1:
            outside.append(txt.get_text())
    assert not outside, f"Text outside page: {outside}"
    boxes = list(MAIN_BOXES.items())
    gaps = []
    for i, (a, aa) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            xgap = max(bb[0] - aa[0] - aa[2], aa[0] - bb[0] - bb[2])
            ygap = max(bb[1] - aa[1] - aa[3], aa[1] - bb[1] - bb[3])
            assert max(xgap, ygap) >= 4, (a, b, xgap, ygap)
            gaps.append(max(xgap, ygap))
    return {"page_mm": [210, 297], "dpi": fs.EXPORT_DPI,
            "panel_letters": list("abcdefghijklm"), "main_axes_boxes_mm": MAIN_BOXES,
            "legend_boxes_mm": LEGEND_BOXES, "minimum_axes_separation_mm": min(gaps),
            "minimum_legend_font_pt": 4.5, "text_outside_page": outside}


def main():
    print("Stage: load frozen tables and P3 decision", flush=True)
    paths = []
    datasets = {}
    for name, ds, fname, state in [("S1", "malaria_late", "cells.csv.gz", "cell_state"),
                                   ("S2", "spont_gc", "spont_gc_fate_cells.csv.gz", "fate"),
                                   ("S3", "gut_gc_reporter", "cells.csv.gz", "celltype")]:
        cell_path, clone_path = CASE / ds / fname, CASE / ds / "clone_table.csv"
        paths += [cell_path, clone_path]
        cells = pd.read_csv(cell_path, index_col=0)
        clones = pd.read_csv(clone_path, index_col=0)
        clones.index = clones.index.astype(str)
        assert clones.index.is_unique
        mapped = cells["clone_id"].map(clones["clone_programme"])
        if "clone_programme" in cells:
            assert cells["clone_programme"].fillna("").eq(mapped.fillna("")).all()
        datasets[name] = {"cells": cells, "clones": clones, "state": state, "mapped": mapped}
    sub_path = CASE / "spont_gc" / "p3_subsplit.json"
    sub = json.loads(sub_path.read_text())
    paths.append(sub_path)
    s2 = datasets["S2"]
    s2["clones"] = s2["clones"].copy()
    if sub["split_accepted"]:
        assignments = pd.Series(sub["assignments_by_clone_id"])
        target = s2["clones"]["clone_programme"].eq("P3")
        assert set(assignments.index) == set(s2["clones"].index[target])
        s2["clones"].loc[target, "clone_programme"] = assignments.reindex(s2["clones"].index[target]).to_numpy()
        s2["mapped"] = s2["cells"]["clone_id"].map(s2["clones"]["clone_programme"])
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
    report = {"input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "panels": {}, "P3_split_accepted": sub["split_accepted"],
              "programme_colors": PROG_COLORS, "state_colors": STATE_COLORS,
              "label_M_definition": {"fate_values": M_LABELS, "minimum_cells_per_clone": 1},
              "analysis_population": {"S2_g_h_i": "all 322 rows of spont_gc_fate_clones.csv, as in the reference panel",
                                      "S2_e": "all 1051 rows of clone_table.csv",
                                      "S2_f": "all 22898 rows of spont_gc_fate_cells.csv.gz",
                                      "S3_m_ii": "all clone_table rows with celltype:plasma >= 0.5"},
              "test_methods": {"correlations": "two-sided scipy.stats.spearmanr",
                               "S3_P2_P3": "two-sided scipy.stats.mannwhitneyu, method=auto, continuity correction=True"}}

    print("Stage: draw panels a-f and j-l", flush=True)
    fig = fs.new_page("Fig. cand.", "Dataset S1 malaria_late (GSE286215)",
                      "Dataset S2 spont_gc (GSE203132)\nDataset S3 gut_gc_reporter (GSE284622)", height=297)
    section(fig, 24, "Section 1 · DS1")
    section(fig, 104, "Section 2 · DS2")
    section(fig, 205, "Section 3 · DS3")
    for name, letters, y, h, ly, lh in [("S1", "abc", 12.5, 50, 65, 17),
                                       ("S2", "def", 94, 33, 130, 11),
                                       ("S3", "jkl", 195, 30, 228, 9)]:
        ds = datasets[name]
        cells, clones, state, mapped = (ds[k] for k in ("cells", "clones", "state", "mapped"))
        pal = PROG_COLORS[name]
        ax = make_ax(fig, 2, y, 53, h, letters[0], f"{name} cells — {state}")
        draw_categories(ax, cells, cells[state], STATE_COLORS[name], ["umap_1", "umap_2"], .35, alpha=.65)
        legend(fig, letters[0], 2, ly, 53, lh, state,
               items_for(cells[state], STATE_COLORS[name]), ncols=2)
        emit(letters[0], {"n_cells": len(cells), "category_counts": counts(cells[state]),
                          "n_missing_coordinates": int(cells[["umap_1", "umap_2"]].isna().any(axis=1).sum())}, report)
        ax = make_ax(fig, 65, y, 53, h, letters[1], f"{name} clones — programme")
        scale = {"S1": 1.5, "S2": .35, "S3": .30}[name]
        draw_categories(ax, clones, clones["clone_programme"], pal, ["x", "y"], clones["n_cells"].to_numpy() * scale)
        legend(fig, letters[1], 65, ly, 53, lh, "programme · size: n_cells",
               items_for(clones["clone_programme"], pal), ncols=2)
        emit(letters[1], {"n_clones": len(clones), "category_counts": counts(clones["clone_programme"]),
                          "n_cells_sum": int(clones["n_cells"].sum()), "n_cells": describe(clones["n_cells"]),
                          "marker_area_pt2_per_cell": scale}, report)
        ax = make_ax(fig, 128, y, 53, h, letters[2], f"{name} map-back — programme")
        draw_categories(ax, cells, mapped, pal, ["umap_1", "umap_2"], .42, alpha=.8)
        legend(fig, letters[2], 128, ly, 53, lh, "programme", items_for(mapped, pal, True), ncols=2)
        emit(letters[2], {"n_cells": len(cells), "category_counts": counts(mapped),
                          "n_labelled": int(mapped.notna().sum()), "n_without_programme": int(mapped.isna().sum()),
                          "n_missing_clone_id": int(cells["clone_id"].isna().sum()),
                          "n_nonmissing_clone_id_not_in_clone_table": int((cells["clone_id"].notna() & ~cells["clone_id"].isin(clones.index)).sum())}, report)

    print("Stage: compute and draw panels g-i", flush=True)
    ax = make_ax(fig, 2, 149, 46, 28, "g", "S2 clones — entropy")
    sc = ax.scatter(w["x"], w["y"], s=w["n_cells"] * .45, c=w["entropy"], cmap="viridis",
                    linewidths=0, alpha=.9, rasterized=True)
    rho = stats.spearmanr(w["entropy"], w["x"])
    ax.text(.035, .975, f"Spearman ρ(entropy, x)\n{rho.statistic:.3f}; n = {len(w)}", transform=ax.transAxes,
            va="top", fontsize=4.9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
    cax = fs.panel(fig, 50.3, 153, 1.8, 24)
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("entropy", fontsize=4.5, pad=2)
    cb.ax.tick_params(labelsize=4.5, pad=1)
    cb.set_ticks([0, .5, 1])
    emit("g", {"n_clones": len(w), "entropy": describe(w["entropy"]),
               "spearman_rho": rho.statistic, "spearman_p": rho.pvalue}, report)

    ax = make_ax(fig, 69, 149, 47, 28, "h", "S2 quartiles — label-M", map_axis=False)
    quartiles, edges = pd.qcut(w["x"], 4, labels=["Q1", "Q2", "Q3", "Q4"], retbins=True)
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        ww = w[quartiles.eq(q)]
        rows.append({"quartile": q, "n": len(ww), "label_M_n": int(ww["label_M"].sum()),
                     "label_M_fraction": ww["label_M"].mean(), "x_min": ww["x"].min(), "x_max": ww["x"].max()})
    ax.bar(np.arange(4), [r["label_M_fraction"] for r in rows], width=.64, color="#526c83", linewidth=0)
    for i, r in enumerate(rows):
        ax.text(i, r["label_M_fraction"] + .018, f"{r['label_M_fraction']:.1%}\nn={r['n']}",
                ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylim(0, .51)
    ax.set_yticks([0, .2, .4], ["0%", "20%", "40%"])
    ax.set_ylabel("label-M", fontsize=5, labelpad=2)
    ax.set_xlabel("x quartile", fontsize=5, labelpad=2)
    emit("h", {"n_clones": len(w), "qcut_edges": edges, "quartiles": rows,
               "label_M_n": int(w["label_M"].sum())}, report)

    ax = make_ax(fig, 131, 149, 50, 28, "i", "S2 donors — ρ(entropy, x)", map_axis=False)
    donor_rows = []
    for d, ww in w.groupby("donor"):
        rr = stats.spearmanr(ww["entropy"], ww["x"])
        donor_rows.append({"donor": d, "n": len(ww), "rho": rr.statistic, "p": rr.pvalue})
    donor_rows.sort(key=lambda d: d["rho"])
    ax.barh(np.arange(len(donor_rows)), [d["rho"] for d in donor_rows], height=.58, color="#526c83", linewidth=0)
    for i, d in enumerate(donor_rows):
        ax.text(d["rho"] - .025, i, f"{d['donor']} (n={d['n']})", ha="right", va="center", fontsize=4.5)
    ax.axvline(0, color=fs.INK2, linewidth=.5)
    ax.set_yticks([])
    ax.set_ylim(-.75, len(donor_rows) - .25)
    ax.invert_yaxis()
    ax.set_xlim(-1.55, .08)
    ax.set_xticks([-1, -.5, 0])
    ax.set_xlabel("Spearman ρ(entropy, x)", fontsize=5, labelpad=2)
    emit("i", {"n_clones": len(w), "donors_sorted_by_rho": donor_rows}, report)

    print("Stage: validate and draw panel m", flush=True)
    t = datasets["S3"]["clones"]
    measured = pd.to_numeric(t["gc_sort:GC"], errors="coerce")
    ax = make_ax(fig, 2, 245, 61, 25, "m", "S3 clones — gc_sort:GC")
    good = measured.notna()
    if (~good).any():
        ax.scatter(t.loc[~good, "x"], t.loc[~good, "y"], s=t.loc[~good, "n_cells"] * .30,
                   c=GRAY, linewidths=0, rasterized=True)
    sc = ax.scatter(t.loc[good, "x"], t.loc[good, "y"], s=t.loc[good, "n_cells"] * .30,
                    c=measured[good], cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
    cax = fs.panel(fig, 66, 249, 1.8, 21)
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("gc_sort:GC", fontsize=4.5, pad=2)
    cb.ax.tick_params(labelsize=4.5, pad=1)
    cb.set_ticks([0, .5, 1])

    ax = make_ax(fig, 98, 245, 78, 25, None, "S3 programmes — gc_sort:GC", map_axis=False, suffix="m_ii")
    selected = t[t["celltype:plasma"].ge(.5)].copy()
    progs = sorted(selected["clone_programme"].dropna().unique())
    arr = [selected.loc[selected["clone_programme"].eq(p), "gc_sort:GC"].dropna().to_numpy() for p in progs]
    group_stats = {p: describe(selected.loc[selected["clone_programme"].eq(p), "gc_sort:GC"]) for p in progs}
    p2, p3 = arr[progs.index("P2")], arr[progs.index("P3")]
    test = stats.mannwhitneyu(p2, p3, alternative="two-sided")
    expected_p = branches["branch_comparison"]["gc_sort"]["p"]
    assert np.isclose(test.pvalue, expected_p, rtol=1e-12, atol=0), (test.pvalue, expected_p)
    rng = np.random.default_rng(0)
    for i, (p, v) in enumerate(zip(progs, arr)):
        ax.scatter(i + rng.uniform(-.19, .19, len(v)), v, s=3, color=PROG_COLORS["S3"][p],
                   alpha=.5, linewidths=0, rasterized=True)
    bp = ax.boxplot(arr, positions=np.arange(len(progs)), widths=.50, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": .8},
                    boxprops={"linewidth": .5}, whiskerprops={"linewidth": .5}, capprops={"linewidth": .5})
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
    ax.plot([1, 1, 2, 2], [1.12, 1.18, 1.18, 1.12], color=fs.INK2, lw=.5)
    ax.text(1.5, 1.22, f"P2–P3: p = {test.pvalue:.3g}", fontsize=4.6, ha="center", va="bottom")
    ax.text(.01, .98, "celltype:plasma ≥ 0.5", transform=ax.transAxes, fontsize=4.5, ha="left", va="top")
    ax.set_xticks(np.arange(len(progs)), [f"{p}\n(n={len(v)})" for p, v in zip(progs, arr)])
    ax.set_yticks([0, .5, 1])
    ax.set_ylim(-.07, 1.60)
    ax.set_xlim(-.6, 2.7)
    ax.set_ylabel("gc_sort:GC", fontsize=5, labelpad=2)
    emit("m", {"map": {"n_clones": len(t), "gc_sort:GC": describe(measured)},
               "comparison": {"threshold_column": "celltype:plasma", "threshold_minimum": .5,
                              "n_all_programmes": len(selected), "programmes": group_stats,
                              "P2_P3_n": len(p2) + len(p3), "U_P2_vs_P3": test.statistic,
                              "p_two_sided": test.pvalue, "reference_json_p": expected_p,
                              "reference_json_matches": True,
                              "display": "seed=0 horizontal jitter and boxplot; all qualifying programmes"}}, report)

    report["layout"] = validate_layout(fig)
    report["panels"] = {letter: report["panels"][letter] for letter in "abcdefghijklm"}
    print("Layout:", json.dumps(native(report["layout"]), ensure_ascii=False), flush=True)
    print("Stage: export A4 PNG and PDF at 1200 dpi", flush=True)
    fs.save(fig, "Figure_clone_states_combined", folder=HERE)
    out = HERE / "reserve" / "provenance" / "figure_clone_states_stats.json"
    out.write_text(json.dumps(native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
