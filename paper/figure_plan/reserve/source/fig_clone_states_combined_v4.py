#!/usr/bin/env python
"""Place the S3 map-back prominently; move its celltype view and boxplot to a supplement."""
from __future__ import annotations

import argparse
import copy
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

import fig_clone_states_combined_v3 as ref

HERE = Path(__file__).resolve().parents[2]
fs = ref.fs
MAIN_HEIGHT = 356
SUPP_HEIGHT = 77


def page(height):
    fig = fs.new_page("", "", height=height)
    for txt in list(fig.texts):
        txt.remove()
    fig.panel_boxes_mm = {}
    fig.legend_boxes_mm = {}
    return fig


def ax_at(fig, x, y, w, h, letter, title="", *, coordinates=None):
    ax = ref.make_ax(fig, x, y, w, h, letter, title, coordinates=coordinates)
    fig.panel_boxes_mm[letter] = [x, y, w, h]
    return ax


def left_key(fig, key, top, values, palette, title):
    ref.left_key(fig, key, top, values, palette, title)
    fig.legend_boxes_mm[key] = list(ref.LEGEND_BOXES[key])


def shared_key(fig, key, top, palette):
    ref.shared_key(fig, key, top, palette)
    fig.legend_boxes_mm[key] = list(ref.LEGEND_BOXES[key])


def large_map_key(fig, palette):
    x, y, w, h = 7, 348, 121, 7.2
    ax = ref.at(fig, x, y, w, h)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.legend_boxes_mm["j_k"] = [x, y, w, h]
    ref.LEGEND_TEXTS.append(ax.text(0, .1, "Programme", fontsize=5.6, fontweight="bold", va="top"))
    items = [*palette.items(), ("no programme", ref.GRAY)]
    for i, (label, color) in enumerate(items):
        xx = i * w / len(items)
        ax.scatter([xx + .9], [3.7], s=11, c=color, linewidths=0, clip_on=False)
        ref.LEGEND_TEXTS.append(ax.text(xx + 2.7, 3.7, label, fontsize=5.6, va="center"))


def validate(fig, name, height):
    print(f"Stage: validate {name} page and legends", flush=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, overlap = [], []
    for txt in fig.findobj(match=matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text():
            continue
        b = txt.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1:
            outside.append(txt.get_text())
    legends = [t for t in ref.LEGEND_TEXTS if t.figure is fig]
    for i, a in enumerate(legends):
        for b in legends[i + 1:]:
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                overlap.append([a.get_text(), b.get_text()])
    assert not outside, outside
    assert not overlap, overlap
    gaps = []
    boxes = list(fig.panel_boxes_mm.items())
    for i, (a, aa) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            dx = max(bb[0] - aa[0] - aa[2], aa[0] - bb[0] - bb[2])
            dy = max(bb[1] - aa[1] - aa[3], aa[1] - bb[1] - bb[3])
            assert max(dx, dy) >= 4, (a, b, dx, dy)
            gaps.append(max(dx, dy))
    return {"page_mm": [210, height], "dpi": fs.EXPORT_DPI,
            "panel_letters": list(fig.panel_boxes_mm), "axes_boxes_mm": fig.panel_boxes_mm,
            "legend_boxes_mm": fig.legend_boxes_mm, "minimum_axes_gap_mm": min(gaps),
            "text_outside_page": outside, "legend_text_collisions": overlap,
            "scatter_coordinate_aspect": "equal; frozen square limits shared across views"}


def main_upper(fig, datasets, w, prior, report):
    print("Stage: draw S1 and S2 using frozen v3 statistics", flush=True)
    for name, letters, top in [("S1", "abc", 8), ("S2", "def", 79)]:
        ds = datasets[name]
        cells, clones, state, mapped = (ds[k] for k in ("cells", "clones", "state", "mapped"))
        palette = ref.PROG_COLORS[name]
        ax = ax_at(fig, 31, top, 53, 53, letters[0], ref.GSE[name], coordinates=("Cell UMAP 1", "Cell UMAP 2"))
        ref.draw_categories(ax, cells, cells[state], ref.STATE_COLORS[name], ["umap_1", "umap_2"], .35, alpha=.7)
        ref.finish_map(ax, ds["cell_limits"])
        left_key(fig, letters[0], top, cells[state], ref.STATE_COLORS[name], state)
        ref.emit(letters[0], prior[letters[0]], report)
        ax = ax_at(fig, 91, top, 53, 53, letters[1], "Clone map", coordinates=("Clone x", "Clone y"))
        scale = {"S1": 1.5, "S2": .35}[name]
        ref.draw_categories(ax, clones, clones["clone_programme"], palette, ["x", "y"], clones["n_cells"].to_numpy() * scale, alpha=.92)
        ref.finish_map(ax, ds["clone_limits"])
        ref.emit(letters[1], prior[letters[1]], report)
        ax = ax_at(fig, 151, top, 53, 53, letters[2], "Map-back", coordinates=("Cell UMAP 1", "Cell UMAP 2"))
        ref.draw_categories(ax, cells, mapped, palette, ["umap_1", "umap_2"], .42, alpha=.88)
        ref.finish_map(ax, ds["cell_limits"])
        shared_key(fig, f"{letters[1]}_{letters[2]}", top, palette)
        ref.emit(letters[2], prior[letters[2]], report)

    top = 153
    ax = ax_at(fig, 7, top, 53, 53, "g", "Entropy", coordinates=("Clone x", "Clone y"))
    sc = ax.scatter(w["x"], w["y"], s=w["n_cells"] * .45, c=w["entropy"], cmap="viridis",
                    linewidths=0, alpha=.9, rasterized=True)
    ref.finish_map(ax, datasets["S2"]["clone_limits"])
    ax.text(.035, .965, f"Spearman ρ(entropy, x)\n{prior['g']['spearman_rho']:.3f}; n = {len(w)}",
            transform=ax.transAxes, va="top", fontsize=5.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1.5})
    cb = fig.colorbar(sc, cax=ref.at(fig, 62, top + 7, 1.9, 40))
    cb.ax.set_title("entropy", fontsize=5.2, pad=3)
    cb.ax.tick_params(labelsize=5, pad=1.5)
    cb.set_ticks([0, .5, 1])
    ref.emit("g", prior["g"], report)

    ax = ax_at(fig, 79, top, 53, 53, "h", "label-M")
    rows = prior["h"]["quartiles"]
    ax.bar(np.arange(4), [r["label_M_fraction"] for r in rows], width=.64, color="#58778B", linewidth=0)
    for i, row in enumerate(rows):
        ax.text(i, row["label_M_fraction"] + .015, f"{row['label_M_fraction']:.1%}\nn={row['n']}",
                ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylim(0, .49)
    ax.set_yticks([0, .2, .4], ["0%", "20%", "40%"])
    ax.set_ylabel("Clones with label-M (%)", fontsize=5.8, labelpad=3)
    ax.set_xlabel("Clone x quartile", fontsize=5.8, labelpad=3)
    ref.emit("h", prior["h"], report)

    ax = ax_at(fig, 145, top, 63, 53, "i", "Donor correlations")
    rows = prior["i"]["donors_sorted_by_rho"]
    ax.barh(np.arange(len(rows)), [r["rho"] for r in rows], height=.58, color="#58778B", linewidth=0)
    for i, row in enumerate(rows):
        ax.text(row["rho"] - .03, i, f"{row['donor']} (n={row['n']})", ha="right", va="center", fontsize=5.2)
    ax.axvline(0, color=fs.INK2, linewidth=.5)
    ax.set_yticks([])
    ax.set_ylim(-.7, len(rows) - .3)
    ax.invert_yaxis()
    ax.set_xlim(-1.62, .06)
    ax.set_xticks([-1, -.5, 0])
    ax.set_ylabel("Donor", fontsize=5.8, labelpad=3)
    ax.set_xlabel("Spearman ρ(entropy, x)", fontsize=5.8, labelpad=3)
    ref.emit("i", prior["i"], report)


def main_s3(fig, ds, prior, report):
    print("Stage: enlarge former panel l; stack former k and m-map on its right", flush=True)
    cells, clones, mapped = (ds[k] for k in ("cells", "clones", "mapped"))
    palette = ref.PROG_COLORS["S3"]
    size = .42 * (121 / 53) ** 2
    ax = ax_at(fig, 7, 220, 121, 121, "j", ref.GSE["S3"], coordinates=("Cell UMAP 1", "Cell UMAP 2"))
    ref.draw_categories(ax, cells, mapped, palette, ["umap_1", "umap_2"], size, alpha=.88)
    ref.finish_map(ax, ds["cell_limits"])
    main_j = copy.deepcopy(prior["l"])
    main_j["source_panel_v3"] = "l"
    main_j["marker_area_pt2"] = size
    ref.emit("j", main_j, report)

    ax = ax_at(fig, 145, 220, 53, 53, "k", "Clone map", coordinates=("Clone x", "Clone y"))
    ref.draw_categories(ax, clones, clones["clone_programme"], palette, ["x", "y"], clones["n_cells"].to_numpy() * .30, alpha=.92)
    ref.finish_map(ax, ds["clone_limits"])
    ref.emit("k", prior["k"], report)

    ax = ax_at(fig, 145, 288, 53, 53, "l", "gc_sort:GC", coordinates=("Clone x", "Clone y"))
    measured = pd.to_numeric(clones["gc_sort:GC"], errors="coerce")
    good = measured.notna()
    if (~good).any():
        ax.scatter(clones.loc[~good, "x"], clones.loc[~good, "y"], s=clones.loc[~good, "n_cells"] * .30,
                   color=ref.GRAY, linewidths=0, rasterized=True)
    sc = ax.scatter(clones.loc[good, "x"], clones.loc[good, "y"], s=clones.loc[good, "n_cells"] * .30,
                    c=measured[good], cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
    ref.finish_map(ax, ds["clone_limits"])
    cb = fig.colorbar(sc, cax=ref.at(fig, 201, 295, 1.9, 40))
    cb.ax.set_title("gc_sort:GC", fontsize=5.1, pad=3)
    cb.ax.tick_params(labelsize=5, pad=1.5)
    cb.set_ticks([0, .5, 1])
    ref.emit("l", prior["m"]["map"], report)
    large_map_key(fig, palette)


def supplement(fig, ds, prior, report):
    print("Stage: assemble supplementary celltype UMAP, its legend, and retained boxplot", flush=True)
    cells, clones = ds["cells"], ds["clones"]
    ax = ax_at(fig, 31, 8, 53, 53, "a", ref.GSE["S3"], coordinates=("Cell UMAP 1", "Cell UMAP 2"))
    ref.draw_categories(ax, cells, cells["celltype"], ref.STATE_COLORS["S3"], ["umap_1", "umap_2"], .35, alpha=.7)
    ref.finish_map(ax, ds["cell_limits"])
    left_key(fig, "a", 8, cells["celltype"], ref.STATE_COLORS["S3"], "celltype")

    ax = ax_at(fig, 99, 8, 109, 53, "b", "gc_sort:GC by programme")
    selected = clones[clones["celltype:plasma"].ge(.5)]
    progs = sorted(selected["clone_programme"].dropna().unique())
    arrays = [selected.loc[selected["clone_programme"].eq(p), "gc_sort:GC"].dropna().to_numpy() for p in progs]
    comp = prior["m"]["comparison"]
    summaries = ref.refined_measurement_boxplot(ax, progs, arrays, ref.PROG_COLORS["S3"], comp["p_two_sided"])
    assert ref.native(summaries) == comp["boxplot_summary"]
    report["supplement_panels"] = {"a": copy.deepcopy(prior["j"]), "b": copy.deepcopy(comp)}
    for key, value in report["supplement_panels"].items():
        print(f"Supplement panel {key}: {json.dumps(ref.native(value), ensure_ascii=False, allow_nan=False)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print("Stage: load frozen v3 statistics and inputs", flush=True)
    previous_path = HERE / "reserve" / "provenance" / "figure_clone_states_stats_v3.json"
    previous = json.loads(previous_path.read_text())
    datasets, w, _, paths, _ = ref.load_data()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes == previous["input_sha256"], "Frozen source hashes differ from v3"
    report = {"input_sha256": hashes, "previous_statistics": str(previous_path),
              "previous_statistics_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
              "panels": {}, "programme_colors": ref.PROG_COLORS, "state_colors": ref.STATE_COLORS,
              "S2_display_rename": previous["S2_display_rename"],
              "panel_mapping_from_v3": {"a-i": "main a-i", "l": "main j (large map-back)",
                                        "k": "main k (upper right)", "m.map": "main l (lower right)",
                                        "j": "supplement a", "m.comparison": "supplement b"},
              "shared_programme_keys": {"main_b_c": "S1", "main_e_f": "S2", "main_j_k": "S3"},
              "retained_standalone_boxplot": {"png": str(HERE / "Figure_clone_states_combined_v3_boxplot.png"),
                                              "pdf": str(HERE / "Figure_clone_states_combined_v3_boxplot.pdf")}}
    fig = page(MAIN_HEIGHT)
    main_upper(fig, datasets, w, previous["panels"], report)
    main_s3(fig, datasets["S3"], previous["panels"], report)
    report["main_layout"] = validate(fig, "main", MAIN_HEIGHT)
    supp = page(SUPP_HEIGHT)
    supplement(supp, datasets["S3"], previous["panels"], report)
    report["supplement_layout"] = validate(supp, "supplement", SUPP_HEIGHT)
    print("Main layout:", json.dumps(report["main_layout"], ensure_ascii=False), flush=True)
    print("Supplement layout:", json.dumps(report["supplement_layout"], ensure_ascii=False), flush=True)
    if args.preview:
        for obj, name in [(fig, "main"), (supp, "supplement")]:
            path = Path(f"/tmp/clone_states_v4_{name}_preview.png")
            obj.savefig(path, dpi=180)
            plt.close(obj)
            print("Preview:", path, flush=True)
    else:
        print("Stage: export main and supplementary PNG/PDF at 1200 dpi", flush=True)
        fs.save(fig, "Figure_clone_states_combined_v4", folder=HERE)
        fs.save(supp, "Figure_clone_states_supplement_v4", folder=HERE)
        out = HERE / "reserve" / "provenance" / "figure_clone_states_stats_v4.json"
        out.write_text(json.dumps(ref.native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print("Statistics:", out, flush=True)


if __name__ == "__main__":
    main()
