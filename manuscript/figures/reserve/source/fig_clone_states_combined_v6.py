#!/usr/bin/env python
"""Add randomized original GC/nonGC labels and a PC zoom to the combined figure.

Reuse the frozen v4/v3 panels and statistics; redraw the gut block as two paired rows.
Run with --preview for a quick check or without it for 1200 dpi PNG/PDF exports.
"""
from __future__ import annotations

import os
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
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import stats

import fig_clone_states_combined_v4 as base
import fig_clone_states_combined_v3 as ref

HERE = Path(__file__).resolve().parents[2]
fs = ref.fs
HEIGHT = 400
GUT_HEIGHT = 185
GUT_SHIFT = 215
GUT_SIZE = 64
RANDOM_SEED = 20261015
GATE_COLORS = {"GC": "#d94845", "nonGC": "#2a78d6"}
METADATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parents[3] / "case_studies" / "data")) / "gse284622_gut_gc" / "gex_reporter" / "metadata.csv.gz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ax_at(fig, x, y, w, h, letter, title="", *, coordinates=None):
    ax = ref.make_ax(fig, x, y, w, h, letter, title, coordinates=coordinates)
    ax.title.set_fontsize(8.2)
    fig.panel_boxes_mm[letter] = [x, y, w, h]
    fig.panel_axes[letter] = ax
    return ax


def legend_box(fig, key, x, y, w, h):
    ax = ref.at(fig, x, y, w, h)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.legend_boxes_mm[key] = [x, y, w, h]
    return ax


def legend_text(ax, x, y, text, **kwargs):
    artist = ax.text(x, y, text, **kwargs)
    ref.LEGEND_TEXTS.append(artist)
    return artist


def left_key(fig, key, top, values, palette, title):
    title = {"cell_state": "Cell state", "fate": "RNA state"}.get(title, title)
    items = [(name, color, int(values.eq(name).sum())) for name, color in palette.items()
             if values.eq(name).any()]
    pitch = 5.65
    h = 4.4 + len(items) * pitch
    y = max(2, top + (53 - h) / 2)
    ax = legend_box(fig, key, 2, y, 23.5, h)
    legend_text(ax, .1, .1, title, fontsize=7.2, fontweight="bold", va="top")
    for i, (label, color, n) in enumerate(items):
        yy = 4 + i * pitch
        ax.scatter([.9], [yy + .6], s=12, c=color, linewidths=0, clip_on=False)
        legend_text(ax, 2.4, yy, label, fontsize=7, va="top")
        legend_text(ax, 2.4, yy + 2.65, f"n = {n:,}", fontsize=6.8,
                    va="top", color=fs.INK2)


def programme_key(fig, key, x, y, w, palette):
    ax = legend_box(fig, key, x, y, w, 8.4)
    legend_text(ax, 0, .1, "Programme", fontsize=7.2, fontweight="bold", va="top")
    items = [*palette.items(), ("no programme", ref.GRAY)]
    for i, (label, color) in enumerate(items):
        xx = i * w / len(items)
        ax.scatter([xx + .9], [5], s=13, c=color, linewidths=0, clip_on=False)
        legend_text(ax, xx + 2.8, 5, label, fontsize=7, va="center")


def shared_key(fig, key, top, palette):
    programme_key(fig, key, 91, top + 60, 113, palette)


def page(height):
    fig = base.page(height)
    fig.panel_axes = {}
    fig.auxiliary_boxes_mm = {}
    return fig


def typography(fig):
    mapping = {4.9: 6.8, 5.0: 6.8, 5.1: 6.8, 5.2: 6.8, 5.3: 6.8,
               5.4: 7, 5.5: 7, 5.6: 7, 5.8: 7.2, 6.0: 7.2, 6.6: 8.2, 8.5: 10}
    for text in fig.findobj(match=matplotlib.text.Text):
        size = round(float(text.get_fontsize()), 2)
        if size in mapping:
            text.set_fontsize(mapping[size])
        elif size < 6.8:
            text.set_fontsize(6.8)


def validate(fig, name, height, dpi):
    typography(fig)
    result = base.validate(fig, name, height)
    result["dpi"] = dpi
    result["visible_font_sizes_pt"] = sorted({
        float(t.get_fontsize()) for t in fig.findobj(match=matplotlib.text.Text)
        if t.get_visible() and t.get_text()
    })
    assert min(result["visible_font_sizes_pt"]) >= 6.8
    result["auxiliary_axes_boxes_mm"] = fig.auxiliary_boxes_mm
    for key, auxiliary in fig.auxiliary_boxes_mm.items():
        for panel, box in fig.panel_boxes_mm.items():
            dx = max(box[0] - auxiliary[0] - auxiliary[2], auxiliary[0] - box[0] - box[2])
            dy = max(box[1] - auxiliary[1] - auxiliary[3], auxiliary[1] - box[1] - box[3])
            assert max(dx, dy) >= 2, (key, panel, dx, dy)
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(match=matplotlib.text.Text) if t.get_visible() and t.get_text()]
    collisions = []
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                collisions.append([a.get_text(), b.get_text()])
    result["all_visible_text_collisions"] = collisions
    print("Text collisions:", collisions, flush=True)
    assert not collisions, collisions
    return result


def validate_ghi(ds, w, prior):
    """Check the saved scientific quantities against their actual cell-level definitions."""
    cells = ds["cells"]
    assigned = cells[cells["clone_id"].notna() & cells["fate"].notna()
                     & ~cells["fate"].astype(str).str.startswith("unassigned")]
    state_counts = pd.crosstab(assigned["clone_id"], assigned["fate"]).reindex(w.index)
    assert state_counts.sum(axis=1).eq(w["n_cells"]).all()
    assert w["n_cells"].ge(5).all()
    fractions = state_counts.div(state_counts.sum(axis=1), axis=0).replace(0, np.nan)
    entropy = -(fractions * np.log(fractions)).sum(axis=1)
    assert np.allclose(entropy, w["entropy"], atol=1e-12, rtol=0)
    rho = stats.spearmanr(w["entropy"], w["x"])
    assert np.isclose(rho.statistic, prior["g"]["spearman_rho"])
    labels = cells["fate"].isin(ref.M_LABELS).groupby(cells["clone_id"]).any().reindex(w.index)
    assert labels.eq(w["label_M"]).all()
    quartiles = pd.qcut(w["x"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for row in prior["h"]["quartiles"]:
        group = w[quartiles.eq(row["quartile"])]
        assert len(group) == row["n"] and int(group["label_M"].sum()) == row["label_M_n"]
    for row in prior["i"]["donors_sorted_by_rho"]:
        group = w[w["donor"].eq(row["donor"])]
        result = stats.spearmanr(group["entropy"], group["x"])
        assert len(group) == row["n"] and np.isclose(result.statistic, row["rho"])
        assert np.isclose(result.pvalue, row["p"], rtol=1e-10, atol=0)
    return {"entropy_recomputed_from_assigned_cell_states": True,
            "max_entropy_difference": float((entropy - w["entropy"]).abs().max()),
            "minimum_assigned_cells_per_clone": 5,
            "entropy_states": state_counts.columns.tolist(), "entropy_log_base": "natural log",
            "unassigned_cells_in_entropy": False,
            "label_M_definition": {"fate_values": ref.M_LABELS, "minimum_cells_per_clone": 1},
            "quartile_counts_recomputed": True, "within_donor_correlations_recomputed": True,
            "state_label_origin": "RNA marker-based annotation; descriptive state composition"}


def load_sort(ds, metadata_path):
    cells, clones = ds["cells"], ds["clones"]
    meta = pd.read_csv(metadata_path, index_col="cell", low_memory=False)
    assert cells.index.is_unique and meta.index.is_unique
    assert cells.index.isin(meta.index).all()
    meta = meta.loc[cells.index]
    for cc, mc in [("sample", "sample_id"), ("tissue", "tissue"), ("celltype", "celltype")]:
        assert cells[cc].eq(meta[mc]).all()
    assert meta["gc"].isin(GATE_COLORS).all()
    labels = meta["gc"]
    fractions = labels.eq("GC").groupby(cells["clone_id"]).mean().reindex(clones.index)
    assert np.allclose(fractions, clones["gc_sort:GC"], rtol=0, atol=1e-12)
    display = cells.copy()
    display["gc_sort_original"] = labels
    display = display.sample(frac=1, random_state=RANDOM_SEED)
    assert len(display) == len(cells) and display.index.is_unique
    assert display[["umap_1", "umap_2"]].sort_index().equals(
        cells[["umap_1", "umap_2"]].sort_index())
    assert display["gc_sort_original"].sort_index().equals(labels.sort_index().rename("gc_sort_original"))
    pc = cells[cells["celltype"].eq("plasma")]
    pc_counts = labels.loc[pc.index].value_counts().to_dict()
    # A square view enclosing the central PC cluster. All cells in this spatial window are drawn.
    q = pc[["umap_1", "umap_2"]].quantile([.01, .99]).to_numpy()
    center = q.mean(axis=0)
    radius = float((q[1] - q[0]).max() * .61)
    zoom_limits = [(float(c - radius), float(c + radius)) for c in center]
    inside = display["umap_1"].between(*zoom_limits[0]) & display["umap_2"].between(*zoom_limits[1])
    zoom_pc = display.loc[inside & display["celltype"].eq("plasma"), "gc_sort_original"]
    metadata = {"source": str(metadata_path), "source_sha256": digest(metadata_path),
                "n_cells": len(cells), "original_sort_counts": labels.value_counts().to_dict(),
                "original_sort_counts_in_author_plasma_cells": pc_counts,
                "draw_order": "one uniform random permutation of every cell; both classes interleaved",
                "random_seed": RANDOM_SEED, "n_points_dropped": 0,
                "umap_coordinates_modified": False, "sort_labels_modified": False,
                "colors": GATE_COLORS,
                "pc_zoom": {"bounds": zoom_limits,
                            "definition": "Square limits from 1st/99th plasma-cell coordinate quantiles; all cell types in the window shown",
                            "n_all_cells_in_window": int(inside.sum()),
                            "n_plasma_cells_in_window": int(len(zoom_pc)),
                            "plasma_sort_counts_in_window": zoom_pc.value_counts().to_dict(),
                            "draw_order": "same randomized cell order as the full UMAP"}}
    return display, metadata


def draw_s3(fig, ds, display, sort_metadata, prior, report, *, shift=0):
    cells, clones, mapped = (ds[k] for k in ("cells", "clones", "mapped"))
    palette = ref.PROG_COLORS["S3"]
    top, bottom = 225 - shift, 315 - shift
    size = .42 * (GUT_SIZE / 53) ** 2
    clone_scale = .30 * (GUT_SIZE / 53) ** 2
    ax = ax_at(fig, 24, top, GUT_SIZE, GUT_SIZE, "j", "GSE284622 | Programme map-back",
               coordinates=("Cell UMAP 1", "Cell UMAP 2"))
    # Interleave programme-coloured cells too, after drawing unassigned cells in gray.
    missing = mapped.isna()
    ax.scatter(cells.loc[missing, "umap_1"], cells.loc[missing, "umap_2"], s=size,
               c=ref.GRAY, alpha=.75, linewidths=0, rasterized=True)
    indices = cells.index[~missing].to_numpy().copy()
    np.random.default_rng(RANDOM_SEED).shuffle(indices)
    foreground = cells.loc[indices]
    ax.scatter(foreground["umap_1"], foreground["umap_2"], s=size,
               c=mapped.loc[indices].map(palette), alpha=.88, linewidths=0, rasterized=True)
    ref.finish_map(ax, ds["cell_limits"])
    panel_j = copy.deepcopy(prior["j"])
    panel_j.update({"marker_area_pt2": size, "draw_order": "randomized coloured cells over gray background"})
    ref.emit("j", panel_j, report)

    ax = ax_at(fig, 128, top, GUT_SIZE, GUT_SIZE, "k", "Clone map",
               coordinates=("Clone x", "Clone y"))
    ref.draw_categories(ax, clones, clones["clone_programme"], palette, ["x", "y"],
                        clones["n_cells"].to_numpy() * clone_scale, alpha=.92)
    ref.finish_map(ax, ds["clone_limits"])
    panel_k = copy.deepcopy(prior["k"])
    panel_k["marker_area_pt2_per_cell"] = clone_scale
    ref.emit("k", panel_k, report)
    programme_key(fig, "j_k", 24, 298 - shift, 168, palette)

    ax = ax_at(fig, 24, bottom, GUT_SIZE, GUT_SIZE, "l", "Original GC/nonGC sort",
               coordinates=("Cell UMAP 1", "Cell UMAP 2"))
    raw_size, raw_alpha = .55, .78
    ax.scatter(display["umap_1"], display["umap_2"], s=raw_size,
               c=display["gc_sort_original"].map(GATE_COLORS), alpha=raw_alpha,
               linewidths=0, rasterized=True)
    ref.finish_map(ax, ds["cell_limits"])
    zoom_limits = sort_metadata["pc_zoom"]["bounds"]
    rectangle = Rectangle((zoom_limits[0][0], zoom_limits[1][0]),
                          zoom_limits[0][1] - zoom_limits[0][0],
                          zoom_limits[1][1] - zoom_limits[1][0],
                          linewidth=.6, edgecolor="#60656a", facecolor="none", linestyle=(0, (3, 2)))
    ax.add_patch(rectangle)
    # Put the zoom in the reserved gutter, so the full UMAP remains unobstructed.
    zoom_box = [95, 327 - shift, 26, 26]
    zoom = ref.at(fig, *zoom_box)
    fig.auxiliary_boxes_mm["l_pc_zoom"] = zoom_box
    zoom.set_facecolor("white")
    zoom.scatter(display["umap_1"], display["umap_2"], s=.40,
                 c=display["gc_sort_original"].map(GATE_COLORS), alpha=.8,
                 linewidths=0, rasterized=True)
    ref.finish_map(zoom, zoom_limits)
    zoom.set_xticks([])
    zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color("#9d9d9d")
        spine.set_linewidth(.45)
    zoom.set_title("PC cluster (zoom)", fontsize=6.8, pad=2)
    connector = ConnectionPatch(xyA=(zoom_limits[0][1], zoom_limits[1][1]),
                                coordsA="data", axesA=ax, xyB=(0, 0),
                                coordsB="axes fraction", axesB=zoom,
                                color="#a0a0a0", linewidth=.4, clip_on=False)
    fig.add_artist(connector)
    ref.emit("l", dict(sort_metadata, marker_area_pt2=raw_size, alpha=raw_alpha), report)
    ax = legend_box(fig, "l_original_sort", 24, 387 - shift, GUT_SIZE, 8.4)
    legend_text(ax, 0, .1, "Original sort", fontsize=7.2, fontweight="bold", va="top")
    for i, (gate, color) in enumerate(GATE_COLORS.items()):
        xx = i * GUT_SIZE / 2
        ax.scatter([xx + .9], [5], s=13, c=color, linewidths=0)
        legend_text(ax, xx + 2.8, 5, f"{gate} (n = {sort_metadata['original_sort_counts'][gate]:,})",
                    fontsize=7, va="center")

    ax = ax_at(fig, 128, bottom, GUT_SIZE, GUT_SIZE, "m", "GC-sort fraction",
               coordinates=("Clone x", "Clone y"))
    measured = pd.to_numeric(clones["gc_sort:GC"], errors="raise")
    ax.scatter(clones["x"], clones["y"], s=clones["n_cells"] * clone_scale,
               c=measured, cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
    ref.finish_map(ax, ds["clone_limits"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
                      cax=ref.at(fig, 196, bottom + 8, 2, 48))
    cb.ax.set_title("GC fraction", fontsize=6.8, pad=8)
    cb.ax.tick_params(labelsize=6.8, pad=1.5)
    cb.set_ticks([0, .5, 1])
    panel_m = copy.deepcopy(prior["l"])
    panel_m["source_panel_v4"] = "l"
    ref.emit("m", panel_m, report)
    ax = legend_box(fig, "m_clone_scale", 128, 389 - shift, GUT_SIZE, 6)
    legend_text(ax, 0, .1, "958 clones; point area = captured cells", fontsize=6.8, va="top")


def clarify_ghi(fig, prior):
    fig.panel_axes["g"].set_title("State-mix entropy", fontsize=8.2, pad=3.5)
    fig.panel_axes["h"].set_title("Memory-containing clones", fontsize=8.2, pad=3.5)
    fig.panel_axes["h"].set_ylabel("Clones with MemB (%)", fontsize=7.2, labelpad=3)
    for text, row in zip(fig.panel_axes["h"].texts, prior["h"]["quartiles"]):
        text.set_text(f"{row['label_M_fraction']:.1%}\n{row['label_M_n']}/{row['n']}")
    fig.panel_axes["i"].set_title("Within-donor correlations", fontsize=8.2, pad=3.5)


def write_notes(report):
    path = HERE / "Figure_clone_states_combined_v6_notes.md"
    path.write_text("""# 组合图 v6：排版和 g/h/i 说明

主图为 `Figure_clone_states_combined_v6.png/.pdf`，下方区域单独导出为
`Figure_clone_states_combined_v6_gut_detail.png/.pdf`。原图 v4 保留。

下方 GSE284622 区域采用 2 × 2 排列。j 从 121 × 121 mm 缩为 64 × 64 mm，
四个主要面板均为 64 mm 正方形：

| | 细胞 UMAP | 克隆 map |
| --- | --- | --- |
| Programme | j：克隆 programme 映射回细胞 | k：克隆 programme |
| GC 分选 | l：原始逐细胞 GC/nonGC 分选标签，新增 | m：克隆 GC 分选比例，原 v4 的 l |

j/l 使用同一套单细胞坐标和相同坐标范围；k/m 使用同一套克隆坐标。
标题约 8.2 pt，坐标标签 7.2 pt，图例和计数 6.8–7.2 pt，面板字母 10 pt。
上下两行的图例分别说明 programme 和原始分选，避免将 P1/P3 颜色与 GC/nonGC 含义混淆。

新增 l 将所有 16,619 个细胞用同一个固定种子的均匀随机排列绘制，
GC 红色、nonGC 蓝色；没有按分选类别分层覆盖，也没有改变坐标、标签、
类别比例或抽样。主图中点面积 0.55 pt²、透明度 0.78。
PC cluster inset 放大虚线框区域，区域内所有细胞类型都显示，绘制顺序
与整个 UMAP 一致。放大窗口来自作者 plasma 注释细胞坐标的 1%/99% 分位数，
扩为正方形；详细边界和计数记录在统计 JSON。
作者注释的全部 plasma 细胞中，GC 3,394 个、nonGC 579 个；
二者存在混杂，但原始比例为约 85.4% / 14.6%，因此没有将它们画成等比例。

## g：克隆内状态组成的熵

g/h/i 都属于 GSE203132（S2），与新增的 GSE284622 分选图属于不同数据集。
它们使用 322 个至少包含 5 个已注释 BCR+ 细胞的克隆；这是 e 中
1,051 个克隆的一个子集。熵计算先排除 `unassigned` 细胞，再计算克隆内
GC、ASC、Fcrl5+ MemB、CD23+ MemB、other MemB 的比例。

```text
H(c) = -Σ p_s(c) × ln(p_s(c))
```

熵为 0 表示所捕获的已注释细胞全部属于同一类别。熵越大，表示组成包含
更多类别、或各类别的比例更均衡。例如 GC/ASC 各半时 H = ln(2) ≈ 0.693。
每个点为一个克隆，位置为 clone map 坐标，颜色为 H，面积随捕获细胞数增加。
H 与当前横坐标 x 的 Spearman ρ = -0.741845（P = 1.738 × 10^-57）；
这表示当前 map 上从左到右，状态混合度总体下降。

这些类别由 RNA marker module score 注释，熵是状态组成的描述，
不是直接测量的后代命运多样性。x 是当前二维 map 的横坐标，不是时间、
谱系先后或分化轨迹。

## h：包含 memory-labelled 细胞的克隆比例

原 `label-M` 的定义是：一个克隆至少捕获到 1 个标为 Fcrl5+ MemB、
CD23+ MemB 或 other MemB 的细胞，即 `label_M = True`。
它不是要求 memory 细胞占多数，也不是每个克隆内 memory 细胞的比例。

将同一批 322 个克隆按当前 x 坐标从小到大分成四个近似等大的分位组：

| 组 | 含 memory-labelled 细胞的克隆 / 全部克隆 | 比例 |
| --- | --- | --- |
| Q1，最左 | 29/81 | 35.8% |
| Q2 | 10/80 | 12.5% |
| Q3 | 9/80 | 11.2% |
| Q4，最右 | 2/81 | 2.5% |

它说明含 memory-labelled 细胞的克隆主要分布在 map 左侧。
v6 将标题改为 `Memory-containing clones`、纵轴改为 `Clones with MemB (%)`，
并在柱上直接显示分子/分母，含义更清楚。

## i：每个 donor 内的熵与 x 的相关性

在 8 个 donor 内分别计算 Spearman ρ(H, x)。每根横条对应一个 donor，
旁边的 n 是该 donor 在上述 322 个克隆中的数量。8 个相关系数都为负，
范围约 -0.740 至 -0.418，说明左侧状态混合度更高的趋势在各 donor 内方向一致。
M23 仅 12 个克隆，P ≈ 0.177，单独不显著；方向一致不等于所有 donor 都显著。

i 是对 donor 内方向一致性的描述检查，不能直接当作 donor 层级的正式
总体推断。g 的总体相关性、h 的空间富集、i 的 donor 内相关性都来源于
已有 RNA 注释与 map，不能视为互相独立的 fate 验证。

本次从原始细胞状态重算熵、四分位组计数和每个 donor 的相关性，均与
冻结结果一致。布局验证检查页面边界、图例文字相互重叠、字号、面板间距，
输入哈希和核验结果见 `figure_clone_states_stats_v6.json`。

重新生成：`python fig_clone_states_combined_v6.py`；快速预览加 `--preview`。
""", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    args = parser.parse_args()
    assert args.dpi > 0
    previous_path = HERE / "reserve" / "provenance" / "figure_clone_states_stats_v4.json"
    previous = json.loads(previous_path.read_text())
    datasets, w, _, paths, _ = ref.load_data()
    hashes = {str(p): digest(p) for p in paths}
    assert hashes == previous["input_sha256"], "Frozen v4 source data changed"
    ghi = validate_ghi(datasets["S2"], w, previous["panels"])
    display, sort_metadata = load_sort(datasets["S3"], args.metadata)
    report = {"input_sha256": hashes, "previous_statistics": str(previous_path),
              "previous_statistics_sha256": digest(previous_path), "panels": {},
              "programme_colors": ref.PROG_COLORS, "state_colors": ref.STATE_COLORS,
              "S2_display_rename": previous["S2_display_rename"], "g_h_i_validation": ghi,
              "panel_mapping_from_v4": {"a-k": "same panels", "new_original_sort": "l",
                                        "l": "m (clone GC-sort fraction)"},
              "layout_rationale": "Two paired gut rows: programme cell/clone views, then original sort cell/clone views",
              "original_sort_source": sort_metadata, "scientific_statistical_results_changed": False}
    base.ax_at, base.left_key, base.shared_key = ax_at, left_key, shared_key
    fig = page(HEIGHT)
    base.main_upper(fig, datasets, w, previous["panels"], report)
    clarify_ghi(fig, previous["panels"])
    draw_s3(fig, datasets["S3"], display, sort_metadata, previous["panels"], report)
    report["main_layout"] = validate(fig, "v6 main", HEIGHT, args.dpi)
    for key in "abcdefghi":
        assert report["panels"][key] == previous["panels"][key], key
    report["frozen_statistics_a_i_identical_to_v4"] = True
    detail = page(GUT_HEIGHT)
    detail_report = {"panels": {}}
    draw_s3(detail, datasets["S3"], display, sort_metadata, previous["panels"], detail_report, shift=GUT_SHIFT)
    report["gut_detail_layout"] = validate(detail, "v6 gut detail", GUT_HEIGHT, args.dpi)
    if args.preview:
        for obj, name in [(fig, "main"), (detail, "gut")]:
            path = Path(f"/tmp/clone_states_v6_{name}_preview.png")
            obj.savefig(path, dpi=180)
            if name == "main":
                preview = HERE / "Figure_clone_states_combined_v6_preview.png"
                obj.savefig(preview, dpi=300)
                print("Review PNG:", preview, flush=True)
            plt.close(obj)
            print("Preview:", path, flush=True)
        print("PC sort counts:", sort_metadata["original_sort_counts_in_author_plasma_cells"], flush=True)
    else:
        print(f"Stage: export main figure and gut detail at {args.dpi} dpi", flush=True)
        fig.savefig(HERE / "Figure_clone_states_combined_v6_preview.png", dpi=300)
        fs.save(fig, "Figure_clone_states_combined_v6", folder=HERE, dpi=args.dpi)
        fs.save(detail, "Figure_clone_states_combined_v6_gut_detail", folder=HERE, dpi=args.dpi)
        path = HERE / "reserve" / "provenance" / "figure_clone_states_stats_v6.json"
        path.write_text(json.dumps(ref.native(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        print("Statistics:", path, flush=True)
        print("Notes:", write_notes(report), flush=True)


if __name__ == "__main__":
    main()
