#!/usr/bin/env python
"""Map saved clone GC fractions and original GC/nonGC labels onto the saved scRNA UMAP.

No expression processing, UMAP refitting, or fluorescence-intensity inference is done.
Run: python map_gc_derived_to_scrna_umap.py [--metadata PATH] [--out PATH] [--dpi 600]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))
import figstyle as fs

CASE = HERE.parents[3] / "case_studies" / "results" / "gut_gc_reporter"
DEFAULT_METADATA = Path(
    "/data/scratch/projects/punim1236/threadfin_data/gse284622_gut_gc/gex_reporter/metadata.csv.gz"
)
MISSING = "#d5d5d5"
GATE_COLORS = {"GC": "#d94845", "nonGC": "#2a78d6"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: pd.Series) -> dict:
    values = values.dropna().astype(float)
    return {"n": int(len(values)), "min": float(values.min()),
            "max": float(values.max()), "mean": float(values.mean()),
            "median": float(values.median())}


def load_and_validate(metadata_path: Path):
    cells_path, clones_path = CASE / "cells.csv.gz", CASE / "clone_table.csv"
    cells = pd.read_csv(cells_path, index_col=0, low_memory=False)
    clones = pd.read_csv(clones_path, index_col="clone_id", low_memory=False)
    meta = pd.read_csv(metadata_path, index_col="cell", low_memory=False)
    assert cells.index.is_unique and clones.index.is_unique and meta.index.is_unique
    assert cells.index.isin(meta.index).all(), "Unmatched UMAP cell IDs"
    meta = meta.loc[cells.index]
    for cell_column, meta_column in [("sample", "sample_id"), ("tissue", "tissue"),
                                     ("celltype", "celltype")]:
        assert cells[cell_column].equals(meta[meta_column].rename(cell_column)), cell_column
    assert cells["donor"].eq("mouse" + meta["mouse"].astype(int).astype(str)).all()
    assert meta["gc"].isin(GATE_COLORS).all(), "Missing or unexpected original GC labels"
    assert np.isfinite(cells[["umap_1", "umap_2"]].to_numpy()).all()
    assert cells["sample"].str.endswith("_GC").eq(meta["gc"].eq("GC")).all()

    mapped = cells.copy()
    mapped.index.name = "cell"
    mapped["gc_sort_original"] = meta["gc"]
    mapped["gc_sort_indicator"] = meta["gc"].eq("GC").astype(int)
    counts = mapped.groupby("clone_id").agg(
        n_cells_recomputed=("gc_sort_original", "count"),
        n_gc_cells=("gc_sort_indicator", "sum"),
        gc_fraction_recomputed=("gc_sort_indicator", "mean"),
    ).reindex(clones.index)
    counts["n_nongc_cells"] = counts["n_cells_recomputed"] - counts["n_gc_cells"]
    counts["n_cells_saved"] = clones["n_cells"]
    counts["gc_fraction_saved"] = pd.to_numeric(clones["gc_sort:GC"], errors="raise")
    counts["absolute_difference"] = (
        counts["gc_fraction_saved"] - counts["gc_fraction_recomputed"]
    ).abs()
    assert counts.notna().all().all(), "Clone table IDs missing from the cell table"
    assert counts["n_cells_saved"].eq(counts["n_cells_recomputed"]).all()
    assert np.allclose(counts["gc_fraction_saved"], counts["gc_fraction_recomputed"],
                       rtol=0, atol=1e-12), "Saved fractions disagree with original sort labels"
    assert counts["gc_fraction_saved"].between(0, 1).all()
    mapped["clone_gc_sort_fraction"] = mapped["clone_id"].map(counts["gc_fraction_saved"])
    mapped["has_saved_clone_fraction"] = mapped["clone_gc_sort_fraction"].notna()
    mapped["fraction_mapping_status"] = np.select(
        [mapped["has_saved_clone_fraction"], mapped["clone_id"].isna()],
        ["saved_clone_fraction", "no_clone_id"], default="clone_not_in_saved_table"
    )
    assert mapped.groupby("clone_id")["clone_gc_sort_fraction"].nunique().le(1).all()
    assert mapped[["umap_1", "umap_2"]].equals(cells[["umap_1", "umap_2"]])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GSE284622 gut_gc_reporter",
        "inputs": {str(p.resolve()): {"sha256": sha256(p)}
                   for p in [cells_path, clones_path, metadata_path]},
        "umap_source": "Existing scRNA coordinates: cells.csv.gz columns umap_1, umap_2",
        "umap_refitted": False,
        "cell_join": "Exact unique cell ID, with sample/tissue/celltype/donor cross-checks",
        "clone_join": "Exact saved Dandelion clone_id, including donor prefix",
        "clone_fraction_definition": "GC-labelled captured cells / all GC-or-nonGC-labelled captured cells in each clone",
        "clone_fraction_source": "clone_table.csv: gc_sort:GC (mapped without replacing saved values)",
        "original_sort_source": "Original Seurat meta.data exported as metadata.csv.gz: gc",
        "measurement_level": {"clone_fraction": "Captured-cell sort proportion per clone",
                              "original_sort": "Per-cell categorical GC/nonGC library sort label"},
        "continuous_tdTomato_fluorescence_available_in_metadata": False,
        "original_metadata_columns": meta.reset_index().columns.tolist(),
        "n_cells_total": int(len(mapped)),
        "n_cells_exactly_matched_to_metadata": int(len(meta)),
        "n_cells_with_clone_id": int(mapped["clone_id"].notna().sum()),
        "n_clones_with_cells": int(mapped["clone_id"].nunique()),
        "n_clones_in_saved_fraction_table": int(len(clones)),
        "n_cells_with_saved_clone_fraction": int(mapped["has_saved_clone_fraction"].sum()),
        "n_cells_without_saved_clone_fraction_gray": int((~mapped["has_saved_clone_fraction"]).sum()),
        "fraction_mapping_status_counts": mapped["fraction_mapping_status"].value_counts().to_dict(),
        "original_sort_counts_all_cells": mapped["gc_sort_original"].value_counts().to_dict(),
        "original_sort_counts_in_mapped_cells": mapped.loc[
            mapped["has_saved_clone_fraction"], "gc_sort_original"].value_counts().to_dict(),
        "original_sort_counts_by_celltype": pd.crosstab(
            mapped["celltype"], mapped["gc_sort_original"]).to_dict(orient="index"),
        "original_sort_counts_by_tissue": pd.crosstab(
            mapped["tissue"], mapped["gc_sort_original"]).to_dict(orient="index"),
        "saved_clone_fraction_summary_unweighted": describe(counts["gc_fraction_saved"]),
        "mapped_cell_fraction_summary_cell_weighted": describe(mapped["clone_gc_sort_fraction"]),
        "fraction_validation": {"all_clones_match": True,
                                "all_clone_sizes_match": True,
                                "max_absolute_error": float(counts["absolute_difference"].max())},
        "clones_by_fraction": {"all_nonGC": int(counts["gc_fraction_saved"].eq(0).sum()),
                               "mixed": int(counts["gc_fraction_saved"].between(0, 1, inclusive="neither").sum()),
                               "all_GC": int(counts["gc_fraction_saved"].eq(1).sum())},
        "clone_fraction_by_programme": clones.groupby("clone_programme")["gc_sort:GC"].agg(
            ["count", "mean", "median"]).to_dict(orient="index"),
        "n_sort_libraries": int(mapped["sample"].nunique()),
        "all_sort_labels_agree_with_sample_suffix": True,
        "color_scale": {"fraction": {"cmap": "viridis", "vmin": 0, "vmax": 1},
                        "original_sort": GATE_COLORS, "no_saved_fraction": MISSING},
    }
    return mapped, counts, report


def format_umap(ax, cells: pd.DataFrame):
    xy = cells[["umap_1", "umap_2"]].to_numpy()
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center, radius = (lo + hi) / 2, max(hi - lo) * .555
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("scRNA UMAP 1", fontsize=9, labelpad=5)
    ax.set_ylabel("scRNA UMAP 2", fontsize=9, labelpad=5)


def fraction_panel(fig, ax, cells: pd.DataFrame, cax):
    ok = cells["has_saved_clone_fraction"]
    ax.scatter(cells.loc[~ok, "umap_1"], cells.loc[~ok, "umap_2"],
               s=2.2, c=MISSING, alpha=.65, linewidths=0, rasterized=True)
    # Draw low-fraction cells last so rare nonGC-majority clones remain visible.
    colored = cells.loc[ok].sort_values("clone_gc_sort_fraction", ascending=False, kind="stable")
    ax.scatter(colored["umap_1"], colored["umap_2"], s=2.2,
               c=colored["clone_gc_sort_fraction"], cmap="viridis", vmin=0, vmax=1,
               alpha=.95, linewidths=0, rasterized=True)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), cax=cax)
    cb.set_ticks([0, .25, .5, .75, 1])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("GC-sort fraction within clone", fontsize=8.5, labelpad=5)
    format_umap(ax, cells)


def gate_panel(ax, cells: pd.DataFrame):
    # Interleave the two classes with a fixed seed to avoid class-order occlusion.
    display = cells.sample(frac=1, random_state=284622)
    ax.scatter(display["umap_1"], display["umap_2"], s=2.2,
               c=display["gc_sort_original"].map(GATE_COLORS),
               alpha=.8, linewidths=0, rasterized=True)
    format_umap(ax, cells)


def gate_legend(fig, cells: pd.DataFrame, center: float):
    counts = cells["gc_sort_original"].value_counts()
    handles = [Line2D([], [], marker="o", linestyle="", color=GATE_COLORS[k], markersize=5,
                      label=f"{k} sort (n = {counts[k]:,})") for k in GATE_COLORS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(center, .09),
               ncol=2, fontsize=8, handletextpad=.45, columnspacing=1.2)


def fraction_footer(fig, cells: pd.DataFrame, center: float):
    n = int(cells["has_saved_clone_fraction"].sum())
    handles = [Line2D([], [], marker="o", linestyle="", color=MISSING, markersize=5,
                      label=f"No saved clone fraction (n = {len(cells) - n:,})")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(center, .09), fontsize=8)
    fig.text(center, .055, f"{n:,} cells in 958 mapped clones", ha="center", fontsize=8)


def save_figure(fig, stem: str, out: Path, dpi: int):
    # Check visible text stays inside the exported page.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt in fig.findobj(match=matplotlib.text.Text):
        if txt.get_visible() and txt.get_text():
            box = txt.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, txt.get_text()
    fs.save(fig, stem, folder=out, dpi=dpi)


def draw_figures(cells: pd.DataFrame, out: Path, dpi: int):
    fig = plt.figure(figsize=(5.5, 5.8))
    ax = fig.add_axes([.11, .18, .71, .71])
    cax = fig.add_axes([.86, .18, .023, .71])
    fraction_panel(fig, ax, cells, cax)
    fig.text(.47, .966, "GSE284622 | GC-derived fraction on scRNA UMAP", ha="center",
             fontsize=10, fontweight="bold")
    fig.text(.47, .926, f"Saved clone GC-sort fraction | {len(cells):,} cells", ha="center", fontsize=9)
    fraction_footer(fig, cells, .47)
    save_figure(fig, "scrna_umap_gc_derived_fraction", out, dpi)

    fig = plt.figure(figsize=(5.5, 5.8))
    ax = fig.add_axes([.13, .18, .75, .71])
    gate_panel(ax, cells)
    fig.text(.5, .966, "GSE284622 | Original GC/nonGC sort on scRNA UMAP", ha="center",
             fontsize=10, fontweight="bold")
    fig.text(.5, .926, f"Original per-cell metadata: gc | {len(cells):,} cells", ha="center", fontsize=9)
    gate_legend(fig, cells, .5)
    fig.text(.5, .055, "GC/nonGC are categorical sort labels", ha="center", fontsize=8)
    save_figure(fig, "scrna_umap_gc_sort_label", out, dpi)

    fig = plt.figure(figsize=(11.4, 5.8))
    left = fig.add_axes([.055, .18, .361, .71])
    cax = fig.add_axes([.435, .18, .011, .71])
    right = fig.add_axes([.584, .18, .361, .71])
    fraction_panel(fig, left, cells, cax)
    gate_panel(right, cells)
    fig.text(.5, .969, "GSE284622 | GC-derived information mapped onto the existing scRNA UMAP",
             ha="center", fontsize=11, fontweight="bold")
    fig.text(.238, .922, "A  Saved clone GC-sort fraction", ha="center", fontsize=10)
    fig.text(.764, .922, "B  Original per-cell GC/nonGC sort", ha="center", fontsize=10)
    fraction_footer(fig, cells, .238)
    gate_legend(fig, cells, .764)
    fig.text(.764, .055, f"All {len(cells):,} cells have an original sort label", ha="center", fontsize=8)
    save_figure(fig, "scrna_umap_gc_derived_comparison", out, dpi)


def write_review(report: dict, out: Path):
    n = report["n_cells_total"]
    mapped = report["n_cells_with_saved_clone_fraction"]
    gates = report["original_sort_counts_all_cells"]
    statuses = report["fraction_mapping_status_counts"]
    mean = report["saved_clone_fraction_summary_unweighted"]["mean"]
    review = f"""# gut_gc_candidates 分析回顾与 GC-derived UMAP 映射

## 原分析在做什么

`fig_gut_gc_candidates.py` 读取 `case_studies/results/gut_gc_reporter/clone_table.csv`
和 `cells.csv.gz`。数据是 GSE284622 的 reporter 小鼠队列，包含 3 只小鼠的
MLN、PP、LP 样本。Dandelion 在每只小鼠内按配对重链/轻链定义克隆；共有
6,957 个克隆，其中 958 个被捕获至少 2 个细胞，进入保存的 clone map。

本文件夹原来的 `embedding_by_*.png` 都使用这 958 个克隆的同一组 `x/y`
坐标，每个点是一个克隆。点面积来自 `n_cells`，在 2–30 之间截断。
它们分别显示 dominant compartment、GC-sort 比例、多数 isotype、平均 V 区
mutation frequency、主要采样 tissue，以及 2/5/10 个细胞的显示门槛。
`cells_context.png` 使用另存的单细胞 `umap_1/umap_2` 坐标，只显示具有
克隆 ID 的 11,723 个细胞；它不是 clone map。

## GC sort 的计算和含义

原始 Seurat metadata 的 `gc` 列只有 `GC` 和 `nonGC` 两类。加载器
`case_studies/datasets.py::load_gut_gc_reporter` 将它保存为 `adata.obs['gc_sort']`。
配置以 `('gc_sort', 'fraction:GC', ...)` 调用
`threadfin/stats.py::clone_labels`，得到 clone table 的 `gc_sort:GC`。

对克隆 c：

```text
gc_sort:GC(c) = 该克隆中 gc_sort == 'GC' 的已捕获细胞数
               / 该克隆中具有分选标签的已捕获细胞总数
```

源码实际先去掉缺失的 clone ID/label，再按 clone ID 对 `(gc_sort == 'GC')`
取平均。本队列所有细胞都有 GC/nonGC 标签，所以分母就是该克隆被捕获的
细胞数。例如 5 个细胞中 4 个来自 GC 分选，则比例为 0.8。
0 表示该克隆被捕获的细胞全部来自 nonGC 分选，1 表示全部来自 GC 分选。

GC/nonGC 在此数据中是 tdTomato GC-fate reporter 相关的分选来源，
不是 RNA 注释 `GC_Bcell`。原始 metadata 的 24 列没有逐细胞的连续
tdTomato 荧光强度；`gc_sort:GC` 也不是平均荧光强度。它是捕获/分选样本内
的比例，不能直接当作组织内总体比例。nonGC 表示 reporter 阴性分选来源，
不能据此确定该细胞从未经过 GC。

保存的 958 个克隆比例无权平均为 {mean:.6f}（约 {mean * 100:.2f}%）；
这是每个克隆各占一个权重，不是所有 scRNA 细胞的 GC 阳性比例。
原始全体细胞中 GC 标签比例为 {gates['GC']}/{n} = {gates['GC'] / n * 100:.2f}%。

## 新生成的图

| 文件 | 着色内容 | 显示细胞 |
| --- | --- | --- |
| `scrna_umap_gc_derived_fraction.png/.pdf` | 已保存的克隆 `gc_sort:GC`，按 clone ID 映射回细胞；同一克隆共享一个值 | 全部 {n:,} 个细胞，其中 {mapped:,} 个有比例，其余灰色 |
| `scrna_umap_gc_sort_label.png/.pdf` | 每个细胞原始 metadata 的 `gc` 标签，GC 红色、nonGC 蓝色 | 全部 {n:,} 个细胞 |
| `scrna_umap_gc_derived_comparison.png/.pdf` | 上述两种映射并排对照，使用完全相同的坐标和坐标范围 | 两侧均为全部 {n:,} 个细胞 |

本次直接使用已有 scRNA UMAP 坐标，不重新拟合 UMAP。
958 个克隆覆盖 {mapped:,} 个细胞；灰色的 {n - mapped:,} 个细胞包含
{statuses['clone_not_in_saved_table']:,} 个不在保存的 clone table 内的细胞
（本队列均为 singleton），以及 {statuses['no_clone_id']:,} 个没有克隆 ID 的细胞。
灰色表示没有保存的克隆比例，不表示 nonGC。
原始 GC/nonGC 标签覆盖全部细胞：GC {gates['GC']:,} 个，nonGC {gates['nonGC']:,} 个。

## 映射核验与可复现文件

- 使用完整、唯一的原始 cell ID 精确匹配全部 {n:,} 个细胞，并交叉检查 sample、tissue、celltype、donor。
- 从原始 `gc` 标签重算全部 958 个克隆的比例，最大差值为
  {report['fraction_validation']['max_absolute_error']:.3g}（浮点舍入）；克隆细胞数全部一致。
- 按包含 mouse 前缀的 clone ID 映射原有比例，没有补算 singleton 的新比例。
- 原始标签与全部 17 个分选文库的 sample 后缀一致。
- UMAP 坐标全部有限，输出表保留原坐标和细胞顺序。

`gc_derived_umap_cells.csv.gz` 包含逐细胞坐标、原始标签、0/1 indicator、
保存的克隆比例和缺失原因。0/1 indicator 是标签编码，也不代表荧光强度。
`gc_sort_fraction_validation.csv` 保存 958 个克隆的原始/重算比例和 GC/nonGC
细胞数。`gc_derived_umap_summary.json` 保存输入 SHA256、计数、核验结果和
绘图参数。原有 `summary.json` 和候选图保持原文件内容。

重新生成：

```bash
python map_gc_derived_to_scrna_umap.py
```

默认输出为本文件夹的 600 dpi PNG 和 PDF；可用 `--metadata`、`--out`、
`--dpi` 指定 metadata 路径、输出目录和分辨率。
"""
    (out / "gc_sort_analysis_review.md").write_text(review, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert args.dpi > 0
    args.out.mkdir(parents=True, exist_ok=True)
    print("Read metadata; validate exact cell matches and saved clone fractions", flush=True)
    cells, counts, report = load_and_validate(args.metadata)
    print(f"Validated {len(cells):,} cells and {len(counts):,} clones; "
          f"{cells['has_saved_clone_fraction'].sum():,} cells receive saved fractions", flush=True)
    print("Export the two views and side-by-side comparison on the existing UMAP", flush=True)
    draw_figures(cells, args.out, args.dpi)
    cells.to_csv(args.out / "gc_derived_umap_cells.csv.gz", compression="gzip")
    counts.to_csv(args.out / "gc_sort_fraction_validation.csv")
    report["plot_dpi"] = args.dpi
    report["plot_population"] = "All cells with existing UMAP coordinates, identical in all new panels"
    report["text_bounds_validated"] = True
    report["outputs"] = [f"{stem}.{ext}" for stem in (
        "scrna_umap_gc_derived_fraction", "scrna_umap_gc_sort_label",
        "scrna_umap_gc_derived_comparison") for ext in ("png", "pdf")] + [
            "gc_derived_umap_cells.csv.gz", "gc_sort_fraction_validation.csv",
            "gc_sort_analysis_review.md", "gc_derived_umap_summary.json"]
    (args.out / "gc_derived_umap_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_review(report, args.out)
    print("Finished:", args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
