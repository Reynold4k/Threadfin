#!/usr/bin/env python
"""Run the standard Threadfin biological-validation pipeline for one dataset.

Usage:
    python run_validation.py configs/<dataset>.json

Produces under benchmarks/biological_validation/results/<name>/:
    qc.json, runs.json, robustness.json, null_permutation.json,
    null_random_communities.json, heldout.json, report.md, figures/*.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import threadfin as tf
import validation as V
from loaders import LOADERS

HERE = Path(__file__).parent


def main(cfg_path: str):
    cfg = json.load(open(cfg_path))
    name = cfg["name"]
    outdir = HERE / "results" / name
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{name}] loading via '{cfg['loader']}' ...", flush=True)

    adata, bcr = LOADERS[cfg["loader"]](cfg)
    print(f"[{name}] raw: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{len(bcr)} BCR records", flush=True)

    adata = V.standard_preprocess(adata, batch_key=cfg.get("batch_key"))
    tf.attach_bcr(adata, bcr, clone_col="clone_id")
    if "state" not in adata.obs.columns:
        raise RuntimeError("standard_preprocess should guarantee a 'state' column")

    # ---- QC ----
    qc = V.compute_qc(adata, state_key="state")
    qc["description"] = cfg.get("description", "")
    (outdir / "qc.json").write_text(json.dumps(qc, indent=2, cls=V.NpEncoder))
    print(f"[{name}] QC: {qc['n_clonotypes']} clones, "
          f"{qc['expanded_clones_min3']} expanded (>=3 cells)", flush=True)

    # ---- parameter grid (incl. basis comparison) ----
    bases = [b for b in ("X_pca", "X_umap") if b in adata.obsm]
    adata, runs = V.run_parameter_grid(adata, bases=bases)
    (outdir / "runs.json").write_text(json.dumps(runs, indent=2, cls=V.NpEncoder))

    main_key = "tf__pca__r0.3__cw0.0" if "X_pca" in bases else f"tf__umap__r0.3__cw0.0"
    umap_key = "tf__umap__r0.3__cw0.0"

    # ---- basis robustness ----
    robust = None
    if len(bases) == 2:
        robust = V.basis_robustness(adata, main_key, umap_key)
        (outdir / "robustness.json").write_text(json.dumps(robust, indent=2))
        print(f"[{name}] basis robustness: clone-level ARI "
              f"{robust['clone_level_ari']:.3f}", flush=True)

    # ---- null models (on the main run) ----
    print(f"[{name}] null 1 (label permutation) ...", flush=True)
    null1 = V.null_permutation_purity(adata, donor_key=cfg.get("donor_key"),
                                      seed=0)
    (outdir / "null_permutation.json").write_text(json.dumps(null1, indent=2))
    print(f"[{name}]   real purity {null1['real_mean_purity']:.3f} vs null "
          f"{null1['null_mean']:.3f} (p={null1['p_value']:.4f})", flush=True)

    print(f"[{name}] null 3 (random communities) ...", flush=True)
    null3 = V.null_random_communities(adata, cluster_key=main_key, seed=0)
    (outdir / "null_random_communities.json").write_text(json.dumps(null3, indent=2))

    # ---- held-out split-clone validation ----
    print(f"[{name}] held-out split validation ...", flush=True)
    heldout = V.heldout_split_validation(adata, basis="X_pca" if "X_pca" in bases else "X_umap")
    (outdir / "heldout.json").write_text(json.dumps(heldout, indent=2))
    print(f"[{name}]   co-cluster {heldout.get('cocluster_rate_mean', float('nan')):.2f} "
          f"vs chance {heldout.get('chance_rate_mean', float('nan')):.2f}", flush=True)

    # ---- figures ----
    print(f"[{name}] figures ...", flush=True)
    figs = outdir / "figures"
    tf.plotting.cells(adata, color="state", save=str(figs / "cells_state.png"))
    tf.plotting.cells(adata, color=main_key, save=str(figs / "cells_clone_cluster.png"))
    # clone map from the main run (rerun cheaply to keep the clone map in uns)
    tf.clonotype_recluster(adata, basis="X_pca" if "X_pca" in bases else "X_umap",
                           min_clone_size=3, resolution=0.3, random_state=0)
    tf.clonal_pseudotime(adata)
    tf.plotting.clone_map(adata, color="clone_cluster",
                          save=str(figs / "clone_map.png"))
    tf.plotting.clone_map(adata, color="n_cells", save=str(figs / "clone_map_size.png"))
    if cfg.get("signatures"):
        tf.plotting.signature_heatmap(adata, cfg["signatures"], groupby="clone_cluster",
                                      save=str(figs / "signature_heatmap.png"))
    if len(bases) == 2:
        _plot_basis_comparison(adata, main_key, umap_key, figs / "basis_comparison.png")

    # null figure
    _plot_null(null1, figs / "null_purity.png")

    # ---- machine-readable summary + human-readable report ----
    summary = {
        "name": name, "description": cfg.get("description", ""),
        "qc": qc, "runs": runs, "robustness": robust,
        "null_permutation": {k: v for k, v in null1.items() if k != "nulls"},
        "null_random_communities": null3, "heldout": heldout,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, cls=V.NpEncoder))
    _write_report_md(summary, outdir / "report.md")
    print(f"[{name}] DONE in {summary['runtime_min']} min -> {outdir}", flush=True)


def _write_report_md(summary: dict, path):
    """Human-readable per-dataset validation report (Markdown)."""
    qc = summary["qc"]
    n1 = summary["null_permutation"]
    n3 = summary["null_random_communities"]
    ho = summary["heldout"]
    rob = summary.get("robustness")

    def fmt(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)

    lines = [
        f"# Threadfin validation report: {summary['name']}",
        "",
        summary["description"],
        "",
        "## Dataset QC",
        "",
        f"- cells: {qc['n_cells']:,}; BCR-positive: {qc['n_bcr_positive_cells']:,} "
        f"({qc['frac_bcr_positive']:.0%})",
        f"- clonotypes: {qc['n_clonotypes']:,}; expanded (>=3 cells): "
        f"{qc['expanded_clones_min3']:,}; (>=5): {qc['expanded_clones_min5']:,}",
        f"- clone size: median {qc['clone_size_median']}, max {qc['clone_size_max']}",
        "",
    ]
    for k, v in qc.items():
        if k.startswith("levels_"):
            lines.append(f"- {k[7:]}: {v}")
    lines += ["", "## Threadfin runs (parameter grid)", "",
              "| run | clusters | NMI vs state | ARI vs state | #sig enrichments |",
              "|---|---|---|---|---|"]
    for r in summary["runs"]:
        c = r.get("concordance") or {}
        lines.append(
            f"| {r['key']} | {r['n_clone_clusters']} | {fmt(c.get('nmi'))} | "
            f"{fmt(c.get('ari'))} | {r.get('n_significant_enrichments', '-')} |")
    lines += ["", "## Robustness and statistical support", ""]
    if rob:
        lines.append(
            f"- **Basis robustness** (PCA vs UMAP): clone-level ARI "
            f"**{rob['clone_level_ari']:.3f}**, NMI {rob['clone_level_nmi']:.3f} "
            f"over {rob['n_clones_compared']} clones. Low ARI means fine-grained "
            "community boundaries shift between bases; check whether the "
            "biologically significant enrichments replicate across bases.")
    lines += [
        f"- **Null 1 (clone-label permutation, within donor)**: observed mean clone "
        f"state-purity **{n1['real_mean_purity']:.3f}** vs null "
        f"{n1['null_mean']:.3f} ± {n1['null_sd']:.3f}; p = {n1['p_value']:.4f}",
        f"- **Null 3 (size-matched random communities)**: observed "
        f"{n3['real_n_significant']} significant state enrichments vs null "
        f"{n3['null_mean']:.2f} ± {n3['null_sd']:.2f}; p = {n3['p_value']:.4f}",
    ]
    if "error" not in ho:
        lines.append(
            f"- **Held-out split-clone validation**: sibling halves of the same clone "
            f"co-cluster at **{ho['cocluster_rate_mean']:.2f}** ± {ho['cocluster_rate_sd']:.2f} "
            f"vs chance {ho['chance_rate_mean']:.2f} "
            f"({ho['n_big_clones']} clones x {ho['n_repeats']} repeats, basis={ho['basis']})")
    else:
        lines.append(f"- **Held-out validation**: skipped ({ho['error']})")
    lines += ["", f"Runtime: {summary['runtime_min']} min.", ""]
    path.write_text("\n".join(lines))


def _plot_basis_comparison(adata, key_a, key_b, path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    xy = adata.obsm["X_umap"]
    for ax, key, title in zip(axes, (key_a, key_b), ("PCA basis", "UMAP basis")):
        vals = adata.obs[key]
        na = vals.isna().to_numpy()
        ax.scatter(xy[na, 0], xy[na, 1], s=4, color="lightgrey", alpha=0.3, rasterized=True)
        for i, cat in enumerate(pd.Categorical(vals).categories):
            m = (np.asarray(pd.Categorical(vals) == cat)) & ~na
            ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.7, label=str(cat),
                       color=plt.get_cmap("tab20")(i % 20), rasterized=True)
        ax.set_title(f"clone clusters ({title})")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_null(null1, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null1["nulls"], bins=30, color="grey", alpha=0.7, label="permutation null")
    ax.axvline(null1["real_mean_purity"], color="red", lw=2,
               label=f"observed (p={null1['p_value']:.4f})")
    ax.set_xlabel("mean clone state-purity")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
