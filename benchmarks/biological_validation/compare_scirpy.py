#!/usr/bin/env python
"""Direct comparison: sequence-similarity clonotype networks vs Threadfin.

Produces a 5-panel figure per dataset:
  A  cells on UMAP colored by reference state
  B  conventional sequence-defined clonotype clusters (same V/J + CDR3
     similarity, the scirpy-style view) projected on the cell UMAP
  C  Threadfin clone map (one point per clonotype, state-space communities)
  D  cells colored by Threadfin clone community
  E  gene-signature heatmap per clone community

Usage:
    python compare_scirpy.py configs/<dataset>.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import threadfin as tf
import validation as V
from loaders import LOADERS

HERE = Path(__file__).parent


def conventional_clonotypes(adata, bcr):
    """Sequence-similarity clonotype clusters via threadfin's BCR graph
    (same-VJ blocked, CDR3 similarity threshold 0.85) — equivalent in
    spirit to scirpy's `define_clonotype_clusters` with the identity
    metric. Returns a pandas Series: cell -> sequence-cluster label."""
    from threadfin.bcrgraph import define_clones

    labels = define_clones(bcr, cdr3_sim_threshold=0.85, same_vj=True,
                           method="connected", out_col="seq_cluster")
    return labels.astype(str)


def main(cfg_path: str):
    cfg = json.load(open(cfg_path))
    name = cfg["name"]
    outdir = HERE / "results" / name / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    adata, bcr = LOADERS[cfg["loader"]](cfg)
    adata = V.standard_preprocess(adata)
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, basis="X_pca", min_clone_size=cfg.get("min_clone_size", 3),
                           resolution=0.3, random_state=0)

    # conventional sequence clonotypes
    conv = conventional_clonotypes(adata, bcr)
    adata.obs["conv_clonotype"] = adata.obs_names.map(
        conv.groupby(level=0).first()).astype("category")
    top_conv = adata.obs["conv_clonotype"].value_counts().head(20).index
    adata.obs["conv_clonotype_top"] = adata.obs["conv_clonotype"].where(
        adata.obs["conv_clonotype"].isin(top_conv))

    fig, axes = plt.subplots(1, 5, figsize=(30, 5.6))
    fig.suptitle(f"{name}: conventional clonotype network vs Threadfin", fontsize=16)

    _scatter_cells(adata, "state", axes[0], "A. cells by reference state")
    _scatter_cells(adata, "conv_clonotype_top", axes[1],
                   "B. sequence clonotypes (top 20) on cell UMAP")

    # Panel C: Threadfin clone map drawn onto axes[2]
    cm = adata.uns["threadfin"]["clone_map"]
    ax = axes[2]
    for i, cat in enumerate(pd.Categorical(cm["clone_cluster"]).categories):
        m = np.asarray(pd.Categorical(cm["clone_cluster"]) == cat)
        ax.scatter(cm["x"][m], cm["y"][m], s=np.log2(cm["n_cells"][m] + 1) * 30,
                   color=plt.get_cmap("tab20")(i % 20), label=str(cat), alpha=0.85)
    ax.set_title("C. Threadfin clone map (1 point = 1 clone)")
    ax.set_xlabel("Clone UMAP 1"); ax.set_ylabel("Clone UMAP 2")

    _scatter_cells(adata, "clone_cluster", axes[3], "D. cells by Threadfin community")

    # Panel E: signature heatmap into axes[4]
    if cfg.get("signatures"):
        tf.plotting.signature_heatmap(adata, cfg["signatures"],
                                      groupby="clone_cluster", ax=axes[4])
        axes[4].set_title("E. signatures per community")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / "comparison_five_panel.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"[{name}] comparison figure -> {out}", flush=True)

    # numeric contrast: how many conventional clonotypes span communities, and
    # how many communities mix sequence-unrelated clones
    both = adata.obs[["conv_clonotype", "clone_cluster"]].dropna()
    spread = both.groupby("conv_clonotype")["clone_cluster"].nunique()
    report = {
        "n_conventional_clonotypes": int(both["conv_clonotype"].nunique()),
        "conv_clonotypes_spanning_multiple_communities": int((spread > 1).sum()),
        "frac_spanning": float((spread > 1).mean()),
    }
    (outdir / "comparison_report.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def _scatter_cells(adata, color, ax, title):
    xy = adata.obsm["X_umap"]
    vals = adata.obs[color]
    if isinstance(vals.dtype, pd.CategoricalDtype):
        vals = vals.cat.remove_unused_categories()
    na = vals.isna().to_numpy()
    ax.scatter(xy[na, 0], xy[na, 1], s=3, color="lightgrey", alpha=0.25, rasterized=True)
    cats = pd.Categorical(vals)
    for i, cat in enumerate(cats.categories):
        m = np.asarray(cats == cat) & ~na
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.7, label=str(cat),
                   color=plt.get_cmap("tab20")(i % 20), rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False,
              markerscale=3, fontsize=6)


if __name__ == "__main__":
    main(sys.argv[1])
