#!/usr/bin/env python3
"""Recluster real GSE246382 families with fixed presets and measured-gate audits.

Reuses the committed donor-restricted family calls and cell coordinates.
Raw expression is independently aligned and embedded without receptor genes.
Gate labels and marker expression never enter parameter selection or clustering.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import adjusted_rand_score

import threadfin as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data"))
GATES = ["light zone", "Myc+ light zone", "dark zone", "plasma cell"]
MARKERS = {
    "GC identity": ["Bcl6", "Aicda", "Rgs13", "S1pr2"],
    "Cycling": ["Top2a", "Mki67", "Ube2c", "Stmn1"],
    "Plasma cell": ["Prdm1", "Xbp1", "Sdc1", "Jchain", "Mzb1", "Irf4"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, pipeline="notebook"):
    output.mkdir(parents=True, exist_ok=True)
    folder = DATA / "gse246382_np_pc"
    files = [folder / name for name in (
        "GSE246382_umiDedup-Exact.mtx.gz", "GSE246382_features.tsv.gz",
        "GSE246382_barcodes.tsv.gz", "cell_metadata.csv")]
    source = ROOT / "case_studies/results/gc_np_pc/cells.csv.gz"
    saved = pd.read_csv(source, index_col=0)
    x = scipy.io.mmread(files[0]).tocsr().T.tocsr().astype(np.float32)
    feats = pd.read_csv(files[1], sep="\t", header=None)
    barcodes = pd.read_csv(files[2], header=None)[0].astype(str)
    data = ad.AnnData(x, obs=pd.DataFrame(index=barcodes.values),
                      var=pd.DataFrame(index=feats[1].values))
    data.var_names_make_unique()
    assert saved.index.is_unique and saved.index.isin(data.obs_names).all()
    data = data[saved.index].copy()
    qc = np.asarray((data.X > 0).sum(axis=1)).ravel() >= 500
    assert qc.all(), "Saved cells do not match the original >=500-gene QC."
    raw_meta = pd.read_csv(files[3], index_col=0).reindex(data.obs_names)
    assert raw_meta["mouse"].astype(str).equals(saved["donor"].astype(str))
    assert raw_meta["compartment"].astype(str).equals(saved["compartment"].astype(str))
    data.obs = saved.copy()
    data.obsm["X_umap"] = saved[["umap_1", "umap_2"]].to_numpy()
    tf.pp.prepare_embedding(data, batch_key=None, n_top_genes=3000, n_comps=30,
                            random_state=123, key_added="X_gc_reclustering")
    print("Raw counts and frozen family calls aligned:", data.shape, flush=True)

    # Marker summaries use log-normalised RNA; they do not define the graph.
    marker_genes = sorted(set(["Myc", "Top2a"] + sum(MARKERS.values(), [])))
    present = [g for g in marker_genes if g in data.var_names]
    expression = data.layers["log_norm"][:, data.var_names.get_indexer(present)]
    expression = expression.toarray() if hasattr(expression, "toarray") else expression
    genes = pd.DataFrame(expression, index=data.obs_names, columns=present)
    scores = genes.copy()
    z = (genes - genes.mean()) / genes.std(ddof=0).replace(0, 1)
    coverage = {}
    for name, wanted in MARKERS.items():
        available = [g for g in wanted if g in genes]
        coverage[name] = {"requested": wanted, "present": available}
        scores[name] = z[available].mean(axis=1)
    cells = saved.join(scores.add_prefix("marker:"))
    cells.to_csv(output / "cells.csv.gz", compression="gzip")
    clone_scores = scores.groupby(data.obs["clone_id"], observed=True).mean()
    occupancy = pd.crosstab(data.obs["clone_id"], data.obs["fate"]).reindex(columns=GATES, fill_value=0)
    fractions = occupancy.div(occupancy.sum(axis=1), axis=0)
    rows, partitions = [], {}
    configurations = {p: {"preset": p} for p in tf.reclustering_presets()}
    if pipeline == "notebook":
        configurations["notebook"] = {"preset": "continuous", "resolution": .3}
    for name_prefix, settings in configurations.items():
        for seed in (123, 7):
            settings = dict(settings)
            if pipeline == "notebook":
                settings.update(n_neighbors=15,
                                umap_n_neighbors=tf.reclustering_presets()[settings["preset"]]["n_neighbors"],
                                embedding_mode="distance_profiles", cluster_on="embedding")
            result = tf.clonotype_recluster(
                data, basis="X_umap" if pipeline == "notebook" else "X_gc_reclustering",
                min_clone_size=3, random_state=seed, copy=True, **settings,
            )
            cm = result.uns["threadfin"]["clone_map"].copy()
            config = result.uns["threadfin"]["clonotype_recluster"]
            cm = cm.join(fractions.add_prefix("gate:"), how="left")
            cm = cm.join(clone_scores.add_prefix("marker:"), how="left")
            cm["dominant_gate"] = fractions.reindex(cm.index).idxmax(axis=1)
            cm["gate_purity"] = fractions.reindex(cm.index).max(axis=1)
            # Ties are explicitly mixed rather than resolved by column order.
            tied = fractions.reindex(cm.index).eq(cm["gate_purity"], axis=0).sum(axis=1) > 1
            cm.loc[tied, "dominant_gate"] = "mixed"
            name = f"{name_prefix}_seed{seed}"
            cm.to_csv(output / (name + "_clone_map.csv"))
            (output / (name + "_parameters.json")).write_text(json.dumps(config, indent=2) + "\n")
            partitions[name] = cm["clone_cluster"].astype(str)
            rows.append({"preset": name_prefix, "seed": seed, "n_clones": len(cm),
                         "n_clone_clusters": cm.clone_cluster.nunique(),
                         "mean_gate_purity": float(cm.gate_purity.mean()),
                         "gate_partition_ari": adjusted_rand_score(cm.dominant_gate, cm.clone_cluster),
                         "n_cells": int(cm.n_cells.sum())})
    pd.DataFrame(rows).to_csv(output / "preset_summary.csv", index=False)
    stability = {p: adjusted_rand_score(partitions[f"{p}_seed123"], partitions[f"{p}_seed7"])
                 for p in configurations}
    primary_name = "notebook_seed123" if pipeline == "notebook" else "continuous_seed123"
    primary = pd.read_csv(output / f"{primary_name}_clone_map.csv", index_col=0)
    marker_cols = ["marker:Myc", "marker:GC identity", "marker:Cycling", "marker:Plasma cell"]
    state_summary = primary.groupby("dominant_gate")[marker_cols].mean()
    state_summary.to_csv(output / "clone_marker_by_dominant_gate.csv")
    primary.groupby("clone_cluster")[marker_cols + ["gate:" + g for g in GATES]].mean().to_csv(
        output / "clone_cluster_annotations.csv")
    summary = {
        "dataset": "GSE246382", "design": "NP-OVA/Alhydrogel, day 14; FACS GC zones/Myc+ LZ and plasma cells",
        "status": "completed", "job_id": os.environ.get("SLURM_JOB_ID"),
        "n_cells": data.n_obs, "n_cells_with_bcr": int(data.obs.clone_id.notna().sum()),
        "n_donors": data.obs.donor.nunique(), "retained_clones": len(primary),
        "retained_cells": int(primary.n_cells.sum()),
        "primary": primary_name,
        "basis": "frozen receptor-excluded cell UMAP" if pipeline == "notebook" else "receptor-excluded 30-component PCA",
        "pipeline": pipeline,
        "selection": "Historical executed notebook geometry/graph settings and seed 123 fixed before examining gates/markers; presets and both seeds retained" if pipeline == "notebook" else "continuous preset and seed 123 fixed before examining gates/markers; all three presets and both seeds retained",
        "recipe": "Cell UMAP clone centroids -> pairwise Euclidean distance rows -> Euclidean UMAP (20 neighbours, min_dist 0.4) -> Scanpy neighbours (15) -> Leiden (resolution 0.3)" if pipeline == "notebook" else "PCA clone centroids -> distance graph Leiden -> precomputed-distance UMAP",
        "clone_definition": "Committed same-mouse V/J/junction-sequence-defined families; never pooled across mice",
        "source_sha256": {str(p): sha(p) for p in files + [source]},
        "marker_coverage": coverage, "seed_partition_ari": stability,
        "versions": {p: metadata.version(p) for p in ("threadfin", "numpy", "scanpy", "umap-learn", "leidenalg")},
        "limitations": ["Exploratory centroid reclustering; not v4 reliability-filtered programme inference.",
                       "Dominant-gate and marker summaries describe captured cells, not future fate.",
                       "Only 49 same-mouse families have >=3 cells; capture limits state-mixture resolution.",
                       "The existing v4 coherence p=0.224 and three reliable profiles remain unchanged.",
                       "MYC-related gate is a measured selection-associated state, not a temporally traced selection event.",
                       "GC interpretation is restricted to this model-antigen GC dataset, not non-GC."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"primary": summary["primary"], "retained_clones": len(primary),
                      "seed_partition_ari": stability, "states": primary.dominant_gate.value_counts().to_dict()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("notebook", "pca"), default="notebook")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ("case_studies/results/gc_np_pc_clone_embedding" if args.pipeline == "notebook" else "case_studies/results/gc_np_pc_reclustering")
    run(output, args.pipeline)
