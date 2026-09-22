#!/usr/bin/env python
"""Real-data benchmark: Stephenson et al. 2021 paired GEX+BCR (COVID-19 PBMC).

Runs the full Threadfin workflow and writes figures + a JSON/Markdown report
into benchmarks/results/.

Usage: run_real_benchmark.py <h5mu_path> <outdir>
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import awkward as ak
import muon as mu
import numpy as np
import pandas as pd
import psutil

import threadfin as tf

H5MU, OUTDIR = sys.argv[1], Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)

report = {"dataset": "stephenson2021_5k (Stephenson et al. 2021, Nat Med)",
          "stages": {}}
proc = psutil.Process()


def stage(name):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            self.m0 = proc.memory_info().rss
            print(f"[stage] {name} ...", flush=True)
            return self

        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            dm = (proc.memory_info().rss - self.m0) / 1e6
            report["stages"][name] = {"seconds": round(dt, 3), "rss_delta_mb": round(dm, 1)}
            print(f"[stage] {name} done in {dt:.2f}s (RSS +{dm:.0f} MB)", flush=True)
    return _T()


print("[load]", H5MU, flush=True)
with stage("load_h5mu"):
    m = mu.read_h5mu(H5MU)
    gex, airr = m["gex"], m["airr"]

with stage("extract_bcr_table"):
    a = airr.obsm["airr"]
    if "bool" in str(ak.type(a["productive"])):
        productive = ak.fill_none(a["productive"], False)
    else:
        p = a["productive"]
        productive = (p == "T") | (p == "True") | (p == "1") | (p == 1)
    igh = a[(a["locus"] == "IGH") & productive]
    first = ak.firsts(igh)  # highest-priority productive IGH record per cell
    bcr = pd.DataFrame(
        {
            "v_call": ak.to_list(first["v_call"]),
            "d_call": ak.to_list(first["d_call"]),
            "j_call": ak.to_list(first["j_call"]),
            "cdr3": ak.to_list(first["cdr3_aa"]),
        },
        index=airr.obs_names,
    )
    bcr = bcr.dropna(subset=["v_call"])
    bcr = tf.build_clone_key(bcr, strategy="vdj")

with stage("attach_bcr"):
    adata = gex.copy()
    adata.obs["state"] = adata.obs["initial_clustering"].astype(str)
    tf.attach_bcr(adata, bcr)

n_clones = adata.obs["clone_id"].nunique()
report["n_cells"] = int(adata.n_obs)
report["n_clones"] = int(n_clones)
report["clone_size_median"] = float(adata.obs["clone_id"].value_counts().median())
print(f"[data] {adata.n_obs} cells, {n_clones} clones", flush=True)

with stage("clonotype_recluster"):
    tf.clonotype_recluster(
        adata, basis="X_umap", min_clone_size=3, n_neighbors=20,
        resolution=0.3, random_state=0,
    )

with stage("clonal_pseudotime"):
    tf.clonal_pseudotime(adata)

with stage("clonotype_recluster_cdr3_blend"):
    adata_seq = adata.copy()
    tf.clonotype_recluster(
        adata_seq, basis="X_umap", min_clone_size=3, n_neighbors=20,
        resolution=0.3, cdr3_weight=0.3, random_state=0,
        key_added="clone_cluster_seq",
    )
    adata.obs["clone_cluster_seq"] = adata_seq.obs["clone_cluster_seq"]
    del adata_seq

with stage("metrics"):
    conc = tf.metrics.state_concordance(adata, "clone_cluster", "state")
    conc_seq = tf.metrics.state_concordance(adata, "clone_cluster_seq", "state")
    purity = tf.metrics.clone_state_purity(adata, state_key="state")
    summary = tf.metrics.clone_cluster_summary(adata)
    enrichment = tf.metrics.state_enrichment(adata, "clone_cluster", "state")
    report["concordance_gex_only"] = conc
    report["concordance_gex_plus_cdr3"] = conc_seq
    report["clone_purity_median"] = float(purity["purity"].median())
    report["clone_purity_frac_gt_0.8"] = float((purity["purity"] > 0.8).mean())
    report["clone_cluster_summary"] = summary.to_dict()
    report["state_enrichment"] = enrichment.to_dict("records")
    enrichment.to_csv(OUTDIR / "state_enrichment.csv", index=False)
print("[metrics]", json.dumps(report["concordance_gex_only"]), flush=True)
print(enrichment.head(8).to_string(), flush=True)

SIGS = {
    "Naive B": ["TCL1A", "IGHD", "FCER2"],
    "Memory B": ["CD27", "CCR6", "S100A10"],
    "Plasmablast": ["PRDM1", "XBP1", "JCHAIN", "SDC1", "MZB1"],
    "IFN response": ["ISG15", "MX1", "IFI6"],
}

with stage("figures"):
    tf.plotting.clone_map(adata, color="clone_cluster",
                          save=str(OUTDIR / "clone_map_cluster.png"))
    tf.plotting.clone_map(adata, color="n_cells",
                          save=str(OUTDIR / "clone_map_size.png"))
    tf.plotting.clone_map(adata, color="clonal_pseudotime",
                          save=str(OUTDIR / "clone_map_pseudotime.png"))
    tf.plotting.cells(adata, color="clone_cluster",
                      save=str(OUTDIR / "cells_clone_cluster.png"))
    tf.plotting.cells(adata, color="state",
                      save=str(OUTDIR / "cells_state.png"))
    tf.plotting.signature_heatmap(
        adata, SIGS, groupby="clone_cluster",
        save=str(OUTDIR / "signature_heatmap.png"),
    )

(OUTDIR / "benchmark_report.json").write_text(json.dumps(report, indent=2))
print("[done] report ->", OUTDIR / "benchmark_report.json", flush=True)
