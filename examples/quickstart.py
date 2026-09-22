#!/usr/bin/env python
"""Threadfin quickstart on public paired GEX+BCR data.

Downloads the Stephenson et al. 2021 COVID-19 PBMC dataset (5,000
BCR-containing cells, prepared by scirpy) and runs the full Threadfin
workflow. Runtime: ~1 minute.

    python examples/quickstart.py

Requires the benchmark extra: pip install muon awkward
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import awkward as ak
import muon as mu
import pandas as pd

import threadfin as tf

DATA = Path(__file__).parent / "stephenson2021_5k.h5mu"
if not DATA.exists():
    print("downloading example dataset ...")
    import subprocess

    subprocess.run(["bash", str(Path(__file__).parent.parent / "benchmarks" / "download_data.sh"),
                    str(DATA.parent)], check=True)

m = mu.read_h5mu(DATA)
gex, airr = m["gex"], m["airr"]

# --- extract a per-cell BCR table from the AIRR (awkward) representation ---
a = airr.obsm["airr"]
igh = a[(a["locus"] == "IGH") & ak.fill_none(a["productive"], False)]
first = ak.firsts(igh)
bcr = pd.DataFrame(
    {"v_call": ak.to_list(first["v_call"]), "d_call": ak.to_list(first["d_call"]),
     "j_call": ak.to_list(first["j_call"]), "cdr3": ak.to_list(first["cdr3_aa"])},
    index=airr.obs_names,
).dropna(subset=["v_call"])

# --- Threadfin workflow ---
bcr = tf.build_clone_key(bcr, strategy="vdj")          # v_call_d_call_j_call
adata = gex.copy()
adata.obs["state"] = adata.obs["initial_clustering"].astype(str)  # B_cell / Plasmablast
tf.attach_bcr(adata, bcr)

tf.clonotype_recluster(adata, basis="X_umap", min_clone_size=3, random_state=0)
tf.clonal_pseudotime(adata)

print("concordance (clone clusters vs curated cell states):",
      tf.metrics.state_concordance(adata, "clone_cluster", "state"))

tf.plotting.clone_map(adata, color="clone_cluster", save="clone_map.png")
tf.plotting.cells(adata, color="clone_cluster", save="cells_clone_cluster.png")
print("figures written: clone_map.png, cells_clone_cluster.png")
