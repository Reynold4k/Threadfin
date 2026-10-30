#!/usr/bin/env python
"""Reanalyse the WATERMELON drug-persister barcoding experiment (GSE150949) with Threadfin.

Oren et al. Nature 2021: PC9 lung-cancer cells under osimertinib, lineage-barcoded and
sampled at days 0/3/7/14; day-14 cells were sorted into mCherry high/med/low gates
(proliferation dilution reporter: high = non-cycling persister, low = cycling).

Clone identity is the lentiviral lineage barcode (not a BCR), so no Dandelion step.
Only single-barcode cells are used as clone members (25,370 cells, 4,636 lineages);
cells carrying multiple comma-separated barcodes (10,422) are set to clone_id=NA.
The measured labels the embedding never sees: sampling day and the mCherry gate.

Outputs: case_studies/results/watermelon/
  cells.csv.gz, clone_table.csv, clone_map_{k{n}_md{d}}.csv, figures, summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import threadfin as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parents[3] / "case_studies" / "data")) / "gse150949_watermelon"
OUT = HERE / "results" / "watermelon"
OUT.mkdir(parents=True, exist_ok=True)

GATE = {"14_high": "mCherry-high", "14_med": "mCherry-med", "14_low": "mCherry-low"}


def load():
    meta = pd.read_csv(DATA / "GSE150949_metaData_with_lineage.txt.gz", sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    meta["day"] = meta["time_point"].astype(int)
    meta["gate"] = meta["sample_type"].map(GATE)
    # 10,422 cells carry 2-3 comma-separated barcodes (multiple lentiviral integrations);
    # they are excluded so that every clone_id is one unambiguous lineage tag.
    lb = meta["lineage_barcode"]
    meta["clone_id"] = lb.where(lb.notna() & ~lb.str.contains(",", na=False))
    cache = DATA / "counts_sparse.npz"
    genes_f = DATA / "gene_names.txt"
    if cache.exists():
        x = sp.load_npz(cache)
        genes = pd.read_csv(genes_f, header=None)[0].astype(str).to_numpy()
    else:
        print("reading 22k x 56k count csv (a few minutes)", flush=True)
        raw = pd.read_csv(DATA / "GSE150949_pc9_count_matrix.csv.gz", index_col=0)
        genes = raw.index.astype(str).to_numpy()
        x = sp.csr_matrix(raw.to_numpy(dtype=np.float32).T)  # cells x genes
        sp.save_npz(cache, x)
        pd.Series(genes).to_csv(genes_f, index=False, header=False)
        del raw
    assert x.shape == (len(meta), len(genes)), (x.shape, len(meta), len(genes))
    a = ad.AnnData(X=x, obs=meta[["nGene", "nUMI", "percent.mito", "day", "sample_type",
                                  "gate", "clone_id", "clone_size"]].copy(),
                   var=pd.DataFrame(index=pd.Index(genes)))
    a.var_names_make_unique()
    return a


def main():
    t0 = time.time()
    def stamp(m): print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

    a = load()
    stamp(f"loaded {a.n_obs} cells x {a.n_vars} genes; "
          f"{a.obs['clone_id'].notna().sum()} with a lineage barcode")
    a.obs["donor"] = "pc9"
    a.obs["sample"] = a.obs["sample_type"].astype(str)

    stamp("embedding")
    tf.pp.prepare_embedding(a, batch_key=None, key_added="X_threadfin", verbose=True)

    # clone profiles on the lineage barcode, no filters (project default)
    stamp("clone profiles")
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key=None, donor_key=None,
                         representation="kernel", min_cells=2, verbose=True)
    prof = a.uns["threadfin"]["profiles"]
    stamp(f"profiles: {len(prof['clone_table'])} clones")

    stamp("clone map")
    tab = tf.tl.embed_clones(a, n_neighbors=30, min_dist=0.5, min_reliability=0.0, verbose=True)
    tab = tab.copy()

    # per-clone measured labels
    for label, how in (("gate", "fraction:mCherry-high"), ("gate", "fraction:mCherry-low"),
                       ("day", "mean")):
        if label == "day":
            v = a.obs.dropna(subset=["clone_id"]).groupby("clone_id")["day"].mean()
            tab["mean_day"] = v.reindex(tab.index)
        else:
            tab[label + ":" + how.split(":")[1]] = tf.tl.clone_labels(
                a, label, how=how).reindex(tab.index)
    tab.to_csv(OUT / "clone_table.csv")

    # measured-label associations with the high-dimensional profiles
    summary = {"n_cells": int(a.n_obs), "n_clones_profiled": int(len(tab)),
               "cells_with_barcode": int(a.obs["clone_id"].notna().sum()),
               "clone_definition": "single WATERMELON lineage barcode per cell "
                                   "(multi-barcode cells excluded)"}
    effects = {}
    for label, how, note in (
        ("gate", "fraction:mCherry-high", "fraction of the clone's day-14 cells in the non-cycling "
                                          "persister gate (mCherry-high, measured)"),
        ("gate", "fraction:mCherry-low", "fraction of the clone's day-14 cells in the cycling gate "
                                         "(mCherry-low, measured)"),
    ):
        if a.obs[label].notna().sum() < 20:
            continue
        r = tf.tl.profile_association(a, label, how=how, n_perm=2000, strata_key="auto", verbose=True)
        r["note"] = note
        effects[f"{label}:{how.split(':', 1)[1]}"] = {k: v for k, v in r.items()
                                                      if k != "axis_correlation"}
    summary["label_effects"] = effects

    # --- early -> late design (leakage-free): clone profiles built ONLY from day 0/3/7
    # cells, tested against the clone's day-14 measured gate fractions.
    stamp("early->late design")
    d14 = a.obs[(a.obs["day"] == 14) & a.obs["clone_id"].notna()]
    ok14 = d14.groupby("clone_id").size()
    ok14 = ok14[ok14 >= 2].index
    d14 = d14[d14["clone_id"].isin(ok14)]
    early = a[a.obs["day"] < 14].copy()
    for g in ("mCherry-high", "mCherry-low"):
        frac = d14.groupby("clone_id")["gate"].apply(lambda s: float((s == g).mean()))
        early.obs[f"gate14:{g}"] = early.obs["clone_id"].map(frac)
    tf.pp.prepare_embedding(early, batch_key=None, key_added="X_threadfin", verbose=False)
    tf.tl.clone_profiles(early, basis="X_threadfin", context_key=None, donor_key=None,
                         representation="kernel", min_cells=2, verbose=True)
    tab_e = tf.tl.embed_clones(early, n_neighbors=30, min_dist=0.5, min_reliability=0.0,
                               verbose=True).copy()
    for g in ("mCherry-high", "mCherry-low"):
        tab_e[f"gate14:{g}"] = tf.tl.clone_labels(early, f"gate14:{g}",
                                                  how="mean").reindex(tab_e.index)
    tab_e.to_csv(OUT / "early_clone_table.csv")
    early_effects = {}
    for g in ("mCherry-high", "mCherry-low"):
        key = f"gate14:{g}"
        if early.obs[key].notna().sum() < 20:
            continue
        r = tf.tl.profile_association(early, key, how="mean", n_perm=2000,
                                      strata_key="auto", verbose=True)
        r["note"] = (f"profiles from day 0/3/7 cells only; label = fraction of the clone's "
                     f"day-14 cells in the {g} gate (measured; clones with >=2 day-14 cells)")
        early_effects[key] = {k: v for k, v in r.items() if k != "axis_correlation"}
    summary["early_to_late"] = {"n_early_cells": int(early.n_obs),
                                "n_clones_profiled_early": int(len(tab_e)),
                                "n_clones_with_day14_label": int(len(ok14)),
                                "label_effects": early_effects}

    # baseline with clone-mean profiles (what the single-cell average alone carries)
    stamp("early->late, mean-profile baseline")
    tf.tl.clone_profiles(early, basis="X_threadfin", context_key=None, donor_key=None,
                         representation="mean", min_cells=2, verbose=True)
    mean_effects = {}
    for g in ("mCherry-high", "mCherry-low"):
        key = f"gate14:{g}"
        if early.obs[key].notna().sum() < 20:
            continue
        r = tf.tl.profile_association(early, key, how="mean", n_perm=2000,
                                      strata_key="auto", verbose=True)
        mean_effects[key] = {k: v for k, v in r.items() if k != "axis_correlation"}
    summary["early_to_late"]["label_effects_mean_profiles"] = mean_effects

    # --- candidate figure: all-time map and early-only map, coloured by measured gate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, t, col, title in (
        (axes[0], tab, "gate:mCherry-high",
         "A  all-time clone map (descriptive)\ncolour = measured mCherry-high fraction"),
        (axes[1], tab_e, "gate14:mCherry-high",
         "B  profiles from day 0/3/7 only\ncolour = measured day-14 mCherry-high fraction"),
    ):
        v = t[col]
        sc = ax.scatter(t["x"], t["y"], c=v, cmap="viridis", vmin=0, vmax=1,
                        s=6 + 2.5 * np.sqrt(t["n_cells"]), alpha=0.85,
                        linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Clone UMAP 1"); ax.set_ylabel("Clone UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8, label="measured mCherry-high fraction")
    fig.suptitle("WATERMELON GSE150949 - Threadfin clone maps vs measured persister gate", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "clone_map_mcherry.png", dpi=200)
    fig.savefig(OUT / "clone_map_mcherry.pdf")
    plt.close(fig)

    # cells table for figures
    cells = a.obs[["clone_id", "day", "sample_type", "gate", "clone_size", "nUMI", "nGene"]].copy()
    cells.to_csv(OUT / "cells.csv.gz")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    stamp("done")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
