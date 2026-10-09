#!/usr/bin/env python
"""Public GSE266219 NSCLC RNA/TCR preparation and Threadfin profile analysis.

This is a reproducibility/sensitivity analysis of the public processed Seurat
objects used by Clonotrace.  Biological clones require an exact paired TRA+TRB
CDR3 amino-acid key within a patient.  A clone-cycle is the profiling unit so
longitudinal occurrences remain separate observations linked by that key.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/data/scratch/projects/punim1236/threadfin_data/gse266219_nsclc")
PRIVATE = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/nsclc")
OUT = HERE / "results" / "clonotrace_nsclc"
RSCRIPT = Path("/data/scratch/projects/punim1236/python_envs/rseurat/bin/Rscript")
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/GSE266219/suppl/"
FILES = [f"GSE266219_Merged_p{x}_CD8_sorted_and_PBMC_extracted.rds.gz"
         for x in (12, 13, 25, 27, 38, 3, 4, 69, 6, 7)]


def paired_tcr(value: object) -> str | None:
    """Return canonical TRA|TRB CDR3 key only when both chains occur once."""
    if not isinstance(value, str):
        return None
    chains = {}
    for part in value.split(";"):
        m = re.fullmatch(r"(TRA|TRB):(.+)", part.strip())
        if m:
            chains.setdefault(m.group(1), []).append(m.group(2))
    if set(chains) != {"TRA", "TRB"} or len(chains["TRA"]) != 1 or len(chains["TRB"]) != 1:
        return None
    return f"TRA:{chains['TRA'][0]}|TRB:{chains['TRB'][0]}"


def outer_to_inner() -> None:
    """GEO's .rds.gz files contain an outer gzip around an inner gzipped RDS."""
    def fetch(name):
        outer = DATA / name
        if not outer.exists():
            partial = outer.with_suffix(outer.suffix + ".part")
            subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
                            "--connect-timeout", "30", "--continue-at", "-", "-o", str(partial), BASE + name], check=True)
            partial.replace(outer)
        inner = DATA / name.replace(".rds.gz", "_inner.rds.gz")
        if not inner.exists():
            partial = inner.with_suffix(inner.suffix + ".part")
            with gzip.open(outer, "rb") as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            partial.replace(inner)
        h = hashlib.sha256()
        with outer.open("rb") as src:
            for block in iter(lambda: src.read(8*1024*1024), b""): h.update(block)
        print("Downloaded and checked", name, flush=True)
        return {"file": name, "url": BASE+name, "bytes": outer.stat().st_size, "sha256": h.hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(fetch, FILES))
    (DATA/"sources.json").write_text(json.dumps(sources, indent=2)+"\n")


def export_r() -> Path:
    exported = PRIVATE / "exported"
    if not (exported / "EXPORT_COMPLETE").exists():
        subprocess.run([str(RSCRIPT), str(HERE / "export_clonotrace_nsclc.R"), str(DATA), str(exported)], check=True)
    return exported


def read_export(path: Path):
    features = pd.read_csv(path / "features.tsv", sep="\t", header=None)[0].astype(str).to_numpy()
    barcodes = pd.read_csv(path / "barcodes.tsv", sep="\t", header=None)[0].astype(str)
    meta = pd.read_csv(path / "cell_metadata.csv").set_index("cell_id").loc[barcodes]
    x = scipy.io.mmread(path / "counts.mtx").tocsr().T.tocsr()  # cells x genes
    return x, barcodes, features, meta


def analyse() -> None:
    import anndata as ad
    import scanpy as sc
    import umap
    import threadfin as tf

    OUT.mkdir(parents=True, exist_ok=True)
    x, cells, genes, meta = read_export(export_r())
    meta["tcr_pair"] = meta["cdr3s_aa"].map(paired_tcr)
    meta["cycle"] = meta["PtCycle"].astype(str)
    meta["clone_biological"] = np.where(meta.tcr_pair.notna(), meta.patient.astype(str) + "|" + meta.tcr_pair, pd.NA)
    meta["clone_cycle"] = np.where(meta.tcr_pair.notna(), meta.clone_biological.astype(str) + "|" + meta.cycle, pd.NA)
    n_pair = meta.tcr_pair.notna().sum()
    # Exact shared CDR3 pair is primary.  The author clonotype_id is audited,
    # not used to link timepoints because its identifier scope is undocumented.
    consistency = (meta.dropna(subset=["tcr_pair"])
                   .groupby(["patient", "clonotype_id"])["tcr_pair"].nunique().rename("n_exact_pairs").reset_index())
    consistency.to_csv(OUT / "author_clonotype_exact_pair_audit.csv", index=False)
    meta.to_csv(OUT / "cell_metadata.csv.gz", compression="gzip")

    keep = meta.clone_cycle.notna().to_numpy()
    a = ad.AnnData(X=x[keep], obs=meta.iloc[np.flatnonzero(keep)].copy(), var=pd.DataFrame(index=genes))
    a.obs_names = cells.iloc[np.flatnonzero(keep)].to_numpy()
    # Common counts, common gene order, one global PCA.  Patient/cycle is not
    # batch-corrected away; hence temporal state structure remains observable.
    tf.pp.prepare_embedding(a, batch_key=None, integrate=None, key_added="X_threadfin", random_state=0, verbose=True)
    sc.pp.neighbors(a, use_rep="X_threadfin", n_neighbors=15, random_state=0)
    a.obsm["X_umap"] = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(a.obsm["X_threadfin"])
    for representation in ("mean", "kernel"):
        tf.tl.clone_profiles(a, clone_key="clone_cycle", basis="X_threadfin", context_key="patient",
                             donor_key="patient", representation=representation, min_cells=2, random_state=0,
                             verbose=True)
        prof = a.uns["threadfin"]["profiles"]
        tab = prof["clone_table"].copy()
        feats = prof["features"].copy()
        # Recover biological clone and cycle exactly from the controlled IDs.
        members = a.obs.groupby("clone_cycle", observed=True).agg(
            patient=("patient", "first"), cycle=("cycle", "first"), biological_clone=("clone_biological", "first"),
            author_clonotype_id=("clonotype_id", "first"), n_cells_observed=("clone_cycle", "size"),
            source=("scRXN", lambda x: ";".join(sorted(pd.unique(x.astype(str)))))
        )
        tab = tab.join(members)
        tab.to_csv(OUT / f"profiles_{representation}.csv.gz", compression="gzip")
        feats.to_csv(OUT / f"features_{representation}.csv.gz", compression="gzip")
        good = tab.index[tab.reliability >= .5]
        if len(good) >= 3:
            xy = umap.UMAP(n_neighbors=min(15, len(good)-1), min_dist=.3, random_state=0).fit_transform(feats.loc[good])
            coords = tab.loc[good, ["patient", "cycle", "biological_clone", "reliability", "n_cells"]].copy()
            coords[["x", "y"]] = xy
            coords.to_csv(OUT / f"clone_umap_{representation}.csv.gz", compression="gzip")
    # Explicitly observed same biological clone across cycles: no arrows or
    # claims of differentiation; displacement is a profile-coordinate summary.
    ktab = pd.read_csv(OUT / "profiles_kernel.csv.gz", index_col=0)
    linked = ktab.groupby(["patient", "biological_clone"], dropna=True).filter(lambda z: z.cycle.nunique() >= 2)
    linked.to_csv(OUT / "longitudinal_clone_cycle_links.csv.gz", compression="gzip")
    per_patient = (ktab.groupby("patient").agg(clone_cycles=("cycle", "size"), biological_clones=("biological_clone", "nunique"),
                       reliable_clone_cycles=("reliability", lambda z: int((z >= .5).sum())))
                   .join(linked.groupby("patient").biological_clone.nunique().rename("biological_clones_multiple_cycles"), how="left")
                   .fillna(0))
    per_patient.to_csv(OUT / "per_patient_summary.csv")
    summary = {"n_exported_cells": int(x.shape[0]), "n_common_genes": int(x.shape[1]),
               "n_cells_exact_paired_tcr": int(n_pair), "n_cells_analysed": int(a.n_obs),
               "n_patients": int(meta.patient.nunique()), "profile_unit": "exact paired TRA+TRB biological clone x cycle",
               "rna_representation": "global PCA from common genes; no patient/cycle batch correction",
               "limits": ["Observed clone-cycle displacement is not differentiation or efficacy prediction.",
                          "Patient-to-response mapping must be supplied/audited separately from GEO sample metadata.",
                          "Cells lacking an unambiguous paired TRA+TRB CDR3 are excluded from biological clone links."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("download", "export", "analyse"), default="analyse")
    args = parser.parse_args(); DATA.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    outer_to_inner()
    if args.stage == "download": return
    export_r()
    if args.stage == "analyse":
        from reanalyse_clonotrace_nsclc import main as reanalyse
        reanalyse()


if __name__ == "__main__": main()
