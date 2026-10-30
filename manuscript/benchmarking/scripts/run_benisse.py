#!/usr/bin/env python
"""Benisse (Zhang et al. 2022, Nat. Mach. Intell.; wooyongc/Benisse v1.1.0), run as in its README.

1. ``AchillesEncoder.py`` embeds each cell's heavy-chain CDR3 amino-acid sequence (contrastive encoder,
   CPU);
2. ``Benisse.R`` learns a sparse graph and a latent space of the BCR clonotypes (unique heavy-chain
   V gene + CDR3 + J gene) supervised by the cells' expression, with the README's recommended
   hyperparameters (lambda2 1610, gamma 1, 100 iterations, lambda1 1, rho 1, m 10, stop 1e-10).
   The expression input is log(1 + x) of normalised counts of the 3,000 highly variable genes from
   which the shared embedding was computed (see prepare_benisse_expression.py for why not the
   embedding itself).

Benisse's unit is the exact clonotype, which is finer than a Dandelion clone (a clone groups somatic
variants of one rearrangement). Its latent distances are therefore aggregated to clones: the distance
between two clones is the mean latent distance over all pairs of their cells. Benisse receives the
cells of the clones with at least ``min_size`` cells (the clones that are evaluated), because its
optimisation scales with the square of the number of clonotypes.

Usage: python run_benisse.py <dataset> <variant> <min_size> <method_id>
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bench_io
from bench_paths import INPUTS, WORK

# a clone of wooyongc/Benisse at f0ff4c1, the Python that runs its encoder (torch) and the Rscript of its R core
SRC = Path(os.environ.get("BENISSE_SRC", WORK.parent / "src" / "Benisse"))
PYTHON = os.environ.get("BENISSE_PYTHON", sys.executable)
RSCRIPT = os.environ.get("BENISSE_RSCRIPT", "Rscript")
HYPER = ["1610", "1", "100", "1", "1", "10", "1e-10"]


def main(dataset, variant, min_size, method):
    out = bench_io.out_dir(dataset, variant, method)
    work = out / "benisse_io"
    work.mkdir(parents=True, exist_ok=True)
    x, cells = bench_io.load(dataset, variant)
    sizes = bench_io.eligible(cells, min_size)
    cdr3 = pd.read_csv(INPUTS / dataset / "cdr3.tsv.gz", sep="\t", index_col=0)
    cdr3.index = cdr3.index.astype(str)
    use = cells.index[cells["clone"].isin(sizes.index)].intersection(cdr3.index)
    if len(use) < 10:
        bench_io.fail(out, "too few cells with a CDR3", {"method": method})
        return
    # log-normalised expression of the 3,000 highly variable genes (prepare_benisse_expression.py);
    # Benisse's R code renames '-' to '.' in barcodes, so plain identifiers are used throughout
    import scipy.sparse as sp
    expr = sp.load_npz(INPUTS / dataset / "expr_hvg.npz").tocsr()
    expr_cells = pd.read_csv(INPUTS / dataset / "expr_hvg_cells.txt", header=None)[0].astype(str)
    genes = pd.read_csv(INPUTS / dataset / "expr_hvg_genes.txt", header=None)[0].astype(str)
    row = pd.Series(np.arange(len(expr_cells)), index=expr_cells.to_numpy())
    use = use[use.isin(row.index)]
    ids = pd.Index([f"c{i}" for i in range(len(use))])
    key = pd.Series(use, index=ids)
    sub = expr[row.loc[use].to_numpy()].toarray().T
    pd.DataFrame(sub, columns=ids, index=genes.to_numpy()).to_csv(work / "expression.csv", float_format="%.4g")
    c = cdr3.loc[use]
    pd.DataFrame({"contigs": [f"{i}_IGH" for i in ids], "cdr3": c["junction_aa"].to_numpy()}, index=ids).to_csv(work / "bcr_cdr3.csv")
    contigs = pd.DataFrame({
        "barcode": ids, "is_cell": "True", "contig_id": [f"{i}_IGH" for i in ids], "high_confidence": "True",
        "length": 0, "chain": "IGH", "v_gene": c["v_call"].to_numpy(), "d_gene": "None",
        "j_gene": c["j_call"].to_numpy(), "c_gene": "None", "full_length": "True", "productive": "True",
        "cdr3": c["junction_aa"].to_numpy(), "cdr3_nt": "N", "reads": 1, "umis": 1,
        "raw_clonotype_id": "None", "raw_consensus_id": "None"})
    contigs.to_csv(work / "contigs.csv", index=False)

    timer = {}
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    t0 = time.perf_counter()
    p = subprocess.run([PYTHON, str(SRC / "AchillesEncoder.py"), "--input_data", str(work / "bcr_cdr3.csv"),
                        "--output_data", str(work / "encoded.csv"), "--cuda", "False"],
                       capture_output=True, text=True, env=env, cwd=SRC)
    timer["encoder"] = round(time.perf_counter() - t0, 3)
    (work / "encoder.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        bench_io.fail(out, "encoder failed: " + p.stderr[-1500:], {"method": method, "seconds": timer})
        return
    t0 = time.perf_counter()
    p = subprocess.run([RSCRIPT, str(SRC / "Benisse.R"), str(work / "expression.csv"), str(work / "contigs.csv"),
                        str(work / "encoded.csv"), str(work / "result"), *HYPER],
                       capture_output=True, text=True, env=env, cwd=SRC)
    timer["benisse_core"] = round(time.perf_counter() - t0, 3)
    (work / "benisse_core.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        bench_io.fail(out, "Benisse.R failed: " + p.stderr[-1500:], {"method": method, "seconds": timer})
        return
    t0 = time.perf_counter()
    res = work / "result"
    meta = pd.read_csv(res / "clone_annotation.csv", index_col=0)
    clonotype = (meta["v_gene"].astype(str) + "_" + meta["cdr3"].astype(str) + "_" + meta["j_gene"].astype(str)).to_numpy()
    lat = np.loadtxt(res / "latent_dist.txt")
    cleaned = pd.read_csv(res / "cleaned_exp.txt", sep=" ", nrows=1).columns.astype(str)
    labels = pd.read_csv(res / "clonality_label.txt", header=None)[0].astype(str).to_numpy()
    cell_type = pd.Series(labels, index=cleaned)               # Benisse clonotype of every cell it kept
    idx = pd.Series(np.arange(len(clonotype)), index=clonotype)
    kept = cell_type.index.intersection(ids)
    cell_clone = cells.loc[key.loc[kept].to_numpy(), "clone"].astype(str).to_numpy()
    cell_ct = idx.loc[cell_type.loc[kept].to_numpy()].to_numpy()
    clones = pd.Index(pd.unique(cell_clone))
    member = np.zeros((len(clones), len(clonotype)))
    np.add.at(member, (clones.get_indexer(cell_clone), cell_ct), 1.0)
    w = member / member.sum(axis=1, keepdims=True)
    dist = w @ lat @ w.T                                       # mean over cell pairs of the two clones
    np.fill_diagonal(dist, 0.0)
    timer["aggregate"] = round(time.perf_counter() - t0, 3)
    n_cells = pd.Series(cell_clone).value_counts().reindex(clones).to_numpy()
    info = {"method": method, "software": "Benisse 1.1.0 (f0ff4c1)", "dataset": dataset, "variant": variant,
            "min_size": min_size, "n_cells_input": int(len(use)), "n_cells_kept_by_benisse": int(len(kept)),
            "n_clonotypes": int(len(clonotype)), "n_clones": int(len(clones)),
            "n_clones_eligible": int(len(sizes)), "hyperparameters": HYPER, "seconds": timer}
    bench_io.write(out, clones, n_cells, distances=dist, info=info)
    print(f"{dataset} {variant} {method}: {len(clones)} clones from {len(clonotype)} clonotypes, {timer}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
