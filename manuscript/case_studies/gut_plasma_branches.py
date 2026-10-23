#!/usr/bin/env python
"""GSE284622 gut GC reporter: what are the two plasma-cell branches on the clone map?

The clone embedding separates plasma-dominant clones into two regions. This script tests
whether that split is explained by annotations the embedding never saw (isotype, tissue,
measured GC-fate sort, donor, SHM, clone size), which genes distinguish the branches, and
whether the same split is visible on the cell-level scRNA UMAP at all.

Outputs: case_studies/results/gut_gc_reporter/plasma_branches.{json,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "gut_gc_reporter"


def frac_table(cells: pd.DataFrame, col: str) -> pd.DataFrame:
    c = cells[cells["clone_id"].notna()]
    t = pd.crosstab(c["clone_id"].astype(str), c[col].astype(str))
    return t.div(t.sum(axis=1), axis=0)


def main():
    tab = pd.read_csv(RES / "clone_table.csv").set_index("clone_id")
    cells = pd.read_csv(RES / "cells.csv.gz")
    cells = cells.set_index(cells.columns[0])          # first column holds the cell barcode
    cells.index = cells.index.astype(str)
    cells["clone_id"] = cells["clone_id"].astype(str)

    # 1. where the programmes sit on the clone map; which are plasma
    prog = tab.groupby("clone_programme").agg(
        n=("n_cells", "size"), x=("x", "median"), y=("y", "median"),
        plasma_frac=("celltype:plasma", "mean"), gc_frac=("celltype:GC_Bcell", "mean"))
    print("== programme overview ==\n", prog.round(3).to_string(), flush=True)

    plasma = tab[tab["celltype:plasma"] >= 0.5].copy()
    # 8 plasma-dominant clones sit in the GC programme P1 (bridge clones); the two plasma
    # branches are P2 and P3
    plasma = plasma[plasma["clone_programme"].isin(("P2", "P3"))]
    branches = sorted(plasma["clone_programme"].unique())
    print(f"\nplasma-dominant clones in P2/P3: {len(plasma)}; "
          f"{plasma['clone_programme'].value_counts().to_dict()}", flush=True)
    assert len(branches) == 2, f"expected 2 plasma branches, got {branches}"
    b0, b1 = branches

    # 2. annotation comparison between the two branches
    tissue_frac = frac_table(cells, "tissue")
    per_clone = pd.DataFrame({
        "isotype": tab["isotype"],
        "mu": tab["mutation_frequency"],
        "gc_sort": tab["gc_sort:GC"],
        "n_cells": tab["n_cells"],
        "donor": tab["donor"],
        "x": tab["x"], "y": tab["y"],
    }).join(tissue_frac)
    per_clone = per_clone.loc[plasma.index]
    per_clone["branch"] = plasma["clone_programme"]

    out = {"n_plasma_clones": int(len(plasma)),
           "branches": {b: int((per_clone["branch"] == b).sum()) for b in branches},
           "programme_overview": json.loads(prog.round(4).to_json())}

    print("\n== branch comparison (median per clone; Mann-Whitney) ==", flush=True)
    comp = {}
    for col in ("mu", "gc_sort", "n_cells", "LP", "PP", "MLN"):
        if col not in per_clone.columns:
            continue
        a = per_clone.loc[per_clone["branch"] == b0, col].dropna()
        b = per_clone.loc[per_clone["branch"] == b1, col].dropna()
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        comp[col] = {b0: float(a.median()), b1: float(b.median()), "p": float(u.pvalue)}
        print(f"  {col:9s} {b0}: {a.median():.3f}  {b1}: {b.median():.3f}   p = {u.pvalue:.2e}",
              flush=True)
    out["branch_comparison"] = comp

    for col in ("isotype", "donor"):
        ct = pd.crosstab(per_clone["branch"], per_clone[col].astype(str))
        chi2 = stats.chi2_contingency(ct)
        out[f"branch_{col}_table"] = ct.to_dict()
        out[f"branch_{col}_chi2_p"] = float(chi2.pvalue)
        print(f"\n== {col} x branch ==\n{ct.to_string()}\n  chi2 p = {chi2.pvalue:.2e}", flush=True)

    # 3. marker genes between the two plasma programmes (from the case-study markers table)
    markers = pd.read_csv(RES / "programme_markers.csv")
    mcols = list(markers.columns)
    print("\nmarker table columns:", mcols, flush=True)
    sub = markers[markers["programme"].isin(branches)] if "programme" in mcols else markers
    for b in branches:
        bb = sub[sub["programme"] == b] if "programme" in mcols else sub
        key = "score" if "score" in bb.columns else bb.columns[-1]
        top = bb.sort_values(key, ascending=False).head(12)
        out[f"markers_{b}"] = top.to_dict("records")
        print(f"\n== top markers {b} ==\n{top.head(12).to_string()}", flush=True)

    # 4. is the split visible on the CELL-level scRNA UMAP?
    #    neighbourhood agreement of the branch label among a plasma cell's 15 nearest cells
    #    (cells inherit their clone's branch), vs the same agreement on the clone map.
    pc = cells[(cells["celltype"] == "plasma") &
               cells["clone_id"].isin(plasma.index)].copy()
    pc["branch"] = pc["clone_id"].map(plasma["clone_programme"])
    from sklearn.neighbors import NearestNeighbors

    def knn_agreement(xy: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
        idx = nn.kneighbors(return_distance=False)[:, 1:]
        return float((labels[idx] == labels[:, None]).mean())

    rng = np.random.default_rng(0)
    cell_xy = pc[["umap_1", "umap_2"]].to_numpy()
    cell_lab = pc["branch"].to_numpy()
    agree_cell = knn_agreement(cell_xy, cell_lab)
    null_cell = float(np.mean([knn_agreement(cell_xy, rng.permutation(cell_lab), 15)
                               for _ in range(50)]))
    pmap = plasma[["x", "y"]].to_numpy()
    plab = plasma["clone_programme"].to_numpy()
    agree_clone = knn_agreement(pmap, plab, 15)
    null_clone = float(np.mean([knn_agreement(pmap, rng.permutation(plab), 15)
                                for _ in range(50)]))
    out["visibility"] = {
        "cell_umap_knn_agreement": agree_cell, "cell_umap_null": null_cell,
        "clone_map_knn_agreement": agree_clone, "clone_map_null": null_clone,
        "n_plasma_cells": int(len(pc))}
    print(f"\n== visibility ==\ncell UMAP 15-NN branch agreement {agree_cell:.3f} "
          f"(null {null_cell:.3f}, {len(pc):,} plasma cells)\nclone map  15-NN branch agreement "
          f"{agree_clone:.3f} (null {null_clone:.3f}, {len(plasma):,} clones)", flush=True)

    # 4b. donor confound: mouse3 contributes only P2 clones. Repeat the key comparisons
    # within mouse2 and mouse4, where both branches exist.
    print("\n== within-donor branch comparison ==", flush=True)
    wd = {}
    for d in ("mouse2", "mouse4"):
        sub = per_clone[per_clone["donor"] == d]
        a = sub[sub["branch"] == b0]
        b = sub[sub["branch"] == b1]
        wd[d] = {"n_P2": int(len(a)), "n_P3": int(len(b))}
        print(f"  {d}: P2 n={len(a)}, P3 n={len(b)}", flush=True)
        for col in ("mu", "gc_sort", "n_cells", "LP"):
            av, bv = a[col].dropna(), b[col].dropna()
            if len(av) < 5 or len(bv) < 5:
                continue
            u = stats.mannwhitneyu(av, bv)
            wd[d][col] = {"P2": float(av.median()), "P3": float(bv.median()), "p": float(u.pvalue)}
            print(f"    {col:8s} P2 {av.median():.3f}  P3 {bv.median():.3f}  p = {u.pvalue:.2e}",
                  flush=True)
    out["within_donor"] = wd

    # 4c. canonical plasma-maturation genes, cell level, from the dataset loader
    import sys
    sys.path.insert(0, str(HERE))
    from datasets import LOADERS
    a, _bcr = LOADERS["gut_gc_reporter"]()
    a.obs_names = a.obs_names.astype(str)
    pc2 = pc[pc.index.isin(a.obs_names)]
    a = a[pc2.index]
    x = a.layers["log_norm"] if "log_norm" in a.layers else a.X
    genes = [g for g in ("Mki67", "Top2a", "Sdc1", "Xbp1", "Prdm1", "Irf4", "Irf8",
                         "Ccr10", "Mzb1", "Jchain", "Cxcr4", "S1pr1", "Itga4", "Itgb7",
                         "Bcl2", "Aicda") if g in a.var_names]
    expr = {}
    bl = pc2["branch"].to_numpy()
    for g in genes:
        colv = x[:, a.var_names.get_loc(g)]
        v = np.asarray(colv.todense()).ravel() if hasattr(colv, "todense") else np.asarray(colv).ravel()
        expr[g] = {"P2": float(v[bl == b0].mean()), "P3": float(v[bl == b1].mean()),
                   "ratio_P3_over_P2": float(v[bl == b1].mean() / max(v[bl == b0].mean(), 1e-9))}
    out["branch_gene_expression"] = expr
    print("\n== cell-level expression (log-norm mean, P2 vs P3) ==", flush=True)
    for g, e in expr.items():
        print(f"    {g:8s} P2 {e['P2']:.3f}  P3 {e['P3']:.3f}  x{e['ratio_P3_over_P2']:.2f}",
              flush=True)

    (RES / "plasma_branches.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\nwrote", RES / "plasma_branches.json", flush=True)


if __name__ == "__main__":
    main()
