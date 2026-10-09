#!/usr/bin/env python3
"""Audit notebook cell geometry, receptor grouping and clone-UMAP parameters.

Uses real GSE246382 counts; never reads points from an image. Pooled VDJ
groups are explicitly historical diagnostics, not donor-restricted clones.
Run prepare in the pinned 2024-era environment, then baseline or sweep.
"legacy" follows the notebook code: PCA implicitly selects the flagged HVGs.
"allgenes" explicitly disables that PCA mask as a sensitivity analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

import threadfin as tf

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "internal_validation/legacy_reproduction"
OUT = ROOT / "case_studies/results/gc_legacy_parameter_audit"
RAW = Path(os.environ.get("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data")) / "gse246382_np_pc"
GATES = ["light zone", "Myc+ light zone", "dark zone", "plasma cell"]
GENES = ["Myc", "Top2a", "Bcl6", "Aicda", "Mki67", "Prdm1", "Xbp1", "Jchain", "Mzb1", "Sdc1", "Irf4"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    files = [RAW / n for n in ("GSE246382_umiDedup-Exact.mtx.gz", "GSE246382_features.tsv.gz",
                              "GSE246382_barcodes.tsv.gz", "cell_metadata.csv", "GSE246382_trust_report.tsv.gz")]
    saved_path = ROOT / "case_studies/results/gc_np_pc/cells.csv.gz"
    saved = pd.read_csv(saved_path, index_col=0)
    names = pd.read_csv(files[2], header=None)[0].astype(str)
    features = pd.read_csv(files[1], sep="\t", header=None)
    x = mmread(files[0]).tocsr().T.tocsr().astype(np.float32)
    data = ad.AnnData(x, obs=pd.DataFrame(index=names.values), var=pd.DataFrame(index=features[1].values))
    data.var_names_make_unique()  # names only; no genes or matrix columns removed
    assert saved.index.equals(data.obs_names)
    raw_meta = pd.read_csv(files[3], index_col=0).reindex(data.obs_names)
    assert raw_meta.mouse.astype(str).equals(saved.donor.astype(str))
    data.obs = saved.copy()
    data.var["mt"] = data.var_names.str.startswith("mt")
    sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], percent_top=None, inplace=True)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    # Capture biological annotations before regression/scaling, not from PCA.
    gene_x = data[:, GENES].X.toarray()
    for j, g in enumerate(GENES):
        data.obs["marker:" + g] = gene_x[:, j]
    z = (gene_x - gene_x.mean(axis=0)) / np.maximum(gene_x.std(axis=0), 1e-12)
    data.obs["marker:Plasma cell"] = z[:, [GENES.index(g) for g in ("Prdm1", "Xbp1", "Jchain", "Mzb1", "Sdc1", "Irf4")]].mean(axis=1)
    data.obs["marker:Cycling"] = z[:, [GENES.index(g) for g in ("Top2a", "Mki67")]].mean(axis=1)
    sc.pp.highly_variable_genes(data, min_mean=.0125, max_mean=3, min_disp=.5, n_top_genes=5500)
    n_hvg = int(data.var.highly_variable.sum())
    print("Legacy cell preparation: all genes retained; PCA auto-selects flagged HVGs:", data.shape, flush=True)
    sc.pp.regress_out(data, ["total_counts", "pct_counts_mt"], n_jobs=2)
    sc.pp.scale(data, max_value=10)
    sc.tl.pca(data, n_comps=50, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(data, n_neighbors=15, n_pcs=40, random_state=0)
    sc.tl.umap(data, min_dist=.5, spread=1, random_state=0)
    sc.tl.leiden(data, resolution=.8, random_state=0)
    cells = data.obs.copy()
    cells[["legacy_umap_1", "legacy_umap_2"]] = data.obsm["X_umap"]
    report = pd.read_csv(files[4], sep="\t")
    report["cell"] = report.cid.str.replace(r"_\d+_S\d+_R1_001_assemble\d+$", "", regex=True)
    heavy = report[report.V.str.startswith("IGH")].copy()
    productive = ~heavy.CDR3aa.astype(str).str.contains(r"out_of_frame|\*|_|\?", regex=True)
    for productive_only, label in [(True, "productive"), (False, "all_heavy")]:
        use = heavy.loc[productive] if productive_only else heavy
        use = use.sort_values("#count", ascending=False, kind="stable").drop_duplicates("cell").set_index("cell")
        calls = use[["V", "D", "J"]].fillna(".")
        ids = calls.agg("_".join, axis=1)
        cells["pooled_vdj_" + label] = ids.reindex(cells.index)
        if productive_only:
            for column in ["V", "D", "J", "cid", "CDR3nt"]:
                cells["trust:" + column] = use[column].reindex(cells.index)
    vdj = cells.pooled_vdj_productive
    donor_vdj = pd.Series(pd.NA, index=cells.index, dtype="object")
    has_vdj = vdj.notna()
    donor_vdj[has_vdj] = (cells.donor.astype(str)[has_vdj].to_numpy()
                          + "|" + vdj[has_vdj].astype(str).to_numpy())
    cells["donor_vdj"] = donor_vdj
    assert cells.donor_vdj.isna().equals(cells.pooled_vdj_productive.isna())
    assert not cells.donor_vdj.dropna().str.endswith("|nan").any()
    cells.to_csv(OUT / "cells.csv.gz", compression="gzip")
    geometry = ad.AnnData(np.zeros((len(cells), 1), dtype=np.float32), obs=cells)
    geometry.obsm["X_legacy"] = data.obsm["X_umap"]
    geometry.obsm["X_current"] = saved[["umap_1", "umap_2"]].to_numpy()
    geometry.obsm["X_legacy_pca"] = data.obsm["X_pca"]
    geometry.write_h5ad(PRIVATE / "cell_geometry.h5ad")
    audit = {"dataset": "GSE246382", "status": "prepared", "job_id": os.environ.get("SLURM_JOB_ID"),
             "n_cells": len(cells), "n_genes": data.n_vars, "n_hvg_flagged_not_subsetted": n_hvg,
             "cell_recipe": "normalize 1e4, log1p, flag HVG5500 without subsetting, regress total_counts/pct_counts_mt, scale max10; PCA50 implicitly uses HVG5500, Scanpy neighbours15/PC40, UMAP min_dist0.5 seed0",
             "pca_parameters": data.uns["pca"]["params"],
             "versions": {p: metadata.version(p) for p in ("numpy", "scanpy", "anndata", "scikit-learn", "umap-learn", "numba", "igraph", "leidenalg")},
             "source_sha256": {str(p): sha(p) for p in files + [saved_path]},
             "grouping_counts": {c: int(cells[c].nunique()) for c in ("clone_id", "donor_vdj", "pooled_vdj_productive", "pooled_vdj_all_heavy")},
             "limitations": ["Original bigplasma.h5ad and IgBLAST result_df.csv are absent; deposited TRUST4 calls are an independently recorded receptor-call source.",
                             "Pooled VDJ groups merge mice and distinct junctions; they are diagnostic historical groups, never lineage calls.",
                             "No missing-receptor cells are grouped into an artificial clone.",
                             "Layouts and marker annotations do not measure future fate."]}
    (OUT / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["grouping_counts"]), flush=True)


def all_genes():
    """Compute the literal all-gene alternative without replacing legacy maps."""
    geometry = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    x = mmread(RAW / "GSE246382_umiDedup-Exact.mtx.gz").tocsr().T.tocsr().astype(np.float32)
    features = pd.read_csv(RAW / "GSE246382_features.tsv.gz", sep="\t", header=None)
    data = ad.AnnData(x, obs=geometry.obs.copy(), var=pd.DataFrame(index=features[1].values))
    data.var_names_make_unique()
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.regress_out(data, ["total_counts", "pct_counts_mt"], n_jobs=2)
    sc.pp.scale(data, max_value=10)
    sc.tl.pca(data, n_comps=50, svd_solver="arpack", random_state=0, mask_var=None)
    sc.pp.neighbors(data, n_neighbors=15, n_pcs=40, random_state=0)
    sc.tl.umap(data, min_dist=.5, spread=1, random_state=0)
    geometry.obsm["X_allgenes"] = data.obsm["X_umap"]
    geometry.obsm["X_allgenes_pca"] = data.obsm["X_pca"]
    geometry.obs[["allgenes_umap_1", "allgenes_umap_2"]] = data.obsm["X_umap"]
    geometry.obs.to_csv(OUT / "cells.csv.gz", compression="gzip")
    geometry.write_h5ad(PRIVATE / "cell_geometry.h5ad")
    audit = json.loads((OUT / "preparation.json").read_text())
    audit["cell_recipe"] = "normalize 1e4, log1p; HVG5500 flagged, not subsetted; regress total_counts/pct_counts_mt; scale max10; PCA50 auto-selects HVG5500; neighbours15/PC40; UMAP min_dist0.5 seed0"
    audit["pca_mask_correction"] = "Scanpy 1.10.1 _handle_mask_var selects highly_variable by default. Keeping the full AnnData does not mean PCA uses all genes. Original processed object/environment are missing."
    audit["allgenes_sensitivity"] = {"job_id": os.environ.get("SLURM_JOB_ID"), "n_genes_in_pca": data.n_vars,
        "pca_parameters": data.uns["pca"]["params"], "other_steps": "Same RNA recipe; explicit mask_var=None; no receptor-gene exclusion"}
    (OUT / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    rows = [run_one(geometry, "allgenes", definition, minimum)
            for definition in ("donor_sequence", "donor_vdj", "pooled_vdj") for minimum in (1, 3)]
    rows.append(run_one(geometry, "allgenes", "pooled_all_heavy", 1))
    pd.DataFrame(rows).to_csv(OUT / "allgenes_summary.csv", index=False)


def stability_runs():
    """Extend candidates fixed before marker inspection to seven UMAP seeds."""
    data = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    candidates = {"donor_sequence": [(20, .4), (40, .65), (40, .85), (80, .4), (80, .65), (80, .85)],
                  "donor_vdj": [(20, .4), (40, .65)], "pooled_vdj": [(20, .4), (80, .4)]}
    rows = []
    for definition, settings in candidates.items():
        for k, md in settings:
            for seed in (123, 7, 0, 1, 42, 99, 2024):
                rows.append(run_one(data, "legacy", definition, 1, k=k, md=md, seed=seed))
    pd.DataFrame(rows).to_csv(OUT / "stability_runs.csv", index=False)


def nearest(x, k=10):
    d = squareform(pdist(x)); np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def neighbour_jaccard(a, b):
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]))


def audit_results():
    """Quantify all outputs without selecting from gates or marker expression."""
    geometry = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    all_rows = []; stability = []
    for file in sorted((OUT / "maps").glob("*.json")):
        meta = json.loads(file.read_text()); row = meta["audit"].copy()
        cm = pd.read_csv(file.with_suffix(".csv"), index_col=0)
        xy = cm[["x", "y"]].to_numpy()
        original = cm[[c for c in cm if c.startswith("X_")]].to_numpy()
        row["neighbour_jaccard10_input"] = neighbour_jaccard(nearest(original), nearest(xy))
        for k in (5, 10, 15):
            graph = ad.AnnData(xy)
            sc.pp.neighbors(graph, n_neighbors=min(k, len(cm)-1), use_rep="X", random_state=0)
            nc, labels = connected_components(graph.obsp["connectivities"], directed=False)
            row[f"components_k{k}"] = int(nc)
            row[f"largest_component_fraction_k{k}"] = float(np.bincount(labels).max()/len(cm))
        row["coordinates_sha256"] = sha(file.with_suffix(".csv"))
        row["parameters_sha256"] = sha(file)
        all_rows.append(row)
    runs = pd.DataFrame(all_rows)
    runs.to_csv(OUT / "all_runs_audit.csv", index=False)
    for keys, rows in runs.groupby(["basis", "definition", "min_clone_size", "umap_k", "min_dist", "resolution"]):
        if len(rows) < 2: continue
        maps = {r.seed: pd.read_csv(OUT / "maps" / (r["name"] + ".csv"), index_col=0)
                for _, r in rows.iterrows()}
        pairs = []
        for seed_a, seed_b in combinations(sorted(maps), 2):
            a, b = maps[seed_a], maps[seed_b].reindex(maps[seed_a].index)
            ari = adjusted_rand_score(a.clone_cluster, b.clone_cluster)
            jac = neighbour_jaccard(nearest(a[["x", "y"]]), nearest(b[["x", "y"]]))
            disparity = procrustes(a[["x", "y"]], b[["x", "y"]])[2]
            pairs.append({"seed_a": seed_a, "seed_b": seed_b, "ari": ari, "jaccard10": jac, "procrustes_disparity": disparity})
        pairs = pd.DataFrame(pairs)
        stability.append(dict(zip(["basis", "definition", "min_clone_size", "umap_k", "min_dist", "resolution"], keys)) |
            {"n_seeds": len(rows), "seeds": ",".join(map(str, sorted(maps))),
             "min_trustworthiness": rows.trustworthiness_centroids.min(),
             "max_components_k15": int(rows.components_k15.max()),
             "min_largest_component_fraction_k15": rows.largest_component_fraction_k15.min(),
             "clusters_min": int(rows.n_clusters.min()), "clusters_max": int(rows.n_clusters.max()),
             "ari_min": pairs.ari.min(), "ari_median": pairs.ari.median(),
             "jaccard10_median": pairs.jaccard10.median(),
             "procrustes_disparity_median": pairs.procrustes_disparity.median()})
    pd.DataFrame(stability).to_csv(OUT / "seed_stability.csv", index=False)
    # Validate all missing-receptor entries and the typed donor/VDJ join.
    c = geometry.obs
    assert c.donor_vdj.isna().equals(c.pooled_vdj_productive.isna())
    assert c.clone_id.notna().sum() == 762 and c.donor_vdj.notna().sum() == 762
    print("Audited", len(runs), "maps; missing-receptor cells remain excluded", flush=True)


def verify_selected():
    """Independently execute the notebook geometry without Threadfin wrappers."""
    import umap
    chosen = json.loads((OUT / "selected.json").read_text())
    data = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    key = {"donor_vdj": "donor_vdj", "donor_sequence": "clone_id"}[chosen["definition"]]
    # get_basis() deliberately normalises package input to float64. A float32
    # group mean differs by <1e-6 but can change the subsequent UMAP optimum.
    df = pd.DataFrame(np.asarray(data.obsm['X_' + chosen['basis']], dtype=np.float64),
                      index=data.obs_names, columns=['x', 'y'])
    df['group'] = data.obs[key].astype(object)
    centres = df.dropna(subset=['group']).groupby('group')[['x', 'y']].mean()
    counts = df.groupby('group').size()
    centres = centres.loc[counts.ge(chosen['min_clone_size'])]
    distances = squareform(pdist(centres.to_numpy(), metric='euclidean'))
    xy = umap.UMAP(n_neighbors=chosen['umap_k'], min_dist=chosen['min_dist'],
                   spread=1, learning_rate=1, random_state=chosen['seed'], metric='euclidean').fit_transform(distances)
    g = ad.AnnData(xy)
    sc.pp.neighbors(g, n_neighbors=15, random_state=0)
    sc.tl.leiden(g, resolution=chosen['resolution'], random_state=0)
    expected = pd.read_csv(OUT / 'maps' / (chosen['name'] + '.csv'), index_col=0).reindex(centres.index)
    np.testing.assert_allclose(xy, expected[['x', 'y']].to_numpy(), atol=2e-5, rtol=1e-6)
    ari = adjusted_rand_score(expected.clone_cluster.astype(str), g.obs.leiden.astype(str))
    assert ari == 1
    result = {'job_id': os.environ.get('SLURM_JOB_ID'), 'name': chosen['name'],
              'method': 'Independent pandas means -> scipy Euclidean distance rows -> umap.UMAP -> Scanpy neighbours/Leiden; no Threadfin reclustering call',
              'input_dtype': 'float64, matching threadfin._utils.get_basis',
              'coordinates_allclose': True, 'partition_ari': ari,
              'max_absolute_coordinate_difference': float(np.max(np.abs(xy-expected[['x', 'y']].to_numpy()))),
              'not_claimed': 'Equivalence to unavailable historical processed objects or IgBLAST receptor calls'}
    (OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def run_one(data, basis, definition, minimum, k=20, md=.4, resolution=.3, seed=123, tag=None):
    keys = {"donor_sequence": "clone_id", "donor_vdj": "donor_vdj",
            "pooled_vdj": "pooled_vdj_productive", "pooled_all_heavy": "pooled_vdj_all_heavy"}
    clone_key = keys[definition]
    for column in ("clone_id", "donor_vdj", "pooled_vdj_productive",
                   "pooled_vdj_all_heavy", "fate", "donor"):
        if column in data.obs.columns and isinstance(data.obs[column].dtype, pd.CategoricalDtype):
            data.obs[column] = data.obs[column].astype(object)
    name = tag or f"{basis}__{definition}__min{minimum}__k{k}_md{md:.2f}_r{resolution:.2f}_seed{seed}"
    path = OUT / "maps" / (name + ".csv")
    parameters = OUT / "maps" / (name + ".json")
    if path.exists() and parameters.exists():
        print("Reuse completed", name, flush=True)
        return json.loads(parameters.read_text())["audit"]
    result = tf.clonotype_recluster(data, clone_key=clone_key, basis="X_" + basis,
        min_clone_size=minimum, preset="continuous", n_neighbors=15, umap_n_neighbors=k,
        min_dist=md, resolution=resolution, embedding_mode="distance_profiles",
        cluster_on="embedding", random_state=seed, cluster_random_state=0, copy=True)
    cm = result.uns["threadfin"]["clone_map"].copy()
    marker_columns = [c for c in data.obs if c.startswith("marker:")]
    means = data.obs.groupby(clone_key, observed=True)[marker_columns].mean()
    cm = cm.join(means)
    gates = pd.crosstab(data.obs[clone_key], data.obs.fate).reindex(columns=GATES, fill_value=0)
    fractions = gates.div(gates.sum(axis=1), axis=0)
    cm = cm.join(fractions.add_prefix("gate:"))
    cm["dominant_gate"] = fractions.reindex(cm.index).idxmax(axis=1)
    counts = data.obs.groupby(clone_key, observed=True).donor.nunique()
    cm["n_donors"] = counts.reindex(cm.index)
    xy = cm[["x", "y"]].to_numpy()
    graph = ad.AnnData(xy)
    sc.pp.neighbors(graph, n_neighbors=min(15, len(cm)-1), use_rep="X", random_state=0)
    n_components = connected_components(graph.obsp["connectivities"], directed=False)[0]
    dist = result.uns["threadfin"]["clone_distances"]
    nn = min(10, (len(cm)-1)//2)
    audit = {"name": name, "basis": basis, "definition": definition, "min_clone_size": minimum,
             "umap_k": k, "min_dist": md, "resolution": resolution, "seed": seed,
             "n_points": len(cm), "n_cells": int(cm.n_cells.sum()), "n_clusters": int(cm.clone_cluster.nunique()),
             "n_graph_components": int(n_components),
             "trustworthiness_centroids": float(trustworthiness(cm[[c for c in cm if c.startswith("X_")]].to_numpy(), xy, n_neighbors=nn)),
             "trustworthiness_distance_profiles": float(trustworthiness(dist, xy, n_neighbors=nn)),
             "max_group_cells": int(cm.n_cells.max()), "multi_donor_groups": int((cm.n_donors>1).sum()),
             "diagnostic_only": definition.startswith("pooled"), "job_id": os.environ.get("SLURM_JOB_ID")}
    path.parent.mkdir(parents=True, exist_ok=True)
    cm.to_csv(path)
    parameters.write_text(json.dumps({"parameters": result.uns["threadfin"]["clonotype_recluster"], "audit": audit}, indent=2) + "\n")
    print(json.dumps(audit), flush=True)
    return audit


def baseline():
    data = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    rows = []
    for basis in ("current", "legacy"):
        for definition in ("donor_sequence", "donor_vdj", "pooled_vdj"):
            for minimum in (1, 3):
                rows.append(run_one(data, basis, definition, minimum))
        rows.append(run_one(data, basis, "pooled_all_heavy", 1))
    pd.DataFrame(rows).to_csv(OUT / "baseline_summary.csv", index=False)


def sweep(definition, minimum, basis):
    data = ad.read_h5ad(PRIVATE / "cell_geometry.h5ad")
    rows = []
    for k in (10, 20, 40, 80):
        for md in (.1, .4, .65, .85):
            for seed in (123, 7):
                rows.append(run_one(data, basis, definition, minimum, k=k, md=md, seed=seed))
    pd.DataFrame(rows).to_csv(OUT / f"sweep_{basis}_{definition}_min{minimum}.csv", index=False)


def panels(summary_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    table = pd.read_csv(summary_path)
    if "sweep" in summary_path.name:
        table = table[table.seed.eq(123)]
    ncol = 4
    nrow = int(np.ceil(len(table) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.3*nrow), squeeze=False)
    bases = {'legacy': 'Notebook RNA', 'current': 'Receptor-excluded RNA', 'allgenes': 'All-gene PCA'}
    units = {'donor_sequence': 'same-mouse sequence', 'donor_vdj': 'same-mouse VDJ',
             'pooled_vdj': 'pooled VDJ (diagnostic)', 'pooled_all_heavy': 'all IGH VDJ (diagnostic)'}
    for ax, (_, row) in zip(axes.ravel(), table.iterrows()):
        cm = pd.read_csv(OUT / "maps" / (row["name"] + ".csv"), index_col=0)
        colours = plt.get_cmap("tab20" if cm.clone_cluster.nunique()>10 else "tab10")
        for label, group in cm.groupby("clone_cluster", observed=True):
            ax.scatter(group.x, group.y, c=[colours(int(label))], s=5*group.n_cells,
                       edgecolors="white", linewidths=.3, alpha=.9)
        ax.set_title(f'{bases[row.basis]} · {units[row.definition]} ≥{row.min_clone_size}\n'
                     f'k={row.umap_k}, min_dist={row.min_dist} | {row.n_points} points / {row.n_clusters} clusters', fontsize=7.4)
        ax.set_aspect("equal", adjustable="box"); ax.margins(.08)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(.02, .01, f'Trust={row.trustworthiness_centroids:.3f}; graph components={row.n_graph_components}', transform=ax.transAxes, fontsize=6)
    for ax in axes.ravel()[len(table):]: ax.set_axis_off()
    fig.suptitle('GSE246382: real-data notebook reconstruction / parameter comparisons', fontsize=12)
    fig.text(.5, .014, 'Pooled VDJ panels are historical diagnostics; they merge donors/junctions. Colours are Leiden clusters, not fate labels.',
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.025, right=.985, bottom=.045, top=.94, hspace=.28, wspace=.13)
    dest = ROOT / "paper/figure_plan/review" / ("GSE246382_" + summary_path.stem)
    fig.savefig(dest.with_suffix(".png"), dpi=160)
    fig.savefig(dest.with_suffix(".pdf")); plt.close(fig)
    print("Saved", dest, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "baseline", "sweep", "single", "panels", "all-genes", "stability", "audit", "verify"))
    p.add_argument("--definition", choices=("donor_sequence", "donor_vdj", "pooled_vdj", "pooled_all_heavy"), default="donor_sequence")
    p.add_argument("--minimum", type=int, default=1)
    p.add_argument("--basis", choices=("legacy", "current", "allgenes"), default="legacy")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-dist", type=float, default=.4)
    p.add_argument("--resolution", type=float, default=.3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "baseline": baseline()
    elif args.stage == "all-genes": all_genes()
    elif args.stage == "stability": stability_runs()
    elif args.stage == "audit": audit_results()
    elif args.stage == "verify": verify_selected()
    elif args.stage == "sweep": sweep(args.definition, args.minimum, args.basis)
    elif args.stage == "single":
        run_one(ad.read_h5ad(PRIVATE / "cell_geometry.h5ad"), args.basis, args.definition,
                args.minimum, k=args.k, md=args.min_dist, resolution=args.resolution, seed=args.seed)
    else: panels(args.summary or OUT / "baseline_summary.csv")
