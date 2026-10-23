#!/usr/bin/env python
"""GSE284622 gut GC reporter: Threadfin on every B-lineage cell.

The first gut analysis drew the clone map over all BCR+ cells, but its figures only
named germinal-centre (GC) and plasma-cell (PC) compartments. This run keeps the
authors' third B-cell compartment -- ``B_cell``: OVA-binding B cells outside the GC
and PC clusters, which contain a recently activated subcluster (Nr4a1/Myc/Cd83/Irf4)
and a resting CD23+ CD62L+ subcluster -- and names it throughout.

Cells: the authors' ``B_cell``, ``GC_Bcell`` and ``plasma`` clusters. The ``other``
clusters (T/NK, myeloid and low-quality cells; 9 of 1,187 carry a BCR) are dropped.
Clones: the saved Dandelion paired heavy+light calls (within mouse, identity 0.85).

Three centring variants of the same clone profiles are run, because donor centring
turned the gut plasma lobe into two donor-specific arms (mouse3 is 52 % plasma):

* ``donor``    -- context_key="donor" (the case-study default; reproduces the old map);
* ``none``     -- no context centring;
* ``balanced`` -- each mouse's shift is estimated *within* B / GC / PC compartments and
                  removed before profiling, so the reference does not depend on how many
                  cells of each compartment a mouse contributed.

Measured labels (tdTomato GC-fate sort, SHM, isotype, tissue) never enter the profiles.

Outputs: ``allB_threadfin/<variant>/`` (clone table, cells, programmes, tests, json) and
``allB_threadfin/cells_allB.csv.gz`` (cell UMAP shared by all variants).

Run:  sbatch allB_threadfin_run.sbatch   (or python allB_threadfin_run.py)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "case_studies"))
os.environ.setdefault("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data")

import threadfin as tf  # noqa: E402
from datasets import load_gut_gc_reporter  # noqa: E402

OUT = HERE / "allB_threadfin"
DATA = Path(os.environ["THREADFIN_DATA"]) / "gse284622_gut_gc"
CLONES = REPO / "case_studies" / "results" / "dandelion_clones" / "gut_gc_reporter.csv"
B_LINEAGE = ["B_cell", "GC_Bcell", "plasma"]
ACTIVATED_CLUSTER = "7"           # author RNA_snn_res.0.5 cluster: Nr4a1+ Myc+ Cd83+ Irf4+
MAP = dict(n_neighbors=30, min_dist=0.3)   # the gut layout used by the current Figure 3
T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def fine_isotype(c_call: pd.Series) -> pd.Series:
    table = {"IGHM": "IgM", "IGHD": "IgD", "IGHG1": "IgG1", "IGHG2B": "IgG2b", "IGHG2C": "IgG2c",
             "IGHG3": "IgG3", "IGHA": "IgA", "IGHE": "IgE"}
    # calls look like "IGHA*06,IGHA*08": first call, allele dropped
    gene = c_call.astype(str).str.split(",").str[0].str.split("*").str[0].str.upper()
    return gene.map(table)


def load():
    a, bcr = load_gut_gc_reporter()
    meta = pd.read_csv(DATA / "gex_reporter" / "metadata.csv.gz", low_memory=False).set_index("cell")
    a.obs["author_cluster"] = meta.loc[a.obs_names, "RNA_snn_res.0.5"].astype(int).astype(str).values
    ids = pd.read_csv(CLONES).set_index("cell")["clone_id"]
    bcr["clone_id"] = ids.reindex(bcr.index).values
    bcr = bcr[bcr["clone_id"].notna()].copy()
    bcr["isotype"] = tf.clones.isotype_class(bcr["c_call"]).values
    bcr["isotype_fine"] = fine_isotype(bcr["c_call"]).values
    tf.attach_bcr(a, bcr.drop(columns=["author_clone_id"]), clone_col="clone_id", summarize=False,
                  verbose=False)
    for col in ("mutation_frequency", "isotype", "isotype_fine"):
        a.obs[col] = a.obs[f"bcr_{col}"]
    n_other_bcr = int((a.obs["celltype"].eq("other") & a.obs["clone_id"].notna()).sum())
    a = a[a.obs["celltype"].isin(B_LINEAGE).to_numpy()].copy()
    ct = a.obs["celltype"].astype(str)
    a.obs["compartment"] = ct.map({"B_cell": "B", "GC_Bcell": "GC", "plasma": "PC"})
    act = ct.eq("B_cell") & a.obs["author_cluster"].eq(ACTIVATED_CLUSTER)
    a.obs["b_state"] = np.where(ct.ne("B_cell"), "not B", np.where(act, "activated", "resting"))
    a.obs["fate"] = a.obs["gc_sort"].map({"GC": "tdTom+", "nonGC": "tdTom-"})
    a.obs["comp_fate"] = a.obs["compartment"] + " " + a.obs["fate"]
    stamp(f"B-lineage cells: {a.n_obs:,}; BCR+: {int(a.obs['clone_id'].notna().sum()):,}; "
          f"'other' cells dropped carried {n_other_bcr} BCRs")
    return a, n_other_bcr


def balanced_basis(a, basis="X_threadfin", key="compartment", out="X_balanced"):
    """Remove each mouse's shift estimated within compartments (composition-invariant)."""
    x = np.asarray(a.obsm[basis], dtype=float)
    donor = a.obs["donor"].astype(str).to_numpy()
    comp = a.obs[key].astype(str).to_numpy()
    levels = sorted(set(comp))
    weight = pd.Series(comp).value_counts(normalize=True).reindex(levels).to_numpy()
    glob = np.vstack([x[comp == k].mean(axis=0) for k in levels])
    shifts = {}
    y = x.copy()
    for d in sorted(set(donor)):
        dev = []
        for k in levels:
            sel = (donor == d) & (comp == k)
            dev.append(x[sel].mean(axis=0) - glob[levels.index(k)] if sel.sum() >= 20 else np.zeros(x.shape[1]))
        shift = np.average(np.vstack(dev), axis=0, weights=weight)
        y[donor == d] -= shift
        shifts[d] = float(np.linalg.norm(shift))
    a.obsm[out] = y
    return shifts


def clone_descriptors(a) -> pd.DataFrame:
    """Per-clone measured composition. None of these columns enters a profile."""
    obs = a.obs[a.obs["clone_id"].notna()].copy()
    obs["clone_id"] = obs["clone_id"].astype(str)
    g = obs.groupby("clone_id", observed=True)
    d = pd.DataFrame({"n_cells": g.size()})
    for k in ("B", "GC", "PC"):
        d[f"n_{k}"] = g["compartment"].apply(lambda s, k=k: int((s == k).sum()))
    d["n_B_activated"] = g["b_state"].apply(lambda s: int((s == "activated").sum()))
    d["n_B_tdpos"] = (obs["compartment"].eq("B") & obs["fate"].eq("tdTom+")).groupby(obs["clone_id"]).sum()
    d["n_B_tdneg"] = (obs["compartment"].eq("B") & obs["fate"].eq("tdTom-")).groupby(obs["clone_id"]).sum()
    d["n_PC_tdneg"] = (obs["compartment"].eq("PC") & obs["fate"].eq("tdTom-")).groupby(obs["clone_id"]).sum()
    for k in ("B", "GC", "PC"):
        d[f"frac_{k}"] = d[f"n_{k}"] / d["n_cells"]
    d["frac_tdpos"] = obs["fate"].eq("tdTom+").groupby(obs["clone_id"]).mean()
    d["shm_mean"] = g["mutation_frequency"].mean()
    for k in ("B", "GC", "PC"):
        sub = obs[obs["compartment"].eq(k)]
        d[f"shm_{k}"] = sub.groupby("clone_id")["mutation_frequency"].mean()
    iso = obs.dropna(subset=["isotype_fine"])
    iso_n = iso.groupby("clone_id").size()
    d["n_isotyped"] = iso_n.reindex(d.index).fillna(0).astype(int)
    groups = {"IgM_D": ["IgM", "IgD"], "IgG1": ["IgG1"], "IgG_other": ["IgG2b", "IgG2c", "IgG3"], "IgA": ["IgA"]}
    for name, members in groups.items():
        hit = iso["isotype_fine"].isin(members).groupby(iso["clone_id"]).sum()
        d[f"frac_{name}"] = (hit / iso_n).reindex(d.index)
    for t in ("PP", "MLN", "LP"):
        d[f"frac_{t}"] = obs["tissue"].eq(t).groupby(obs["clone_id"]).mean()
    d["n_tissues"] = g["tissue"].nunique()
    d["donor"] = g["donor"].agg(lambda s: s.iloc[0])
    return d


def knn_donor_mixing(feats: np.ndarray, donor: np.ndarray, strata: np.ndarray, k=15, n_perm=200, seed=0):
    """Share of each clone's k nearest clones from its own mouse, vs donor labels shuffled
    within strata (dominant compartment), which keeps compartment-donor structure."""
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, len(feats) - 1))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(feats).kneighbors(feats, return_distance=False)[:, 1:]
    obs = float((donor[idx] == donor[:, None]).mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = donor.copy()
        for s in np.unique(strata):
            sel = np.flatnonzero(strata == s)
            perm[sel] = rng.permutation(perm[sel])
        null.append(float((perm[idx] == perm[:, None]).mean()))
    null = np.asarray(null)
    return {"observed": obs, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "excess": obs - float(null.mean()), "k": k, "n_perm": n_perm,
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def run_variant(a, name, basis, context_key, desc):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    res = {"variant": name, "basis": basis, "context_key": context_key}

    # Q1: is cell state inherited within clones? (clone means, unsmoothed cells)
    tf.tl.clone_profiles(a, basis=basis, context_key=context_key, donor_key="donor",
                         representation="mean", min_cells=2, verbose=False)
    coh = tf.tl.clonal_coherence(a, strata_key="sample", n_perm=500, verbose=True)
    res["coherence"] = {k: coh[k] for k in ("icc", "null_mean", "null_q95", "excess", "p_value",
                                            "n_clones", "n_cells", "strata_key")}

    # Q2: clone map and programmes (kernel profiles; every clone with >= 2 cells)
    table = tf.tl.clone_profiles(a, basis=basis, context_key=context_key, donor_key="donor",
                                 representation="kernel", min_cells=2, verbose=True)
    res["profiles"] = {"n_clones": int(table.shape[0]), "n_cells": int(table["n_cells"].sum()),
                       "bandwidth": float(a.uns["threadfin"]["profiles"]["params"]["bandwidth"])}
    tf.tl.embed_clones(a, **MAP, min_reliability=0.0, verbose=True)
    try:
        prog = tf.tl.find_programmes(a, n_boot=50, verbose=True)
        prog.to_csv(out / "programmes.csv")
        res["programmes"] = {"n": int(prog.shape[0]), "stability": prog["stability"].round(3).to_dict(),
                             "n_clones": prog["n_clones"].astype(int).to_dict()}
        splits = a.uns["threadfin"]["programmes"]["split_tests"]
        if not splits.empty:
            splits.to_csv(out / "programme_split_tests.csv", index=False)
        tf.tl.programme_composition(a, "comp_fate").to_csv(out / "programme_composition.csv")
    except ValueError as e:
        res["programmes"] = {"n": 0, "note": str(e)}

    prof = a.uns["threadfin"]["profiles"]
    clones = prof["clone_table"].copy()
    feats = prof["features"].copy()
    keyed = a.obs.dropna(subset=["clone_id"])
    for col in ("clone_programme", "clone_programme_assigned"):
        if col in a.obs:
            clones[col] = keyed.groupby(keyed["clone_id"].astype(str), observed=True)[col].first() \
                .reindex(clones.index)
    clones = clones.join(desc.drop(columns=["n_cells", "donor"]), how="left")
    clones.to_csv(out / "clone_table.csv")
    feats.to_csv(out / "clone_features.csv.gz")

    # Q3: do measured labels explain how clones differ? (permutations within mouse)
    tests = [("gc_sort", "fraction:GC"), ("celltype", "fraction:B_cell"), ("celltype", "fraction:GC_Bcell"),
             ("celltype", "fraction:plasma"), ("isotype", "majority"), ("mutation_frequency", "mean"),
             ("tissue", "fraction:LP"), ("tissue", "fraction:PP"), ("tissue", "fraction:MLN")]
    effects = {}
    for label, how in tests:
        try:
            r = tf.tl.profile_association(a, label, how=how, n_perm=2000, verbose=False)
        except ValueError as e:
            stamp(f"  {label}/{how} skipped: {e}")
            continue
        key = label if not how.startswith("fraction:") else f"{label}:{how.split(':', 1)[1]}"
        effects[key] = {k: r[k] for k in ("r2", "null_mean", "excess_r2", "p_value", "n_clones", "kind")}
    res["label_effects"] = effects
    pd.DataFrame(effects).T.to_csv(out / "label_effects.csv")

    # Donor diagnostics: is the map organised by mouse beyond what compartments imply?
    dom = clones[["frac_B", "frac_GC", "frac_PC"]].idxmax(axis=1).to_numpy()
    res["donor_mixing_profiles"] = knn_donor_mixing(feats.loc[clones.index].to_numpy(),
                                                    clones["donor"].astype(str).to_numpy(), dom)
    pcl = clones["frac_PC"] >= 0.5
    if pcl.sum() > 30:
        res["donor_mixing_plasma_clones"] = knn_donor_mixing(
            feats.loc[clones.index[pcl]].to_numpy(), clones.loc[pcl, "donor"].astype(str).to_numpy(),
            np.zeros(int(pcl.sum()), dtype=int))
    if "clone_programme" in clones:
        tab = pd.crosstab(clones["clone_programme"], clones["donor"])
        tab.to_csv(out / "programme_by_donor.csv")
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(tab)
        n = tab.to_numpy().sum()
        res["programme_by_donor"] = {"table": tab.to_dict(), "chi2_p": float(p),
                                     "cramers_v": float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))}

    cells = a.obs[["clone_id", "donor", "sample", "tissue", "gc_sort", "celltype", "author_cluster",
                   "compartment", "b_state", "fate", "isotype_fine", "mutation_frequency"]].copy()
    for col in ("clone_programme", "clone_programme_assigned"):
        if col in a.obs:
            cells[col] = a.obs[col]
    cells.to_csv(out / "cells.csv.gz")
    (out / "summary.json").write_text(json.dumps(tf_jsonable(res), indent=1))
    stamp(f"{name}: {json.dumps(tf_jsonable({k: res[k] for k in ('coherence', 'programmes')}))}")
    for col in ("clone_programme", "clone_programme_assigned"):
        if col in a.obs:
            del a.obs[col]
    return res


def tf_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): tf_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tf_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, n_other_bcr = load()
    tf.pp.prepare_embedding(a, batch_key=None, key_added="X_threadfin", verbose=True)
    import scanpy as sc

    sc.pp.neighbors(a, use_rep="X_threadfin", n_neighbors=15, random_state=0, key_added="cells")
    sc.tl.umap(a, neighbors_key="cells", random_state=0)
    a.obs["umap_1"], a.obs["umap_2"] = a.obsm["X_umap"][:, 0], a.obsm["X_umap"][:, 1]
    stamp("embedding + cell UMAP done")
    shifts = balanced_basis(a)
    stamp(f"balanced donor shifts (norm in PC units): {shifts}")

    cells = a.obs[["clone_id", "donor", "sample", "tissue", "gc_sort", "celltype", "author_cluster",
                   "compartment", "b_state", "fate", "comp_fate", "isotype_fine", "mutation_frequency",
                   "umap_1", "umap_2"]].copy()
    cells.to_csv(OUT / "cells_allB.csv.gz")
    desc = clone_descriptors(a)
    desc.to_csv(OUT / "clone_descriptors_all_clones.csv")

    summary = {"dataset": "GSE284622 gut GC reporter (Siniscalco et al. 2025 Immunity)",
               "cells": {"n_b_lineage": int(a.n_obs), "n_bcr": int(a.obs["clone_id"].notna().sum()),
                         "by_compartment": a.obs["compartment"].value_counts().to_dict(),
                         "by_comp_fate": a.obs["comp_fate"].value_counts().to_dict(),
                         "other_cells_dropped_with_bcr": n_other_bcr},
               "map_params": MAP, "balanced_shift_norms": shifts, "variants": {}}
    for name, basis, ctx in (("donor", "X_threadfin", "donor"), ("none", "X_threadfin", None),
                             ("balanced", "X_balanced", None)):
        stamp(f"variant {name}")
        summary["variants"][name] = run_variant(a, name, basis, ctx, desc)
    (OUT / "summary.json").write_text(json.dumps(tf_jsonable(summary), indent=1))
    # slim copy for figure panels: raw counts, the two embeddings and plain-text cell labels
    import anndata as ad

    slim = ad.AnnData(X=a.X.copy(), obs=cells.astype(str).copy(), var=a.var[[]].copy())
    slim.obs["mutation_frequency"] = pd.to_numeric(a.obs["mutation_frequency"], errors="coerce").values
    for key in ("X_threadfin", "X_balanced", "X_umap"):
        slim.obsm[key] = np.asarray(a.obsm[key])
    slim.write_h5ad(OUT / "allB_embedding.h5ad", compression="gzip")
    stamp("done")


if __name__ == "__main__":
    main()
