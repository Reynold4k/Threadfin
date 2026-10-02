#!/usr/bin/env python
"""Threadfin v4 analysis of one public dataset, plus the v3-vs-v4 audit.

Usage:
    python run_dataset.py <dataset>      # ln_vaccine | flu | tonsil | ebv | stephenson | mouse_np | mouse_rbd

Writes results/<dataset>/: summary.json, tables (*.csv), figures/, and a short
machine-generated report.md (the interpretation for readers lives in
../../report/PUBLIC_DATASETS_REPORT.md).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import threadfin as tf  # noqa: E402
from datasets import LOADERS  # noqa: E402

HERE = Path(__file__).resolve().parent

# Clone-level labels tested against programmes: (obs column, aggregation, note)
CONFIGS = {
    "ln_vaccine": dict(
        title="Kim et al. 2022 Nature - SARS-CoV-2 mRNA vaccine, lymph node FNA + blood (8 donors)",
        batch_key="donor", state_key="state",
        tests=[("spike_binding", "majority", "spike binding of the clone (probe sorting / mAb ELISA)"),
               ("elisa", "majority", "ELISA of the clone's expressed mAb"),
               ("isotype", "majority", "heavy-chain isotype"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("tissue", "fraction:blood", "fraction of the clone's cells sampled in blood")],
        memory=["timepoint", "tissue"],
    ),
    "flu": dict(
        title="Wang et al. 2023 - seasonal influenza vaccine, PBMC B cells (6 donors, d0 + d7)",
        batch_key="donor", state_key=None,
        tests=[("isotype", "majority", "heavy-chain isotype"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("timepoint", "fraction:d7", "fraction of the clone's cells sampled at day 7")],
        donor_level_tests=[("age_group", "majority", "donor age group (donor-level; confounded with donor)")],
        memory=["timepoint"],
    ),
    "tonsil": dict(
        title="King et al. 2021 Sci Immunol - paediatric tonsil (6 donors)",
        batch_key="donor", state_key="state",
        tests=[("isotype", "majority", "heavy-chain isotype")],
        memory=[],
    ),
    "ebv": dict(
        title="Mitul et al. 2026 PNAS - EBV infection of tonsil organoids (d0-d21, GFP+/- sorts)",
        batch_key=None, state_key=None, context_key="timepoint",
        tests=[("gfp", "fraction:GFP+", "fraction of the clone's d14/d21 cells that are EBV-GFP+"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=["timepoint", "gfp"],
    ),
    "stephenson": dict(
        title="Stephenson et al. 2021 Nat Med - COVID-19 PBMC, 5k BCR+ B-lineage cells",
        batch_key="donor", state_key="state",
        tests=[("isotype", "majority", "heavy-chain isotype"),
               ("mutation_frequency", "mean", "V-region mutation frequency")],
        donor_level_tests=[("severity", "majority", "clinical severity (donor-level)")],
        memory=[],
    ),
    "mouse_np": dict(
        title="Merkenschlager et al. 2025 Nature - NP-OVA germinal centres, H2B-mCherry division reporter (7 mice)",
        batch_key=None, state_key=None,
        tests=[("division_gate", "fraction:mCherry-low", "fraction of the clone's cells that divided most (mCherry-low gate)"),
               ("w33l", "majority", "VH186.2 W33L high-affinity mutation (IGHV1-72 clones)"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=["division_gate"],
    ),
    "mouse_rbd": dict(
        title="Merkenschlager et al. 2025 Nature - RBD protein / mRNA vaccine germinal centres (10 mice)",
        batch_key=None, state_key=None,
        tests=[("rbd_bait", "fraction:RBD+", "fraction of the clone's cells binding RBD bait (protein arm)"),
               ("division_gate", "fraction:mCherry-low", "fraction of the clone's cells in the mCherry-low gate"),
               ("zone_gate", "fraction:DZ", "fraction of the clone's cells sorted as dark zone (mRNA arm)"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=["division_gate", "rbd_bait", "zone_gate"],
    ),
}

GENE_SETS_HUMAN = {
    "plasma cell": ["PRDM1", "XBP1", "IRF4", "JCHAIN", "MZB1", "SDC1", "TNFRSF17", "DERL3", "FKBP11", "SSR4"],
    "germinal centre": ["BCL6", "AICDA", "RGS13", "MEF2B", "S1PR2", "LMO2", "MYBL1", "GCSAM", "ELL3", "SERPINA9"],
    "dark zone / cycling": ["CXCR4", "MKI67", "TOP2A", "AURKB", "CCNB1", "HMGB2", "TUBB", "STMN1", "FOXP1", "AICDA"],
    "light zone": ["CD83", "CD86", "MYC", "NFKBIA", "BATF", "EGR2", "IL4I1", "CCL22", "BCL2A1", "NME1"],
    "memory": ["CD27", "CCR6", "GPR183", "TNFRSF13B", "SELL", "CD44", "S1PR1", "KLF2", "ZBTB32", "FCRL4"],
    "naive": ["TCL1A", "FCER2", "IL4R", "BACH2", "SELL", "PLPP5", "YBX3", "BTG1", "CD200", "IGHD"],
    "interferon": ["ISG15", "IFIT1", "IFIT3", "MX1", "IFI44L", "OAS1", "IFI6", "XAF1", "STAT1", "IRF7"],
}


def _mouse_case(genes):
    return [g[:1] + g[1:].lower() for g in genes]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 200 else f"<array {obj.shape}>"
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def leiden_states(adata, basis="X_threadfin", resolution=1.0):
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=basis, n_neighbors=15, random_state=0, key_added="tf_cells")
    sc.tl.leiden(adata, resolution=resolution, neighbors_key="tf_cells", key_added="leiden_state",
                 random_state=0, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(adata, neighbors_key="tf_cells", random_state=0)
    return "leiden_state"


def main(name: str):
    cfg = CONFIGS[name]
    out = HERE / "results" / name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {"dataset": name, "title": cfg["title"]}

    def stamp(msg):
        print(f"[{name} {time.time() - t0:7.0f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ 1. data
    adata, bcr = LOADERS[name]()
    stamp(f"loaded {adata.n_obs} cells x {adata.n_vars} genes; {len(bcr)} cells with a heavy chain")
    mouse = name.startswith("mouse")
    bcr["isotype"] = tf.clones.isotype_class(bcr["c_call"]).values
    bcr["donor"] = adata.obs["donor"].astype(str).reindex(bcr.index).values

    # ------------------------------------------------------------------ 2. clones (within donors)
    bcr = tf.define_clones(bcr, donor_key="donor", out_col="clone_id")
    cdef = dict(bcr.attrs["clone_definition"])
    if "author_clone_id" in bcr.columns:
        from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

        both = bcr.dropna(subset=["clone_id", "author_clone_id"])
        h, c, v = homogeneity_completeness_v_measure(both["author_clone_id"], both["clone_id"])
        cdef["vs_author_clones"] = {"ari": adjusted_rand_score(both["author_clone_id"], both["clone_id"]),
                                    "homogeneity": h, "completeness": c, "n_cells": int(len(both))}
    summary["clone_definition"] = cdef
    tf.attach_bcr(adata, bcr.drop(columns=["donor"]), clone_col="clone_id", summarize=False)
    for col in [c for c in bcr.columns if c not in ("clone_id", "donor", "v_call", "j_call", "junction",
                                                     "cdr3_nt", "c_call", "author_clone_id")]:
        adata.obs[col] = adata.obs[f"bcr_{col}"]
    sizes = adata.obs["clone_id"].value_counts()
    summary["qc"] = {
        "n_cells": int(adata.n_obs), "n_cells_bcr": int(adata.obs["clone_id"].notna().sum()),
        "n_clones": int(sizes.size), "clones_ge2": int((sizes >= 2).sum()), "clones_ge3": int((sizes >= 3).sum()),
        "clones_ge5": int((sizes >= 5).sum()), "clones_ge10": int((sizes >= 10).sum()),
        "cells_in_clones_ge2": int(sizes[sizes >= 2].sum()), "largest_clone": int(sizes.max()),
        "n_donors": int(adata.obs["donor"].nunique()), "n_samples": int(adata.obs["sample"].nunique()),
    }
    stamp(f"clones: {summary['qc']}")

    # ------------------------------------------------------------------ 3. embeddings (with / without IG genes)
    tf.pp.prepare_embedding(adata, batch_key=cfg["batch_key"], key_added="X_threadfin", verbose=True)
    hvg_clean = adata.var["highly_variable"].copy()
    tf.pp.prepare_embedding(adata, batch_key=cfg["batch_key"], key_added="X_withIG",
                            exclude_receptor_genes=False, verbose=True)
    n_ig_hvg = int((adata.var["highly_variable"] & adata.var["threadfin_receptor_gene"]).sum())
    adata.var["highly_variable"] = hvg_clean
    summary["ig_genes_in_naive_hvg"] = n_ig_hvg
    state_key = cfg["state_key"] or leiden_states(adata)
    if cfg["state_key"]:
        leiden_states(adata)  # still compute a cell UMAP for figures
    stamp(f"embeddings done; {n_ig_hvg} IG genes would have entered a naive HVG set")

    context_key = cfg.get("context_key", "donor")
    strata_key = "sample"

    # ------------------------------------------------------------------ 4. clonal coherence (+ audit)
    coh = {}
    for label, basis in (("without IG genes", "X_threadfin"), ("with IG genes (v3 practice)", "X_withIG")):
        tf.tl.clone_profiles(adata, basis=basis, context_key=context_key, donor_key="donor",
                             representation="mean", verbose=False)
        r = tf.tl.clonal_coherence(adata, strata_key=strata_key, n_perm=200, verbose=False)
        coh[label] = {k: r[k] for k in ("icc", "null_mean", "null_q95", "p_value", "n_clones", "n_cells")}
    r = tf.tl.clonal_coherence(adata, strata_key="donor", n_perm=200, verbose=False)
    coh["with IG genes, donor-only null (v3 practice)"] = {k: r[k] for k in ("icc", "null_mean", "null_q95", "p_value")}
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=context_key, donor_key="donor",
                         representation="mean", verbose=False)
    main_coh = tf.tl.clonal_coherence(adata, strata_key=strata_key, n_perm=500, verbose=True)
    coh["main"] = {k: main_coh[k] for k in ("icc", "null_mean", "null_q95", "excess", "p_value", "n_clones", "n_cells")}
    vc = adata.uns["threadfin"]["profiles"]["variance_components"]
    coh["main"]["min_cells_reliability_0.5"] = tf.profiles.VarianceComponents(
        np.asarray(vc["sigma2"]), np.asarray(vc["tau2"]), vc["n0"], vc["n_clones"], vc["n_cells"],
        np.zeros(len(vc["tau2"]))).min_cells_for(0.5)
    summary["coherence"] = coh
    stamp(f"coherence: {json.dumps(coh['main'])}")
    tf.pl.set_style()
    tf.pl.coherence(adata, save=out / "figures" / "coherence.png")
    plt.close("all")

    # ------------------------------------------------------------------ 5. programmes (kernel profiles)
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=context_key, donor_key="donor",
                         representation="kernel", verbose=True)
    try:
        prog = tf.tl.find_programmes(adata, n_boot=50, verbose=True)
    except ValueError as e:
        prog = None
        summary["programmes"] = {"n": 0, "error": str(e)}
        stamp(f"programmes not defined: {e}")
    if prog is not None:
        progs = adata.uns["threadfin"]["programmes"]
        prog.to_csv(out / "programmes.csv")
        progs["resolution_scan"].to_csv(out / "resolution_scan.csv", index=False)
        comp = tf.tl.programme_composition(adata, state_key)
        comp.to_csv(out / "programme_composition.csv")
        markers = tf.tl.programme_markers(adata, programme_key="clone_programme", layer="log_norm",
                                          context_key=context_key, n_top=50, verbose=True)
        markers.to_csv(out / "programme_markers.csv", index=False)
        signatures = _programme_signatures(adata, GENE_SETS_HUMAN if not mouse else
                                           {k: _mouse_case(v) for k, v in GENE_SETS_HUMAN.items()})
        signatures.to_csv(out / "programme_signatures.csv")
        summary["programme_markers"] = {p_: g["gene"].head(12).tolist() for p_, g in markers.groupby("programme")}
        summary["programme_signatures"] = signatures.round(3).to_dict(orient="index")
        summary["programmes"] = {"n": int(prog.shape[0]), "resolution": progs["params"]["resolution"],
                                 "n_clones": int(prog["n_clones"].sum()), "n_cells": int(prog["n_cells"].sum()),
                                 "n_clones_with_assigned": int(prog["n_clones_with_assigned"].sum()),
                                 "bandwidth": adata.uns["threadfin"]["profiles"]["params"].get("bandwidth"),
                                 "stability": prog["stability"].round(3).to_dict(),
                                 "n_stable": int((prog["stability"] >= 0.75).sum())}
        stamp(f"programmes: {summary['programmes']}")

    # ------------------------------------------------------------------ 6. clone-level tests
    assoc = []
    assoc_assigned = []
    v3_vs_v4 = []
    for label, how, note in (cfg["tests"] if prog is not None else []):
        if label not in adata.obs or adata.obs[label].notna().sum() < 20:
            continue
        try:
            res = tf.tl.association_test(adata, label, how=how, n_perm=2000, verbose=True)
        except ValueError as e:
            stamp(f"  skip {label}: {e}")
            continue
        res["note"] = note
        res["how"] = how
        res["strata"] = res.attrs["strata_key"]
        res["programmes"] = "core"
        assoc.append(res)
        v3_vs_v4.append(_v3_cell_level(adata, label, how, res))
        # the same test on core + confidently assigned clones (more clones, less certain labels)
        try:
            res2 = tf.tl.association_test(adata, label, how=how, programme_key="clone_programme_assigned",
                                          n_perm=2000, verbose=True)
            res2["note"], res2["how"], res2["strata"] = note, how, res2.attrs["strata_key"]
            res2["programmes"] = "core + assigned"
            assoc_assigned.append(res2)
        except (ValueError, KeyError) as e:
            stamp(f"  assigned-programme test {label} skipped: {e}")
    for label, how, note in (cfg.get("donor_level_tests", []) if prog is not None else []):
        res = tf.tl.association_test(adata, label, how=how, strata_key=None, n_perm=2000, verbose=True)
        res["note"] = note + " - tested WITHOUT donor stratification; donor-level labels cannot be separated from donor effects"
        res["how"], res["strata"] = how, "none"
        assoc.append(res)
    if assoc_assigned:
        pd.concat(assoc_assigned, ignore_index=True).to_csv(out / "associations_assigned.csv", index=False)
    if assoc:
        pd.concat(assoc, ignore_index=True).to_csv(out / "associations.csv", index=False)
        pd.concat(v3_vs_v4, ignore_index=True).to_csv(out / "v3_vs_v4_tests.csv", index=False)
    summary["associations"] = {
        a["label"].iloc[0]: {"n_clones": int(a.attrs.get("n_clones", 0)),
                             "n_significant_fdr05": int((a["fdr"] < 0.05).sum()),
                             "kruskal_p": a.attrs.get("kruskal_pvalue")} for a in assoc}

    # ------------------------------------------------------------------ 7. clonal memory
    mem = {}
    for key in cfg["memory"]:
        if key not in adata.obs or adata.obs[key].nunique() < 2:
            continue
        try:
            r = tf.tl.clonal_memory(adata, key, verbose=True)
        except ValueError as e:
            stamp(f"  memory {key} skipped: {e}")
            continue
        mem[key] = {k: r.get(k) for k in ("n_clones", "n_pairs", "memory_index", "memory_index_ci", "p_value",
                                          "persistence", "persistence_null", "levels")}
        r["pairs"].to_csv(out / f"memory_pairs_{key}.csv", index=False)
        if "transitions" in r:
            r["transitions"].to_csv(out / f"memory_transitions_{key}.csv")
            r["transitions_expected"].to_csv(out / f"memory_transitions_expected_{key}.csv")
        # the v3 "transition matrix" on clone-level labels, for the audit
        if prog is not None:
            tr = tf.clones.community_transition(adata, time_key=key, cluster_key="clone_programme")
            mem[key]["v3_transition_diagonal"] = float(np.trace(tr.to_numpy()) / max(tr.to_numpy().sum(), 1))
    summary["memory"] = mem

    # ------------------------------------------------------------------ 8. gene-level clonal heritability
    gene_sets = GENE_SETS_HUMAN if not mouse else {k: _mouse_case(v) for k, v in GENE_SETS_HUMAN.items()}
    try:
        herit = tf.tl.gene_heritability(adata, context_key=context_key, strata_key=strata_key,
                                        layer="log_norm", n_perm=100, verbose=True)
        herit.to_csv(out / "gene_heritability.csv")
        gs = tf.tl.geneset_heritability(herit, gene_sets)
        gs.to_csv(out / "geneset_heritability.csv", index=False)
        summary["heritability"] = {"n_genes": int(herit.shape[0]), "n_fdr05": int((herit["fdr"] < 0.05).sum()),
                                   "median_icc": float(herit["icc"].median()),
                                   "top_genes": herit.head(25).index.tolist(),
                                   "gene_sets": gs.set_index("gene_set")[["median_icc", "frac_significant"]].round(4)
                                   .to_dict(orient="index") if not gs.empty else {}}
    except Exception as e:  # keep the run going; report the failure
        summary["heritability"] = {"error": repr(e)}

    # ------------------------------------------------------------------ 9. sensitivity: centre on samples
    try:
        tf.tl.clone_profiles(adata, basis="X_threadfin", context_key="sample", donor_key="donor",
                             representation="mean", verbose=False)
        r = tf.tl.clonal_coherence(adata, strata_key=strata_key, n_perm=200, verbose=False)
        summary["sensitivity_context_sample"] = {k: r[k] for k in ("icc", "null_mean", "p_value")}
    except Exception as e:
        summary["sensitivity_context_sample"] = {"error": repr(e)}
    # restore the main (kernel, donor-context) profiles for the figures and tables
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=context_key, donor_key="donor",
                         representation="kernel", verbose=False)
    if prog is not None:
        tf.tl.find_programmes(adata, n_boot=50, verbose=False)

    # ------------------------------------------------------------------ 10. tables + figures
    table = adata.uns["threadfin"]["profiles"]["clone_table"].copy()
    for label, how, _ in cfg["tests"] + cfg.get("donor_level_tests", []):
        if label in adata.obs:
            table[label] = tf.tl.clone_labels(adata, label, how=how).reindex(table.index)
    table.to_csv(out / "clone_table.csv")
    if prog is not None:
        _figures(adata, state_key, out, cfg)
    cols = [c for c in ("clone_id", "clone_programme", "clone_programme_assigned", state_key, "donor", "sample",
                        "timepoint", "tissue", "isotype", "mutation_frequency", "spike_binding", "gfp",
                        "division_gate", "rbd_bait", "zone_gate", "w33l", "severity") if c in adata.obs]
    cells = adata.obs[list(dict.fromkeys(cols))].copy()
    if "X_umap" in adata.obsm:
        cells["umap_1"], cells["umap_2"] = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
    cells.to_csv(out / "cells.csv.gz")

    summary["runtime_min"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, default=str))
    stamp("done")


def _programme_signatures(adata, gene_sets, programme_key="clone_programme"):
    """Mean gene-set score per programme, averaging cells within clones first."""
    import scanpy as sc

    tmp = sc.AnnData(X=adata.layers["log_norm"], obs=adata.obs[[programme_key, "clone_id"]].copy(),
                     var=pd.DataFrame(index=adata.var_names))
    rows = {}
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in tmp.var_names]
        if len(present) < 3:
            continue
        sc.tl.score_genes(tmp, present, score_name="_s", random_state=0)
        per_clone = tmp.obs.dropna(subset=[programme_key, "clone_id"]).groupby("clone_id", observed=True).agg(
            programme=(programme_key, "first"), score=("_s", "mean"))
        rows[name] = per_clone.groupby("programme", observed=True)["score"].mean()
    return pd.DataFrame(rows)


def _v3_cell_level(adata, label, how, v4_table):
    """Re-test the same label the v3 way: one-sided Fisher on cells (pseudoreplicated)."""
    from scipy.stats import fisher_exact

    rows = []
    obs = adata.obs[["clone_programme", label]].dropna()
    if pd.api.types.is_numeric_dtype(obs[label]) and not how.startswith("fraction:"):
        return pd.DataFrame()
    level = how.split(":", 1)[1] if how.startswith("fraction:") else None
    for prog in obs["clone_programme"].astype(str).unique():
        in_p = (obs["clone_programme"].astype(str) == prog).to_numpy()
        levels = [level] if level else obs[label].astype(str).unique()
        for lv in levels:
            has = (obs[label].astype(str) == lv).to_numpy()
            a, b = int((in_p & has).sum()), int((in_p & ~has).sum())
            c, d = int((~in_p & has).sum()), int((~in_p & ~has).sum())
            p = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
            rows.append({"label": label, "programme": prog, "level": lv, "v3_cell_pvalue": p, "v3_n_cells": a})
    v3 = pd.DataFrame(rows)
    if "level" in v4_table.columns:
        merged = v3.merge(v4_table[["programme", "level", "pvalue", "odds_ratio"]].astype({"level": str}),
                          on=["programme", "level"], how="left")
    else:
        merged = v3.merge(v4_table[["programme", "pvalue", "rank_effect"]], on="programme", how="left")
    return merged.rename(columns={"pvalue": "v4_clone_pvalue"})


def _figures(adata, state_key, out, cfg):
    figs = out / "figures"
    tf.pl.set_style()
    tf.pl.overview(adata, state_key=state_key, save=figs / "overview.png")
    tf.pl.clone_map(adata, save=figs / "clone_map_programmes.png")
    tf.pl.stability(adata, save=figs / "stability.png")
    tf.pl.programme_composition(adata, state_key, save=figs / "programme_composition.png")
    assoc = adata.uns["threadfin"].get("associations", {})
    for label, tab in assoc.items():
        try:
            tf.pl.association(tab, save=figs / f"association_{label}.png")
        except Exception:
            pass
    for key, res in adata.uns["threadfin"].get("memory", {}).items():
        tf.pl.memory(res, save=figs / f"memory_{key}.png")
    herit = adata.uns["threadfin"].get("gene_heritability")
    if herit is not None:
        sets = GENE_SETS_HUMAN if not str(out.name).startswith("mouse") else \
            {k: _mouse_case(v) for k, v in GENE_SETS_HUMAN.items()}
        tf.pl.heritability(herit, highlight={k: sets[k] for k in ("plasma cell", "germinal centre", "dark zone / cycling")},
                           save=figs / "gene_heritability.png")
    plt.close("all")


if __name__ == "__main__":
    main(sys.argv[1])
