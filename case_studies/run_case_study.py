#!/usr/bin/env python
"""Threadfin case study on one public dataset.

Usage:
    python run_case_study.py <dataset>
    # ln_vaccine | flu | tonsil | ebv | stephenson | mouse_np | mouse_rbd

For one dataset, this script asks the questions Threadfin is built for:

1. Does clone identity shape B-cell state?            (clonal coherence)
2. Which clones behave alike?                          (clonal programmes, or a continuum)
3. Do the measured labels - antigen binding, isotype,  (clone-level label tests)
   division history, sort gate - explain how clones differ?
4. Do clones keep their state across time points,      (clonal memory)
   tissues or sort gates?
5. Which genes are clonally inherited?                 (gene-level clonal heritability)

Results go to results/<dataset>/ (summary.json, tables, figures/). The
step-by-step biological interpretation is in ../report/PUBLIC_DATASETS_REPORT.md.
"""

from __future__ import annotations

import json
import os
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

# What each dataset asks. "tests": cell-level labels collapsed per clone and tested against clone
# states; "donor_level_tests": labels that are constant within a donor (tested without donor
# stratification, so they cannot be separated from other donor differences); "memory": columns
# across which the same clone is compared with itself.
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
        donor_level_tests=[("age_group", "majority", "donor age group")],
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
        donor_level_tests=[("severity", "majority", "clinical severity of the donor")],
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
    "gc_np_pc": dict(
        title="ElTanbouly et al. 2023 J Exp Med - NP-OVA day 14, sorted GC B cells (DZ, LZ, Myc+ LZ) and plasma cells, "
              "Smart-seq2 + TRUST4 (11 mice)",
        batch_key=None, state_key="fate",
        tests=[("fate", "fraction:plasma cell", "fraction of the clone's cells sorted as plasma cells"),
               ("fate", "fraction:Myc+ light zone", "fraction of the clone's cells sorted as Myc-GFP+ light zone "
                                                    "(recently positively selected; c-Myc-GFP mice)"),
               ("fate", "fraction:dark zone", "fraction of the clone's cells sorted as dark zone"),
               ("w33l", "fraction:W33L", "fraction of the clone's IGHV1-72 cells carrying the high-affinity W33L"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=["plasma_cell"],
    ),
    "malaria": dict(
        title="GSE286215 - terminal splenic sampling during PcAS infection: day 0 (one naive mouse), "
              "day 4 (four infected mice), days 7/10/14 (five each); author transcriptomic annotations",
        batch_key=None, state_key="cell_state",
        tests=[("cell_state", "fraction:GC", "fraction of the clone's cells in a germinal centre"),
               ("cell_state", "fraction:PB", "fraction of the clone's cells that are plasmablasts"),
               ("cell_state", "fraction:Memory", "fraction of the clone's cells that are memory cells"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=[],
    ),
    "malaria_late": dict(
        title="GSE286215 experiment 2 - terminal splenic sampling at days 10/14/21/28/35/42: "
              "three saline, three artesunate+pyrimethamine-treated and one naive mouse per day",
        batch_key=None, state_key="cell_state",
        tests=[("cell_state", "fraction:GC", "fraction of the clone's cells in a germinal centre"),
               ("cell_state", "fraction:PB", "fraction of the clone's cells that are plasmablasts"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        donor_level_tests=[("treatment", "majority", "antimalarial treatment of the mouse")],
        memory=[],
    ),
    "bone_marrow_pc": dict(
        title="GSE253857 - human bone-marrow plasma cells and memory B cells, with blood counterparts; some "
              "sorted by what their antibody binds (SARS-CoV-2 spike, recent; tetanus toxoid, decades old)",
        batch_key="donor", state_key=None,
        tests=[("sorted_as", "fraction:plasma cells", "fraction of the clone's cells sorted as plasma cells"),
               ("tissue", "fraction:bone marrow", "fraction of the clone's cells found in the bone marrow"),
               ("antigen", "majority", "antigen the clone's antibody was sorted on"),
               ("isotype", "majority", "heavy-chain isotype")],
        memory=["tissue", "sorted_as"],
    ),
    "flu_lung": dict(
        title="GSE317692 - influenza A (PR8) infection, mouse lung and mediastinal lymph node at day 20, "
              "haemagglutinin tetramers and a B-cell alpha-v integrin knockout (15 mice)",
        batch_key=None, state_key=None,
        tests=[("ha_binding", "fraction:PR8HA", "fraction of the clone's cells binding the infecting strain's "
                                                "haemagglutinin"),
               ("ha_binding", "fraction:both strains", "fraction of the clone's cells binding both strains "
                                                       "(cross-reactive)"),
               ("tissue", "fraction:Lung", "fraction of the clone's cells found in the infected lung"),
               ("mutation_frequency", "mean", "V-region mutation frequency"),
               ("isotype", "majority", "heavy-chain isotype")],
        donor_level_tests=[("genotype", "majority", "B-cell alpha-v integrin knockout or control")],
        memory=["tissue", "ha_binding"],
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

# Marker gene sets used to describe programmes and to ask which B-cell programmes are clonally inherited
GENE_SETS_HUMAN = {
    "plasma cell": ["PRDM1", "XBP1", "IRF4", "JCHAIN", "MZB1", "SDC1", "TNFRSF17", "DERL3", "FKBP11", "SSR4"],
    "germinal centre": ["BCL6", "AICDA", "RGS13", "MEF2B", "S1PR2", "LMO2", "MYBL1", "GCSAM", "ELL3", "SERPINA9"],
    "dark zone / cycling": ["CXCR4", "MKI67", "TOP2A", "AURKB", "CCNB1", "HMGB2", "TUBB", "STMN1", "FOXP1", "AICDA"],
    "light zone": ["CD83", "CD86", "MYC", "NFKBIA", "BATF", "EGR2", "IL4I1", "CCL22", "BCL2A1", "NME1"],
    "memory": ["CD27", "CCR6", "GPR183", "TNFRSF13B", "SELL", "CD44", "S1PR1", "KLF2", "ZBTB32", "FCRL4"],
    "naive": ["TCL1A", "FCER2", "IL4R", "BACH2", "SELL", "PLPP5", "YBX3", "BTG1", "CD200", "IGHD"],
    "interferon": ["ISG15", "IFIT1", "IFIT3", "MX1", "IFI44L", "OAS1", "IFI6", "XAF1", "STAT1", "IRF7"],
}


def test_name(label, how):
    """Result key of a label test: the label, or label:level for a per-clone fraction."""
    return f"{label}:{how.split(':', 1)[1]}" if how.startswith("fraction:") else label


def mouse_case(genes):
    return [g[:1] + g[1:].lower() for g in genes]


def gene_sets_for(name):
    mouse_datasets = {"mouse_np", "mouse_rbd", "gc_np_pc", "flu_lung", "malaria", "malaria_late"}
    return {k: mouse_case(v) for k, v in GENE_SETS_HUMAN.items()} if name in mouse_datasets else GENE_SETS_HUMAN


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 200 else f"<array {obj.shape}>"
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def leiden_states(adata, basis="X_threadfin", resolution=1.0):
    """Unsupervised cell states (and a cell UMAP) when the authors' annotation is not available."""
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=basis, n_neighbors=15, random_state=0, key_added="tf_cells")
    sc.tl.leiden(adata, resolution=resolution, neighbors_key="tf_cells", key_added="leiden_state",
                 random_state=0, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(adata, neighbors_key="tf_cells", random_state=0)
    return "leiden_state"


def prepare(name: str, stamp=print, summary: dict | None = None):
    """Load a dataset, define clones within donors, build the expression embedding and cell states.

    Returns ``(adata, cfg, state_key, summary)``.
    """
    cfg = CONFIGS[name]
    summary = {} if summary is None else summary

    # 1. data
    adata, bcr = LOADERS[name]()
    stamp(f"loaded {adata.n_obs} cells x {adata.n_vars} genes; {len(bcr)} cells with a heavy chain")
    bcr["isotype"] = tf.clones.isotype_class(bcr["c_call"]).values
    bcr["donor"] = adata.obs["donor"].astype(str).reindex(bcr.index).values

    # 2. clones, within donors
    bcr = tf.define_clones(bcr, donor_key="donor", out_col="clone_id")
    summary["clone_definition"] = dict(bcr.attrs["clone_definition"])
    tf.attach_bcr(adata, bcr.drop(columns=["donor", "author_clone_id"], errors="ignore"), clone_col="clone_id",
                  summarize=False)
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

    # 3. expression embedding without immunoglobulin genes, and cell states
    tf.pp.prepare_embedding(adata, batch_key=cfg["batch_key"], key_added="X_threadfin", verbose=True)
    state_key = cfg["state_key"] or leiden_states(adata)
    if cfg["state_key"]:
        leiden_states(adata)  # still compute a cell UMAP for the figures
    stamp("embedding and cell states done")
    return adata, cfg, state_key, summary


def programme_signatures(adata, gene_sets, programme_key="clone_programme"):
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


def clone_gene_scores(adata, gene_sets):
    """Descriptive expression signatures; these reuse expression and are not external validation."""
    import scanpy as sc

    tmp = sc.AnnData(X=adata.layers["log_norm"], obs=adata.obs[["clone_id"]].copy(),
                    var=pd.DataFrame(index=adata.var_names))
    coverage = {}
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in tmp.var_names]
        coverage[name] = {"genes_present": present, "n_present": len(present), "n_requested": len(genes)}
        if len(present) >= 3:
            sc.tl.score_genes(tmp, present, score_name=name, random_state=0)
    cols = [name for name in gene_sets if name in tmp.obs]
    scores = tmp.obs.dropna(subset=["clone_id"]).groupby("clone_id", observed=True)[cols].mean()
    return scores, coverage


def main(name: str):
    out = Path(os.environ.get("THREADFIN_RESULTS", HERE / "results")) / name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def stamp(msg):
        print(f"[{name} {time.time() - t0:7.0f}s] {msg}", flush=True)

    summary = {"dataset": name, "title": CONFIGS[name]["title"]}
    adata, cfg, state_key, summary = prepare(name, stamp=stamp, summary=summary)
    context_key = cfg.get("context_key", "donor")
    tf.pl.set_style()

    # Q1. does clone identity shape cell state?
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=context_key, donor_key="donor",
                         representation="mean", verbose=False)
    coh = tf.tl.clonal_coherence(adata, strata_key="sample", n_perm=500, verbose=True)
    vc = adata.uns["threadfin"]["profiles"]["variance_components"]
    summary["coherence"] = {k: coh[k] for k in ("icc", "null_mean", "null_q95", "excess", "p_value",
                                                "n_clones", "n_cells")}
    summary["coherence"]["min_cells_reliability_0.5"] = tf.profiles.VarianceComponents(
        np.asarray(vc["sigma2"]), np.asarray(vc["tau2"]), vc["n0"], vc["n_clones"], vc["n_cells"],
        np.zeros(len(vc["tau2"]))).min_cells_for(0.5)
    stamp(f"coherence: {json.dumps(summary['coherence'])}")
    tf.pl.coherence(adata, save=out / "figures" / "coherence.png")
    plt.close("all")

    # Q2. which clones behave alike?
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=context_key, donor_key="donor",
                         representation="kernel", verbose=True)
    try:
        prog = tf.tl.find_programmes(adata, n_boot=50, verbose=True)
    except ValueError as e:
        prog = None
        summary["programmes"] = {"n": 0, "note": str(e)}
        stamp(f"programmes not defined: {e}")
    if prog is not None:
        progs = adata.uns["threadfin"]["programmes"]
        splits = progs["split_tests"]
        summary["programme_splits"] = {
            "n_leiden_communities": int(len(splits["communities_left"].iloc[0]) +
                                        len(splits["communities_right"].iloc[0])) if not splits.empty else 1,
            "n_merged": int((~splits["split"]).sum()) if not splits.empty else 0,
            "root_pvalue": float(splits["pvalue"].iloc[0]) if not splits.empty else None,
        }
        prog.to_csv(out / "programmes.csv")
        if prog.shape[0] < 2:
            summary["programmes"] = {"n": 1, "n_clones": int(prog["n_clones"].sum()),
                                     "note": "no distinct programmes: clones vary along a continuum"}
            stamp("programmes: none distinct (continuum)")
            prog = None
        else:
            tf.tl.programme_composition(adata, state_key).to_csv(out / "programme_composition.csv")
            markers = tf.tl.programme_markers(adata, layer="log_norm", context_key=context_key, n_top=50,
                                              verbose=True)
            markers.to_csv(out / "programme_markers.csv", index=False)
            signatures = programme_signatures(adata, gene_sets_for(name))
            signatures.to_csv(out / "programme_signatures.csv")
            summary["programme_markers"] = {p_: g["gene"].head(12).tolist() for p_, g in markers.groupby("programme")}
            summary["programme_signatures"] = signatures.round(3).to_dict(orient="index")
            summary["programmes"] = {
                "n": int(prog.shape[0]), "resolution": progs["params"]["resolution"],
                "n_clones": int(prog["n_clones"].sum()), "n_cells": int(prog["n_cells"].sum()),
                "n_clones_with_assigned": int(prog["n_clones_with_assigned"].sum()),
                "stability": prog["stability"].round(3).to_dict(),
                "n_stable": int((prog["stability"] >= 0.75).sum()),
            }
            stamp(f"programmes: {summary['programmes']}")

    # Q3. do the measured labels explain how clones differ?
    label_effects = {}
    tests = cfg["tests"] + cfg.get("donor_level_tests", [])
    for label, how, note in tests:
        if label not in adata.obs or adata.obs[label].notna().sum() < 20:
            continue
        donor_level = (label, how, note) in cfg.get("donor_level_tests", [])
        try:
            r = tf.tl.profile_association(adata, label, how=how, n_perm=2000,
                                          strata_key=None if donor_level else "auto", verbose=True)
        except ValueError as e:
            stamp(f"  {label} skipped: {e}")
            continue
        r["note"] = note + (" (donor-level label: not separable from other donor differences)"
                            if donor_level else "")
        r["label"] = test_name(label, how)
        label_effects[test_name(label, how)] = r
    summary["label_effects"] = label_effects
    if label_effects:
        pd.DataFrame([{k: v for k, v in r.items() if k != "axis_correlation"} for r in label_effects.values()]) \
            .to_csv(out / "label_effects.csv", index=False)
        tf.pl.label_effects(label_effects, save=out / "figures" / "label_effects.png")
        plt.close("all")

    assoc, assoc_assigned = [], []
    for label, how, note in (tests if prog is not None else []):
        if label not in adata.obs or adata.obs[label].notna().sum() < 20:
            continue
        donor_level = (label, how, note) in cfg.get("donor_level_tests", [])
        try:
            res = tf.tl.association_test(adata, label, how=how, n_perm=2000,
                                         strata_key=None if donor_level else "auto", verbose=True)
        except ValueError as e:
            stamp(f"  programme test {label} skipped: {e}")
            continue
        res["label"] = test_name(label, how)
        res["note"], res["how"], res["strata"], res["programmes"] = note, how, res.attrs["strata_key"], "core"
        assoc.append(res)
        try:  # the same test on core + confidently assigned clones
            res2 = tf.tl.association_test(adata, label, how=how, programme_key="clone_programme_assigned",
                                          strata_key=None if donor_level else "auto", n_perm=2000, verbose=True)
            res2["label"] = test_name(label, how)
            res2["note"], res2["how"], res2["strata"] = note, how, res2.attrs["strata_key"]
            res2["programmes"] = "core + assigned"
            assoc_assigned.append(res2)
        except (ValueError, KeyError) as e:
            stamp(f"  assigned-programme test {label} skipped: {e}")
    if assoc:
        pd.concat(assoc, ignore_index=True).to_csv(out / "programme_associations.csv", index=False)
    if assoc_assigned:
        pd.concat(assoc_assigned, ignore_index=True).to_csv(out / "programme_associations_assigned.csv", index=False)
    summary["programme_associations"] = {
        a["label"].iloc[0]: {"n_clones": int(a.attrs.get("n_clones", 0)),
                             "n_significant_fdr05": int((a["fdr"] < 0.05).sum())} for a in assoc}

    # Q4. do clones keep their state?
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
    summary["memory"] = mem

    # Q5. which genes are clonally inherited?
    try:
        herit = tf.tl.gene_heritability(adata, context_key=context_key, strata_key="sample",
                                        layer="log_norm", n_perm=100, verbose=True)
        herit.to_csv(out / "gene_heritability.csv")
        gs = tf.tl.geneset_heritability(herit, gene_sets_for(name))
        gs.to_csv(out / "geneset_heritability.csv", index=False)
        summary["heritability"] = {"n_genes": int(herit.shape[0]),
                                   "median_icc": float(herit["icc"].median()),
                                   "top_genes": herit.head(25).index.tolist(),
                                   "receptor_gene_icc": adata.uns["threadfin"].get("gene_heritability_receptor_icc"),
                                   "gene_sets": gs.set_index("gene_set")[
                                       ["median_icc", "matched_background_median_icc", "pvalue_vs_matched"]]
                                   .round(4).to_dict(orient="index") if not gs.empty else {}}
    except Exception as e:  # keep the run going; report the failure
        summary["heritability"] = {"error": repr(e)}

    # tables and figures
    table = adata.uns["threadfin"]["profiles"]["clone_table"].copy()
    for label, how, _ in tests:
        if label in adata.obs:
            table[test_name(label, how)] = tf.tl.clone_labels(adata, label, how=how).reindex(table.index)
    table.to_csv(out / "clone_table.csv")
    scores, coverage = clone_gene_scores(adata, gene_sets_for(name))
    scores.to_csv(out / "clone_gene_scores.csv")
    summary["signature_coverage"] = coverage
    figures(adata, state_key, out, has_programmes=prog is not None, gene_sets=gene_sets_for(name))
    cols = [c for c in ("clone_id", "clone_programme", "clone_programme_assigned", state_key, "donor", "sample",
                        "timepoint", "tissue", "isotype", "mutation_frequency", "spike_binding", "gfp",
                        "division_gate", "rbd_bait", "zone_gate", "w33l", "severity", "fate", "compartment",
                        "n_mutations", "treatment", "sorted_as", "antigen", "vaccine", "genotype") if c in adata.obs]
    cells = adata.obs[list(dict.fromkeys(cols))].copy()
    if "X_umap" in adata.obsm:
        cells["umap_1"], cells["umap_2"] = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
    cells.to_csv(out / "cells.csv.gz")

    summary["runtime_min"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1, default=str))
    stamp("done")


def figures(adata, state_key, out, *, has_programmes, gene_sets):
    figs = out / "figures"
    tf.pl.set_style()
    if has_programmes:
        tf.pl.overview(adata, state_key=state_key, save=figs / "overview.png")
        tf.pl.clone_map(adata, save=figs / "clone_map_programmes.png")
        tf.pl.stability(adata, save=figs / "stability.png")
        tf.pl.programme_composition(adata, state_key, save=figs / "programme_composition.png")
        for label, tab in adata.uns["threadfin"].get("associations", {}).items():
            try:
                tf.pl.association(tab, save=figs / f"programme_association_{label}.png")
            except Exception:
                pass
    for key, res in adata.uns["threadfin"].get("memory", {}).items():
        tf.pl.memory(res, save=figs / f"memory_{key}.png")
    herit = adata.uns["threadfin"].get("gene_heritability")
    if herit is not None:
        tf.pl.heritability(herit, highlight={k: gene_sets[k] for k in ("plasma cell", "germinal centre",
                                                                          "dark zone / cycling")},
                           save=figs / "gene_heritability.png")
    plt.close("all")


if __name__ == "__main__":
    main(sys.argv[1])
