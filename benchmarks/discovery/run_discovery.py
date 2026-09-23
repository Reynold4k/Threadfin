"""Discovery-layer biological reconstruction per dataset.

Reuses the biological_validation loaders/preprocessing, reclusters clones
with the same reference parameters (PCA basis, resolution 0.3), then runs
the v3 discovery analyses that connect clone communities to antigen
specificity, affinity maturation (SHM), tissue/time migration, gene
programmes and public clones.

Usage:
    python run_discovery.py <config.json>     # configs are shared with
                                              # benchmarks/biological_validation
Outputs go to benchmarks/discovery/results/<name>/ :
    discovery.json, *.csv tables, figures/*.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "biological_validation"))

import threadfin as tf  # noqa: E402
from loaders import LOADERS  # noqa: E402
from validation import NpEncoder, standard_preprocess  # noqa: E402

SIGNATURES = {
    "plasmablast": ["XBP1", "JCHAIN", "MZB1", "PRDM1", "SDC1", "IRF4", "DERL3"],
    "germinal_centre": ["AICDA", "BCL6", "MKI67", "RGS13", "MEF2B", "S1PR2"],
    "naive": ["TCL1A", "IL4R", "FCER2", "IGHD"],
    "memory": ["CD27", "GPR183", "FCRL5", "TNFRSF13B"],
    "interferon": ["ISG15", "IFIT1", "MX1", "IFI44L", "OAS1"],
    "atypical_b": ["ITGAX", "TBX21", "FCRL5", "ZEB2"],
}

_TP_ORDER = {"d0": 0, "d4": 4, "d7": 7, "d14": 14, "d21": 21,
             "d28": 28, "d35": 35, "d28+d35": 31.5, "d60": 60,
             "d110": 110, "d201": 201, "pre": 0}


def _tp_sort_key(v):
    return _TP_ORDER.get(str(v), 1e9)


def _save_fig(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[discovery] wrote {out / name}", flush=True)


def _enrichment_table(adata, cluster_key, label_key):
    """Fisher enrichment of label_key values per community (reuse metrics)."""
    return tf.metrics.state_enrichment(adata, cluster_key, label_key)


# ---------------------------------------------------------------- LN vaccine


def discover_ln(adata, out: Path, summary: dict):
    """Spike specificity (author s_pos_clone + ELISA), SHM maturation,
    LN<->blood migration, community programmes."""
    # --- per-clone specificity labels from author annotation
    spos = adata.obs["bcr_s_pos_clone"].astype("string")
    lab = pd.DataFrame({"clone_id": adata.obs["clone_id"], "spos": spos}).dropna()
    lab["is_spos"] = lab["spos"] == "TRUE"
    frac = lab.groupby("clone_id")["is_spos"].mean()
    labels = pd.DataFrame({
        "clone_id": frac.index,
        "spike_specific": np.where(frac >= 0.5, "S+", "S-"),
    })
    adata = tf.annotate_specificity(adata, labels=labels, label_col="spike_specific",
                                    key_added="spike_specific")
    n_spos_clones = int((frac >= 0.5).sum())
    summary["n_spike_pos_clones"] = n_spos_clones
    summary["n_spike_neg_clones"] = int((frac < 0.5).sum())

    # ELISA-validated mAb labels (stricter, clone-level any-TRUE)
    elisa = adata.obs.get("bcr_elisa")
    if elisa is not None:
        el = pd.DataFrame({"clone_id": adata.obs["clone_id"],
                           "hit": elisa.astype("string") == "TRUE"}).dropna(subset=["clone_id"])
        any_hit = el.groupby("clone_id")["hit"].max()
        el_labels = pd.DataFrame({"clone_id": any_hit.index,
                                  "elisa_validated": np.where(any_hit, "ELISA+", "ELISA-")})
        adata = tf.annotate_specificity(adata, labels=el_labels,
                                        label_col="elisa_validated",
                                        key_added="elisa_validated")
        summary["n_elisa_validated_clones"] = int(any_hit.sum())

    enr = tf.specificity_enrichment(adata, specificity_key="spike_specific")
    enr.to_csv(out / "specificity_enrichment.csv", index=False)
    top = enr[enr["specificity"] == "S+"].sort_values("odds_ratio", ascending=False)
    summary["spike_enrichment_top"] = top.head(5).to_dict("records")

    # --- SHM (author per-sequence V-region mutation frequency)
    shm = pd.to_numeric(adata.obs.get("bcr_nuc_RS_freq_19_312"), errors="coerce")
    adata.obs["shm"] = shm
    if shm.notna().sum() > 100:
        grad = tf.clones.shm_gradient_test(adata, cluster_key="clone_cluster",
                                           mut_col="shm")
        summary["shm_gradient"] = {
            "kruskal_H": grad["kruskal_H"], "kruskal_p": grad["kruskal_p"],
            "community_medians": {str(k): v for k, v in
                                  grad["community_medians"].items()},
        }
        # maturation curve: median SHM per timepoint x specificity
        df = pd.DataFrame({
            "tp": adata.obs["timepoint"].astype(str),
            "spec": adata.obs["spike_specific"].astype(str),
            "shm": shm,
        }).dropna()
        df = df[df["spec"].isin(("S+", "S-"))]
        curve = (df.groupby(["spec", "tp"])["shm"]
                 .agg(["median", "count"]).reset_index())
        curve = curve[curve["count"] >= 20]
        curve["order"] = curve["tp"].map(_tp_sort_key)
        curve = curve.sort_values("order")
        curve.to_csv(out / "shm_maturation_curve.csv", index=False)
        # per-timepoint S+ vs S- Mann-Whitney
        mw = []
        for tp, sub in df.groupby("tp"):
            a = sub.loc[sub["spec"] == "S+", "shm"]
            b = sub.loc[sub["spec"] == "S-", "shm"]
            if len(a) >= 20 and len(b) >= 20:
                u = stats.mannwhitneyu(a, b, alternative="greater")
                mw.append({"timepoint": tp, "median_S+": a.median(),
                           "median_S-": b.median(), "p_S+_greater": u.pvalue})
        mwd = pd.DataFrame(mw).sort_values("timepoint", key=lambda s: s.map(_tp_sort_key))
        mwd.to_csv(out / "shm_spos_vs_sneg.csv", index=False)
        summary["shm_maturation"] = mwd.to_dict("records")

    # --- LN <-> blood migration, S+ vs S- clones
    mig = {}
    for label, mask in (("all", adata.obs["clone_id"].notna()),
                        ("S+", adata.obs["spike_specific"].astype(str) == "S+"),
                        ("S-", adata.obs["spike_specific"].astype(str) == "S-")):
        sub = adata[mask & adata.obs["tissue"].notna()].copy()
        if sub.obs["clone_id"].nunique() < 10:
            continue
        m = tf.migration_index(sub, group_key="tissue", min_clone_size=3)
        m.to_csv(out / f"migration_tissue_{label.replace('+','pos').replace('-','neg')}.csv")
        if {"LN", "blood"} <= set(m.index):
            mig[label] = float(m.loc["LN", "blood"])
    summary["migration_LN_blood"] = mig

    # --- programmes
    _programmes(adata, out, summary)
    _figure_ln(adata, out, enr)
    return summary


def _figure_ln(adata, out: Path, enr: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # (a) S+ enrichment per community
    e = enr[enr["specificity"] == "S+"].copy()
    e["community"] = e[enr.columns[0]].astype(str)
    e = e.sort_values("odds_ratio", ascending=False)
    ax = axes[0]
    ax.bar(e["community"], np.log10(e["odds_ratio"].clip(lower=1e-3)),
           color=["#b2182b" if f < 0.05 else "#999999" for f in e["fdr"]])
    ax.set_ylabel("log10 OR (S+ clones)")
    ax.set_xlabel("clone community")
    ax.set_title("Spike-specific clones per community\n(red: FDR<0.05)")
    # (b) SHM maturation curve
    ax = axes[1]
    curve_path = out / "shm_maturation_curve.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        for spec, color in (("S+", "#b2182b"), ("S-", "#2166ac")):
            c = curve[curve["spec"] == spec]
            ax.plot(c["order"], c["median"], "o-", color=color, label=spec)
        ax.set_xlabel("days post vaccination")
        ax.set_ylabel("median SHM frequency")
        ax.set_title("Affinity maturation of S+ clones")
        ax.legend()
    # (c) signature scores per community
    ax = axes[2]
    sig_path = out / "community_scores.csv"
    if sig_path.exists():
        sc_df = pd.read_csv(sig_path)
        pivot = sc_df.pivot_table(index="signature", columns="community",
                                  values="score")
        z = pivot.sub(pivot.mean(axis=1), axis=0)
        im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r")
        ax.set_xticks(range(len(z.columns)), [str(c) for c in z.columns])
        ax.set_yticks(range(len(z.index)), z.index)
        ax.set_xlabel("clone community")
        ax.set_title("Gene programmes per community")
        fig.colorbar(im, ax=ax, shrink=0.8)
    _save_fig(fig, out / "figures", "discovery_ln.png")


# ---------------------------------------------------------------- Stephenson


def discover_stephenson(adata, out: Path, summary: dict, cfg: dict):
    """Severity enrichment + CoV-AbDab reference matching + fine states."""
    # severity column (h5mu gex obs, names may vary)
    sev_col = None
    for cand in ("Status_on_day_collection_summary", "gex:Status_on_day_collection_summary",
                 "Worst_Clinical_Status"):
        if cand in adata.obs.columns:
            sev_col = cand
            break
    if sev_col:
        adata.obs["severity"] = adata.obs[sev_col].astype(str)
        enr = _enrichment_table(adata, "clone_cluster", "severity")
        enr.to_csv(out / "severity_enrichment.csv", index=False)
        summary["severity_enrichment_top"] = (
            enr.sort_values("odds_ratio", ascending=False).head(6).to_dict("records"))

    # CoV-AbDab reference matching
    ref = cfg.get("cov_abdab")
    if ref and Path(ref).exists():
        adata = tf.annotate_specificity(adata, reference=ref, identity=0.9,
                                        key_added="cov_abdab")
        hits = adata.uns.get("cov_abdab_clones")
        summary["cov_abdab_matched_clones"] = 0 if hits is None else int(len(hits))
        if hits is not None and len(hits):
            hits.to_csv(out / "cov_abdab_matches.csv", index=False)
            enr2 = tf.specificity_enrichment(adata, specificity_key="cov_abdab")
            enr2.to_csv(out / "cov_abdab_enrichment.csv", index=False)
            summary["cov_abdab_enrichment"] = enr2.to_dict("records")

    # fine-grained author states
    fine = None
    for cand in ("full_clustering", "gex:full_clustering"):
        if cand in adata.obs.columns:
            fine = cand
            break
    if fine:
        conc = tf.metrics.state_concordance(adata, "clone_cluster", fine)
        summary["fine_state_concordance"] = conc
        _enrichment_table(adata, "clone_cluster", fine).to_csv(
            out / "fine_state_enrichment.csv", index=False)
    _programmes(adata, out, summary)
    return summary


# ---------------------------------------------------------------- flu


def discover_flu(adata, out: Path, summary: dict):
    """Age x timepoint expansion dynamics + public clones."""
    adata.obs["age_tp"] = (adata.obs["condition"].astype(str) + "_" +
                           adata.obs["timepoint"].astype(str))
    expa = tf.expansion_index(adata, group_key="age_tp")
    expa.to_csv(out / "expansion_index_age_tp.csv")
    summary["expansion_index_age_tp"] = {str(k): float(v) for k, v in expa.items()}

    mig = tf.migration_index(adata, group_key="timepoint", min_clone_size=3)
    mig.to_csv(out / "migration_timepoint.csv")

    pub = tf.clones.public_clone_summary(adata, donor_key="donor")
    pub.to_csv(out / "public_clones.csv", index=False)
    summary["n_public_clones"] = int(len(pub))
    if len(pub):
        summary["public_clones_top"] = pub.head(10).to_dict("records")
    _programmes(adata, out, summary)
    return summary


# ---------------------------------------------------------------- EBV


def discover_ebv(adata, out: Path, summary: dict):
    """GFP+ infection enrichment + time migration + programmes."""
    enr = _enrichment_table(adata, "clone_cluster", "condition")
    enr.to_csv(out / "gfp_enrichment.csv", index=False)
    top = enr[enr["state"] == "GFP+"].sort_values("odds_ratio", ascending=False)
    summary["gfp_enrichment_top"] = top.head(5).to_dict("records")
    mig = tf.migration_index(adata, group_key="timepoint", min_clone_size=3)
    mig.to_csv(out / "migration_timepoint.csv")
    _programmes(adata, out, summary)
    return summary


# ---------------------------------------------------------------- tonsil


def discover_tonsil(adata, out: Path, summary: dict):
    """Programmes + markers for the maturation-axis communities."""
    _programmes(adata, out, summary)
    shm_col = "bcr_mu_count" if "bcr_mu_count" in adata.obs.columns else None
    if shm_col or "bcr_v_identity" in adata.obs.columns:
        grad = tf.clones.shm_gradient_test(
            adata, cluster_key="clone_cluster",
            mut_col=shm_col or "bcr_mu_count")
        summary["shm_gradient"] = {
            "kruskal_H": grad["kruskal_H"], "kruskal_p": grad["kruskal_p"],
            "community_medians": {str(k): v for k, v in
                                  grad["community_medians"].items()},
        }
    return summary


# ---------------------------------------------------------------- shared


def _programmes(adata, out: Path, summary: dict):
    markers = tf.community_markers(adata, method="t-test_overestim_var", n_genes=15)
    markers.to_csv(out / "community_markers.csv", index=False)
    scores = tf.community_score(adata, SIGNATURES)
    scores.to_csv(out / "community_scores.csv", index=False)
    top = (markers[markers["pval_adj"] < 0.05]
           .sort_values("logfoldchange", ascending=False)
           .groupby("community").head(3))
    summary["community_top_markers"] = {
        str(c): g["gene"].tolist() for c, g in top.groupby("community")}


DISCOVERY = {
    "ln_vaccine_gse195673": discover_ln,
    "stephenson2021": discover_stephenson,
    "flu_gse175522": discover_flu,
    "ebv_organoid_gse317492": discover_ebv,
    "tonsil_king2021": discover_tonsil,
}


def main():
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    name = cfg["name"]
    out = HERE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[discovery] {name}: loading", flush=True)
    adata, bcr = LOADERS[cfg["loader"]](cfg)
    print(f"[discovery] {name}: preprocess ({adata.n_obs} cells)", flush=True)
    adata = standard_preprocess(adata, batch_key=cfg.get("batch_key"))
    tf.attach_bcr(adata, bcr, clone_col="clone_id")

    min_cs = cfg.get("min_clone_size", 3)
    print(f"[discovery] {name}: recluster (min_clone_size={min_cs})", flush=True)
    tf.clonotype_recluster(adata, basis="X_pca", min_clone_size=min_cs,
                           resolution=0.3, random_state=0)
    n_com = adata.obs["clone_cluster"].nunique()
    print(f"[discovery] {name}: {n_com} communities", flush=True)

    summary = {"name": name, "n_communities": int(n_com),
               "min_clone_size": int(min_cs)}
    fn = DISCOVERY.get(name)
    if fn is not None:
        print(f"[discovery] {name}: running discovery block", flush=True)
        summary = fn(adata, out, summary, cfg) if name == "stephenson2021" \
            else fn(adata, out, summary)

    summary["runtime_min"] = round((time.time() - t0) / 60, 1)
    (out / "discovery.json").write_text(
        json.dumps(summary, indent=1, cls=NpEncoder))
    print(f"[discovery] {name}: done in {summary['runtime_min']} min", flush=True)


if __name__ == "__main__":
    main()
