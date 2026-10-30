#!/usr/bin/env python
"""What the activated / non-GC B cells add to the gut clone map (GSE284622).

Reads the outputs of ``allB_threadfin_run.py`` and asks four questions, all with
measurements that never entered a clone profile (tdTomato GC-fate sort, SHM,
heavy-chain isotype, tissue):

1. Cells: can RNA alone tell GC-experienced (tdTomato+) from GC-naive (tdTomato-)
   B cells inside the authors' non-GC B ("naive/memory") cluster?
2. Clones: where do clones carrying B cells sit on the all-B clone map, and what
   are the B-only clones?
3. Axis: do GC history, SHM, class-switch order (IgM/IgD -> IgG1 -> IgA) and tissue
   change along the main axis of clone profiles, also *within* a compartment?
4. Memory output (held-out design): profiles are rebuilt from each clone's GC and PC
   cells only; which of those clones also contain GC-experienced B cells?

Outputs: ``allB_threadfin/story/`` (tables + story_stats.json).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
import threadfin as tf  # noqa: E402

RUN = HERE / "allB_threadfin"
OUT = RUN / "story"
VARIANT = os.environ.get("ALLB_VARIANT", "none")
MAP = dict(n_neighbors=30, min_dist=0.3)
SEED = 0


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def pattern(t: pd.DataFrame) -> pd.Series:
    """Which compartments a clone was captured in (B = non-GC B, GC, PC)."""
    parts = []
    for k in ("B", "GC", "PC"):
        parts.append(np.where(t[f"n_{k}"] > 0, k, ""))
    lab = ["+".join([p for p in trio if p]) for trio in zip(*parts)]
    return pd.Series(lab, index=t.index)


# ------------------------------------------------------------------ 1. cells
def cell_separability(adata) -> dict:
    """Cross-validated tdTom+ vs tdTom- within non-GC B cells, from the RNA embedding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier

    obs = adata.obs
    b = (obs["compartment"] == "B").to_numpy()
    x = np.asarray(adata.obsm["X_threadfin"])[b]
    y = (obs.loc[b, "fate"] == "tdTom+").to_numpy().astype(int)
    out = {"n_B": int(b.sum()), "n_tdpos": int(y.sum())}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_lr = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), x, y, cv=cv,
                             method="predict_proba")[:, 1]
    p_knn = cross_val_predict(KNeighborsClassifier(15), x, y, cv=cv, method="predict_proba")[:, 1]
    out["auc_logistic"] = float(roc_auc_score(y, p_lr))
    out["auc_knn15"] = float(roc_auc_score(y, p_knn))
    # the same, restricted to resting (non-activated) B cells
    rest = (obs.loc[b, "b_state"] == "resting").to_numpy()
    if y[rest].sum() >= 20:
        p = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), x[rest], y[rest],
                              cv=cv, method="predict_proba")[:, 1]
        out["auc_logistic_resting"] = float(roc_auc_score(y[rest], p))
    # measured SHM of the two groups (the reporter's own validation)
    shm = pd.to_numeric(obs.loc[b, "mutation_frequency"], errors="coerce").to_numpy()
    for lab, sel in (("tdpos", y == 1), ("tdneg", y == 0)):
        v = shm[sel & np.isfinite(shm)]
        out[f"shm_{lab}"] = {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                             "frac_mutated": float((v > 0).mean())}
    return out


# ------------------------------------------------------------------ 3. axis
def axis_gradients(t: pd.DataFrame, axis: str) -> dict:
    """Spearman of measured clone labels with the profile axis, overall and within strata."""
    res = {}
    labels = ["frac_tdpos", "shm_mean", "frac_IgM_D", "frac_IgG1", "frac_IgA", "frac_LP", "frac_PP",
              "frac_MLN", "frac_B", "frac_GC", "frac_PC"]
    for lab in labels:
        ok = t[[axis, lab]].dropna()
        r, p = stats.spearmanr(ok[axis], ok[lab])
        res[lab] = {"rho": float(r), "p": float(p), "n": int(len(ok))}
        per = {}
        for d, g in ok.join(t["donor"]).groupby("donor"):
            if len(g) >= 15:
                per[d] = float(stats.spearmanr(g[axis], g[lab])[0])
        res[lab]["per_mouse"] = per
    # within compartment-dominated clones: does the axis still order isotype / SHM / tissue?
    within = {}
    for comp in ("GC", "PC"):
        sub = t[t[f"frac_{comp}"] >= 0.8]
        w = {}
        for lab in ("frac_IgA", "frac_IgG1", "shm_mean", "frac_LP", "frac_tdpos"):
            ok = sub[[axis, lab]].dropna()
            if len(ok) >= 20 and ok[lab].nunique() > 1:
                r, p = stats.spearmanr(ok[axis], ok[lab])
                w[lab] = {"rho": float(r), "p": float(p), "n": int(len(ok))}
        within[f"{comp}>=0.8"] = w
    res["within_compartment"] = within
    return res


def quantile_profile(t: pd.DataFrame, axis: str, n=5) -> pd.DataFrame:
    q = pd.qcut(t[axis].rank(method="first"), n, labels=[f"Q{i + 1}" for i in range(n)])
    cols = ["frac_B", "frac_GC", "frac_PC", "frac_tdpos", "shm_mean", "frac_IgM_D", "frac_IgG1",
            "frac_IgA", "frac_PP", "frac_MLN", "frac_LP", "n_cells"]
    prof = t.groupby(q, observed=True)[cols].mean()
    prof["n_clones"] = t.groupby(q, observed=True).size()
    prof["has_B_tdpos"] = t.groupby(q, observed=True)["n_B_tdpos"].apply(lambda s: float((s > 0).mean()))
    return prof


# ------------------------------------------------------------------ 4. memory output
def held_out_memory(adata, desc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Profiles from GC + PC cells only; B cells are held out and overlaid afterwards."""
    a = adata.copy()
    a.obs["clone_gcpc"] = a.obs["clone_id"].where(a.obs["compartment"].isin(["GC", "PC"]))
    a.obs.loc[a.obs["clone_gcpc"].astype(str).isin(["nan", "None"]), "clone_gcpc"] = np.nan
    tf.tl.clone_profiles(a, clone_key="clone_gcpc", basis="X_threadfin", context_key=None, donor_key="donor",
                         representation="kernel", min_cells=2, verbose=True)
    tf.tl.embed_clones(a, **MAP, min_reliability=0.0, verbose=True)
    prof = a.uns["threadfin"]["profiles"]
    t = prof["clone_table"].copy()
    t["kpc0"] = prof["features"]["kpc0"].reindex(t.index)
    t["kpc1"] = prof["features"]["kpc1"].reindex(t.index)
    t = t.join(desc.drop(columns=["n_cells", "donor"]), how="left")
    gcpc = a.obs[a.obs["compartment"].isin(["GC", "PC"])].dropna(subset=["clone_gcpc"])
    t["n_tis_GCPC"] = gcpc.groupby(gcpc["clone_gcpc"].astype(str))["tissue"].nunique().reindex(t.index)
    # clone-level held-out labels: B cells of the same clone (never in the profile)
    t["has_B"] = t["n_B"] > 0
    t["has_B_tdpos"] = t["n_B_tdpos"] > 0
    t["has_B_tdneg"] = t["n_B_tdneg"] > 0
    t["n_GCPC"] = t["n_GC"] + t["n_PC"]
    t["frac_PC_of_GCPC"] = t["n_PC"] / t["n_GCPC"]
    # orient kpc0 so that it increases from GC to PC
    if stats.spearmanr(t["kpc0"], t["frac_PC_of_GCPC"])[0] < 0:
        t["kpc0"] = -t["kpc0"]
    res = {"n_clones": int(len(t)), "n_with_B_tdpos": int(t["has_B_tdpos"].sum()),
           "n_with_B_tdneg": int(t["has_B_tdneg"].sum())}
    # where are memory-carrying clones? three classes of GC/PC composition
    t["output_class"] = pd.cut(t["frac_PC_of_GCPC"], [-0.01, 0.0001, 0.9999, 1.01],
                               labels=["GC only", "GC + PC", "PC only"])
    tab = t.groupby("output_class", observed=True).agg(n=("has_B_tdpos", "size"),
                                                       with_memB=("has_B_tdpos", "sum"),
                                                       median_size=("n_GCPC", "median"))
    tab["frac"] = tab["with_memB"] / tab["n"]
    res["by_output_class"] = tab.to_dict(orient="index")
    # logistic model: memory B presence ~ output class + log clone size + mouse
    import statsmodels.formula.api as smf

    df = t[["has_B_tdpos", "output_class", "n_GCPC", "donor", "kpc0", "frac_LP", "n_tis_GCPC"]].dropna().copy()
    df["y"] = df["has_B_tdpos"].astype(int)
    df["log_n"] = np.log(df["n_GCPC"])
    df["oc"] = df["output_class"].astype(str)
    try:
        m = smf.logit("y ~ C(oc, Treatment('GC only')) + log_n + C(donor)", data=df).fit(disp=False)
        res["logit_output_class"] = {k: {"coef": float(m.params[k]), "p": float(m.pvalues[k]),
                                         "odds_ratio": float(np.exp(m.params[k]))} for k in m.params.index}
        m2 = smf.logit("y ~ kpc0 + I(kpc0**2) + log_n + C(donor)", data=df).fit(disp=False)
        res["logit_axis_quadratic"] = {k: {"coef": float(m2.params[k]), "p": float(m2.pvalues[k])}
                                       for k in m2.params.index}
        m3 = smf.logit("y ~ C(oc, Treatment('GC only')) + log_n + frac_LP + C(donor)", data=df).fit(disp=False)
        res["logit_output_class_plus_LP"] = {k: {"coef": float(m3.params[k]), "p": float(m3.pvalues[k]),
                                                 "odds_ratio": float(np.exp(m3.params[k]))}
                                             for k in m3.params.index}
        # caveat check: GC + PC clones are captured in more tissues, so more B-cell libraries see them
        m4 = smf.logit("y ~ C(oc, Treatment('GC only')) + log_n + C(n_tis_GCPC) + C(donor)", data=df).fit(disp=False)
        res["logit_output_class_plus_ntissues"] = {k: {"coef": float(m4.params[k]), "p": float(m4.pvalues[k]),
                                                       "odds_ratio": float(np.exp(m4.params[k]))}
                                                   for k in m4.params.index}
    except Exception as e:  # noqa: BLE001
        res["logit_error"] = str(e)
    # size-matched permutation: shuffle memory-B presence among clones of the same size bin and mouse
    rng = np.random.default_rng(SEED)
    bins = pd.cut(df["n_GCPC"], [1, 2, 3, 5, 10, 1e6], labels=False)
    strata = df["donor"].astype(str) + "_" + bins.astype(str)
    mixed = (df["oc"] == "GC + PC").to_numpy()
    obs_rate = float(df.loc[mixed, "y"].mean())
    null = []
    y = df["y"].to_numpy()
    for _ in range(2000):
        perm = y.copy()
        for s in strata.unique():
            sel = np.flatnonzero((strata == s).to_numpy())
            perm[sel] = rng.permutation(perm[sel])
        null.append(perm[mixed].mean())
    null = np.asarray(null)
    res["mixed_clone_memB_rate"] = {"observed": obs_rate, "null_mean": float(null.mean()),
                                    "null_q975": float(np.quantile(null, 0.975)),
                                    "p_two_sided": float(min(1, 2 * min((null >= obs_rate).mean(),
                                                                        (null <= obs_rate).mean()) + 1 / 2000))}
    return t, res


def within_clone_shm(cells: pd.DataFrame) -> dict:
    """Paired within-clone SHM: GC-experienced B vs GC vs PC cells of the same clone."""
    c = cells.dropna(subset=["clone_id"]).copy()
    c["shm"] = pd.to_numeric(c["mutation_frequency"], errors="coerce")
    c["grp"] = np.where(c["compartment"] == "B", np.where(c["fate"] == "tdTom+", "B_tdpos", "B_tdneg"),
                        c["compartment"])
    med = c.groupby(["clone_id", "grp"])["shm"].mean().unstack()
    out = {}
    for a_, b_ in (("B_tdpos", "GC"), ("B_tdpos", "PC"), ("PC", "GC"), ("B_tdneg", "GC"), ("B_tdneg", "PC")):
        ok = med[[a_, b_]].dropna()
        if len(ok) >= 5:
            d = (ok[a_] - ok[b_]).round(12)   # equal per-clone means are ties, not float noise
            w = stats.wilcoxon(d) if (d != 0).any() else None
            out[f"{a_}_minus_{b_}"] = {"n_clones": int(len(ok)), "median_diff": float(d.median()),
                                       "n_lower": int((d < 0).sum()), "n_higher": int((d > 0).sum()),
                                       "wilcoxon_p": float(w.pvalue) if w is not None else None}
    return out


def within_clone_isotype(cells: pd.DataFrame) -> dict:
    """Paired within-clone IgG1 share (of IgG1 + IgA cells): GC-experienced B vs GC vs PC."""
    c = cells.dropna(subset=["clone_id"]).copy()
    c = c[c["isotype_fine"].isin(["IgG1", "IgA"])]
    c["g1"] = (c["isotype_fine"] == "IgG1").astype(float)
    c["grp"] = np.where(c["compartment"] == "B", np.where(c["fate"] == "tdTom+", "B_tdpos", "B_tdneg"),
                        c["compartment"])
    share = c.groupby(["clone_id", "grp"])["g1"].mean().unstack()
    out = {}
    for a_, b_ in (("B_tdpos", "GC"), ("B_tdpos", "PC"), ("PC", "GC")):
        ok = share[[a_, b_]].dropna()
        d = (ok[a_] - ok[b_]).round(12)
        w = stats.wilcoxon(d) if (d != 0).any() else None
        out[f"{a_}_minus_{b_}"] = {"n_clones": int(len(ok)), "mean_diff": float(d.mean()),
                                   "n_lower": int((d < 0).sum()), "n_higher": int((d > 0).sum()),
                                   "wilcoxon_p": float(w.pvalue) if w is not None else None}
    return out


def main():
    import anndata as ad

    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(RUN / "allB_embedding.h5ad")
    for col in ("compartment", "fate", "b_state", "donor", "tissue", "clone_id"):
        adata.obs[col] = adata.obs[col].astype(str).replace({"nan": np.nan})
    desc = pd.read_csv(RUN / "clone_descriptors_all_clones.csv", index_col=0)
    t = pd.read_csv(RUN / VARIANT / "clone_table.csv", index_col=0)
    feats = pd.read_csv(RUN / VARIANT / "clone_features.csv.gz", index_col=0)
    t["kpc0"], t["kpc1"] = feats["kpc0"].reindex(t.index), feats["kpc1"].reindex(t.index)
    if stats.spearmanr(t["kpc0"], t["frac_PC"])[0] < 0:
        t["kpc0"] = -t["kpc0"]
    t["pattern"] = pattern(t)
    stats_out = {"variant": VARIANT}

    stats_out["cells"] = cell_separability(adata)
    print("cells:", json.dumps(stats_out["cells"]), flush=True)

    # 2. clone classes on the all-B map
    cls = t.groupby("pattern").agg(n=("n_cells", "size"), median_cells=("n_cells", "median"),
                                   tdpos=("frac_tdpos", "mean"), shm=("shm_mean", "median"),
                                   IgM_D=("frac_IgM_D", "mean"), IgG1=("frac_IgG1", "mean"),
                                   IgA=("frac_IgA", "mean"), LP=("frac_LP", "mean"), PP=("frac_PP", "mean"),
                                   MLN=("frac_MLN", "mean"), x=("x", "median"), y=("y", "median"),
                                   activated=("n_B_activated", "sum"))
    cls.to_csv(OUT / f"clone_classes_{VARIANT}.csv")
    stats_out["clone_classes"] = cls.round(4).to_dict(orient="index")
    b_only = t[t["pattern"] == "B"]
    stats_out["b_only"] = {"n": int(len(b_only)), "cells": int(b_only["n_cells"].sum()),
                           "frac_tdneg_cells": float(1 - (b_only["frac_tdpos"] * b_only["n_cells"]).sum()
                                                     / b_only["n_cells"].sum()),
                           "shm_median": float(b_only["shm_mean"].median()),
                           "frac_clones_unmutated": float((b_only["shm_mean"] == 0).mean()),
                           "IgM_D_mean": float(b_only["frac_IgM_D"].mean()),
                           "by_donor": b_only["donor"].value_counts().to_dict(),
                           "activated_cells": int(b_only["n_B_activated"].sum())}

    # 3. axis
    stats_out["axis"] = axis_gradients(t, "kpc0")
    qp = quantile_profile(t, "kpc0")
    qp.to_csv(OUT / f"axis_quintiles_{VARIANT}.csv")
    print(qp.round(3).to_string(), flush=True)

    # 4. memory output, held-out B cells
    ho, ho_res = held_out_memory(adata, desc)
    ho.to_csv(OUT / "heldout_gcpc_clone_table.csv")
    stats_out["held_out_memory"] = ho_res
    print("held-out:", json.dumps(jsonable(ho_res), indent=1), flush=True)
    stats_out["held_out_axis"] = axis_gradients(ho.assign(frac_B=np.nan), "kpc0")

    cells = pd.read_csv(RUN / "cells_allB.csv.gz", index_col=0)
    stats_out["within_clone_shm"] = within_clone_shm(cells)
    stats_out["within_clone_isotype"] = within_clone_isotype(cells)
    print("within-clone isotype:", json.dumps(stats_out["within_clone_isotype"], indent=1), flush=True)
    print("within-clone SHM:", json.dumps(stats_out["within_clone_shm"], indent=1), flush=True)
    t.to_csv(OUT / f"clone_table_{VARIANT}_with_axis.csv")
    (OUT / f"story_stats_{VARIANT}.json").write_text(json.dumps(jsonable(stats_out), indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
