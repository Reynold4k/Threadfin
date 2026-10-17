#!/usr/bin/env python
"""Gut GC fate-reporter cohort (GSE284622, Siniscalco et al. 2025): analyses for the 2026-10-17 revision.

1. Reproduce the Figure 3 g-j clone profiles (kernel, centred on each mouse) from the cached embedding and
   check them against the saved clone table.
2. Centring diagnostics: the same cells and clones profiled without centring and with a
   composition-balanced mouse shift; donor composition of programmes and regions; mouse mixing among
   plasma cells (cell level) and plasma-cell-dominated clones (clone level); within-mouse association of
   clone profiles with GC history.
3. Clone relationships on the Figure 3 embedding: state span (GC B cells and plasma cells), tissue span,
   isotype (IgG1, IgA), GC-fate sort and V-region mutation frequency, with clone-level permutation tests.

Nothing here changes the saved Figure 3 run. Outputs: reserve/revision_2026-10-17/gut/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

import threadfin as tf

ROOT = Path(__file__).resolve().parents[4]                       # Threadfin/
CACHE = (ROOT.parent / "internal_validation" / "algorithm_revision" / "embedding_cache"
         / "gut_gc_reporter_embedding.h5ad")
RES = ROOT / "case_studies" / "results" / "gut_gc_reporter"
ALLB = ROOT / "paper" / "figure_plan" / "reserve" / "gut_gc_candidates" / "allB_threadfin"
OUT = ROOT / "paper" / "figure_plan" / "reserve" / "revision_2026-10-17" / "gut"
MAP = dict(n_neighbors=30, min_dist=0.3)                           # Figure 3h display settings
MICE = ["mouse2", "mouse3", "mouse4"]
T0 = time.time()


def stamp(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# --------------------------------------------------------------------------- data


def load() -> ad.AnnData:
    a = ad.read_h5ad(CACHE)
    ddl = pd.read_csv(ROOT / "case_studies" / "results" / "dandelion_clones" / "gut_gc_reporter.csv")
    a.obs["clone_id"] = ddl.set_index("cell")["clone_id"].reindex(a.obs_names).values
    cells = pd.read_csv(RES / "cells.csv.gz", index_col=0)
    assert (cells.index == a.obs_names).all(), "cell order differs from the Figure 3 run"
    both = cells["clone_id"].notna() & a.obs["clone_id"].notna()
    assert (cells.loc[both, "clone_id"].values == a.obs.loc[both, "clone_id"].values).all()
    assert cells["clone_id"].notna().sum() == a.obs["clone_id"].notna().sum()
    a.obs["mutation_frequency"] = cells["mutation_frequency"].values
    a.obs["cell_umap_1"], a.obs["cell_umap_2"] = cells["umap_1"].values, cells["umap_2"].values
    fine = pd.read_csv(ALLB / "cells_allB.csv.gz", index_col=0)["isotype_fine"]
    a.obs["isotype_fine"] = fine.reindex(a.obs_names).values
    a.obs["fig3_programme"] = cells["clone_programme"].values
    return a


def balanced_basis(a: ad.AnnData, key: str = "celltype", out: str = "X_balanced") -> dict:
    """Remove each mouse's shift estimated within cell types, so that the shift does not depend on
    how many cells of each type a mouse contributed (same procedure as allB_threadfin_run.py)."""
    x = np.asarray(a.obsm["X_threadfin"], dtype=float)
    donor = a.obs["donor"].astype(str).to_numpy()
    comp = a.obs[key].astype(str).to_numpy()
    levels = sorted(set(comp))
    weight = pd.Series(comp).value_counts(normalize=True).reindex(levels).to_numpy()
    glob = np.vstack([x[comp == k].mean(axis=0) for k in levels])
    y, shifts = x.copy(), {}
    for d in sorted(set(donor)):
        dev = [x[(donor == d) & (comp == k)].mean(axis=0) - glob[i]
               if ((donor == d) & (comp == k)).sum() >= 20 else np.zeros(x.shape[1])
               for i, k in enumerate(levels)]
        shift = np.average(np.vstack(dev), axis=0, weights=weight)
        y[donor == d] -= shift
        shifts[d] = float(np.linalg.norm(shift))
    a.obsm[out] = y
    return shifts


def run_variant(a: ad.AnnData, basis: str, context: str | None, programmes: bool = True) -> dict:
    tf.tl.clone_profiles(a, basis=basis, context_key=context, donor_key="donor", representation="kernel",
                         min_cells=2, verbose=False)
    tf.tl.embed_clones(a, min_reliability=0.0, verbose=False, **MAP)
    summary = {}
    if programmes:
        prog = tf.tl.find_programmes(a, n_boot=50, verbose=False)
        summary = {"n": int(len(prog)), "stability": prog["stability"].round(3).to_dict(),
                   "n_clones": prog["n_clones"].to_dict()}
    prof = a.uns["threadfin"]["profiles"]
    return {"table": prof["clone_table"].copy(), "features": prof["features"].copy(), "programmes": summary}


# --------------------------------------------------------------------------- statistics


def knn_same_mouse(x: np.ndarray, mouse: np.ndarray, k: int = 15, clone: np.ndarray | None = None,
                   n_perm: int = 500, seed: int = 0) -> dict:
    """Fraction of each item's k nearest neighbours from the same mouse, against mouse labels permuted
    among the items. With ``clone``, neighbours from the item's own clone are skipped (cell level)."""
    extra = 250 if clone is not None else 0
    nn = NearestNeighbors(n_neighbors=min(k + 1 + extra, len(x))).fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    items, nbs = [], []
    for i, nb in enumerate(idx):
        if clone is not None and not pd.isna(clone[i]):
            nb = nb[clone[nb] != clone[i]]          # skip the item's own clone
        nb = nb[:k]
        items.append(np.full(nb.size, i)), nbs.append(nb)
    items, nbs = np.concatenate(items), np.concatenate(nbs)
    counts = np.bincount(items, minlength=len(x)).astype(float)
    has = counts > 0

    def per_item(lab):
        same = np.bincount(items, weights=(lab[nbs] == lab[items]).astype(float), minlength=len(x))
        return same[has] / counts[has]

    obs_item = per_item(mouse)
    rng = np.random.default_rng(seed)
    null = np.array([per_item(rng.permutation(mouse)).mean() for _ in range(n_perm)])
    mouse = mouse[has]
    per_mouse = {m: float(obs_item[mouse == m].mean()) for m in MICE if (mouse == m).any()}
    share = {m: float((mouse == m).mean()) for m in MICE}
    return {"n": int(len(x)), "k": k, "observed": float(obs_item.mean()), "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + (null >= obs_item.mean()).sum()) / (1 + n_perm)),
            "observed_by_mouse": per_mouse, "mouse_share": share, "n_perm": n_perm}


def within_mouse_neighbourhood(feats: pd.DataFrame, clones: pd.DataFrame, k: int = 10, n_perm: int = 2000,
                               seed: int = 20261017) -> dict:
    """Within one mouse: do plasma-cell-dominated clones with nonGC-sorted plasma cells neighbour each other?
    Statistic: Spearman correlation between a clone's nonGC share of plasma cells and the mean share of its
    k nearest clones in profile space; null: shares permuted within the mouse (any / within LP strata)."""
    rng = np.random.default_rng(seed)
    out = {}
    pl = clones[(clones["frac_plasma"] >= 0.5) & (clones["n_plasma"] > 0)]
    for m in MICE:
        d = pl[pl["donor"] == m]
        share = (d["n_plasma_nonGC"] / d["n_plasma"]).to_numpy()
        if len(d) < 20 or (share > 0).sum() < 5:
            out[m] = {"n_clones": int(len(d)), "n_with_nonGC_plasma": int((share > 0).sum()),
                      "note": "not testable (fewer than 5 clones with nonGC plasma cells)"}
            continue
        f = feats.loc[d.index].to_numpy()
        idx = NearestNeighbors(n_neighbors=k + 1).fit(f).kneighbors(f, return_distance=False)[:, 1:]

        def stat(s):
            return float(stats.spearmanr(s, s[idx].mean(axis=1)).statistic)

        obs = stat(share)
        lp = (d["frac_LP"] >= 0.5).to_numpy()
        null_any = np.array([stat(rng.permutation(share)) for _ in range(n_perm)])
        null_lp = []
        for _ in range(n_perm):
            s = share.copy()
            for g in (True, False):
                w = np.flatnonzero(lp == g)
                s[w] = rng.permutation(share[w])
            null_lp.append(stat(s))
        null_lp = np.asarray(null_lp)
        out[m] = {"n_clones": int(len(d)), "n_with_nonGC_plasma": int((share > 0).sum()), "rho": obs,
                  "null_mean": float(null_any.mean()), "null_q95": float(np.quantile(null_any, 0.95)),
                  "p": float((1 + (null_any >= obs).sum()) / (1 + n_perm)),
                  "null_mean_lp_strata": float(null_lp.mean()),
                  "p_lp_strata": float((1 + (null_lp >= obs).sum()) / (1 + n_perm)), "k": k, "n_perm": n_perm,
                  "null_any": null_any.round(4).tolist()}
    return out


def clone_descriptors(a: ad.AnnData, table: pd.DataFrame) -> pd.DataFrame:
    obs = a.obs[a.obs["clone_id"].notna()].copy()
    obs["clone_id"] = obs["clone_id"].astype(str)
    obs = obs[obs["clone_id"].isin(table.index)]
    g = obs.groupby("clone_id", observed=True)
    d = pd.DataFrame({"n_cells": g.size(), "donor": g["donor"].first().astype(str)})
    for ct, col in (("GC_Bcell", "n_GC"), ("plasma", "n_plasma"), ("B_cell", "n_B"), ("other", "n_other")):
        d[col] = g["celltype"].apply(lambda s, ct=ct: int((s == ct).sum()))
    for t in ("PP", "MLN", "LP"):
        d[f"n_{t}"] = g["tissue"].apply(lambda s, t=t: int((s == t).sum()))
    d["n_nonGC"] = g["gc_sort"].apply(lambda s: int((s == "nonGC").sum()))
    pcs = obs[obs["celltype"] == "plasma"].groupby("clone_id", observed=True)
    d["n_plasma_nonGC"] = pcs["gc_sort"].apply(lambda s: int((s == "nonGC").sum())).reindex(d.index).fillna(0)
    iso = obs.dropna(subset=["isotype_fine"]).groupby("clone_id", observed=True)["isotype_fine"]
    d["n_isotyped"] = iso.size().reindex(d.index).fillna(0).astype(int)
    for k in ("IgA", "IgG1", "IgM", "IgD", "IgG2b"):
        d[f"n_{k}"] = iso.apply(lambda s, k=k: int((s == k).sum())).reindex(d.index).fillna(0).astype(int)
    d["shm_mean"] = g["mutation_frequency"].mean()
    d["shm_plasma"] = pcs["mutation_frequency"].mean().reindex(d.index)
    d["shm_GC"] = obs[obs["celltype"] == "GC_Bcell"].groupby("clone_id", observed=True)["mutation_frequency"] \
        .mean().reindex(d.index)
    d["frac_plasma"] = d["n_plasma"] / d["n_cells"]
    d["frac_GC"] = d["n_GC"] / d["n_cells"]
    d["frac_LP"] = d["n_LP"] / d["n_cells"]
    d["frac_GCsort"] = 1 - d["n_nonGC"] / d["n_cells"]
    d["frac_IgA"] = d["n_IgA"] / d["n_isotyped"].replace(0, np.nan)
    d["frac_IgG1"] = d["n_IgG1"] / d["n_isotyped"].replace(0, np.nan)
    d["state_span"] = np.select([(d.n_GC > 0) & (d.n_plasma > 0), d.n_GC > 0, d.n_plasma > 0],
                                ["GC B and plasma", "GC B only", "plasma only"], "other")
    d["tissue_span"] = ((d.n_PP + d.n_MLN) > 0) & (d.n_LP > 0)
    d["iga_igg1"] = (d.n_IgA > 0) & (d.n_IgG1 > 0)
    d["gc_history"] = np.select([d.n_nonGC == 0, d.n_nonGC == d.n_cells], ["all GC-sorted", "all nonGC"],
                                "mixed")
    t = table.reindex(d.index)
    d["x"], d["y"], d["programme"] = t["x"], t["y"], t["clone_programme"]
    d["region"] = np.where(d["programme"] == "P2", np.where(d["y"] < 3.5, "P2 lower", "P2 bridge"),
                           d["programme"])
    return d


def region_tables(d: pd.DataFrame) -> dict:
    res = {}
    order = ["P1", "P2 bridge", "P2 lower", "P3"]
    res["region_by_mouse_all"] = pd.crosstab(d["region"], d["donor"]).reindex(order).to_dict(orient="index")
    pl = d[d["frac_plasma"] >= 0.5]
    res["region_by_mouse_plasma_dominated"] = pd.crosstab(pl["region"], pl["donor"]).reindex(order) \
        .fillna(0).astype(int).to_dict(orient="index")
    # nonGC share of plasma cells per region vs a mouse-only expectation
    rate = pl.groupby("donor")[["n_plasma", "n_plasma_nonGC"]].sum()
    rate = (rate["n_plasma_nonGC"] / rate["n_plasma"]).to_dict()
    rows = {}
    for reg, x in pl.groupby("region"):
        obs = x["n_plasma_nonGC"].sum() / x["n_plasma"].sum()
        exp = (x["n_plasma"] * x["donor"].map(rate)).sum() / x["n_plasma"].sum()
        rows[reg] = {"n_clones": int(len(x)), "n_plasma_cells": int(x["n_plasma"].sum()),
                     "observed_nonGC_share": float(obs), "expected_from_mouse": float(exp)}
    res["nonGC_share_by_region"] = rows
    res["mouse_nonGC_share_of_plasma_in_plasma_dominated_clones"] = rate
    # composition of each region (clone-weighted means of cell fractions)
    comp = {}
    for reg, x in d.groupby("region"):
        comp[reg] = {"n_clones": int(len(x)), "frac_GC": float(x["frac_GC"].mean()),
                     "frac_plasma": float(x["frac_plasma"].mean()), "frac_LP": float(x["frac_LP"].mean()),
                     "frac_IgA": float(x["frac_IgA"].mean()), "frac_IgG1": float(x["frac_IgG1"].mean()),
                     "frac_GCsort": float(x["frac_GCsort"].mean()),
                     "state_span": x["state_span"].value_counts().to_dict(),
                     "tissue_span": int(x["tissue_span"].sum()), "iga_igg1": int(x["iga_igg1"].sum()),
                     "gc_history": x["gc_history"].value_counts().to_dict(),
                     "shm_mean_median": float(x["shm_mean"].median())}
    res["region_composition"] = comp
    return res


def shm_by_sort(a: ad.AnnData, d: pd.DataFrame, n_perm: int = 2000, seed: int = 7) -> dict:
    """V-region mutation frequency of plasma cells by GC-fate sort, compared within mice (clone level:
    plasma-cell-containing clones with any nonGC plasma cell versus clones whose plasma cells were all
    GC-sorted; difference of medians, permutation within mice)."""
    obs = a.obs[(a.obs["celltype"] == "plasma") & a.obs["mutation_frequency"].notna()]
    cell = obs.groupby(["donor", "gc_sort"], observed=True)["mutation_frequency"].agg(["median", "size"])
    x = d[(d["n_plasma"] > 0) & d["shm_plasma"].notna()].copy()
    x["has_nonGC_plasma"] = x["n_plasma_nonGC"] > 0
    rng = np.random.default_rng(seed)

    def diff(lab):
        vals = []
        for m in MICE:
            sel = (x["donor"] == m).to_numpy()
            yes, no = x["shm_plasma"].to_numpy()[sel & lab], x["shm_plasma"].to_numpy()[sel & ~lab]
            if len(yes) >= 3 and len(no) >= 3:
                vals.append(np.median(yes) - np.median(no))
        return float(np.mean(vals)) if vals else np.nan

    lab = x["has_nonGC_plasma"].to_numpy()
    obs_d = diff(lab)
    null = []
    for _ in range(n_perm):
        p = lab.copy()
        for m in MICE:
            w = np.flatnonzero((x["donor"] == m).to_numpy())
            p[w] = rng.permutation(lab[w])
        null.append(diff(p))
    null = np.asarray(null)
    per_mouse = {}
    for m in MICE:
        s = x[x["donor"] == m]
        yes, no = s.loc[s.has_nonGC_plasma, "shm_plasma"], s.loc[~s.has_nonGC_plasma, "shm_plasma"]
        per_mouse[m] = {"n_with_nonGC": int(len(yes)), "n_GC_only": int(len(no)),
                        "median_with_nonGC": float(yes.median()) if len(yes) else None,
                        "median_GC_only": float(no.median()) if len(no) else None,
                        "mannwhitney_p": float(stats.mannwhitneyu(yes, no).pvalue) if len(yes) >= 3 and len(no) >= 3
                        else None}
    return {"cell_level_median_by_mouse_sort": {f"{k[0]}|{k[1]}": {"median": float(v["median"]), "n": int(v["size"])}
                                                for k, v in cell.iterrows()},
            "clone_level_mean_within_mouse_median_difference": obs_d,
            "p_two_sided": float((1 + (np.abs(null) >= abs(obs_d)).sum()) / (1 + n_perm)), "n_perm": n_perm,
            "per_mouse": per_mouse}


def associations(a: ad.AnnData, d: pd.DataFrame) -> dict:
    """Profile associations (2,000 permutations within mice) for clone-level annotations not used to build
    the profiles. Needs the Figure 3 profiles to be the ones stored in ``a``."""
    obs = a.obs
    clone = obs["clone_id"].astype(object)
    obs["state_span_GCPC"] = clone.map((d["state_span"] == "GC B and plasma").astype(float))
    obs["tissue_span"] = clone.map(d["tissue_span"].astype(float))
    tests = [("gc_sort", "fraction:GC", "GC-sorted proportion"),
             ("celltype", "fraction:plasma", "Plasma-cell proportion"),
             ("celltype", "fraction:GC_Bcell", "GC B-cell proportion"),
             ("tissue", "fraction:LP", "Lamina-propria proportion"),
             ("tissue", "fraction:PP", "Peyer's-patch proportion"),
             ("isotype_fine", "fraction:IgA", "IgA proportion"),
             ("isotype_fine", "fraction:IgG1", "IgG1 proportion"),
             ("mutation_frequency", "mean", "V-region mutation frequency"),
             ("state_span_GCPC", "mean", "Clone contains GC B and plasma cells"),
             ("tissue_span", "mean", "Clone spans PP/MLN and LP")]
    out = {}
    for label, how, name in tests:
        r = tf.tl.profile_association(a, label, how=how, n_perm=2000, verbose=False)
        out[name] = {k: r[k] for k in ("r2", "null_mean", "excess_r2", "p_value", "n_clones", "strata_key")}
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = load()
    stamp(f"loaded {a.n_obs} cells; {a.obs['clone_id'].notna().sum()} with a clone")

    # mean representation: clonal ICC must match the saved run
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor", representation="mean",
                         min_cells=2, verbose=False)
    icc = a.uns["threadfin"]["profiles"]["variance_components"]["icc"]
    saved = json.loads((RES / "summary.json").read_text())
    stamp(f"ICC {icc:.6f} (saved {saved['coherence']['icc']:.6f})")

    # 1. Figure 3 profiles (centred on each mouse)
    donor = run_variant(a, "X_threadfin", "donor")
    ref = pd.read_csv(RES / "clone_table.csv").set_index("clone_id")
    t = donor["table"].join(ref[["x", "y", "clone_programme"]], rsuffix="_saved")
    check = {"n_clones": int(len(t)), "max_abs_dx": float((t.x - t.x_saved).abs().max()),
             "max_abs_dy": float((t.y - t.y_saved).abs().max()),
             "programme_agreement": float((t.clone_programme == t.clone_programme_saved).mean()),
             "icc_mean_profiles": float(icc), "icc_saved": saved["coherence"]["icc"]}
    stamp(f"reproduction: {check}")
    # use the saved coordinates and programmes of Figure 3 for every downstream summary
    table3 = ref.copy()
    feats_donor = donor["features"]
    d = clone_descriptors(a, table3)
    res = {"reproduction": check, "programmes_donor": donor["programmes"]}

    # clone-level associations on the Figure 3 profiles (stored in a.uns right now)
    res["associations"] = associations(a, d)
    stamp("associations done")

    # 2. centring variants
    none = run_variant(a, "X_threadfin", None)
    res["programmes_none"] = none["programmes"]
    stamp(f"no centring: {none['programmes']}")
    res["balanced_shift_norms"] = balanced_basis(a)
    bal = run_variant(a, "X_balanced", None)
    res["programmes_balanced"] = bal["programmes"]
    stamp(f"balanced: {bal['programmes']}")

    # programme labels of the variants per clone, and their coordinates
    for name, v in (("none", none), ("balanced", bal)):
        vt = v["table"].reindex(d.index)
        d[f"x_{name}"], d[f"y_{name}"] = vt["x"], vt["y"]
        d[f"programme_{name}"] = vt["clone_programme"]
        res[f"programme_{name}_by_fig3_region"] = pd.crosstab(d["region"], d[f"programme_{name}"]).to_dict(
            orient="index")
        res[f"programme_{name}_by_mouse"] = pd.crosstab(d[f"programme_{name}"], d["donor"]).to_dict(
            orient="index")

    # donor composition and region tables (Figure 3 run)
    obs = a.obs
    res["mouse_composition"] = {
        "celltype": pd.crosstab(obs["donor"], obs["celltype"]).to_dict(orient="index"),
        "tissue": pd.crosstab(obs["donor"], obs["tissue"]).to_dict(orient="index"),
        "gc_sort": pd.crosstab(obs["donor"], obs["gc_sort"]).to_dict(orient="index")}
    res.update(region_tables(d))

    # mouse mixing: plasma cells (cell embedding and cell UMAP) versus plasma-cell-dominated clones
    pcs = obs[obs["celltype"] == "plasma"]
    pidx = np.flatnonzero((obs["celltype"] == "plasma").to_numpy())
    mouse_c = pcs["donor"].astype(str).to_numpy()
    clone_c = pcs["clone_id"].astype(object).to_numpy()
    res["mixing_cells_embedding"] = knn_same_mouse(np.asarray(a.obsm["X_threadfin"])[pidx], mouse_c,
                                                   clone=clone_c)
    res["mixing_cells_umap"] = knn_same_mouse(pcs[["cell_umap_1", "cell_umap_2"]].to_numpy(), mouse_c,
                                              clone=clone_c)
    pl = d[d["frac_plasma"] >= 0.5]
    mouse_k = pl["donor"].astype(str).to_numpy()
    for name, feats in (("donor", feats_donor), ("none", none["features"]), ("balanced", bal["features"])):
        res[f"mixing_clones_{name}"] = knn_same_mouse(feats.loc[pl.index].to_numpy(), mouse_k)
    res["mixing_clones_donor_umap"] = knn_same_mouse(pl[["x", "y"]].to_numpy(), mouse_k)
    stamp("mixing done")

    # within-mouse GC-history neighbourhoods, for each centring
    res["within_mouse_nonGC"] = {name: within_mouse_neighbourhood(feats, d) for name, feats in
                                 (("donor", feats_donor), ("none", none["features"]),
                                  ("balanced", bal["features"]))}
    stamp("within-mouse tests done")

    # mutation frequency of plasma cells by GC-fate sort
    res["shm_by_sort"] = shm_by_sort(a, d)

    d.to_csv(OUT / "clone_descriptors.csv")
    feats_donor.to_csv(OUT / "features_donor.csv.gz")
    none["features"].to_csv(OUT / "features_none.csv.gz")
    bal["features"].to_csv(OUT / "features_balanced.csv.gz")
    obs[["donor", "sample", "tissue", "gc_sort", "celltype", "clone_id", "isotype_fine", "mutation_frequency",
         "cell_umap_1", "cell_umap_2", "fig3_programme"]].to_csv(OUT / "cells.csv.gz")
    (OUT / "summary.json").write_text(json.dumps(jsonable(res), indent=1))
    stamp(f"wrote {OUT}")


if __name__ == "__main__":
    main()
