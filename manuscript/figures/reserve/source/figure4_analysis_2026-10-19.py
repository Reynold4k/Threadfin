#!/usr/bin/env python
"""Figure 4 analyses: division history, antigen binding and zone in the RBD germinal-centre cohorts.

GSE287123 (Merkenschlager et al. 2025): H2B-mCherry division-reporter mice immunized with RBD protein
(five mice, sorted RBD-bait+/- x mCherry-high/low) or RBD mRNA (five mice, sorted light/dark zone x
mCherry-high/low). The four gates of each cohort were sequenced in four libraries (tubes A-D) shared by the
two cohorts, with the division gates crossed: recently divided (mCherry-low) cells were sequenced in
libraries A and C for protein-immunized mice and in libraries B and D for mRNA-immunized mice.

1. Reproduce the saved clone profiles (kernel, centred on each mouse) and clone map from the cached
   embedding and check them against the saved clone table.
2. Orient the first principal component of the clone profiles (PC1) and describe it with the clone-level
   RNA module scores of the main run.
3. Clone-level tests of the sorted annotations: profile association (Threadfin) for the division gate in
   each cohort separately, the RBD bait (protein cohort) and the zone gate (mRNA cohort).
4. Library crossover: a difference between libraries would associate the profiles with recent division in
   opposite directions in the two cohorts. Compare the division-associated profile directions (cosine
   similarity, permutations within mice) and the within-mouse slopes of PC1 on the recently divided fraction.
5. Within-mouse Spearman correlations and decile summaries along PC1 for the figure.

Nothing here changes the saved case-study run. Outputs: reserve/figure4_rbd_2026-10-19/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

import threadfin as tf

ROOT = Path(__file__).resolve().parents[4]                       # Threadfin/
CACHE = ROOT.parent / "internal_validation" / "algorithm_revision" / "embedding_cache" / "mouse_rbd_embedding.h5ad"
RES = ROOT / "manuscript" / "case_studies" / "results" / "mouse_rbd"
OUT = ROOT / "manuscript" / "figures" / "reserve" / "figure4_rbd_2026-10-19"
MAP = dict(n_neighbors=30, min_dist=0.8)                           # case-study display settings for mouse_rbd
COHORTS = {"RBD protein": "protein", "mRNA": "mRNA"}
MODULES = ["dark zone / cycling", "light zone", "memory", "naive", "plasma cell"]
N_PERM, N_BOOT, SEED = 2000, 2000, 0
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


# --------------------------------------------------------------------------- data and profiles


def load() -> ad.AnnData:
    a = ad.read_h5ad(CACHE)
    cells = pd.read_csv(RES / "cells.csv.gz", index_col=0)
    assert (cells.index == a.obs_names).all(), "cell order differs from the case-study run"
    # clone labels exactly as in the case-study run (cells with a productive heavy chain and a paired call)
    a.obs["clone_id"] = cells["clone_id"].values
    for col in ("isotype", "mutation_frequency", "umap_1", "umap_2"):
        a.obs[col] = cells[col].values
    for col in a.obs.columns:
        if isinstance(a.obs[col].dtype, pd.CategoricalDtype):
            a.obs[col] = a.obs[col].astype(object)
    a.obs["tube"] = a.obs["library"].astype(str).str[-1]
    return a


def profile(a: ad.AnnData) -> dict:
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", min_cells=2, verbose=False)
    tf.tl.embed_clones(a, min_reliability=0.0, verbose=False, **MAP)
    prof = a.uns["threadfin"]["profiles"]
    saved = pd.read_csv(RES / "clone_table.csv", index_col=0)
    table = prof["clone_table"]
    assert table.index.sort_values().equals(saved.index.sort_values())
    diffs = {c: float(np.abs(table.loc[saved.index, c].astype(float) - saved[c].astype(float)).max())
             for c in ("n_cells", "reliability", "x", "y")}
    assert max(diffs.values()) < 1e-8, diffs
    stamp(f"profiles reproduced: {len(table)} clones; max |difference| {diffs}")
    return {"table": table.copy(), "features": prof["features"].copy(), "check": diffs}


def clone_table(a: ad.AnnData, prof: dict) -> pd.DataFrame:
    obs = a.obs.dropna(subset=["clone_id"])
    obs = obs[obs["clone_id"].isin(prof["table"].index)]
    g = obs.groupby("clone_id")
    assert int(g["arm"].nunique().max()) == 1 and int(g["donor"].nunique().max()) == 1

    def frac(col, level):
        return g[col].apply(lambda s: float((s == level).sum() / s.notna().sum()) if s.notna().any() else np.nan)

    t = pd.DataFrame({
        "n_cells": g.size(), "donor": g["donor"].first(), "cohort": g["arm"].first().map(COHORTS),
        "recently_divided": frac("division_gate", "mCherry-low"),
        "rbd_binding": frac("rbd_bait", "RBD+"),
        "dark_zone": frac("zone_gate", "DZ"),
        "tubes_A_C": g["tube"].apply(lambda s: float(s.isin(["A", "C"]).mean())),
        "igm_igd": g["isotype"].apply(lambda s: float(s.isin(["IGHM", "IGHD"]).sum() / s.notna().sum())
                                      if s.notna().any() else np.nan),
        "mutation_frequency": g["mutation_frequency"].mean(),
    })
    t = t.join(prof["table"][["reliability", "x", "y"]])
    scores = pd.read_csv(RES / "clone_gene_scores.csv", index_col=0)
    t = t.join(scores[MODULES])
    # the division gate is fixed by the tubes, in opposite directions in the two cohorts
    prot, mrna = t["cohort"].eq("protein"), t["cohort"].eq("mRNA")
    assert np.allclose(t.loc[prot, "recently_divided"], t.loc[prot, "tubes_A_C"])
    assert np.allclose(t.loc[mrna, "recently_divided"], 1 - t.loc[mrna, "tubes_A_C"])
    return t


def principal_axes(features: pd.DataFrame, t: pd.DataFrame) -> dict:
    x = features.loc[t.index].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:3].T
    # orientation: PC1 increases with the clone-level dark-zone/cycling module score
    sign = np.sign(stats.spearmanr(pcs[:, 0], t["dark zone / cycling"]).statistic)
    t["pc1"] = sign * pcs[:, 0]
    t["pc2"], t["pc3"] = pcs[:, 1], pcs[:, 2]
    t["pc1_decile"] = pd.qcut(t["pc1"], 10, labels=False) + 1
    var = s ** 2 / np.sum(s ** 2)
    return {"variance_explained": var[:5].tolist(), "orientation_sign": float(sign),
            "spearman_with_modules": {m: float(stats.spearmanr(t["pc1"], t[m]).statistic) for m in MODULES},
            "spearman_with_clone_map": {c: float(stats.spearmanr(t["pc1"], t[c]).statistic) for c in ("x", "y")},
            "spearman_with_log_cells": float(stats.spearmanr(t["pc1"], np.log(t["n_cells"])).statistic)}


# --------------------------------------------------------------------------- statistics


def permute_within(v: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    out = v.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = rng.permutation(v[idx])
    return out


def direction(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Least-squares shift of the clone profiles per unit of the annotation (profiles centred)."""
    x = x - x.mean(axis=0)
    v = v - v.mean()
    return x.T @ v / (v @ v)


def cosine(u: np.ndarray, w: np.ndarray) -> float:
    return float(u @ w / np.linalg.norm(u) / np.linalg.norm(w))


def associations(a: ad.AnnData) -> dict:
    """Threadfin profile association, as in the case-study run, per cohort where the design requires it."""
    obs = a.obs
    obs["division_gate_protein"] = obs["division_gate"].where(obs["arm"].eq("RBD protein"))
    obs["division_gate_mrna"] = obs["division_gate"].where(obs["arm"].eq("mRNA"))
    tests = {"recently_divided_all": ("division_gate", "fraction:mCherry-low"),
             "recently_divided_protein": ("division_gate_protein", "fraction:mCherry-low"),
             "recently_divided_mrna": ("division_gate_mrna", "fraction:mCherry-low"),
             "rbd_binding_protein": ("rbd_bait", "fraction:RBD+"),
             "dark_zone_mrna": ("zone_gate", "fraction:DZ")}
    out = {}
    for key, (label, how) in tests.items():
        r = tf.tl.profile_association(a, label, how=how, n_perm=N_PERM, random_state=SEED, verbose=False)
        out[key] = {k: r[k] for k in ("r2", "null_mean", "excess_r2", "p_value", "n_clones", "n_perm", "strata_key")}
        stamp(f"association {key}: R2 {r['r2']:.4f} vs {r['null_mean']:.4f}, P {r['p_value']:.4g}, "
              f"{r['n_clones']} clones")
    return out


def crossover(features: pd.DataFrame, t: pd.DataFrame) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = {c: features.loc[t.index[t["cohort"].eq(c)]].to_numpy(dtype=float) for c in ("protein", "mRNA")}
    v = {c: t.loc[t["cohort"].eq(c), "recently_divided"].to_numpy(dtype=float) for c in ("protein", "mRNA")}
    d = {c: t.loc[t["cohort"].eq(c), "donor"].to_numpy() for c in ("protein", "mRNA")}
    w = {c: direction(x[c], v[c]) for c in x}
    observed = cosine(w["protein"], w["mRNA"])
    null = np.array([cosine(direction(x["protein"], permute_within(v["protein"], d["protein"], rng)),
                            direction(x["mRNA"], permute_within(v["mRNA"], d["mRNA"], rng)))
                     for _ in range(N_PERM)])
    p = float((1 + np.sum(null >= observed)) / (N_PERM + 1))
    # within-mouse slopes of PC1 on the recently divided fraction; with the tubes crossed, the division
    # effect is the mean of the two slopes and the tube (A,C versus B,D) effect is half their difference
    slopes = {}
    for c in ("protein", "mRNA"):
        sub = t[t["cohort"].eq(c)]
        xc = sub["recently_divided"] - sub.groupby("donor")["recently_divided"].transform("mean")
        yc = sub["pc1"] - sub.groupby("donor")["pc1"].transform("mean")
        slopes[c] = float(np.sum(xc * yc) / np.sum(xc * xc))
    effect = (slopes["protein"] + slopes["mRNA"]) / 2
    tube = (slopes["protein"] - slopes["mRNA"]) / 2
    res = {"cosine": observed, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, .95)),
           "null_q99": float(np.quantile(null, .99)), "null_min": float(null.min()), "null_max": float(null.max()),
           "p_value": p, "n_perm": N_PERM, "permutation": "recently divided fraction among clones within each mouse",
           "expected_if_tube_effect_only": -1.0,
           "pc1_slopes_within_mouse": slopes, "pc1_division_effect": effect, "pc1_tube_AC_vs_BD_effect": tube,
           "tube_effect_relative": tube / effect}
    stamp(f"crossover: cosine {observed:.3f} vs null mean {null.mean():.3f} (q95 {res['null_q95']:.3f}), P {p:.4g}; "
          f"slopes {slopes}; tube/division {tube / effect:.3f}")
    return res, null


def correlations(t: pd.DataFrame) -> dict:
    out = {"cohort": {}, "mouse": {}}
    for label, cohorts in (("recently_divided", ("protein", "mRNA")), ("rbd_binding", ("protein",)),
                           ("dark_zone", ("mRNA",))):
        for c in cohorts:
            sub = t[t["cohort"].eq(c) & t[label].notna()]
            r = stats.spearmanr(sub[label], sub["pc1"])
            out["cohort"][f"{label}|{c}"] = {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": int(len(sub))}
            for donor, s in sub.groupby("donor"):
                r = stats.spearmanr(s[label], s["pc1"])
                out["mouse"][f"{label}|{donor}"] = {"cohort": c, "donor": donor, "rho": float(r.statistic),
                                                    "p_value": float(r.pvalue), "n": int(len(s))}
    # partial correlation with clone size (log cells) removed from both ranks
    for c in ("protein", "mRNA"):
        sub = t[t["cohort"].eq(c)]
        rx, ry, rz = (stats.rankdata(sub[k]) for k in ("recently_divided", "pc1", "n_cells"))
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        out["cohort"][f"recently_divided|{c}"]["partial_rho_given_size"] = float(np.corrcoef(ex, ey)[0, 1])
    return out


def deciles(t: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    specs = [("recently_divided", "protein"), ("recently_divided", "mRNA"), ("rbd_binding", "protein"),
             ("dark_zone", "mRNA"), *[(m, "all") for m in MODULES]]
    for label, c in specs:
        sub = t if c == "all" else t[t["cohort"].eq(c)]
        sub = sub[sub[label].notna()]
        for dec, s in sub.groupby("pc1_decile"):
            vals = s[label].to_numpy(dtype=float)
            boot = np.array([rng.choice(vals, vals.size).mean() for _ in range(N_BOOT)])
            rows.append({"annotation": label, "cohort": c, "pc1_decile": int(dec), "n_clones": int(vals.size),
                         "mean": float(vals.mean()), "ci_low": float(np.quantile(boot, .025)),
                         "ci_high": float(np.quantile(boot, .975)), "median_pc1": float(s["pc1"].median())})
    return pd.DataFrame(rows)


def cell_level_contrasts(a: ad.AnnData) -> dict:
    """Library contrasts of mouse-centred cell coordinates in each cohort (descriptive record)."""
    x = np.asarray(a.obsm["X_threadfin"], dtype=float).copy()
    donor = a.obs["donor"].astype(str).to_numpy()
    for d in np.unique(donor):
        x[donor == d] -= x[donor == d].mean(axis=0)
    tube, arm = a.obs["tube"].to_numpy(), a.obs["arm"].to_numpy()
    out, vec = {}, {}
    for c in ("RBD protein", "mRNA"):
        mu = {k: x[(arm == c) & (tube == k)].mean(axis=0) for k in "ABCD"}
        vec[c] = {"A,C vs B,D": (mu["A"] + mu["C"] - mu["B"] - mu["D"]) / 2,
                  "C,D vs A,B": (mu["C"] + mu["D"] - mu["A"] - mu["B"]) / 2,
                  "A,D vs B,C": (mu["A"] + mu["D"] - mu["B"] - mu["C"]) / 2}
    for k in vec["mRNA"]:
        u, w = vec["RBD protein"][k], vec["mRNA"][k]
        out[k] = {"norm_protein": float(np.linalg.norm(u)), "norm_mrna": float(np.linalg.norm(w)),
                  "cosine": cosine(u, w)}
    u, w = vec["RBD protein"]["A,C vs B,D"], vec["mRNA"]["A,C vs B,D"]
    out["division_decomposition"] = {"division_norm": float(np.linalg.norm((u - w) / 2)),
                                     "tube_AC_vs_BD_norm": float(np.linalg.norm((u + w) / 2))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = load()
    prof = profile(a)
    t = clone_table(a, prof)
    axes = principal_axes(prof["features"], t)
    stamp(f"PC1: {axes['variance_explained'][0]:.3f} of profile variance; modules {axes['spearman_with_modules']}")
    assoc = associations(a)
    cross, null = crossover(prof["features"], t)
    corr = correlations(t)
    dec = deciles(t)
    cells = cell_level_contrasts(a)
    design = (a.obs.groupby(["arm", "tube"])["sample"].apply(lambda s: sorted(set(x.split("_", 1)[1] for x in s)))
              .unstack().to_dict())
    summary = {
        "dataset": "GSE287123 RBD cohorts (Merkenschlager et al. 2025)",
        "reproduction_max_abs_difference": prof["check"],
        "n_cells": int(a.n_obs), "n_cells_in_profiled_clones": int(t["n_cells"].sum()),
        "n_clones": int(len(t)), "n_clones_by_cohort": t["cohort"].value_counts().to_dict(),
        "n_mice_by_cohort": t.groupby("cohort")["donor"].nunique().to_dict(),
        "cells_by_cohort_and_gate": a.obs.groupby(["arm", "division_gate"]).size().unstack().to_dict(),
        "tube_design": design,
        "pc1": axes, "associations": assoc, "crossover": cross, "correlations": corr,
        "clones_without_recently_divided_cell_by_decile": t.groupby("pc1_decile")["recently_divided"]
        .apply(lambda s: float((s == 0).mean())).to_dict(),
        "igm_igd_by_decile": t.groupby("pc1_decile")["igm_igd"].mean().to_dict(),
        "cell_level_tube_contrasts": cells,
        "seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
        "threadfin_version": tf.__version__ if hasattr(tf, "__version__") else None,
    }
    t.to_csv(OUT / "clones.csv")
    dec.to_csv(OUT / "pc1_deciles.csv", index=False)
    np.save(OUT / "crossover_null.npy", null)
    a.obs[["clone_id", "donor", "arm", "tube", "division_gate", "rbd_bait", "zone_gate", "umap_1", "umap_2"]] \
        .to_csv(OUT / "cells.csv.gz")
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    stamp(f"wrote {OUT}")


if __name__ == "__main__":
    main()
