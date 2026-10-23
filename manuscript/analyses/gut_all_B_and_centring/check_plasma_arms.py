#!/usr/bin/env python
"""Are the two gut plasma arms of Figure 3h, and the nonGC enrichment of the upper arm, a mouse effect?

Uses the saved run behind the current Figure 3 g-j (case_studies/results/gut_gc_reporter) for the arm
geometry, and the all-B no-centring run (allB_threadfin/none) for the within-mouse profile test.
Output: allB_threadfin/story/plasma_arms_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIG = ROOT / "case_studies" / "results" / "gut_gc_reporter"
ALLB = HERE / "allB_threadfin"
MICE = ["mouse2", "mouse3", "mouse4"]
RNG = np.random.default_rng(20261016)


def pc_counts(cells: pd.DataFrame) -> pd.DataFrame:
    pc = cells[cells["celltype"] == "plasma"]
    return pc.groupby("clone_id").agg(n_PC=("gc_sort", "size"),
                                      n_PC_non=("gc_sort", lambda s: int((s == "nonGC").sum())))


def figure_arms() -> tuple[dict, pd.DataFrame]:
    tab = pd.read_csv(ORIG / "clone_table.csv")
    cells = pd.read_csv(ORIG / "cells.csv.gz")
    cells["gc_sort"] = cells["sample"].str.split("_").str[-1]
    tab = tab.merge(pc_counts(cells), left_on="clone_id", right_index=True, how="left")
    tab[["n_PC", "n_PC_non"]] = tab[["n_PC", "n_PC_non"]].fillna(0)
    pl = tab[(tab["celltype:plasma"] >= 0.5) & tab["clone_programme"].isin(["P2", "P3"])].copy()
    pl["region"] = np.where(pl["clone_programme"] == "P3", "upper arm (P3)",
                            np.where(pl["y"] < 3.5, "lower arm (P2)", "bridge (P2)"))

    out = {"n_plasma_clones": int(len(pl))}
    out["region_by_mouse"] = pd.crosstab(pl["region"], pl["donor"]).to_dict(orient="index")

    pcs = cells[cells["celltype"] == "plasma"]
    out["pc_nonGC_share_by_mouse_all_pc_cells"] = (
        pcs.groupby("donor")["gc_sort"].apply(lambda s: round(float((s == "nonGC").mean()), 4)).to_dict())
    # mouse rate among PCs inside the 431 plasma clones; expected arm share if only the mouse mattered
    rate = pl.groupby("donor")[["n_PC", "n_PC_non"]].sum()
    rate = (rate["n_PC_non"] / rate["n_PC"]).to_dict()
    out["pc_nonGC_share_by_mouse_plasma_clones"] = {k: round(v, 4) for k, v in rate.items()}
    rows = {}
    for reg, d in pl.groupby("region"):
        obs = d["n_PC_non"].sum() / d["n_PC"].sum()
        exp = (d["n_PC"] * d["donor"].map(rate)).sum() / d["n_PC"].sum()
        rows[reg] = {"n_clones": int(len(d)), "n_pc": int(d["n_PC"].sum()),
                     "observed_nonGC_share": round(float(obs), 4),
                     "expected_from_mouse_only": round(float(exp), 4),
                     "clones_gc_sort_below_0.5": int((d["gc_sort:GC"] < 0.5).sum()),
                     "median_gc_sort": float(d["gc_sort:GC"].median())}
    out["region_nonGC"] = rows
    out["low_gc_clones_by_mouse"] = pl[pl["gc_sort:GC"] < 0.5]["donor"].value_counts().to_dict()

    # within mouse: upper arm vs bridge, PCs only (Fisher on pooled PC cells; Mann-Whitney on clones)
    within = {}
    for m in ("mouse2", "mouse4"):
        d = pl[(pl["donor"] == m) & pl["region"].isin(["upper arm (P3)", "bridge (P2)"]) & (pl["n_PC"] > 0)]
        a, b = d[d["region"] == "upper arm (P3)"], d[d["region"] == "bridge (P2)"]
        t = [[int(a["n_PC_non"].sum()), int(a["n_PC"].sum() - a["n_PC_non"].sum())],
             [int(b["n_PC_non"].sum()), int(b["n_PC"].sum() - b["n_PC_non"].sum())]]
        fa, fb = a["n_PC_non"] / a["n_PC"], b["n_PC_non"] / b["n_PC"]
        within[m] = {"upper_n": int(len(a)), "bridge_n": int(len(b)),
                     "upper_pc_nonGC": round(t[0][0] / max(sum(t[0]), 1), 4),
                     "bridge_pc_nonGC": round(t[1][0] / max(sum(t[1]), 1), 4),
                     "fisher_p": float(stats.fisher_exact(t)[1]),
                     "mannwhitney_p": float(stats.mannwhitneyu(fa, fb).pvalue)}
    out["within_mouse_upper_vs_bridge"] = within
    return out, pl


def knn_stat(F: np.ndarray, share: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(F)
    idx = nn.kneighbors(F, return_distance=False)[:, 1:]
    return float(stats.spearmanr(share, share[idx].mean(axis=1)).statistic)


def within_mouse_profile_test(k: int = 10, n_perm: int = 2000) -> dict:
    """No centring: do nonGC-rich plasma clones sit together in profile space, within one mouse?"""
    tab = pd.read_csv(ALLB / "none" / "clone_table.csv").set_index("clone_id")
    feat = pd.read_csv(ALLB / "none" / "clone_features.csv.gz").set_index("clone_id")
    kcols = [c for c in feat.columns if c.startswith("kpc")]
    pl = tab[(tab["frac_PC"] >= 0.5) & (tab["n_PC"] > 0)].copy()
    pl["share"] = pl["n_PC_tdneg"] / pl["n_PC"]
    pl["lp_dominant"] = pl["frac_LP"] >= 0.5
    res = {}
    for m in ("mouse2", "mouse4"):
        d = pl[pl["donor"] == m]
        F, s = feat.loc[d.index, kcols].to_numpy(), d["share"].to_numpy()
        obs = knn_stat(F, s, k)
        strata = d["lp_dominant"].to_numpy()
        null_any, null_lp = [], []
        for _ in range(n_perm):
            null_any.append(knn_stat(F, RNG.permutation(s), k))
            p = s.copy()
            for g in (True, False):
                w = np.where(strata == g)[0]
                p[w] = RNG.permutation(s[w])
            null_lp.append(knn_stat(F, p, k))
        null_any, null_lp = np.array(null_any), np.array(null_lp)
        res[m] = {"n_clones": int(len(d)), "n_with_nonGC_pc": int((d["n_PC_tdneg"] > 0).sum()),
                  "observed_rho": round(obs, 4),
                  "null_mean": round(float(null_any.mean()), 4),
                  "p_perm_within_mouse": float((1 + (null_any >= obs).sum()) / (n_perm + 1)),
                  "null_mean_tissue_strata": round(float(null_lp.mean()), 4),
                  "p_perm_within_mouse_and_LP_stratum": float((1 + (null_lp >= obs).sum()) / (n_perm + 1))}
    return {"k": k, "n_perm": n_perm, "features": "kernel PCs, no context centring", "by_mouse": res}


def main():
    arms, _ = figure_arms()
    prof = within_mouse_profile_test()
    out = {"figure_3_run": arms, "no_centring_within_mouse": prof}
    (ALLB / "story").mkdir(exist_ok=True)
    (ALLB / "story" / "plasma_arms_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
