#!/usr/bin/env python
"""Mouse germinal-centre deep dive (Merkenschlager et al. 2025, Nature; GSE287123).

Questions
---------
1. Is the high-division programme a library artefact? The two division gates
   (mCherry-high / mCherry-low) were sequenced as separate libraries, so
   profiles are re-estimated relative to each mouse x gate library and the
   programmes re-tested against each clone's division fraction.
2. Within each library separately, do clones whose cells carry a higher
   ribosome/translation score have divided more?
3. Sanity checks of the data against the paper's biology: do clones carrying
   the high-affinity VH186.2 W33L mutation (NP arm), or binding RBD bait (RBD
   arm), have divided more? In the RBD experiment every sort gate is a
   separate library, so gate labels are also re-tested against profiles
   centred within each library.
4. Which programmes are clonally heritable (translation, light zone, dark
   zone / cycling, plasma cell)? Gene-set clonal ICCs from the main pipeline.

Usage: python mouse_gc_deep_dive.py   (writes results/mouse_gc_deep_dive/)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")
import threadfin as tf  # noqa: E402
from run_case_study import HERE, prepare  # noqa: E402

OUT = HERE / "results" / "mouse_gc_deep_dive"
OUT.mkdir(parents=True, exist_ok=True)


def ribosome_genes(var_names):
    names = pd.Index(var_names)
    rp = names[names.str.match(r"^Rp[sl]\d+[a-z]?\d*$") & ~names.str.contains("-ps")]
    return list(rp) + [g for g in ("Eef1a1", "Tpt1", "Fau", "Eef1b2", "Eef1g", "Npm1") if g in names]


def cell_score(adata, genes):
    x = adata[:, genes].layers["log_norm"]
    return np.asarray(x.mean(axis=1)).ravel()


def clone_table(adata, score, gate_level, gate_col="division_gate", min_cells=5, within=None):
    """Per clone: mean library-centred score, fraction of cells in ``gate_level``, donor."""
    obs = adata.obs
    df = pd.DataFrame({"clone": obs["clone_id"], "donor": obs["donor"].astype(str),
                       "sample": obs["sample"].astype(str), "gate": obs[gate_col].astype(str),
                       "score": score}).dropna(subset=["clone"])
    df["score_c"] = df["score"] - df.groupby("sample")["score"].transform("mean")  # centre within library
    frac = df.groupby("clone")["gate"].apply(lambda s: float((s == gate_level).mean()))
    n = df.groupby("clone").size()
    sc_df = df if within is None else df[df["gate"] == within]
    score_clone = sc_df.groupby("clone")["score_c"].mean()
    n_within = sc_df.groupby("clone").size()
    out = pd.DataFrame({"frac": frac, "n": n, "score": score_clone, "n_scored": n_within,
                        "donor": df.groupby("clone")["donor"].first()}).dropna()
    return out[(out["n"] >= min_cells) & (out["n_scored"] >= 2)]


def perm_spearman(tab, x, y, n_perm=2000, seed=0):
    """Spearman rho with a permutation p-value that shuffles y among clones within donors."""
    rng = np.random.default_rng(seed)
    rho = spearmanr(tab[x], tab[y]).statistic
    yv = tab[y].to_numpy().copy()
    donors = tab["donor"].to_numpy()
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = yv.copy()
        for d in np.unique(donors):
            idx = np.flatnonzero(donors == d)
            perm[idx] = yv[rng.permutation(idx)]
        null[b] = spearmanr(tab[x], perm).statistic
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (n_perm + 1)
    return float(rho), float(p), int(len(tab))


def main():
    t0 = time.time()
    res = {}

    def stamp(msg):
        print(f"[mouse deep dive {time.time() - t0:6.0f}s] {msg}", flush=True)

    # ---------------------------------------------------------------- NP arm
    adata, cfg, state_key, _ = prepare("mouse_np", stamp=stamp)
    rib = ribosome_genes(adata.var_names)
    score = cell_score(adata, rib)
    res["n_ribosome_genes"] = len(rib)

    # 1) library-confound check: profiles relative to each mouse x gate library
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key="sample", donor_key="donor",
                         representation="kernel", verbose=True)
    r = tf.tl.profile_association(adata, "division_gate", how="fraction:mCherry-low", n_perm=5000)
    res["np_within_library_profile_vs_division"] = {k: r[k] for k in ("r2", "null_mean", "excess_r2", "p_value",
                                                                      "n_clones", "axis_correlation")}
    try:
        prog = tf.tl.find_programmes(adata, n_boot=50, verbose=True)
        assoc = tf.tl.association_test(adata, "division_gate", how="fraction:mCherry-low", n_perm=5000)
        assoc.to_csv(OUT / "np_programmes_within_library_vs_division.csv", index=False)
        res["np_within_library_programmes"] = {
            "n_programmes": int(prog.shape[0]), "stability": prog["stability"].round(3).to_dict(),
            "kruskal_p": assoc.attrs.get("kruskal_pvalue"),
            "medians": assoc.set_index("programme")["median_in_programme"].round(3).to_dict(),
        }
        markers = tf.tl.programme_markers(adata, layer="log_norm", context_key="sample", n_top=30)
        markers.to_csv(OUT / "np_programmes_within_library_markers.csv", index=False)
        res["np_within_library_markers"] = {p: g["gene"].head(10).tolist() for p, g in markers.groupby("programme")}
    except ValueError as e:
        res["np_within_library_programmes"] = {"error": str(e)}

    # 2) within-library clone-level correlation: ribosome score vs division history
    for within in (None, "mCherry-high", "mCherry-low"):
        tab = clone_table(adata, score, "mCherry-low", within=within)
        rho, p, n = perm_spearman(tab, "score", "frac")
        key = f"np_ribosome_vs_division[{within or 'all cells'}]"
        res[key] = {"spearman_rho": rho, "p_within_donor_perm": p, "n_clones": n}
        tab.to_csv(OUT / f"np_ribosome_vs_division_{(within or 'all').replace('-', '_')}.csv")
        stamp(f"{key}: rho={rho:.3f} p={p:.3g} n={n}")

    # 3a) W33L clones divide more? (IGHV1-72 clones only)
    obs = adata.obs.dropna(subset=["clone_id"])
    w = obs.dropna(subset=["w33l"]).groupby("clone_id")["w33l"].agg(lambda s: s.value_counts().index[0])
    div = obs.groupby("clone_id")["division_gate"].apply(lambda s: float((s == "mCherry-low").mean()))
    n = obs.groupby("clone_id").size()
    wt = pd.DataFrame({"w33l": w, "frac": div.reindex(w.index), "n": n.reindex(w.index)})
    wt = wt[wt["n"] >= 3]
    a, b = wt.loc[wt["w33l"] == "W33L", "frac"], wt.loc[wt["w33l"] == "germline W33", "frac"]
    res["np_w33l_vs_division"] = {
        "n_w33l_clones": int(a.size), "n_germline_clones": int(b.size),
        "median_frac_low_w33l": float(a.median()) if a.size else None,
        "median_frac_low_germline": float(b.median()) if b.size else None,
        "mannwhitney_p_greater": float(mannwhitneyu(a, b, alternative="greater").pvalue) if a.size and b.size else None,
    }
    wt.to_csv(OUT / "np_w33l_vs_division.csv")
    stamp(f"W33L vs division: {res['np_w33l_vs_division']}")

    # ---------------------------------------------------------------- RBD experiment
    adata, cfg, state_key, _ = prepare("mouse_rbd", stamp=stamp)

    # 0) every sort gate is its own library: re-test gate labels against profiles centred per library,
    #    separately in each arm, so library (technical) effects cannot create the association
    for arm, labels in (("RBD protein", (("division_gate", "fraction:mCherry-low"), ("rbd_bait", "fraction:RBD+"))),
                        ("mRNA", (("division_gate", "fraction:mCherry-low"), ("zone_gate", "fraction:DZ")))):
        sub = adata[adata.obs["arm"] == arm].copy()
        tf.tl.clone_profiles(sub, basis="X_threadfin", context_key="sample", donor_key="donor",
                             representation="kernel", verbose=True)
        for label, how in labels:
            try:
                r = tf.tl.profile_association(sub, label, how=how, n_perm=5000)
            except ValueError as e:
                res[f"rbd_within_library_profile[{arm}][{label}]"] = {"error": str(e)}
                continue
            res[f"rbd_within_library_profile[{arm}][{label}]"] = {
                k: r[k] for k in ("r2", "null_mean", "excess_r2", "p_value", "n_clones", "axis_correlation")}
            stamp(f"RBD {arm} within-library {label}: R2 {r['r2']:.3f} vs {r['null_mean']:.3f}, p {r['p_value']:.3g}")

    prot = adata[adata.obs["arm"] == "RBD protein"].copy()
    obs = prot.obs.dropna(subset=["clone_id"])
    rbd = obs.groupby("clone_id")["rbd_bait"].apply(lambda s: float((s == "RBD+").mean()))
    div = obs.groupby("clone_id")["division_gate"].apply(lambda s: float((s == "mCherry-low").mean()))
    n = obs.groupby("clone_id").size()
    donor = obs.groupby("clone_id")["donor"].first()
    tab = pd.DataFrame({"rbd": rbd, "frac": div, "n": n, "donor": donor.astype(str)})
    tab = tab[tab["n"] >= 5]
    rho, p, nn = perm_spearman(tab, "rbd", "frac")
    # the four gates were sorted in unequal numbers, which by itself anti-correlates the two
    # clone-level fractions; shuffle division gates within mouse x bait to keep those quotas
    cells = obs[["clone_id", "donor", "rbd_bait", "division_gate"]].copy()
    rng = np.random.default_rng(0)

    def fractions_rho(df):
        g = df.groupby("clone_id")
        t = pd.DataFrame({"r": g["rbd_bait"].apply(lambda s: (s == "RBD+").mean()),
                          "d": g["division_gate"].apply(lambda s: (s == "mCherry-low").mean()), "n": g.size()})
        t = t[t["n"] >= 5]
        return spearmanr(t["r"], t["d"]).statistic

    null = []
    for _ in range(500):
        d = cells.copy()
        for _, idx in d.groupby(["donor", "rbd_bait"]).groups.items():
            d.loc[idx, "division_gate"] = rng.permutation(d.loc[idx, "division_gate"].to_numpy())
        null.append(fractions_rho(d))
    null = np.asarray(null)
    res["rbd_binding_vs_division"] = {
        "spearman_rho": rho, "p_within_donor_perm": p, "n_clones": nn,
        "design_null_mean_rho": float(null.mean()),
        "design_null_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "p_vs_design_null": float((1 + np.sum(np.abs(null - null.mean()) >= abs(rho - null.mean()))) / (null.size + 1)),
        "cells_per_gate": pd.crosstab(cells["rbd_bait"], cells["division_gate"]).to_dict(),
    }
    tab.to_csv(OUT / "rbd_binding_vs_division.csv")
    stamp(f"RBD binding vs division: {res['rbd_binding_vs_division']}")
    rib = ribosome_genes(prot.var_names)
    score = cell_score(prot, rib)
    for within in (None, "mCherry-high", "mCherry-low"):
        t2 = clone_table(prot, score, "mCherry-low", within=within)
        rho, p, nn = perm_spearman(t2, "score", "frac")
        res[f"rbd_ribosome_vs_division[{within or 'all cells'}]"] = {"spearman_rho": rho,
                                                                   "p_within_donor_perm": p, "n_clones": nn}
        stamp(f"RBD arm ribosome vs division [{within}]: rho={rho:.3f} p={p:.3g} n={nn}")

    # 4) gene-set clonal heritability from the main runs, against expression-matched genes
    for ds in ("mouse_np", "mouse_rbd"):
        path = HERE / "results" / ds / "gene_heritability.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path, index_col=0)
        sets = {
            "translation (ribosomal)": [g for g in ribosome_genes(h.index) if g in h.index],
            "dark zone / cell cycle": ["Mki67", "Top2a", "Ccnb1", "Cdk1", "Aurkb", "Hmgb2", "Stmn1", "Tubb5",
                                       "Pclaf", "Rrm2", "Cxcr4", "Aicda"],
            "light zone": ["Cd83", "Cd86", "Myc", "Nfkbia", "Batf", "Egr2", "Il4i1", "Fcer2a", "H2-Oa", "Cd74",
                           "Nme1", "Bcl2a1a"],
            "plasma cell": ["Prdm1", "Xbp1", "Irf4", "Jchain", "Mzb1", "Sdc1", "Fkbp11", "Ssr4"],
            "interferon": ["Isg15", "Ifit1", "Ifit3", "Irf7", "Usp18", "Ifi203", "Ifitm3", "Cmpk2", "Ifi27l2a",
                           "Mndal", "Stat1", "Oasl1"],
        }
        gs = tf.tl.geneset_heritability(h, sets)
        gs.to_csv(OUT / f"{ds}_geneset_heritability.csv", index=False)
        res[f"{ds}_geneset_heritability"] = gs.drop(columns="genes").round(4).to_dict(orient="records")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    stamp("done")


if __name__ == "__main__":
    main()
