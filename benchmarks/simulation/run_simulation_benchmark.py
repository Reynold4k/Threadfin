#!/usr/bin/env python
"""Simulation benchmark for Threadfin v4 (ground truth known).

Tasks (each writes one CSV to benchmarks/simulation/results/):

  recovery <scenario> <seed>   programme recovery (ARI) for 8 methods
  calibration <seed>           false-positive rates of coherence/association nulls
  memory <memory> <seed>       clonal memory index vs the true keep-rate
  scaling <n_cells>            runtime and peak memory

Usage:
  python run_simulation_benchmark.py recovery default 1
  python run_simulation_benchmark.py calibration 7
  python run_simulation_benchmark.py memory 0.5 3
  python run_simulation_benchmark.py scaling 200000
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
import threadfin as tf  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESOLUTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0)

SCENARIOS = {
    "default": dict(),
    "strong_batch": dict(context_shift=2.5),
    "no_batch": dict(context_shift=0.0),
    "bifurcation": dict(scenario="bifurcation"),
    "small_clones": dict(clone_size_exponent=2.6, n_clones=6000),
    "spread": dict(clone_nesting="spread"),
}
BASE = dict(n_clones=3000, clone_size_exponent=2.0)


def _ari(truth, labels):
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, labels))


def _leiden_scan(features: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray):
    """Best ARI over Leiden resolutions (oracle tuning, identical for every method)."""
    graph = tf.programmes.knn_graph(features, 15)
    best = (-1.0, None, None)
    for r in RESOLUTIONS:
        lab = tf.programmes.leiden(graph, r)
        a = _ari(truth[eval_mask], lab[eval_mask])
        if a > best[0]:
            best = (a, r, lab)
    return best


def _size_bins(n):
    return pd.cut(n, [2, 4, 9, np.inf], labels=["3-4", "5-9", ">=10"])


def task_recovery(scenario: str, seed: int):
    import scanpy as sc

    kw = {**BASE, **SCENARIOS[scenario]}
    ad = tf.sim.simulate_repertoire(random_state=seed, **kw)
    sizes = ad.obs["clone_id"].value_counts()
    clones = sizes.index[sizes >= 3]
    truth_all = ad.obs.groupby("clone_id")["true_programme"].first()
    truth = truth_all.loc[clones].to_numpy()
    nbin = _size_bins(sizes.loc[clones].to_numpy())
    rows = []

    def record(method, labels, extra=None):
        labels = np.asarray(labels)
        row = {"scenario": scenario, "seed": seed, "method": method, "ari": _ari(truth, labels),
               "n_clones": int(len(clones)), "n_found": int(len(np.unique(labels)))}
        for b in ["3-4", "5-9", ">=10"]:
            m = np.asarray(nbin == b)
            row[f"ari_{b}"] = _ari(truth[m], labels[m]) if m.sum() >= 10 else np.nan
        row.update(extra or {})
        rows.append(row)

    def from_features(method, feats: pd.DataFrame, t0):
        f = feats.reindex(clones).to_numpy()
        ari, res, lab = _leiden_scan(f, truth, np.ones(len(clones), bool))
        record(method, lab, {"resolution": res, "seconds": time.time() - t0})

    # 1) Threadfin v3: centroid of PCA coordinates, Leiden (best resolution)
    t0 = time.time()
    x = ad.obsm["X_pca"].astype(float)
    cent = pd.DataFrame(x, index=ad.obs_names).groupby(ad.obs["clone_id"].to_numpy()).mean()
    from_features("v3 centroid", cent, t0)

    # 2-3) cell clusters -> modal cluster per clone / composition per clone
    t0 = time.time()
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=15, random_state=0)
    sc.tl.leiden(ad, resolution=1.0, key_added="cell_cluster", random_state=0, flavor="igraph",
                 n_iterations=2, directed=False)
    modal = ad.obs.groupby("clone_id")["cell_cluster"].agg(lambda s: s.value_counts().index[0])
    record("modal cell state", modal.loc[clones].to_numpy(), {"seconds": time.time() - t0})
    comp = pd.crosstab(ad.obs["clone_id"], ad.obs["cell_cluster"])
    from_features("state composition", np.sqrt(comp.div(comp.sum(axis=1), axis=0)), t0)

    # 4-5) clone2vec on the raw and on the context-centred embedding
    for label, rep in (("clone2vec", "X_pca"), ("clone2vec (centred)", "X_centred")):
        t0 = time.time()
        try:
            import clone2vec as c2v

            if rep == "X_centred":
                ctx = ad.obs["context"].to_numpy()
                mu = pd.DataFrame(x).groupby(ctx).transform("mean").to_numpy()
                ad.obsm["X_centred"] = (x - mu).astype(np.float32)
            cl = c2v.pp.clones_adata(ad, obs_name="clone_id", min_size=3)
            c2v.tl.clonal_nn(ad, cl, obs_name="clone_id", use_rep=rep, k=15)
            c2v.tl.clone2vec(cl, z_dim=10, progress_bar=False)
            from_features(label, pd.DataFrame(cl.obsm["clone2vec"], index=cl.obs_names), t0)
        except Exception as e:  # pragma: no cover - record failures instead of crashing
            rows.append({"scenario": scenario, "seed": seed, "method": label, "error": repr(e)})

    # 6-7) Threadfin v4 profiles (mean and kernel), best resolution and the full auto pipeline
    variants = (("mean", "median", 0, "mean"), ("kernel", "median", 15, "kernel"),
                ("kernel", "median", 0, "kernel, unsmoothed"), ("kernel", "local", 15, "kernel, local"))
    for rep, bw, smooth, label in variants:
        t0 = time.time()
        tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                             representation=rep, bandwidth=bw, smooth=smooth, min_cells=3, verbose=False)
        feats = ad.uns["threadfin"]["profiles"]["features"]
        from_features(f"Threadfin v4 ({label})", feats, t0)
        t0 = time.time()
        try:
            tf.tl.find_programmes(ad, n_boot=20, embed=False, verbose=False)
        except ValueError as e:
            rows.append({"scenario": scenario, "seed": seed, "method": f"Threadfin v4 ({label}) auto",
                         "error": str(e)})
            continue
        table = ad.uns["threadfin"]["profiles"]["clone_table"]
        for col, tag in (("clone_programme", "core"), ("clone_programme_assigned", "core + assigned")):
            lab = table[col].reindex(clones)
            covered = lab.notna().to_numpy()
            rows.append({"scenario": scenario, "seed": seed, "method": f"Threadfin v4 ({label}) auto, {tag}",
                         "ari": _ari(truth[covered], lab[covered].astype(str).to_numpy()),
                         "coverage": float(covered.mean()), "n_clones": int(len(clones)),
                         "n_found": int(lab.nunique()),
                         "resolution": ad.uns["threadfin"]["programmes"]["params"]["resolution"],
                         "seconds": time.time() - t0})

    # 8) reference: composition over the TRUE states (noisy for small clones; not a ceiling)
    t0 = time.time()
    comp_true = pd.crosstab(ad.obs["clone_id"], ad.obs["true_state"])
    from_features("true-state composition (reference)", np.sqrt(comp_true.div(comp_true.sum(axis=1), axis=0)), t0)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"recovery_{scenario}_{seed}.csv", index=False)
    print(out[["method", "ari", "n_found"]].to_string(index=False), flush=True)


def task_calibration(seed: int):
    """Null data: how often does each test call a non-existent effect significant?"""
    import scanpy as sc
    from scipy.stats import fisher_exact

    rows = []
    # (a) coherence: no clonal signal beyond the sampling context
    ad = tf.sim.simulate_repertoire(scenario="null", n_clones=1500, clone_size_exponent=2.0,
                                    random_state=1000 + seed)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key=None, donor_key="donor", verbose=False)
    p_within = tf.tl.clonal_coherence(ad, strata_key="context", n_perm=100, verbose=False)["p_value"]
    p_global = tf.tl.clonal_coherence(ad, strata_key=None, n_perm=100, verbose=False)["p_value"]
    # v3 "null 1": mean clone purity over Leiden cell states vs within-donor label permutation
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=15, random_state=0)
    sc.tl.leiden(ad, resolution=0.8, key_added="state", random_state=0, flavor="igraph", n_iterations=2,
                 directed=False)
    rng = np.random.default_rng(seed)

    def purity(clone):
        df = pd.DataFrame({"c": clone, "s": ad.obs["state"].to_numpy()})
        size = df["c"].map(df["c"].value_counts())
        df = df[size >= 3]
        return df.groupby("c")["s"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()

    real = purity(ad.obs["clone_id"].to_numpy())
    donors = ad.obs["donor"].to_numpy()
    null = []
    for _ in range(100):
        perm = ad.obs["clone_id"].to_numpy().copy()
        for d in np.unique(donors):
            idx = np.flatnonzero(donors == d)
            perm[idx] = perm[rng.permutation(idx)]
        null.append(purity(perm))
    p_v3 = (1 + np.sum(np.asarray(null) >= real)) / 101
    rows += [{"seed": seed, "test": "coherence", "method": "Threadfin v4 (within-sample null)", "pvalue": p_within},
             {"seed": seed, "test": "coherence", "method": "global shuffle", "pvalue": p_global},
             {"seed": seed, "test": "coherence", "method": "v3 null 1 (purity, within-donor)", "pvalue": p_v3}]

    # (b) association: programmes are real, but the label is a coin flipped per CLONE
    ad = tf.sim.simulate_repertoire(n_clones=2500, clone_size_exponent=1.9, random_state=2000 + seed)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    tf.tl.find_programmes(ad, n_boot=10, embed=False, verbose=False)
    clones = ad.obs["clone_id"].unique()
    coin = pd.Series(rng.random(clones.size) < 0.3, index=clones)
    ad.obs["coin"] = ad.obs["clone_id"].map(coin).map({True: "yes", False: "no"})
    v4 = tf.tl.association_test(ad, "coin", n_perm=500, verbose=False)
    for p in v4.loc[v4["level"] == "yes", "pvalue"]:
        rows.append({"seed": seed, "test": "association", "method": "Threadfin v4 (clone-level)", "pvalue": p})
    # v3: one-sided Fisher on CELLS (programme x label), as in state_enrichment
    obs = ad.obs[["clone_programme", "coin"]].dropna()
    for prog in obs["clone_programme"].cat.categories:
        in_p = (obs["clone_programme"] == prog).to_numpy()
        yes = (obs["coin"] == "yes").to_numpy()
        a, b = int((in_p & yes).sum()), int((in_p & ~yes).sum())
        c, d = int((~in_p & yes).sum()), int((~in_p & ~yes).sum())
        p = fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]
        rows.append({"seed": seed, "test": "association", "method": "v3 (cell-level Fisher)", "pvalue": p})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"calibration_{seed}.csv", index=False)
    print(out.groupby(["test", "method"])["pvalue"].apply(lambda p: (p < 0.05).mean()), flush=True)


def task_memory(memory: float, seed: int):
    ad = tf.sim.simulate_repertoire(n_clones=3000, clone_size_exponent=1.8, n_timepoints=2,
                                    memory=memory, random_state=3000 + seed)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    tf.tl.find_programmes(ad, n_boot=10, embed=False, verbose=False)
    res = tf.tl.clonal_memory(ad, "timepoint", verbose=False)
    tr = tf.clones.community_transition(ad, time_key="timepoint", cluster_key="clone_programme")
    first = ad.obs.groupby(["clone_id", "timepoint"])["true_programme"].first().unstack()
    evaluated = res["pairs"]["clone"].unique()
    true_keep = float((first.loc[evaluated, "t0"] == first.loc[evaluated, "t1"]).mean())
    out = pd.DataFrame([{
        "memory": memory, "seed": seed, "true_keep_rate": true_keep,
        "v4_memory_index": res["memory_index"], "ci_low": res["memory_index_ci"][0],
        "ci_high": res["memory_index_ci"][1], "p_value": res["p_value"],
        "v4_programme_persistence": res.get("persistence"), "v4_persistence_null": res.get("persistence_null"),
        "v3_transition_diagonal": float(np.trace(tr.to_numpy()) / tr.to_numpy().sum()),
        "n_clones": res["n_clones"],
    }])
    out.to_csv(OUT / f"memory_{memory}_{seed}.csv", index=False)
    print(out.T, flush=True)


def task_scaling(n_cells: int):
    import psutil

    proc = psutil.Process()
    n_clones = int(n_cells / 6.3)  # mean clone size of this size distribution
    t0 = time.time()
    ad = tf.sim.simulate_repertoire(n_clones=n_clones, clone_size_exponent=1.9, max_clone_size=1000,
                                    random_state=0)
    rec = {"n_cells_requested": n_cells, "n_cells": ad.n_obs, "n_clones": int(ad.obs.clone_id.nunique()),
           "simulate_s": time.time() - t0}
    steps = [
        ("profiles_kernel_s", lambda: tf.tl.clone_profiles(ad, basis="X_pca", context_key="context",
                                                           donor_key="donor", representation="kernel",
                                                           verbose=False)),
        ("coherence_50perm_s", lambda: tf.tl.clonal_coherence(ad, n_perm=50, verbose=False)),
        ("programmes_20boot_s", lambda: tf.tl.find_programmes(ad, n_boot=20, embed=False, verbose=False)),
    ]
    for name, fn in steps:
        t0 = time.time()
        fn()
        rec[name] = time.time() - t0
        rec[name.replace("_s", "_rss_gb")] = proc.memory_info().rss / 1e9
    rec["n_programme_clones"] = int(ad.uns["threadfin"]["profiles"]["clone_table"]["clone_programme"].notna().sum())
    rec["peak_rss_gb"] = proc.memory_info().rss / 1e9
    pd.DataFrame([rec]).to_csv(OUT / f"scaling_{n_cells}.csv", index=False)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    task, *args = sys.argv[1:]
    if task == "recovery":
        task_recovery(args[0], int(args[1]))
    elif task == "calibration":
        task_calibration(int(args[0]))
    elif task == "memory":
        task_memory(float(args[0]), int(args[1]))
    elif task == "scaling":
        task_scaling(int(args[0]))
    else:
        raise SystemExit(__doc__)
