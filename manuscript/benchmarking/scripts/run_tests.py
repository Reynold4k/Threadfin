#!/usr/bin/env python
"""Calibration and power of tests for "do cells of one clone share a state?".

Three tests answer this question in different software:

* Threadfin ``tl.clonal_coherence``: clonal intraclass correlation (share of embedding variance explained
  by clone) against clone labels permuted *within samples* (one test per dataset);
* scirpy ``tl.clonotype_modularity`` (Sturm et al. 2020, inspired by CoNGA): for every clone, the excess
  of edges between its cells in the cell neighbour graph over random subgraphs of the same size drawn
  from the *whole* graph; Benjamini-Hochberg across clones (one test per clone);
* a cell-level analysis of variance: each of the first 10 embedding dimensions compared between clones
  with cells as replicates (F test, Bonferroni over dimensions) — the analysis that ignores how cells
  were sampled.

The data are real; only the clone labels are changed:

``observed``           the real clone labels;
``null_within_sample`` labels permuted among clone-assigned cells within each sample: clone structure is
                       destroyed but every clone still comes from one sample, as in real data, so any
                       sample-level difference in cell state remains confounded with clone;
``null_global``        labels permuted among all clone-assigned cells;
``mix_<f>``            a fraction f of the cells keep their real label and the rest receive a label
                       permuted within sample (power as real clone structure is diluted).

A calibrated test rejects in about 5 % of null replicates at alpha = 0.05.

Usage: python run_tests.py <dataset> [n_replicates]
"""
from __future__ import annotations

import json
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import bench_io
from bench_paths import RESULTS

FRACTIONS = (0.1, 0.25, 0.5)


def permute_within(labels: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    out = labels.copy()
    lab = np.flatnonzero(pd.notna(labels))
    for g in np.unique(groups[lab]):
        idx = lab[groups[lab] == g]
        out[idx] = labels[rng.permutation(idx)]
    return out


def scenario_labels(labels, samples, scenario, rng):
    if scenario == "observed":
        return labels.copy()
    if scenario == "null_within_sample":
        return permute_within(labels, samples, rng)
    if scenario == "null_global":
        return permute_within(labels, np.zeros(len(labels), dtype=int), rng)
    f = float(scenario.split("_")[1])
    shuffled = permute_within(labels, samples, rng)
    keep = rng.random(len(labels)) < f
    return np.where(keep, labels, shuffled)


def threadfin_test(x, cells, labels, n_perm=99):
    import threadfin as tf

    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32), obs=cells[["donor", "sample"]].copy())
    a.obs["clone"] = pd.Series(labels, index=cells.index, dtype=object)
    a.obsm["X_emb"] = x
    tf.tl.clone_profiles(a, clone_key="clone", basis="X_emb", context_key="sample", representation="mean",
                         min_cells=2, verbose=False)
    res = tf.tl.clonal_coherence(a, strata_key="sample", n_perm=n_perm, verbose=False)
    return res["p_value"], res["icc"], res["null_mean"]


def scirpy_test(graph_adata, labels):
    import scirpy as ir

    g = graph_adata
    g.obs["clone"] = pd.Series(labels, index=g.obs_names, dtype=object)
    sizes = g.obs["clone"].value_counts()
    g.obs.loc[~g.obs["clone"].isin(sizes.index[sizes >= 2]), "clone"] = None
    ir.tl.clonotype_modularity(g, target_col="clone", connectivity_key="connectivities", permutation_test="approx",
                               random_state=0)
    per_clone = g.obs.dropna(subset=["clone"]).groupby("clone")["clonotype_modularity_fdr"].first()
    return float((per_clone < 0.05).mean()), int((per_clone < 0.05).sum()), int(len(per_clone))


def anova_test(x, labels, n_dims=10):
    s = pd.Series(labels)
    sizes = s.value_counts()
    keep = s.isin(sizes.index[sizes >= 2]).to_numpy()
    groups = s[keep].to_numpy()
    xs = x[keep, :n_dims]
    codes = pd.factorize(groups)[0]
    ps = []
    for j in range(xs.shape[1]):
        samples = [xs[codes == c, j] for c in range(codes.max() + 1)]
        ps.append(stats.f_oneway(*samples).pvalue)
    return float(min(1.0, np.min(ps) * len(ps)))


def main(dataset, n_rep=50):
    import scanpy as sc

    x, cells = bench_io.load(dataset, "full")
    labels = cells["clone"].astype(object).where(cells["clone"].notna(), None).to_numpy()
    samples = cells["sample"].astype(str).to_numpy()
    g = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32), obs=pd.DataFrame(index=cells.index))
    g.obsm["X_emb"] = x
    sc.pp.neighbors(g, use_rep="X_emb", n_neighbors=15, random_state=0)
    rng = np.random.default_rng(7)
    rows = []
    scenarios = ["observed"] + ["null_within_sample"] * n_rep + ["null_global"] * n_rep + \
                [f"mix_{f}" for f in FRACTIONS for _ in range(max(10, n_rep // 5))]
    for i, sc_name in enumerate(scenarios):
        lab = scenario_labels(labels, samples, sc_name, rng)
        t0 = time.perf_counter()
        p_tf, icc, null_mean = threadfin_test(x, cells, lab)
        t1 = time.perf_counter()
        frac, n_sig, n_clones = scirpy_test(g, lab)
        t2 = time.perf_counter()
        p_an = anova_test(x, lab)
        rows.append({"dataset": dataset, "scenario": sc_name.split("_")[0] if sc_name.startswith("mix") else sc_name,
                     "fraction_real_labels": float(sc_name.split("_")[1]) if sc_name.startswith("mix") else
                     (1.0 if sc_name == "observed" else 0.0), "replicate": i,
                     "threadfin_p": p_tf, "threadfin_icc": icc, "threadfin_null_mean": null_mean,
                     "scirpy_fraction_clones_fdr05": frac, "scirpy_n_clones_fdr05": n_sig, "scirpy_n_clones": n_clones,
                     "cell_anova_p": p_an, "seconds_threadfin": round(t1 - t0, 2), "seconds_scirpy": round(t2 - t1, 2)})
        if i % 10 == 0:
            print(json.dumps(rows[-1]), flush=True)
    out = RESULTS / "tests"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{dataset}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
