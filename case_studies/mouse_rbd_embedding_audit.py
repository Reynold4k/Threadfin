#!/usr/bin/env python
"""Parameter audit of the mouse_rbd clone-profile UMAP (Figure 2C-E).

Question: is the flat mutation-history display on the RBD clone map a
parameter artifact, and is the division / zone-gate structure robust?
Rebuilds the Threadfin family features exactly as run_case_study.py does,
sweeps UMAP n_neighbors / min_dist / spread / seed, and quantifies per
parameter set how strongly each measured label or programme score
associates with the 2-D map position.

Metrics per variable per parameter set:
  * lin_r2   - R^2 of value ~ x + y (linear gradient on the map)
  * knn_r2   - legacy name: explained variance (EV), from the 10 nearest
               map neighbours, excluding self; not standard predictive R^2
  * excess   - observed EV minus donor-stratified permutation mean (n_perm)
  * p_value  - empirical permutation p of EV

Outputs to results/mouse_rbd_embedding_audit/:
  sweep.csv, verify.json (default-parameter reproduction + axis test).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "mouse_rbd_embedding_audit"
N_PERM = 200
K_NN = 10


def knn_loo_pred(xy, k):
    from scipy.spatial import cKDTree

    k = min(k + 1, len(xy) - 1)
    idx = cKDTree(xy).query(xy, k=k)[1][:, 1:]
    return idx


def assoc_metrics(xy, v, donors, rng, n_perm=N_PERM):
    """OLS R² and kNN-LOO EV plus a donor-stratified permutation null.

    The historical knn_r2 column stores EV for compatibility with saved
    sweeps. EV uses residual variance; standard R² instead uses residual
    mean square and also penalises nonzero mean prediction bias.
    """
    m = np.isfinite(v)
    n = int(m.sum())
    if n < 30:
        return None
    xy, v, d = xy[m], v[m], np.asarray(donors)[m]
    X = np.column_stack([np.ones(n), xy])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    lin_r2 = 1.0 - float((v - X @ beta).var() / v.var())
    N = knn_loo_pred(xy, K_NN)
    obs = 1.0 - float((v - v[N].mean(axis=1)).var() / v.var())
    blocks = [np.flatnonzero(d == g) for g in pd.unique(d)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        vp = v.copy()
        for bl in blocks:
            vp[bl] = rng.permutation(vp[bl])
        null[b] = 1.0 - float((vp - vp[N].mean(axis=1)).var() / vp.var())
    return {"n": n, "lin_r2": lin_r2, "knn_r2": obs,
            "null_mean": float(null.mean()), "excess": obs - float(null.mean()),
            "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1))}


def faithfulness(feats, xy, rng, n_pairs=30000):
    from scipy.stats import spearmanr

    n = len(xy)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_hi = np.linalg.norm(feats[i] - feats[j], axis=1)
    d_lo = np.linalg.norm(xy[i] - xy[j], axis=1)
    return float(spearmanr(d_hi, d_lo).statistic)


def main():
    t0 = time.time()

    def stamp(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    import umap

    import threadfin as tf
    from datasets import LOADERS

    OUT.mkdir(parents=True, exist_ok=True)
    stamp("loading mouse_rbd")
    adata, bcr = LOADERS["mouse_rbd"]()
    bcr["isotype"] = tf.clones.isotype_class(bcr["c_call"]).values
    bcr["donor"] = adata.obs["donor"].astype(str).reindex(bcr.index).values
    bcr = tf.define_clones(bcr, donor_key="donor", out_col="clone_id")
    tf.attach_bcr(adata, bcr.drop(columns=["donor", "author_clone_id"], errors="ignore"),
                  clone_col="clone_id", summarize=False)
    stamp(f"{adata.n_obs} cells; building expression embedding and family profiles")
    tf.pp.prepare_embedding(adata, batch_key=None, key_added="X_threadfin", verbose=False)
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", verbose=False)
    prof = adata.uns["threadfin"]["profiles"]
    table, feats_all = prof["clone_table"], prof["features"]
    eligible = table.index[table["reliability"] >= 0.5]
    feats = feats_all.loc[eligible].to_numpy()
    stamp(f"features {feats.shape} for {len(eligible)} reliable clones")

    committed = pd.read_csv(HERE / "results" / "mouse_rbd" / "clone_table.csv", index_col=0)
    scores = pd.read_csv(HERE / "results" / "mouse_rbd" / "clone_gene_scores.csv", index_col=0)
    lab = committed.reindex(eligible).join(scores)
    donors = lab["donor"].astype(str).to_numpy()

    # verify the default configuration reproduces the committed map
    rng = np.random.default_rng(0)
    xy0 = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(feats)
    saved = lab[["x", "y"]].to_numpy()
    repro = {ax: float(np.corrcoef(xy0[:, k], saved[:, k])[0, 1]) for k, ax in enumerate("xy")}
    stamp(f"default-config reproduction corr: {repro}")

    variables = {
        "mutation_frequency": lab["mutation_frequency"].to_numpy(float),
        "division_gate:mCherry-low": lab["division_gate:mCherry-low"].to_numpy(float),
        "zone_gate:DZ": lab["zone_gate:DZ"].to_numpy(float),
        "rbd_bait:RBD+": lab["rbd_bait:RBD+"].to_numpy(float),
        "dark zone / cycling": lab["dark zone / cycling"].to_numpy(float),
        "light zone": lab["light zone"].to_numpy(float),
        "LZ minus DZ": (lab["light zone"] - lab["dark zone / cycling"]).to_numpy(float),
        "germinal centre": lab["germinal centre"].to_numpy(float),
        "plasma cell": lab["plasma cell"].to_numpy(float),
        "memory": lab["memory"].to_numpy(float),
    }

    # axis-alignment test on the committed map: LZ-DZ programme vs measured division
    v_div = variables["division_gate:mCherry-low"]
    v_ax = variables["LZ minus DZ"]
    m = np.isfinite(v_div) & np.isfinite(v_ax)
    obs_c = float(np.corrcoef(v_ax[m], v_div[m])[0, 1])
    blocks = [np.flatnonzero(donors[m] == g) for g in pd.unique(donors[m])]
    null_c = np.empty(N_PERM)
    va, vd = v_ax[m], v_div[m]
    for b in range(N_PERM):
        vp = vd.copy()
        for bl in blocks:
            vp[bl] = rng.permutation(vp[bl])
        null_c[b] = np.corrcoef(va, vp)[0, 1]
    axis_test = {"corr_lz_minus_dz_vs_mcherry_low": obs_c,
                 "null_mean_abs": float(np.abs(null_c).mean()),
                 "p_value_two_sided": float((1 + (np.abs(null_c) >= abs(obs_c)).sum()) / (N_PERM + 1)),
                 "n_clones": int(m.sum())}
    stamp(f"axis alignment: r={obs_c:.3f} p={axis_test['p_value_two_sided']}")

    grid = []
    for nn in (5, 10, 15, 25, 50, 100):
        for md in (0.0, 0.1, 0.25, 0.5, 0.9):
            for seed in (0, 1):
                grid.append(dict(n_neighbors=nn, min_dist=md, spread=1.0, random_state=seed))
    for sp in (0.5, 2.0):
        for nn in (10, 15, 30):
            for md in (0.1, 0.5):
                for seed in (0, 1):
                    grid.append(dict(n_neighbors=nn, min_dist=md, spread=sp, random_state=seed))
    grid.append(dict(n_neighbors=15, min_dist=0.1, spread=1.0, random_state=2))
    stamp(f"sweep: {len(grid)} parameter sets")

    rows = []
    for gi, g in enumerate(grid):
        xy = umap.UMAP(**g).fit_transform(feats)
        faith = faithfulness(feats, xy, rng)
        for name, v in variables.items():
            r = assoc_metrics(xy, v, donors, rng)
            if r is None:
                continue
            rows.append({**g, "variable": name, "faithfulness": faith, **r})
        if (gi + 1) % 10 == 0:
            stamp(f"{gi + 1}/{len(grid)} done")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sweep.csv", index=False)
    (OUT / "verify.json").write_text(json.dumps(
        {"default_config": {"n_neighbors": 15, "min_dist": 0.1, "spread": 1.0, "random_state": 0},
         "reproduction_axis_corr": repro, "n_reliable_clones": int(len(eligible)),
         "feature_shape": list(feats.shape), "axis_alignment": axis_test,
         "n_parameter_sets": len(grid), "n_perm": N_PERM, "k_nn": K_NN}, indent=1))
    stamp(f"done -> {OUT / 'sweep.csv'}")


if __name__ == "__main__":
    main()
