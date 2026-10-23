#!/usr/bin/env python
"""How many clones does the clonal-coherence test need? Power by simulation.

Each simulated dataset has two donors with two samples each. Clones with two or more cells (2 + Poisson(1.5)
cells, so 3.5 on average) are sampled within one sample; as many cells again are singletons. A cell's
30-dimensional state is its sample's shift (SD 0.5), plus its clone's shift (SD s), plus noise (SD 1), so clone
identity explains a share ICC = s² / (s² + 1) of the variance within samples. The test is the one `tf.run`
performs: mean clone profiles centred on the sample, clonal ICC against 200 permutations of clone labels within
samples, significant at P < 0.05. ICC 0 gives the false-positive rate.

For orientation, the published case studies show clonal ICCs of 0.05-0.74 with shuffled values of 0.00-0.30,
that is, excesses over shuffled clones of 0.04-0.73, from 92 to 11,126 clones with two or more cells.

Usage: python coherence_power.py <out_dir> [n_jobs]
       python coherence_power.py --plot <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ICCS = (0.0, 0.02, 0.05, 0.1, 0.2)
N_CLONES = (10, 20, 40, 80, 160, 320)
N_REP = 100
DIM = 30


def simulate(n_clones, icc, seed):
    from anndata import AnnData

    rng = np.random.default_rng(seed)
    s = np.sqrt(icc / (1 - icc))
    sizes = 2 + rng.poisson(1.5, n_clones)
    sample_of_clone = rng.integers(0, 4, n_clones)
    clone = np.repeat(np.arange(n_clones), sizes)
    sample = sample_of_clone[clone]
    n_single = int(sizes.sum())
    sample = np.concatenate([sample, rng.integers(0, 4, n_single)])
    shift = rng.normal(0, 0.5, (4, DIM))
    offset = np.vstack([rng.normal(0, s, (n_clones, DIM))[clone], np.zeros((n_single, DIM))])
    x = shift[sample] + offset + rng.normal(0, 1, (sample.size, DIM))
    ids = [f"c{c}" for c in clone] + [f"s{i}" for i in range(n_single)]
    obs = pd.DataFrame({"clone_id": ids, "sample": [f"S{k}" for k in sample],
                        "donor": [f"D{k // 2}" for k in sample]}, index=[f"cell{i}" for i in range(sample.size)])
    obs["clone_id"] = obs["donor"] + "|" + obs["clone_id"]
    adata = AnnData(X=np.zeros((sample.size, 0), dtype=np.float32), obs=obs)
    adata.obsm["X_emb"] = x.astype(np.float32)
    return adata


def one(n_clones, icc, rep):
    import threadfin as tf

    adata = simulate(n_clones, icc, seed=[n_clones, int(round(1000 * icc)), rep])
    tf.tl.clone_profiles(adata, basis="X_emb", context_key="sample", donor_key="donor", representation="mean",
                         min_cells=2, verbose=False)
    coh = tf.tl.clonal_coherence(adata, strata_key="sample", n_perm=200, random_state=rep, verbose=False)
    return {"n_clones": n_clones, "icc_true": icc, "rep": rep, "icc": coh["icc"], "icc_shuffled": coh["null_mean"],
            "p_value": coh["p_value"]}


def main(out: Path, n_jobs: int):
    from joblib import Parallel, delayed

    import threadfin as tf

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(n, icc, r) for icc in ICCS for n in N_CLONES for r in range(N_REP)]
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "coherence_power_runs.csv", index=False)
    power = (runs.assign(significant=runs["p_value"] < 0.05)
             .groupby(["icc_true", "n_clones"])
             .agg(power=("significant", "mean"), icc_median=("icc", "median"),
                  excess_median=("icc", lambda v: float(np.median(v - runs.loc[v.index, "icc_shuffled"]))))
             .reset_index())
    power.to_csv(out / "coherence_power.csv", index=False)
    (out / "run.json").write_text(json.dumps({"threadfin": tf.__version__, "replicates": N_REP, "n_perm": 200,
                                              "minutes": round((time.time() - t0) / 60, 1)}, indent=1) + "\n")
    print(power.pivot(index="n_clones", columns="icc_true", values="power").to_string())


def plot(out: Path):
    import matplotlib.pyplot as plt

    import threadfin as tf
    from threadfin.plotting import BLUE_RAMP, INK_2

    tf.pl.set_style()
    power = pd.read_csv(out / "coherence_power.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    colours = dict(zip(ICCS, ["#898781"] + BLUE_RAMP[2::1][:4]))
    for icc, sub in power.groupby("icc_true"):
        label = "0 (no clonal effect)" if icc == 0 else f"{icc:g}"
        ax.plot(sub["n_clones"], sub["power"], "-o", ms=4, lw=1.2, color=colours[icc], label=label)
    ax.axhline(0.05, color=INK_2, lw=0.5, ls=":")
    ax.axhline(0.8, color=INK_2, lw=0.5, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_CLONES, [str(n) for n in N_CLONES])
    ax.set_xlabel("clones with ≥ 2 cells")
    ax.set_ylabel("share of datasets with P < 0.05")
    ax.set_ylim(0, 1.02)
    ax.legend(title="clonal ICC", loc="center right", fontsize=5.5, title_fontsize=6)
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "coherence_power.png")
    fig.savefig(out / "figures" / "coherence_power.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1)
