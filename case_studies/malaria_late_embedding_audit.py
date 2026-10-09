#!/usr/bin/env python
"""Parameter audit of the malaria_late clone-profile UMAP (Figure 3C context).

Question: the later-infection dataset is a dynamic B-cell landscape
(d10-d42, treatment arms, 11 author cell states). Does the Threadfin clone
map encode the infection dynamics (sampling day), and is the GC-biased
colouring used in Figure 3C the most informative representation? Rebuilds
the committed family features exactly as run_case_study.py does, sweeps
UMAP n_neighbors / min_dist / spread / seed, and quantifies per parameter
set the association of the 2-D map with state fractions, mutation history,
sampling day and treatment.

Metrics per variable per parameter set (assoc_metrics from
mouse_rbd_embedding_audit):
  * lin_r2   - R^2 of value ~ x + y
  * knn_r2   - leave-one-out R^2 from the 10 nearest map neighbours
  * excess   - knn_r2 minus permutation mean (donor-stratified, or
               unstratified for donor-level variables: day, treatment)
  * p_value  - empirical permutation p of knn_r2

Outputs to results/malaria_late_embedding_audit/:
  sweep.csv, verify.json, gallery_coords.csv.gz,
  gallery_{gc,pb,memory,day,mutation}.png
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "malaria_late_embedding_audit"

from mouse_rbd_embedding_audit import assoc_metrics, faithfulness  # noqa: E402

GALLERY_NNS = [5, 15, 50]
GALLERY_MDS = [0.0, 0.1, 0.5, 0.9]
COMMITTED = (15, 0.1)


def gallery(coords, mets, varr, sizes, fname, cmap, clabel, vmin=None, vmax=None):
    fig, axs = plt.subplots(len(GALLERY_NNS), len(GALLERY_MDS), figsize=(13.5, 10.8))
    for i, nn in enumerate(GALLERY_NNS):
        for j, md in enumerate(GALLERY_MDS):
            ax = axs[i, j]
            xy = coords[(nn, md)]
            m = mets[(nn, md)]
            im = ax.scatter(xy[:, 0], xy[:, 1], c=varr, s=sizes, cmap=cmap,
                            edgecolors="white", linewidths=.15, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ttl = ax.set_title(f"k={nn}  min_dist={md}\nlin R²={m['lin_r2']:.2f} · local ΔR²={m['excess']:.2f} "
                               f"(p={m['p_value']:.3f})", fontsize=7.5)
            ttl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.2))
            if (nn, md) == COMMITTED:
                for s in ax.spines.values():
                    s.set_color("#c65359")
                    s.set_linewidth(2.2)
                ax.text(.02, .02, "committed\nconfiguration", transform=ax.transAxes,
                        fontsize=7, color="#c65359", fontweight="bold", va="bottom",
                        bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#c65359", lw=.8))
    fig.suptitle(f"malaria_late clone map across UMAP parameters — {clabel}\n"
                 f"1,183 reliable clones; rows = n_neighbors, columns = min_dist (spread=1, seed=0); "
                 f"red frame = configuration used in Figure 3", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=.03, pad=.02, aspect=40)
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def main():
    t0 = time.time()

    def stamp(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    import umap

    import threadfin as tf
    from datasets import LOADERS

    OUT.mkdir(parents=True, exist_ok=True)
    stamp("loading malaria_late")
    adata, bcr = LOADERS["malaria_late"]()
    bcr["isotype"] = tf.clones.isotype_class(bcr["c_call"]).values
    bcr["donor"] = adata.obs["donor"].astype(str).reindex(bcr.index).values
    bcr = tf.define_clones(bcr, donor_key="donor", out_col="clone_id")
    tf.attach_bcr(adata, bcr.drop(columns=["donor", "author_clone_id"], errors="ignore"),
                  clone_col="clone_id", summarize=False)
    stamp(f"{adata.n_obs} cells; expression embedding + family profiles")
    tf.pp.prepare_embedding(adata, batch_key=None, key_added="X_threadfin", verbose=False)
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", verbose=False)
    prof = adata.uns["threadfin"]["profiles"]
    table, feats_all = prof["clone_table"], prof["features"]
    eligible = table.index[table["reliability"] >= 0.5]
    feats = feats_all.loc[eligible].to_numpy()
    stamp(f"features {feats.shape} for {len(eligible)} reliable clones")

    committed = pd.read_csv(HERE / "results" / "malaria_late" / "clone_table.csv", index_col=0)
    scores = pd.read_csv(HERE / "results" / "malaria_late" / "clone_gene_scores.csv", index_col=0)
    cells = pd.read_csv(HERE / "results" / "malaria_late" / "cells.csv.gz", index_col=0)
    lab = committed.reindex(eligible).join(scores)
    mem_frac = pd.crosstab(cells.clone_id, cells.cell_state, normalize="index")["Memory"]
    lab["cell_state:Memory"] = mem_frac.reindex(lab.index)
    lab["day"] = pd.to_numeric(lab.donor.astype(str).str.extract(r"D(\d+)")[0])
    lab["treated"] = lab.treatment.map({"Saline": 0.0, "Artesunate": 1.0})
    donors = lab["donor"].astype(str).to_numpy()
    same = np.array(["all"] * len(lab))  # donor-level variables: unstratified null

    rng = np.random.default_rng(0)
    xy0 = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(feats)
    saved = lab[["x", "y"]].to_numpy()
    repro = {ax: float(np.corrcoef(xy0[:, k], saved[:, k])[0, 1]) for k, ax in enumerate("xy")}
    stamp(f"default-config reproduction corr: {repro}")

    variables = {
        "cell_state:GC": (lab["cell_state:GC"].to_numpy(float), donors),
        "cell_state:PB": (lab["cell_state:PB"].to_numpy(float), donors),
        "cell_state:Memory": (lab["cell_state:Memory"].to_numpy(float), donors),
        "mutation_frequency": (lab["mutation_frequency"].to_numpy(float), donors),
        "germinal centre": (lab["germinal centre"].to_numpy(float), donors),
        "dark zone / cycling": (lab["dark zone / cycling"].to_numpy(float), donors),
        "plasma cell": (lab["plasma cell"].to_numpy(float), donors),
        "memory score": (lab["memory"].to_numpy(float), donors),
        "day": (lab["day"].to_numpy(float), same),
        "treated": (lab["treated"].to_numpy(float), same),
    }

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
    coords, gmets = {}, {}
    for gi, g in enumerate(grid):
        xy = umap.UMAP(**g).fit_transform(feats)
        faith = faithfulness(feats, xy, rng)
        for name, (v, strata) in variables.items():
            r = assoc_metrics(xy, v, strata, rng)
            if r is not None:
                rows.append({**g, "variable": name, "faithfulness": faith, **r})
        key = (g["n_neighbors"], g["min_dist"])
        if g["spread"] == 1.0 and g["random_state"] == 0 and g["n_neighbors"] in GALLERY_NNS \
                and g["min_dist"] in GALLERY_MDS:
            coords[key] = xy
            gmets[key] = {name: assoc_metrics(xy, v, strata, np.random.default_rng(1))
                          for name, (v, strata) in variables.items()}
        if (gi + 1) % 10 == 0:
            stamp(f"{gi + 1}/{len(grid)} done")
    pd.DataFrame(rows).to_csv(OUT / "sweep.csv", index=False)

    co = pd.DataFrame({"clone_id": eligible})
    for (nn, md), xy in coords.items():
        co[f"x_k{nn}_md{md}"] = xy[:, 0]
        co[f"y_k{nn}_md{md}"] = xy[:, 1]
    co.to_csv(OUT / "gallery_coords.csv.gz", index=False)

    sizes = 1 + 5 * np.sqrt(lab["n_cells"].to_numpy())
    gallery(coords, {k: gmets[k]["cell_state:GC"] for k in coords}, lab["cell_state:GC"].to_numpy(float),
            sizes, "gallery_gc.png", "viridis", "Fraction GC cells", 0, 1)
    gallery(coords, {k: gmets[k]["cell_state:PB"] for k in coords}, lab["cell_state:PB"].to_numpy(float),
            sizes, "gallery_pb.png", "viridis", "Fraction PB cells", 0, 1)
    gallery(coords, {k: gmets[k]["cell_state:Memory"] for k in coords},
            lab["cell_state:Memory"].to_numpy(float), sizes, "gallery_memory.png", "viridis",
            "Fraction memory cells", 0, 1)
    gallery(coords, {k: gmets[k]["day"] for k in coords}, lab["day"].to_numpy(float),
            sizes, "gallery_day.png", "plasma", "Sampling day (donor-level)", 10, 42)
    gallery(coords, {k: gmets[k]["mutation_frequency"] for k in coords},
            lab["mutation_frequency"].to_numpy(float), sizes, "gallery_mutation.png", "magma",
            "Mean V mutation frequency (clipped at 0.06)", 0, 0.06)
    stamp("galleries written")

    (OUT / "verify.json").write_text(json.dumps(
        {"default_config": {"n_neighbors": 15, "min_dist": 0.1, "spread": 1.0, "random_state": 0},
         "reproduction_axis_corr": repro, "n_reliable_clones": int(len(eligible)),
         "feature_shape": list(feats.shape), "n_parameter_sets": len(grid),
         "note": "day and treated are donor-level: their permutation null is unstratified."}, indent=1))
    stamp(f"done -> {OUT}")


if __name__ == "__main__":
    main()
