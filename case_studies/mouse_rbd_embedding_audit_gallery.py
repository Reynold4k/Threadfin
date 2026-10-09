#!/usr/bin/env python
"""Gallery of mouse_rbd clone-profile UMAPs across parameter settings.

Rebuilds the committed family features (identical to run_case_study.py), then
renders the clone map for 12 representative UMAP parameter sets, coloured by
measured division history, measured mutation history, or the LZ-DZ programme
axis. Each panel is annotated with the linear-gradient R^2 and the local
association (excess kNN R^2 vs donor-stratified null) from
mouse_rbd_embedding_audit.assoc_metrics. The committed configuration
(n_neighbors=15, min_dist=0.1, spread=1, seed=0) is framed in red.

Outputs to results/mouse_rbd_embedding_audit/:
  gallery_division.png, gallery_mutation.png, gallery_lzdz.png,
  gallery_coords.csv.gz
"""
from __future__ import annotations

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
OUT = HERE / "results" / "mouse_rbd_embedding_audit"

from mouse_rbd_embedding_audit import assoc_metrics  # noqa: E402

NNS = [5, 15, 50]
MDS = [0.0, 0.1, 0.5, 0.9]
COMMITTED = (15, 0.1)


def main():
    t0 = time.time()

    def stamp(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

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
    stamp("expression embedding + family profiles")
    tf.pp.prepare_embedding(adata, batch_key=None, key_added="X_threadfin", verbose=False)
    tf.tl.clone_profiles(adata, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", verbose=False)
    prof = adata.uns["threadfin"]["profiles"]
    table, feats_all = prof["clone_table"], prof["features"]
    eligible = table.index[table["reliability"] >= 0.5]
    feats = feats_all.loc[eligible].to_numpy()
    lab = pd.read_csv(HERE / "results" / "mouse_rbd" / "clone_table.csv", index_col=0).reindex(eligible)
    scores = pd.read_csv(HERE / "results" / "mouse_rbd" / "clone_gene_scores.csv", index_col=0).reindex(eligible)
    lab = lab.join(scores)
    donors = lab["donor"].astype(str).to_numpy()
    sizes = 1 + 5 * np.sqrt(lab["n_cells"].to_numpy())
    stamp(f"{len(eligible)} reliable clones; running {len(NNS) * len(MDS)} UMAP configurations")

    varr = {"division": lab["division_gate:mCherry-low"].to_numpy(float),
            "mutation": lab["mutation_frequency"].to_numpy(float),
            "lzdz": (lab["light zone"] - lab["dark zone / cycling"]).to_numpy(float)}
    rng = np.random.default_rng(0)
    coords, mets = {}, {v: {} for v in varr}
    for nn in NNS:
        for md in MDS:
            xy = umap.UMAP(n_neighbors=nn, min_dist=md, spread=1.0, random_state=0).fit_transform(feats)
            coords[(nn, md)] = xy
            for vname, v in varr.items():
                mets[vname][(nn, md)] = assoc_metrics(xy, v, donors, rng)
            stamp(f"  k={nn} md={md} done")

    co = pd.DataFrame({"clone_id": eligible})
    for (nn, md), xy in coords.items():
        co[f"x_k{nn}_md{md}"] = xy[:, 0]
        co[f"y_k{nn}_md{md}"] = xy[:, 1]
    co.to_csv(OUT / "gallery_coords.csv.gz", index=False)

    for vname, cmap, clabel in [("division", "viridis", "Fraction mCherry-low (≥6 divisions)"),
                                ("mutation", "magma", "Mean V mutation frequency (clipped at 0.03)"),
                                ("lzdz", "RdBu_r", "LZ − DZ programme score (2–98 pct clipped)")]:
        v = varr[vname]
        if vname == "lzdz":
            v = np.clip(v, *np.nanpercentile(v, [2, 98]))
        kw = (dict(vmin=0, vmax=1) if vname == "division"
              else dict(vmin=0, vmax=.03) if vname == "mutation"
              else dict(vmin=-np.abs(v).max(), vmax=np.abs(v).max()))
        fig, axs = plt.subplots(len(NNS), len(MDS), figsize=(13.5, 10.8))
        for i, nn in enumerate(NNS):
            for j, md in enumerate(MDS):
                ax = axs[i, j]
                xy = coords[(nn, md)]
                m = mets[vname][(nn, md)]
                im = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=sizes, cmap=cmap,
                                edgecolors="white", linewidths=.15, **kw)
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
        fig.suptitle(f"mouse_rbd clone map across UMAP parameters — {clabel}\n"
                     f"381 reliable clones; rows = n_neighbors, columns = min_dist (spread=1, seed=0); "
                     f"red frame = configuration used in Figure 2", fontsize=10)
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        fig.colorbar(im, ax=axs, orientation="horizontal", fraction=.03, pad=.02, aspect=40)
        fig.savefig(OUT / f"gallery_{vname}.png", dpi=110)
        plt.close(fig)
        stamp(f"gallery_{vname}.png written")
    stamp("done")


if __name__ == "__main__":
    main()
