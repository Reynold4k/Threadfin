#!/usr/bin/env python
"""Sweep the clone-map UMAP parameters on the **Dandelion** clones, so the layout can be re-chosen.

Why this exists. The swept layouts that the figures used (GSE286215 experiment 2 at n_neighbors 50 /
min_dist 0.5, GSE195673 at 50 / 0.3) were chosen on clones defined from the heavy chain alone. The clone
definition has since moved to Dandelion's paired heavy + light clones, which changes *which clones exist*
— so the right layout for the new clone set is an open question rather than something to inherit or to
discard. This script re-runs the sweep on the new clones and renders one gallery per dataset to choose
from by eye, with two numbers under each panel so the choice is not only by eye.

**faithfulness** — Spearman correlation between pairwise distances in the clone-profile feature space and
in the 2-D map, on 30,000 random pairs. How much of the real geometry survived the projection.

**compartment agreement** — the fraction of each clone's 15 nearest map neighbours that share its
dominant compartment, against a label-permutation null. This is the "coherence" of the picture: whether
germinal-centre, plasmablast, plasma-cell and memory clones land near their own kind. Reported as the
excess over the null, with an empirical p.

Features are rebuilt the way `run_case_study.py` builds them — cached expression embedding, Dandelion
clone ids, `clone_profiles(representation="kernel")`, clones reaching reliability 0.5 — so a setting
chosen here can be copied straight into `CONFIGS[...]["map_params"]`.

Usage:  python clone_map_sweep.py [--datasets ...] [--n-perm 200]
Output: case_studies/results/clone_map_sweep/<dataset>_gallery.png  +  sweep.csv
"""
from __future__ import annotations

import argparse
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
CACHE = ROOT / "internal_validation" / "algorithm_revision" / "embedding_cache"
CLONES = HERE / "results" / "dandelion_clones"
OUT = HERE / "results" / "clone_map_sweep"

NNS = [10, 15, 30, 50, 100, 200]
MDS = [0.0, 0.1, 0.3, 0.5, 0.8]

#: the figure's own vocabulary and colours, so the gallery reads like panel C
COMPARTMENT = {
    "GC": "germinal centre", "dark zone": "germinal centre", "light zone": "germinal centre",
    "Myc+ light zone": "germinal centre", "GC_Bcell": "germinal centre",
    "PB": "plasmablast", "PB-like": "plasmablast",
    "LNPC": "plasma cell", "plasma cell": "plasma cell", "plasma": "plasma cell",
    "Memory": "memory", "RMB": "memory", "Memory-like": "memory",
}
COL = {"germinal centre": "#2a78d6", "plasmablast": "#e34948", "plasma cell": "#4a3aa7",
       "memory": "#008300", "other": "#dcdbd3"}
ORDER = ["other", "memory", "plasma cell", "plasmablast", "germinal centre"]

#: settings the old figures used, framed in the gallery for reference
PREVIOUS = {"malaria_late": (50, 0.5), "ln_vaccine": (50, 0.3)}
DEFAULT = (15, 0.1)
SUFFIX = ""
MIN_CELLS = 1
#: clones captured once are drawn as triangles, so a reader can see at a glance how much of a
#: map rests on a single cell -- in these deposits that is 75 to 98 % of clones
SINGLETON_MARKER, EXPANDED_MARKER = "^", "o"


def continuity(feats, xy, rng, k=15, n_sub=4000):
    """Manifold *continuity*: do points that are neighbours in the profile space stay neighbours on
    the map? This is the formal dual of trustworthiness (which asks the reverse), and it is the
    quantity "how continuous does the map look" actually refers to. 1.0 is perfect.

    Computed on a random subsample because both metrics are O(n^2) in the number of points.
    """
    from sklearn.manifold import trustworthiness

    n = len(xy)
    idx = rng.choice(n, size=int(min(n_sub, n)), replace=False)
    f, x = np.asarray(feats)[idx], np.asarray(xy)[idx]
    kk = int(min(k, len(idx) // 3))
    # trustworthiness(A -> B) penalises points that are far in A but near in B; swapping the roles
    # gives continuity, i.e. penalising points that are near in the features but far on the map
    return float(trustworthiness(x, f, n_neighbors=kk))


def fragmentation(xy, k=6):
    """Fraction of clones in the largest visually connected piece of the map.

    UMAP always returns a kNN-connected graph, so connectivity is measured geometrically: two clones
    are linked when they are closer than twice the median nearest-neighbour distance. 1.0 means the
    map is one piece; 0.4 means the biggest piece holds only 40 % of the clones.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

    d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    r = 2.0 * float(np.median(d[:, 1]))
    g = radius_neighbors_graph(xy, radius=max(r, 1e-9), mode="connectivity")
    n_comp, lab = connected_components(g, directed=False)
    return float(np.bincount(lab).max() / len(xy)), int(n_comp)


def faithfulness(feats, xy, rng, n_pairs=30000):
    from scipy.stats import spearmanr

    n = len(xy)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_hi = np.linalg.norm(feats[i] - feats[j], axis=1)
    d_lo = np.linalg.norm(xy[i] - xy[j], axis=1)
    return float(spearmanr(d_hi, d_lo).statistic)


def _knn(xy, k=15):
    from sklearn.neighbors import NearestNeighbors

    kk = int(min(k, len(xy) - 1))
    return NearestNeighbors(n_neighbors=kk + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]


def agreement(xy, labels, rng, k=15, n_perm=200):
    """Fraction of map neighbours sharing a clone's compartment, and the excess over a label null."""
    lab = np.asarray(labels)
    ok = lab != "other"
    if ok.sum() < k + 1 or len(np.unique(lab[ok])) < 2:
        return np.nan, np.nan, np.nan
    idx = _knn(xy, k)
    obs = float((lab[idx][ok] == lab[ok][:, None]).mean())
    null = np.empty(n_perm)
    for p in range(n_perm):
        q = rng.permutation(lab)
        null[p] = (q[idx][ok] == q[ok][:, None]).mean()
    return obs, obs - float(null.mean()), float((null >= obs).mean())


def smoothness(xy, values, rng, k=15, n_perm=200):
    """For a measured *quantity* rather than a compartment: how well a clone's 15 map neighbours
    predict its value, against a value-permutation null. Same role as `agreement`, for the sorts that
    kept one compartment and so have nothing categorical to separate."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < k + 1 or np.nanstd(v[ok]) == 0:
        return np.nan, np.nan, np.nan
    idx = _knn(xy, k)

    def r2(x):
        pred = np.nanmean(np.where(np.isfinite(x[idx]), x[idx], np.nan), axis=1)
        m = ok & np.isfinite(pred)
        return 1 - np.sum((x[m] - pred[m]) ** 2) / np.sum((x[m] - x[m].mean()) ** 2)

    obs = float(r2(v))
    null = np.array([r2(rng.permutation(v)) for _ in range(n_perm)])
    return obs, obs - float(null.mean()), float((null >= obs).mean())


def features_for(dataset: str, min_cells: int = 2):
    """Clone-profile features and the dominant compartment per clone, on the Dandelion clones.

    Every profiled clone is kept: no reliability filter.
    """
    import anndata as ad

    import threadfin as tf

    a = ad.read_h5ad(CACHE / f"{dataset}_embedding.h5ad")
    a.obs_names = a.obs_names.astype(str)
    ids = pd.read_csv(CLONES / f"{dataset}.csv").set_index("cell")["clone_id"]
    a.obs["clone_id"] = ids.reindex(a.obs_names).values
    a = a[a.obs["clone_id"].notna()].copy()
    ctx = "donor" if "donor" in a.obs else None
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key=ctx, donor_key="donor",
                         representation="kernel", min_cells=min_cells, verbose=False)
    prof = a.uns["threadfin"]["profiles"]
    table, feats_all = prof["clone_table"], prof["features"]
    # No reliability gate: the clone map is a display of the clones that exist, and the 0.5 floor
    # removes 75-100% of the model-antigen datasets (see embed_clones).
    eligible = table.index
    feats = feats_all.loc[eligible].to_numpy()

    state = next((c for c in ("cell_state", "state", "fate", "leiden_state", "celltype") if c in a.obs), None)
    comp = (a.obs[state].astype(str).map(COMPARTMENT) if state else pd.Series(index=a.obs_names, dtype=object))
    tab = pd.crosstab(a.obs["clone_id"].astype(str), comp.fillna("other"))
    real = [k for k in ORDER if k != "other" and k in tab.columns]
    if real:
        frac = tab[real].div(tab[real].sum(axis=1).replace(0, np.nan), axis=0)
        dom = pd.Series("other", index=tab.index, dtype=object)
        hit = frac.max(axis=1).notna()
        dom[hit] = frac[hit].idxmax(axis=1)
    else:
        dom = pd.Series("other", index=tab.index, dtype=object)
    sizes = table.loc[eligible, "n_cells"].to_numpy(float)
    icc = a.uns["threadfin"]["profiles"]["variance_components"]["icc"]
    dom = dom.reindex(eligible).fillna("other")

    # a sort that kept one compartment has nothing categorical to separate (GSE287123 is GL7+ Fas+
    # germinal centre only, and its states are unnamed Leiden clusters). Use the measured reporter
    # instead, which is the quantity panel D paints on the same map.
    measured, mname = None, None
    if (dom == "other").all():
        for col, label in (("division_gate", "fraction mCherry-low (>= 6 divisions)"),
                           ("gfp", "fraction GFP+"), ("tissue", "fraction in blood")):
            if col in a.obs and a.obs[col].notna().any():
                top = a.obs[col].astype(str).value_counts().index[-1]
                frac = (a.obs.assign(_v=a.obs[col].astype(str).eq(top))
                        .groupby(a.obs["clone_id"].astype(str), observed=True)["_v"].mean())
                measured = frac.reindex(eligible).to_numpy(float)
                mname = label.replace("mCherry-low", top) if col == "division_gate" else label
                break
    return feats, dom.to_numpy(), sizes, len(table), float(icc), measured, mname


def gallery(dataset, coords, stats, dom, sizes, n_total, icc, measured=None, mname=None):
    fig, axs = plt.subplots(len(NNS), len(MDS), figsize=(3.0 * len(MDS), 3.0 * len(NNS)))
    for i, nn in enumerate(NNS):
        for j, md in enumerate(MDS):
            ax = axs[i, j]
            xy, s = coords[(nn, md)], stats[(nn, md)]
            single = sizes <= 1
            if measured is None:
                for kind in ORDER:
                    for sel, mk, sc in ((single, SINGLETON_MARKER, 1.0),
                                        (~single, EXPANDED_MARKER, 1.3)):
                        m = (dom == kind) & sel
                        if m.any():
                            ax.scatter(xy[m, 0], xy[m, 1], s=np.clip(sizes[m] * sc, 2, 26),
                                       c=COL[kind], marker=mk, linewidths=0, alpha=0.85,
                                       rasterized=True)
            else:
                ok = np.isfinite(measured)
                for sel, mk, sc in ((single, SINGLETON_MARKER, 1.0), (~single, EXPANDED_MARKER, 1.3)):
                    m = ~ok & sel
                    if m.any():
                        ax.scatter(xy[m, 0], xy[m, 1], s=3, c="#ececea", marker=mk, linewidths=0,
                                   rasterized=True)
                    m = ok & sel
                    if m.any():
                        im = ax.scatter(xy[m, 0], xy[m, 1], s=np.clip(sizes[m] * sc, 2, 26),
                                        c=measured[m], cmap="viridis", vmin=0, vmax=1, marker=mk,
                                        linewidths=0, alpha=0.9, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            t = ax.set_title(f"k={nn}  min_dist={md}\ncontinuity {s['continuity']:.3f} · "
                             f"largest piece {s['largest_piece']:.2f}\nfaithful "
                             f"{s['faithfulness']:.3f} · agreement {s['agreement']:.3f}",
                             fontsize=7.2)
            t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2))
            tag = ("previously used" if PREVIOUS.get(dataset) == (nn, md)
                   else "package default" if (nn, md) == DEFAULT else None)
            if tag:
                col = "#c65359" if tag == "previously used" else "#52514e"
                for sp in ax.spines.values():
                    sp.set_color(col); sp.set_linewidth(2.2)
                ax.text(0.02, 0.02, tag, transform=ax.transAxes, fontsize=7, color=col,
                        fontweight="bold", va="bottom",
                        bbox=dict(boxstyle="round,pad=.25", fc="white", ec=col, lw=0.8))
    if measured is None:
        present = [k for k in ORDER if (dom == k).any()]
        handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=COL[k],
                              label=f"{k} ({(dom == k).sum()})") for k in reversed(present)]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
        colour_line = ("colour = compartment holding most of the clone's cells; "
                       "triangle = clone captured once, circle = two or more cells")
    else:
        fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.022, pad=0.03, aspect=50)
        colour_line = (f"colour = {mname} (measured; the embedding never saw it); "
                       "triangle = clone captured once, circle = two or more cells")
    fig.suptitle(
        f"{dataset} — clone map across UMAP parameters, on Dandelion paired heavy + light clones\n"
        f"all {len(dom):,} profiled clones, min_cells={MIN_CELLS} (no reliability filter); "
        f"clonal ICC {icc:.3f};  "
        f"rows = n_neighbors, columns = min_dist (spread 1, seed 0)\n"
        f"{colour_line}; point size = captured cells",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    fig.savefig(OUT / f"{dataset}{SUFFIX}_gallery.png", dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["malaria", "malaria_late", "ln_vaccine", "mouse_rbd", "mouse_np",
                             "gc_np_pc"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--min-cells", type=int, default=2,
                    help="cells a clone needs to be profiled; 1 includes singleton clones, which "
                         "are 75-98%% of clones in these deposits")
    a = ap.parse_args()
    global SUFFIX, MIN_CELLS
    MIN_CELLS = a.min_cells
    SUFFIX = f"_min{a.min_cells}"
    OUT.mkdir(parents=True, exist_ok=True)
    import umap

    t0 = time.time()
    rows = []
    for ds in a.datasets:
        print(f"\n=== {ds}  [{time.time() - t0:.0f}s]", flush=True)
        try:
            feats, dom, sizes, n_total, icc, measured, mname = features_for(ds, a.min_cells)
        except Exception as e:
            print(f"  skipped: {e}", flush=True)
            continue
        print(f"  {feats.shape[0]:,} reliable clones x {feats.shape[1]} features; ICC {icc:.3f}; "
              f"compartments {pd.Series(dom).value_counts().to_dict()}", flush=True)
        if feats.shape[0] < 15:
            print("  too few reliable clones to embed; skipped", flush=True)
            continue
        coords, stats = {}, {}
        for nn in NNS:
            for md in MDS:
                rng = np.random.default_rng(0)
                xy = umap.UMAP(n_neighbors=int(min(nn, len(feats) - 1)), min_dist=md, spread=1.0,
                               random_state=0).fit_transform(feats)
                f = faithfulness(feats, xy, rng)
                cont = continuity(feats, xy, rng)
                frag, ncomp = fragmentation(xy)
                ag, ex, p = (agreement(xy, dom, rng, n_perm=a.n_perm) if measured is None
                             else smoothness(xy, measured, rng, n_perm=a.n_perm))
                coords[(nn, md)] = xy
                stats[(nn, md)] = {"faithfulness": f, "agreement": ag, "excess": ex, "p": p,
                                   "continuity": cont, "largest_piece": frag}
                rows.append(dict(dataset=ds, n_neighbors=nn, min_dist=md, n_clones=len(feats),
                                 continuity=cont, largest_piece=frag, n_pieces=ncomp,
                                 faithfulness=f, agreement=ag, excess=ex, p_value=p))
                print(f"  k={nn:3d} md={md:.1f}  continuity {cont:.3f}  largest piece {frag:.2f} "
                      f"({ncomp} pieces)  faithful {f:.3f}  agreement {ag:.3f}", flush=True)
                np.save(OUT / f"{ds}{SUFFIX}_k{nn}_md{md}.npy", xy)
        gallery(ds, coords, stats, dom, sizes, n_total, icc, measured, mname)
        pd.DataFrame({"compartment": dom, "n_cells": sizes,
                      "measured": measured if measured is not None else np.nan}).to_csv(
            OUT / f"{ds}{SUFFIX}_points.csv", index=False)
        print(f"  wrote {OUT / f'{ds}{SUFFIX}_gallery.png'}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"sweep{SUFFIX}.csv", index=False)
    print(f"\nwrote {OUT / f'sweep{SUFFIX}.csv'}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
