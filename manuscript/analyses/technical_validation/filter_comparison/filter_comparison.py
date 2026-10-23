#!/usr/bin/env python
"""Three ways to handle thinly observed clones, drawn on the same real clone map.

The question is what to do with a clone whose profile rests on one or two cells. Three answers:

* **no filter, unweighted** — cluster every clone, all edges equal
* **no filter, reliability-weighted** — cluster every clone, but scale each graph edge by
  ``sqrt(rel_i * rel_j)`` so a thin profile pulls weakly
* **threshold 0.5** — cluster only the clones that reach reliability 0.5, which is what the package
  used to do and what removed 75-100 % of the model-antigen datasets

All three are drawn on **one** clone map, laid out once with no filter at all, so the only thing that
changes between panels is the programme labelling. The last panel of each row shows the measured
compartment on the same coordinates, which is the thing the programmes are supposed to recover: a
programme split that does not line up with it is a split of noise.

Usage:  python filter_comparison.py [--datasets gc_np_pc mouse_rbd ...] [--n-boot 20]
Output: case_studies/results/filter_comparison/<dataset>.png  +  summary.csv
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
OUT = HERE / "results" / "filter_comparison"

COMPARTMENT = {
    "GC": "germinal centre", "dark zone": "germinal centre", "light zone": "germinal centre",
    "Myc+ light zone": "germinal centre",
    "PB": "plasmablast", "PB-like": "plasmablast",
    "LNPC": "plasma cell", "plasma cell": "plasma cell",
    "Memory": "memory", "RMB": "memory", "Memory-like": "memory",
}
COL = {"germinal centre": "#2a78d6", "plasmablast": "#e34948", "plasma cell": "#4a3aa7",
       "memory": "#008300", "other": "#dcdbd3"}
ORDER = ["other", "memory", "plasma cell", "plasmablast", "germinal centre"]

SETTINGS = [
    ("no filter, unweighted", dict(min_reliability=0.0, weight_by_reliability=False)),
    ("no filter, reliability-weighted", dict(min_reliability=0.0, weight_by_reliability=True)),
    ("threshold: reliability >= 0.5", dict(min_reliability=0.5, weight_by_reliability=False)),
]


def prepared(dataset: str):
    import anndata as ad

    import threadfin as tf

    a = ad.read_h5ad(CACHE / f"{dataset}_embedding.h5ad")
    a.obs_names = a.obs_names.astype(str)
    ids = pd.read_csv(CLONES / f"{dataset}.csv").set_index("cell")["clone_id"]
    a.obs["clone_id"] = ids.reindex(a.obs_names).values
    a = a[a.obs["clone_id"].notna()].copy()
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                         representation="kernel", verbose=False)
    tf.tl.embed_clones(a, min_reliability=0.0, verbose=False)   # one layout for every panel
    return a


def compartments(a) -> pd.Series:
    state = next((c for c in ("cell_state", "state", "fate", "leiden_state") if c in a.obs), None)
    comp = (a.obs[state].astype(str).map(COMPARTMENT) if state
            else pd.Series(index=a.obs_names, dtype=object))
    tab = pd.crosstab(a.obs["clone_id"].astype(str), comp.fillna("other"))
    real = [k for k in ORDER if k != "other" and k in tab.columns]
    if not real:
        return pd.Series("other", index=tab.index, dtype=object)
    frac = tab[real].div(tab[real].sum(axis=1).replace(0, np.nan), axis=0)
    dom = pd.Series("other", index=tab.index, dtype=object)
    hit = frac.max(axis=1).notna()
    dom[hit] = frac[hit].idxmax(axis=1)
    return dom


def agreement_with_compartment(labels: pd.Series, dom: pd.Series) -> float:
    """Adjusted Rand index between the programme split and the measured compartments."""
    from sklearn.metrics import adjusted_rand_score

    both = labels.notna() & dom.reindex(labels.index).ne("other")
    if both.sum() < 10:
        return np.nan
    return float(adjusted_rand_score(dom.reindex(labels.index)[both], labels[both]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["gc_np_pc", "mouse_rbd"])
    ap.add_argument("--n-boot", type=int, default=20)
    a_ = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import threadfin as tf

    rows, t0 = [], time.time()
    for ds in a_.datasets:
        print(f"\n=== {ds}  [{time.time() - t0:.0f}s]", flush=True)
        a = prepared(ds)
        table = a.uns["threadfin"]["profiles"]["clone_table"]
        dom = compartments(a).reindex(table.index).fillna("other")
        xy = table[["x", "y"]].to_numpy()
        sizes = np.clip(table["n_cells"].to_numpy(float) * 1.4, 2.5, 30)
        rel = table["reliability"]
        print(f"  {len(table):,} clones, reliability median {rel.median():.2f}, "
              f">=0.5 in {(rel >= 0.5).sum():,}", flush=True)

        fig, axs = plt.subplots(1, 4, figsize=(19, 5.0))
        for ax, (name, kw) in zip(axs, SETTINGS):
            try:
                summ = tf.tl.find_programmes(a, n_boot=a_.n_boot, verbose=False, **kw)
                lab = a.uns["threadfin"]["profiles"]["clone_table"]["clone_programme"]
                n_prog, stab = summ.shape[0], summ["stability"].round(2).tolist()
            except ValueError as e:
                ax.text(0.5, 0.5, f"refused:\n{str(e)[:70]}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#c65359")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=11)
                rows.append(dict(dataset=ds, setting=name, n_programmes=0, error=str(e)[:90]))
                continue
            ari = agreement_with_compartment(lab, dom)
            cats = [c for c in pd.Categorical(lab).categories]
            pal = plt.get_cmap("tab10")
            miss = lab.isna().to_numpy()
            ax.scatter(xy[miss, 0], xy[miss, 1], s=3, c="#ececea", linewidths=0, rasterized=True)
            for i, c in enumerate(cats):
                m = (lab == c).to_numpy()
                ax.scatter(xy[m, 0], xy[m, 1], s=sizes[m], color=pal(i % 10), linewidths=0,
                           alpha=0.85, rasterized=True, label=f"{c} ({m.sum():,})")
            ax.legend(frameon=False, fontsize=7.5, loc="best", markerscale=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}\n{n_prog} programmes · {int(lab.notna().sum()):,} clones labelled\n"
                         f"ARI vs measured compartment = {ari:.3f}", fontsize=10)
            rows.append(dict(dataset=ds, setting=name, n_programmes=n_prog,
                             n_labelled=int(lab.notna().sum()), ari=ari, stability=str(stab)))
            print(f"  {name:34s} {n_prog} programmes, ARI {ari:.3f}, stability {stab}", flush=True)

        ax = axs[3]
        for k in ORDER:
            m = (dom == k).to_numpy()
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=sizes[m], c=COL[k], linewidths=0, alpha=0.85,
                           rasterized=True, label=f"{k} ({m.sum():,})")
        ax.legend(frameon=False, fontsize=7.5, loc="best", markerscale=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("measured compartment\n(what the programmes should recover)", fontsize=10)

        fig.suptitle(f"{ds} — same clone map, three ways of handling thinly observed clones\n"
                     f"{len(table):,} clones (every clone profiled, min_cells=1); "
                     f"median reliability {rel.median():.2f}; "
                     f"{(rel >= 0.5).sum():,} would survive a 0.5 threshold", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(OUT / f"{ds}.png", dpi=115)
        plt.close(fig)
        print(f"  wrote {OUT / f'{ds}.png'}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
