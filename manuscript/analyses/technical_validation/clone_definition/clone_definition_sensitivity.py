#!/usr/bin/env python
"""Do the results depend on how clones are called?

Clones are called with Dandelion at junction identities from 0.75 to 1.0 (0.85 in the manuscript), and at 0.85
also after Dandelion's own contig check (`check_contigs=True`: cells with several chains of one kind keep the one
that dominates in UMIs; cells without a dominant chain, often doublets, are set aside). Without the check, a cell
with two productive chains of one kind gets a compound id ("a|b") and so usually forms a clone of its own; the
number of such cells is recorded. For each clone definition the analysis of the case study is repeated on its
cached embedding, with the same settings: the clonal ICC from mean profiles centred on the mouse, tested by
permutation within samples (200 permutations), and the variance of kernel clone profiles explained by the
sorting label of the dataset (profile association, 1,000 permutations within mice). The number of clones and of
cells in clones of two or more is recorded too.

Datasets: GSE287123 RBD (Figure 4; label: recently divided fraction) and GSE284622 gut (Figure 3g-j; label:
GC-fate sorted fraction).

Usage: PYTHONHASHSEED=0 python clone_definition_sensitivity.py <out_dir>    (needs sc-dandelion)
       python clone_definition_sensitivity.py --plot <out_dir>               (figure from the saved table)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd

import threadfin as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "case_studies"))
from dandelion_clones import CACHE, contigs_for  # noqa: E402

SETTINGS = {"mouse_rbd": ("division_gate", "fraction:mCherry-low"), "gut_gc_reporter": ("gc_sort", "fraction:GC")}
IDENTITIES = (0.75, 0.80, 0.85, 0.90, 0.95, 1.0)


def labels(ds):
    """Cell annotations of the case study (the cached embedding keeps only donor and sample)."""
    cells = pd.read_csv(HERE.parents[2] / "case_studies" / "results" / ds / "cells.csv.gz", index_col=0,
                        low_memory=False)
    cells.index = cells.index.astype(str)
    return cells


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, (label, how) in SETTINGS.items():
        emb = ad.read_h5ad(CACHE / f"{ds}_embedding.h5ad")
        emb.obs_names = emb.obs_names.astype(str)
        if label not in emb.obs:                           # not kept in the cache: take it from the case study
            emb.obs[label] = labels(ds)[label].reindex(emb.obs_names).to_numpy()
        contigs = contigs_for(ds)
        contigs = contigs[contigs["cell"].astype(str).isin(emb.obs_names)].copy()
        contigs["donor"] = contigs["cell"].astype(str).map(emb.obs["donor"].astype(str))
        runs = [(i, False) for i in IDENTITIES] + [(0.85, True)]
        for identity, check in runs:
            t0 = time.time()
            ids = tf.tl.define_clones_dandelion(contigs, cell_key="cell", donor_key="donor", identity=identity,
                                                check_contigs=check, verbose=False)
            a = emb.copy()
            a.obs["clone_id"] = ids.reindex(a.obs_names).to_numpy()
            # as in run_case_study.py: ICC on mean profiles, tested within samples; labels on kernel profiles
            tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                                 representation="mean", min_cells=2, verbose=False)
            coh = tf.tl.clonal_coherence(a, strata_key="sample", n_perm=200, verbose=False)
            tf.tl.clone_profiles(a, basis="X_threadfin", context_key="donor", donor_key="donor",
                                 representation="kernel", min_cells=2, verbose=False)
            assoc = tf.tl.profile_association(a, label, how=how, n_perm=1000, verbose=False)
            sizes = ids.value_counts()
            compound = ids.astype(str).str.split("|").str.len() > 2          # donor|a|b: two chains of one kind
            rows.append({"dataset": ds, "identity": identity, "check_contigs": check, "cells_with_clone": int(len(ids)),
                         "clones": int(sizes.size), "clones_ge2": int((sizes >= 2).sum()),
                         "cells_in_clones_ge2": int(sizes[sizes >= 2].sum()), "largest": int(sizes.max()),
                         "cells_with_compound_id": int(compound.sum()),
                         "contigs_set_aside": ids.attrs.get("n_contigs_ambiguous", 0), "icc": coh["icc"],
                         "icc_shuffled": coh["null_mean"], "icc_p": coh["p_value"],
                         "label": label, "label_r2": float(assoc["r2"]), "label_excess_r2": float(assoc["excess_r2"]),
                         "label_p": float(assoc["p_value"]), "seconds": round(time.time() - t0, 1)})
            print(json.dumps(rows[-1]), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "clone_definition_sensitivity.csv", index=False)
    (out / "run.json").write_text(json.dumps({"python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                                             "threadfin": tf.__version__}, indent=1) + "\n")


NAMES = {"mouse_rbd": "GSE287123 RBD (label: recently divided)", "gut_gc_reporter": "GSE284622 gut (label: GC-fate sorted)"}


def plot(out: Path):
    """Clones, clonal ICC and label effect against the junction identity; the contig check as open markers."""
    import matplotlib.pyplot as plt

    tf.pl.set_style()
    from threadfin.plotting import INK_2, PALETTE

    tab = pd.read_csv(out / "clone_definition_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
    panels = [("clones_ge2", "clones with ≥ 2 cells"), ("icc", "clonal ICC (dashed: shuffled clones)"),
              ("label_excess_r2", "label: excess R² of clone profiles")]
    for ax, (col, title) in zip(axes, panels):
        for colour, (ds, sub) in zip(PALETTE, tab.groupby("dataset", sort=False)):
            plain, checked = sub[~sub["check_contigs"]], sub[sub["check_contigs"]]
            ax.plot(plain["identity"], plain[col], "-o", color=colour, ms=4, lw=1.2, label=NAMES.get(ds, ds))
            ax.plot(checked["identity"], checked[col], "o", ms=7, mfc="none", mec=colour, mew=1.2)
            if col == "icc":
                ax.plot(plain["identity"], plain["icc_shuffled"], "--", color=colour, lw=1.0)
        ax.axvline(0.85, color=INK_2, lw=0.5, ls=":")
        ax.set_xlabel("junction identity")
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec=INK_2, mew=1.2))
    labels.append("identity 0.85 after Dandelion's contig check")
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "clone_definition_sensitivity.png")
    fig.savefig(out / "figures" / "clone_definition_sensitivity.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]))
