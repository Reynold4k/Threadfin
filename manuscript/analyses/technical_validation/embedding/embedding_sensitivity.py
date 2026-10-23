#!/usr/bin/env python
"""Do the results depend on how the expression embedding is built?

The case study is prepared exactly as in `run_case_study.py` (same data, same Dandelion clone ids), and the
embedding is rebuilt with 2,000, 3,000 (the default) or 5,000 highly variable genes and 20, 30 (the default) or
50 principal components, and once with the receptor genes left in. For each embedding the analysis of the case
study is repeated: the clonal ICC on mean profiles centred on the mouse, tested by 200 permutations within
samples, and the variance of kernel clone profiles explained by the sorting label (1,000 permutations within
mice).

Usage: PYTHONHASHSEED=0 python embedding_sensitivity.py <dataset> <out_dir>
       python embedding_sensitivity.py --plot <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "case_studies"))

import threadfin as tf  # noqa: E402

LABELS = {"mouse_rbd": ("division_gate", "fraction:mCherry-low"), "gut_gc_reporter": ("gc_sort", "fraction:GC")}
SETTINGS = [(3000, 30, True), (2000, 30, True), (5000, 30, True), (3000, 20, True), (3000, 50, True),
            (3000, 30, False)]
NAMES = {"mouse_rbd": "GSE287123 RBD (label: recently divided)", "gut_gc_reporter": "GSE284622 gut (label: GC-fate sorted)"}


def main(ds: str, out: Path):
    import run_case_study as case

    out.mkdir(parents=True, exist_ok=True)
    adata, cfg, _, _ = case.prepare(ds)
    label, how = LABELS[ds]
    rows = []
    for genes, comps, exclude in SETTINGS:
        t0 = time.time()
        tf.pp.prepare_embedding(adata, batch_key=cfg["batch_key"], n_top_genes=genes, n_comps=comps,
                                exclude_receptor_genes=exclude, key_added="X_threadfin", verbose=False)
        ctx = cfg.get("context_key", "donor")
        tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=ctx, donor_key="donor", representation="mean",
                             min_cells=cfg.get("min_cells", 2), verbose=False)
        coh = tf.tl.clonal_coherence(adata, strata_key="sample", n_perm=200, verbose=False)
        tf.tl.clone_profiles(adata, basis="X_threadfin", context_key=ctx, donor_key="donor", representation="kernel",
                             min_cells=cfg.get("min_cells", 2), verbose=False)
        assoc = tf.tl.profile_association(adata, label, how=how, n_perm=1000, verbose=False)
        rows.append({"dataset": ds, "hvg": genes, "pcs": comps, "receptor_genes_excluded": exclude,
                     "clones": int(coh["n_clones"]), "icc": coh["icc"], "icc_shuffled": coh["null_mean"],
                     "icc_p": coh["p_value"], "label": label, "label_r2": assoc["r2"],
                     "label_excess_r2": assoc["excess_r2"], "label_p": assoc["p_value"],
                     "seconds": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(out / f"embedding_sensitivity_{ds}.csv", index=False)
    (out / f"run_{ds}.json").write_text(json.dumps({"threadfin": tf.__version__}, indent=1) + "\n")


def plot(out: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    from threadfin.plotting import INK_2, PALETTE

    tf.pl.set_style()
    tab = pd.concat([pd.read_csv(f) for f in sorted(out.glob("embedding_sensitivity_*.csv"))])
    order = ["default", "2,000\ngenes", "5,000\ngenes", "20 PCs", "50 PCs", "receptor\ngenes kept"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    width = 0.38
    datasets = [ds for ds in NAMES if ds in set(tab["dataset"])]      # the colours of the clone-definition figure
    for k, (colour, ds) in enumerate(zip(PALETTE, datasets)):
        sub = tab[tab["dataset"] == ds]
        x = np.arange(len(sub)) + (k - 0.5) * width
        axes[0].bar(x, sub["icc"], width, color=colour, edgecolor="white", linewidth=0.8, label=NAMES.get(ds, ds))
        axes[0].scatter(x, sub["icc_shuffled"], marker="_", s=60, color=INK_2, zorder=3)
        axes[1].bar(x, sub["label_excess_r2"], width, color=colour, edgecolor="white", linewidth=0.8)
    for ax, title in zip(axes, ["clonal ICC (dash: shuffled clones)", "label: excess R² of clone profiles"]):
        ax.set_xticks(np.arange(len(order)), order, fontsize=5.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("embedding (default: 3,000 variable genes, 30 PCs, receptor genes left out)", fontsize=5.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "embedding_sensitivity.png")
    fig.savefig(out / "figures" / "embedding_sensitivity.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(Path(sys.argv[2]))
    else:
        main(sys.argv[1], Path(sys.argv[2]))
