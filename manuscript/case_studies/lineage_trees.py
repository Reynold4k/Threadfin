#!/usr/bin/env python
"""Somatic-hypermutation lineage trees for the model-antigen germinal-centre case studies.

A clone's heavy-chain V regions are written out with their germline as the
root, GCtree infers an abundance-aware parsimony tree per clone, and each
cell is then placed on its clone's tree. This gives an independent, sequence-
based ordering inside each clone: cells further from the germline carry more
somatic mutations and so arose later in the clone's history. Comparing the
sorted compartments of cells at different depths asks whether a cell's place
in its clone's family tree relates to what it was doing when it was sampled.

Usage:
    python lineage_trees.py prepare <dataset>     # write results/<dataset>/lineage/*.fasta
    bash run_gctree.sh <clone>                     # in that folder, per clone (see the script)
    python lineage_trees.py summarise <dataset>   # place cells on the trees and compare compartments

Datasets: gc_np_pc (sorted GC zones and plasma cells), mouse_np (division reporter).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent

# per dataset: the obs column holding the cell's sorted compartment, and the comparisons to make
GROUPS = {
    "gc_np_pc": {"column": "fate",
                 "comparisons": [(["plasma cell"], ["light zone", "Myc+ light zone", "dark zone"]),
                                 (["dark zone"], ["light zone", "Myc+ light zone"]),
                                 (["Myc+ light zone"], ["light zone"])]},
    "mouse_np": {"column": "division_gate",
                 "comparisons": [(["mCherry-low"], ["mCherry-high"])]},
    "mouse_rbd": {"column": "division_gate",
                  "comparisons": [(["mCherry-low"], ["mCherry-high"])]},
    "malaria": {"column": "cell_state",
                "comparisons": [(["PB"], ["GC"]), (["GC"], ["Activated", "Bystanders"]),
                                (["Memory"], ["GC", "PB"])]},
    "malaria_late": {"column": "cell_state", "comparisons": []},
    "bone_marrow_pc": {"column": "sorted_as",
                       "comparisons": [(["plasma cells"], ["memory B cells"])]},
    "flu_lung": {"column": "tissue", "comparisons": [(["Lung"], ["medLN"])]},
}


def _sequences(adata, dataset):
    """One row per cell: clone, sorted compartment, the V-region sequence, its germline and where it starts.

    ``start`` is the germline position of the first base, so that cells
    sequenced over different parts of the V region can be aligned to each
    other before a tree is built.
    """
    import threadfin as tf

    obs = adata.obs
    cols = {c: c[4:] for c in obs.columns if c.startswith("bcr_")}
    bcr = obs[list(cols)].rename(columns=cols)
    if "heavy_v_seq" in bcr.columns:                      # written by the GSE246382 loader
        seq, germ, start = bcr["heavy_v_seq"], bcr["heavy_v_germline"], bcr["heavy_v_start"]
    else:                                                  # AIRR alignments: locate the V region
        need = {"sequence_alignment", "germline_alignment"}
        if not need <= set(bcr.columns):
            raise SystemExit(f"{dataset}: the deposited contig files carry no V-region sequence "
                             f"(need {sorted(need)}), so no lineage tree can be built for it")
        win = tf.clones.v_region_windows(bcr, "sequence_alignment", "germline_alignment")
        def cut(col, start_col):
            return pd.Series([v[int(a):int(a + n)] if isinstance(v, str) and np.isfinite(n) else None
                              for v, a, n in zip(bcr[col], win[start_col], win["length"])], index=bcr.index)
        seq, germ, start = cut("sequence_alignment", "query_start"), cut("germline_alignment", "germline_start"), \
            win["germline_start"]
    out = pd.DataFrame({"clone": obs["clone_id"], "group": obs[GROUPS[dataset]["column"]],
                        "donor": obs["donor"], "seq": seq, "germline": germ, "start": start})
    return out.dropna(subset=["clone", "seq", "germline", "start"])


def prepare(dataset: str, min_cells: int = 3):
    from lineage import write_clone_fastas
    from run_case_study import prepare as prepare_dataset

    adata, *_ = prepare_dataset(dataset, stamp=lambda m: None)
    cells = _sequences(adata, dataset)
    out = HERE / "results" / dataset / "lineage"
    table = write_clone_fastas(cells, out, min_cells=min_cells)
    table.to_csv(out / "cells.csv")
    print(f"{dataset}: {table['clone'].nunique()} clones, {len(table)} cells -> {out}")
    print(f"next:  cd {out} && ls *.fasta | sed 's/.fasta//' | xargs -P 8 -n1 bash {HERE}/run_gctree.sh")


def summarise(dataset: str):
    from lineage import place_cells, within_clone_difference

    out = HERE / "results" / dataset / "lineage"
    cells = pd.read_csv(out / "cells.csv", index_col=0)
    tab = place_cells(out, cells)
    if tab.empty:
        raise SystemExit(f"no trees found in {out / 'trees'}; run run_gctree.sh first")
    tab.to_csv(out / "cell_tree_positions.csv", index=False)
    print(f"{tab['clone'].nunique()} trees, {len(tab)} cells placed")
    print(tab.groupby("group")[["depth", "mutations", "ancestral"]].mean().round(3).to_string())
    rows = [within_clone_difference(tab, "group", a, b, measure)
            for a, b in GROUPS[dataset]["comparisons"] for measure in ("depth", "ancestral")]
    res = pd.DataFrame(rows)
    res.to_csv(out / "within_clone_tree_comparisons.csv", index=False)
    print(res.round(3).to_string(index=False))


def tally(dataset: str):
    """Commit the per-clone tree outcome, so the figures do not need the trees themselves.

    ``ok`` means a tree was built; ``trivial`` means the clone's cells carry no
    sequence difference at all, so there is nothing for a tree to resolve.
    """
    out = HERE / "results" / dataset / "lineage"
    status = out / "status.txt"
    if not status.exists():
        raise SystemExit(f"{status} not found; run run_gctree.sh first")
    rows = [line.split() for line in status.read_text().splitlines() if len(line.split()) >= 2]
    counts = pd.Series([r[1] for r in rows]).value_counts()
    tab = pd.DataFrame({"dataset": dataset, "outcome": counts.index, "clones": counts.to_numpy()})
    tab["share"] = tab["clones"] / tab["clones"].sum()
    tab.to_csv(out / "tree_status.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    action, dataset = sys.argv[1], sys.argv[2]
    {"prepare": prepare, "summarise": summarise, "tally": tally}[action](dataset)
