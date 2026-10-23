#!/usr/bin/env python
"""Paired heavy + light chain clones of GSE203132.

Figure 3a-f defined clones on the heavy chain, from the deposited AIRR table (one heavy chain per cell). Here the
same heavy-chain rows are kept and the light chains of the same cells are added from the IgBLAST annotation of
the deposited contigs (annotate_contigs.sh), so the only change is the light chain. Clones are called with
Dandelion through ``threadfin.tl.define_clones_dandelion`` (heavy and light chain, junction identity 0.85,
within each mouse, cells without a heavy chain left without a clone), as for the other datasets of the
manuscript. The deposit holds no UMI counts for the light chains; Dandelion's clone calls do not use them.

As a check on the annotation, the heavy chains are annotated by IgBLAST too and compared with the deposit.

Usage: PYTHONHASHSEED=0 python paired_clones.py <igblast_dir> <GSE203132_10X_clone-pass_germ-pass.tsv.gz> <out_dir>
       (needs sc-dandelion; <igblast_dir> holds the <mouse>_airr.tsv files written by annotate_contigs.sh)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

import threadfin as tf

HERE = Path(__file__).resolve().parent
CELLS = HERE.parents[1] / "case_studies" / "results" / "spont_gc" / "cells.csv.gz"
KEEP = ["sequence_id", "cell", "donor", "locus", "v_call", "d_call", "j_call", "junction", "junction_aa",
        "productive", "umi_count"]


def gene(call):
    return call.astype(str).str.split(",").str[0].str.split("*").str[0]


def main(igblast_dir: Path, deposit: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    donor = pd.read_csv(CELLS, index_col=0, usecols=lambda c: c in ("Unnamed: 0", "donor"))["donor"].astype(str)
    donor.index = donor.index.astype(str)

    heavy = pd.read_csv(deposit, sep="\t", low_memory=False)
    heavy = heavy.loc[:, ~heavy.columns.duplicated()]
    heavy["cell"] = heavy["mid"].astype(str) + "|" + heavy["barcode_10x"].astype(str)
    heavy = heavy[heavy["cell"].isin(donor.index)].copy()

    frames = []
    for f in sorted(igblast_dir.glob("M*_airr.tsv")):
        mouse = f.name.split("_")[0]
        t = pd.read_csv(f, sep="\t", low_memory=False)
        t["cell"] = mouse + "|" + t["sequence_id"].astype(str).str.split("_contig").str[0]
        frames.append(t)
    igb = pd.concat(frames, ignore_index=True)
    igb = igb[igb["cell"].isin(donor.index)].copy()
    igb["productive"] = igb["productive"].astype(str).str.upper().isin(["T", "TRUE"]).map({True: "T", False: "F"})

    # annotation check: the deposited heavy chains against IgBLAST's annotation of the same contigs
    key = ["cell", "sequence_id"]
    h_igb = igb[igb["locus"].eq("IGH")].set_index(key)
    both = heavy.set_index(key).join(h_igb[["junction_aa", "v_call", "j_call"]], rsuffix="_igblast", how="inner")
    check = {"deposited_heavy_chains": int(len(heavy)), "found_in_igblast": int(len(both)),
             "same_junction_aa": float((both["junction_aa"] == both["junction_aa_igblast"]).mean()),
             "same_v_gene": float((gene(both["v_call"]) == gene(both["v_call_igblast"])).mean()),
             "same_j_gene": float((gene(both["j_call"]) == gene(both["j_call_igblast"])).mean())}
    print("annotation check:", json.dumps(check), flush=True)

    light = igb[igb["locus"].isin(["IGK", "IGL"]) & igb["productive"].eq("T")
                & igb["cell"].isin(set(heavy["cell"]))].copy()
    light["umi_count"] = 1
    contigs = pd.concat([heavy.assign(productive=heavy["productive"].astype(str)), light], ignore_index=True)
    contigs["donor"] = contigs["cell"].map(donor)
    contigs["sequence_id"] = contigs["cell"].str.split("|").str[0] + "_" + contigs["sequence_id"].astype(str)
    contigs = contigs[KEEP]
    n_light = light.groupby("cell").size()
    chains = {"cells_with_heavy_chain": int(heavy["cell"].nunique()),
              "with_light_chain": int(n_light.size), "with_one_light_chain": int((n_light == 1).sum()),
              "with_two_or_more_light_chains": int((n_light >= 2).sum()),
              "light_chain_loci": light["locus"].value_counts().to_dict()}
    print("chains:", json.dumps(chains), flush=True)
    contigs.to_csv(out / "gse203132_paired_contigs.tsv.gz", sep="\t", index=False)

    ids = tf.tl.define_clones_dandelion(contigs, cell_key="cell", donor_key="donor", identity=0.85)
    ids.rename("clone_id").rename_axis("cell").to_frame().to_csv(out / "spont_gc_paired_clones.csv")
    meta = {"annotation_check": check, "chains": chains, "clone_call": dict(ids.attrs)}
    (out / "spont_gc_paired_clones.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}; wrote {out / 'spont_gc_paired_clones.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
