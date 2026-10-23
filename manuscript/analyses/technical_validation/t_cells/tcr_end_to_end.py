#!/usr/bin/env python
"""T cells, end to end: scirpy's example dataset through `tf.run`, receptor data passed as the scirpy object.

Data: Wu et al. 2020 (Nature 579:274), 3,000 T cells from tumour, normal adjacent tissue and blood of 14
patients, as distributed with scirpy (`scirpy.datasets.wu2020_3k`). Clones are called by Threadfin's Dandelion
adapter from the scirpy MuData; the analysis is `tf.run` with the patient as donor and the sample as context,
testing the tissue of origin ('source'). The clone calls are also compared with and without Dandelion's contig
check.

Usage: PYTHONHASHSEED=0 python tcr_end_to_end.py <out_dir>    (needs scirpy and sc-dandelion)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import scirpy as ir

import threadfin as tf
from threadfin.dandelion import define_clones_dandelion, read_contigs


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mdata = ir.datasets.wu2020_3k()
    gex = mdata["gex"].copy()
    result = tf.run(gex, bcr=mdata, donor_key="patient", sample_key="sample", test=["source"], n_perm=200, n_jobs=4)
    (out / "summary.txt").write_text(result.summary() + "\n")
    (out / "methods.txt").write_text(result.methods() + "\n")
    sizes = gex.obs["clone_id"].dropna().value_counts()
    rec = {"threadfin": tf.__version__, "scirpy": ir.__version__, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
           "cells": int(gex.n_obs), "cells_with_clone": int(sizes.sum()), "clones": int(sizes.size),
           "clones_ge2": int((sizes >= 2).sum()), "receptor": result.settings["receptor"],
           "icc": result.coherence["icc"], "icc_shuffled": result.coherence["null_mean"],
           "icc_p": result.coherence["p_value"], "programmes": int(result.programmes.shape[0]),
           "source_r2": result.profile_tests["source"]["r2"], "source_p": result.profile_tests["source"]["p_value"],
           "minutes": round((time.time() - t0) / 60, 1)}

    # the same clone calls with and without Dandelion's contig check
    contigs = read_contigs(mdata)
    contigs["donor"] = contigs["cell_id"].map(mdata["gex"].obs["patient"].astype(str))
    rec["contigs_by_locus"] = {k: int(v) for k, v in contigs["locus"].value_counts().items()}
    rec["clone_calls"] = []
    for check in (False, True):
        ids = define_clones_dandelion(contigs, donor_key="donor", check_contigs=check, verbose=False)
        s = ids.value_counts()
        rec["clone_calls"].append({
            "check_contigs": check, "cells": int(ids.size), "clones": int(s.size), "clones_ge2": int((s >= 2).sum()),
            "cells_in_clones_ge2": int(s[s >= 2].sum()), "combined_ids": ids.attrs["n_cells_with_combined_id"],
            "combined_two_beta": ids.attrs["n_cells_with_combined_id_two_heavy"],
            "contigs_set_aside": ids.attrs["n_contigs_ambiguous"]})
    (out / "tcr_end_to_end.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
