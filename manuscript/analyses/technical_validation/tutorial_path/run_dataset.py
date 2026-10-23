#!/usr/bin/env python
"""Follow Tutorial 2 on one public dataset, as a user would, and record everything that goes wrong.

The user's part is reading their own files: raw counts with donor and sample columns (the case-study loaders,
`case_studies/datasets.py`) and the contig-level BCR table as deposited, with the barcodes renamed to match the
expression data (Tutorial 2, section 4; `case_studies/dandelion_clones.contigs_for`). Everything after that is
the documented call, with every setting at its default:

    result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key=..., test=[...], n_jobs=8)
    print(result.summary()); result.methods(); result.plot(...); result.clones.to_csv(...); adata.write_h5ad(...)

LARRY has lineage barcodes instead of a BCR; it follows Tutorial 2, section 6 (clone ids defined elsewhere) with
the published embedding (`basis=`). Every exception, every warning and the run time are written to
<out>/<dataset>.json.

Usage: PYTHONHASHSEED=0 python run_dataset.py <dataset> <out_dir>
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[2]
sys.path.insert(0, str(MANUSCRIPT / "case_studies"))
sys.path.insert(0, str(MANUSCRIPT / "benchmarking" / "scripts"))

import threadfin as tf  # noqa: E402

# the settings of the case studies (run_case_study.CONFIGS): context, batch, annotation and test columns
SETTINGS = {
    "ln_vaccine": dict(sample_key="donor", batch_key="donor", state_key="state",
                       test=["spike_binding", "tissue"], time_key="timepoint"),
    "flu": dict(sample_key="donor", batch_key="donor", test=["timepoint"], time_key="timepoint"),
    "tonsil": dict(sample_key="donor", batch_key="donor", state_key="state", test=[]),
    "ebv": dict(sample_key="timepoint", test=["gfp"], time_key="timepoint"),
    "mouse_np": dict(sample_key="donor", test=["division_gate"], time_key="division_gate"),
    "mouse_rbd": dict(sample_key="donor", test=["rbd_bait", "division_gate", "zone_gate"],
                      time_key="division_gate"),
    "gc_np_pc": dict(sample_key="donor", state_key="fate", test=["fate"]),
    "flu_lung": dict(sample_key="donor", test=["ha_binding", "tissue"], time_key="tissue"),
    "bone_marrow_pc": dict(sample_key="donor", batch_key="donor", test=["sorted_as", "tissue", "antigen"],
                           time_key="tissue"),
    "malaria": dict(sample_key="donor", state_key="cell_state", test=["cell_state"]),
    "malaria_late": dict(sample_key="donor", state_key="cell_state", test=["cell_state"]),
    "gut_gc_reporter": dict(sample_key="donor", state_key="celltype", test=["gc_sort", "tissue"],
                            time_key="tissue"),
    "covid_bcell": dict(sample_key="donor", batch_key="donor", test=["antigen_call"], time_key="visit"),
    "spont_gc": dict(sample_key="donor", test=[]),
    "larry_d2": dict(sample_key="sample", state_key="state", test=[]),
    "larry_d46": dict(sample_key="sample", state_key="state", test=[]),
}


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def user_inputs(ds):
    """What the user prepares before calling Threadfin."""
    if ds.startswith("larry"):
        import anndata as ad
        import bench_io

        x, cells = bench_io.load(ds, "full")
        adata = ad.AnnData(X=np.zeros((len(cells), 0), dtype=np.float32), obs=cells.drop(columns="clone"))
        adata.obs["clone_id"] = cells["clone"].astype(object).where(cells["clone"].notna(), None).to_numpy()
        adata.obsm["X_published"] = np.asarray(x, dtype=np.float32)
        return adata, None, {"basis": "X_published"}
    import datasets
    import dandelion_clones

    adata, bcr = datasets.LOADERS[ds]()
    if "c_call" in bcr:                       # the isotype from the user's BCR table, to test (Tutorial 2, section 7)
        adata.obs["isotype"] = tf.clones.isotype_class(bcr["c_call"].reindex(adata.obs_names)).to_numpy()
    contigs = dandelion_clones.contigs_for(ds)
    contigs["cell_id"] = contigs["cell"].astype(str)        # barcodes harmonised as in Tutorial 2, section 4
    contigs = contigs.drop(columns=[c for c in ("cell", "barcode") if c in contigs.columns])
    return adata, contigs, {}


def main(ds, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": ds, "threadfin": tf.__version__, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
           "steps": {}, "warnings": [], "error": None}
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            adata, contigs, extra = user_inputs(ds)
            rec["steps"]["user_inputs"] = round(time.time() - t0, 1)
            rec["cells"] = int(adata.n_obs)
            if contigs is not None:
                rec["contigs"] = int(len(contigs))
                rec["contigs_matching_a_cell"] = float(contigs["cell_id"].isin(adata.obs_names.astype(str)).mean())
            s = SETTINGS[ds]
            test = [c for c in s.get("test", []) if c in adata.obs] + (["isotype"] if "isotype" in adata.obs else [])
            rec["test"] = test
            kw = dict(donor_key="donor", sample_key=s["sample_key"], batch_key=s.get("batch_key"),
                      state_key=s.get("state_key"), test=test or None, time_key=s.get("time_key"), n_jobs=8, **extra)
            if contigs is not None:
                kw["bcr"] = contigs
            t1 = time.time()
            result = tf.run(adata, **kw)
            rec["steps"]["threadfin_run"] = round(time.time() - t1, 1)
            summary = result.summary()
            (out / f"{ds}_summary.txt").write_text(summary + "\n")
            (out / f"{ds}_methods.txt").write_text(result.methods() + "\n")
            t2 = time.time()
            result.plot(str(out / f"{ds}_overview.png"))
            result.clones.to_csv(out / f"{ds}_clones.csv")
            h5 = out / f"{ds}_result.h5ad"
            adata.write_h5ad(h5)
            rec["h5ad_gb"] = round(h5.stat().st_size / 1024 ** 3, 2)
            h5.unlink()                                   # written and readable is what is checked; not kept
            rec["steps"]["plot_and_save"] = round(time.time() - t2, 1)
            sizes = adata.obs["clone_id"].dropna().astype(str).value_counts()
            rec.update({"clones": int(sizes.size), "clones_ge2": int((sizes >= 2).sum()),
                        "icc": float(result.coherence["icc"]), "icc_shuffled": float(result.coherence["null_mean"]),
                        "p_value": float(result.coherence["p_value"]), "programmes": int(result.programmes.shape[0]),
                        "skipped": result.skipped,
                        "map": {k: adata.uns["threadfin"]["profiles"]["map_params"].get(k)
                                for k in ("n_neighbors", "min_dist")}})
        except Exception as e:                            # noqa: BLE001 - the point is to record any failure
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
    seen = {}
    for w in caught:
        key = (w.category.__name__, str(w.message)[:300], Path(w.filename).name)
        seen[key] = seen.get(key, 0) + 1
    rec["warnings"] = [{"category": c, "message": m, "file": f, "count": n} for (c, m, f), n in seen.items()]
    rec["seconds_total"] = round(time.time() - t0, 1)
    rec["peak_rss_gb"] = round(peak_gb(), 2)
    (out / f"{ds}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("dataset", "error", "cells", "clones_ge2", "icc", "programmes",
                                              "seconds_total", "peak_rss_gb")}, default=str))
    print(f"{len(rec['warnings'])} distinct warnings")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
