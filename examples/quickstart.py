"""Threadfin quick start.

    python examples/quickstart.py --simulated   # simulated data with a known answer (~30 s, no download)
    python examples/quickstart.py               # public COVID-19 PBMC data (5,000 BCR+ cells, ~50 MB)

Both write ``threadfin_summary.txt`` and ``threadfin_overview.png`` to the
current directory.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

import threadfin as tf  # noqa: E402

COVID_URL = "https://exampledata.scverse.org/scirpy/stephenson2021_5k.h5mu"


def run_simulated():
    """A repertoire with four true clonal programmes, sampled in shifted batches."""
    adata = tf.sim.simulate_repertoire(random_state=0)
    return tf.run(adata, donor_key="donor", sample_key="context", basis="X_pca",
                  state_key="true_state", test=["true_programme"])


def run_covid(path: Path):
    """Stephenson et al. 2021 (Nat Med) COVID-19 PBMC, 5,000 B-lineage cells."""
    import awkward as ak
    import mudata as md

    if not path.exists():
        print(f"downloading {COVID_URL} ...")
        urllib.request.urlretrieve(COVID_URL, path)
    m = md.read_h5mu(path)
    adata = m["gex"].copy()
    adata.X = adata.layers["raw"].copy()   # raw counts
    adata.obs["donor"] = adata.obs["patient_id"].astype(str)
    adata.obs["sample"] = adata.obs["sample_id"].astype(str)

    # one productive heavy chain per cell from the AIRR modality
    airr = m["airr"].obsm["airr"]
    first = ak.firsts(airr[(airr["locus"] == "IGH") & ak.fill_none(airr["productive"], False)])
    bcr = pd.DataFrame({k: ak.to_list(first[k]) for k in ("v_call", "j_call", "junction", "c_call")},
                       index=m["airr"].obs_names).dropna(subset=["v_call", "junction"])

    return tf.run(adata, bcr=bcr, donor_key="donor", sample_key="sample",
                  state_key="full_clustering")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulated", action="store_true", help="use simulated data (no download)")
    args = parser.parse_args()

    result = run_simulated() if args.simulated else run_covid(Path("stephenson2021_5k.h5mu"))
    text = result.summary()
    print(text)
    Path("threadfin_summary.txt").write_text(text + "\n")
    if not result.programmes.empty:
        result.plot("threadfin_overview.png")
        print("wrote threadfin_summary.txt and threadfin_overview.png")


if __name__ == "__main__":
    main()
