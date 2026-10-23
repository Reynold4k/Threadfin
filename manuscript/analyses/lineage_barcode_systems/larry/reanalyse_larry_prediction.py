"""Rerun the fixed LARRY prediction comparison from existing day-2-only features."""
import json
import anndata as ad
import pandas as pd
from reanalyse_clonotrace_larry import CACHE, OUT, predict_future

if __name__ == "__main__":
    predict_future(ad.read_h5ad(CACHE / "early_embedding.h5ad"),
                   pd.read_csv(OUT / "cell_metadata.csv.gz", index_col=0))
    path = OUT / "summary.json"
    summary = json.loads(path.read_text())
    summary["future_prediction"] = json.loads((OUT / "future_audit.json").read_text())
    path.write_text(json.dumps(summary, indent=2))
