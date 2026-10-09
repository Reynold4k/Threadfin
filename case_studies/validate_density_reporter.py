"""Leave-one-mouse-out reporter validation with fully train-only RNA preprocessing.

Fixed, predeclared settings: 2,000 receptor-excluded HVGs, 30 PCs, 16 regions,
Dirichlet strength 2, ridge alpha 10, no label-based model/parameter selection.
Every representation gets log captured-cell count as an extra feature.
This tests division-gate readout, not mutation rate, affinity or future fate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import threadfin as tf
from datasets import LOADERS

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "algorithm_revision"


def run(name):
    OUT.mkdir(parents=True, exist_ok=True)
    a, _ = LOADERS[name]()
    source = HERE / "results" / name / "cells.csv.gz"
    cells = pd.read_csv(source, index_col=0)
    if len(cells) != a.n_obs or set(cells.index) != set(a.obs_names):
        raise ValueError("Saved clone definitions do not match the raw count input.")
    a.obs["clone_id"] = cells.clone_id.reindex(a.obs_names).to_numpy()
    obs = a.obs.copy()
    obs["donor"] = obs.donor.astype(str)
    obs["target"] = obs.division_gate.astype(str).eq("mCherry-low").astype(float)
    target = obs.dropna(subset=["clone_id"]).groupby("clone_id", observed=True).agg(
        target=("target", "mean"), donor=("donor", "first"), n_cells=("target", "size"))
    target.index = target.index.astype(str)
    predictions, scores, audits = [], [], []
    for donor in sorted(obs.donor.unique()):
        start = time.perf_counter()
        train_mask = obs.donor.ne(donor).to_numpy()
        test_mask = ~train_mask
        assert set(obs.loc[train_mask, "clone_id"].dropna()).isdisjoint(
            obs.loc[test_mask, "clone_id"].dropna())
        embedding = tf.pp.FrozenExpressionModel.fit(a[train_mask], n_top_genes=2000, n_comps=30)
        train_x, test_x = embedding.transform(a[train_mask]), embedding.transform(a[test_mask])
        train_obs, test_obs = obs.loc[train_mask], obs.loc[test_mask]
        train_valid = train_obs.clone_id.notna().to_numpy()
        test_valid = test_obs.clone_id.notna().to_numpy()
        train_cl = train_obs.loc[train_valid, "clone_id"].astype(str).to_numpy()
        test_cl = test_obs.loc[test_valid, "clone_id"].astype(str).to_numpy()
        density = tf.StateDensityModel.fit(train_x[train_valid], train_cl, n_states=16, prior_strength=2)
        train_density = density.transform(train_x[train_valid], train_cl)
        test_density = density.transform(test_x[test_valid], test_cl)
        train_mean = pd.DataFrame(train_x[train_valid], index=train_cl).groupby(level=0).mean()
        test_mean = pd.DataFrame(test_x[test_valid], index=test_cl).groupby(level=0).mean()
        train_ids = target.index[(target.donor != donor) & (target.n_cells >= 2)]
        test_ids = target.index[(target.donor == donor) & (target.n_cells >= 2)]
        train_size = np.log1p(target.loc[train_ids, "n_cells"].to_numpy())[:, None]
        test_size = np.log1p(target.loc[test_ids, "n_cells"].to_numpy())[:, None]
        representations = {
            "size_only": (np.empty((len(train_ids), 0)), np.empty((len(test_ids), 0))),
            "RNA_mean": (train_mean.loc[train_ids].to_numpy(), test_mean.loc[test_ids].to_numpy()),
            "state_counts": (train_density.raw_proportions.loc[train_ids].to_numpy(),
                             test_density.raw_proportions.loc[test_ids].to_numpy()),
            "state_posterior": (train_density.proportions.loc[train_ids].to_numpy(),
                                test_density.proportions.loc[test_ids].to_numpy()),
        }
        y_train, y_test = target.loc[train_ids, "target"].to_numpy(), target.loc[test_ids, "target"].to_numpy()
        for method, (xt, xv) in representations.items():
            predictor = make_pipeline(StandardScaler(), Ridge(alpha=10))
            predictor.fit(np.c_[xt, train_size], y_train)
            pred = predictor.predict(np.c_[xv, test_size]).clip(0, 1)
            for key, y, estimate in zip(test_ids, y_test, pred):
                predictions.append({"dataset": name, "heldout_mouse": donor, "clone": key,
                                    "n_cells": int(target.loc[key, "n_cells"]), "method": method,
                                    "observed": float(y), "predicted": float(estimate)})
            for min_cells in (2, 10):
                keep = (target.loc[test_ids, "n_cells"] >= min_cells).to_numpy()
                if keep.sum() < 3:
                    continue
                yy, pp = y_test[keep], pred[keep]
                scores.append({"dataset": name, "heldout_mouse": donor, "method": method,
                               "min_cells": min_cells, "n_train_clones": len(train_ids),
                               "n_test_clones": int(keep.sum()), "mae": mean_absolute_error(yy, pp),
                               "r2": r2_score(yy, pp) if np.ptp(yy) else np.nan,
                               "spearman": spearmanr(yy, pp).statistic if np.ptp(yy) and np.ptp(pp) else np.nan})
        audits.append({"heldout_mouse": donor, "n_train_cells": int(train_mask.sum()),
                       "n_test_cells": int(test_mask.sum()), "n_train_clones": len(train_ids),
                       "n_test_clones": len(test_ids), "gene_count": len(embedding.genes),
                       "genes_sha256": hashlib.sha256("\n".join(embedding.genes).encode()).hexdigest(),
                       "pca_sha256": hashlib.sha256(embedding.components.tobytes()).hexdigest(),
                       "density_reference_sha256": density.metadata()["reference_sha256"],
                       "elapsed_seconds": time.perf_counter()-start})
        print(f"{name}: held out {donor}, {len(test_ids)} clones, {time.perf_counter()-start:.1f}s", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / f"{name}_density_reporter_predictions.csv.gz", index=False)
    pd.DataFrame(scores).to_csv(OUT / f"{name}_density_reporter_scores.csv", index=False)
    audit = {"dataset": name, "target": "captured mCherry-low division-gate fraction", "fold_unit": "whole mouse",
             "rna_training_scope": "normalization per cell; HVGs/scaling/clipping/PCA fitted only on training mice",
             "density_training_scope": "regions and clone-balanced reference fitted only on training mice",
             "readout_model": "training-fold StandardScaler + Ridge(alpha=10); all include log1p(n_cells)",
             "hyperparameters_selected_on_test": False, "clones_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "folds": audits, "limitations": [
                 "Reporter gates were physically sorted into separate libraries; gate/library confounding remains.",
                 "No affinity, mutation-rate or fate prediction is implied.",
                 "Fixed model settings; no claim that these are optimised settings for each baseline.",
                 "Clonotrace is not included in this new validation.",
             ]}
    (OUT / f"{name}_density_reporter_audit.json").write_text(json.dumps(audit, indent=2))


def summarize_saved_scores():
    """Record every method, threshold and contributing mouse/clone count."""
    rows = []
    for path in sorted(OUT.glob("*_density_reporter_scores.csv")):
        table = pd.read_csv(path)
        for (dataset, method, size), group in table.groupby(["dataset", "method", "min_cells"]):
            rows.append({"dataset": dataset, "method": method, "min_cells": int(size),
                         "n_mice": len(group), "n_clones": int(group.n_test_clones.sum()),
                         "median_mae": group.mae.median(), "median_r2": group.r2.median()})
    pd.DataFrame(rows).to_csv(OUT / "density_reporter_summary.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1])
    summarize_saved_scores()
