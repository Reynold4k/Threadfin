"""Predeclared stress tests; retain negative results and interval undercoverage.

Independent reference clones; fixed K=4 and alpha=2 before evaluation.
Regimes: matching Dirichlet(2q), heterogeneous Dirichlet(.2q),
near-shared Dirichlet(20q), and pure states. n=1,2,5,10,30,100.
An alpha='auto' comparator is calibrated only on the reference.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadfin.density import StateDensityModel

OUT = Path(__file__).resolve().parent / "results" / "algorithm_revision"
CENTRES = np.array([[-4., 0.], [0., 0.], [4., 0.], [0., 5.]])
Q = np.array([.35, .35, .28, .02])


def sample_profiles(probabilities, n, rng, prefix):
    states = np.concatenate([rng.choice(4, n, p=p) for p in probabilities])
    x = CENTRES[states] + rng.normal(0, .08, (len(states), 2))
    ids = np.repeat([f"{prefix}{i:04d}" for i in range(len(probabilities))], n)
    return x, ids


def simulation():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    ref_prob = rng.dirichlet(2*Q, 200)
    ref_x, ref_ids = sample_profiles(ref_prob, 80, rng, "ref")
    models = {
        "fixed_prior_2": StateDensityModel.fit(ref_x, ref_ids, n_states=4, prior_strength=2),
        "reference_calibrated": StateDensityModel.fit(ref_x, ref_ids, n_states=4, prior_strength="auto"),
    }
    models["reference_calibrated"].calibration.to_csv(OUT / "prior_calibration.csv", index=False)
    rows = []
    for regime, concentration in [("matched_prior", 2), ("heterogeneous", .2), ("near_shared", 20), ("pure", None)]:
        truth = (rng.dirichlet(concentration*Q, 500) if concentration is not None
                 else np.eye(4)[rng.choice(4, 500, p=Q)])
        for n in (1, 2, 5, 10, 30, 100):
            x, ids = sample_profiles(truth, n, rng, f"{regime}_")
            for method, model in models.items():
                start = time.perf_counter()
                result = model.transform(x, ids)
                # Align independently known biological components to reference regions.
                mapping = cdist(model.centres, CENTRES).argmin(axis=1)
                assert len(np.unique(mapping)) == 4
                known = truth[:, mapping]
                estimate = result.proportions.to_numpy()
                empirical = result.raw_proportions.to_numpy()
                credible_covered = ((result.lower.to_numpy() <= known) & (result.upper.to_numpy() >= known))
                sampling_covered = ((result.sampling_lower.to_numpy() <= known) &
                                    (result.sampling_upper.to_numpy() >= known))
                rare = int(np.flatnonzero(mapping == 3)[0])
                omitted = result.counts.iloc[:, rare].to_numpy() == 0
                rows.append({
                    "regime": regime, "n_cells": n, "method": method, "n_clones": len(truth),
                    "prior_strength": model.prior_strength,
                    "mse": float(np.mean((estimate-known)**2)),
                    "raw_mse": float(np.mean((empirical-known)**2)),
                    "credible_marginal_coverage": float(credible_covered.mean()),
                    "credible_all_regions_coverage": float(credible_covered.all(axis=1).mean()),
                    "sampling_all_regions_coverage": float(sampling_covered.all(axis=1).mean()),
                    "credible_mean_width": float((result.upper-result.lower).to_numpy().mean()),
                    "sampling_mean_width": float((result.sampling_upper-result.sampling_lower).to_numpy().mean()),
                    "rare_region_mse": float(np.mean((estimate[:, rare]-known[:, rare])**2)),
                    "rare_region_credible_coverage": float(credible_covered[:, rare].mean()),
                    "rare_region_sampling_coverage": float(sampling_covered[:, rare].mean()),
                    "unobserved_rare_true_mass": float(known[omitted, rare].mean()) if omitted.any() else None,
                    "elapsed_seconds": time.perf_counter()-start,
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "density_simulation.csv", index=False)
    # Null contrast diagnostic: marginal credible intervals are not multiplicity-controlled tests.
    null_rows = []
    for n in (2, 10, 30):
        x, ids = sample_profiles(np.tile(Q, (200, 1)), n, rng, "null")
        result = models["fixed_prior_2"].transform(x, ids)
        for i in range(0, 200, 2):
            contrast = result.contrast(f"null{i:04d}", f"null{i+1:04d}", n_draws=2000, random_state=i)
            flagged = (contrast.lower > 0) | (contrast.upper < 0)
            null_rows.append({"n_cells": n, "pair": i//2, "marginal_flag_fraction": flagged.mean(),
                              "any_region_flag": bool(flagged.any())})
    pd.DataFrame(null_rows).to_csv(OUT / "density_null_contrasts.csv", index=False)
    summary = {
        "random_seed": 1729, "n_reference_clones": 200, "reference_cells_per_clone": 80,
        "evaluation_clones_per_regime": 500, "n_states": 4, "query_sizes": [1, 2, 5, 10, 30, 100],
        "reference_sha256": models["fixed_prior_2"].metadata()["reference_sha256"],
        "automatic_prior_strength": models["reference_calibrated"].prior_strength,
        "minimum_simultaneous_sampling_coverage": float(table.sampling_all_regions_coverage.min()),
        "minimum_marginal_credible_coverage": float(table.credible_marginal_coverage.min()),
        "limits": [
            "Known, well-separated regions; does not validate arbitrary manifolds or donor/capture effects.",
            "Posterior intervals are conditional and can under-cover under prior misspecification.",
            "Pure-state empirical proportions are exact; shrinkage then adds bias.",
            "No comparison with official Clonotrace is made by this experiment.",
            "Sampling intervals assume independent multinomial draws conditional on a fixed reference.",
        ],
    }
    (OUT / "density_simulation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def larry():
    """Disjoint biological-barcode reference in fixed existing RNA coordinates."""
    import anndata as ad
    import hashlib
    path = Path(__file__).resolve().parents[2] / "internal_validation/paper_673503/larry/all_embedding.h5ad"
    a = ad.read_h5ad(path)
    obs = a.obs.loc[a.obs.barcode.notna()].copy()
    x = np.asarray(a.obsm["X_threadfin"])[a.obs_names.get_indexer(obs.index)]
    parity = obs.barcode.astype(str).map(lambda b: int(hashlib.sha256(b.encode()).hexdigest()[:8], 16) % 2)
    train = parity.to_numpy() == 0
    ref_ids = obs.loc[train, "barcode"].astype(str).to_numpy()
    ref_ctx = obs.loc[train, "day"].astype(str).to_numpy()
    models = {
        "fixed_prior_2": StateDensityModel.fit(x[train], ref_ids, contexts=ref_ctx, n_states=16),
        "reference_calibrated": StateDensityModel.fit(x[train], ref_ids, contexts=ref_ctx,
                                                      n_states=16, prior_strength="auto"),
    }
    groups = obs.loc[~train].groupby("clone_day_well", observed=True).groups
    eligible = [key for key, idx in groups.items() if len(idx) >= 16]
    rng = np.random.default_rng(8675309)
    selected = rng.choice(sorted(eligible), min(250, len(eligible)), replace=False)
    assert set(ref_ids).isdisjoint(obs.loc[~train, "barcode"].astype(str))
    rows = []
    for rep in range(20):
        query, truth_rows, labels, contexts, barcodes = {n: [] for n in (2, 4, 8)}, [], [], [], []
        # Shared split and target, independently of method or alpha.
        for key in selected:
            loc = obs.index.get_indexer(groups[key])
            perm = rng.permutation(loc)
            half = len(loc)//2
            small, heldout = perm[:half], perm[half:]
            reference_state = models["fixed_prior_2"]._assign(x[heldout])
            truth_rows.append(np.bincount(reference_state, minlength=16)/len(heldout))
            for n in query:
                query[n].append(x[small[:n]])
            labels.append(str(key))
            contexts.append(str(obs.iloc[loc[0]].day))
            barcodes.append(str(obs.iloc[loc[0]].barcode))
        truth = pd.DataFrame(truth_rows, index=labels)
        for n in query:
            for name, model in models.items():
                result = model.transform(np.vstack(query[n]), np.repeat(labels, n),
                                         contexts=np.repeat(contexts, n))
                target = truth.loc[result.proportions.index].to_numpy()
                for method, estimate in [(name, result.proportions.to_numpy()),
                                         ("raw_region_counts", result.raw_proportions.to_numpy())]:
                    if method == "raw_region_counts" and name != "fixed_prior_2":
                        continue
                    mse = ((estimate-target)**2).mean(axis=1)
                    bc = dict(zip(labels, barcodes))
                    rows.extend({"repeat": rep, "n_cells": n, "method": method,
                                 "profile": key, "barcode": bc[key], "mse": value}
                                for key, value in zip(result.proportions.index, mse))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "density_larry_split_errors.csv.gz", index=False)
    biological = table.groupby(["method", "n_cells", "barcode"], observed=True).mse.mean().reset_index()
    biological.to_csv(OUT / "density_larry_barcode_errors.csv", index=False)
    report = {"n_reference_barcodes": len(set(ref_ids)), "n_reference_cells": int(train.sum()),
              "n_evaluation_profiles": len(selected), "n_evaluation_barcodes": table.barcode.nunique(),
              "repeats": 20, "n_states": 16, "reference_query_barcodes_disjoint": True,
              "fixed_strength": 2, "automatic_strength": models["reference_calibrated"].prior_strength,
              "rna_preprocessing": "shared precomputed embedding; transductive RNA coordinates",
              "target": "independent half-sample region fractions; not full latent lineage potential",
              "aggregation": "mean within biological barcode, then equal weight per barcode",
              "official_clonotrace_comparison": False}
    summary = biological.groupby(["method", "n_cells"]).mse.mean().unstack("method")
    summary.to_csv(OUT / "density_larry_split_summary.csv")
    report["mse_by_size"] = summary.to_dict(orient="index")
    (OUT / "density_larry_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "larry":
        larry()
    else:
        simulation()
