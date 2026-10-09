#!/usr/bin/env python3
"""Audit final state-model tables and optional before/after prediction snapshots."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "case_studies/results/algorithm_revision"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unchanged_predictions(old, new, keys):
    a, b = pd.read_csv(old), pd.read_csv(new)
    b = b[b.method.isin(a.method.unique())]
    a, b = [x.sort_values(keys).reset_index(drop=True) for x in (a, b)]
    assert a[keys].equals(b[keys])
    np.testing.assert_array_equal(a.observed, b.observed)
    delta = float(np.abs(a.predicted.to_numpy()-b.predicted.to_numpy()).max())
    assert delta <= 1e-12, delta
    return {"n_predictions": len(a), "maximum_absolute_prediction_difference": delta}

def main(previous_reporter=None, previous_larry=None):
    checks = {}
    integration = json.loads((OUT / "case_integration_audit.json").read_text())
    assert integration["status"] == "passed" and len(integration["datasets"]) == 12
    for x in integration["datasets"]:
        assert all(x[k] for k in ["identities_equal", "capture_counts_equal", "reliable_family_set_equal"])
    checks["twelve_case_identity_capture_and_eligibility"] = True
    for name, mice, clones in [("mouse_np", 7, 373), ("mouse_rbd", 10, 1414)]:
        pred = pd.read_csv(OUT / f"{name}_density_reporter_predictions.csv.gz")
        scores = pd.read_csv(OUT / f"{name}_density_reporter_scores.csv")
        audit = json.loads((OUT / f"{name}_density_reporter_audit.json").read_text())
        assert audit["clones_source_sha256"] == sha(ROOT / f"case_studies/results/{name}/cells.csv.gz")
        assert len(audit["folds"]) == mice and pred.heldout_mouse.nunique() == mice
        assert len(pred) == 4*clones
        assert set(pred.method) == {"size_only", "RNA_mean", "state_counts", "state_posterior"}
        assert not pred.duplicated(["method", "clone"]).any()
        assert pred.groupby("clone")["observed"].nunique().eq(1).all()
        assert pred.groupby("clone").heldout_mouse.nunique().eq(1).all()
        assert pred.groupby("method").clone.nunique().eq(clones).all()
        assert np.isfinite(pred[["observed", "predicted"]]).all().all()
        assert pred.predicted.between(0, 1).all()
        assert scores.groupby(["min_cells", "method"]).heldout_mouse.nunique().eq(mice).all()
        checks[f"{name}_common_coverage_and_current_input_hash"] = True
        if previous_reporter:
            checks[f"{name}_predictions_preserved"] = unchanged_predictions(
                previous_reporter / f"{name}_density_reporter_predictions.csv.gz",
                OUT / f"{name}_density_reporter_predictions.csv.gz",
                ["method", "heldout_mouse", "clone"])
    if previous_larry:
        checks["larry_original_five_methods_preserved"] = unchanged_predictions(
            previous_larry / "future_predictions.csv.gz",
            ROOT / "case_studies/results/clonotrace_larry/future_predictions.csv.gz",
            ["method", "repeat", "fold", "barcode", "target"])
    sim = pd.read_csv(OUT / "density_simulation.csv")
    assert set(sim.regime) == {"matched_prior", "heterogeneous", "near_shared", "pure"}
    assert sim.sampling_all_regions_coverage.min() >= .95
    pure = sim[sim.regime.eq("pure")]
    assert pure.credible_marginal_coverage.eq(0).all()
    assert (pure.mse > pure.raw_mse).all()
    checks["simulation_retains_pure_state_bias_and_credible_undercoverage"] = True
    larry = json.loads((OUT / "density_larry_summary.json").read_text())
    assert larry["reference_query_barcodes_disjoint"]
    assert (larry["n_reference_barcodes"], larry["n_evaluation_barcodes"], larry["n_evaluation_profiles"]) == (2920, 216, 250)
    checks["larry_reference_scope_and_evaluation_denominators"] = True
    sources = [ROOT / f"threadfin/{x}.py" for x in ["density", "expression", "profiles", "dynamics", "programmes"]]
    sources += [ROOT / "case_studies" / x for x in ["validate_density_model.py", "validate_density_reporter.py",
                                                  "benchmark_representation_runtime.py", "integrate_algorithm_revision.py"]]
    result = {"status": "passed", "checks": checks,
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
              "limits": ["Consistency checks are not independent biological validation.",
                         "Simulation coverage is conditional on fixed well-separated regions and independent sampling.",
                         "The official Clonotrace implementation is not a comparator here."]}
    (OUT / "final_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-reporter", type=Path)
    parser.add_argument("--previous-larry", type=Path)
    args = parser.parse_args()
    main(args.previous_reporter, args.previous_larry)
