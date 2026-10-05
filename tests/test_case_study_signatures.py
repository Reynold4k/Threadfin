"""Prevent silently empty mouse pathway outputs caused by human gene symbols."""
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "case_studies"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("case_run", HERE / "run_case_study.py")
case_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_run)


@pytest.mark.parametrize("name", ["mouse_np", "mouse_rbd", "gc_np_pc", "flu_lung", "malaria", "malaria_late"])
def test_mouse_signatures_are_not_human_symbols(name):
    sets = case_run.gene_sets_for(name)
    assert "Aicda" in sets["germinal centre"]
    assert "Mzb1" in sets["plasma cell"]
    assert "AICDA" not in sets["germinal centre"]


def test_human_signatures_and_missing_gene_coverage():
    sets = case_run.gene_sets_for("ln_vaccine")
    assert "AICDA" in sets["germinal centre"]
    a = ad.AnnData(np.ones((3, 3)), obs=pd.DataFrame({"clone_id": ["c1", "c1", None]}),
                   var=pd.DataFrame(index=["AICDA", "MZB1", "XBP1"]))
    a.layers["log_norm"] = a.X.copy()
    scores, coverage = case_run.clone_gene_scores(a, sets)
    assert coverage["germinal centre"]["n_present"] == 1
    assert "germinal centre" not in scores
    assert scores.index.tolist() == ["c1"]


def test_missing_spike_binding_is_not_a_negative_measurement():
    from datasets import spike_binding_status
    assert spike_binding_status(True) == 'S+'
    assert spike_binding_status(' false ') == 'S-'
    for value in [None, np.nan, pd.NA, '', 'unknown']:
        assert pd.isna(spike_binding_status(value))
