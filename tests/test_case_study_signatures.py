"""Prevent silently empty mouse pathway outputs caused by human gene symbols."""
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "manuscript" / "case_studies"
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


def test_loaders_do_not_reference_deleted_constants():
    """A loader must not reference a module-level constant that no longer exists.

    Deleting a retired loader once took the `_GSE287123` path constant with it. Nothing caught it:
    the module still imports, because a name used only inside a function body is resolved at call
    time, so it died eleven datasets into a SLURM re-run. This checks the exact failure class --
    module-level constants (UPPER_CASE, with or without a leading underscore) used inside a function
    but never assigned at module level.
    """
    import ast
    import re

    src = (HERE / "datasets.py").read_text()
    tree = ast.parse(src)
    module_level = {t.id for n in tree.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
    module_level |= {(al.asname or al.name).split(".")[0]
                     for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                     for al in n.names}
    const = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        assigned = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        assigned |= {(al.asname or al.name).split(".")[0]          # imports inside the function body
                     for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))
                     for al in n.names}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and const.match(node.id) and node.id not in module_level
                    and node.id not in assigned):
                pytest.fail(f"datasets.{fn.name} uses undefined constant {node.id!r} "
                            f"(datasets.py line {node.lineno})")


def test_every_registered_loader_is_callable():
    import datasets

    assert set(datasets.LOADERS), "the loader registry is empty"
    for name, fn in datasets.LOADERS.items():
        assert callable(fn), f"{name} is not callable"
