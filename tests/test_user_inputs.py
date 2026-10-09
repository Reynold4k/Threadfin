"""Regression tests for errors a new Threadfin user can act on."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf


def _adata():
    return AnnData(
        X=np.ones((6, 3)),
        obs=pd.DataFrame({"clone_id": ["a", "a", "b", "b", "c", "c"]},
                         index=[f"cell{i}" for i in range(6)]),
    )


def test_run_reports_all_missing_user_columns_before_analysis():
    with pytest.raises(KeyError, match="'sample' not found.*threadfin.run"):
        tf.run(_adata(), basis="X_pca", sample_key="sample", verbose=False)


def test_run_does_not_silently_skip_a_missing_requested_test_label():
    with pytest.raises(KeyError, match="'antigen' not found.*threadfin.run"):
        tf.run(_adata(), basis="X_pca", test=["antigen"], verbose=False)


@pytest.mark.parametrize("key, value", [("donor", np.nan), ("sample", "   "), ("batch", pd.NA)])
def test_run_rejects_missing_or_blank_grouping_labels(key, value):
    adata = _adata()
    adata.obs[key] = "group1"
    adata.obs.iloc[0, adata.obs.columns.get_loc(key)] = value
    with pytest.raises(ValueError, match=rf"Grouping column '{key}'.*fill or remove"):
        tf.run(adata, basis="X_pca", **{f"{key}_key": key}, verbose=False)


@pytest.mark.parametrize("kwargs, name", [
    ({"n_perm": 1.5}, "n_perm"), ({"n_perm": 0}, "n_perm"),
    ({"n_boot": -1}, "n_boot"), ({"n_boot": True}, "n_boot"),
])
def test_run_requires_positive_integer_resampling_counts(kwargs, name):
    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        tf.run(_adata(), basis="X_pca", verbose=False, **kwargs)


def test_run_explains_duplicate_bcr_barcodes():
    adata = _adata()
    bcr = pd.DataFrame(
        {"v_call": ["IGHV1", "IGHV1"], "j_call": ["IGHJ1", "IGHJ1"],
         "junction": ["TGT", "TGT"]},
        index=["cell0", "cell0"],
    )
    with pytest.raises(ValueError, match="duplicate cell barcodes.*one row per cell"):
        tf.run(adata, bcr=bcr, basis="X_pca", verbose=False)


def test_clone_profiles_explains_too_few_expanded_clones():
    adata = _adata()
    adata.obsm["X_pca"] = np.arange(12, dtype=float).reshape(6, 2)
    adata.obs["clone_id"] = ["a", "a", "b", "b", None, None]
    with pytest.raises(ValueError, match="at least 3 expanded clones"):
        tf.tl.clone_profiles(adata, basis="X_pca", verbose=False)


def test_prepare_embedding_rejects_negative_counts_with_fix():
    adata = _adata()
    adata.X[0, 0] = -1
    with pytest.raises(ValueError, match="finite non-negative raw counts.*precomputed embedding"):
        tf.pp.prepare_embedding(adata, verbose=False)


@pytest.mark.parametrize("n_comps", [0, -1, 1.5, True])
def test_prepare_embedding_requires_positive_integer_n_comps(n_comps):
    with pytest.raises(ValueError, match="n_comps must be a positive integer"):
        tf.pp.prepare_embedding(_adata(), n_comps=n_comps, verbose=False)
