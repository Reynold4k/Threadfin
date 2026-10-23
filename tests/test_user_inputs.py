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


def test_run_rejects_per_cell_bcr_table_with_guidance():
    adata = _adata()
    bcr = pd.DataFrame(
        {"v_call": ["IGHV1", "IGHV1"], "j_call": ["IGHJ1", "IGHJ1"],
         "junction": ["TGT", "TGT"]},
        index=["cell0", "cell0"],
    )
    with pytest.raises(ValueError, match="contig-level"):
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


def test_prepare_embedding_integrates_batches_with_harmony():
    """The Harmony step runs with every harmonypy the package allows (pure-Python 0.x and compiled 2.x)."""
    rng = np.random.default_rng(0)
    n, g = 240, 300
    genes = [f"G{i}" for i in range(g - 4)] + ["IGHM", "IGKC", "IGHV1-2", "IGLC2"]
    counts = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    batch = np.repeat(["a", "b", "c"], n // 3)
    counts[batch == "b", :40] += rng.poisson(3.0, size=(int((batch == "b").sum()), 40))   # a batch effect
    adata = AnnData(X=counts, obs=pd.DataFrame({"batch": batch}, index=[f"c{i}" for i in range(n)]),
                    var=pd.DataFrame(index=genes))
    tf.pp.prepare_embedding(adata, batch_key="batch", n_top_genes=150, n_comps=10, verbose=False)
    emb = adata.obsm["X_threadfin"]
    assert emb.shape == (n, 10) and np.isfinite(emb).all()
    assert not np.allclose(emb, adata.obsm["X_pca"][:, :10])          # Harmony changed the embedding


def test_a_misspelled_column_is_answered_with_the_closest_names():
    adata = _adata()
    adata.obs["sort_gate"] = "A"
    with pytest.raises(KeyError, match="Did you mean 'sort_gate'"):
        tf.run(adata, basis="X_pca", test=["sort_gat"], verbose=False)


def test_normalised_data_passed_as_counts_is_flagged():
    rng = np.random.default_rng(1)
    counts = rng.poisson(1.0, size=(60, 80)).astype(np.float32)
    adata = AnnData(X=np.log1p(counts / counts.sum(1, keepdims=True) * 1e4),
                    obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
                    var=pd.DataFrame(index=[f"G{i}" for i in range(80)]))
    with pytest.warns(UserWarning, match="does not look like raw counts"):
        tf.pp.prepare_embedding(adata, n_top_genes=40, n_comps=5, verbose=False)



def test_a_subset_without_copy_runs_and_keeps_its_results():
    """adata[mask] without .copy() is a view; results used to vanish and surface as 'run clone_profiles first'."""
    full = tf.sim.simulate_repertoire(random_state=0, n_clones=300)
    view = full[full.obs["donor"].isin(["D0", "D1"])]
    assert view.is_view
    res = tf.run(view, basis="X_pca", donor_key="donor", sample_key="context", n_perm=10, n_boot=3, verbose=False)
    assert not view.is_view and "threadfin" in view.uns and len(res.clones) > 0
