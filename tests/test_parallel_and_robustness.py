"""Parallel layouts give the same result as one process; inputs that cannot be analysed fail with guidance."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf
from threadfin.maptune import resolve_n_jobs, tune_clone_map

FAST = dict(top=2, n_layouts=2, ladder_seeds=1, search_epochs=60)


def _branches(n=120, d=8, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    arms = rng.normal(size=(3, d))
    arm = rng.integers(0, 3, n)
    t = rng.uniform(0, 1, n)
    return t[:, None] * 3 * arms[arm] + rng.normal(0, noise, size=(n, d))


def test_parallel_layouts_choose_exactly_what_one_process_chooses():
    x = _branches()
    one = tune_clone_map(x, random_state=3, n_jobs=1, verbose=False, **FAST)
    two = tune_clone_map(x, random_state=3, n_jobs=2, verbose=False, **FAST)
    assert (one.n_neighbors, one.min_dist, one.repulsion_strength) == (two.n_neighbors, two.min_dist,
                                                                        two.repulsion_strength)
    cols = ["stage", "n_neighbors", "min_dist", "repulsion_strength", "seed", "score", "coherence"]
    pd.testing.assert_frame_equal(one.history[cols].reset_index(drop=True), two.history[cols].reset_index(drop=True))


def test_n_jobs_is_validated_and_capped():
    assert resolve_n_jobs(None) == 1 and resolve_n_jobs(1) == 1
    assert 1 <= resolve_n_jobs(-1) and resolve_n_jobs(10**6) == resolve_n_jobs(-1)
    for bad in (0, 1.5, True, "4"):
        with pytest.raises(ValueError, match="n_jobs"):
            resolve_n_jobs(bad)


def _repertoire(n_clones=40, size=3, singletons=60, seed=0):
    rng = np.random.default_rng(seed)
    rows, xs = [], []
    for c in range(n_clones):
        centre = rng.normal(size=6) * 2
        for _ in range(size):
            rows.append({"clone_id": f"m{c % 2}|c{c}", "donor": f"m{c % 2}", "sample": f"m{c % 2}", "day": "d7"})
            xs.append(centre + rng.normal(size=6))
    for i in range(singletons):
        rows.append({"clone_id": f"m{i % 2}|s{i}", "donor": f"m{i % 2}", "sample": f"m{i % 2}", "day": "d7"})
        xs.append(rng.normal(size=6) * 2)
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    adata = AnnData(X=np.zeros((len(obs), 2), dtype=np.float32), obs=obs)
    adata.obsm["X_emb"] = np.asarray(xs, dtype=np.float32)
    return adata


def test_run_explains_data_without_repeated_clones():
    adata = _repertoire(n_clones=0, singletons=80)
    with pytest.raises(ValueError, match="clones seen at least twice"):
        tf.run(adata, basis="X_emb", donor_key="donor", verbose=False)


def test_an_untestable_memory_axis_is_skipped_not_fatal():
    adata = _repertoire()
    res = tf.run(adata, basis="X_emb", donor_key="donor", sample_key="sample", time_key="day", n_perm=10,
                 n_boot=5, map_n_neighbors=10, map_min_dist=0.5, verbose=False)
    assert res.memory is None and len(res.clones) == 40
    assert list(res.skipped) == ["clonal memory over 'day'"]
    assert "Not run" in res.summary() and "clonal memory over 'day'" in res.summary()


def test_a_label_without_values_is_reported_in_the_summary():
    adata = _repertoire()
    adata.obs["empty"] = np.nan
    res = tf.run(adata, basis="X_emb", donor_key="donor", sample_key="sample", test=["empty"], n_perm=10,
                 n_boot=5, map_n_neighbors=10, map_min_dist=0.5, verbose=False)
    assert "clone-level test of 'empty'" in res.skipped
    assert "clone-level test of 'empty'" in res.summary()


def test_shared_clone_ids_name_the_problem_and_the_fix():
    adata = _repertoire()
    adata.obs.loc[adata.obs.index[:1], "donor"] = "m1"    # one cell of clone m0|c0 now belongs to another mouse
    with pytest.raises(ValueError, match="more than one donor.*donor-private"):
        tf.tl.clone_profiles(adata, basis="X_emb", donor_key="donor", verbose=False)


def test_non_finite_embedding_rows_are_counted_with_a_fix():
    adata = _repertoire()
    adata.obsm["X_emb"][:4] = np.nan
    with pytest.raises(ValueError, match="4 of .* cells.*drop those cells"):
        tf.tl.clone_profiles(adata, basis="X_emb", verbose=False)


def test_parallel_bootstrap_gives_the_same_programmes():
    adata = _repertoire(n_clones=60, size=4)
    tf.tl.clone_profiles(adata, basis="X_emb", donor_key="donor", context_key="sample", representation="kernel",
                         verbose=False)
    one = tf.tl.find_programmes(adata, n_boot=6, n_jobs=1, verbose=False)
    labels_one = adata.obs["clone_programme"].astype(str).copy()
    two = tf.tl.find_programmes(adata, n_boot=6, n_jobs=2, verbose=False)
    pd.testing.assert_frame_equal(one, two)
    assert labels_one.equals(adata.obs["clone_programme"].astype(str))
