"""The clone-map tuner: scoring, Adam search and its use in embed_clones."""

import numpy as np
import pytest

import threadfin as tf
from threadfin.maptune import map_quality, tune_clone_map

FAST = dict(top=2, n_layouts=2, ladder_seeds=1, search_epochs=100)
ADAM = dict(method="adam", adam_options=dict(n_iter=4, patience=4), **FAST)


def _curve(n=300, d=8, noise=0.05, seed=0):
    """Clones spread along one continuous curve: the map should come out as one piece."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 3 * np.pi, n))
    basis = rng.normal(size=(3, d))
    x = np.c_[np.cos(t), np.sin(t), t / 3] @ basis
    return x + rng.normal(0, noise, size=x.shape)


def test_map_quality_separates_a_torn_map_from_an_intact_one():
    x = _curve()
    t = np.linspace(0, 1, len(x))
    intact = np.c_[t, np.zeros_like(t)]                   # the curve, unrolled in order
    torn = intact.copy()
    torn[len(x) // 2:, 0] += 5.0                          # same order, cut in two
    q_intact, q_torn = map_quality(x, intact), map_quality(x, torn)
    assert q_intact["coherence"] == pytest.approx(1.0)
    assert q_torn["coherence"] == pytest.approx(0.5, abs=0.01)
    assert q_intact["score"] > q_torn["score"]
    assert 0 < q_intact["coherence_target"] <= 0.99


def test_map_quality_checks_shapes_and_target():
    x = _curve(n=60)
    with pytest.raises(ValueError, match="one row per clone"):
        map_quality(x, np.zeros((10, 2)))
    with pytest.raises(ValueError, match="coherence_target"):
        map_quality(x, np.zeros((60, 2)), coherence_target=1.5)


def _branches(n=360, d=10, noise=0.15, seed=0):
    """Clones along three arms that leave one root, like germinal-centre clones fanning out into outputs."""
    rng = np.random.default_rng(seed)
    arms = rng.normal(size=(3, d))
    arm = rng.integers(0, 3, n)
    t = rng.uniform(0, 1, n)
    return t[:, None] * 3 * arms[arm] + rng.normal(0, noise, size=(n, d))


def test_tuner_finds_a_coherent_map_and_is_reproducible():
    x = _branches()
    a = tune_clone_map(x, random_state=1, verbose=False, **FAST)
    b = tune_clone_map(x, random_state=1, verbose=False, **FAST)
    assert (a.n_neighbors, a.min_dist) == (b.n_neighbors, b.min_dist)
    assert 5 <= a.n_neighbors <= 250 and 0 <= a.min_dist <= 0.99
    assert a.coherence >= 0.9 and a.method == "ladder" and a.repulsion_strength in (1.0, 0.5, 0.25)
    assert set(a.history["stage"]) == {"ladder", "confirm"}
    assert {"n_neighbors", "min_dist", "repulsion_strength", "coherence"} <= set(a.candidates.columns)
    assert a.n_layouts == len(a.history)
    record = a.as_dict()
    assert record["n_neighbors"] == a.n_neighbors and "history" in record


def test_adam_warm_up_widens_a_fragmented_start():
    x = _branches()
    opts = dict(n_iter=4, patience=4, start=(5, 0.01))
    res = tune_clone_map(x, method="adam", adam_options=opts, random_state=0, verbose=False, **FAST)
    assert res.method == "adam"
    warm = res.history[res.history["stage"] == "warm-up"]
    assert warm["n_neighbors"].iloc[-1] >= warm["n_neighbors"].iloc[0]
    assert res.coherence >= 0.9


def test_tuner_holds_a_fixed_setting():
    x = _curve(n=200)
    res = tune_clone_map(x, n_neighbors=20, min_dist="auto", random_state=0, verbose=False, **FAST)
    assert res.n_neighbors == 20
    with pytest.raises(ValueError, match="Nothing to tune"):
        tune_clone_map(x, n_neighbors=20, min_dist=0.3, verbose=False)
    with pytest.raises(ValueError, match="at least 15"):
        tune_clone_map(x[:10], verbose=False)
    with pytest.raises(ValueError, match="method"):
        tune_clone_map(x, method="grid", verbose=False)
    res = tune_clone_map(x, n_neighbors="auto", min_dist=0.3, random_state=0, verbose=False, **FAST)
    assert res.min_dist == 0.3


def test_tuner_on_a_subset_rescales_neighbours():
    x = _curve(n=400)
    res = tune_clone_map(x, max_clones=200, random_state=0, verbose=False, **FAST)
    assert 5 <= res.n_neighbors <= 250
    assert (res.candidates["n_neighbors_full_data"] >= res.candidates["n_neighbors"]).all()


def test_embed_clones_auto_and_manual():
    ad = tf.sim.simulate_repertoire(n_clones=500, clone_size_exponent=1.9, random_state=3)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    tf.tl.embed_clones(ad, tune_kwargs=FAST, verbose=False)
    mp = ad.uns["threadfin"]["profiles"]["map_params"]
    assert mp["chosen_by"] == "tune_clone_map" and len(mp["attempts"]) >= 1
    assert 0 <= mp["quality"]["coherence"] <= 1 and mp["repulsion_strength"] > 0
    table = tf.tl.embed_clones(ad, n_neighbors=30, min_dist=0.3, verbose=False)
    mp = ad.uns["threadfin"]["profiles"]["map_params"]
    assert mp["chosen_by"] == "user" and "tuning" not in mp and mp["n_neighbors"] == 30
    assert table[["x", "y"]].notna().all().all()
    with pytest.raises(ValueError, match="'auto' or a number"):
        tf.tl.embed_clones(ad, n_neighbors="many", verbose=False)


def test_map_record_survives_h5ad(tmp_path):
    import anndata

    ad = tf.sim.simulate_repertoire(n_clones=500, clone_size_exponent=1.9, random_state=4)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    tf.tl.embed_clones(ad, tune_kwargs=FAST, verbose=False)
    ad.write_h5ad(tmp_path / "auto.h5ad")
    back = anndata.read_h5ad(tmp_path / "auto.h5ad")
    mp = back.uns["threadfin"]["profiles"]["map_params"]
    assert mp["chosen_by"] == "tune_clone_map" and len(mp["tuning"]["candidates"]) >= 1
    tf.tl.embed_clones(ad, n_neighbors=20, min_dist=0.5, verbose=False)
    ad.write_h5ad(tmp_path / "manual.h5ad")


def test_weaker_repulsion_rescue_runs_when_the_target_cannot_be_met():
    x = _branches()
    res = tune_clone_map(x, coherence_target=1.0, random_state=0, verbose=False, **FAST)
    reps = set(res.history["repulsion_strength"])
    assert 1.0 in reps
    if res.coherence < 1.0:          # the rescue was needed: weaker repulsion must have been tried
        assert reps & {0.5, 0.25}
