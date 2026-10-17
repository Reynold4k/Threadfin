"""Tests for :func:`threadfin.tl.export_lag`."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import threadfin as tf


def build(n_clones=60, lag=0.004, slope=0.001, seed=0):
    """A germinal centre maturing at `slope` per day, exporting cells that left `lag` ago."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clones):
        for day in (10, 20, 30, 40):
            gc = slope * day + rng.normal(0, 0.0002)
            for _ in range(3):
                rows.append((f"c{c}", "GC", max(gc + rng.normal(0, 1e-4), 0), day, "IGHG"))
            for _ in range(2):
                rows.append((f"c{c}", "PB", max(gc - lag + rng.normal(0, 1e-4), 0), day, "IGHG"))
    obs = pd.DataFrame(rows, columns=["clone_id", "state", "mutation_frequency", "timepoint", "isotype"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.zeros((len(obs), 1), dtype="float32"), obs=obs)


def test_recovers_a_planted_lag():
    a = build(lag=0.004, slope=0.001)
    r = tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"],
                         within=("timepoint",), time_key="timepoint")
    assert r["n_clones"] == 240                      # 60 clones x 4 visits
    assert r["n_output_earlier"] > 0.95 * r["n_clones"]
    assert r["median_lag"] == pytest.approx(-0.004, abs=5e-4)
    # the planted maturation slope converts the gap to the planted delay
    assert r["maturation_slope"] == pytest.approx(0.001, rel=0.2)
    assert r["median_lag_days"] == pytest.approx(-4.0, abs=1.0)


def test_no_lag_gives_no_direction():
    a = build(lag=0.0)
    r = tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"],
                         within=("timepoint",))
    assert r["median_lag"] == pytest.approx(0.0, abs=5e-4)
    assert 0.3 < r["n_output_earlier"] / r["n_clones"] < 0.7


def test_within_separates_the_visits():
    """Without `within`, a clone's visits are pooled and the clock is diluted."""
    a = build(lag=0.004, slope=0.001)
    pooled = tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"])
    per_visit = tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"],
                                 within=("timepoint",))
    assert pooled["n_clones"] == 60 and per_visit["n_clones"] == 240


def test_switched_only_needs_an_isotype_column():
    a = build()
    del a.obs["isotype"]
    with pytest.raises(ValueError, match="isotype"):
        tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"])
    r = tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"],
                         switched_only=False)
    assert r["n_clones"] == 60


def test_overlapping_states_are_refused():
    a = build()
    with pytest.raises(ValueError, match="overlap"):
        tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["GC", "PB"])


def test_unsequenced_clones_are_dropped_not_imputed():
    a = build()
    a.obs.loc[a.obs["state"].eq("PB"), "mutation_frequency"] = np.nan
    with pytest.raises(ValueError, match="both sides"):
        tf.tl.export_lag(a, state_key="state", source_states=["GC"], output_states=["PB"],
                         within=("timepoint",))
