"""Clonal memory: which clones a clone is compared with, and how sampling noise is removed.

A clone seen at two levels (time points or tissues) can share a state with the other clones seen twice that the
clones seen once lack. Compared with all clones (the Threadfin 4.5 estimator, ``reference="all"``), that selection
alone reads as memory; compared with the other clones seen twice (the default), it does not. The noise removed from
each squared change must be that of the snapshots compared: one variance pooled over all snapshots
(``noise="pooled"``) over-corrects when the compared snapshots are more homogeneous than the rest. Where a test
guards against one of these bugs, it also shows that the 4.5 estimator fails the same check, so the test can see
the bug. The data are small hand-built repertoires with a known memory index; each test runs in seconds.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.stats import binom

import threadfin as tf

ALPHA = 0.05


def _max_rejections(n_rep: int, alpha: float = ALPHA) -> int:
    """Largest number of rejections at ``alpha`` that a valid test gives in ``n_rep`` replicates (99.9 %)."""
    return int(binom.ppf(0.999, n_rep, alpha))


def _repertoire(seed, *, select=True, keep=0.0, sd=(1.0, 1.0, 1.0), sd_twice=1.0, n_twice=60, n_once=240,
                n_donors=2, dim=6, sep=4.0, cap=20):
    """Two levels, t0 and t1; per donor ``n_twice`` clones seen at both levels and ``n_once`` seen at one.

    A cell sits in state A (at 0), B or C (``sep`` along one of two axes), plus noise with the state's SD ``sd``,
    times ``sd_twice`` for clones seen twice. With ``select``, clones seen twice are mostly A and the others mostly
    B or C; without, every clone draws its state composition from one mixture. A clone seen twice keeps its
    composition with probability ``keep``, the true memory index, and otherwise draws a new one. Contexts are
    donor x level; snapshot sizes follow a Zipf law (most are one cell).
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, dim))
    centres[1, 0] = centres[2, 1] = sep
    sd = np.asarray(sd, dtype=float)
    mostly_a, mostly_bc, mixed = np.array([.8, .1, .1]), np.array([.1, .6, .3]), np.array([.45, .35, .2])
    x, clone, donor, level = [], [], [], []
    for d in range(n_donors):
        n = n_twice + n_once
        twice = np.arange(n) < n_twice
        theta = np.where(twice[:, None], mostly_a, mostly_bc) if select else np.tile(mixed, (n, 1))
        comp = [np.vstack([rng.dirichlet(t) for t in theta]) for _ in range(2)]
        kept = rng.random(n) < keep
        comp[1][kept] = comp[0][kept]
        once_at = rng.integers(0, 2, size=n)
        shift = rng.normal(0, 1, size=(2, dim))
        for t in (0, 1):
            ids = np.flatnonzero(twice | (once_at == t))
            rep = np.repeat(ids, np.minimum(rng.zipf(2.0, size=ids.size), cap))
            state = (rng.random(rep.size)[:, None] > np.cumsum(comp[t][rep], axis=1)).sum(axis=1).clip(max=2)
            scale = sd[state] * np.where(twice[rep], sd_twice, 1.0)
            x.append(centres[state] + shift[t] + rng.normal(size=(rep.size, dim)) * scale[:, None])
            clone += [f"d{d}c{i}" for i in rep]
            donor += [f"d{d}"] * rep.size
            level += [f"t{t}"] * rep.size
    obs = pd.DataFrame({"clone_id": clone, "donor": donor, "time": level},
                       index=[f"cell{i}" for i in range(len(clone))])
    obs["context"] = obs["donor"] + "_" + obs["time"]
    adata = AnnData(np.zeros((len(clone), 1), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.vstack(x)
    tf.tl.clone_profiles(adata, context_key="context", donor_key="donor", representation="mean", verbose=False)
    return adata


def _memory(adata, **kwargs):
    kwargs = {"n_null": 99, "n_boot": 50, "verbose": False, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.tl.clonal_memory(adata, "time", **kwargs)


def _memory_or_none(adata, **kwargs):
    """The result, or None when the noise-corrected distance between clones is not positive (no index)."""
    try:
        return _memory(adata, **kwargs)
    except ValueError as e:
        assert "not identifiable" in str(e)
        return None


# --------------------------------------------------------------------------- the reference set


@pytest.mark.parametrize("select", [False, True], ids=["pure null", "selection only"])
def test_memory_test_is_calibrated_when_no_clone_keeps_its_state(select):
    """No clone keeps its state between the levels. With all clones alike, and with clones seen twice mostly in a
    state that clones seen once lack (selection), the default rejects at most at the nominal rate and its index
    centres on 0. Under selection the 4.5 estimator reads the shared state as memory and rejects every time."""
    n_rep = 40
    new, old = [], []
    for seed in range(n_rep):
        adata = _repertoire(seed, select=select)
        new.append(_memory(adata))
        old.append(_memory(adata, reference="all", noise="pooled"))
    assert sum(r["p_value"] < ALPHA for r in new) <= _max_rejections(n_rep)
    assert abs(np.median([r["memory_index"] for r in new])) < 0.15
    if select:
        assert all(r["p_value"] < ALPHA for r in old)
        assert np.median([r["memory_index"] for r in old]) > 0.5


def _three_level_repertoire(seed, n_type=40, n_once=150, n_donors=2, dim=6, sep=4.0):
    """Three levels, t0 < t1 < t2, and no clone memory: every clone draws a new state composition at each level.
    Clones seen at t0 and t1 are mostly in state C, at t1 and t2 mostly B, at t0 and t2 mostly A; clones seen once
    are mixed."""
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, dim))
    centres[1, 0] = centres[2, 1] = sep
    kinds = {(0, 1): [.1, .1, .8], (1, 2): [.1, .8, .1], (0, 2): [.8, .1, .1]}
    x, clone, donor, level = [], [], [], []
    for d in range(n_donors):
        shift = rng.normal(0, 1, size=(3, dim))
        clones = [(levels, theta) for levels, theta in kinds.items() for _ in range(n_type)]
        clones += [((int(rng.integers(0, 3)),), [.34, .33, .33]) for _ in range(n_once)]
        for c, (levels, theta) in enumerate(clones):
            for t in levels:
                state = rng.choice(3, size=min(int(rng.zipf(2.0)), 20), p=rng.dirichlet(theta))
                x.append(centres[state] + shift[t] + rng.normal(size=(state.size, dim)))
                clone += [f"d{d}c{c}"] * state.size
                donor += [f"d{d}"] * state.size
                level += [f"t{t}"] * state.size
    obs = pd.DataFrame({"clone_id": clone, "donor": donor, "time": level},
                       index=[f"cell{i}" for i in range(len(clone))])
    obs["context"] = obs["donor"] + "_" + obs["time"]
    adata = AnnData(np.zeros((len(clone), 1), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.vstack(x)
    tf.tl.clone_profiles(adata, context_key="context", donor_key="donor", representation="mean", verbose=False)
    return adata


def test_memory_test_is_calibrated_when_pairs_of_levels_select_different_clones():
    """Three levels and no clone memory, but clones seen at t0 and t2 sit in one state and clones seen at t1 and
    t2 in another. Their t2 snapshots are not interchangeable, so each pair is compared only with pairs of the same
    donor between the same two levels: the test stays calibrated. (Strata of donor and later level alone reject
    in every replicate of this design; so does the 4.5 estimator, shown here.)"""
    n_rep = 30
    new, old = [], []
    for seed in range(n_rep):
        adata = _three_level_repertoire(seed)
        new.append(_memory(adata))
        old.append(_memory(adata, reference="all", noise="pooled"))
    assert sum(r["p_value"] < ALPHA for r in new) <= _max_rejections(n_rep)
    assert abs(np.median([r["memory_index"] for r in new])) < 0.1
    assert all(r["p_value"] < ALPHA for r in old)


@pytest.mark.parametrize("keep", [1.0, 0.5])
def test_real_memory_is_detected_and_measured(keep):
    """Under the same selection, clones seen twice keep their state composition with probability ``keep``, which
    is the true index. The default detects the memory and its index is close to ``keep``; the 4.5 estimator adds
    the selection to it."""
    new, old = [], []
    for seed in range(5):
        adata = _repertoire(seed, keep=keep, n_twice=150)
        new.append(_memory(adata))
        old.append(_memory(adata, reference="all", noise="pooled"))
    assert sum(r["p_value"] < ALPHA for r in new) >= 4
    assert abs(np.median([r["memory_index"] for r in new]) - keep) < 0.2
    assert np.median([r["memory_index"] for r in old]) - keep > 0.3


def test_the_old_estimator_is_still_available_and_unchanged():
    """reference="all", noise="pooled" is the Threadfin 4.5 estimator: on a fixed repertoire it returns what 4.5.3
    returned (the numbers below were computed with the 4.5.3 code). The default is the corrected estimator, and
    unknown settings are refused."""
    adata = _repertoire(11)
    old = _memory(adata, reference="all", noise="pooled", n_null=200, n_boot=100, random_state=3)
    assert (old["reference"], old["noise"]) == ("all", "pooled")
    assert (old["n_pairs"], old["n_clones"]) == OLD_45["n_pairs"]
    assert old["memory_index"] == pytest.approx(OLD_45["memory_index"], rel=1e-9)
    assert old["p_value"] == pytest.approx(OLD_45["p_value"], rel=1e-12)
    np.testing.assert_allclose(old["memory_index_ci"], OLD_45["memory_index_ci"], rtol=1e-9)
    assert old["mean_change_random"] == pytest.approx(OLD_45["mean_change_random"], rel=1e-9)
    new = _memory(adata, n_null=200, n_boot=100, random_state=3)
    assert (new["reference"], new["noise"]) == ("persistent", "snapshot")
    assert new["memory_index"] < old["memory_index"] - 0.3        # the selection in this repertoire is gone
    with pytest.raises(ValueError, match="reference must be"):
        _memory(adata, reference="every clone")
    with pytest.raises(ValueError, match="noise must be"):
        _memory(adata, noise="global")


# values returned by Threadfin 4.5.3 (dynamics.py before the reference change) for _repertoire(11)
OLD_45 = {"n_pairs": (120, 120), "memory_index": 0.6634808564756305, "p_value": 0.009950248756218905,
          "memory_index_ci": (0.2497011029586316, 1.0948795615967786), "mean_change_random": 4.142473623660328}


# --------------------------------------------------------------------------- the noise term


def _expected_noise_terms(adata):
    """Noise term v / n of every (clone, level) snapshot, computed here with pandas from the embedding.

    A snapshot of >= 2 cells: the unbiased variance of its own cells, summed over features. A one-cell snapshot:
    the mean of that variance over the multi-cell snapshots of the same level of clones seen at both levels (the
    fixture has at least five at each level). Context centring does not change these variances, since all cells
    of a snapshot share one context. Also returns the 4.5 term: the within-snapshot mean square pooled over all
    multi-cell snapshots.
    """
    obs = adata.obs
    key = (obs["clone_id"] + "|" + obs["time"]).to_numpy()
    x = pd.DataFrame(adata.obsm["X_pca"], index=obs.index)
    snaps = pd.DataFrame({"n": pd.Series(key).value_counts()})
    snaps["v"] = x.groupby(key).var(ddof=1).sum(axis=1, min_count=1).reindex(snaps.index)
    snaps["clone"] = snaps.index.str.split("|").str[0]
    snaps["time"] = snaps.index.str.split("|").str[1]
    snaps["twice"] = snaps["clone"].map(obs.groupby("clone_id", observed=True)["time"].nunique()).to_numpy() == 2
    multi = snaps[snaps["n"] >= 2]
    compared = multi[multi["twice"]]
    assert compared.groupby("time").size().min() >= 5
    v = np.where(snaps["n"] >= 2, snaps["v"], snaps["time"].map(compared.groupby("time")["v"].mean()))
    pooled = float((multi["v"] * (multi["n"] - 1)).sum() / (multi["n"] - 1).sum())
    return pd.Series(v / snaps["n"].to_numpy(), index=snaps.index), pooled


def test_noise_removed_from_each_pair_is_that_of_its_own_snapshots():
    """Guards a pooled noise term: each pair's noise term is v_a / n_a + v_b / n_b with every snapshot's own
    variance (one-cell snapshots: the mean of the multi-cell snapshots of their level among clones seen twice),
    and noise="pooled" gives the 4.5 term. Checked against a computation with pandas from the embedding."""
    adata = _repertoire(5, sd=(0.6, 1.4, 1.0))
    terms, pooled = _expected_noise_terms(adata)
    res = _memory(adata)
    pairs = res["pairs"]
    expected = (terms[(pairs["clone"] + "|" + pairs["earlier"]).to_numpy()].to_numpy()
                + terms[(pairs["clone"] + "|" + pairs["later"]).to_numpy()].to_numpy())
    np.testing.assert_allclose(pairs["noise_removed"], expected, rtol=1e-8)
    assert res["mean_noise_removed"] == pytest.approx(pairs["noise_removed"].mean())
    assert res["variance_source"]["own"] > 0 and res["variance_source"]["level"] > 0
    # (with the default reference the pooled term over-corrects so much here that no distance between clones is
    # left, so the pooled term is read from the 4.5 estimator)
    with pytest.raises(ValueError, match="not identifiable"):
        _memory(adata, noise="pooled")
    old = _memory(adata, reference="all", noise="pooled")["pairs"]
    np.testing.assert_allclose(old["noise_removed"],
                               pooled * (1 / old["n_cells_earlier"] + 1 / old["n_cells_later"]), rtol=1e-8)


def test_one_cell_snapshots_take_the_variance_of_comparable_snapshots():
    """Guards the rule for one-cell snapshots, whose own variance is unknown: the mean over the multi-cell
    snapshots of the same level and kind of clone; with fewer than five of those, that kind at every level; then
    every snapshot. A bootstrap weighting re-estimates the means, and a group that a resample leaves empty keeps
    its full-data mean. Hand-built snapshots with known variances."""
    from threadfin.dynamics import _snapshot_noise

    rows = ([(0, 2, 1.0, True)] * 5 + [(0, 3, 3.0, True)] * 5        # level 0, seen twice: ten multi-cell snapshots
            + [(1, 2, 1.0, True)] * 2 + [(1, 4, 5.0, True)] * 2      # level 1, seen twice: only four
            + [(1, 2, 10.0, False)]                                  # level 1, seen once: one
            + [(0, 1, np.nan, True), (1, 1, np.nan, True), (0, 1, np.nan, False)])   # one-cell snapshots
    snaps = pd.DataFrame(rows, columns=["t", "n_cells", "v", "twice"])
    snaps["within_ss"] = (snaps["v"] * (snaps["n_cells"] - 1)).fillna(0.0)
    terms, source = _snapshot_noise(snaps, snaps["twice"].to_numpy())
    got = terms()
    multi = snaps["n_cells"].to_numpy() >= 2
    np.testing.assert_allclose(got[multi], (snaps["v"] / snaps["n_cells"]).to_numpy()[multi])  # own variance / n
    seen_twice = (5 * 1.0 + 5 * 3.0 + 2 * 1.0 + 2 * 5.0) / 14
    np.testing.assert_allclose(got[-3:], [(5 * 1.0 + 5 * 3.0) / 10, seen_twice, (14 * seen_twice + 10.0) / 15])
    assert list(source[-3:]) == ["level", "kind", "all"]
    # bootstrap weights: leave out the five three-cell snapshots of level 0, then all of level 0
    w = np.ones(len(snaps))
    w[5:10] = 0.0
    assert terms(w)[-3] == pytest.approx(1.0)
    w[:10] = 0.0
    assert terms(w)[-3] == pytest.approx(2.0)                      # emptied group: the full-data mean


def test_snapshot_noise_keeps_perfect_memory_at_one_when_compared_snapshots_are_homogeneous():
    """Every clone seen twice keeps its state composition (true index 1), and its cells are tight (noise SD 0.5)
    while the cells of clones seen once are wide (SD 2). The variance pooled over all snapshots is mostly that of
    the wide clones: the 4.5 noise term over-corrects and pushes the index far above 1, or leaves no positive
    distance between clones. Each snapshot's own variance keeps the index near 1."""
    new, old = [], []
    for seed in range(5):
        adata = _repertoire(seed, select=False, keep=1.0, sd=(2.0, 2.0, 2.0), sd_twice=0.25)
        new.append(_memory(adata)["memory_index"])
        res = _memory_or_none(adata, reference="all", noise="pooled")
        old.append(np.inf if res is None else res["memory_index"])
    assert 0.8 < np.median(new) < 1.2
    assert np.median(old) > 1.5


def test_permutation_p_value_does_not_depend_on_the_noise_term():
    """With the default reference every permutation pairs the same snapshots, so the noise terms are the same in
    each one: the P value is identical with either noise term (same random_state), while the index differs."""
    adata = _repertoire(2, select=False, keep=0.5)
    own = _memory(adata, random_state=4)
    pooled = _memory(adata, noise="pooled", random_state=4)
    assert own["p_value"] == pooled["p_value"]
    assert own["memory_index"] != pooled["memory_index"]


# --------------------------------------------------------------------------- reporting


def test_an_index_without_interval_is_not_reported_as_an_estimate(capsys):
    """Four clones; three keep their state and one moves far away. Resamples that draw the moved clone at least
    twice leave no positive distance between clones, so the bootstrap interval is undefined: the result says so
    (NaN interval, a warning) and the log calls the index not interpretable instead of printing '[nan, nan]'."""
    rows, x = [], []
    for c in range(4):
        for t, centre in (("t0", (0.0, 0.0)), ("t1", (np.sqrt(12.0), 0.0) if c == 3 else (0.0, 0.0))):
            for sign in (1.0, -1.0):                       # two cells 1 apart from the snapshot mean: v = 2
                x.append(np.asarray(centre) + sign * np.array([0.0, 1.0]))
                rows.append({"clone_id": f"c{c}", "time": t})
    adata = AnnData(np.zeros((len(rows), 1), dtype=np.float32),
                    obs=pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))]))
    adata.obsm["X_pca"] = np.asarray(x)
    tf.tl.clone_profiles(adata, representation="mean", verbose=False)
    with pytest.warns(UserWarning, match="could not be computed"):
        res = tf.tl.clonal_memory(adata, "time", min_pairs=4, n_null=20, n_boot=50, random_state=0)
    assert np.isnan(res["memory_index_ci"]).all() and res["bootstrap_identifiable_fraction"] < 1
    # D_other = 2, 2, 2, -2 and D_same = -2, -2, -2, 10 for the four clones: the index itself is 0
    assert res["mean_change_random"] == pytest.approx(1.0) and res["memory_index"] == pytest.approx(0.0, abs=1e-12)
    log = capsys.readouterr().out
    assert "not interpretable" in log and "nan" not in log


def test_programme_transitions_expected_under_the_default_reference_are_exact():
    """Guards the descriptive programme table: its expectation pairs each clone with another clone seen at the same
    two levels, exactly (no sampling), so each row sums to the row of the observed table and the diagonal sums to
    persistence_null x n_pairs."""
    adata = _repertoire(4, keep=0.5)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    first_axis = pd.DataFrame(adata.obsm["X_pca"][:, 0], index=adata.obs["clone_id"].to_numpy()).groupby(level=0)[0].mean()
    table["clone_programme"] = np.where(first_axis.reindex(table.index.astype(str)).to_numpy() > 1.0, "P1", "P2")
    res = _memory(adata)
    observed, expected = res["transitions"], res["transitions_expected"]
    np.testing.assert_allclose(expected.sum(axis=1), observed.sum(axis=1))
    assert np.trace(expected.to_numpy()) == pytest.approx(res["persistence_null"] * res["n_pairs"])
    assert 0 < res["persistence_null"] < 1
