"""Gene-set test of `tl.geneset_heritability` and intervals of `tl.export_lag` (statistical review F05, F10)."""

import io

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, mannwhitneyu

import threadfin as tf
from threadfin._utils import permute_within
from threadfin.export import _median_interval, _sen_interval, _theil_sen
from threadfin.heritability import _clone_bootstrap_weights, _mean_squares, _shuffled_mean_squares
from threadfin.profiles import variance_components

# --------------------------------------------------------------------------- gene sets


def modules(seed, *, n_clones=500, n_mod=12, per=10, tau2=0.0, planted=(), planted_tau2=0.0):
    """Clones of Zipf sizes; genes in co-expressed modules.

    A module's genes share a module activity with a cell-level part (co-expression) and, with ``tau2 > 0``,
    a clone-level part that is the same for every module, so all genes are equally clonal. ``planted``
    modules get an extra clone-level part of variance ``planted_tau2``.
    """
    rng = np.random.default_rng(seed)
    size = np.minimum(rng.zipf(1.8, n_clones), 20)
    clone = np.repeat(np.arange(n_clones), size)
    n = clone.size
    tau = np.full(n_mod, tau2)
    tau[list(planted)] += planted_tau2
    act = rng.normal(size=(n, n_mod)) + rng.normal(size=(n_clones, n_mod))[clone] * np.sqrt(tau)
    base = rng.uniform(0.5, 2.5, n_mod * per)
    y = np.repeat(act, per, axis=1) * 0.6 + base + rng.normal(size=(n, n_mod * per))
    genes = [f"m{g // per}_g{g % per}" for g in range(n_mod * per)]
    obs = pd.DataFrame({"clone_id": [f"c{c}" for c in clone], "sample": "s1"},
                       index=[f"cell{i}" for i in range(n)])
    adata = ad.AnnData(np.clip(y, 0, None).astype("float32"), obs=obs, var=pd.DataFrame(index=genes))
    sets = {f"m{m}": [f"m{m}_g{k}" for k in range(per)] for m in range(n_mod)}
    return adata, sets


def heritability(adata, seed=0, n_perm=20, **kw):
    return tf.tl.gene_heritability(adata, clone_key="clone_id", context_key="sample", layer=None, n_perm=n_perm,
                                   min_frac_expressed=0.0, random_state=seed, verbose=False, **kw)


def test_mean_squares_reproduce_variance_components_and_the_shuffles():
    rng = np.random.default_rng(0)
    clone = np.repeat(np.arange(80), rng.integers(1, 9, 80))
    resid = rng.normal(size=(clone.size, 5)) + rng.normal(size=(80, 5))[clone]
    multi = np.bincount(clone)[clone] >= 2
    cl = clone[multi]
    idx = np.unique(cl, return_inverse=True)[1]
    strata = rng.integers(0, 3, idx.size)
    w = _clone_bootstrap_weights(idx, strata, 3, rng)
    home = pd.crosstab(idx, strata).to_numpy().argmax(axis=1)
    for s in range(3):                                          # every replicate keeps the clones per stratum
        assert (w[:, home == s].sum(axis=1) == (home == s).sum()).all()
    msb, msw, n0 = _mean_squares(resid[multi], idx, np.ones((1, w.shape[1])))
    vc = variance_components(resid[multi], cl)
    assert np.allclose(np.maximum((msb[0] - msw[0]) / n0[0], 0), vc.tau2) and np.allclose(msw[0], vc.sigma2)
    # the exact expectation under shuffling within strata agrees with shuffles
    e_msb, e_msw = _shuffled_mean_squares(resid[multi], idx, strata)
    shuffled = [_mean_squares(resid[multi], np.unique(permute_within(idx, strata, rng), return_inverse=True)[1],
                              np.ones((1, w.shape[1]))) for _ in range(2000)]
    assert np.allclose(np.mean([s[0][0] for s in shuffled], axis=0), e_msb, rtol=0.03)
    assert np.allclose(np.mean([s[1][0] for s in shuffled], axis=0), e_msw, rtol=0.01)


def test_bootstrap_does_not_change_the_gene_table():
    adata, _ = modules(1, tau2=0.3)
    with_boot = heritability(adata.copy(), n_boot=50)
    without = heritability(adata.copy(), n_boot=0)
    pd.testing.assert_frame_equal(with_boot, without)          # permutations and p-values are unaffected
    assert "clone_bootstrap" in with_boot.attrs and "clone_bootstrap" not in without.attrs
    stored = adata.copy()
    heritability(stored, n_boot=50)
    assert stored.uns["threadfin"]["gene_heritability_bootstrap"].shape == (with_boot.shape[0], 50)
    # Fisher z of the untruncated ICC, which moves with the truncated ICC wherever that is positive
    positive = with_boot["icc"] > 0.01
    assert np.corrcoef(with_boot.loc[positive, "excess_fisher_z"], with_boot.loc[positive, "excess_icc"])[0, 1] > 0.9


def test_geneset_test_is_calibrated_with_coexpressed_clonal_genes():
    """Every gene equally clonal, genes co-expressed in modules: no set differs from its background.

    Treating genes as independent (the previous Mann-Whitney test) rejects far too often; the clone bootstrap
    keeps the correlation between genes and stays near the nominal 5 %.
    """
    boot, mw = [], []
    for seed in range(8):
        adata, sets = modules(seed, tau2=0.5)
        gs = tf.tl.geneset_heritability(heritability(adata, seed, n_boot=100), sets, seed=seed)
        boot.extend(gs["pvalue_vs_matched"])
        mw.extend(gs["pvalue_mw_genes_independent"])
    boot, mw = np.array(boot), np.array(mw)
    assert boot.size == 96
    assert (boot < 0.05).mean() <= 0.10
    assert (mw < 0.05).mean() >= 0.25


def test_geneset_test_is_calibrated_with_coexpressed_null_genes():
    """The statistical review's design: co-expressed modules, no clonal effect anywhere. The test does not
    depend on the number of permutations, because its centring is exact."""
    boot = []
    for seed in range(8):
        adata, sets = modules(100 + seed)
        gs = tf.tl.geneset_heritability(heritability(adata, seed, n_perm=5, n_boot=100), sets, seed=seed)
        boot.extend(gs["pvalue_vs_matched"])
    assert (np.array(boot) < 0.05).mean() <= 0.10


def test_geneset_test_detects_a_clonal_set():
    adata, sets = modules(7, n_clones=800, planted=(0,), planted_tau2=0.3)
    gs = tf.tl.geneset_heritability(heritability(adata, n_boot=100), sets).set_index("gene_set")
    planted = gs.loc["m0"]
    assert planted["pvalue_vs_matched"] < 1e-3 and planted["fdr"] < 0.01
    assert planted["ci_low"] > 0
    assert planted["ci_low"] < planted["difference"] < planted["ci_high"]
    assert gs["difference"].idxmax() == "m0"
    assert planted["median_icc"] > planted["matched_background_median_icc"]


def test_geneset_keeps_the_previous_test_for_comparison_only():
    adata, sets = modules(3, tau2=0.3)
    table = heritability(adata, n_boot=100)
    gs = tf.tl.geneset_heritability(table, sets, seed=5)
    # the previous implementation: Mann-Whitney against a random draw of up to 20 matched genes per set gene
    rng = np.random.default_rng(5)
    bins = pd.Series(pd.qcut(table["mean"].rank(method="first"), 10, labels=False), index=table.index)
    previous = []
    for genes in sets.values():
        others = table.index.difference(genes)
        background = []
        for b, k in bins.loc[genes].value_counts().items():
            pool = others[bins.loc[others].to_numpy() == b]
            background.extend(rng.choice(pool, size=min(pool.size, 20 * k), replace=False))
        previous.append(mannwhitneyu(table.loc[genes, "excess_icc"], table.loc[background, "excess_icc"],
                                     alternative="two-sided").pvalue)
    assert np.allclose(gs["pvalue_mw_genes_independent"], previous)
    assert gs["method"].str.contains("clone bootstrap").all()


def test_geneset_needs_the_bootstrap_for_a_table_read_from_disk():
    adata, sets = modules(4)
    table = heritability(adata, n_boot=100)
    reloaded = pd.read_csv(io.StringIO(table.to_csv()), index_col=0)
    with pytest.raises(ValueError, match="bootstrap"):
        tf.tl.geneset_heritability(reloaded, sets)
    again = tf.tl.geneset_heritability(reloaded, sets,
                                       bootstrap=adata.uns["threadfin"]["gene_heritability_bootstrap"])
    direct = tf.tl.geneset_heritability(table, sets)
    assert np.allclose(again["pvalue_vs_matched"], direct["pvalue_vs_matched"])
    with pytest.raises(ValueError, match="bootstrap"):                   # n_boot=0: nothing to test with
        tf.tl.geneset_heritability(heritability(adata, n_boot=0), sets)


def test_gene_heritability_says_shared_not_inherited(capsys):
    adata, _ = modules(2)
    tf.tl.gene_heritability(adata, clone_key="clone_id", context_key="sample", layer=None, n_perm=5,
                            n_boot=0, min_frac_expressed=0.0, verbose=True)
    out = capsys.readouterr().out
    assert "shared within clones" in out and "inherited" not in out


# --------------------------------------------------------------------------- export lag


def lag_data(n_clones=60, lag=0.004, slope=0.001, days=(10, 20, 30, 40, 50, 60), donors=3, seed=0):
    """A GC maturing at `slope` per day; output cells `lag` behind; mutation counts out of 270 positions."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clones):
        donor = f"D{c % donors}"
        day = days[c % len(days)]
        gc = 0.01 + slope * day + rng.normal(0, 0.002)
        for _ in range(int(rng.integers(1, 4))):
            rows.append((f"c{c}", donor, "GC", rng.binomial(270, gc) / 270, day, "IGHG1"))
        for _ in range(int(rng.integers(1, 3))):
            rows.append((f"c{c}", donor, "PB", rng.binomial(270, max(gc - lag, 0)) / 270, day, "IGHG1"))
    obs = pd.DataFrame(rows, columns=["clone_id", "donor", "state", "mutation_frequency", "timepoint", "isotype"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.zeros((len(obs), 1), dtype="float32"), obs=obs)


def run(adata, **kw):
    return tf.tl.export_lag(adata, state_key="state", source_states=["GC"], output_states=["PB"],
                            within=("timepoint",), **kw)


def test_median_interval_covers_with_ties_and_is_not_degenerate():
    """Mutation frequencies are counts out of a fixed length, so lags repeat; a bootstrap of the median then
    clusters on a few values. The order-statistic interval covers at least 95 % whatever the ties."""
    rng = np.random.default_rng(1)
    values = np.arange(-6, 4) / 270
    p = np.array([1, 2, 4, 8, 12, 14, 10, 6, 3, 1], float)
    p /= p.sum()
    truth = values[np.searchsorted(np.cumsum(p), 0.5)]
    hits, widths = 0, []
    for _ in range(400):
        x = rng.choice(values, size=80, p=p)
        lo, hi = _median_interval(x)
        hits += lo <= truth <= hi
        widths.append(hi - lo)
        assert lo <= np.median(x) <= hi
    assert hits / 400 >= 0.94
    assert np.mean(np.array(widths) > 0) > 0.9
    # without ties it is the textbook order-statistic interval: n = 20 gives x_(6) and x_(15)
    x = np.arange(20.0)
    assert _median_interval(x) == (5.0, 14.0)
    assert np.isnan(_median_interval(np.arange(5.0))[0])        # n < 6 cannot reach 95 %


def test_export_lag_reports_an_interval_and_a_sign_test():
    r = run(lag_data(n_clones=120))
    lo, hi = r["median_lag_ci"]
    assert lo <= r["median_lag"] <= hi and lo < hi
    assert lo <= -0.004 <= hi
    below, above = r["n_output_earlier"], r["n_output_later"]
    assert r["sign_test_pvalue"] == pytest.approx(binomtest(below, below + above, 0.5).pvalue)
    assert r["n_output_earlier"] + r["n_output_later"] + r["n_equal"] == r["n_clones"]


def test_days_come_with_an_interval():
    r = run(lag_data(n_clones=180, lag=0.004, slope=0.001), time_key="timepoint")
    assert r["n_timepoints"] == 6 and len(r["clock_points"]) == 6
    s_lo, s_hi = r["maturation_slope_ci"]
    assert s_lo <= 0.001 <= s_hi
    d_lo, d_hi = r["median_lag_days_ci"]
    assert d_lo < r["median_lag_days"] < d_hi
    assert d_lo <= -4.0 <= d_hi
    # the days interval is at least as wide as the gap interval divided by the rate
    lo, hi = r["median_lag_ci"]
    assert d_lo <= lo / r["maturation_slope"] and d_hi >= hi / r["maturation_slope"]


def test_days_interval_needs_five_timepoints():
    r = run(lag_data(days=(10, 20, 30, 40)), time_key="timepoint")
    assert np.isfinite(r["maturation_slope"]) and np.isfinite(r["median_lag_days"])
    assert np.isnan(r["maturation_slope_ci"][0]) and np.isnan(r["median_lag_days_ci"][0])


def test_clock_is_fitted_on_timepoint_medians_and_cells_remain_available():
    a = lag_data(n_clones=180)
    r = run(a, time_key="timepoint")
    pts = r["clock_points"]
    assert r["maturation_slope"] == pytest.approx(_theil_sen(pts["time"].to_numpy(float), pts["median"].to_numpy()))
    cells = run(a, time_key="timepoint", clock="cells")
    gc = a.obs[a.obs["state"] == "GC"]
    assert cells["maturation_slope"] == pytest.approx(
        _theil_sen(gc["timepoint"].to_numpy(float), gc["mutation_frequency"].to_numpy(float)))
    assert np.isnan(cells["median_lag_days_ci"][0])
    with pytest.raises(ValueError, match="clock"):
        run(a, time_key="timepoint", clock="pairs")


def test_sen_interval_is_exact_for_few_points():
    rng = np.random.default_rng(2)
    x = np.arange(6.0)
    hits = 0
    for _ in range(600):
        y = 0.5 * x + rng.standard_t(3, size=6)
        slope, lo, hi = _sen_interval(x, y)
        hits += lo <= 0.5 <= hi
        assert lo <= slope <= hi
    assert hits / 600 >= 0.95 - 0.02                            # exact level 0.983 for n = 6
    assert np.isnan(_sen_interval(np.arange(4.0), np.arange(4.0))[1])


def test_per_donor_estimates():
    a = lag_data(n_clones=180, donors=3)
    r = run(a, donor_key="donor")
    per = r["per_donor"].set_index("donor")
    assert per.index.tolist() == ["D0", "D1", "D2"] and per["n_clones"].sum() == r["n_clones"]
    pairs = r["pairs"]
    for donor, row in per.iterrows():
        assert row["median_lag"] == pytest.approx(pairs.loc[pairs["donor"] == donor, "lag"].median())
        assert row["ci_low"] <= row["median_lag"] <= row["ci_high"]
    assert np.isnan(r["median_lag_across_donors_ci"][0])         # three donors cannot give a 95 % interval
    six = run(lag_data(n_clones=180, donors=6), donor_key="donor")
    lo, hi = six["median_lag_across_donors_ci"]
    assert lo <= six["median_lag_across_donors"] <= hi
