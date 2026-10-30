"""Statistical properties of Threadfin's tests and estimators.

Each test guards one class of realistic bug: a permutation that ignores its strata, a p-value without the +1,
Benjamini-Hochberg without the step-up, an estimator that departs from its closed form, or a documented
option that is silently ignored. Calibration is checked under nulls that have the structure of real data
(clones nested in samples, labels whose frequency differs between donors), and each such test also shows
that the same data analysed without the strata would reject, so the test can see the bug it guards against.
The data are small simulations (``threadfin.sim``) or hand-built AnnData; the module runs in seconds.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.stats import binom, norm

import threadfin as tf
from threadfin._utils import bh_fdr, codes, nested_strata, perm_pvalue, permute_within
from threadfin.profiles import neighbour_average, variance_components

ALPHA = 0.05


def _max_rejections(n_rep: int, alpha: float = ALPHA) -> int:
    """Largest number of rejections at ``alpha`` that a valid test gives in ``n_rep`` replicates (99.9 %)."""
    return int(binom.ppf(0.999, n_rep, alpha))


def _centred(x: np.ndarray, groups) -> np.ndarray:
    """``x`` minus the mean of its group (computed here with pandas, independently of the package)."""
    x = np.asarray(x, dtype=float)
    return x - pd.DataFrame(x).groupby(np.asarray(groups)).transform("mean").to_numpy()


# --------------------------------------------------------------------------- permutation machinery


def test_permute_within_keeps_every_label_in_its_stratum():
    """Guards permutations that ignore their strata: labels stay in their stratum and missing labels stay put."""
    rng = np.random.default_rng(0)
    strata = np.repeat(np.arange(4), 25)
    labels = rng.integers(0, 5, size=strata.size)
    labels[::7] = -1                                            # missing labels
    moved = False
    for _ in range(20):
        perm = permute_within(labels, strata, rng)
        assert np.array_equal(perm == -1, labels == -1)
        for s in range(4):
            assert sorted(perm[strata == s]) == sorted(labels[strata == s])
        moved |= not np.array_equal(perm, labels)
    assert moved
    # text labels with missing values take the same path
    text = np.array(["a", "b", None, "c", "a", "b"], dtype=object)
    perm = permute_within(text, np.array([0, 0, 0, 1, 1, 1]), rng)
    assert perm[2] is None and sorted(perm[:2]) == ["a", "b"] and sorted(perm[3:]) == ["a", "b", "c"]
    # the check above can fail: without strata, labels cross between strata
    glob = permute_within(labels, None, np.random.default_rng(1))
    assert any(sorted(glob[strata == s]) != sorted(labels[strata == s]) for s in range(4))


def test_perm_pvalue_counts_the_observed_statistic_as_one_permutation():
    """Guards a permutation p-value without the +1: p = (1 + b) / (1 + n), never below 1 / (n + 1)."""
    # ties with the observed statistic count as at least as extreme
    null = np.array([0.125, 0.5, 0.5, 0.875, 1.25, np.nan])     # the NaN replicate is dropped: n = 5
    assert perm_pvalue(0.5, null) == pytest.approx((1 + 4) / 6)
    assert perm_pvalue(2.0, null) == pytest.approx(1 / 6)       # beyond every permutation: 1 / (n + 1), not 0
    assert perm_pvalue(0.0, null) == pytest.approx(1.0)
    assert perm_pvalue(0.5, null, "less") == pytest.approx((1 + 3) / 6)
    # two-sided: distance from the median of the null (0.5)
    assert perm_pvalue(1.5, null, "two-sided") == pytest.approx(1 / 6)
    assert perm_pvalue(0.125, null, "two-sided") == pytest.approx((1 + 3) / 6)
    assert np.isnan(perm_pvalue(1.0, np.array([np.nan])))


def _bh_by_definition(p):
    """BH adjusted p-values straight from the definition: min over j >= i of m p_(j) / j, capped at 1."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    order = ok[np.argsort(p[ok], kind="stable")]
    for i, idx in enumerate(order):
        out[idx] = min(1.0, min(m * p[order[j]] / (j + 1) for j in range(i, m)))
    return out


def test_bh_fdr_matches_a_hand_computed_step_up():
    """Guards Benjamini-Hochberg without the step-up: hand-worked example and definition, in any input order."""
    p = np.array([0.020, 0.010, np.nan, 0.5, 0.021])
    # sorted 0.010, 0.020, 0.021, 0.5 (m = 4): m p / rank = 0.04, 0.04, 0.028, 0.5; the step-up takes the
    # minimum over larger ranks, so the first three all become 0.028
    np.testing.assert_allclose(bh_fdr(p), [0.028, 0.028, np.nan, 0.5, 0.028], rtol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(size=12) ** 3
        q[rng.random(12) < 0.15] = np.nan
        np.testing.assert_allclose(bh_fdr(q), _bh_by_definition(q), rtol=1e-12)
        order = rng.permutation(q.size)
        np.testing.assert_allclose(bh_fdr(q[order]), bh_fdr(q)[order], rtol=1e-12)


# --------------------------------------------------------------------------- estimators


def test_variance_components_match_the_closed_form_unbalanced_anova():
    """Guards a balanced n0 and an untruncated tau2: closed-form unbalanced one-way ANOVA on a hand-worked design."""
    # sigma2 = MSW, tau2 = max((MSB - MSW) / n0, 0), n0 = (N - sum n_c^2 / N) / (C - 1)
    # clones of 2, 3 and 5 cells; a singleton (code 3) and a cell without a clone (-1) must be ignored
    codes = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, -1])
    f1 = [1, 3, 4, 6, 8, 0, 1, 2, 3, 4, 100, 50]               # clone means 2, 6, 2: between-clone signal
    f2 = [0, 10, 2, 5, 8, 1, 9, 5, 3, 7, -40, 70]              # clone means 5, 5, 5: none
    vc = variance_components(np.column_stack([f1, f2]).astype(float), codes)
    # by hand: N = 10 cells, C = 3 clones, grand means 3.2 and 5
    #   f1: SSW = 2 + 8 + 10 = 20 -> MSW = 20 / 7;  SSB = 2(1.2^2) + 3(2.8^2) + 5(1.2^2) = 33.6 -> MSB = 16.8
    #   f2: SSW = 50 + 18 + 40 = 108 -> MSW = 108 / 7;  SSB = 0 -> (MSB - MSW) / n0 < 0, truncated to 0
    #   n0 = (10 - (4 + 9 + 25) / 10) / 2 = 3.1   (the mean clone size, 10 / 3, would be wrong)
    msw = np.array([20 / 7, 108 / 7])
    tau2_1 = (16.8 - 20 / 7) / 3.1
    assert vc.n_clones == 3 and vc.n_cells == 10
    assert vc.n0 == pytest.approx(3.1)
    np.testing.assert_allclose(vc.sigma2, msw, rtol=1e-12)
    np.testing.assert_allclose(vc.tau2, [tau2_1, 0.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vc.icc_per_feature, [tau2_1 / (tau2_1 + msw[0]), 0.0], atol=1e-12)
    assert vc.icc == pytest.approx(tau2_1 / (tau2_1 + msw.sum()))
    # shrinkage of a clone of n cells: n tau2 / (n tau2 + sigma2), zero where tau2 = 0
    np.testing.assert_allclose(vc.shrinkage([4])[0], [4 * tau2_1 / (4 * tau2_1 + msw[0]), 0.0], atol=1e-12)


def test_neighbour_average_uses_k_cells_including_the_cell_itself():
    """Guards smoothing with k - 1 neighbours: each cell is the mean of its k nearest cells, itself included."""
    # points chosen so that every neighbour set is unambiguous
    x = np.array([[0.0], [1.0], [3.0], [7.0], [15.0]])
    np.testing.assert_allclose(neighbour_average(x, 1), x)
    np.testing.assert_allclose(neighbour_average(x, 2)[:, 0], [0.5, 0.5, 2.0, 5.0, 11.0])
    np.testing.assert_allclose(neighbour_average(x, 3)[:, 0], [4 / 3, 4 / 3, 4 / 3, 11 / 3, 25 / 3])
    np.testing.assert_allclose(neighbour_average(x, 50)[:, 0], np.full(5, x.mean()))   # k is capped at n


def test_receptor_genes_never_enter_the_highly_variable_genes():
    """Guards IG/TR genes kept in the embedding: none is highly variable, though they are the most variable genes."""
    # genes that only look like receptor genes must not be flagged
    rng = np.random.default_rng(0)
    n_cells, n_background = 400, 150
    receptor = ["IGHV3-23", "IGHV1-69", "IGKV1-5", "IGLV2-14", "IGHG1", "IGHM", "IGKC", "IGLC2",
                "TRBV7-2", "TRAC"]
    lookalikes = ["IGHMBP2", "JCHAIN", "IGLON5", "TRAF3"]
    clone = rng.integers(0, 15, n_cells)
    background = rng.poisson(rng.uniform(0.5, 5.0, n_background + len(lookalikes)),
                             size=(n_cells, n_background + len(lookalikes)))
    on = rng.random((15, len(receptor))) < 0.3                  # clone-specific receptor expression
    counts = np.hstack([background, rng.poisson(np.where(on[clone], 12.0, 0.05))]).astype(np.float32)
    genes = [f"G{i}" for i in range(n_background)] + lookalikes + receptor
    adata = AnnData(counts, var=pd.DataFrame(index=genes), obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]))
    kept = adata.copy()
    tf.pp.prepare_embedding(kept, n_top_genes=12, n_comps=5, integrate=None, exclude_receptor_genes=False,
                            verbose=False)
    assert kept.var.loc[receptor, "highly_variable"].sum() >= len(receptor) // 2   # the fixture can see the bug
    tf.pp.prepare_embedding(adata, n_top_genes=12, n_comps=5, integrate=None, verbose=False)
    assert not adata.var.loc[receptor, "highly_variable"].any()
    assert adata.var["highly_variable"].sum() == 12
    assert set(adata.var.index[adata.var["threadfin_receptor_gene"]]) == set(receptor)
    # a user's own HVG selection: exclude_ig_genes drops the receptor genes and keeps every other gene
    others = kept.var.index[kept.var["highly_variable"] & ~kept.var.index.isin(receptor)]
    tf.pp.exclude_ig_genes(kept, tr_genes=True, verbose=False)
    assert set(kept.var.index[kept.var["highly_variable"]]) == set(others)


# --------------------------------------------------------------------------- clonal coherence


def _nested_null(seed: int):
    """Null repertoire: every clone sampled in one context, contexts shifted strongly, no clonal state."""
    return tf.sim.simulate_repertoire(scenario="null", n_clones=250, n_donors=2, contexts_per_donor=3,
                                      context_shift=3.0, dim=8, random_state=seed)


def test_coherence_is_calibrated_within_strata_and_not_without():
    """Guards coherence permutations that ignore their strata: calibrated within contexts, rejecting across them."""
    # profiles are not centred on the contexts, so the context shifts inflate the ICC itself; only the
    # permutation strata keep the test calibrated
    n_rep, n_perm = 20, 49
    p_within, p_across = [], []
    for seed in range(n_rep):
        adata = _nested_null(seed)
        tf.tl.clone_profiles(adata, basis="X_pca", context_key=None, donor_key="donor", verbose=False)
        res = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=n_perm, random_state=seed, verbose=False)
        assert res["strata_key"] == "context"
        p_within.append(res["p_value"])
        p_across.append(tf.tl.clonal_coherence(adata, strata_key=None, n_perm=n_perm, random_state=seed,
                                               verbose=False)["p_value"])
    p_within, p_across = np.array(p_within), np.array(p_across)
    assert (p_within <= ALPHA).sum() <= _max_rejections(n_rep)
    assert 0.25 < p_within.mean() < 0.75
    assert (p_across <= ALPHA).all()


def test_coherence_is_calibrated_in_the_default_design():
    """Guards an anti-conservative coherence test in the run() design: null rejections within binomial limits."""
    # the run() design: profiles centred on the sampling context, clone labels shuffled within it
    n_rep, n_perm = 20, 49
    pvals = []
    for seed in range(n_rep):
        adata = _nested_null(seed)
        tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
        res = tf.tl.clonal_coherence(adata, n_perm=n_perm, random_state=seed, verbose=False)
        null = np.asarray(res["null"])
        assert res["p_value"] == pytest.approx((1 + np.sum(null >= res["icc"])) / (n_perm + 1))
        assert res["p_value"] >= 1 / (n_perm + 1)
        pvals.append(res["p_value"])
    pvals = np.array(pvals)
    assert (pvals <= ALPHA).sum() <= _max_rejections(n_rep)
    assert 0.25 < pvals.mean() < 0.75


def test_coherence_strata_default_to_the_profile_context():
    """Guards coherence permuting globally when no strata_key is given: the default is the profiles' context."""
    adata = _nested_null(0)
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    default = tf.tl.clonal_coherence(adata, n_perm=30, random_state=1, verbose=False)
    explicit = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=30, random_state=1, verbose=False)
    assert default["strata_key"] == "context"
    np.testing.assert_array_equal(np.asarray(default["null"]), np.asarray(explicit["null"]))
    assert default["p_value"] == explicit["p_value"]


def test_coherence_strata_stay_within_the_profile_context():
    """Guards clone labels shuffled between donors when one library pools several donors."""
    adata = _nested_null(0)
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="donor", donor_key="donor", verbose=False)
    # library S<k> pools sample S<k> of every donor
    adata.obs["library"] = adata.obs["context"].astype(str).str.split("_").str[1]
    clone, donor = codes(adata.obs["clone_id"])[0], codes(adata.obs["donor"])[0]
    # the fixture can see the bug: shuffled within the raw libraries, clone labels land in both donors
    raw = permute_within(clone, codes(adata.obs["library"])[0], np.random.default_rng(0))
    assert (pd.Series(donor).groupby(raw).nunique() > 1).any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # strata nested in the donors: no warning
        nested = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=30, random_state=1, verbose=False)
    assert nested["strata_key"] == "context"
    with pytest.warns(UserWarning, match="pool several donors"):
        pooled = tf.tl.clonal_coherence(adata, strata_key="library", n_perm=30, random_state=1, verbose=False)
    assert pooled["strata_key"] == "donor x library"
    # donor x library is the partition into contexts: the same null, draw for draw
    np.testing.assert_array_equal(np.asarray(pooled["null"]), np.asarray(nested["null"]))
    with pytest.warns(UserWarning, match="pool several donors"):
        strata, name = nested_strata(adata.obs, "library", "donor")
    assert name == "donor x library"
    assert (pd.Series(donor).groupby(strata).nunique() == 1).all()
    np.testing.assert_array_equal(strata, codes(adata.obs["context"])[0])


def test_coherence_is_measured_on_unsmoothed_context_centred_cells():
    """Guards coherence computed on smoothed cells: the ICC is that of the context-centred embedding itself."""
    adata = tf.sim.simulate_repertoire(n_clones=300, random_state=4)
    clone_codes = pd.factorize(adata.obs["clone_id"])[0]
    expected = variance_components(_centred(adata.obsm["X_pca"], adata.obs["context"]), clone_codes).icc
    for kwargs in ({"representation": "mean"}, {"representation": "mean", "smooth": 10},
                   {"representation": "kernel"}):              # kernel profiles smooth over 15 cells by default
        tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False,
                             **kwargs)
        assert tf.tl.clonal_coherence(adata, n_perm=5, verbose=False)["icc"] == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- profile association


def _clone_labels_for(adata, clone_values: pd.Series, column: str) -> None:
    """Write a clone-level label to every cell of the clone (cells of other clones get NaN)."""
    adata.obs[column] = adata.obs["clone_id"].astype(str).map(clone_values).to_numpy()


def test_profile_association_is_calibrated_within_donors_and_keeps_power():
    """Guards profile_association permuting labels across donors: calibrated within donors, with power kept."""
    # the label's frequency differs between donors but it is random within each donor
    adata = tf.sim.simulate_repertoire(scenario="null", n_clones=600, n_donors=2, contexts_per_donor=1,
                                       context_shift=3.0, dim=6, random_state=11)
    # no centring: the clone profiles keep their donor's shift, so a donor-confounded label tracks them
    tf.tl.clone_profiles(adata, basis="X_pca", context_key=None, donor_key="donor", verbose=False)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    rng = np.random.default_rng(0)
    p_yes = table["donor"].map({"D0": 0.8, "D1": 0.2}).to_numpy()
    n_rep, n_perm = 40, 99
    p_within, p_across = [], []
    for rep in range(n_rep):
        labels = pd.Series(np.where(rng.random(len(table)) < p_yes, "yes", "no"), index=table.index)
        _clone_labels_for(adata, labels, "label")
        res = tf.tl.profile_association(adata, "label", n_perm=n_perm, random_state=rep, verbose=False)
        assert res["strata_key"] == "donor" and res["p_value"] >= 1 / (n_perm + 1)
        p_within.append(res["p_value"])
        p_across.append(tf.tl.profile_association(adata, "label", strata_key=None, n_perm=n_perm, random_state=rep,
                                                  verbose=False)["p_value"])
    p_within, p_across = np.array(p_within), np.array(p_across)
    assert (p_within <= ALPHA).sum() <= _max_rejections(n_rep)
    assert 0.25 < p_within.mean() < 0.75
    assert (p_across <= ALPHA).mean() >= 0.9

    # power: the same kind of label, now shifting the states of its clones' cells
    planted = tf.sim.simulate_repertoire(scenario="null", n_clones=600, n_donors=2, contexts_per_donor=1,
                                         context_shift=3.0, dim=6, random_state=11)
    clone_ids = planted.obs["clone_id"].astype(str)
    donor_of = planted.obs.groupby(clone_ids.to_numpy(), observed=True)["donor"].first()
    hit = pd.Series(rng.random(len(donor_of)) < donor_of.map({"D0": 0.8, "D1": 0.2}).to_numpy(),
                    index=donor_of.index)
    _clone_labels_for(planted, hit.map({True: "yes", False: "no"}), "label")
    planted.obsm["X_pca"] = planted.obsm["X_pca"].copy()
    planted.obsm["X_pca"][:, 0] += 1.0 * (planted.obs["label"] == "yes").to_numpy()
    tf.tl.clone_profiles(planted, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    res = tf.tl.profile_association(planted, "label", n_perm=n_perm, verbose=False)
    assert res["p_value"] <= 2 / (n_perm + 1) and res["excess_r2"] > 0


# --------------------------------------------------------------------------- programme x label association


def _association_adata(rows, seed=0):
    """Two cells per clone; ``rows`` = (clone, donor, programme, label). Profiles of a random embedding, with
    the programmes written into the clone table as find_programmes would."""
    obs = pd.DataFrame([{"clone_id": c, "donor": d, "label": lab} for c, d, _, lab in rows for _ in range(2)])
    obs.index = [f"cell{i}" for i in range(len(obs))]
    adata = AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.random.default_rng(seed).normal(size=(len(obs), 3))
    tf.tl.clone_profiles(adata, basis="X_pca", donor_key="donor", verbose=False)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    table["clone_programme"] = pd.Series({c: p for c, _, p, _ in rows}).reindex(table.index).to_numpy()
    return adata


def _rows_from_tables(tables: dict):
    """Clones laid out from per-donor 2 x 2 tables {donor: (P1 yes, P1 no, P2 yes, P2 no)}."""
    rows = []
    for donor, cells in tables.items():
        for (prog, lab), n in zip((("P1", "yes"), ("P1", "no"), ("P2", "yes"), ("P2", "no")), cells):
            rows += [(f"{donor}_{prog}_{lab}_{i}", donor, prog, lab) for i in range(n)]
    return rows


def _mantel_haenszel_by_hand(tables):
    """MH odds ratio and Robins-Breslow-Greenland 95 % interval, term by term, from (a, b, c, d) per stratum."""
    r_sum = s_sum = pr = ps_qr = qs = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r, s, p, q = a * d / n, b * c / n, (a + d) / n, (b + c) / n
        r_sum, s_sum = r_sum + r, s_sum + s
        pr, ps_qr, qs = pr + p * r, ps_qr + p * s + q * r, qs + q * s
    se = np.sqrt(pr / (2 * r_sum**2) + ps_qr / (2 * r_sum * s_sum) + qs / (2 * s_sum**2))
    z = norm.ppf(0.975)
    return r_sum / s_sum, np.exp(np.log(r_sum / s_sum) - z * se), np.exp(np.log(r_sum / s_sum) + z * se)


def test_association_counts_expectation_and_mantel_haenszel_match_hand_computation():
    """Guards an inverted Mantel-Haenszel odds ratio and an unstratified expectation: both match a hand computation."""
    # two donors with different margins; the RBG interval and the BH FDR are checked too
    tables = {"d1": (5, 1, 2, 4), "d2": (2, 4, 3, 11)}         # (P1 yes, P1 no, P2 yes, P2 no) per donor
    n_perm = 199
    res = tf.tl.association_test(_association_adata(_rows_from_tables(tables)), "label", n_perm=n_perm,
                                 verbose=False)
    assert res.attrs["strata_key"] == "donor"
    rows = res.set_index(["programme", "level"])
    assert {("P1", "yes"), ("P2", "yes")} <= set(rows.index)  # a two-level label may report its rarer level only
    # each (programme, level) reorders the four cells as (in & level, in & other, out & level, out & other)
    reorder = {("P1", "yes"): (0, 1, 2, 3), ("P1", "no"): (1, 0, 3, 2),
               ("P2", "yes"): (2, 3, 0, 1), ("P2", "no"): (3, 2, 1, 0)}
    for key, row in rows.iterrows():
        per_donor = [tuple(cells[i] for i in reorder[key]) for cells in tables.values()]
        a = sum(t[0] for t in per_donor)
        # expectation under shuffling within donors: sum over donors of (row total x column total / n)
        expected = sum((t[0] + t[1]) * (t[0] + t[2]) / sum(t) for t in per_donor)
        or_mh, lo, hi = _mantel_haenszel_by_hand(per_donor)
        assert row["observed"] == a
        assert row["expected"] == pytest.approx(expected)
        assert row["odds_ratio"] == pytest.approx(or_mh, rel=1e-9)
        assert row["ci_low"] == pytest.approx(lo, rel=1e-6) and row["ci_high"] == pytest.approx(hi, rel=1e-6)
        assert row["pvalue"] >= 1 / (n_perm + 1)
    # P1 is enriched for "yes" within both donors: OR = (5*4/12 + 2*11/20) / (1*2/12 + 4*3/20) = 3.61, not 1/3.61
    odds_ratio = (5 * 4 / 12 + 2 * 11 / 20) / (1 * 2 / 12 + 4 * 3 / 20)
    assert rows.loc[("P1", "yes"), "odds_ratio"] == pytest.approx(odds_ratio)
    # pooled over donors the expectation would be 12 x 12 / 32 = 4.5; within donors it is 6*7/12 + 6*5/20 = 5
    assert rows.loc[("P1", "yes"), "expected"] == pytest.approx(5.0)
    np.testing.assert_allclose(res["fdr"], bh_fdr(res["pvalue"].to_numpy()))


def test_association_p_value_counts_ties_as_at_least_as_extreme():
    """Guards a permutation p-value that ignores ties: observed counts equal to their expectation give p = 1."""
    # no shuffle can be less extreme than a deviation of zero
    tables = {"d1": (2, 2, 2, 2), "d2": (2, 2, 2, 2), "d3": (2, 2, 2, 2)}
    res = tf.tl.association_test(_association_adata(_rows_from_tables(tables)), "label", n_perm=200, verbose=False)
    np.testing.assert_allclose(res["expected"], res["observed"])
    assert (res["pvalue"] == 1.0).all()
    np.testing.assert_allclose(res["odds_ratio"], 1.0)


def test_association_is_calibrated_within_donors():
    """Guards an anti-conservative programme-by-label test: a Simpson's-paradox null is calibrated within donors."""
    # programmes and labels both follow the donor but are independent within donors
    rows = []
    for donor, n_p1, n_p2 in (("d1", 30, 10), ("d2", 10, 30)):
        rows += [(f"{donor}_c{i}", donor, "P1" if i < n_p1 else "P2", "no") for i in range(n_p1 + n_p2)]
    adata = _association_adata(rows)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    p_yes = table["donor"].map({"d1": 0.8, "d2": 0.2}).to_numpy()
    rng = np.random.default_rng(1)
    n_rep, n_perm = 30, 99
    p_within, p_across = [], []
    for rep in range(n_rep):
        labels = pd.Series(np.where(rng.random(len(table)) < p_yes, "yes", "no"), index=table.index)
        _clone_labels_for(adata, labels, "label")
        # with two programmes and two levels every row carries the same p-value; take programme P1's
        res = tf.tl.association_test(adata, "label", n_perm=n_perm, random_state=rep, verbose=False)
        p_within.append(res.loc[res["programme"] == "P1", "pvalue"].iloc[0])
        res = tf.tl.association_test(adata, "label", strata_key=None, n_perm=n_perm, random_state=rep, verbose=False)
        p_across.append(res.loc[res["programme"] == "P1", "pvalue"].iloc[0])
    p_within, p_across = np.array(p_within), np.array(p_across)
    assert (p_within <= ALPHA).sum() <= _max_rejections(n_rep)
    assert (p_across <= ALPHA).mean() >= 0.5


# --------------------------------------------------------------------------- clonal memory


def _memory_adata(memory: bool, seed: int, n_clones: int = 100, tau: float = 1.0, sigma: float = 2.0, dim: int = 4):
    """Two donors x two time points (one context each, strongly shifted). Each clone has a true state drawn
    with variance tau^2; at t1 it keeps that state (memory) or draws a new one (no memory). Cells scatter
    around it with variance sigma^2, and with 2-5 cells per snapshot the sampling noise of a snapshot mean is
    as large as the differences between clones."""
    rng = np.random.default_rng(seed)
    rows, x = [], []
    for d in range(2):
        shift = rng.normal(0, 2.0, size=(2, dim))
        for c in range(n_clones):
            u0 = rng.normal(0, tau, dim)
            u1 = u0 if memory else rng.normal(0, tau, dim)
            for t, u in enumerate((u0, u1)):
                for _ in range(int(rng.integers(2, 6))):
                    x.append(u + shift[t] + rng.normal(0, sigma, dim))
                    rows.append({"clone_id": f"d{d}_c{c}", "donor": f"d{d}", "time": f"t{t}",
                                 "context": f"d{d}_t{t}"})
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    adata = AnnData(np.asarray(x, dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()
    return adata


def _uncorrected_memory_index(adata) -> float:
    """1 - mean squared change of a clone / mean squared distance to another clone of its donor, from raw
    context-centred snapshot means (no sampling-noise correction), computed by hand."""
    centred = _centred(adata.obsm["X_pca"], adata.obs["context"])
    snap = pd.DataFrame(centred).groupby([adata.obs["clone_id"].to_numpy(), adata.obs["time"].to_numpy()]).mean()
    m0, m1 = snap.xs("t0", level=1), snap.xs("t1", level=1)
    m1 = m1.loc[m0.index]
    donor = m0.index.str.split("_").str[0].to_numpy()
    same = ((m0.to_numpy() - m1.to_numpy()) ** 2).sum(axis=1).mean()
    dist = ((m0.to_numpy()[:, None, :] - m1.to_numpy()[None, :, :]) ** 2).sum(axis=-1)
    other = (donor[:, None] == donor[None, :]) & ~np.eye(donor.size, dtype=bool)
    return float(1.0 - same / dist[other].mean())


def _memory(adata, seed):
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.tl.clonal_memory(adata, "time", n_null=99, n_boot=20, random_state=seed, verbose=False)


def test_memory_index_is_near_zero_without_memory_and_near_one_with_perfect_memory():
    """Guards a memory noise correction of the wrong sign: M near 0 without memory, near 1 with perfect memory."""
    # the snapshot noise is as large as the clonal differences, so the uncorrected index of the perfect-memory
    # data is about 0.4; tolerances are generous because dynamics.py is being revised
    for seed in range(3):
        none = _memory(_memory_adata(memory=False, seed=seed), seed)
        assert abs(none["memory_index"]) < 0.4
        assert none["p_value"] >= 1 / 100
        perfect_data = _memory_adata(memory=True, seed=seed)
        assert _uncorrected_memory_index(perfect_data) < 0.65    # the correction matters in this fixture
        perfect = _memory(perfect_data, seed)
        assert 0.7 < perfect["memory_index"] < 1.3
        assert perfect["p_value"] <= ALPHA


# --------------------------------------------------------------------------- programmes


def _programme_adata():
    adata = tf.sim.simulate_repertoire(n_clones=600, clone_size_exponent=1.9, random_state=3)
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    return adata


def test_assignment_respects_min_posterior():
    """Guards assignment that ignores min_posterior: a clone outside the core needs posterior >= min_posterior."""
    adata = _programme_adata()
    floor = float(adata.uns["threadfin"]["profiles"]["clone_table"]["reliability"].median())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.tl.find_programmes(adata, resolution=1.0, n_boot=0, test_splits=False, min_reliability=floor,
                              min_posterior=0.9, verbose=False)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    rest = table[~table["core"]]
    post, assigned = rest["assignment_posterior"], rest["clone_programme_assigned"]
    assert (post < 0.9).any() and (post >= 0.9).any()          # the fixture has clones on both sides
    assert assigned[post < 0.9].isna().all()
    assert assigned[post >= 0.9].notna().all()
    core = table[table["core"]]
    assert (core["clone_programme_assigned"].astype(str) == core["clone_programme"].astype(str)).all()


def test_find_programmes_passes_reliabilities_to_the_clone_graph(monkeypatch):
    """Guards a programme graph that ignores the reliability weights: the clone graph receives them."""
    from threadfin import programmes

    adata = _programme_adata()
    calls = []
    original = programmes.knn_graph

    def spy(features, n_neighbors=15, reliability=None):
        calls.append(None if reliability is None else np.asarray(reliability, dtype=float).copy())
        return original(features, n_neighbors, reliability)

    monkeypatch.setattr(programmes, "knn_graph", spy)
    kwargs = {"resolution": 1.0, "n_boot": 0, "test_splits": False, "assign_remaining": False, "verbose": False}
    tf.tl.find_programmes(adata, **kwargs)
    table = adata.uns["threadfin"]["profiles"]["clone_table"]
    assert len(calls) == 1 and calls[0] is not None
    np.testing.assert_allclose(calls[0], table["reliability"].to_numpy())
    calls.clear()
    tf.tl.find_programmes(adata, weight_by_reliability=False, **kwargs)
    assert len(calls) == 1 and calls[0] is None


def test_knn_graph_weights_are_gaussian_and_scaled_by_reliability():
    """Guards an unweighted programme graph: edges carry Gaussian weights times sqrt(rel_i * rel_j)."""
    # weight exp(-d^2 / sigma_i^2), sigma_i the distance to the k-th neighbour; the larger direction is kept
    from threadfin.programmes import knn_graph

    x = np.array([[0.0], [1.0], [3.0], [7.0], [12.0]])          # no ties among neighbour distances
    k = 2
    expected = {}
    for i in range(len(x)):
        d = np.abs(x[:, 0] - x[i, 0])
        d[i] = np.inf
        nbrs = np.argsort(d)[:k]
        sigma = d[nbrs[-1]]
        for j in nbrs:
            key = (min(i, j), max(i, j))
            expected[key] = max(expected.get(key, 0.0), float(np.exp(-d[j] ** 2 / sigma**2)))
    rel = np.array([1.0, 0.25, 1.0, 0.64, 1.0])
    for reliability in (None, rel):
        g = knn_graph(x, n_neighbors=k, reliability=reliability)
        got = {tuple(sorted(e)): w for e, w in zip(g.get_edgelist(), g.es["weight"])}
        assert set(got) == set(expected)
        for (i, j), w in expected.items():
            scale = 1.0 if reliability is None else np.sqrt(rel[i] * rel[j])
            assert got[(i, j)] == pytest.approx(w * scale, rel=1e-12)
