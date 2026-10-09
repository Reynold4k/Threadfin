"""Statistical contracts: fixed-reference inference, count uncertainty and leakage boundaries."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta
from scipy.integrate import trapezoid

import threadfin as tf
from threadfin.density import StateDensityModel


@pytest.fixture
def reference():
    rng = np.random.default_rng(11)
    x = np.r_[rng.normal(-2, .15, (80, 1)), rng.normal(0, .15, (80, 1)),
              rng.normal(2, .15, (80, 1))]
    clones = np.tile(np.repeat([f"r{i}" for i in range(8)], 10), 3)
    return x, clones


def test_posterior_agrees_with_binomial_conjugacy(reference):
    x, clones = reference
    model = StateDensityModel.fit(x, clones, n_states=2, prior_strength=4)
    query = np.array([[-2.], [-2.], [2.]])
    result = model.transform(query, ["q"] * 3)
    observed = result.counts.loc["q"].to_numpy()
    a = 4 * result.background.loc["q"].to_numpy()
    expected = (observed + a) / 7
    np.testing.assert_allclose(result.proportions.loc["q"], expected)
    np.testing.assert_allclose(result.lower.loc["q"], beta.ppf(.025, observed+a, 7-observed-a))
    assert result.clone_table.loc["q", "data_weight"] == 3/7


def test_clone_balance_and_leave_whole_clone_out():
    # One huge clone must not dominate the background. All clone A contexts are excluded.
    x = np.r_[np.full((1000, 1), -2.), [[0.], [2.]]]
    labels = ["A"]*1000 + ["B", "C"]
    ctx = ["left"]*500 + ["right"]*500 + ["right", "right"]
    model = StateDensityModel.fit(x, labels, contexts=ctx, n_states=3, background_floor=.06)
    result = model.transform([[-2.]], ["A"], contexts=["left"])
    order = np.argsort(model.centres[:, 0])
    np.testing.assert_allclose(result.background.loc["A"].to_numpy()[order], [.02, .49, .49])
    assert result.clone_table.loc["A", "global_background_fraction"] == 1
    assert result.clone_table.loc["A", "own_clone_excluded_fraction"] == 1
    new = model.transform([[-2.]], ["new"], contexts=["unknown"])
    np.testing.assert_allclose(new.background.loc["new"], np.repeat(1/3, 3))


def test_different_compositions_with_identical_centroids():
    x = np.repeat([[-2.], [0.], [2.]], 20, axis=0)
    model = StateDensityModel.fit(x, np.repeat(["a", "b", "c"], 20), n_states=3)
    mixed = np.r_[np.full((20, 1), -2.), np.full((20, 1), 2.)]
    middle = np.zeros((40, 1))
    assert mixed.mean() == middle.mean()
    result = model.transform(np.r_[mixed, middle], ["mixed"]*40 + ["middle"]*40)
    assert np.abs(result.proportions.loc["mixed"] - result.proportions.loc["middle"]).sum()/2 > .9
    contrast = result.contrast("mixed", "middle", n_draws=2000)
    center = model.state_names[np.argmin(np.abs(model.centres[:, 0]))]
    assert contrast.loc[center, "upper"] < -.7
    assert (result.contrast("mixed", "mixed")["difference"] == 0).all()


def test_streaming_and_roundtrip_are_exact(reference, tmp_path):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3, prior_strength="auto", batch_size=11)
    before = model.metadata()
    expected = model.transform(x, ids)
    actual = model.transform_batches((x[i:i+7], ids[i:i+7], None) for i in range(0, len(x), 7))
    for name in ("counts", "proportions", "lower", "upper", "background"):
        pd.testing.assert_frame_equal(getattr(expected, name), getattr(actual, name))
    assert model.metadata() == before  # queries cannot retune model
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = StateDensityModel.load(path)
    assert loaded.metadata() == before
    pd.testing.assert_frame_equal(expected.proportions, loaded.transform(x, ids).proportions)


def test_intervals_contract_at_large_counts(reference):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3)
    query = np.r_[x[:2], np.tile(x, (4, 1))]
    labels = ["small"]*2 + ["large"]*(len(query)-2)
    result = model.transform(query, labels)
    width = result.upper - result.lower
    assert width.loc["large"].mean() < width.loc["small"].mean()
    assert result.clone_table.loc["large", "data_weight"] > .99
    assert np.isfinite(result.lower.to_numpy()).all()
    assert ((result.upper-result.lower).to_numpy() >= 0).all()


def test_normalised_smooth_density(reference):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3, scales=(.5, 1, 2))
    grid = np.linspace(-6, 6, 20001)
    density = np.exp(model.log_density(grid[:, None], [.2, .5, .3]))
    assert trapezoid(density, grid) == pytest.approx(1, abs=1e-4)
    assert np.isfinite(model.log_density([[1e5]], [0, 1, 0])).all()


def test_anndata_wrapper_writes_and_keeps_legacy_results(reference, tmp_path):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3)
    a = ad.AnnData(np.zeros((len(x), 1)))
    a.obsm["X_pca"] = x
    a.obs["clone_id"] = ids
    a.uns["threadfin"] = {"profiles": {"sentinel": 123}}
    result = tf.tl.clone_densities(a, reference_model=model)
    assert a.uns["threadfin"]["profiles"]["sentinel"] == 123
    path = tmp_path / "result.h5ad"
    a.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    pd.testing.assert_frame_equal(loaded.uns["threadfin"]["densities"]["proportions"], result.proportions)


@pytest.mark.parametrize("setting,value", [
    ("prior_strength", 0), ("background_floor", 0), ("n_states", 1),
    ("scales", [0, 1]), ("batch_size", 0),
])
def test_invalid_reference_settings(reference, setting, value):
    x, ids = reference
    kwargs = {"n_states": 3, setting: value}
    with pytest.raises(ValueError):
        StateDensityModel.fit(x, ids, **kwargs)


def test_missing_clones_and_mismatched_coordinates_are_rejected(reference):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3)
    with pytest.raises(ValueError, match="label"):
        model.transform([[0]], [None])
    with pytest.raises(ValueError, match="coordinates"):
        model.transform([[0, 1]], ["q"])
    with pytest.raises(ValueError):
        model.transform_batches([])


def test_donor_inputs_and_empty_gene_scan_fail_cleanly(reference):
    from threadfin.clones import define_clones
    x, ids = reference
    tab = pd.DataFrame({"donor": ["d", None], "v_call": ["IGHV1"]*2,
                        "j_call": ["IGHJ1"]*2, "junction": ["ACTG"]*2})
    with pytest.raises(ValueError, match="donor"):
        define_clones(tab, threshold=.1)
    a = ad.AnnData(np.ones((20, 2)))
    a.obs["clone_id"] = None
    with pytest.raises(ValueError, match="Too few"):
        tf.tl.gene_heritability(a, layer=None, genes=["0"], verbose=False)

def test_sampling_intervals_cover_pure_state_boundaries(reference):
    x, ids = reference
    model = StateDensityModel.fit(x, ids, n_states=3)
    result = model.transform([[-2.]]*10, ["pure"]*10)
    count = result.counts.loc["pure"]
    assert (result.sampling_lower.loc["pure", count == 0] == 0).all()
    assert result.sampling_upper.loc["pure", count == 10].iloc[0] == 1
    assert (result.sampling_upper >= result.sampling_lower).to_numpy().all()


def test_frozen_expression_never_refits_on_query_and_aligns_genes(tmp_path):
    rng = np.random.default_rng(41)
    ref = ad.AnnData(rng.poisson(2, (60, 30)).astype(float),
                     var=pd.DataFrame(index=["IGHV1-1", "TRBV1"]+[f"g{i}" for i in range(28)]))
    model = tf.pp.FrozenExpressionModel.fit(ref, n_top_genes=20, n_comps=4, batch_size=7)
    assert not set(model.genes) & {"IGHV1-1", "TRBV1"}
    before = model.components.copy(), model.mean.copy()
    query = ref[:10].copy()
    projected = model.transform(query)
    order = rng.permutation(query.n_vars)
    np.testing.assert_allclose(projected, model.transform(query[:, order]), atol=1e-10)
    changed = query.copy()
    changed.X[:, 2:10] *= 10
    assert not np.allclose(projected, model.transform(changed))
    np.testing.assert_array_equal(model.components, before[0])
    np.testing.assert_array_equal(model.mean, before[1])
    path = tmp_path / "expression.npz"
    model.save(path)
    loaded = tf.pp.FrozenExpressionModel.load(path)
    np.testing.assert_allclose(loaded.transform(query), projected)
    with pytest.raises(ValueError, match="gene universe"):
        model.transform(query[:, 1:])


def test_region_gene_annotations_count_clones_equally():
    x = np.r_[np.full((101, 1), -2.), np.full((10, 1), 2.)]
    ids = ["large"]*100 + ["small"] + ["other"]*10
    model = StateDensityModel.fit(x, ids, n_states=2)
    genes = pd.DataFrame({"gene": [10.]*100 + [0.] + [-3.]*10})
    annotation = model.annotate_regions(x, genes, ids)
    region = model.state_names[np.argmin(model.centres[:, 0])]
    assert annotation["values"].loc[region, "gene"] == 5
    assert annotation["coverage"].loc[region, "n_cells"] == 101
    assert annotation["coverage"].loc[region, "n_clones"] == 2
