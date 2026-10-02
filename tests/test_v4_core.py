"""Tests for the v4 core: profiles, coherence, programmes, tests, memory, heritability."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf
from threadfin.profiles import VarianceComponents, variance_components


@pytest.fixture(scope="module")
def sim():
    return tf.sim.simulate_repertoire(n_clones=1200, clone_size_exponent=1.9, random_state=3)


# --------------------------------------------------------------------------- variance components


def test_variance_components_recover_known_icc():
    rng = np.random.default_rng(0)
    n_clones, per_clone, tau, sigma = 400, 6, 1.0, 2.0
    clone = np.repeat(np.arange(n_clones), per_clone)
    x = rng.normal(0, tau, size=(n_clones, 3))[clone] + rng.normal(0, sigma, size=(clone.size, 3))
    vc = variance_components(x, clone)
    expected = tau**2 / (tau**2 + sigma**2)
    assert vc.icc == pytest.approx(expected, abs=0.04)
    assert np.allclose(vc.sigma2, sigma**2, rtol=0.1)


def test_spearman_brown_min_cells():
    vc = VarianceComponents(np.array([0.8]), np.array([0.2]), 3.0, 10, 50, np.zeros(1))
    # ICC 0.2 -> reliability 0.5 needs n = 4 cells
    assert vc.min_cells_for(0.5) == 4
    assert vc.reliability(np.array([4]))[0] == pytest.approx(0.5)


def test_singletons_do_not_enter_variance_components():
    x = np.arange(10, dtype=float)[:, None]
    codes = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])  # only clones 0 and 1 have 2 cells
    vc = variance_components(x, codes)
    assert vc.n_clones == 2 and vc.n_cells == 4


# --------------------------------------------------------------------------- profiles + coherence


def test_profiles_and_coherence(sim):
    ad = sim.copy()
    table = tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    assert {"n_cells", "reliability", "donor", "n_contexts"} <= set(table.columns)
    assert table["reliability"].between(0, 1).all()
    # bigger clones are more reliable
    assert np.corrcoef(np.log(table["n_cells"]), table["reliability"])[0, 1] > 0.5
    res = tf.tl.clonal_coherence(ad, n_perm=30, verbose=False)
    assert res["icc"] > res["null_q95"]
    assert res["p_value"] <= 1 / 31 + 1e-9


def test_coherence_null_is_calibrated_within_context():
    ad = tf.sim.simulate_repertoire(scenario="null", n_clones=800, random_state=11)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key=None, verbose=False)
    within = tf.tl.clonal_coherence(ad, strata_key="context", n_perm=40, verbose=False)
    naive = tf.tl.clonal_coherence(ad, strata_key=None, n_perm=40, verbose=False)
    assert within["p_value"] > 0.05  # no clonal signal beyond sampling context
    assert naive["p_value"] < 0.05   # a global shuffle mistakes context for clonality


def test_kernel_separates_equal_centroid_programmes():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    ad = tf.sim.simulate_repertoire(scenario="bifurcation", n_clones=1500, random_state=2)
    truth = ad.obs.groupby("clone_id")["true_programme"].first()
    sizes = ad.obs["clone_id"].value_counts()
    ab = sizes.index[(sizes >= 5).to_numpy() & truth.loc[sizes.index].isin(["G0", "G1"]).to_numpy()]
    scores = {}
    for rep in ("mean", "kernel"):
        tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", representation=rep,
                             min_cells=1, verbose=False)
        feats = ad.uns["threadfin"]["profiles"]["features"].loc[ab].to_numpy()
        lab = KMeans(2, n_init=10, random_state=0).fit_predict(feats)
        scores[rep] = adjusted_rand_score(truth.loc[ab], lab)
    assert scores["kernel"] > 0.8 > scores["mean"] + 0.3


# --------------------------------------------------------------------------- programmes + tests


def test_programmes_and_association(sim):
    ad = sim.copy()
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = tf.tl.find_programmes(ad, n_boot=6, verbose=False)
    assert summary.shape[0] >= 2
    assert summary["stability"].between(0, 1).all()
    assert ad.obs["clone_programme"].notna().any()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    assert table["confidence"].dropna().between(0, 1).all()

    # a label that tracks the true programme is detected; a random one is not
    res = tf.tl.association_test(ad, "true_programme", n_perm=200, verbose=False)
    assert (res["fdr"] < 0.05).any()
    ad.obs["coin"] = np.random.default_rng(0).choice(["heads", "tails"], size=ad.n_obs)
    res_coin = tf.tl.association_test(ad, "coin", n_perm=200, verbose=False)
    assert res_coin["fdr"].min() > 0.05
    # numeric label path: per-cell indicator of the most common true programme
    top = ad.obs.loc[ad.obs["clone_programme"].notna(), "true_programme"].value_counts().index[0]
    ad.obs["score"] = (ad.obs["true_programme"] == top).astype(float)
    res_num = tf.tl.association_test(ad, "score", n_perm=200, verbose=False)
    assert {"rank_effect", "median_in_programme"} <= set(res_num.columns)
    assert res_num.attrs["kruskal_pvalue"] < 0.05


def test_clone_labels_aggregation():
    obs = pd.DataFrame({
        "clone_id": ["a", "a", "a", "b", "b", None],
        "iso": ["IGHG", "IGHG", "IGHM", "IGHM", "IGHA", "IGHG"],
        "shm": [0.02, 0.04, 0.06, 0.0, 0.01, 0.5],
    }, index=[f"c{i}" for i in range(6)])
    ad = AnnData(X=np.zeros((6, 2)), obs=obs)
    maj = tf.tl.clone_labels(ad, "iso", clone_key="clone_id")
    assert maj["a"] == "IGHG" and pd.isna(maj["b"])  # b is a 50/50 tie below min_frac
    frac = tf.tl.clone_labels(ad, "iso", clone_key="clone_id", how="fraction:IGHM")
    assert frac["a"] == pytest.approx(1 / 3)
    mean = tf.tl.clone_labels(ad, "shm", clone_key="clone_id")
    assert mean["a"] == pytest.approx(0.04)


# --------------------------------------------------------------------------- memory


@pytest.mark.parametrize("memory, lo, hi", [(1.0, 0.8, 1.2), (0.0, -0.2, 0.3)])
def test_clonal_memory_tracks_truth(memory, lo, hi):
    ad = tf.sim.simulate_repertoire(n_clones=1500, clone_size_exponent=1.7, n_timepoints=2,
                                    memory=memory, random_state=5)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    res = tf.tl.clonal_memory(ad, "timepoint", programme_key="absent", n_null=100, n_boot=100,
                              verbose=False)
    assert lo < res["memory_index"] < hi


def test_community_transition_warns_on_clone_level_labels(sim):
    ad = sim.copy()
    ad.obs["tp"] = np.random.default_rng(0).choice(["d0", "d7"], size=ad.n_obs)
    ad.obs["clone_level"] = ad.obs["true_programme"]  # constant within every clone
    with pytest.warns(UserWarning, match="diagonal by construction"):
        tf.clones.community_transition(ad, time_key="tp", cluster_key="clone_level")


# --------------------------------------------------------------------------- heritability


def test_gene_heritability_ranks_clonal_gene_first():
    rng = np.random.default_rng(1)
    n_clones, per = 150, 6
    clone = np.repeat(np.arange(n_clones), per)
    clonal = rng.normal(0, 1, n_clones)[clone] + rng.normal(0, 0.5, clone.size)
    noisy = rng.normal(0, 1, clone.size)
    x = np.column_stack([clonal, noisy]) + 5
    obs = pd.DataFrame({"clone_id": [f"c{c}" for c in clone], "sample": "s1"},
                       index=[f"cell{i}" for i in range(clone.size)])
    ad = AnnData(X=x, obs=obs, var=pd.DataFrame(index=["clonal_gene", "noise_gene"]))
    out = tf.tl.gene_heritability(ad, clone_key="clone_id", context_key="sample", layer=None,
                                  n_perm=50, verbose=False)
    assert out.index[0] == "clonal_gene"
    assert out.loc["clonal_gene", "fdr"] < 0.05 < out.loc["noise_gene", "pvalue"] + 0.05


# --------------------------------------------------------------------------- clone definition


def _bcr_rows():
    rows = []
    # donor A: one lineage of 3 SHM variants (1 nt apart) + an unrelated clone
    base = "TGTGCGAGAGATCGGGGCTACTACTTTGACTACTGG"
    variants = [base, base[:10] + "T" + base[11:], base[:20] + "A" + base[21:]]
    for i, s in enumerate(variants):
        rows.append(("A", f"a{i}", "IGHV3-23*01", "IGHJ4*02", s))
    rows.append(("A", "a9", "IGHV1-2*02", "IGHJ6*01", "TGTGCGAGAGGGGGTTATGATTACGTCTGGGGC"))
    # donor B: an identical sequence to donor A's lineage must NOT join it
    rows.append(("B", "b0", "IGHV3-23*01", "IGHJ4*02", base))
    return pd.DataFrame(rows, columns=["donor", "barcode", "v_call", "j_call", "junction"]).set_index("barcode")


def test_define_clones_within_donor_single_linkage():
    out = tf.define_clones(_bcr_rows(), donor_key="donor", threshold=0.1, verbose=False)
    ids = out["clone_id"]
    assert ids["a0"] == ids["a1"] == ids["a2"]  # hypermutated variants merged
    assert ids["a9"] != ids["a0"]               # different V/J -> different clone
    assert ids["b0"] != ids["a0"]               # never across donors
    assert ids["b0"].startswith("B|")
    assert out.attrs["clone_definition"]["sequence"] == "nt"


def test_define_clones_threshold_zero_is_exact_matching():
    out = tf.define_clones(_bcr_rows(), donor_key="donor", threshold=0.0, verbose=False)
    assert out.loc[["a0", "a1", "a2"], "clone_id"].nunique() == 3


def test_find_threshold_bimodal():
    rng = np.random.default_rng(0)
    d = np.concatenate([rng.normal(0.04, 0.02, 400).clip(0.005), rng.normal(0.45, 0.08, 1200)])
    info = tf.tl.find_threshold(d)
    assert info["method"] == "density"
    assert 0.1 < info["threshold"] < 0.3


def test_mutation_frequency():
    airr = pd.DataFrame({
        "sequence_alignment": ["ACGTACGTAC", "ACGT-CGTAN"],
        "germline_alignment": ["ACGTACGTAA", "ACGTACGTAC"],
        "v_germline_end": [10, 10],
    })
    mf = tf.tl.mutation_frequency(airr)
    assert mf.iloc[0] == pytest.approx(0.1)
    assert mf.iloc[1] == pytest.approx(0.0)  # gap and N positions ignored


# --------------------------------------------------------------------------- preprocessing + run()


def test_ig_gene_mask():
    genes = ["IGHV3-23", "IGHM", "IGHG1", "Ighg2c", "IGKC", "IGLC2", "IGLL5", "IGKV1OR1-1",
             "IGHMBP2", "JCHAIN", "IGLON5", "CD19", "TRBV7-2"]
    mask = dict(zip(genes, tf.pp.ig_gene_mask(genes)))
    assert all(mask[g] for g in ["IGHV3-23", "IGHM", "IGHG1", "Ighg2c", "IGKC", "IGLC2", "IGLL5", "IGKV1OR1-1"])
    assert not any(mask[g] for g in ["IGHMBP2", "JCHAIN", "IGLON5", "CD19", "TRBV7-2"])
    assert dict(zip(genes, tf.pp.ig_gene_mask(genes, tr_genes=True)))["TRBV7-2"]


def test_run_end_to_end(sim, tmp_path):
    ad = sim.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tf.run(ad, donor_key="donor", sample_key="context", state_key="true_state",
                        basis="X_pca", test=["true_programme"], n_perm=20, n_boot=6, verbose=False)
    text = result.summary()
    assert "clonal coherence" in text and "programmes" in text
    assert "clone_programme" in ad.obs
    fig = result.plot(save=str(tmp_path / "overview.png"))
    assert (tmp_path / "overview.png").exists()
    assert fig is not None


def test_programme_markers_clone_level(sim):
    import scipy.sparse as sp

    ad = sim.copy()
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.tl.find_programmes(ad, n_boot=6, embed=False, verbose=False)
    # plant a marker in the true programme that dominates the purest recovered programme
    comp = pd.crosstab(ad.obs["clone_programme"], ad.obs["true_programme"], normalize="index")
    best_prog = comp.max(axis=1).idxmax()
    planted = comp.loc[best_prog].idxmax()
    rng = np.random.default_rng(0)
    expr = rng.poisson(1.0, size=(ad.n_obs, 30)).astype(float)
    expr[:, 0] += 3.0 * (ad.obs["true_programme"] == planted).to_numpy()
    ad2 = AnnData(X=sp.csr_matrix(np.log1p(expr)), obs=ad.obs.copy(), obsm=dict(ad.obsm), uns=dict(ad.uns),
                  var=pd.DataFrame(index=[f"gene{i}" for i in range(30)]))
    markers = tf.tl.programme_markers(ad2, layer=None, verbose=False)
    assert {"programme", "gene", "mean_diff", "auc", "fdr"} <= set(markers.columns)
    top = markers[markers["programme"] == str(best_prog)].sort_values("rank")
    assert top["gene"].iloc[0] == "gene0"
    assert top["fdr"].iloc[0] < 0.05
