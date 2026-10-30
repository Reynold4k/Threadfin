"""Functional tests for the v4 workflow: clones, profiles, coherence, programmes, tests, memory."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf
from threadfin.profiles import variance_components


@pytest.fixture(scope="module")
def sim():
    return tf.sim.simulate_repertoire(n_clones=1200, clone_size_exponent=1.9, random_state=3)


# --------------------------------------------------------------------------- variance components


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


def test_continuum_gives_a_single_programme():
    # clones differ (clonal ICC > 0) but come from one programme: no distinct programmes
    ad = tf.sim.simulate_repertoire(n_programmes=1, random_state=1)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor", representation="kernel",
                         verbose=False)
    summ = tf.tl.find_programmes(ad, n_boot=10, verbose=False)
    assert summ.shape[0] == 1
    assert not ad.uns["threadfin"]["programmes"]["split_tests"]["split"].any()
    # without the split test, Leiden cuts the continuum into pieces
    summ = tf.tl.find_programmes(ad, n_boot=10, test_splits=False, verbose=False)
    assert summ.shape[0] >= 2


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


def test_mutation_frequency_handles_shifted_germline():
    # the query still carries its 5' UTR while the germline starts at the V gene:
    # comparing from position 0 would make an unmutated cell look hypermutated
    v = "ACGTACGTACGTTTGACCAGTTACGGATCCAAG" * 3
    airr = pd.DataFrame({"sequence_alignment": ["GGGGTTTCCCAAA" + v], "germline_alignment": [v],
                         "v_sequence_start": [14], "v_sequence_end": [13 + len(v)]})
    assert tf.tl.mutation_frequency(airr).iloc[0] == pytest.approx(0.0)
    assert tf.tl.mutation_frequency(airr, align="none").iloc[0] > 0.3
    win = tf.tl.mutation_frequency(airr).attrs["windows"].iloc[0]
    assert (win["query_start"], win["germline_start"], win["length"]) == (13, 0, len(v))


def test_mutation_frequency_counts_v_region_only():
    v = "ACGTACGTACGTTTGACCAGTTACGGATCCAAG" * 3
    mutated = v[:50] + ("T" if v[50] != "T" else "A") + v[51:]
    airr = pd.DataFrame({"sequence_alignment": [mutated + "CCCCCCCCCC"],      # junction after the V
                         "germline_alignment": [v + "AAAAAAAAAA"],
                         "v_sequence_start": [1], "v_sequence_end": [len(v)]})
    assert tf.tl.mutation_frequency(airr).iloc[0] == pytest.approx(1 / len(v))


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
    assert result.settings["composition_weight"] == "cells"
    assert result.programmes.shape[0] < 2 or "State composition: share of the programme's cells" in text
    assert "clone_programme" in ad.obs
    fig = result.plot(save=str(tmp_path / "overview.png"))
    assert (tmp_path / "overview.png").exists()
    assert fig is not None

    # a Methods paragraph from the settings that were used, with numbered references
    methods = result.methods()
    assert f"Threadfin {tf.__version__}" in methods and "Clone identities were taken from" in methods
    assert "20 permutations of clone labels within 'context'" in methods and "6 bootstrap" in methods
    assert "ref. 1" in methods and "\nReferences\n1. " in methods and "ref. dandelion" not in methods
    # clones called by Dandelion: the settings recorded by define_clones_dandelion are described
    result.settings["clone_calling"] = {"dandelion_version": "0.5.7", "donor_key": "donor", "use_light_chain": True,
                                        "require_heavy_chain": True, "check_contigs": True,
                                        "identity": {"ig": 0.85, "tr-ab": 1.0, "tr-gd": 1.0},
                                        "python_hash_seed": "0"}
    methods = result.methods()
    assert "Dandelion 0.5.7 (ref. 1) within each donor ('donor')" in methods
    assert "heavy and light chains and a junction amino-acid identity of at least 0.85" in methods
    assert "after Dandelion's contig check" in methods and "Suo C et al." in methods
    result.settings["receptor"] = "TCR"
    assert "beta and alpha chains and identical junction amino-acid sequences" in result.methods()
    # clones called before run(): the Series returned by define_clones_dandelion carries the settings
    result.settings.update(clone_calling=None, receptor=None)
    assert "Clone identities were taken from" in result.methods()
    ids = pd.Series(["c1"], index=["cell0"])
    ids.attrs.update({"method": "dandelion", "dandelion_version": "0.5.7", "donor_key": "mouse", "identity": 0.85,
                      "receptor": "BCR", "use_light_chain": True, "require_heavy_chain": True})
    assert "Dandelion 0.5.7 (ref. 1) within each donor ('mouse')" in result.methods(ids)


def test_programme_composition_counts_cells_or_clones():
    # P1: clone a (8 cells, all GC) and clone b (2 cells, all ASC); P2: clone c (1 GC, 1 ASC) and d (4 ASC).
    # One cell without a state and one without a programme are not counted.
    obs = pd.DataFrame({
        "clone_id": ["a"] * 9 + ["b"] * 2 + ["c"] * 2 + ["d"] * 4 + ["e"],
        "clone_programme": ["P1"] * 11 + ["P2"] * 6 + [None],
        "state": ["GC"] * 8 + [None] + ["ASC"] * 2 + ["GC", "ASC"] + ["ASC"] * 4 + ["GC"],
    }, index=[f"cell{i}" for i in range(18)])
    ad = AnnData(np.zeros((18, 1)), obs=obs)
    cells = tf.tl.programme_composition(ad, "state", key="clone_programme")
    clones = tf.tl.programme_composition(ad, "state", key="clone_programme", weight="clones", clone_key="clone_id")
    assert cells.loc["P1", "GC"] == pytest.approx(0.8) and clones.loc["P1", "GC"] == pytest.approx(0.5)
    assert cells.loc["P2", "GC"] == pytest.approx(1 / 6) and clones.loc["P2", "GC"] == pytest.approx(0.25)
    assert list(cells.index) == list(clones.index) and list(cells.columns) == list(clones.columns)
    assert np.allclose(cells.sum(axis=1), 1) and np.allclose(clones.sum(axis=1), 1)
    # clones of equal size: both weightings give the same composition
    even = ad[ad.obs["clone_id"].isin(["b", "c"]) & ad.obs["state"].notna()].copy()
    even.obs["clone_programme"] = "P1"
    assert np.allclose(tf.tl.programme_composition(even, "state", key="clone_programme"),
                       tf.tl.programme_composition(even, "state", key="clone_programme", weight="clones",
                                                   clone_key="clone_id"))
    with pytest.raises(ValueError, match="weight must be 'cells' or 'clones'"):
        tf.tl.programme_composition(ad, "state", key="clone_programme", weight="donors")


def test_run_can_weight_the_composition_by_clones(sim, tmp_path):
    ad = sim.copy()
    with pytest.raises(ValueError, match="composition_weight must be"):
        tf.run(ad, donor_key="donor", sample_key="context", state_key="true_state", basis="X_pca",
               composition_weight="donor", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tf.run(ad, donor_key="donor", sample_key="context", state_key="true_state", basis="X_pca",
                        test=["true_programme"], composition_weight="clones", map_n_neighbors=15, map_min_dist=0.5,
                        n_perm=10, n_boot=6,
                        verbose=False)
    assert result.settings["composition_weight"] == "clones"
    pd.testing.assert_frame_equal(result.composition,
                                  tf.tl.programme_composition(ad, "true_state", weight="clones"))
    assert not np.allclose(result.composition, tf.tl.programme_composition(ad, "true_state"))
    assert result.programmes.shape[0] > 1
    assert "every clone counts once" in result.summary()
    assert "so that every clone counted once" in result.methods()
    result.plot(save=str(tmp_path / "overview_clones.png"))
    assert (tmp_path / "overview_clones.png").exists()

    # the run is recorded in the AnnData, so the result can be rebuilt after saving and reading it
    import anndata

    record = ad.uns["threadfin"]["run"]
    assert record["versions"]["threadfin"] == tf.__version__ and record["settings"]["composition_weight"] == "clones"
    ad.write_h5ad(tmp_path / "result.h5ad")
    again = tf.ThreadfinResult.from_adata(anndata.read_h5ad(tmp_path / "result.h5ad"))
    assert again.summary() == result.summary()
    assert again.methods() == result.methods()
    again.plot(save=str(tmp_path / "overview_again.png"))


def test_run_warns_once_about_a_sample_level_label(sim):
    """Both label tests check a label fixed within each sample; run() says so once, not once per test."""
    ad = sim.copy()
    ad.obs["gate"] = np.where(ad.obs["context"].astype(str).str.endswith("S0"), "A", "B")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tf.run(ad, donor_key="donor", sample_key="context", state_key="true_state", basis="X_pca",
                        test=["gate"], map_n_neighbors=15, map_min_dist=0.5, n_perm=10, n_boot=6, verbose=False)
    assert result.programmes.shape[0] > 1 and "gate" in result.tests       # both tests ran
    assert result.tests["gate"].attrs["sample_level"] and result.profile_tests["gate"]["sample_level"]
    assert sum("'gate' is the same for every cell" in str(w.message) for w in caught) == 1


def test_programme_markers_clone_level(sim):
    import scipy.sparse as sp

    ad = sim.copy()
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.tl.find_programmes(ad, n_boot=6, verbose=False)
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


def test_profile_association_runs(sim):
    ad = sim.copy()
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    res = tf.tl.profile_association(ad, "true_programme", n_perm=100, verbose=False)
    assert {"r2", "null_mean", "excess_r2", "p_value", "n_clones", "axis_correlation"} <= set(res)
    assert res["kind"] == "categorical" and 0 <= res["r2"] <= 1
    assert res["p_value"] < 0.05  # the true programme explains how clones differ
    num = tf.tl.profile_association(ad, "true_state", how="fraction:S0", n_perm=50, verbose=False)
    assert num["kind"] == "numeric"


def test_clonal_memory_runs():
    ad = tf.sim.simulate_repertoire(n_clones=1500, clone_size_exponent=1.7, n_timepoints=2, random_state=5)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    res = tf.tl.clonal_memory(ad, "timepoint", n_null=50, n_boot=50, verbose=False)
    assert {"memory_index", "memory_index_ci", "p_value", "n_clones", "pairs"} <= set(res)
    assert res["n_clones"] > 10 and res["memory_index_ci"][0] <= res["memory_index"] <= res["memory_index_ci"][1]


def test_clonal_memory_declines_on_too_few_clones():
    """Three clones cannot support an index: the function must refuse, not return a collapsed interval."""
    import pytest

    ad = tf.sim.simulate_repertoire(n_clones=1500, clone_size_exponent=1.7, n_timepoints=2, random_state=5)
    tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                         representation="kernel", verbose=False)
    with pytest.raises(ValueError, match="no memory index is reported"):
        tf.tl.clonal_memory(ad, "timepoint", min_pairs=10_000, n_null=10, n_boot=10, verbose=False)


def test_results_can_be_written_to_h5ad(tmp_path):
    """Every result Threadfin stores in adata.uns must survive adata.write_h5ad."""
    import anndata

    ad = tf.sim.simulate_repertoire(n_clones=1500, n_timepoints=2, random_state=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.run(ad, donor_key="donor", sample_key="context", basis="X_pca", n_perm=20, n_boot=5,
               test=["true_programme"], time_key="timepoint", map_n_neighbors=30, map_min_dist=0.5,
               verbose=False)
    ad.write_h5ad(tmp_path / "result.h5ad")
    back = anndata.read_h5ad(tmp_path / "result.h5ad")
    assert {"profiles", "coherence", "programmes", "memory", "associations"} <= set(back.uns["threadfin"])
    lo, hi = back.uns["threadfin"]["memory"]["timepoint"]["memory_index_ci"]
    assert lo <= hi


def test_programme_markers_within_one_state_separate_fate_from_composition():
    """Programme P2 holds more plasma cells (composition); its GC cells also express 'fate' more (within state).
    Markers across all cells find the plasma gene; markers within GC cells find 'fate' and not the plasma gene."""
    import scipy.sparse as sp
    from anndata import AnnData

    rng = np.random.default_rng(0)
    rows, expr = [], []
    for c in range(80):
        prog = "P1" if c < 40 else "P2"
        for k in range(6):
            state = "PC" if (prog == "P2" and k < 4) or (prog == "P1" and k == 0) else "GC"
            g = rng.normal(0, 0.3, size=20)
            g[0] += 3.0 if state == "PC" else 0.0                         # plasma-cell gene
            g[1] += 1.5 if (state == "GC" and prog == "P2") else 0.0       # within-GC difference
            rows.append({"clone_id": f"c{c}", "donor": "d1", "sample": "s1", "state": state, "prog": prog})
            expr.append(np.clip(g + 2, 0, None))
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    x = np.asarray(expr, dtype=np.float32)
    ad = AnnData(X=sp.csr_matrix(x), obs=obs, var=pd.DataFrame(index=["plasma"] + ["fate"] + [f"g{i}" for i in range(18)]))
    ad.layers["log_norm"] = ad.X.copy()
    ad.obsm["X_emb"] = rng.normal(size=(ad.n_obs, 5)).astype(np.float32)
    tf.tl.clone_profiles(ad, basis="X_emb", donor_key="donor", context_key="sample", verbose=False)
    ad.obs["clone_programme"] = ad.obs["prog"].astype(str)
    allc = tf.tl.programme_markers(ad, context_key=None, n_top=3, verbose=False)
    gc = tf.tl.programme_markers(ad, context_key=None, n_top=3, state_key="state", state="GC", verbose=False)
    assert "plasma" in set(allc.loc[allc["programme"] == "P2", "gene"])
    sig = gc[(gc["programme"] == "P2") & (gc["fdr"] < 0.05)]
    assert sig["gene"].iloc[0] == "fate" and "plasma" not in set(sig["gene"])
    assert (gc["state"] == "GC").all()


def test_landscape_genes_find_the_gene_that_follows_a_continuum_of_clones():
    """Clones lie along a continuum (no programmes); one gene follows it, the others are noise."""
    import scipy.sparse as sp
    from anndata import AnnData

    rng = np.random.default_rng(3)
    rows, emb, expr = [], [], []
    for c in range(150):
        t = rng.uniform()
        for _ in range(4):
            tc = t + rng.normal(0, 0.05)
            emb.append([3 * tc, 0.0] + list(rng.normal(0, 0.3, 4)))
            g = rng.normal(0, 0.4, 30)
            g[0] += 2.5 * tc                                     # follows the continuum
            expr.append(np.clip(g + 2, 0, None))
            rows.append({"clone_id": f"c{c}", "donor": "d1", "sample": "s1"})
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    ad = AnnData(X=sp.csr_matrix(np.asarray(expr, dtype=np.float32)), obs=obs,
                 var=pd.DataFrame(index=["trend"] + [f"noise{i}" for i in range(29)]))
    ad.layers["log_norm"] = ad.X.copy()
    ad.obsm["X_emb"] = np.asarray(emb, dtype=np.float32)
    tf.tl.clone_profiles(ad, basis="X_emb", donor_key="donor", context_key="sample", verbose=False)
    out = tf.tl.landscape_genes(ad, context_key=None, verbose=False)
    assert out["gene"].iloc[0] == "trend" and abs(out["corr_axis1"].iloc[0]) > 0.8
    assert (out["gene"] != "trend").sum() <= 3                  # noise genes rarely pass FDR 5 %


def test_landscape_genes_report_the_right_gene_when_rare_genes_are_left_out():
    """Genes expressed in too few cells are dropped before scoring; names and correlations must stay aligned."""
    import scipy.sparse as sp
    from anndata import AnnData

    rng = np.random.default_rng(4)
    rows, emb, expr = [], [], []
    for c in range(150):
        t = rng.uniform()
        for _ in range(4):
            tc = t + rng.normal(0, 0.05)
            emb.append([3 * tc, 0.0] + list(rng.normal(0, 0.3, 4)))
            g = np.clip(rng.normal(0, 0.4, 40) + 2, 0, None)
            g[:20] = 0                                           # 20 genes never expressed: filtered out
            g[30] = max(0.0, 2 + 2.5 * tc + rng.normal(0, 0.4))  # follows the continuum, after the gap
            expr.append(g)
            rows.append({"clone_id": f"c{c}", "donor": "d1", "sample": "s1"})
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    names = [f"silent{i}" for i in range(20)] + [f"noise{i}" for i in range(10)] + ["trend"] + \
        [f"noise{i}" for i in range(10, 19)]
    ad = AnnData(X=sp.csr_matrix(np.asarray(expr, dtype=np.float32)), obs=obs, var=pd.DataFrame(index=names))
    ad.layers["log_norm"] = ad.X.copy()
    ad.obsm["X_emb"] = np.asarray(emb, dtype=np.float32)
    tf.tl.clone_profiles(ad, basis="X_emb", donor_key="donor", context_key="sample", verbose=False)
    out = tf.tl.landscape_genes(ad, context_key=None, verbose=False)
    assert out["gene"].iloc[0] == "trend" and abs(out["corr_axis1"].iloc[0]) > 0.8
    assert not out["gene"].str.startswith("silent").any()
    mean_profiles = tf.tl.clone_profiles(ad, basis="X_emb", donor_key="donor", context_key="sample",
                                         representation="mean", verbose=False)
    assert len(mean_profiles)
    out = tf.tl.landscape_genes(ad, context_key=None, verbose=False)
    assert out["gene"].iloc[0] == "trend" and abs(out["corr_axis1"].iloc[0]) > 0.8
