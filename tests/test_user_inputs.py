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


def _tcr_csv_for(adata, path):
    """A Cell Ranger TCR filtered_contig_annotations.csv for the simulated cells: one beta and one alpha chain
    per cell, identical within a clone and distinct between clones."""
    rng = np.random.default_rng(0)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    junctions = {c: ("CASS" + "".join(rng.choice(aa, 7)) + "F", "CAV" + "".join(rng.choice(aa, 7)) + "F")
                 for c in adata.obs["clone_id"].astype(str).unique()}
    rows = []
    for cell, clone in adata.obs["clone_id"].astype(str).items():
        beta, alpha = junctions[clone]
        for chain, v, j, cdr3 in (("TRB", "TRBV7-9", "TRBJ2-7", beta), ("TRA", "TRAV1-2", "TRAJ33", alpha)):
            rows.append({"barcode": cell, "is_cell": True, "high_confidence": True, "chain": chain, "v_gene": v,
                         "d_gene": "None", "j_gene": j, "c_gene": chain[:2] + "C", "productive": True,
                         "cdr3": cdr3, "cdr3_nt": "TGT" * len(cdr3), "reads": 500, "umis": 5})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_t_cell_contigs_can_be_passed_as_a_file_path(tmp_path):
    """The README's T-cell route: a TCR filtered_contig_annotations.csv path used to crash after clone calling
    ('No productive BCR contigs left') because the per-cell summary read B-cell chains only."""
    pytest.importorskip("dandelion")
    adata = tf.sim.simulate_repertoire(random_state=0, n_clones=150)
    true_clone = adata.obs["clone_id"].astype(str).copy()
    _tcr_csv_for(adata, tmp_path / "tcr.csv")
    del adata.obs["clone_id"]
    res = tf.run(adata, bcr=str(tmp_path / "tcr.csv"), basis="X_pca", donor_key="donor", sample_key="context",
                 n_perm=10, n_boot=3, verbose=False)
    called = adata.obs["clone_id"]
    assert res.settings.get("receptor") == "TCR" and called.notna().all()
    # the called clones are the simulated ones (within donors)
    assert pd.crosstab(called, true_clone).gt(0).sum(axis=1).eq(1).all()


def test_define_clones_accepts_cell_ranger_column_names():
    """Tutorial 2 passes the Cell Ranger contig table (barcode, v_gene, j_gene, cdr3_nt) directly."""
    pytest.importorskip("dandelion")
    adata = tf.sim.simulate_repertoire(random_state=0, n_clones=40)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        _tcr_csv_for(adata, os.path.join(d, "tcr.csv"))
        contigs = pd.read_csv(os.path.join(d, "tcr.csv"))
    contigs["donor"] = contigs["barcode"].map(adata.obs["donor"].astype(str))
    ids = tf.define_clones_dandelion(contigs, donor_key="donor", verbose=False)
    assert ids.notna().all() and ids.index.isin(adata.obs_names).all()


def test_fraction_of_a_level_that_never_occurs_warns():
    """fraction:IGHA on a column holding IGHA1/IGHA2 used to return 0 for every clone without a word."""
    adata = tf.sim.simulate_repertoire(random_state=0, n_clones=60)
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    adata.obs["iso"] = np.where(np.arange(adata.n_obs) % 2, "IGHA1", "IGHG1")
    with pytest.warns(UserWarning, match="Levels starting with 'IGHA': IGHA1"):
        out = tf.tl.clone_labels(adata, "iso", how="fraction:IGHA")
    assert (out == 0).all()


def test_overview_shows_the_coherence_of_the_result(monkeypatch):
    """Panel b used to show whatever clonal_coherence ran last, not result.coherence."""
    import threadfin.plotting as plotting
    adata = tf.sim.simulate_repertoire(random_state=0, n_clones=1000)
    res = tf.run(adata, basis="X_pca", donor_key="donor", sample_key="context", n_perm=10, n_boot=3, verbose=False)
    tf.tl.clonal_coherence(adata, strata_key="donor", n_perm=10, verbose=False)     # a later, different test
    seen = {}
    original = plotting.coherence

    def spy(adata, *, result=None, ax=None, save=None):
        seen["icc"] = (result or adata.uns["threadfin"]["coherence"])["icc"]
        return original(adata, result=result, ax=ax, save=save)

    monkeypatch.setattr(plotting, "coherence", spy)
    res.plot()
    assert seen["icc"] == res.coherence["icc"]


def test_overview_is_drawn_when_no_programmes_were_found():
    """A small dataset gets 'not enough reliably profiled clones' for programmes; result.plot() used to crash."""
    adata = tf.sim.simulate_repertoire(random_state=0, n_clones=200)
    res = tf.run(adata, basis="X_pca", donor_key="donor", sample_key="context", n_perm=10, n_boot=3, verbose=False)
    assert res.programmes.empty
    assert res.plot() is not None
