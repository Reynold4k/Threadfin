import sys

import numpy as np
import pandas as pd
import pytest

import threadfin as tf


def test_build_clone_key_vdj():
    tab = pd.DataFrame(
        {"v_call": ["IGHV1-2*01", "IGHV3-3*01"], "d_call": ["IGHD1-1*01", None],
         "j_call": ["IGHJ4*01", "IGHJ2*01"], "cdr3": ["ARAAA", "ARBBB"]},
        index=["c1", "c2"],
    )
    out = tf.build_clone_key(tab)
    assert out.loc["c1", "clone_id"] == "IGHV1-2*01_IGHD1-1*01_IGHJ4*01"
    out2 = tf.build_clone_key(tab, strategy="cdr3")
    assert out2.loc["c2", "clone_id"] == "ARBBB"
    with pytest.raises(ValueError):
        tf.build_clone_key(tab, strategy="nope")


def test_attach_bcr(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    assert "clone_id" in adata.obs.columns
    n_with = adata.obs["clone_id"].notna().sum()
    assert n_with == len(bcr)
    assert "threadfin_clones" in adata.uns
    assert adata.uns["threadfin_clones"].shape[0] == 130  # 120 + 10 singletons
    # no-BCR cells stay NaN
    assert adata.obs["clone_id"].isna().sum() == 50


def test_clone_centroids(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    cent = tf.clone_centroids(adata, basis="X_umap", min_clone_size=3)
    assert cent.shape[0] == 120  # singletons filtered
    clone0 = adata.obs_names[adata.obs["clone_id"] == "CLONE000"]
    manual = adata[clone0].obsm["X_umap"].mean(axis=0)
    assert np.allclose(cent.loc["CLONE000", ["X_umap_0", "X_umap_1"]].to_numpy(), manual)


def test_clonotype_recluster_recovers_states(synthetic):
    adata, bcr, truth = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, n_neighbors=15, resolution=0.5, random_state=0)

    assert "clone_cluster" in adata.obs.columns
    assert pd.api.types.is_categorical_dtype(adata.obs["clone_cluster"])
    cm = adata.uns["threadfin"]["clone_map"]
    assert {"x", "y", "n_cells", "clone_cluster"} <= set(cm.columns)

    # clone-level agreement with the ground-truth home state
    from sklearn.metrics import adjusted_rand_score

    clone_labels = cm["clone_cluster"].astype(str)
    clone_truth = pd.Series(truth).reindex(cm.index)
    ari = adjusted_rand_score(clone_truth, clone_labels)
    assert ari > 0.6, f"clone-level ARI too low: {ari}"
    assert 2 <= cm["clone_cluster"].nunique() <= 8

    # singletons are NaN back in obs
    singles = adata.obs[adata.obs["clone_id"].astype(str).str.startswith("SINGLE")]
    assert singles["clone_cluster"].isna().all()


@pytest.mark.xfail(sys.version_info >= (3, 12), strict=False,
                   reason="legacy v3 clonotype_recluster: nondeterministic on 3.12 dependency stack (issue #2)")
def test_determinism(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    a1 = tf.clonotype_recluster(adata.copy(), random_state=7)
    a2 = tf.clonotype_recluster(adata.copy(), random_state=7)
    assert (a1.obs["clone_cluster"].astype(str) == a2.obs["clone_cluster"].astype(str)).all()


def test_cdr3_blending(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, cdr3_weight=0.3, n_neighbors=15, random_state=0)
    assert "clone_cluster" in adata.obs.columns


def test_clonal_pseudotime(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, n_neighbors=15, random_state=0)
    tf.clonal_pseudotime(adata)
    pt = adata.obs["clonal_pseudotime"]
    assert pt.notna().sum() > 0
    vals = pt.dropna().to_numpy()
    assert np.isfinite(vals).all()
    assert vals.min() >= 0 and vals.max() <= 1


def test_metrics(synthetic):
    adata, bcr, truth = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, n_neighbors=15, random_state=0)

    res = tf.metrics.state_concordance(adata, "clone_cluster", "state")
    assert 0 <= res["nmi"] <= 1
    assert res["n_cells_used"] > 0

    purity = tf.metrics.clone_state_purity(adata, state_key="state")
    assert purity["purity"].between(0, 1).all()
    assert (purity["purity"] > 0.6).mean() > 0.8  # most clones are state-pure

    summ = tf.metrics.clone_cluster_summary(adata)
    assert summ["n_clones"].sum() == 120


def test_state_enrichment(synthetic):
    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, n_neighbors=15, random_state=0)

    enr = tf.metrics.state_enrichment(adata, "clone_cluster", "state")
    assert {"clone_cluster", "state", "n_cells", "odds_ratio", "pvalue", "fdr"} <= set(enr.columns)
    assert (enr["pvalue"] >= 0).all() and (enr["fdr"] >= 0).all()
    # synthetic clone clusters are built from state-pure clones -> strong signals
    assert (enr["fdr"] < 0.05).sum() >= 3


def test_plotting(synthetic, tmp_path):
    import matplotlib.figure

    adata, bcr, _ = synthetic
    tf.attach_bcr(adata, bcr)
    tf.clonotype_recluster(adata, n_neighbors=15, random_state=0)

    fig, ax = tf.plotting.clone_map(adata)
    assert isinstance(fig, matplotlib.figure.Figure)
    fig, ax = tf.plotting.cells(adata)
    assert isinstance(fig, matplotlib.figure.Figure)
    fig, ax, mat = tf.plotting.signature_heatmap(
        adata, {"sigA": ["gene1", "gene2"], "sigB": ["gene3"]}
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert mat.shape == (adata.obs["clone_cluster"].nunique(), 2)

    out = tmp_path / "clone_map.png"
    tf.plotting.clone_map(adata, save=str(out))
    assert out.exists() and out.stat().st_size > 1000


def test_deprecated_alias(synthetic):
    adata, bcr, _ = synthetic
    with pytest.warns(DeprecationWarning):
        adata = tf.bcr_reclustering(adata, bcr, n_neighbors=15, random_state=0)
    assert "clone_cluster" in adata.obs.columns


def test_sequence_utils():
    from threadfin.sequence import blend_distances, cdr3_distance_matrix

    d = cdr3_distance_matrix(["ARAAAA", "ARAAAA", "ARAAAT", None, "ARAA"])
    assert d[0, 1] == 0
    assert 0 < d[0, 2] < 0.2  # one mismatch in 6 aa
    assert d[0, 3] == 1.0     # missing sequence
    assert d[0, 4] > d[0, 2]  # length penalty

    b = blend_distances(np.random.default_rng(0).random((5, 5)), np.eye(5), 0.5)
    assert b.shape == (5, 5)
    assert (np.diag(b) == 0).all()
