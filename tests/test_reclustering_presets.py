"""Contracts for exploratory reclustering controls and parameter provenance."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import threadfin as tf


@pytest.fixture
def clone_data():
    rng = np.random.default_rng(34)
    centres = np.r_[rng.normal(-3, .3, (5, 3)), rng.normal(3, .3, (5, 3))]
    obs = pd.DataFrame({"clone_id": np.repeat([f"c{i}" for i in range(10)], 4)},
                       index=[f"cell{i}" for i in range(40)])
    data = ad.AnnData(np.zeros((40, 2)), obs=obs)
    data.obsm["X_pca"] = np.repeat(centres, 4, axis=0) + rng.normal(0, .02, (40, 3))
    return data


def test_presets_do_not_mutate_defaults():
    settings = tf.reclustering_presets()
    settings["continuous"]["min_dist"] = 99
    assert tf.reclustering_presets()["continuous"]["min_dist"] == .4
    assert set(settings) == {"cohesive", "continuous", "discrete"}


def test_default_compatibility_and_overrides(clone_data):
    default = tf.clonotype_recluster(clone_data, basis="X_pca", embed_clones=False, copy=True)
    explicit = tf.clonotype_recluster(clone_data, basis="X_pca", n_neighbors=20,
                                     resolution=.3, embed_clones=False, copy=True)
    pd.testing.assert_frame_equal(default.uns["threadfin"]["clone_map"],
                                  explicit.uns["threadfin"]["clone_map"])
    result = tf.clonotype_recluster(clone_data, basis="X_pca", preset="continuous",
                                   n_neighbors=7, resolution=.6, min_dist=.25,
                                   embed_clones=False, copy=True)
    config = result.uns["threadfin"]["clonotype_recluster"]
    assert (config["n_neighbors"], config["resolution"], config["min_dist"]) == (7, .6, .25)
    assert default.uns["threadfin"]["clonotype_recluster"]["effective_n_neighbors"] == 9
    assert "threadfin" not in clone_data.uns
    assert "clone_cluster" not in clone_data.obs


def test_display_controls_leave_distance_partition_unchanged(clone_data):
    first = tf.clonotype_recluster(clone_data, basis="X_pca", n_neighbors=3,
                                  min_dist=.05, random_state=3, copy=True)
    second = tf.clonotype_recluster(clone_data, basis="X_pca", n_neighbors=3,
                                   min_dist=.4, umap_n_neighbors=7,
                                   embedding_mode="distance_profiles", random_state=3, copy=True)
    left, right = first.uns["threadfin"], second.uns["threadfin"]
    pd.testing.assert_series_equal(left["clone_map"]["clone_cluster"], right["clone_map"]["clone_cluster"])
    np.testing.assert_array_equal(left["clone_distances"], right["clone_distances"])
    assert not np.allclose(left["clone_map"][["x", "y"]], right["clone_map"][["x", "y"]])
    assert np.isfinite(right["clone_map"][["x", "y"]]).all().all()
    assert right["clonotype_recluster"]["effective_umap_n_neighbors"] == 7
    # Parameters and coordinates survive normal AnnData serialization.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "result.h5ad"
        second.write_h5ad(path)
        saved = ad.read_h5ad(path)
        assert saved.uns["threadfin"]["clonotype_recluster"]["embedding_mode"] == "distance_profiles"


@pytest.mark.parametrize("kwargs, message", [
    ({"preset": "auto"}, "preset must"),
    ({"n_neighbors": 1}, "at least 2"),
    ({"umap_n_neighbors": 2.5}, "positive integer"),
    ({"min_dist": 1.2}, "between zero and spread"),
    ({"resolution": 0}, "greater than zero"),
    ({"learning_rate": float("nan")}, "finite number"),
    ({"embedding_mode": "unknown"}, "embedding_mode must"),
    ({"cluster_on": "unknown"}, "cluster_on must"),
    ({"cluster_on": "embedding", "embed_clones": False}, "requires embed_clones=True"),
])
def test_invalid_controls_are_rejected_without_mutation(clone_data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tf.clonotype_recluster(clone_data, basis="X_pca", **kwargs)
    assert "threadfin" not in clone_data.uns
    assert "clone_cluster" not in clone_data.obs


def test_missing_clone_strings_are_not_reclustered(clone_data):
    clone_data.obs.loc[clone_data.obs.index[:4], "clone_id"] = ["nan", "", "None", None]
    result = tf.clonotype_recluster(clone_data, basis="X_pca", embed_clones=False)
    assert result.obs.iloc[:4]["clone_cluster"].isna().all()
    assert len(result.uns["threadfin"]["clone_map"]) == 9


def test_invalid_distances_fail_before_outputs(clone_data):
    distances = np.ones((10, 10))
    np.fill_diagonal(distances, 0)
    distances[0, 1] = -1
    with pytest.raises(ValueError, match="finite, non-negative, symmetric"):
        tf.clonotype_recluster(clone_data, basis="X_pca", distances=distances)
    assert "clone_cluster" not in clone_data.obs


def test_embedding_partition_matches_notebook_scanpy_graph(clone_data, tmp_path):
    import scanpy as sc

    result = tf.clonotype_recluster(
        clone_data, basis="X_pca", preset="continuous", n_neighbors=5,
        umap_n_neighbors=7, resolution=.3, embedding_mode="distance_profiles",
        cluster_on="embedding", random_state=123, copy=True,
    )
    cm = result.uns["threadfin"]["clone_map"]
    expected = ad.AnnData(cm[["x", "y"]].to_numpy())
    sc.pp.neighbors(expected, n_neighbors=5, use_rep="X", random_state=123)
    sc.tl.leiden(expected, resolution=.3, random_state=123,
                 flavor="leidenalg", directed=True, n_iterations=-1)
    np.testing.assert_array_equal(cm.clone_cluster.astype(str), expected.obs.leiden.astype(str))
    np.testing.assert_array_equal(result.obs.clone_cluster.astype(str),
                                  result.obs.clone_id.map(cm.clone_cluster).astype(str))
    config = result.uns["threadfin"]["clonotype_recluster"]
    assert config["cluster_on"] == "embedding"
    assert config["graph_method"] == "scanpy_umap_connectivities"
    assert config["effective_n_neighbors"] == 5
    assert config["effective_umap_n_neighbors"] == 7
    path = tmp_path / "notebook_graph.h5ad"
    result.write_h5ad(path)
    saved = ad.read_h5ad(path)
    pd.testing.assert_frame_equal(cm, saved.uns["threadfin"]["clone_map"])
    assert saved.uns["threadfin"]["clonotype_recluster"]["cluster_on"] == "embedding"
