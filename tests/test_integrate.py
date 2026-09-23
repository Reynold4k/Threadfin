import importlib.util

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import threadfin as tf
from threadfin import integrate

HAS_BCRGRAPH = importlib.util.find_spec("threadfin.bcrgraph") is not None

_AA = "ACDEFGHIKLMNPQRSTVWY"
_N_CLONES = 60
_GROUPS = 3


def _mutate(seq, rng, n_mut):
    seq = list(seq)
    for pos in rng.choice(len(seq), size=n_mut, replace=False):
        seq[pos] = rng.choice(list(_AA))
    return "".join(seq)


@pytest.fixture()
def joint_synthetic():
    """60 clones x ~240 cells with planted joint structure: 3 GEX clusters,
    and a BCR table where the same 3 groups share V/J and near-identical
    CDR3s. Returns (adata, bcr_graph, clone_truth)."""
    rng = np.random.default_rng(0)
    d = 8
    centers = rng.normal(0, 8.0, size=(_GROUPS, d))
    consensus = ["".join(rng.choice(list(_AA), size=14)) for _ in range(_GROUPS)]

    clone_group = np.arange(_N_CLONES) % _GROUPS
    obs_rows, emb_rows, info_rows = [], [], []
    cell_i = 0
    for c in range(_N_CLONES):
        g = clone_group[c]
        cid = f"CLONE{c:03d}"
        cdr3 = _mutate(consensus[g], rng, n_mut=int(rng.integers(0, 2)))
        info_rows.append(
            {"clone_id": cid, "v_call": f"IGHV{g + 1}-1*01",
             "j_call": f"IGHJ{g + 1}*01", "cdr3": cdr3}
        )
        for _ in range(int(rng.integers(3, 6))):
            emb_rows.append(centers[g] + rng.normal(0, 1.0, size=d))
            obs_rows.append({"clone_id": cid})
            cell_i += 1

    n_cells = cell_i
    obs = pd.DataFrame(obs_rows, index=[f"cell{i:05d}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(50)])
    x = rng.poisson(1.0, size=(n_cells, 50)).astype(np.float32)
    adata = AnnData(X=x, obs=obs, var=var)
    emb = np.array(emb_rows)
    adata.obsm["X_pca"] = emb
    adata.obsm["X_umap"] = emb[:, :2].copy()

    clone_info = pd.DataFrame(info_rows).set_index("clone_id")
    adata.uns["threadfin_clones"] = clone_info

    # user-supplied BCR graph aligned with clone_info: strong within-group,
    # weak cross-group background (keeps the coupled graph connected)
    w = np.where(
        clone_group[:, None] == clone_group[None, :], 0.9, 0.02
    ).astype(float)
    np.fill_diagonal(w, 0.0)
    bcr_graph = sp.csr_matrix(w)

    truth = pd.Series(clone_group, index=clone_info.index, name="group")
    return adata, bcr_graph, truth


def _make_joint(adata, bcr_graph, **kw):
    params = dict(n_neighbors=10, lam=0.5, n_components=5, random_state=0)
    params.update(kw)
    return integrate.joint_embedding(adata, bcr_graph=bcr_graph, **params)


def test_joint_embedding_shape_and_storage(joint_synthetic):
    adata, bcr_graph, _ = joint_synthetic
    joint = _make_joint(adata, bcr_graph)
    assert joint.shape == (_N_CLONES, 6)  # 5 joint dims + n_cells
    assert list(joint.columns) == [f"joint_{i}" for i in range(5)] + ["n_cells"]
    assert (joint["n_cells"] >= 3).all()
    assert np.isfinite(joint[[f"joint_{i}" for i in range(5)]].to_numpy()).all()
    pd.testing.assert_frame_equal(joint, adata.uns["threadfin"]["joint"])
    params = adata.uns["threadfin"]["joint_params"]
    assert params["lam"] == 0.5 and params["n_clones"] == _N_CLONES
    assert adata.uns["threadfin"]["joint_graph"].shape == (_N_CLONES, _N_CLONES)


def test_joint_embedding_deterministic(joint_synthetic):
    adata, bcr_graph, _ = joint_synthetic
    j1 = _make_joint(adata.copy(), bcr_graph)
    j2 = _make_joint(adata.copy(), bcr_graph)
    pd.testing.assert_frame_equal(j1, j2)


def test_joint_embedding_recovers_groups(joint_synthetic):
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import adjusted_rand_score

    from threadfin.core import _leiden_on_distances

    adata, bcr_graph, truth = joint_synthetic
    joint = _make_joint(adata, bcr_graph)
    cols = [c for c in joint.columns if c != "n_cells"]
    dist = squareform(pdist(joint[cols].to_numpy()))
    labels = _leiden_on_distances(
        dist, n_neighbors=15, resolution=1.0, random_state=0
    )
    ari = adjusted_rand_score(truth.loc[joint.index], labels)
    assert ari > 0.8, f"joint embedding failed to recover planted groups: ARI={ari}"


def test_integration_diagnostics(joint_synthetic):
    adata, bcr_graph, _ = joint_synthetic
    joint = _make_joint(adata, bcr_graph)
    diag = integrate.integration_diagnostics(adata)
    assert {"testcor_gex", "testcor_bcr", "n_edges", "modality_contribution"} <= set(diag)
    assert diag["n_edges"] > 0
    for key in ("testcor_gex", "testcor_bcr"):
        assert -1.0 <= diag[key] <= 1.0
    contrib = diag["modality_contribution"]
    assert set(contrib) == {"gex", "bcr"}
    assert contrib["gex"] + contrib["bcr"] == pytest.approx(1.0)
    # explicit joint argument gives the same result
    diag2 = integrate.integration_diagnostics(adata, joint=joint)
    assert diag2["n_edges"] == diag["n_edges"]
    assert adata.uns["threadfin"]["joint_diagnostics"]["n_edges"] == diag["n_edges"]


def test_recluster_joint_and_distances_agree(joint_synthetic):
    from scipy.spatial.distance import pdist

    adata, bcr_graph, _ = joint_synthetic
    joint = _make_joint(adata, bcr_graph)
    cols = [c for c in joint.columns if c != "n_cells"]

    a1 = tf.clonotype_recluster(
        adata.copy(), basis="joint", n_neighbors=10, resolution=0.5,
        random_state=0,
    )
    a2 = tf.clonotype_recluster(
        adata.copy(), basis="joint", distances=pdist(joint[cols].to_numpy()),
        n_neighbors=10, resolution=0.5, random_state=0,
    )
    l1 = a1.obs["clone_cluster"].astype(str)
    l2 = a2.obs["clone_cluster"].astype(str)
    assert (l1 == l2).all()
    cm = a1.uns["threadfin"]["clone_map"]
    assert {"x", "y", "n_cells", "clone_cluster"} <= set(cm.columns)
    # distances= also works on a plain expression basis
    cent = tf.clone_centroids(adata, basis="X_pca", min_clone_size=3)
    coord_cols = [c for c in cent.columns if c != "n_cells"]
    d = pdist(cent[coord_cols].to_numpy())
    a3 = tf.clonotype_recluster(
        adata.copy(), basis="X_pca", distances=d, n_neighbors=10,
        resolution=0.5, random_state=0,
    )
    assert "clone_cluster" in a3.obs.columns


def test_recluster_distances_shape_checked(joint_synthetic):
    adata, _, _ = joint_synthetic
    with pytest.raises(ValueError, match="distances"):
        tf.clonotype_recluster(
            adata, basis="X_pca", distances=np.zeros((5, 5)), embed_clones=False
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        tf.clonotype_recluster(
            adata, basis="X_pca", distances=np.zeros(1770), cdr3_weight=0.2,
            embed_clones=False,
        )


def test_weighted_clone_centroids(joint_synthetic):
    adata, _, _ = joint_synthetic
    rng = np.random.default_rng(1)
    adata.obs["w"] = rng.uniform(0.1, 2.0, size=adata.n_obs)
    cent = tf.clone_centroids(
        adata, basis="X_pca", min_clone_size=3, weight_col="w"
    )
    cid = "CLONE000"
    mask = (adata.obs["clone_id"] == cid).to_numpy()
    w = adata.obs["w"].to_numpy()[mask]
    manual = (adata.obsm["X_pca"][mask] * w[:, None]).sum(axis=0) / w.sum()
    got = cent.loc[cid, [f"X_pca_{i}" for i in range(8)]].to_numpy()
    assert np.allclose(got, manual)


def test_pseudotime_use_clone_graph(joint_synthetic):
    adata, _, _ = joint_synthetic
    tf.clonotype_recluster(
        adata, basis="X_pca", n_neighbors=10, resolution=0.5, random_state=0
    )
    assert adata.uns["threadfin"]["clone_distances"].shape == (_N_CLONES, _N_CLONES)
    tf.clonal_pseudotime(adata, use_clone_graph=True)
    pt = adata.obs["clonal_pseudotime"]
    assert pt.notna().sum() > 0
    vals = pt.dropna().to_numpy()
    assert np.isfinite(vals).all()
    assert vals.min() >= 0 and vals.max() <= 1


def test_bcr_graph_none_branch(joint_synthetic):
    adata, _, _ = joint_synthetic
    if HAS_BCRGRAPH:
        # k > group size so the GEX kNN graph links the planted groups and
        # the coupled graph stays connected
        joint = _make_joint(adata, None, n_neighbors=25)
        assert joint.shape[0] == _N_CLONES
    else:
        with pytest.raises(ImportError, match="bcrgraph"):
            _make_joint(adata, None)
    # missing clone info always errors, regardless of bcrgraph availability
    adata2 = adata.copy()
    del adata2.uns["threadfin_clones"]
    with pytest.raises(KeyError, match="threadfin_clones"):
        _make_joint(adata2, None)
