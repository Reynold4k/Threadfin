import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from threadfin import clones as tf_clones


def _shm_adata():
    """Three communities with increasing SHM (naive < gc < pb)."""
    rng = np.random.default_rng(0)
    means = {"naive": 2.0, "gc": 8.0, "pb": 15.0}
    rows = []
    i = 0
    for community, mean in means.items():
        for _ in range(30):
            rows.append(
                {
                    "barcode": f"cell{i}",
                    "clone_cluster": community,
                    "bcr_mu_count": float(rng.normal(mean, 0.5)),
                }
            )
            i += 1
    # cells with missing community or SHM are excluded
    rows.append({"barcode": f"cell{i}", "clone_cluster": None, "bcr_mu_count": 5.0})
    i += 1
    rows.append(
        {"barcode": f"cell{i}", "clone_cluster": "naive", "bcr_mu_count": np.nan}
    )
    obs = pd.DataFrame(rows).set_index("barcode")
    x = rng.poisson(1.0, size=(len(obs), 5)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    return AnnData(X=x, obs=obs, var=var)


def test_shm_gradient_test():
    adata = _shm_adata()
    res = tf_clones.shm_gradient_test(adata, order=["naive", "gc", "pb"])
    assert res["kruskal_H"] > 0
    assert res["kruskal_p"] < 1e-6
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["spearman_p"] < 0.05

    med = res["community_medians"]
    assert isinstance(med, pd.Series)
    assert len(med) == 3
    assert med["naive"] < med["gc"] < med["pb"]


def test_shm_gradient_test_no_order():
    res = tf_clones.shm_gradient_test(_shm_adata())
    assert res["spearman_rho"] is None
    assert res["spearman_p"] is None
    assert res["kruskal_p"] < 1e-6


def test_shm_gradient_test_v_identity_fallback():
    adata = _shm_adata()
    adata.obs["bcr_v_identity"] = 100.0 - adata.obs["bcr_mu_count"].astype(float)
    del adata.obs["bcr_mu_count"]
    res = tf_clones.shm_gradient_test(adata, order=["naive", "gc", "pb"])
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["kruskal_p"] < 1e-6


def test_shm_gradient_test_errors():
    adata = _shm_adata()
    with pytest.raises(ValueError, match="absent"):
        tf_clones.shm_gradient_test(adata, order=["naive", "gc", "plasma"])

    del adata.obs["clone_cluster"]
    with pytest.raises(KeyError):
        tf_clones.shm_gradient_test(adata)

    adata2 = _shm_adata()
    del adata2.obs["bcr_mu_count"]
    with pytest.raises(ValueError, match="bcr_v_identity"):
        tf_clones.shm_gradient_test(adata2)


def _public_adata():
    rows = []
    i = 0

    def add(clone, donor, cluster, n):
        nonlocal i
        for _ in range(n):
            rows.append(
                {
                    "barcode": f"cell{i}",
                    "clone_id": clone,
                    "donor": donor,
                    "clone_cluster": cluster,
                }
            )
            i += 1

    add("PUB1", "d1", "0", 3)
    add("PUB1", "d2", "0", 1)
    add("PUB2", "d1", "1", 2)
    add("PUB2", "d3", "1", 2)
    add("PUB3", "d1", "0", 3)
    add("PUB3", "d2", "1", 1)
    add("PRIV", "d1", "0", 4)  # single donor -> excluded
    rows.append(
        {"barcode": f"cell{i}", "clone_id": None, "donor": "d1", "clone_cluster": "0"}
    )
    obs = pd.DataFrame(rows).set_index("barcode")
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(len(obs), 5)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    return AnnData(X=x, obs=obs, var=var)


def test_public_clone_summary():
    adata = _public_adata()
    df = tf_clones.public_clone_summary(adata, donor_key="donor")
    assert list(df.columns) == [
        "clone_id",
        "n_donors",
        "n_cells",
        "donor_list",
        "dominant_cluster",
        "state_purity",
    ]
    assert set(df["clone_id"]) == {"PUB1", "PUB2", "PUB3"}  # PRIV excluded
    assert df["n_donors"].is_monotonic_decreasing
    # within n_donors ties, n_cells descends
    assert df["n_cells"].is_monotonic_decreasing

    p1 = df[df["clone_id"] == "PUB1"].iloc[0]
    assert p1["n_donors"] == 2
    assert p1["n_cells"] == 4
    assert list(p1["donor_list"]) == ["d1", "d2"]
    assert p1["dominant_cluster"] == "0"
    assert p1["state_purity"] == pytest.approx(1.0)

    p2 = df[df["clone_id"] == "PUB2"].iloc[0]
    assert p2["n_donors"] == 2
    assert p2["donor_list"] == ["d1", "d3"]
    assert p2["dominant_cluster"] == "1"

    p3 = df[df["clone_id"] == "PUB3"].iloc[0]
    assert p3["dominant_cluster"] == "0"
    assert p3["state_purity"] == pytest.approx(0.75)


def test_public_clone_summary_no_cluster_key():
    df = tf_clones.public_clone_summary(
        _public_adata(), donor_key="donor", cluster_key=None
    )
    assert list(df.columns) == ["clone_id", "n_donors", "n_cells", "donor_list"]


def test_public_clone_summary_missing_columns():
    adata = _public_adata()
    with pytest.raises(KeyError):
        tf_clones.public_clone_summary(adata, donor_key="nope")

    del adata.obs["clone_cluster"]
    with pytest.raises(KeyError):
        tf_clones.public_clone_summary(adata, donor_key="donor")

    del adata.obs["clone_id"]
    with pytest.raises(KeyError):
        tf_clones.public_clone_summary(adata, donor_key="donor", cluster_key=None)
