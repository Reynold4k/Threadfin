import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from threadfin import programs


def _markers_adata():
    """Two communities; g7 strongly up in B, g3 strongly up in A."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 40, 50
    x = rng.poisson(1.0, size=(2 * n_per, n_genes)).astype(np.float32)
    x[n_per:, 7] = rng.poisson(12.0, size=n_per)
    x[:n_per, 3] = rng.poisson(12.0, size=n_per)
    obs = pd.DataFrame(
        {"clone_cluster": ["A"] * n_per + ["B"] * n_per},
        index=[f"cell{i}" for i in range(2 * n_per)],
    )
    # cells without a community label are excluded
    x = np.vstack([x, rng.poisson(1.0, size=(2, n_genes)).astype(np.float32)])
    obs = pd.concat(
        [
            obs,
            pd.DataFrame(
                {"clone_cluster": [None, None]}, index=["cell_na0", "cell_na1"]
            ),
        ]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return AnnData(X=x, obs=obs, var=var)


def test_community_markers():
    adata = _markers_adata()
    df = programs.community_markers(adata, n_genes=10)
    assert list(df.columns) == ["community", "gene", "score", "logfoldchange", "pval_adj"]
    assert set(df["community"]) == {"A", "B"}
    assert (df.groupby("community", observed=True).size() == 10).all()

    b = df[df["community"] == "B"].set_index("gene")
    assert "g7" in b.index
    assert b.loc["g7", "logfoldchange"] > 0
    assert b.loc["g7", "pval_adj"] < 0.05

    a = df[df["community"] == "A"].set_index("gene")
    assert "g3" in a.index
    assert a.loc["g3", "logfoldchange"] > 0


def test_community_markers_missing_cluster_key():
    adata = _markers_adata()
    del adata.obs["clone_cluster"]
    with pytest.raises(KeyError):
        programs.community_markers(adata)


def _score_adata():
    """Two communities; programme genes g0-g2 high in community A.

    Background genes have heterogeneous expression levels so that
    ``score_genes`` finds control genes in every expression bin.
    """
    rng = np.random.default_rng(1)
    n_per, n_bg = 30, 200
    lam = rng.uniform(0.2, 20.0, size=n_bg)
    bg = rng.poisson(lam, size=(2 * n_per, n_bg)).astype(np.float32)
    prog = np.vstack(
        [rng.poisson(15.0, size=(n_per, 3)), rng.poisson(1.0, size=(n_per, 3))]
    ).astype(np.float32)
    x = np.hstack([prog, bg])
    obs = pd.DataFrame(
        {"clone_cluster": ["A"] * n_per + ["B"] * n_per},
        index=[f"cell{i}" for i in range(2 * n_per)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(3 + n_bg)])
    return AnnData(X=x, obs=obs, var=var)


def test_community_score():
    adata = _score_adata()
    sigs = {
        "prog": ["g0", "g1", "g2", "not_a_gene"],  # partial missing is fine
        "absent": ["nope1", "nope2"],  # fully missing -> skipped
    }
    with pytest.warns(UserWarning, match="absent"):
        out = programs.community_score(adata, sigs)

    assert list(out.columns) == ["community", "signature", "score", "n_cells"]
    assert set(out["signature"]) == {"prog"}
    assert set(out["community"]) == {"A", "B"}
    assert (out["n_cells"] == 30).all()

    s = out.set_index("community")["score"]
    assert s["A"] > s["B"]
    # no score columns leak into obs
    assert not any(c.startswith("_threadfin_score_") for c in adata.obs.columns)


def test_community_score_missing_cluster_key():
    adata = _score_adata()
    del adata.obs["clone_cluster"]
    with pytest.raises(KeyError):
        programs.community_score(adata, {"prog": ["g0"]})
