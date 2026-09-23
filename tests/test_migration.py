import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from threadfin import migration as tf_mig


def _mini_adata():
    rows = [
        # clone, tissue, state
        ("A", "g1", "s1"),  # clone A: 4 cells, all in g1 / s1
        ("A", "g1", "s1"),
        ("A", "g1", "s1"),
        ("A", "g1", "s1"),
        ("B", "g1", "s1"),  # clone B: half g1/s1, half g2/s2
        ("B", "g1", "s1"),
        ("B", "g2", "s2"),
        ("B", "g2", "s2"),
        ("C", "g2", "s2"),  # singleton clone, filtered by min_clone_size=2
        (None, "g1", "s1"),  # no clone assigned -> excluded
    ]
    obs = pd.DataFrame(
        rows,
        columns=["clone_id", "tissue", "state"],
        index=[f"cell{i}" for i in range(len(rows))],
    )
    x = np.ones((len(rows), 3), dtype=np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    return AnnData(X=x, obs=obs, var=var)


def test_clone_distribution():
    adata = _mini_adata()
    d = tf_mig.clone_distribution(adata, group_key="tissue")

    assert d.index.tolist() == ["A", "B"]  # C filtered, NaN clone excluded
    assert d.columns.tolist() == ["g1", "g2"]
    assert d.index.name == "clone_id"
    assert d.columns.name == "tissue"
    assert np.allclose(d.sum(axis=1), 1.0)
    assert d.loc["A", "g1"] == pytest.approx(1.0)
    assert d.loc["A", "g2"] == pytest.approx(0.0)
    assert d.loc["B", "g1"] == pytest.approx(0.5)
    assert d.loc["B", "g2"] == pytest.approx(0.5)


def test_clone_distribution_min_clone_size():
    adata = _mini_adata()
    d1 = tf_mig.clone_distribution(adata, group_key="tissue", min_clone_size=1)
    assert d1.index.tolist() == ["A", "B", "C"]
    assert d1.loc["C", "g2"] == pytest.approx(1.0)
    assert np.allclose(d1.sum(axis=1), 1.0)

    d5 = tf_mig.clone_distribution(adata, group_key="tissue", min_clone_size=5)
    assert d5.index.tolist() == []  # all clones below threshold


def test_migration_index():
    adata = _mini_adata()
    m = tf_mig.migration_index(adata, group_key="tissue")

    assert m.index.tolist() == ["g1", "g2"]
    assert m.columns.tolist() == ["g1", "g2"]
    # A contributes (1, 0), B contributes (0.5, 0.5)
    assert m.loc["g1", "g1"] == pytest.approx(1.0**2 + 0.5**2)
    assert m.loc["g2", "g2"] == pytest.approx(0.5**2)
    assert m.loc["g1", "g2"] == pytest.approx(1.0 * 0.0 + 0.5 * 0.5)
    assert np.allclose(m.values, m.values.T)  # symmetric
    # diagonal equals sum_c p[c, g]**2
    d = tf_mig.clone_distribution(adata, group_key="tissue")
    assert np.allclose(np.diag(m.values), (d.values**2).sum(axis=0))


def test_transition_index():
    adata = _mini_adata()
    t = tf_mig.transition_index(adata, state_key="state")

    assert t.index.tolist() == ["s1", "s2"]
    assert t.columns.tolist() == ["s1", "s2"]
    # A: s1=1, s2=0; B: s1=0.5, s2=0.5
    assert t.loc["s1", "s1"] == pytest.approx(1.0**2 + 0.5**2)
    assert t.loc["s2", "s2"] == pytest.approx(0.5**2)
    assert t.loc["s1", "s2"] == pytest.approx(1.0 * 0.0 + 0.5 * 0.5)
    assert np.allclose(t.values, t.values.T)


def _expansion_adata():
    rows = []
    rows += [("X", "single")] * 3  # one clone only -> defined as 0
    rows += [("Y", "uniform")] * 2  # two equal clones -> entropy = log 2 -> 0
    rows += [("Z", "uniform")] * 2
    rows += [("P", "skewed")] * 3  # 3:1 split -> hand-computed below
    rows += [("Q", "skewed")]
    rows += [(None, "single")]  # no clone -> excluded
    obs = pd.DataFrame(
        rows,
        columns=["clone_id", "tissue"],
        index=[f"cell{i}" for i in range(len(rows))],
    )
    x = np.ones((len(rows), 3), dtype=np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    return AnnData(X=x, obs=obs, var=var)


def test_expansion_index():
    adata = _expansion_adata()
    e = tf_mig.expansion_index(adata, group_key="tissue")

    assert e.name == "expansion_index"
    assert e.index.tolist() == ["single", "skewed", "uniform"]

    assert e.loc["single"] == pytest.approx(0.0)  # n_clones == 1
    assert e.loc["uniform"] == pytest.approx(0.0)  # equal sizes -> H = log n

    p = np.array([0.75, 0.25])
    h = float(-(p * np.log(p)).sum())
    assert e.loc["skewed"] == pytest.approx(1.0 - h / np.log(2.0))

    # stronger skew -> index moves towards 1
    rows = [("D", "g")] * 9 + [("E", "g")]
    obs = pd.DataFrame(rows, columns=["clone_id", "tissue"],
                       index=[f"c{i}" for i in range(len(rows))])
    ad2 = AnnData(X=np.ones((len(rows), 1), dtype=np.float32), obs=obs,
                  var=pd.DataFrame(index=["g0"]))
    e2 = tf_mig.expansion_index(ad2, group_key="tissue")
    assert e2.loc["g"] > e.loc["skewed"]


def test_missing_columns():
    adata = _mini_adata()
    with pytest.raises(KeyError, match="no_such_col"):
        tf_mig.clone_distribution(adata, group_key="no_such_col")
    with pytest.raises(KeyError, match="no_clone"):
        tf_mig.clone_distribution(adata, clone_key="no_clone", group_key="tissue")
    with pytest.raises(KeyError, match="no_such_col"):
        tf_mig.migration_index(adata, group_key="no_such_col")
    with pytest.raises(KeyError, match="no_such_col"):
        tf_mig.transition_index(adata, state_key="no_such_col")
    with pytest.raises(KeyError, match="no_such_col"):
        tf_mig.expansion_index(adata, group_key="no_such_col")
