"""Lineage heritability: does the estimator recover a known answer, and refuse a wrong one?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threadfin.phylo import (
    KERNELS,
    lineage_forest,
    lineage_heritability,
    lineage_power,
    simulate_lineage_trait,
)


def random_forest(n_clones=250, seed=0, max_cells=8):
    """A forest of small random trees with integer branch lengths, as a repertoire produces."""
    rng = np.random.default_rng(seed)
    clones, depth, pairs = [], [], []
    for c in range(n_clones):
        n = int(rng.integers(2, max_cells + 1))
        parent, bl = [-1], [0]
        for k in range(1, 2 * n):
            parent.append(int(rng.integers(0, k)))
            bl.append(int(rng.poisson(1.5)))
        node_depth = [0] * len(parent)
        for k in range(1, len(parent)):
            node_depth[k] = node_depth[parent[k]] + bl[k]
        nodes = rng.choice(len(parent), size=n, replace=True)

        def ancestors(k):
            out = [k]
            while parent[out[-1]] != -1:
                out.append(parent[out[-1]])
            return out

        base = len(clones)
        for k in range(n):
            clones.append(f"c{c}")
            depth.append(node_depth[nodes[k]])
        for a in range(n):
            for b in range(a + 1, n):
                anc = next(x for x in ancestors(nodes[a]) if x in set(ancestors(nodes[b])))
                d = node_depth[nodes[a]] + node_depth[nodes[b]] - 2 * node_depth[anc]
                pairs.append((base + a, base + b, d, node_depth[anc]))
    table = pd.DataFrame(pairs, columns=["i", "j", "d", "shared"])
    return lineage_forest(np.array(clones), table, depth=np.array(depth, dtype=float))


def test_forest_shapes_and_subset():
    f = random_forest(n_clones=40, seed=1)
    s = f.summary()
    assert s["units"] == f.n_units > 0
    assert s["cells"] == f.rows.size
    assert all(d.shape[0] == u.size for u, d in zip(f.units, f.dist))
    keep = np.zeros(f.n_cells, dtype=bool)
    keep[f.units[0]] = True
    sub = f.subset(keep)
    assert sub.n_units == 1 and sub.units[0].size == f.units[0].size


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_is_symmetric_and_positive_semidefinite(kernel):
    f = random_forest(n_clones=20, seed=2)
    for k in f.kernel(kernel, length_scale=2.0):
        assert np.allclose(k, k.T)
        assert np.linalg.eigvalsh(k).min() > -1e-8
        if kernel == "brownian":
            # a cell still at the germline root has accumulated no divergence, so its variance is 0
            assert np.all(np.diag(k) >= 0)
        else:
            assert np.all(np.diag(k) > 0)


def test_no_signal_is_not_called_heritable():
    f = random_forest(n_clones=300, seed=3)
    y = simulate_lineage_trait(f, 0.0, random_state=11)
    res = lineage_heritability(np.nan_to_num(y), f, n_perm=199, random_state=5, verbose=False)
    assert res.table["h2"].iloc[0] < 0.15
    assert res.table["p_perm"].iloc[0] > 0.05


def test_planted_signal_is_recovered():
    f = random_forest(n_clones=300, seed=4)
    y = simulate_lineage_trait(f, 0.3, random_state=12)
    res = lineage_heritability(np.nan_to_num(y), f, n_perm=199, random_state=6, verbose=False)
    assert 0.15 < res.table["h2"].iloc[0] < 0.5
    assert res.table["p_perm"].iloc[0] <= 0.05
    lo, hi = res.table["h2_lo"].iloc[0], res.table["h2_hi"].iloc[0]
    assert lo <= res.table["h2"].iloc[0] <= hi


def test_a_pure_depth_effect_is_not_mistaken_for_heritability():
    """The reason this module exists: distance on a tree grows with depth, so depth must be a fixed effect."""
    f = random_forest(n_clones=300, seed=5)
    rng = np.random.default_rng(13)
    y = 0.5 * np.nan_to_num(f.depth) + rng.normal(size=f.n_cells)
    with_depth = lineage_heritability(y, f, depth_effect=True, n_perm=199, random_state=7, verbose=False)
    without = lineage_heritability(y, f, depth_effect=False, n_perm=199, random_state=7, verbose=False)
    assert with_depth.table["h2"].iloc[0] < 0.1
    assert without.table["h2"].iloc[0] > with_depth.table["h2"].iloc[0]


def test_several_features_and_combined_p():
    f = random_forest(n_clones=200, seed=6)
    y = np.column_stack([simulate_lineage_trait(f, h, random_state=20 + k) for k, h in enumerate((0.0, 0.3))])
    res = lineage_heritability(np.nan_to_num(y), f, n_perm=99, feature_names=["null", "planted"],
                               random_state=8, verbose=False)
    assert list(res.table["feature"]) == ["null", "planted"]
    assert res.table.set_index("feature").loc["planted", "h2"] > res.table.set_index("feature").loc["null", "h2"]
    assert 0 < res.combined_p <= 1


def test_power_table_is_monotone_enough_and_controls_type_one_error():
    f = random_forest(n_clones=200, seed=7)
    tab = lineage_power(f, h2_grid=(0.0, 0.3), n_sim=12, n_perm=99, random_state=9, verbose=False)
    null_row = tab[tab["h2_true"] == 0.0].iloc[0]
    planted = tab[tab["h2_true"] == 0.3].iloc[0]
    assert null_row["rejection_perm"] <= 0.25  # 12 simulations: a loose but real bound
    assert planted["rejection_perm"] > null_row["rejection_perm"]
    assert planted["h2_mean"] > null_row["h2_mean"]


def test_input_validation():
    f = random_forest(n_clones=20, seed=8)
    with pytest.raises(ValueError):
        lineage_heritability(np.zeros(f.n_cells + 1), f, n_perm=9, verbose=False)
    y = np.full(f.n_cells, np.nan)
    with pytest.raises(ValueError):
        lineage_heritability(y, f, n_perm=9, verbose=False)
    with pytest.raises(ValueError):
        f.kernel("not-a-kernel")
    with pytest.raises(ValueError):
        simulate_lineage_trait(f, 1.5)
