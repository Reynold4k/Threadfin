"""The two steps that use approximate neighbour search, with whichever pynndescent and umap-learn are installed.

Above 50,000 cells the smoothing of clone profiles finds each cell's neighbours with pynndescent, and UMAP lays out
4,096 or more clones with pynndescent too; below that both searches are exact. pyproject.toml allows pynndescent
0.6 (with umap-learn 0.5) on Python 3.10-3.11 and pynndescent 0.7 (with umap-learn 0.6) from Python 3.12, so the
test matrix runs these paths with both. scanpy reads pynndescent 0.7 only from 1.12.5, which import threadfin checks.
"""

import warnings

import numpy as np
import pytest

from threadfin import _utils
from threadfin.maptune import layout_clone_map
from threadfin.profiles import neighbour_average


@pytest.mark.parametrize("scanpy, pynndescent, flagged", [
    ("1.11.5", "0.7.0", True), ("1.12.4", "0.7.0", True), ("1.12.5", "0.7.0", False), ("1.13.0", "0.7.0", False),
    ("1.11.5", "0.6.0", False), ("1.10.0", "0.5.8", False), ("1.12.4", "0.8.1", True), ("1.11.5", "0.7.0rc1", True),
    (None, "0.7.0", False), ("1.11.5", None, False), ("not a version", "0.7.0", False),
])
def test_scanpy_that_cannot_read_pynndescent_07_is_flagged(scanpy, pynndescent, flagged):
    problem = _utils.neighbour_stack_problem(scanpy, pynndescent)
    assert (problem is not None) == flagged
    if flagged:
        assert "pip install \"pynndescent<0.7\"" in problem and "getnnz" in problem


def test_the_import_check_warns_once_and_names_the_fix(monkeypatch):
    versions = {"scanpy": "1.11.5", "pynndescent": "0.7.0"}
    monkeypatch.setattr(_utils, "_installed_version", versions.get)
    with pytest.warns(UserWarning, match="scanpy 1.12.5 or later"):
        assert _utils.check_neighbour_stack() is not None
    versions["pynndescent"] = "0.6.0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _utils.check_neighbour_stack() is None


def test_smoothing_above_50000_cells_averages_the_approximate_neighbours(monkeypatch):
    import pynndescent
    from sklearn.neighbors import NearestNeighbors

    graphs = []
    original = pynndescent.NNDescent

    class Spy(original):
        @property
        def neighbor_graph(self):
            graph = super().neighbor_graph
            graphs.append(graph[0])
            return graph

    monkeypatch.setattr(pynndescent, "NNDescent", Spy)
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, size=(20, 5))
    x = centres[rng.integers(0, 20, 50_001)] + rng.normal(size=(50_001, 5))
    out = neighbour_average(x, 5)
    assert len(graphs) == 1, "more than 50,000 cells: the neighbours come from pynndescent"
    idx = graphs[0]
    np.testing.assert_allclose(out, x[idx].mean(axis=1), atol=1e-12)
    exact = NearestNeighbors(n_neighbors=5).fit(x).kneighbors(x, return_distance=False)
    recall = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(idx, exact)])
    assert recall > 0.9, recall


def test_clone_map_of_more_than_4096_clones():
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(1)
    centres = rng.normal(0, 8, size=(3, 8))
    group = rng.integers(0, 3, 4_200)
    features = centres[group] + rng.normal(size=(4_200, 8))
    xy, quality, setting, attempts = layout_clone_map(features, n_neighbors=15, min_dist=0.5)
    assert xy.shape == (4_200, 2) and np.isfinite(xy).all()
    assert setting == (15, 0.5, 1.0, 0) and len(attempts) == 1 and set(quality) >= {"coherence", "local_fidelity"}
    # three well-separated groups of clones stay apart on the map
    near = NearestNeighbors(n_neighbors=11).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    assert (group[near] == group[:, None]).mean() > 0.95


def test_cells_with_too_few_approximate_neighbours_get_exact_ones(monkeypatch):
    """pynndescent pads the neighbour row of a cell for which it found fewer than k neighbours with -1."""
    import pynndescent
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(2)
    x = rng.normal(size=(50_001, 3))
    exact = NearestNeighbors(n_neighbors=4).fit(x).kneighbors(x, return_distance=False)
    padded = exact.copy()
    padded[[0, 7, 50_000], 2:] = -1

    class Padded:
        def __init__(self, data, n_neighbors, random_state=None, **kw):
            assert data.shape == x.shape and n_neighbors == 4

        @property
        def neighbor_graph(self):
            return padded.copy(), np.zeros(padded.shape)

    monkeypatch.setattr(pynndescent, "NNDescent", Padded)
    np.testing.assert_allclose(neighbour_average(x, 4), x[exact].mean(axis=1), atol=1e-12)


def test_umap_keeps_the_layout_pipeline_the_clone_map_was_validated_with():
    """umap-learn 0.6 warns that a later release will switch to a new layout pipeline; Threadfin keeps the 0.5 one."""
    import inspect

    import umap

    opts = _utils.umap_layout_options()
    assert opts == ({"compatibility_layout": True} if "compatibility_layout" in inspect.signature(umap.UMAP).parameters
                    else {})
    rng = np.random.default_rng(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xy, *_ = layout_clone_map(rng.normal(size=(300, 6)), n_neighbors=10, min_dist=0.5)
    assert np.isfinite(xy).all()
    assert not [w for w in caught if "compatibility_layout" in str(w.message)]
