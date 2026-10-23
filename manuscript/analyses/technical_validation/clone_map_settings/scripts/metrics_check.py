import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph
from scipy.stats import spearmanr

def extent(xy):
    lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
    return float(np.max(hi - lo))

def largest_piece(xy, r):
    g = radius_neighbors_graph(xy, radius=max(r, 1e-12), mode="connectivity")
    _, lab = connected_components(g, directed=False)
    return np.bincount(lab).max() / len(xy)

def faith(f, xy, rng, n=30000):
    i, j = rng.integers(0, len(xy), n), rng.integers(0, len(xy), n)
    k = i != j
    return spearmanr(np.linalg.norm(f[i[k]] - f[j[k]], axis=1), np.linalg.norm(xy[i[k]] - xy[j[k]], axis=1)).statistic
