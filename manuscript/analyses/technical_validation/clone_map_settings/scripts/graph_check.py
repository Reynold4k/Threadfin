import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from load_feats import features
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ds = sys.argv[1]; mc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
f, s, r, ids = features(ds, mc)
N = len(f)
dist, idx = NearestNeighbors(n_neighbors=min(N - 1, 301)).fit(f).kneighbors(f)
print(ds, N)
for k in [3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]:
    if k >= N: break
    rows = np.repeat(np.arange(N), k); cols = idx[:, 1:k + 1].ravel()
    g = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(N, N))
    nc, lab = connected_components(g, directed=False)
    sizes = np.sort(np.bincount(lab))[::-1]
    # mutual kNN graph too
    gm = g.multiply(g.T)
    ncm, labm = connected_components(gm, directed=False)
    sm = np.sort(np.bincount(labm))[::-1]
    print(f"k={k:3d}: union components {nc} (largest {sizes[0]/N:.3f}; next {sizes[1:4].tolist()})  "
          f"mutual components {ncm} (largest {sm[0]/N:.3f})")
