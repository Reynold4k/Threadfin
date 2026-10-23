import numpy as np, time, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
from load_feats import features
from metrics_check import extent, largest_piece
from scipy.stats import spearmanr
import umap

ds = sys.argv[1]
f, s, r, ids = features(ds)
N = len(f)
kmax = min(N - 1, 300)
t = time.time()
dist, idx = NearestNeighbors(n_neighbors=kmax).fit(f).kneighbors(f)
print("knn", time.time() - t)
rng = np.random.default_rng(0)
P = 30000
i, j = rng.integers(0, N, P), rng.integers(0, N, P); keep = i != j; i, j = i[keep], j[keep]
dhi = np.linalg.norm(f[i] - f[j], axis=1)
nn10 = idx[:, 1:11]

def score(xy):
    nm = NearestNeighbors(n_neighbors=11).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    knn = np.mean((nm[:, :, None] == nn10[:, None, :]).any(axis=2))
    cpd = spearmanr(dhi, np.linalg.norm(xy[i] - xy[j], axis=1)).statistic
    C = largest_piece(xy, 1.5 * extent(xy) / np.sqrt(N))
    return knn, cpd, C

for k, md in [(30, 0.3), (100, 0.5), (200, 0.3)]:
    for seed in range(4):
        t = time.time()
        xy = umap.UMAP(n_neighbors=k, min_dist=md, random_state=seed,
                       precomputed_knn=(idx[:, :k].copy(), dist[:, :k].copy())).fit_transform(f)
        t1 = time.time() - t
        knn, cpd, C = score(xy)
        print(f"k={k} md={md} seed={seed}  {t1:.1f}s  knn {knn:.3f} cpd {cpd:.3f} C {C:.3f}", flush=True)
