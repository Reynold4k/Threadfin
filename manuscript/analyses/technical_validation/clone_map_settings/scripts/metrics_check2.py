from pathlib import Path
import numpy as np, pandas as pd, sys
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, ".")
from load_feats import features
from metrics_check import extent, largest_piece, faith
SWEEP = str(Path(__file__).resolve().parents[4] / "case_studies" / "results" / "clone_map_sweep")
ds = sys.argv[1]
f, s, r, ids = features(ds)
N = len(f)
kq = 15
nnf = NearestNeighbors(n_neighbors=kq + 1).fit(f).kneighbors(f, return_distance=False)[:, 1:]
rows = []
for k in [10, 15, 30, 50, 100, 200]:
    for md in [0.0, 0.1, 0.3, 0.5, 0.8]:
        xy = np.load(f"{SWEEP}/{ds}_min2_k{k}_md{md}.npy")
        nnm = NearestNeighbors(n_neighbors=kq + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        recall = np.mean([len(set(a) & set(b)) / kq for a, b in zip(nnf, nnm)])
        D = extent(xy)
        rows.append(dict(k=k, md=md, C=largest_piece(xy, 1.5 * D / np.sqrt(N)), F=faith(f, xy, np.random.default_rng(0)), knn_recall=recall))
d = pd.DataFrame(rows)
print(d.round(3).pivot(index="k", columns="md", values="knn_recall"))
print(d.round(3).pivot(index="k", columns="md", values="F"))
print(d.round(3).pivot(index="k", columns="md", values="C"))
