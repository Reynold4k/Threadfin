import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from load_feats import features
from metrics_check import extent
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
import umap
ds = sys.argv[1]
f, s, r, ids = features(ds)
N = len(f)
xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(f)
g = radius_neighbors_graph(xy, radius=1.5 * extent(xy) / np.sqrt(N))
nc, lab = connected_components(g, directed=False)
big = np.bincount(lab).argmax()
isl = lab != big
print("island clones:", isl.sum(), "components:", nc)
d1, i1 = NearestNeighbors(n_neighbors=11).fit(f).kneighbors(f)
tab = pd.DataFrame(dict(id=ids, size=s, rel=r, norm=np.linalg.norm(f, axis=1), nn1=d1[:, 1], nn10=d1[:, 10],
                        island=isl))
print(tab.groupby("island")[["size", "rel", "norm", "nn1", "nn10"]].describe().T.round(3).to_string())
print(tab[tab.island].head(30).to_string())
# are island members each other's neighbours?
frac_in = np.mean([isl[i1[i, 1:]].mean() for i in np.where(isl)[0]])
print("mean fraction of island members' 10-NN inside island:", round(frac_in, 3))
