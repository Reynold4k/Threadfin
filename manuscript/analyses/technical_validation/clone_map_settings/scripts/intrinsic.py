import numpy as np, warnings
warnings.filterwarnings("ignore")
from load_feats import features
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
for ds, mc in [("gut_gc_reporter",2),("spont_gc",2),("covid_bcell",2),("gc_np_pc",2),("mouse_np",2),("mouse_rbd",2),("malaria",2),("malaria_late",2),("ln_vaccine",10),("ln_vaccine",2)]:
    f, s, r, ids = features(ds, mc)
    N = len(f)
    idx = NearestNeighbors(n_neighbors=min(N - 1, 31)).fit(f).kneighbors(f, return_distance=False)
    out = []
    for k in (10, 15, 20, 30):
        rows = np.repeat(np.arange(N), k); cols = idx[:, 1:k + 1].ravel()
        g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        gm = g.multiply(g.T)
        nc, lab = connected_components(gm, directed=False)
        sz = np.sort(np.bincount(lab))[::-1]
        out.append(f"k{k}: {sz[0]/N:.3f} (n>=5 pieces {int((sz >= 5).sum())}, next {sz[1:3].tolist()})")
    print(f"{ds}_min{mc} N={N}: " + " | ".join(out))
