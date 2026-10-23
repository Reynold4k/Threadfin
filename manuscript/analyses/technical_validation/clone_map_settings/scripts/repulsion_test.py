import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin")
from load_feats import features
from threadfin.maptune import _Reference, _target
import umap
ds, mc = sys.argv[1], int(sys.argv[2])
f, s, r, ids = features(ds, mc)
rng = np.random.default_rng(0)
if len(f) > 5000:
    f = f[np.sort(rng.choice(len(f), 5000, replace=False))]
ref = _Reference(f, k_max=200, random_state=0)
tgt = _target(ref, "auto")
rows = []
for k in (10, 32, 100):
    for md in (0.3, 0.5, 0.8):
        for rep in (1.0, 0.5, 0.25):
            for seed in (1, 2, 3):
                t = time.time()
                kk = min(k, ref.knn_idx.shape[1])
                xy = umap.UMAP(n_neighbors=kk, min_dist=md, repulsion_strength=rep, random_state=seed,
                               precomputed_knn=(ref.knn_idx[:, :kk].copy(), ref.knn_dist[:, :kk].copy())).fit_transform(ref.features)
                q = ref.quality(xy, tgt, 5.0)
                rows.append(dict(k=k, md=md, rep=rep, seed=seed, sec=time.time() - t, **q))
        print(k, md, "done", flush=True)
d = pd.DataFrame(rows)
g = d.groupby(["k", "md", "rep"]).agg(C_min=("coherence", "min"), C_mean=("coherence", "mean"),
                                       local=("local_fidelity", "mean"), glob=("global_fidelity", "mean"),
                                       sec=("sec", "mean")).round(3)
print(ds, mc, "target", round(tgt, 3)); print(g.to_string())
d.to_csv(f"repulsion_{ds}_min{mc}.csv", index=False)
