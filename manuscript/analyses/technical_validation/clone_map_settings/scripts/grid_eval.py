"""Dense grid of clone-map layouts per dataset, scored on several candidate quality measures."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_feats import features
from metrics_check import extent, largest_piece
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

KS = [10, 15, 20, 30, 50, 75, 100, 150, 200]
MDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
SEEDS = [0, 1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid")

def recall(nf, nm):
    return float(np.mean((nm[:, :, None] == nf[:, None, :]).any(axis=2)))

def run(job):
    ds, mc = job
    import umap
    f, s, r, ids = features(ds, mc)
    N = len(f)
    kmax = min(N - 1, max(KS) + 1)
    dist, idx = NearestNeighbors(n_neighbors=kmax).fit(f).kneighbors(f)
    nf10, nf30 = idx[:, 1:11], idx[:, 1:31]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, N, 30000), rng.integers(0, N, 30000); keep = i != j; i, j = i[keep], j[keep]
    dhi = np.linalg.norm(f[i] - f[j], axis=1)
    rows, coords = [], {}
    for k in KS:
        if k >= N: continue
        for md in MDS:
            for seed in SEEDS:
                t = time.time()
                xy = umap.UMAP(n_neighbors=k, min_dist=md, random_state=seed,
                               precomputed_knn=(idx[:, :k].copy(), dist[:, :k].copy())).fit_transform(f)
                sec = time.time() - t
                nm = NearestNeighbors(n_neighbors=31).fit(xy).kneighbors(xy, return_distance=False)
                D = extent(xy)
                row = dict(dataset=f"{ds}_min{mc}", N=N, k=k, md=md, seed=seed, sec=sec,
                           local10=recall(nf10, nm[:, 1:11]), local30=recall(nf30, nm[:, 1:31]),
                           glob=spearmanr(dhi, np.linalg.norm(xy[i] - xy[j], axis=1)).statistic)
                for c in (1.0, 1.5, 2.0, 3.0):
                    row[f"C{c}"] = largest_piece(xy, c * D / np.sqrt(N))
                rows.append(row)
                coords[f"k{k}_md{md}_s{seed}"] = xy.astype(np.float32)
        print(ds, mc, "k", k, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{ds}_min{mc}.csv", index=False)
    np.savez_compressed(f"{OUT}/{ds}_min{mc}_coords.npz", **coords)
    return ds

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [("gut_gc_reporter", 2), ("spont_gc", 2), ("covid_bcell", 2), ("gc_np_pc", 2), ("mouse_np", 2),
            ("mouse_rbd", 2), ("malaria", 2), ("malaria_late", 2), ("ln_vaccine", 10), ("ln_vaccine", 2)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        for ds in p.imap_unordered(run, jobs):
            print("finished", ds, flush=True)
