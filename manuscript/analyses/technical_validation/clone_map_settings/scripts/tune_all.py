import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_feats import features
MANUAL = {("gut_gc_reporter", 2): (30, 0.3), ("spont_gc", 2): (30, 0.3), ("covid_bcell", 2): (30, 0.3),
          ("ln_vaccine", 10): (50, 0.3), ("malaria_late", 2): (200, 0.1), ("mouse_np", 2): (15, 0.1),
          ("mouse_rbd", 2): (30, 0.8), ("gc_np_pc", 2): (50, 0.8), ("malaria", 2): (50, 0.8),
          ("ln_vaccine", 2): (50, 0.3)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned")

def job(args):
    ds, mc, rs, max_clones = args
    import umap
    from threadfin.maptune import tune_clone_map, map_quality
    f, s, r, ids = features(ds, mc)
    t = time.time()
    res = tune_clone_map(f, random_state=rs, max_clones=max_clones, verbose=False)
    sec = time.time() - t
    rows = []
    for lab, (k, md) in (("tuned", (res.n_neighbors, res.min_dist)), ("manual", MANUAL[(ds, mc)]),
                         ("default", (50, 0.8))):
        for seed in range(3):
            xy = umap.UMAP(n_neighbors=int(min(k, len(f) - 1)), min_dist=md, random_state=seed).fit_transform(f)
            q = map_quality(f, xy)
            rows.append(dict(dataset=ds, min_cells=mc, rs=rs, max_clones=max_clones, setting=lab, k=k, md=md,
                             seed=seed, **q))
            if seed == 0:
                np.save(f"{OUT}/{ds}_min{mc}_rs{rs}_mc{max_clones}_{lab}.npy", xy)
    summ = dict(dataset=ds, min_cells=mc, N=len(f), rs=rs, max_clones=max_clones, k=res.n_neighbors,
                md=res.min_dist, target=res.coherence_target, score=res.score, coherence=res.coherence,
                n_layouts=res.n_layouts, seconds=sec)
    res.history.to_csv(f"{OUT}/{ds}_min{mc}_rs{rs}_mc{max_clones}_history.csv", index=False)
    res.candidates.to_csv(f"{OUT}/{ds}_min{mc}_rs{rs}_mc{max_clones}_candidates.csv", index=False)
    print(json.dumps(summ), flush=True)
    return summ, rows

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ds, mc, rs, 5000) for (ds, mc) in MANUAL for rs in (0, 1)] + [("ln_vaccine", 2, 0, None)]
    jobs.sort(key=lambda j: -features(j[0], j[1])[0].shape[0])  # big ones first
    allsum, allrows = [], []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        for summ, rows in p.imap_unordered(job, jobs):
            allsum.append(summ); allrows += rows
            pd.DataFrame(allsum).to_csv(f"{OUT}/summary.csv", index=False)
            pd.DataFrame(allrows).to_csv(f"{OUT}/quality.csv", index=False)
    print("all done", flush=True)
