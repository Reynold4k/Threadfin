"""Validation of the clone-map auto mode: stability across random states and quality on fresh seeds."""
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation")
EVAL_SEEDS = [101, 202, 303, 404, 505]

def tune(args):
    ds, mc, method, rs = args
    from threadfin.maptune import tune_clone_map
    f, s, r, ids = features(ds, mc)
    t = time.time()
    res = tune_clone_map(f, method=method, random_state=rs, verbose=False)
    res.history.to_csv(f"{OUT}/{ds}_min{mc}_{method}_rs{rs}_history.csv", index=False)
    res.candidates.to_csv(f"{OUT}/{ds}_min{mc}_{method}_rs{rs}_candidates.csv", index=False)
    out = dict(dataset=ds, min_cells=mc, N=len(f), method=method, rs=rs, k=res.n_neighbors, md=res.min_dist,
               target=res.coherence_target, tuned_score=res.score, tuned_coherence=res.coherence,
               n_layouts=res.n_layouts, seconds=time.time() - t)
    print(json.dumps(out), flush=True)
    return out

def evaluate(args):
    ds, mc, k, md = args
    import umap
    from threadfin.maptune import map_quality
    f, s, r, ids = features(ds, mc)
    rows = []
    for seed in EVAL_SEEDS:
        xy = umap.UMAP(n_neighbors=int(min(k, len(f) - 1)), min_dist=float(md), random_state=seed).fit_transform(f)
        q = map_quality(f, xy)
        rows.append(dict(dataset=ds, min_cells=mc, k=k, md=md, seed=seed, **q))
        if seed == EVAL_SEEDS[0]:
            np.save(f"{OUT}/{ds}_min{mc}_k{k}_md{md}.npy", xy)
    return rows

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    jobs = [(ds, mc, m, rs) for (ds, mc) in MANUAL for m in ("ladder", "adam") for rs in (0, 1, 2)]
    jobs.sort(key=lambda j: -features(j[0], j[1])[0].shape[0])
    with Pool(cpus) as p:
        tuned = list(p.imap_unordered(tune, jobs))
    tdf = pd.DataFrame(tuned); tdf.to_csv(f"{OUT}/tuned.csv", index=False)
    settings = {(t["dataset"], t["min_cells"], t["k"], t["md"]) for t in tuned}
    settings |= {(ds, mc, k, md) for (ds, mc), (k, md) in MANUAL.items()}
    settings |= {(ds, mc, 50, 0.8) for (ds, mc) in MANUAL}
    settings = sorted(settings, key=lambda s: -features(s[0], s[1])[0].shape[0])
    rows = []
    with Pool(cpus) as p:
        for rr in p.imap_unordered(evaluate, settings):
            rows += rr
            pd.DataFrame(rows).to_csv(f"{OUT}/evaluation.csv", index=False)
    print("all done", flush=True)
