"""Validation of the clone-map auto mode (v2): delivered map, robustness on fresh seeds, timing."""
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation3")
EVAL_SEEDS = [101, 202, 303, 404, 505]

def fresh(f, k, md, rep):
    import umap
    from threadfin.maptune import map_quality
    rows = []
    for seed in EVAL_SEEDS:
        xy = umap.UMAP(n_neighbors=int(min(k, len(f) - 1)), min_dist=float(md), repulsion_strength=float(rep),
                       random_state=seed).fit_transform(f)
        rows.append(map_quality(f, xy))
    q = pd.DataFrame(rows)
    return dict(fresh_score=q.score.mean(), fresh_Cmin=q.coherence.min(), fresh_Cmean=q.coherence.mean(),
                fresh_local=q.local_fidelity.mean(), fresh_global=q.global_fidelity.mean())

def job(args):
    ds, mc, method, rs = args
    from threadfin.maptune import tune_clone_map, layout_clone_map
    f, s, r, ids = features(ds, mc)
    out = dict(dataset=ds, min_cells=mc, N=len(f), method=method, rs=rs)
    t = time.time()
    if method in ("ladder", "adam"):
        tun = tune_clone_map(f, method=method, random_state=rs, verbose=False)
        t_tune = time.time() - t
        xy, q, setting, attempts = layout_clone_map(f, n_neighbors=tun.n_neighbors, min_dist=tun.min_dist,
                                                     repulsion_strength=tun.repulsion_strength, tuning=tun,
                                                     random_state=rs)
        tun.history.to_csv(f"{OUT}/{ds}_min{mc}_{method}_rs{rs}_history.csv", index=False)
        tun.candidates.to_csv(f"{OUT}/{ds}_min{mc}_{method}_rs{rs}_candidates.csv", index=False)
        out.update(target=tun.coherence_target, tuned_coherence=tun.coherence, n_layouts=tun.n_layouts,
                   tune_seconds=t_tune)
    else:
        k, md = MANUAL[(ds, mc)] if method == "manual" else (50, 0.8)
        xy, q, setting, attempts = layout_clone_map(f, n_neighbors=k, min_dist=md, random_state=rs)
        out.update(target=q["coherence_target"])
    out.update(k=setting[0], md=setting[1], rep=setting[2], layout_seed=setting[3], n_attempts=len(attempts),
               delivered_C=q["coherence"], delivered_score=q["score"], total_seconds=time.time() - t)
    np.save(f"{OUT}/{ds}_min{mc}_{method}_rs{rs}.npy", xy)
    out.update(fresh(f, setting[0], setting[1], setting[2]))
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ds, mc, m, rs) for (ds, mc) in MANUAL for m in ("ladder",) for rs in (0, 1, 2)]
    pass
    jobs.sort(key=lambda j: -features(j[0], j[1])[0].shape[0])
    rows = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        for out in p.imap_unordered(job, jobs):
            rows.append(out)
            pd.DataFrame(rows).to_csv(f"{OUT}/results.csv", index=False)
    print("all done", flush=True)
