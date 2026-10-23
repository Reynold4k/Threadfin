import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from load_feats import features
CASE = "/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin/case_studies/results"
r3 = pd.read_csv("validation3/results.csv"); r2 = pd.read_csv("validation2/results.csv")
sets = [("gut_gc_reporter", 2), ("ln_vaccine", 10), ("malaria_late", 2), ("spont_gc", 2), ("covid_bcell", 2),
        ("mouse_rbd", 2), ("malaria", 2), ("mouse_np", 2), ("gc_np_pc", 2)]
fig, axs = plt.subplots(len(sets), 5, figsize=(17, 3.3 * len(sets)))
for i, (ds, mc) in enumerate(sets):
    f, s, r, ids = features(ds, mc)
    ct = pd.read_csv(f"{CASE}/{ds}/clone_table.csv", index_col=0)
    prog = ct["clone_programme"].reindex(ids).astype(str).replace("nan", "none").to_numpy()
    cats = sorted(set(prog) - {"none"}); cmap = plt.get_cmap("tab10")
    col = np.array([cmap(cats.index(p) % 10) if p != "none" else (0.85, 0.85, 0.85, 1) for p in prog])
    panels = [("manual", f"validation2/{ds}_min{mc}_manual_rs0.npy", r2[(r2.dataset == ds) & (r2.min_cells == mc) & (r2.method == "manual")]),
              ("default 50/0.8", f"validation2/{ds}_min{mc}_default_rs0.npy", r2[(r2.dataset == ds) & (r2.min_cells == mc) & (r2.method == "default")])]
    for rs in (0, 1, 2):
        panels.append((f"auto rs{rs}", f"validation3/{ds}_min{mc}_ladder_rs{rs}.npy", r3[(r3.dataset == ds) & (r3.min_cells == mc) & (r3.rs == rs)]))
    for j, (lab, path, row) in enumerate(panels):
        ax = axs[i, j]
        try:
            xy = np.load(path)
        except FileNotFoundError:
            ax.axis("off"); continue
        o = np.argsort(prog != "none")
        ax.scatter(xy[o, 0], xy[o, 1], s=max(1, 60000 / len(xy)) * 0.15, c=col[o], linewidths=0)
        rw = row.iloc[0]
        ax.set_title(f"{ds}_min{mc} {lab}: k={int(rw.k)} md={rw.md:.2f} rep={rw.rep:g}\nC={rw.delivered_C:.3f} (target {rw.target:.3f}) fresh Cmin {rw.fresh_Cmin:.3f}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig("gallery_v3.png", dpi=60)
