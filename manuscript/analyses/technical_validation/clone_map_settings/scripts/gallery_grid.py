from pathlib import Path
import sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from load_feats import features
from analyze_grid import objectives, MANUAL
CASE = str(Path(__file__).resolve().parents[4] / "case_studies" / "results")
OBJ = sys.argv[1] if len(sys.argv) > 1 else "bal30_C1.5_adapt"
sets = sorted(glob.glob("grid/*.csv"))
fig, axs = plt.subplots(len(sets), 4, figsize=(16, 4 * len(sets)))
for row, f in enumerate(sets):
    d = pd.read_csv(f); ds = d.dataset.iloc[0]
    name, mc = ds.rsplit("_min", 1)
    feats, sizes, rel, ids = features(name, int(mc))
    ct = pd.read_csv(f"{CASE}/{name}/clone_table.csv", index_col=0)
    prog = ct["clone_programme"].reindex(ids).astype(str).replace("nan", "none").to_numpy()
    cats = sorted(set(prog) - {"none"})
    cmap = plt.get_cmap("tab10")
    col = np.array([cmap(cats.index(p) % 10) if p != "none" else (0.85, 0.85, 0.85, 1) for p in prog])
    m = d.groupby(["k", "md"]).mean(numeric_only=True)
    obj = objectives(m, ds)[OBJ]
    best = obj.idxmax(); alt = obj.drop(best).idxmax()
    coords = np.load(f.replace(".csv", "_coords.npz"))
    for c, (lab, key) in enumerate([("package default", (50, 0.8)), ("manual", MANUAL[ds]),
                                    (f"best {OBJ}", best), ("runner-up", alt)]):
        ax = axs[row, c]
        k, md = key
        kk = f"k{k}_md{md}_s0"
        if kk not in coords:
            ax.set_title(f"{lab}: {key} not in grid", fontsize=8); ax.axis("off"); continue
        xy = coords[kk]
        order = np.argsort(prog != "none")
        ax.scatter(xy[order, 0], xy[order, 1], s=np.clip(120000 / len(xy), 1, 40) * 0.3, c=col[order], linewidths=0)
        r = m.loc[key]
        ax.set_title(f"{ds} | {lab}: k={k}, md={md}\nC1.5 {r['C1.5']:.3f}  local30 {r.local30:.3f}  glob {r.glob:.3f}  obj {obj.loc[key]:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"gallery_{OBJ}.png", dpi=70)
print("saved")
