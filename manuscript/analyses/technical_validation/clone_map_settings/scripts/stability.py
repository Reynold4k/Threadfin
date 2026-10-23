import glob, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
INTR = {"gut_gc_reporter_min2": .966, "spont_gc_min2": .993, "covid_bcell_min2": .978, "gc_np_pc_min2": .970,
        "mouse_np_min2": .928, "mouse_rbd_min2": .970, "malaria_min2": .975, "malaria_late_min2": .982,
        "ln_vaccine_min10": .982, "ln_vaccine_min2": .985}
MANUAL = {"gut_gc_reporter_min2": (30, 0.3), "spont_gc_min2": (30, 0.3), "covid_bcell_min2": (30, 0.3),
          "ln_vaccine_min10": (50, 0.3), "malaria_late_min2": (200, 0.1), "mouse_np_min2": (15, 0.1),
          "mouse_rbd_min2": (30, 0.8), "gc_np_pc_min2": (50, 0.8), "malaria_min2": (50, 0.8), "ln_vaccine_min2": (50, 0.3)}
def score(d, tgt):
    return 0.5 * (d.local30 + d.glob) - 5 * np.maximum(0, tgt - d["C1.5"])
rows = []
for f in sorted(glob.glob("grid/*.csv")):
    d = pd.read_csv(f); ds = d.dataset.iloc[0]; tgt = min(0.99, INTR[ds])
    d["S"] = score(d, tgt)
    ks, mds = sorted(d.k.unique()), sorted(d.md.unique())
    truth = d.groupby(["k", "md"]).S.mean().unstack()           # mean over both seeds
    cohm = d.groupby(["k", "md"])["C1.5"].mean().unstack()
    best_true = truth.values.max()
    out = dict(dataset=ds, N=int(d.N.iloc[0]), manual=MANUAL[ds], best_mean=truth.stack().idxmax())
    for seed in (0, 1):
        g = d[d.seed == seed].pivot(index="k", columns="md", values="S").loc[ks, mds]
        raw = g.stack().idxmax()
        sm = pd.DataFrame(gaussian_filter(g.values, sigma=1.0, mode="nearest"), index=ks, columns=mds)
        smo = sm.stack().idxmax()
        out[f"raw_s{seed}"] = raw; out[f"smooth_s{seed}"] = smo
        out[f"raw_s{seed}_regret"] = round(best_true - truth.loc[raw], 4)
        out[f"smooth_s{seed}_regret"] = round(best_true - truth.loc[smo], 4)
        out[f"smooth_s{seed}_C"] = round(cohm.loc[smo], 3)
    rows.append(out)
r = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(r[["dataset", "N", "manual", "best_mean", "raw_s0", "raw_s1", "smooth_s0", "smooth_s1"]].to_string(index=False))
print(r[["dataset", "raw_s0_regret", "raw_s1_regret", "smooth_s0_regret", "smooth_s1_regret", "smooth_s0_C", "smooth_s1_C"]].to_string(index=False))
