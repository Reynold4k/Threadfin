import glob, numpy as np, pandas as pd, sys
MANUAL = {"gut_gc_reporter_min2": (30, 0.3), "spont_gc_min2": (30, 0.3), "covid_bcell_min2": (30, 0.3),
          "ln_vaccine_min10": (50, 0.3), "malaria_late_min2": (200, 0.1), "mouse_np_min2": (15, 0.1),
          "mouse_rbd_min2": (30, 0.8), "gc_np_pc_min2": (50, 0.8), "malaria_min2": (50, 0.8),
          "ln_vaccine_min2": (50, 0.3)}
INTR = {"gut_gc_reporter_min2": .966, "spont_gc_min2": .993, "covid_bcell_min2": .978, "gc_np_pc_min2": .970,
        "mouse_np_min2": .928, "mouse_rbd_min2": .970, "malaria_min2": .975, "malaria_late_min2": .982,
        "ln_vaccine_min10": .982, "ln_vaccine_min2": .985}
def objectives(d, ds):
    d = d.copy()
    d["Cms"] = d[["C1.5", "C2.0", "C3.0"]].mean(axis=1)
    out = {}
    for tgt in (0.95, 0.99):
        pen = lambda C: 5 * np.maximum(0, tgt - C)
        out[f"bal10_C1.5_{tgt}"] = 0.5 * (d.local10 + d.glob) - pen(d["C1.5"])
        out[f"bal10_Cms_{tgt}"] = 0.5 * (d.local10 + d.glob) - pen(d["Cms"])
        out[f"bal30_Cms_{tgt}"] = 0.5 * (d.local30 + d.glob) - pen(d["Cms"])
        out[f"glob_Cms_{tgt}"] = d.glob - pen(d["Cms"])
        out[f"g+.25l_Cms_{tgt}"] = (d.glob + 0.25 * d.local10) / 1.25 - pen(d["Cms"])
    tgt = min(0.99, INTR[ds]); pen = lambda C: 5 * np.maximum(0, tgt - C)
    out["bal10_C1.5_adapt"] = 0.5 * (d.local10 + d.glob) - pen(d["C1.5"])
    out["bal10_Cms_adapt"] = 0.5 * (d.local10 + d.glob) - pen(d["Cms"])
    out["bal30_C1.5_adapt"] = 0.5 * (d.local30 + d.glob) - pen(d["C1.5"])
    out["bal30_Cms_adapt"] = 0.5 * (d.local30 + d.glob) - pen(d["Cms"])
    out["glob_C1.5_adapt"] = d.glob - pen(d["C1.5"])
    return pd.DataFrame(out, index=d.index)
rows = []
for f in sorted(glob.glob("grid/*.csv")):
    d = pd.read_csv(f)
    ds = d.dataset.iloc[0]
    m = d.groupby(["k", "md"]).mean(numeric_only=True)
    obj = objectives(m, ds)
    man = MANUAL.get(ds)
    for name in obj.columns:
        s = obj[name]
        best = s.idxmax()
        r = dict(dataset=ds, N=int(m.N.iloc[0]), objective=name, best_k=best[0], best_md=best[1], best=s.max())
        if man in s.index:
            r["manual_rank"] = int((s > s.loc[man]).sum()) + 1
            r["manual_regret"] = s.max() - s.loc[man]
            r["manual_C1.5"] = m.loc[man, "C1.5"]
        rows.append(r)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for name, g in res.groupby("objective"):
    print(f"\n=== {name}")
    print(g.drop(columns="objective").round(3).to_string(index=False))
    print("median manual rank:", g.manual_rank.median(), " mean regret:", round(g.manual_regret.mean(), 4))
