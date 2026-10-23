#!/usr/bin/env python
"""Turn the result tables into the Markdown tables of ``manuscript/benchmarking/README.md``.

Each table is written between ``<!-- BEGIN:<name> -->`` and ``<!-- END:<name> -->`` in the README, so the
numbers in the README always come from ``results/`` and can be refreshed after any re-run:

    python evaluate.py all && python make_figures.py && python summarise.py

All tables are also written to ``results/summary_tables.md``.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from bench_paths import RESULTS
from make_figures import DATASET, ORDER, name

README = RESULTS.parent / "README.md"
SHORT = {
    "gut_gc_reporter": "Gut GC reporter", "mouse_rbd": "RBD reporter", "mouse_np": "NP reporter",
    "gc_np_pc": "GC + plasma cells", "malaria_late": "Plasmodium d10-42", "malaria": "Plasmodium d0-14",
    "ln_vaccine": "Human LN vaccine", "spont_gc": "Spontaneous GC", "covid_bcell": "Human COVID-19",
    "larry_d2": "LARRY d2", "larry_d46": "LARRY d4/d6",
}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def read(fname):
    p = RESULTS / fname
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------------- tables

def coverage_table():
    cov = read("coverage_runtime.csv")
    if cov is None:
        return None
    cols = [("threadfin_kernel", "Threadfin"), ("clone2vec_min3", "clone2vec (default ≥ 3)"),
            ("clone2vec_min2", "clone2vec (forced ≥ 2)"), ("clonotrace_revised_min10", "Clonotrace (default ≥ 10)"),
            ("clonotrace_revised_min2", "Clonotrace (forced ≥ 2)"), ("benisse_min2", "Benisse")]
    rows = []
    for ds in ORDER:
        sub = cov[cov["dataset"] == ds].set_index("method")
        if sub.empty:
            continue
        row = {"dataset": SHORT[ds], "clones ≥ 2 cells": f"{int(sub['n_clones_ge2'].iloc[0]):,}",
               "of which 2 cells": f"{int(sub['n_two_cell_clones'].iloc[0]):,}"}
        for m, label in cols:
            if m == "benisse_min2" and sub.loc[m, "status"] != "completed" and sub.loc["benisse_min10", "status"] == "completed":
                m = "benisse_min10"
            r = sub.loc[m]
            if r["status"] != "completed":
                reason = str(r["reason"])
                row[label] = ("n/a (no BCR sequences)" if "receptor" in reason else
                              "n/a (< 3 clones ≥ 10 cells)" if "fewer than 3" in reason else
                              reason.split(":")[0] if r["status"] == "stopped" else
                              "failed (disconnected cell graph)" if "disconnected" in reason else
                              "failed: " + reason[:60] if r["status"] == "failed" else "not completed")
                continue
            two = int(r["n_two_cell_clones_represented"])
            row[label] = f"{int(r['n_clones_represented']):,} ({two:,} two-cell)" + (" [≥ 10 cells]" if m == "benisse_min10" else "")
        rows.append(row)
    return md(pd.DataFrame(rows))


def pairwise_summary(pw, methods, rna=True):
    rows = []
    for m in methods:
        s = pw[pw["method"] == m]
        if not rna:
            s = s[s["tag"] != "rna"]
        if s.empty:
            continue
        rows.append({"tool": name(m), "datasets": s["dataset"].nunique(), "annotations": len(s),
                     "tool better (95 % CI > 0)": int((s["delta_ci_low"] > 0).sum()),
                     "Threadfin better (CI < 0)": int((s["delta_ci_high"] < 0).sum()),
                     "no clear difference": int(((s["delta_ci_low"] <= 0) & (s["delta_ci_high"] >= 0)).sum()),
                     "median ΔR² (tool − Threadfin)": f"{s['delta_method_minus_reference'].median():+.3f}",
                     "median clones per comparison": int(s["n_clones"].median())})
    return md(pd.DataFrame(rows)) if rows else None


def pairwise_tables():
    pw = read("annotation_pairwise.csv")
    if pw is None:
        return {}
    forced = ["clone2vec_min2", "clonotrace_revised_min2", "clonotrace_raw_min2", "benisse_min2", "mean_pca",
              "composition", "conga_medoid"]
    native = ["clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10", "benisse_min10"]
    ablation = ["threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth", "threadfin_kernel_noshrink",
                "mean_pca"]
    out = {"pairwise_forced": pairwise_summary(pw, forced),
           "pairwise_forced_no_rna": pairwise_summary(pw, forced, rna=False),
           "pairwise_native": pairwise_summary(pw, native),
           "pairwise_ablation": pairwise_summary(pw, ablation)}
    # two-cell clones
    rows = []
    for m in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca", "composition", "conga_medoid",
              "threadfin_mean", "threadfin_kernel_noshrink"]:
        s = pw[pw["method"] == m].dropna(subset=["r2_two_cell_reference", "r2_two_cell_method"])
        if s.empty:
            continue
        rows.append({"tool": name(m), "annotations": len(s),
                     "median R² on two-cell clones, tool": f"{s['r2_two_cell_method'].median():.3f}",
                     "median R² on the same clones, Threadfin": f"{s['r2_two_cell_reference'].median():.3f}",
                     "tool higher": int((s["r2_two_cell_method"] > s["r2_two_cell_reference"]).sum()),
                     "Threadfin higher": int((s["r2_two_cell_method"] < s["r2_two_cell_reference"]).sum()),
                     "annotations with R² > 0.1 for the tool": int((s["r2_two_cell_method"] > 0.1).sum())})
    out["two_cell"] = md(pd.DataFrame(rows)) if rows else None
    # per-annotation detail of the forced comparison
    det = pw[pw["method"].isin(["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca"])].copy()
    if not det.empty:
        det["cell"] = det.apply(lambda r: f"{r['r2_method']:.2f} vs {r['r2_reference']:.2f}"
                                + (" ▲" if r["delta_ci_low"] > 0 else " ▼" if r["delta_ci_high"] < 0 else ""), axis=1)
        wide = det.pivot_table(index=["dataset", "target"], columns="method", values="cell", aggfunc="first")
        wide = wide.reindex(columns=[c for c in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca"]
                                     if c in wide.columns])
        wide.columns = [name(c) for c in wide.columns]
        wide = wide.reset_index()
        wide["order"] = wide["dataset"].map({d: i for i, d in enumerate(ORDER)})
        wide = wide.sort_values(["order"]).drop(columns="order")
        wide["dataset"] = wide["dataset"].map(SHORT)
        out["pairwise_detail"] = md(wide.fillna("—"))
    return out


def tests_table():
    files = sorted(glob.glob(str(RESULTS / "tests" / "*.csv")))
    if not files:
        return None
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    for ds in ORDER:
        s = t[t["dataset"] == ds]
        if s.empty:
            continue
        nw, ng = s[s["scenario"] == "null_within_sample"], s[s["scenario"] == "null_global"]
        obs = s[s["scenario"] == "observed"].iloc[0]
        mix = {f: s[(s["scenario"] == "mix") & np.isclose(s["fraction_real_labels"], f)] for f in (0.1, 0.25, 0.5)}
        rows.append({
            "dataset": SHORT[ds],
            "Threadfin: false positives, shuffled within sample": f"{(nw['threadfin_p'] < 0.05).mean():.2f}",
            "scirpy: ≥ 1 clone flagged, shuffled within sample": f"{(nw['scirpy_n_clones_fdr05'] > 0).mean():.2f}",
            "scirpy: share of clones flagged, shuffled within sample": f"{nw['scirpy_fraction_clones_fdr05'].mean():.3f}",
            "ANOVA: false positives, shuffled within sample": f"{(nw['cell_anova_p'] < 0.05).mean():.2f}",
            "Threadfin, shuffled across dataset": f"{(ng['threadfin_p'] < 0.05).mean():.2f}",
            "scirpy, shuffled across dataset": f"{(ng['scirpy_n_clones_fdr05'] > 0).mean():.2f}",
            "ANOVA, shuffled across dataset": f"{(ng['cell_anova_p'] < 0.05).mean():.2f}",
            "Threadfin power at 10 / 25 / 50 % real labels": " / ".join(f"{(mix[f]['threadfin_p'] < 0.05).mean():.1f}"
                                                                      for f in (0.1, 0.25, 0.5)),
            "Threadfin on the real labels: ICC (null) P": f"{obs['threadfin_icc']:.3f} ({obs['threadfin_null_mean']:.3f}) "
                                                          f"{obs['threadfin_p']:.2f}",
            "scirpy on the real labels: clones flagged": f"{int(obs['scirpy_n_clones_fdr05']):,} of {int(obs['scirpy_n_clones']):,}",
        })
    return md(pd.DataFrame(rows))


def simulation_table():
    sim = read("simulation.csv")
    if sim is None:
        return None
    small = sim["regime"] == "small"
    keep = (small & sim["set"].isin(["forced: clones >= 2 cells", "ablation: clones >= 2 cells"])) | \
           (~small & sim["set"].isin(["native: clones >= 10 cells", "ablation: clones >= 2 cells"]))
    sim = sim[keep].drop_duplicates(["dataset", "method"])
    rows = []
    for (reg, b), g in sim.groupby(["regime", "batch"], sort=False):
        for m, h in g.groupby("method", sort=False):
            rows.append({"clones": "small (half 2 cells)" if reg == "small" else "large (≥ 10 cells)",
                         "donor shift (SD)": f"{b:g}", "method": name(m), "seeds": len(h),
                         "programme purity (chance)": f"{h['programme_purity'].mean():.3f} ({h['purity_chance'].mean():.3f})",
                         "purity, two-cell clones": "" if h["purity_two_cell_clones"].isna().all()
                         else f"{h['purity_two_cell_clones'].mean():.3f}",
                         "mixture vs intermediate AUROC": f"{h['mixture_vs_intermediate_auroc'].mean():.3f}",
                         "same-donor excess": f"{h['same_donor_excess'].mean():+.3f}"})
    return md(pd.DataFrame(rows))


def runtime_table():
    cov = read("coverage_runtime.csv")
    if cov is None:
        return None
    tools = [("threadfin_kernel", "Threadfin"), ("clone2vec_min3", "clone2vec ≥ 3"), ("clone2vec_min2", "clone2vec ≥ 2"),
             ("clonotrace_revised_min10", "Clonotrace ≥ 10"), ("clonotrace_revised_min2", "Clonotrace ≥ 2"),
             ("benisse_min2", "Benisse")]
    rows = []
    for ds in ORDER:
        sub = cov[cov["dataset"] == ds].set_index("method")
        if sub.empty:
            continue
        row = {"dataset": SHORT[ds]}
        for m, label in tools:
            if m == "benisse_min2" and sub.loc[m, "status"] != "completed" and sub.loc["benisse_min10", "status"] == "completed":
                m = "benisse_min10"
            r = sub.loc[m]
            if r["status"] != "completed":
                row[label] = "—"
                continue
            sec = r["seconds_method"] if pd.notna(r["seconds_method"]) else r["wall_seconds_task"]
            mem = r["peak_rss_gb_method"] if m == "threadfin_kernel" else r["peak_rss_gb_task"]
            flag = " †" if m.startswith("clonotrace") and r.get("distance_workers") == 1 else ""
            row[label] = f"{fmt_seconds(sec)}, {mem:.1f} GB{flag}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def fmt_seconds(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "?"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def clonotrace_table():
    cov = read("coverage_runtime.csv")
    if cov is None:
        return None
    import json
    rows = []
    for _, r in cov[cov["method"].isin(["clonotrace_revised_min10", "clonotrace_revised_min2"])].iterrows():
        if r["dataset"] not in ORDER or r["status"] != "completed":
            continue
        st = json.loads(r["seconds_stages"])
        rows.append({"dataset": SHORT[r["dataset"]], "setting": "default ≥ 10" if r["method"].endswith("10") else "forced ≥ 2",
                     "clones offered": f"{int(r['n_clones_ge10'] if r['method'].endswith('10') else r['n_clones_ge2']):,}",
                     "clones kept": f"{int(r['n_clones_represented']):,}",
                     "label spreading": fmt_seconds(st.get("label_spreading")),
                     "clone distances": fmt_seconds(st.get("clone_distance")) + (" (1 process)" if r["distance_workers"] == 1
                                                                                  else f" ({int(r['distance_workers'])} workers)"),
                     "graph k used": "" if pd.isna(r["graph_k_used"]) else int(r["graph_k_used"])})
    return md(pd.DataFrame(rows)) if rows else None


def halves_tables():
    out = {}
    rep = read("split_half_reproducibility.csv")
    if rep is not None:
        rep = rep[rep["set"].str.startswith("forced") | rep["set"].str.startswith("ablation")]
        rep = rep.drop_duplicates(["dataset", "method", "seed"])
        g = rep.groupby("method").agg(datasets=("dataset", "nunique"),
                                      distance_spearman=("distance_spearman", "median"),
                                      knn_overlap=("knn_overlap", "median"), knn_chance=("knn_overlap_chance", "median"),
                                      reid_top1=("reid_top1", "median"), reid_chance=("reid_chance_top1", "median"))
        g = g.sort_values("distance_spearman", ascending=False).reset_index()
        g["method"] = g["method"].map(name)
        for c in ("distance_spearman", "knn_overlap", "knn_chance", "reid_top1"):
            g[c] = g[c].map(lambda v: f"{v:.3f}")
        g["reid_chance"] = g["reid_chance"].map(lambda v: f"{v:.4f}")
        g.columns = ["method", "datasets", "median distance Spearman", "median 10-NN overlap", "overlap by chance",
                     "median re-identification (top 1)", "re-identification by chance"]
        out["reproducibility"] = md(g)
    dis = read("annotation_recovery_disjoint.csv")
    if dis is not None:
        dis = dis[dis["set"].str.startswith("forced") | dis["set"].str.startswith("ablation")]
        dis = dis.drop_duplicates(["dataset", "target", "method", "seed", "representation_half"])
        mean = dis.groupby(["dataset", "target", "method"])["r2"].mean().reset_index()
        ref = mean[mean["method"] == "threadfin_kernel"].set_index(["dataset", "target"])["r2"]
        mean["delta"] = mean["r2"] - ref.reindex(pd.MultiIndex.from_frame(mean[["dataset", "target"]])).to_numpy()
        g = mean[mean["method"] != "threadfin_kernel"].groupby("method").agg(
            annotations=("delta", "size"), median_delta=("delta", "median"),
            tool_higher=("delta", lambda v: int((v > 0).sum())), threadfin_higher=("delta", lambda v: int((v < 0).sum())))
        g = g.reset_index()
        g["method"] = g["method"].map(name)
        g["median_delta"] = g["median_delta"].map(lambda v: f"{v:+.3f}")
        g.columns = ["method", "annotations", "median ΔR² (tool − Threadfin)", "tool higher", "Threadfin higher"]
        out["disjoint"] = md(g)
    return out


def larry_table():
    lk = read("larry_clone_linking.csv")
    if lk is None or lk.empty:
        return None
    lk = lk.sort_values(["set", "mrr"], ascending=[True, False])
    rows = [{"setting": r["set"], "method": name(r["method"]), "query clones": f"{int(r['n_query_clones']):,}",
             "day-6 candidates": f"{int(r['n_candidate_day6_clones']):,}", "top 1": f"{r['top1']:.3f}",
             "top 10": f"{r['top10']:.3f}", "mean reciprocal rank": f"{r['mrr']:.3f}",
             "median rank": f"{r['median_rank']:.0f}", "chance top 1": f"{r['chance_top1']:.4f}"} for _, r in lk.iterrows()]
    return md(pd.DataFrame(rows))


def datasets_table():
    import json

    from bench_paths import INPUTS
    from evaluate import TARGETS

    overview = pd.read_csv(RESULTS.parents[1] / "dataset_overview.csv").set_index("dataset")
    rows = []
    for ds in ORDER:
        meta_p = INPUTS / ds / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        study = overview.loc[ds, "study"] if ds in overview.index else "Weinreb et al. 2020 Science"
        acc = overview.loc[ds, "accession"] if ds in overview.index else "GSE140802"
        rows.append({"dataset": SHORT[ds], "study": study, "accession": acc, "cells": f"{meta['n_cells']:,}",
                     "clones ≥ 2 cells": f"{meta['n_clones_ge2']:,}", "≥ 10 cells": f"{meta['n_clones_ge10']:,}",
                     "donors / samples": f"{meta.get('n_donors', '—')} / {meta.get('n_samples', '—')}",
                     "annotations scored": "; ".join(t.name for t in TARGETS.get(ds, [])) or
                                           "re-identify a barcode's day-6 clone from day 4"})
    return md(pd.DataFrame(rows))


def status_block():
    """Which full-data runs had not completed, and why, when the tables were generated."""
    import datetime

    cov = read("coverage_runtime.csv")
    if cov is None:
        return None
    tools = ["clone2vec_min3", "clone2vec_min2", "clonotrace_revised_min10", "clonotrace_revised_min2", "benisse_min2",
             "benisse_min10", "threadfin_kernel"]
    sub = cov[cov["method"].isin(tools) & cov["dataset"].isin(ORDER)]
    running, failed, stopped = [], [], []
    for _, r in sub.iterrows():
        reason = str(r["reason"]) if pd.notna(r["reason"]) else ""
        if r["status"] == "completed" or reason.startswith("not applicable"):
            continue
        is_ln = r["dataset"] == "ln_vaccine"           # Benisse: >= 10 cells on GSE195673, >= 2 elsewhere
        if (r["method"] == "benisse_min2" and is_ln) or (r["method"] == "benisse_min10" and not is_ln):
            continue
        label = f"{name(r['method'])} on {SHORT[r['dataset']]}"
        if r["status"] == "stopped":
            stopped.append(f"{label} — {reason}")
            continue
        if r["status"] == "failed":
            if "graph_k=" in reason and "disconnected" in reason:
                tried = ", ".join(x.split(":")[0].split("=")[1] for x in reason.split(" | "))
                reason = (f"the cell graph stayed disconnected at graph_k = {tried} (Clonotrace: \"Clone-distance output "
                          "contains infinite distances because the cell graph has disconnected clone-bearing components\")")
            failed.append(f"{label} — {reason[:300]}")
        else:
            running.append(label)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"*Status of the full-data runs when these tables were generated ({stamp}, Melbourne time).*", ""]
    lines.append("Still running: " + ("; ".join(running) if running else "none") + ".")
    lines.append("")
    lines.append("Failed (each with the reason the tool gave): " + ("; ".join(failed) if failed else "none") + ".")
    if stopped:
        lines.append("")
        lines.append("Stopped before completion: " + "; ".join(stopped))
    from bench_paths import RUNS
    pending = []
    for ds in ORDER:
        for half in ("halfA_s0", "halfB_s0"):
            d = RUNS / ds / half
            if not d.is_dir():
                continue
            for run in sorted(d.glob("clonotrace_min*")) + sorted(d.glob("benisse_min*")):
                if not (run / "run.json").exists():
                    pending.append(f"{run.name} on {SHORT[ds]} ({half})")
    lines.append("")
    lines.append("Split-half runs still running (sections 6.4–6.5 use the halves that are complete): "
                 + ("; ".join(pending) if pending else "none") + ".")
    return "\n".join(lines)


def overview_tables():
    """Rules of the overview figure, and the usability checklist as a matrix (evidence in the CSV)."""
    import json

    out = {}
    rules = RESULTS / "overview_rules.json"
    if rules.exists():
        from make_overview import COLUMNS
        label = {k: lab.replace("-\n", "").replace("\n", " ") for k, _, lab in COLUMNS}
        r = json.loads(rules.read_text())
        out["overview_rules"] = md(pd.DataFrame({"column": [label[k] for k in r], "how the verdict is reached": list(r.values())}))
    u = read("usability.csv")
    if u is not None:
        m = u.pivot_table(index=["group", "criterion"], columns="tool", values="score", aggfunc="first", sort=False)
        m = m.reindex(columns=["Threadfin", "Clonotrace", "clone2vec", "Benisse", "CoNGA", "scirpy"])
        m = m.map(lambda v: "n.a." if pd.isna(v) else {1.0: "yes", 0.5: "partly", 0.0: "no"}[float(v)]) \
            if hasattr(m, "map") else m.applymap(lambda v: "n.a." if pd.isna(v) else {1.0: "yes", 0.5: "partly", 0.0: "no"}[float(v)])
        out["usability"] = md(m.reset_index().drop(columns="group"))
    return out


def resource_tables():
    out = {}
    res = read("resources.csv")
    if res is not None:
        tools = [("threadfin_run", "Threadfin, complete analysis"), ("clone2vec_min2", "clone2vec"),
                 ("clonotrace_revised_min2", "Clonotrace"), ("benisse_min2", "Benisse")]
        c = res[res["experiment"].eq("cores")]
        rows = []
        for n in sorted(c["cores"].unique()):
            row = {"cores": int(n)}
            for t, lab in tools:
                r = c[c["tool"].eq(t) & c["cores"].eq(n)]
                row[lab] = ("—" if r.empty else "running" if r["status"].iloc[0] != "completed"
                            else f"{fmt_seconds(r['seconds'].iloc[0])}, {r['peak_rss_gb'].iloc[0]:.1f} GB")
            rows.append(row)
        out["resources_cores"] = md(pd.DataFrame(rows))
        g = res[res["experiment"].eq("size")]
        tools_g = [("threadfin_run", "Threadfin, complete analysis"), ("threadfin_kernel", "Threadfin, clone profiles"),
                   ("clone2vec_min2", "clone2vec"), ("clonotrace_revised_min2", "Clonotrace"), ("benisse_min2", "Benisse")]
        rows = []
        for ds in sorted(g["dataset"].unique(), key=lambda d: float(d.split("_f")[1])):
            sub = g[g["dataset"].eq(ds)]
            row = {"subset of GSE195673": f"{100 * float(ds.split('_f')[1]):g} %",
                   "clones ≥ 2 cells": f"{int(sub['n_clones_ge2'].iloc[0]):,}"}
            for t, lab in tools_g:
                r = sub[sub["tool"].eq(t)]
                row[lab] = ("—" if r.empty else "running" if r["status"].iloc[0] != "completed"
                            else f"{fmt_seconds(r['seconds'].iloc[0])}, {r['peak_rss_gb'].iloc[0]:.1f} GB")
            rows.append(row)
        out["resources_size"] = md(pd.DataFrame(rows))
    rep = read("resources_reproducibility.csv")
    if rep is not None:
        lab = {"threadfin_run": "Threadfin (clone map and programmes)", "clone2vec_min2": "clone2vec",
               "clonotrace_revised_min2": "Clonotrace", "benisse_min2": "Benisse"}
        rows = []
        for t, g in rep.groupby("tool", sort=False):
            ident = bool(g["identical"].all())
            rho = g["distance_spearman"].min() if g["distance_spearman"].notna().any() else np.nan
            diff = g["max_abs_difference"].max() if g["max_abs_difference"].notna().any() else np.nan
            if t == "threadfin_run":
                compared = (f"4.4.0 with 1 and with 8 cores vs the released 4.3.1, on {g['dataset'].nunique()} datasets "
                            "(clone-map coordinates and programme labels)")
            else:
                compared = f"{int(g['reference_cores'].iloc[0])} core(s) vs " + ", ".join(str(int(c)) for c in g["cores"])
            rows.append({"tool": lab.get(t, t), "compared": compared, "identical": "yes" if ident else "no",
                         "lowest Spearman correlation of clone distances": "" if np.isnan(rho) else f"{rho:.6f}",
                         "largest difference of a clone distance": "" if np.isnan(diff) else f"{diff:.3g}"})
        out["resources_repro"] = md(pd.DataFrame(rows))
    full = read("threadfin_full_run.csv")
    if full is not None:
        rows = []
        for _, r in full.iterrows():
            rows.append({"dataset": SHORT.get(r["dataset"], r["dataset"]), "cells": f"{int(r['n_cells']):,}",
                         "clones ≥ 2 cells": f"{int(r['n_clones_ge2']):,}",
                         "profiles": fmt_seconds(r["seconds_clone_profiles"]),
                         "coherence test (200 permutations)": fmt_seconds(r["seconds_clonal_coherence"]),
                         "clone map (automatic settings)": fmt_seconds(r["seconds_clone_map"]),
                         "programmes (30 bootstraps)": fmt_seconds(r["seconds_programmes"]),
                         "one label test": fmt_seconds(r.get("seconds_profile_association", np.nan)),
                         "total": fmt_seconds(r["total_seconds"]), "peak memory": f"{r['peak_rss_gb']:.1f} GB"})
        out["threadfin_full_run"] = md(pd.DataFrame(rows))
    return out


def write(tables: dict):
    body = []
    for k, v in tables.items():
        if v:
            body.append(f"## {k}\n\n{v}\n")
    (RESULTS / "summary_tables.md").write_text("# Benchmark summary tables (generated by summarise.py)\n\n" + "\n".join(body))
    if README.exists():
        text = README.read_text()
        for k, v in tables.items():
            if not v:
                continue
            pat = re.compile(rf"(<!-- BEGIN:{k} -->).*?(<!-- END:{k} -->)", re.S)
            if pat.search(text):
                text = pat.sub(lambda m: m.group(1) + "\n\n" + v + "\n\n" + m.group(2), text)
            else:
                print(f"marker {k} not found in README")
        README.write_text(text)
    print("tables:", ", ".join(k for k, v in tables.items() if v))


def main():
    tables = {"status": status_block(), "datasets": datasets_table(), "coverage": coverage_table(), **pairwise_tables(), "tests": tests_table(),
              "simulation": simulation_table(), "runtime": runtime_table(), "clonotrace_stages": clonotrace_table(),
              **halves_tables(), "larry": larry_table(), **overview_tables(), **resource_tables()}
    write(tables)


if __name__ == "__main__":
    main()
