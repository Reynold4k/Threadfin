#!/usr/bin/env python
"""Turn the result tables into the Markdown tables and figure legends of ``manuscript/benchmarking/README.md``.

Each table or legend is written between ``<!-- BEGIN:<name> -->`` and ``<!-- END:<name> -->`` in the README, so
the numbers in the README always come from ``results/`` and can be refreshed after any re-run
(``src/refresh_benchmark.sh all`` in the working folder runs every step). All tables are also written to
``results/summary_tables.md``.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from bench_paths import BENCH, RESULTS
from make_figures import (FIG, ORDER, SHORT_DS, matched_disjoint, matched_reproducibility, name, simulation_frame,
                          wilson)

README = RESULTS.parent / "README.md"
SHORT = SHORT_DS
TF = "threadfin_kernel"


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def read(fname):
    p = RESULTS / fname
    return pd.read_csv(p) if p.exists() else None


def fmt_p(p):
    return "" if p is None or not np.isfinite(p) else f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def ci_text(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f} % ({100 * float(lo):.0f}–{100 * float(hi):.0f} %)"


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


def calls(s):
    c = s["call"]
    return int(c.eq("better").sum()), int(c.eq("worse").sum()), int(c.eq("similar").sum()), int(c.eq("unresolved").sum())


def pairwise_summary(pw, methods, rna=True):
    rows = []
    for m in methods:
        s = pw[pw["method"] == m]
        if not rna:
            s = s[s["tag"] != "rna"]
        if s.empty:
            continue
        b, w, si, u = calls(s)
        rows.append({"tool": name(m), "datasets": s["dataset"].nunique(), "annotations": len(s),
                     "tool better (95 % CI > 0)": b, "Threadfin better (CI < 0)": w, "no clear difference": si,
                     "unresolved": u,
                     "median ΔR² (tool − Threadfin)": f"{s['delta_method_minus_reference'].median():+.3f}",
                     "median clones per comparison": int(s["n_clones"].median())})
    return md(pd.DataFrame(rows)) if rows else None


FORCED = ["clone2vec_min2", "clonotrace_revised_min2", "clonotrace_raw_min2", "benisse_min2", "mean_pca", "composition",
          "conga_medoid"]
NATIVE = ["clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10", "benisse_min10"]
ABLATION = ["threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth", "threadfin_kernel_noshrink",
            "mean_pca"]
EXACT = ["clonotrace_exact_min2", "clonotrace_exact_raw_min2", "clonotrace_exact_min10", "clonotrace_exact_raw_min10"]


def pairwise_tables():
    pw = read("annotation_pairwise.csv")
    if pw is None:
        return {}
    out = {"pairwise_forced": pairwise_summary(pw, FORCED),
           "pairwise_forced_no_rna": pairwise_summary(pw, FORCED, rna=False),
           "pairwise_native": pairwise_summary(pw, NATIVE),
           "pairwise_ablation": pairwise_summary(pw, ABLATION),
           "pairwise_exact": pairwise_summary(pw, EXACT)}
    # two-cell clones: every annotation, as in the figure, and the medians over the measured annotations alone
    # (the RNA-derived annotations are easy by construction)
    rows = []
    for m in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca", "composition", "conga_medoid",
              "threadfin_mean", "threadfin_kernel_noshrink"]:
        s = pw[pw["method"] == m].dropna(subset=["r2_two_cell_reference", "r2_two_cell_method"])
        if s.empty:
            continue
        meas = s[s["tag"] != "rna"]
        c2 = s["call_two_cell"] if "call_two_cell" in s else pd.Series("similar", index=s.index)
        rows.append({"tool": name(m), "annotations": len(s),
                     "median R² on two-cell clones, tool": f"{s['r2_two_cell_method'].median():.3f}",
                     "median R² on the same clones, Threadfin": f"{s['r2_two_cell_reference'].median():.3f}",
                     "the same, measured annotations only: tool / Threadfin":
                         f"{meas['r2_two_cell_method'].median():.3f} / {meas['r2_two_cell_reference'].median():.3f} ({len(meas)})",
                     "tool higher": int((s["r2_two_cell_method"] > s["r2_two_cell_reference"]).sum()),
                     "Threadfin higher": int((s["r2_two_cell_method"] < s["r2_two_cell_reference"]).sum()),
                     "interval excludes 0: tool / Threadfin better": f"{int(c2.eq('better').sum())} / {int(c2.eq('worse').sum())}",
                     "annotations with R² > 0.1 for the tool": int((s["r2_two_cell_method"] > 0.1).sum())})
    out["two_cell"] = md(pd.DataFrame(rows)) if rows else None
    # per-annotation detail of the forced comparison
    det = pw[pw["method"].isin(["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca"])].copy()
    if not det.empty:
        mark = {"better": " ▲", "worse": " ▼", "unresolved": " ?"}
        det["cell"] = det.apply(lambda r: f"{r['r2_method']:.2f} vs {r['r2_reference']:.2f}" + mark.get(r["call"], ""),
                                axis=1)
        wide = det.pivot_table(index=["dataset", "target"], columns="method", values="cell", aggfunc="first")
        wide = wide.reindex(columns=[c for c in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca"]
                                     if c in wide.columns])
        wide.columns = [name(c) for c in wide.columns]
        wide = wide.reset_index()
        wide["order"] = wide["dataset"].map({d: i for i, d in enumerate(ORDER)})
        wide = wide.sort_values(["order"]).drop(columns="order")
        wide["dataset"] = wide["dataset"].map(SHORT)
        out["pairwise_detail"] = md(wide.fillna("—"))
    out["seed_control"] = seed_control_table(pw)
    return out


def seed_control_table(pw):
    """The negative control: Threadfin with random seed 1 or 2 against the default seed 0. Every call is false."""
    rows = []
    for lab, ids in (("all clones ≥ 2 cells (Fig. 4)", ["threadfin_kernel_seed1", "threadfin_kernel_seed2"]),
                     ("clones ≥ 3 cells (Fig. 5)", ["threadfin_kernel_seed1_ge3", "threadfin_kernel_seed2_ge3"]),
                     ("clones ≥ 10 cells (Fig. 5)", ["threadfin_kernel_seed1_ge10", "threadfin_kernel_seed2_ge10"])):
        for rna in (False, True):
            s = pw[pw["method"].isin(ids)]
            if not rna:
                s = s[s["tag"] != "rna"]
            if s.empty:
                continue
            b, w, si, u = calls(s)
            ad = np.abs(s["delta_method_minus_reference"])
            rows.append({"comparisons": lab, "annotations": "measured" if not rna else "all (incl. RNA)",
                         "comparisons (2 seeds)": len(s), "called better": b, "called worse": w, "unresolved": u,
                         "false-call rate (Wilson 95 % CI)": ci_text(b + w, len(s)),
                         "binomial P against 5 %": fmt_p(stats.binomtest(b + w, len(s), 0.05).pvalue),
                         "median |ΔR²|": f"{ad.median():.3f}", "largest |ΔR²|": f"{ad.max():.3f}"})
    return md(pd.DataFrame(rows)) if rows else None


def exact_table():
    """Clonotrace's exact optimal-transport runs of 2026-10-25: what finished and what failed."""
    cov, hosts = read("coverage_runtime.csv"), read("run_hosts.csv")
    if cov is None:
        return None
    rows = []
    for m in ("clonotrace_exact_min2", "clonotrace_exact_min10"):
        sub = cov[cov["method"].eq(m) & (cov["dataset"].isin(ORDER) | cov["dataset"].str.startswith("sim_"))]
        for _, r in sub.iterrows():
            if r["status"] in ("not run",):
                continue
            job = None
            if hosts is not None:
                h = hosts[hosts["dataset"].eq(r["dataset"]) & hosts["variant"].eq("full") & hosts["tool_run"].eq(m)]
                job = h.iloc[0] if len(h) else None
            approx = cov[cov["dataset"].eq(r["dataset"]) & cov["method"].eq(m.replace("exact_", "revised_"))]
            ds = r["dataset"]
            if ds.startswith("sim_"):
                _, regime, b, seed = ds.split("_")
                ds_label = f"simulation, {regime} clones, shift {b[1:]}, seed {seed[1:]}"
            else:
                ds_label = SHORT.get(ds, ds)
            rows.append({"dataset": ds_label, "setting": "≥ 2 cells" if m.endswith("2") else "≥ 10 cells",
                         "status": r["status"],
                         "clones kept": f"{int(r['n_clones_represented']):,}" if r["status"] == "completed" else "",
                         "clone distances": (f"{json.loads(r['seconds_stages']).get('clone_distance', np.nan) / 60:.1f} min"
                                             if r["status"] == "completed" else ""),
                         "approximate distance": (f"{json.loads(approx['seconds_stages'].iloc[0]).get('clone_distance', np.nan) / 60:.1f} min"
                                                  if len(approx) and approx["status"].iloc[0] == "completed" else ""),
                         "job peak memory": (f"{job['job_max_rss_gb']:.1f} GB" if job is not None and pd.notna(job["job_max_rss_gb"])
                                             and r["status"] != "cancelled" else ""),
                         "reason": ("the tool: \"" + str(r["reason"])[:80] + "\"") if r["status"] == "failed" else
                                   "cancelled by the authors" if r["status"] == "cancelled" else ""})
    return md(pd.DataFrame(rows)) if rows else None


def tests_table():
    files = sorted(glob.glob(str(RESULTS / "tests" / "*.csv")))
    if not files:
        return None
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pf = sorted(glob.glob(str(RESULTS / "tests" / "power" / "*.csv")))
    power = pd.concat([pd.read_csv(f) for f in pf], ignore_index=True) if pf else None
    rows = []
    for ds in ORDER:
        s = t[t["dataset"] == ds]
        if s.empty:
            continue
        nw, ng = s[s["scenario"] == "null_within_sample"], s[s["scenario"] == "null_global"]
        obs = s[s["scenario"] == "observed"].iloc[0]
        src = power[power["dataset"] == ds] if power is not None else s[s["scenario"] == "mix"]
        mix = {f: src[np.isclose(src["fraction_real_labels"], f)] for f in (0.1, 0.25, 0.5)}
        rows.append({
            "dataset": SHORT[ds],
            "Threadfin: false positives, shuffled within sample": ci_text(int((nw["threadfin_p"] < 0.05).sum()), len(nw)),
            "scirpy: ≥ 1 clone flagged, shuffled within sample": f"{(nw['scirpy_n_clones_fdr05'] > 0).mean():.2f}",
            "scirpy: share of clones flagged, shuffled within sample": f"{nw['scirpy_fraction_clones_fdr05'].mean():.3f}",
            "ANOVA: false positives, shuffled within sample": f"{(nw['cell_anova_p'] < 0.05).mean():.2f}",
            "Threadfin, shuffled across dataset": ci_text(int((ng["threadfin_p"] < 0.05).sum()), len(ng)),
            "scirpy, shuffled across dataset": f"{(ng['scirpy_n_clones_fdr05'] > 0).mean():.2f}",
            "ANOVA, shuffled across dataset": f"{(ng['cell_anova_p'] < 0.05).mean():.2f}",
            f"Threadfin power at 10 / 25 / 50 % real labels ({len(mix[0.1])} replicates each)":
                " / ".join(f"{(mix[f]['threadfin_p'] < 0.05).mean():.2f}" for f in (0.1, 0.25, 0.5)),
            "Threadfin on the real labels: ICC (null) P": f"{obs['threadfin_icc']:.3f} ({obs['threadfin_null_mean']:.3f}) "
                                                          f"{obs['threadfin_p']:.2f}",
            "scirpy on the real labels: clones flagged": f"{int(obs['scirpy_n_clones_fdr05']):,} of {int(obs['scirpy_n_clones']):,}",
        })
    # pooled calibration of Threadfin's test
    pooled = []
    for scen, lab in (("null_within_sample", "shuffled within sample"), ("null_global", "shuffled across the dataset")):
        k, n = int((t.loc[t["scenario"].eq(scen), "threadfin_p"] < 0.05).sum()), int(t["scenario"].eq(scen).sum())
        pooled.append(f"{lab}: {k} of {n}, {ci_text(k, n)}")
    return md(pd.DataFrame(rows)) + "\n\nThreadfin's coherence test over all nine datasets: " + "; ".join(pooled) + "."


def simulation_tables():
    sim = read("simulation.csv")
    if sim is None:
        return {}
    f = simulation_frame(sim)
    rows = []
    for (reg, b), g in f.groupby(["regime", "batch"], sort=False):
        for tool in ["threadfin_kernel", "threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
                     "threadfin_kernel_noshrink", "clone2vec", "clonotrace", "mean_pca", "composition", "conga_medoid"]:
            h = g[g["tool"].eq(tool)]
            if h.empty:
                continue
            lab = {"clone2vec": "clone2vec (" + ("forced to ≥ 2 cells" if reg == "small" else "default, ≥ 3 cells") + ")",
                   "clonotrace": "Clonotrace (" + ("forced to ≥ 2 cells" if reg == "small" else "default, ≥ 10 cells") + ")"}
            rows.append({"clones": "small (half 2 cells)" if reg == "small" else "large (≥ 10 cells)",
                         "donor shift (SD)": f"{b:g}", "method": lab.get(tool, name(tool)), "seeds": len(h),
                         "programme purity (chance)": f"{h['programme_purity'].mean():.3f} ({h['purity_chance'].mean():.3f})",
                         "purity, two-cell clones": "" if h["purity_two_cell_clones"].isna().all()
                         else f"{h['purity_two_cell_clones'].mean():.3f}",
                         "mixture vs intermediate AUROC": "" if h["mixture_vs_intermediate_auroc"].isna().all()
                         else f"{h['mixture_vs_intermediate_auroc'].mean():.3f}",
                         "same-donor excess": f"{h['same_donor_excess'].mean():+.3f}"})
    out = {"simulation": md(pd.DataFrame(rows))}
    # seed-level comparisons with Threadfin on the same clones (the overview's rule)
    from make_overview import Overview
    ov = Overview()
    rows = []
    for key in ["clone2vec_min3", "clone2vec_min2", "clonotrace_revised_min10", "clonotrace_revised_min2",
                "clonotrace_exact", "mean_pca", "composition", "conga_medoid", "threadfin_kernel_seed1"]:
        for what, lab in (("programme_purity", "programme purity"), ("mixture_vs_intermediate_auroc", "mixture vs intermediate AUROC")):
            c = ov.simulation(key, what)
            if c.verdict == "na":
                continue
            rows.append({"tool": name(key) if key != "clonotrace_exact" else "Clonotrace, exact OT (≥ 2)",
                         "readout": lab, "tool vs Threadfin (mean over seeds)": c.text.replace("\n", " "),
                         "seeds": c.n, "Wilcoxon P over seeds": fmt_p(c.p) if c.p is not None else "",
                         "verdict": {"descriptive": "too few seeds to test"}.get(c.verdict, c.verdict)})
    out["simulation_tests"] = md(pd.DataFrame(rows))
    return out


def runtime_table():
    cov, hosts = read("coverage_runtime.csv"), read("run_hosts.csv")
    if cov is None:
        return None
    peaks = {}
    if hosts is not None:
        h = hosts[hosts["variant"].eq("full") & hosts["alone_in_job"].eq(True)]
        peaks = {(r.dataset, r.tool_run): (r.job_max_rss_gb, r.job_cpu_hours) for r in h.itertuples()}
    tools = [("threadfin_kernel", "Threadfin"), ("clone2vec_min3", "clone2vec ≥ 3"), ("clone2vec_min2", "clone2vec ≥ 2"),
             ("clonotrace_revised_min10", "Clonotrace ≥ 10"), ("clonotrace_revised_min2", "Clonotrace ≥ 2"),
             ("benisse_min2", "Benisse")]
    rundir = {"clonotrace_revised_min10": "clonotrace_min10", "clonotrace_revised_min2": "clonotrace_min2"}
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
            text = f"{fmt_seconds(sec)}, "
            if m.startswith("clonotrace"):
                jp = peaks.get((ds, rundir[m]))
                text += (f"{jp[0]:.1f} GB (job)" if jp and pd.notna(jp[0]) else f"≥ {mem:.1f} GB")
                flag = " †" if r.get("distance_workers") == 1 else ""
                text += f", {r['core_hours']:,.1f} core-h{flag}" if pd.notna(r.get("core_hours")) else flag
            else:
                text += f"{mem:.1f} GB"
            row[label] = text
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
                     "core-hours": f"{r['core_hours']:,.1f}" if pd.notna(r.get("core_hours")) else "",
                     "graph k used": "" if pd.isna(r["graph_k_used"]) else int(r["graph_k_used"])})
    return md(pd.DataFrame(rows)) if rows else None


def halves_tables():
    out = {}
    rep = read("split_half_reproducibility.csv")
    if rep is not None:
        mr = matched_reproducibility(rep)
        rows = []
        for m, g in mr.groupby("method", sort=False):
            rows.append({"method": name(m), "set": g["set"].iloc[0].split(":")[0], "datasets": len(g),
                         "median distance Spearman": f"{g['distance_spearman'].median():.3f}",
                         "Threadfin, same clones": f"{g['distance_spearman_tf'].median():.3f}",
                         "median 10-NN overlap": f"{g['knn_overlap'].median():.3f}",
                         "Threadfin, same clones ": f"{g['knn_overlap_tf'].median():.3f}",
                         "overlap by chance": f"{g['knn_overlap_chance'].median():.3f}",
                         "median re-identification (top 1)": f"{g['reid_top1'].median():.3f}",
                         "Threadfin, same clones  ": f"{g['reid_top1_tf'].median():.3f}",
                         "higher / lower than Threadfin": (f"{int((g['reid_top1'] > g['reid_top1_tf']).sum())} / "
                                                           f"{int((g['reid_top1'] < g['reid_top1_tf']).sum())}"
                                                           if m != TF else ""),
                         "re-identification by chance": f"{g['reid_chance_top1'].median():.4f}"})
        out["reproducibility"] = md(pd.DataFrame(rows))
    dis = read("annotation_recovery_disjoint.csv")
    if dis is not None:
        mdj = matched_disjoint(dis)
        rows = []
        for m, g in mdj.groupby("method", sort=False):
            up, down = int((g["delta"] > 0).sum()), int((g["delta"] < 0).sum())
            gm = g[g["tag"] != "rna"]
            um, dm = int((gm["delta"] > 0).sum()), int((gm["delta"] < 0).sum())
            rows.append({"method": name(m), "set": g["set"].iloc[0].split(":")[0], "annotations": len(g),
                         "median ΔR² (tool − Threadfin, same halves and clones)": f"{g['delta'].median():+.3f}",
                         "tool higher": up, "Threadfin higher": down,
                         "measured annotations only: higher / lower (sign test P)":
                             f"{um} / {dm} ({fmt_p(stats.binomtest(um, um + dm).pvalue) if um + dm else ''})"})
        out["disjoint"] = md(pd.DataFrame(rows))
    return out


def larry_table():
    lk = read("larry_clone_linking.csv")
    if lk is None or lk.empty:
        return None
    lk = lk.sort_values(["set", "mrr"], ascending=[True, False])
    rows = [{"setting": r["set"], "method": name(r["method"]), "query clones": f"{int(r['n_query_clones']):,}",
             "day-6 candidates": f"{int(r['n_candidate_day6_clones']):,}", "top 1": f"{r['top1']:.3f}",
             "top 10": f"{r['top10']:.3f}", "mean reciprocal rank": f"{r['mrr']:.3f}",
             "difference to Threadfin (95 % CI)": ("" if r["method"] == TF else
                                                   f"{r['mrr_delta_vs_threadfin']:+.3f} [{r['mrr_delta_ci_low']:+.3f}, "
                                                   f"{r['mrr_delta_ci_high']:+.3f}]"),
             "queries with tied candidates": f"{r['share_queries_tied']:.0%}",
             "median rank": f"{r['median_rank']:.0f}", "chance top 1": f"{r['chance_top1']:.4f}"} for _, r in lk.iterrows()]
    return md(pd.DataFrame(rows))


def datasets_table():
    from bench_paths import INPUTS
    from evaluate import TARGETS

    overview = pd.read_csv(BENCH.parent / "dataset_overview.csv").set_index("dataset")
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
        elif r["status"] == "failed":
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
    ex = cov[cov["method"].isin(["clonotrace_exact_min2", "clonotrace_exact_min10"]) & cov["status"].ne("not run")]
    if not ex.empty:
        real = ex[ex["dataset"].isin(ORDER)]
        lines.append("")
        lines.append(f"Clonotrace with the exact OT distance (runs of 2026-10-25): {len(ex)} started, "
                     f"{int(ex['status'].eq('completed').sum())} completed ({int(real['status'].eq('completed').sum())} on real "
                     f"data), {int(ex['status'].eq('failed').sum())} failed, {int(ex['status'].eq('cancelled').sum())} "
                     "cancelled by the authors (section 6.11).")
    return "\n".join(lines)


def overview_tables():
    """Rules of the overview figure, the usability checklist as a matrix (evidence in the CSV) and the α sensitivity."""
    out = {}
    rules = RESULTS / "overview_rules.json"
    if rules.exists():
        from make_overview import COLUMNS
        label = {k: " ".join(lab.replace("-\n", "").split()) for k, _, lab in COLUMNS}
        r = json.loads(rules.read_text())
        out["overview_rules"] = md(pd.DataFrame({"column": [label[k] for k in r], "how the verdict is reached": list(r.values())}))
    u = read("usability.csv")
    if u is not None:
        m = u.pivot_table(index=["group", "criterion"], columns="tool", values="score", aggfunc="first", sort=False)
        m = m.reindex(columns=["Threadfin", "Clonotrace", "clone2vec", "Benisse", "CoNGA", "scirpy"])
        m = m.map(lambda v: "n.a." if pd.isna(v) else {1.0: "yes", 0.5: "partly", 0.0: "no"}[float(v)])
        out["usability"] = md(m.reset_index().drop(columns="group"))
    a = read("overview_alpha_sensitivity.csv")
    ov = read("overview.csv")
    if a is not None and ov is not None:
        tested = ov[ov["p"].notna()]
        txt = (f"Of the {len(tested)} tiles whose verdict comes from a test, {len(a)} change when the tests use P < 0.01 "
               "instead of 0.05")
        if len(a):
            a = a.assign(**{"P at 0.05": a["p_005"].map(fmt_p)})
            txt += ":\n\n" + md(a.rename(columns={"verdict_005": "verdict at 0.05", "verdict_001": "verdict at 0.01",
                                                  "shown_005": "shown"})[["row", "column", "shown", "P at 0.05",
                                                                         "verdict at 0.05", "verdict at 0.01"]])
        else:
            txt += "."
        out["alpha_sensitivity"] = txt
    if ov is not None:
        t = ov[ov["verdict"].eq("count")][["row", "shown"]]
        before = {"better": [], "worse": []}
        out["tally"] = md(t.rename(columns={"row": "row", "shown": "tiles better / worse"}))
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
                lb = "≥ " if t.startswith("clonotrace") else ""
                row[lab] = ("—" if r.empty else "running" if r["status"].iloc[0] != "completed"
                            else f"{fmt_seconds(r['seconds'].iloc[0])}, {lb}{r['peak_rss_gb'].iloc[0]:.1f} GB")
            rows.append(row)
        out["resources_cores"] = md(pd.DataFrame(rows))
        g = res[res["experiment"].eq("size")]
        hosts = read("run_hosts.csv")
        jp = {}
        if hosts is not None:
            h = hosts[hosts["alone_in_job"].eq(True)]
            jp = {(r.dataset, r.tool_run): r.job_max_rss_gb for r in h.itertuples()}
        tools_g = [("threadfin_run", "Threadfin, complete analysis"), ("threadfin_kernel", "Threadfin, clone profiles"),
                   ("clone2vec_min2", "clone2vec"), ("clonotrace_revised_min2", "Clonotrace"), ("benisse_min2", "Benisse")]
        rows = []
        for ds in sorted(g["dataset"].unique(), key=lambda d: float(d.split("_f")[1])):
            sub = g[g["dataset"].eq(ds)]
            row = {"subset of GSE195673": f"{100 * float(ds.split('_f')[1]):g} %",
                   "clones ≥ 2 cells": f"{int(sub['n_clones_ge2'].iloc[0]):,}"}
            for t, lab in tools_g:
                r = sub[sub["tool"].eq(t)]
                if r.empty or r["status"].iloc[0] != "completed":
                    row[lab] = "—" if r.empty else "running"
                    continue
                mem = f"{r['peak_rss_gb'].iloc[0]:.1f} GB"
                if t.startswith("clonotrace"):
                    j = jp.get((ds, "clonotrace_min2"))
                    mem = f"{j:.1f} GB (job)" if j is not None and pd.notna(j) else "≥ " + mem
                row[lab] = f"{fmt_seconds(r['seconds'].iloc[0])}, {mem}"
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
    out["hosts"] = hosts_table()
    out["budget"] = budget_table()
    return out


def hosts_table():
    h = read("run_hosts.csv")
    if h is None:
        return None
    real = h[h["dataset"].isin(ORDER) & h["variant"].eq("full")]
    groups = [("per-dataset runs, Threadfin and the baselines", real[real["tool_run"].str.startswith(("threadfin", "mean_pca", "composition", "conga"))]),
              ("per-dataset runs, clone2vec", real[real["tool_run"].str.startswith("clone2vec")]),
              ("per-dataset runs, Clonotrace (approximate distance)", real[real["tool_run"].isin(["clonotrace_min2", "clonotrace_min10"])]),
              ("per-dataset runs, Clonotrace (exact OT)", real[real["tool_run"].str.startswith("clonotrace_exact")]),
              ("per-dataset runs, Benisse", real[real["tool_run"].str.startswith("benisse")])]
    res = h[h["variant"].str.startswith("full_w") | h["dataset"].str.contains("_f0.")]
    for tool, lab in (("threadfin", "Threadfin"), ("clone2vec", "clone2vec"), ("clonotrace", "Clonotrace"), ("benisse", "Benisse")):
        groups.append((f"core-count and data-size runs, {lab}", res[res["tool_run"].str.startswith(tool)]))
    rows = []
    for lab, g in groups:
        if g.empty:
            continue
        kinds = g.groupby("node_kind").size()
        rows.append({"runs": lab, "number": len(g),
                     "nodes": "; ".join(f"{k}: {v}" for k, v in kinds.items()),
                     "node names": ", ".join(sorted(g["node"].dropna().unique()))})
    return md(pd.DataFrame(rows))


def budget_table():
    b = read("compute_budget.csv")
    if b is None:
        return None
    rows = []
    for tool, g in b.groupby("tool"):
        done = g[g["status"].eq("completed")]
        rows.append({"tool": tool, "runs with a time record": len(g), "completed": len(done),
                     "wall-hours, completed runs": f"{done['wall_hours'].sum():,.1f}",
                     "core-hours, completed runs": f"{done['core_hours'].sum():,.1f}",
                     "wall-hours, all runs (incl. failed, stopped, replaced)": f"{g['wall_hours'].sum():,.1f}",
                     "core-hours, all runs": f"{g['core_hours'].sum():,.1f}",
                     "longest run": fmt_seconds(3600 * g["wall_hours"].max())})
    return md(pd.DataFrame(rows))


# --------------------------------------------------------------------------- legends

def legends():
    """Figure legends with the numbers behind each figure (n, test, interval)."""
    pw, ov = read("annotation_pairwise.csv"), read("overview.csv")
    out = {}
    if ov is not None:
        n_tested = int(ov["p"].notna().sum())
        out["legend_fig1"] = (
            f"**Fig. 1 | The benchmark at a glance.** **a**, Each row is a tool at its own default and forced down to clones "
            "of two cells, Clonotrace with the exact optimal-transport distance (runs of 2026-10-25, section 6.11), three "
            "simple baselines, and a negative control: Threadfin against itself with another random seed. Each column "
            "is one question; the rule behind each verdict is in the table below. Every tile compares the tool with "
            "Threadfin on the same clones: green ▲ better, grey no clear difference, magenta ▼ worse; a dashed outline "
            "marks a comparison with too few datasets or simulation seeds for a two-sided paired test to reach P < 0.05 "
            "(shown descriptively). The numbers are the tool's and, after \"vs\", Threadfin's. Tests: two-sided sign "
            "tests over annotations, Wilcoxon signed-rank tests over datasets or simulation seeds, and a 95 % paired "
            "bootstrap interval over day-4 clones (10,000 resamples) for LARRY linking; P < 0.05 with no correction for "
            f"the {n_tested} tiles that come from a test (sensitivity to α below); n of each tile in results/overview.csv. "
            "The last column counts the tiles better and worse than Threadfin; the columns are not independent, so it "
            "summarises a row and is not a score. **b**, Usability checklist: the authors' assessment, with the evidence "
            "for every score in scripts/usability_checklist.csv.")
    out["legend_fig2"] = (
        "**Fig. 2 | Coverage.** Clones with ≥ 2 cells (**a**) and of exactly 2 cells (**b**) that each tool represents, "
        "per dataset (row label: the dataset's number of such clones); cell: number represented and share. Hatched: no "
        "result, with the reason. Benisse was run at ≥ 10 cells on GSE195673.")
    if pw is not None:
        def n_ann(ms, rna=True):
            s = pw[pw["method"].isin(ms)]
            s = s if rna else s[s["tag"] != "rna"]
            return s.groupby("method").size()

        two = pw[pw["method"].isin(["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca", "composition"])]
        two = two.dropna(subset=["r2_two_cell_method"])
        nn = two.groupby("method").size()
        top = int(nn.max())
        others = ", ".join(f"{name(k).split(' (')[0]} {int(v)}" for k, v in nn.items() if v != top)
        out["legend_fig3"] = (
            "**Fig. 3 | Two-cell clones carry information.** R² of each annotation computed on the two-cell clones only: "
            "tool (y) against Threadfin (x) on the same clones; one dot per annotation with at least 20 two-cell clones "
            f"(n = {top} per panel" + (f"; {others}" if others else "") + "), including the RNA-derived annotations "
            "(medians over the measured annotations alone are in the table below). Predictions are the mean annotation of "
            "each clone's 10 nearest clones among all clones of the comparison, after centring within donor. Bars: 95 % "
            "bootstrap percentile intervals (10,000 resamples of clones; R² of the two-cell clones of each resample). "
            "Filled: the 95 % paired bootstrap interval of the difference excludes 0. Text: annotations with the higher "
            "point estimate.")
        ctrl = pw[pw["method"].isin(["threadfin_kernel_seed1", "threadfin_kernel_seed2"]) & (pw["tag"] != "rna")]
        k = int(ctrl["call"].isin(["better", "worse"]).sum())
        common = (
            "Dot: R² of the tool minus R² of Threadfin on the clones the tool represents (k-nearest-neighbour regression "
            "with k = 10, annotations centred within donor); line: 95 % paired bootstrap percentile interval (10,000 "
            "resamples of clones, the same resamples for every method on the same clones; arrow: continues beyond the "
            "axis); filled: the interval excludes 0; diamond: an interval end within the Monte-Carlo error of 0 "
            "(unresolved). Rows: annotations that no tool used; \"(RNA)\" marks annotations derived from expression, which "
            "are easy by construction. Counts under each panel: annotations better, worse and in all.")
        out["legend_fig4"] = (
            "**Fig. 4 | Recovering measured clone annotations, every tool forced to clones of ≥ 2 cells.** " + common
            + " Last panel, a negative control: Threadfin with random-feature seed 1 (upper line of each row) and seed 2 "
            "(lower) against the default seed, on the same clones; every call there is false. Over the measured "
            f"annotations {k} of {len(ctrl)} control comparisons get a call ({ci_text(k, len(ctrl))}, Wilson interval), "
            "against 5 % expected from a 95 % interval: the intervals hold each representation fixed and do not include "
            "the variation from Threadfin's random features (section 6.10).")
        out["legend_fig5"] = (
            "**Fig. 5 | Recovering measured clone annotations, every tool at its own default.** clone2vec keeps clones of "
            "≥ 3 cells, Clonotrace ≥ 10; Benisse was run at ≥ 10 cells on GSE195673 only. " + common
            + " Last two panels: the seed control restricted to the clones of ≥ 3 and ≥ 10 cells.")
        out["legend_fig11"] = (
            "**Fig. 11 | Which parts of Threadfin matter?** Each variant switches off one step of the default profile: the "
            "kernel mean embedding (mean profile), centring on the donor, neighbour smoothing, shrinkage; \"mean of the "
            "embedding\" switches off all of them. " + common + " Last panel: the seed control of Fig. 4.")
    dis = read("annotation_recovery_disjoint.csv")
    if dis is not None:
        mdj = matched_disjoint(dis)
        out["legend_fig6"] = (
            "**Fig. 6 | Annotation of one half of each clone's cells predicted from a representation of the other half.** "
            "R² of the tool minus R² of Threadfin on the same halves and clones (the same split, direction and set of "
            "clones), averaged over 3 random splits × 2 directions (Clonotrace and Benisse: split 0 only). Dots: "
            f"annotations ({mdj.groupby('method').size().min()}–{mdj.groupby('method').size().max()} per method, "
            "including the RNA-derived ones); diamond: median; right: annotations higher / lower than Threadfin. Clones "
            "need ≥ 2 cells in each half. The tools come from the comparison of every tool (forced to ≥ 2 cells), "
            "Threadfin's variants from the ablation comparison and the control (seed 1) from the sensitivity comparison, "
            "each against Threadfin in the same comparison.")
    rep = read("split_half_reproducibility.csv")
    if rep is not None:
        mr = matched_reproducibility(rep)
        out["legend_fig7"] = (
            "**Fig. 7 | The same clones represented from two disjoint halves of their cells.** **a**, Spearman correlation "
            "of all clone–clone distances between the halves; **b**, overlap of the 10 nearest clones (dashed: chance); "
            "**c**, share of clones whose representation from one half is nearest to their own other half "
            "(re-identification; ties counted as mid-ranks; dashed: chance). Dots: datasets "
            f"({mr.groupby('method').size().min()}–{mr.groupby('method').size().max()} per method; mean over 3 splits, "
            "Clonotrace and Benisse split 0 only); diamond: median over datasets; black tick: Threadfin's median on the "
            "same halves and clones. Clones need ≥ 2 cells in each half; at most 4,000 clones and 300,000 clone pairs "
            "per dataset (the same subsample for every method).")
    lk = read("larry_clone_linking.csv")
    if lk is not None:
        f = lk[lk["set"].str.startswith("forced")].iloc[0]
        n = lk[lk["set"].str.startswith("native")].iloc[0]
        out["legend_fig8"] = (
            "**Fig. 8 | Linking day-4 LARRY clones to the same barcode's day-6 clone.** Each day-4 clone's own day-6 "
            "clone is ranked among all day-6 clones by the distance between representations (ties counted as mid-ranks). "
            f"**a**, **b**: clones of ≥ 2 cells kept by every tool ({int(f['n_query_clones']):,} day-4 queries, "
            f"{int(f['n_candidate_day6_clones']):,} day-6 candidates); **c**, **d**: clones of ≥ 10 cells "
            f"({int(n['n_query_clones']):,} queries, {int(n['n_candidate_day6_clones']):,} candidates). **a**, **c**: mean "
            "reciprocal rank (dashed: chance); **b**, **d**: difference to Threadfin with its 95 % paired bootstrap "
            "interval over day-4 queries (10,000 resamples); filled: the interval excludes 0.")
    files = sorted(glob.glob(str(RESULTS / "tests" / "*.csv")))
    pf = sorted(glob.glob(str(RESULTS / "tests" / "power" / "*.csv")))
    if files:
        reps = pd.read_csv(pf[0]).groupby("fraction_real_labels").size().iloc[0] if pf else 10
        out["legend_fig9"] = (
            "**Fig. 9 | Are tests of \"cells of one clone share a state\" calibrated?** Real data, clone labels shuffled. "
            "**a**, **b**: share of 50 null replicates per dataset in which each test reports clonal structure, with "
            "Wilson 95 % intervals (dashed: 5 %). Threadfin: clonal coherence test, 99 within-sample permutations, "
            "P < 0.05; scirpy: clonotype modularity, at least one clone at FDR < 5 %; ANOVA: cells as replicates, "
            "Bonferroni over the first 10 embedding dimensions. **c**, power of Threadfin's test when 10, 25 or 50 % of "
            f"cells keep their real clone label and the rest are shuffled within sample ({reps} replicates per fraction; "
            "0 %: the 50 within-sample null replicates), with Wilson 95 % intervals under each value; right: P of the "
            "test on the real labels.")
    sim = read("simulation.csv")
    if sim is not None:
        fr = simulation_frame(sim)
        ns = fr.groupby("tool")["seed"].nunique()
        out["legend_fig10"] = (
            "**Fig. 10 | Simulated repertoires with a known answer.** 600 clones and 8,000 unlabelled cells per dataset, "
            "six donors; rows of panels: clone sizes and donor shift. Small dots: simulated datasets (one per seed); large "
            f"dot: mean over seeds. Every tool except Clonotrace: {int(ns.drop('clonotrace', errors='ignore').min())} "
            f"seeds; Clonotrace: {int(ns.get('clonotrace', 0))} seeds (its runs take hours per large simulation). Columns: "
            "share of each clone's 10 nearest clones that follow the same planted programme (dashed: chance); AUROC "
            "for telling mixture from intermediate clones by the share of mixture clones among the 10 nearest (clones "
            "≥ 10 cells; dashed: chance); share of same-donor neighbours minus its expectation (dashed: none). The "
            "seed-level tests against Threadfin are in the table below.")
    out["legend_fig12"] = (
        "**Fig. 12 | Computing resources of each tool relative to Threadfin** (log scale; 1 = the same). Top blocks: "
        "time and peak memory of the clone representation on each dataset (4 cores; Clonotrace forced to ≥ 2 cells used "
        "6, 8 and 32 R sessions for the clone distances on LARRY days 4/6, Plasmodium d10-42 and GSE195673). Lower "
        "blocks: time with 1–16 cores (GSE287123 RBD; Threadfin: its complete analysis, the other tools: their clone "
        "representation) and on 5–50 % subsets of GSE195673 (8 cores). Hollow: Clonotrace runs whose clone distances "
        "used one R session (slower than its parallel plan). In the two lower blocks the default and forced panels of a "
        "tool show the same runs (each program was run once there, on clones of ≥ 2 cells). Triangles: Clonotrace "
        "memory of the main R session only, a "
        "lower bound (its parallel sessions are not measured by /usr/bin/time; the job's peak from the scheduler is "
        "used where the run had a job of its own). Counts under each panel: differences of more than 25 %. Hardware "
        "(results/run_hosts.csv): the per-dataset runs ran on 128-core Sapphire Rapids nodes, except Clonotrace forced "
        "to ≥ 2 cells on GSE195673 (a 72-core node of the older cascade generation); in the core-count and data-size "
        "blocks Threadfin ran on a Sapphire Rapids node and clone2vec, Clonotrace and Benisse on 72-core cascade nodes "
        "(Benisse's 50 % subset on a Sapphire Rapids node), so cross-tool times there mix CPU generations. Nodes were "
        "shared with other jobs; times are indicative to within about a factor of two.")
    out["legend_fig13"] = (
        "**Fig. 13 | Scaling.** **a**, speed-up with cores on GSE287123 RBD (1,887 clones; each configuration alone in a "
        "job holding 16 cores, one after another). **b**, **c**, time and peak memory on 5, 10, 25 and 50 % subsets of "
        "GSE195673 (8 cores). Threadfin's runs were on a Sapphire Rapids node, the other tools' on 72-core cascade nodes "
        "(Benisse's 50 % subset on a Sapphire Rapids node); Clonotrace's memory is its main R session (a lower bound; "
        "job peaks for its two largest subsets in the table). Growth exponents are fits to 4 subsets.")
    if pw is not None:
        sens = pw[pw["method"].str.match(r"threadfin_kernel_(rff|bw|smooth|pc|seed)\d") & ~pw["method"].str.contains("_ge")
                  & (pw["tag"] != "rna")]
        out["legend_fig14"] = (
            "**Fig. 14 | Sensitivity of Threadfin to its settings** (one setting changed at a time). **a**, R² of the "
            f"setting minus the default per measured annotation (n = {sens.groupby('method').size().max()} per setting); "
            "dark: the 95 % paired bootstrap interval (10,000 resamples) excludes 0; bar: median; right: annotations "
            "better ▲ and worse ▼. The two seed rows are the negative control of Figs. 4, 5 and 11. **b**, "
            "re-identification across halves (percentage points; one dot per dataset, mean over 3 splits). **c**, **d**, "
            "simulation: one dot per seed (mean over the seed's simulated datasets), programme purity and the mixture "
            "vs intermediate AUROC (clones ≥ 10 cells).")
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
    tables = {"status": status_block(), "datasets": datasets_table(), "coverage": coverage_table(), **pairwise_tables(),
              "exact": exact_table(), "tests": tests_table(), **simulation_tables(), "runtime": runtime_table(),
              "clonotrace_stages": clonotrace_table(), **halves_tables(), "larry": larry_table(), **overview_tables(),
              **resource_tables(), **legends()}
    write(tables)


if __name__ == "__main__":
    main()
