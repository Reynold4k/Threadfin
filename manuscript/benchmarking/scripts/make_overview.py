#!/usr/bin/env python
"""Figure 0 of the benchmark: every tool compared with Threadfin at a glance, and how easy each tool is to use.

a  One column per question of the benchmark, in plain words, and one row per tool. A tile says whether the
   tool did better than Threadfin, about the same, or worse, and gives the number behind the verdict;
   Threadfin's own row gives its value. Failed and stopped runs are counted, not dropped. The rule for each
   column is in RULES (also printed in the README).
b  Usability checklist (scripts/usability_checklist.csv): 13 criteria in five groups, each scored 1 (yes),
   0.5 (partly) or 0 (no), with the evidence for every score in the same file. The two scores that this
   benchmark measured itself (same result with any number of cores) come from results/resources_reproducibility.csv.

Everything in panel a is computed from the tables in results/. Writes figures/fig0_overview.png and
results/overview.csv. Usage: python make_overview.py
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

from bench_paths import HERE, RESULTS  # noqa: E402
from make_figures import INK, INK2, MUTED, ORDER, SURFACE, save  # noqa: E402

BETTER, SIMILAR, WORSE = "#2b6cb5", "#e4e3df", "#c2511f"       # white text on BETTER and WORSE
TF = "threadfin_kernel"
ALPHA, TOL = 0.05, 0.25           # paired-test level; tolerance for single measurements (25 %)
STOP_SECONDS = 72 * 3600

ROWS = [  # label, key, kind
    ("Threadfin", TF, "ref"),
    ("clone2vec, its default (clones ≥ 3 cells)", "clone2vec_min3", "tool"),
    ("clone2vec, forced to clones ≥ 2 cells", "clone2vec_min2", "tool"),
    ("Clonotrace, its default (clones ≥ 10 cells)", "clonotrace_revised_min10", "tool"),
    ("Clonotrace, forced to clones ≥ 2 cells", "clonotrace_revised_min2", "tool"),
    ("Benisse", "benisse", "tool"),
    ("CoNGA's one-cell-per-clone rule", "conga_medoid", "baseline"),
    ("Mean of the embedding (baseline)", "mean_pca", "baseline"),
    ("Cluster composition (baseline)", "composition", "baseline"),
]
COLUMNS = [  # key, group, label
    ("coverage", "Which clones it can use", "Share of\nclones it\ndescribes"),
    ("two_cell", "Which clones it can use", "Keeps\ntwo-cell\nclones"),
    ("datasets", "Which clones it can use", "Datasets\nit could\nanalyse"),
    ("traits", "How well it describes clones", "Predicts\nmeasured\nclone traits"),
    ("traits_disjoint", "How well it describes clones", "… when traits\ncome from\nother cells"),
    ("reid", "How well it describes clones", "Recognises\na clone from\nother cells"),
    ("fate", "How well it describes clones", "Links day-2\nclones to\nday 6"),
    ("sim_groups", "How well it describes clones", "Finds\nplanted clone\ngroups"),
    ("sim_mixed", "How well it describes clones", "Tells mixed\nfrom in-between\nclones"),
    ("time", "Computer resources", "Run\ntime"),
    ("memory", "Computer resources", "Peak\nmemory"),
    ("cores", "Computer resources", "Gain from\n16 CPU\ncores"),
    ("growth", "Computer resources", "Slow-down\nas data\ngrow"),
    ("finished", "Computer resources", "Runs that\nended with-\nout help"),
    ("tally", "Summary", "Tally of\ncolumns"),
]
RULES = {
    "coverage": "median over the 11 datasets of the share of clones with ≥ 2 cells that the tool represents; a run "
                "that failed, stopped or could not start counts as 0. Better or worse: more than 5 percentage points "
                "from Threadfin.",
    "two_cell": "the same for clones of exactly two cells.",
    "datasets": "datasets (of 11) on which the tool produced a result.",
    "traits": "measured annotations that no tool used (sorting gates, isotype, mutation, tissue, time, later fate; "
              "RNA-derived calls excluded), each scored on the clones the tool represents: the tool is better (worse) "
              "on an annotation when the 95 % paired bootstrap interval of R² − R²(Threadfin) excludes 0. Column "
              "verdict: two-sided sign test of better vs worse annotations, P < 0.05.",
    "traits_disjoint": "as above, but the representation and the annotation come from disjoint halves of each "
                       "clone's cells (3 random splits, both directions); sign test over annotations.",
    "reid": "share of clones whose representation from one half of their cells is nearest to their own other half "
            "(median over datasets); Wilcoxon signed-rank test over datasets.",
    "fate": "LARRY: mean reciprocal rank of each day-4 clone's own day-6 clone among all day-6 clones; 95 % paired "
            "bootstrap interval over clones.",
    "sim_groups": "simulated data with a known answer: share of clones whose nearest clone has the same planted "
                  "programme; Wilcoxon over simulated datasets.",
    "sim_mixed": "AUROC for telling clones made of two kinds of cells from clones of cells halfway between them "
                 "(clones ≥ 10 cells); Wilcoxon over simulated datasets.",
    "time": "time to compute the clone representation, divided by Threadfin's on the same dataset (median); "
            "Clonotrace: only runs whose clone distances used several R sessions, a run stopped at 72 h counted as "
            "72 h. Better or worse: every ratio beyond 25 % in the same direction, otherwise a Wilcoxon test on "
            "the log ratios (≥ 6 datasets).",
    "memory": "peak memory, the same way.",
    "cores": "time with 1 core divided by time with 16 cores (GSE287123 RBD; Threadfin: its complete analysis). "
             "Better or worse: more than 25 % from Threadfin's gain.",
    "growth": "how much longer the clone representation takes for 10 times more clones, from 5-50 % subsets of "
              "GSE195673 (8 cores). Better or worse: an exponent more than 0.25 from Threadfin's.",
    "finished": "runs on the full data and on the disjoint halves that finished with the tool's default settings; "
                "failed or stopped runs, and runs that needed a larger cell graph, are counted against the tool.",
}


@dataclass
class Cell:
    verdict: str                  # ref, better, similar, worse, na, pending
    text: str = ""
    detail: str = ""
    n: int | None = None
    p: float | None = None


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def pct(v):
    return f"{100 * v:.0f} %" if v >= 0.095 or v == 0 else f"{100 * v:.1f} %"


def sign_test(n_up, n_down):
    n = n_up + n_down
    if n == 0:
        return "similar", None
    p = float(stats.binomtest(n_up, n).pvalue)
    return ("better" if n_up > n_down else "worse") if p < ALPHA else "similar", p


def paired(diffs, higher_better=True):
    """Verdict from paired differences (tool - Threadfin) over datasets or annotations."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if not higher_better:
        d = -d
    if d.size == 0:
        return "na", None
    if np.allclose(d, 0):
        return "similar", 1.0
    if d.size < 5:
        return "similar", None
    p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    return ("better" if (d > 0).sum() > (d < 0).sum() else "worse") if p < ALPHA else "similar", p


def ratio_verdict(ratios, lower_better=True):
    """Verdict from tool / Threadfin ratios over datasets."""
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return "na", None
    lr = np.log(r) * (-1 if lower_better else 1)
    if (lr > np.log(1 + TOL)).all():          # beyond 25 % on every dataset
        return "better", None
    if (lr < -np.log(1 + TOL)).all():
        return "worse", None
    if r.size >= 6:
        p = float(stats.wilcoxon(lr).pvalue)
        if p < ALPHA:
            return ("better" if np.median(lr) > 0 else "worse"), p
        return "similar", p
    return "similar", None


def times_text(r, more="slower", less="faster"):
    if not np.isfinite(r):
        return "?"
    if r >= 1 + TOL:
        return f"{r:,.0f}× {more}" if r >= 10 else f"{r:.1f}× {more}"
    if r <= 1 / (1 + TOL):
        return f"{1 / r:,.0f}× {less}" if 1 / r >= 10 else f"{1 / r:.1f}× {less}"
    return "about the same"


# --------------------------------------------------------------------------- columns

class Overview:
    def __init__(self):
        self.cov = read("coverage_runtime.csv")
        self.cov = self.cov[self.cov["dataset"].isin(ORDER)]
        self.pw = read("annotation_pairwise.csv")
        self.dis = read("annotation_recovery_disjoint.csv")
        self.rep = read("split_half_reproducibility.csv")
        self.larry = read("larry_clone_linking.csv")
        self.sim = read("simulation.csv")
        self.res = read("resources.csv")
        self.status = read("run_status.csv")

    # helpers -----------------------------------------------------------------
    def runs(self, key):
        """One coverage row per dataset for a row of the figure (Benisse: >= 2 cells, else >= 10 cells)."""
        out = []
        for ds in ORDER:
            sub = self.cov[self.cov["dataset"] == ds].set_index("method")
            if sub.empty:
                continue
            if key == "benisse":
                r = sub.loc["benisse_min2"]
                if r["status"] != "completed" and sub.loc["benisse_min10", "status"] == "completed":
                    r = sub.loc["benisse_min10"]
                r = r.copy()
                r["method_used"] = r.name
            else:
                r = sub.loc[key].copy()
                r["method_used"] = key
            out.append(r)
        return pd.DataFrame(out)

    @staticmethod
    def native(key):
        return key in ("clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10")

    def method_names(self, key):
        return ["benisse_min2", "benisse_min10"] if key == "benisse" else [key]

    # which clones -------------------------------------------------------------
    def coverage(self, key, two=False):
        r = self.runs(key)
        num, den = ("n_two_cell_clones_represented", "n_two_cell_clones") if two else ("n_clones_represented", "n_clones_ge2")
        r = r[~r["reason"].astype(str).str.contains("receptor")]      # Benisse has nothing to read in LARRY
        r = r[r[den] > 0]
        v = np.where(r["status"].eq("completed"), r[num] / r[den], 0.0)
        if key == TF:
            return Cell("ref", pct(np.median(v)), n=len(v))
        med = float(np.median(v))
        verdict = "worse" if med < 1 - 0.05 else "better" if med > 1 + 0.05 else "similar"
        return Cell(verdict, pct(med), f"per dataset: {', '.join(pct(x) for x in v)}", n=len(v))

    def datasets(self, key):
        r = self.runs(key)
        ok = r["status"].eq("completed")
        n = int(ok.sum())
        if key == TF:
            return Cell("ref", f"{n} of {len(r)}", n=len(r))
        why = []
        reasons = r.loc[~ok, "reason"].astype(str)
        if reasons.str.contains("fewer than 3").any():
            why.append(f"{int(reasons.str.contains('fewer than 3').sum())} too small")
        if reasons.str.contains("receptor").any():
            why.append("2 without BCR")
        if r.loc[~ok, "status"].eq("stopped").any():
            why.append("1 stopped")
        if r.loc[~ok, "status"].eq("failed").any():
            why.append(f"{int(r.loc[~ok, 'status'].eq('failed').sum())} failed")
        text = f"{n} of {len(r)}" + ("\n" + why[0] if why else "")
        return Cell("worse" if n < len(r) else "similar", text, "; ".join(why), n=len(r))

    # how well ----------------------------------------------------------------
    def traits(self, key):
        if key == TF:
            return Cell("ref", "reference")
        s = self.pw[self.pw["method"].isin(self.method_names(key)) & (self.pw["tag"] != "rna")]
        if s.empty:
            return Cell("na", "—")
        up, down = int((s["delta_ci_low"] > 0).sum()), int((s["delta_ci_high"] < 0).sum())
        v, p = sign_test(up, down)
        return Cell(v, f"{up} better\n{down} worse", f"{up} better, {down} worse, {len(s)} annotations", len(s), p)

    def _set(self, frame, key, forced, native):
        return frame[frame["set"].eq(native if self.native(key) else forced)]

    def traits_disjoint(self, key):
        if key == TF:
            return Cell("ref", "reference")
        d = self._set(self.dis, key, "forced: >= 2 cells per half", "native: >= 10 cells per half")
        d = d[d["tag"] != "rna"]
        m = d.groupby(["dataset", "target", "method"])["r2"].mean().unstack("method")
        cols = [c for c in self.method_names(key) if c in m]
        if not cols or TF not in m:
            return Cell("na", "—")
        tool = m[cols].bfill(axis=1).iloc[:, 0]
        diff = (tool - m[TF]).dropna()
        up, down = int((diff > 0).sum()), int((diff < 0).sum())
        v, p = sign_test(up, down)
        return Cell(v, f"{up} better\n{down} worse", f"{len(diff)} annotations", len(diff), p)

    def reid(self, key):
        r = self._set(self.rep, key, "forced: >= 2 cells per half", "native: >= 10 cells per half")
        m = r.groupby(["dataset", "method"])["reid_top1"].mean().unstack("method")
        if key == TF:
            tf = self.rep[self.rep["set"].eq("forced: >= 2 cells per half") & self.rep["method"].eq(TF)]
            return Cell("ref", pct(tf.groupby("dataset")["reid_top1"].mean().median()))
        cols = [c for c in self.method_names(key) if c in m]
        if not cols:
            return Cell("na", "—")
        tool = m[cols].bfill(axis=1).iloc[:, 0]
        both = pd.concat([tool, m[TF]], axis=1).dropna()
        v, p = paired(both.iloc[:, 0] - both.iloc[:, 1])
        return Cell(v, f"{pct(both.iloc[:, 0].median())}\nvs {pct(both.iloc[:, 1].median())}",
                    f"{len(both)} datasets", len(both), p)

    def fate(self, key):
        lk = self.larry[self.larry["set"].eq("native: clones >= 10 cells" if self.native(key) else "forced: clones >= 2 cells")]
        if key == TF:
            r = self.larry[self.larry["set"].eq("forced: clones >= 2 cells") & self.larry["method"].eq(TF)]
            return Cell("ref", f"{r['mrr'].iloc[0]:.3f}" if len(r) else "—")
        r = lk[lk["method"].isin(self.method_names(key))]
        tf = lk[lk["method"].eq(TF)]
        if r.empty or tf.empty:
            return Cell("na", "—")
        r = r.iloc[0]
        lo, hi = r.get("mrr_delta_ci_low", np.nan), r.get("mrr_delta_ci_high", np.nan)
        v = "better" if lo > 0 else "worse" if hi < 0 else "similar"
        return Cell(v, f"{r['mrr']:.3f}\nvs {tf['mrr'].iloc[0]:.3f}", f"95 % CI of difference [{lo:.3f}, {hi:.3f}]",
                    int(r["n_query_clones"]))

    def simulation(self, key, what):
        s = self.sim
        if what == "mixture_vs_intermediate_auroc":
            s = s[s["regime"].eq("large")]
        if self.native(key):
            s = s[s["set"].eq("native: clones >= 10 cells")]
        else:
            s = s[s["set"].eq("forced: clones >= 2 cells")]
        m = s.pivot_table(index="dataset", columns="method", values=what, aggfunc="first")
        if key == TF:
            base = self.sim[self.sim["set"].eq("forced: clones >= 2 cells") & self.sim["method"].eq(TF)]
            if what == "mixture_vs_intermediate_auroc":
                base = base[base["regime"].eq("large")]
            return Cell("ref", f"{base[what].median():.2f}")
        cols = [c for c in self.method_names(key) if c in m]
        if not cols or TF not in m:
            return Cell("na", "—")
        both = pd.concat([m[cols[0]], m[TF]], axis=1).dropna()
        v, p = paired(both.iloc[:, 0] - both.iloc[:, 1])
        return Cell(v, f"{both.iloc[:, 0].median():.2f}\nvs {both.iloc[:, 1].median():.2f}", f"{len(both)} simulated datasets",
                    len(both), p)

    # resources -----------------------------------------------------------------
    def _usable_runs(self, key):
        r = self.runs(key)
        tf = self.runs(TF).set_index("dataset")
        keep = r["status"].eq("completed")
        if key.startswith("clonotrace"):
            keep &= r["distance_workers"].fillna(1) > 1
            keep |= r["status"].eq("stopped")
        return r[keep], tf

    def time(self, key):
        if key == TF:
            t = self.runs(TF)["seconds_method"]
            return Cell("ref", f"{t.median():.1f} s\n(median)")
        if key in ("conga_medoid", "mean_pca", "composition"):
            return Cell("na", "not timed")
        r, tf = self._usable_runs(key)
        sec = np.where(r["status"].eq("stopped"), STOP_SECONDS,
                       r["seconds_method"].fillna(r["wall_seconds_task"]).to_numpy(float))
        ratios = sec / tf.loc[r["dataset"], "seconds_method"].to_numpy(float)
        v, p = ratio_verdict(ratios)
        stopped = int(r["status"].eq("stopped").sum())
        text = times_text(float(np.median(ratios))).replace(" ", "\n", 1)
        if stopped:
            text = "≥ " + text.replace("\n", "\n", 1) + f", {stopped} stopped"
        return Cell(v, text, f"{len(ratios)} datasets", len(ratios), p)

    def memory(self, key):
        if key == TF:
            return Cell("ref", f"{self.runs(TF)['peak_rss_gb_method'].median():.1f} GB\n(median)")
        if key in ("conga_medoid", "mean_pca", "composition"):
            return Cell("na", "not measured")
        r, tf = self._usable_runs(key)
        r = r[r["status"].eq("completed")]
        ratios = r["peak_rss_gb_task"].to_numpy(float) / tf.loc[r["dataset"], "peak_rss_gb_method"].to_numpy(float)
        v, p = ratio_verdict(ratios)
        return Cell(v, times_text(float(np.median(ratios)), "more", "less"), f"{len(ratios)} datasets", len(ratios), p)

    def cores(self, key):
        res = self.res
        if res is None:
            return Cell("pending", "…")
        c = res[res["experiment"].eq("cores") & res["status"].eq("completed")]
        tool = {TF: "threadfin_run", "clone2vec_min3": "clone2vec_min2", "clone2vec_min2": "clone2vec_min2",
                "clonotrace_revised_min10": "clonotrace_revised_min2", "clonotrace_revised_min2": "clonotrace_revised_min2",
                "benisse": "benisse_min2"}.get(key)
        if tool is None:
            return Cell("na", "—")

        def speedup(t):
            s = c[c["tool"].eq(t)].set_index("cores")["seconds"]
            return (float(s[1] / s[16]), "1→16") if {1, 16} <= set(s.index) else (np.nan, "")

        s_tf, _ = speedup("threadfin_run")
        s, _ = speedup(tool)
        if not np.isfinite(s):
            return Cell("pending", "running")
        text = f"{s:.1f}× faster" if s >= 1 else f"{1 / s:.1f}× slower"
        if key == TF:
            return Cell("ref", text)
        text += f"\nThreadfin {s_tf:.1f}×"
        v = "better" if s > s_tf * (1 + TOL) else "worse" if s < s_tf / (1 + TOL) else "similar"
        return Cell(v, text, f"Threadfin {s_tf:.1f}×")

    def growth(self, key):
        res = self.res
        if res is None:
            return Cell("pending", "…")
        g = res[res["experiment"].eq("size") & res["status"].eq("completed")]
        tool = {TF: "threadfin_kernel", "clone2vec_min3": "clone2vec_min2", "clone2vec_min2": "clone2vec_min2",
                "clonotrace_revised_min10": "clonotrace_revised_min2", "clonotrace_revised_min2": "clonotrace_revised_min2",
                "benisse": "benisse_min2"}.get(key)
        if tool is None:
            return Cell("na", "—")

        def exponent(t):
            s = g[g["tool"].eq(t)]
            if len(s) < 3:
                return np.nan
            return float(np.polyfit(np.log(s["n_clones_ge2"]), np.log(s["seconds"]), 1)[0])

        b_tf, b = exponent("threadfin_kernel"), exponent(tool)
        if not np.isfinite(b):
            return Cell("pending", "running")
        text = f"{10 ** b:,.0f}× time\nfor 10× clones"
        if key == TF:
            return Cell("ref", text)
        v = "better" if b < b_tf - TOL else "worse" if b > b_tf + TOL else "similar"
        return Cell(v, text, f"exponent {b:.2f} vs {b_tf:.2f}")

    def finished(self, key):
        st = self.status
        if st is None:
            return Cell("na", "—")
        runs = {"clonotrace_revised_min10": ["clonotrace_min10"], "clonotrace_revised_min2": ["clonotrace_min2"],
                "benisse": ["benisse_min2", "benisse_min10"]}.get(key, [key])
        s = st[st["tool_run"].isin(runs)]
        if s.empty:
            return Cell("na", "—")
        big_graph = s["status"].eq("completed") & (s["graph_k_used"].fillna(10) > 10)
        unaided = s["status"].eq("completed") & ~big_graph
        n, u = len(s), int(unaided.sum())
        notes = []
        for what, mask in (("stopped", s["status"].eq("stopped")), ("failed", s["status"].eq("failed")),
                           ("re-run", big_graph)):
            if mask.any():
                notes.append(f"{int(mask.sum())} {what}")
        text = f"{u} of {n}" + ("\n" + ", ".join(notes[:2]) if notes else "")
        if key == TF:
            return Cell("ref", text)
        return Cell("worse" if u < n else "similar", text, "; ".join(notes), n)

    def cell(self, key, col):
        f = {"coverage": lambda: self.coverage(key), "two_cell": lambda: self.coverage(key, two=True),
             "datasets": lambda: self.datasets(key), "traits": lambda: self.traits(key),
             "traits_disjoint": lambda: self.traits_disjoint(key), "reid": lambda: self.reid(key),
             "fate": lambda: self.fate(key), "sim_groups": lambda: self.simulation(key, "programme_purity"),
             "sim_mixed": lambda: self.simulation(key, "mixture_vs_intermediate_auroc"),
             "time": lambda: self.time(key),
             "memory": lambda: self.memory(key), "cores": lambda: self.cores(key),
             "growth": lambda: self.growth(key), "finished": lambda: self.finished(key)}[col]
        if key == "benisse" and col in ("fate", "sim_groups", "sim_mixed"):
            return Cell("na", "no BCR")
        try:
            return f()
        except (KeyError, IndexError, ValueError) as e:      # a readout that this row does not have
            return Cell("na", "—", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- usability

def usability():
    u = pd.read_csv(HERE / "usability_checklist.csv")
    rep = read("resources_reproducibility.csv")
    if rep is not None:
        for tool, key in (("clone2vec", "clone2vec_min2"), ("Benisse", "benisse_min2")):
            r = rep[rep["tool"].eq(key)]
            if r.empty:
                continue
            same = bool(r["identical"].all())
            rho = float(r["distance_spearman"].min())
            mask = u["tool"].eq(tool) & u["criterion"].eq("Same result when re-run")
            u.loc[mask, "score"] = 1.0 if same else 0.5
            u.loc[mask, "evidence"] = ("identical with 1-16 cores (this benchmark)" if same else
                                       f"slightly different with 1-16 cores: clone distances correlate {rho:.4f} "
                                       f"(Spearman), max difference {r['max_abs_difference'].max():.3g} (this benchmark)")
    return u


# --------------------------------------------------------------------------- drawing

def draw(cells: dict, u: pd.DataFrame):
    groups = []
    xs, x = [], 0.0
    for i, (key, group, _) in enumerate(COLUMNS):
        if i and group != COLUMNS[i - 1][1]:
            x += 0.35
        xs.append(x)
        x += 1.0
    width_units = xs[-1] + 1
    n_rows = len(ROWS)
    tools_u = ["Threadfin", "Clonotrace", "clone2vec", "Benisse", "CoNGA", "scirpy"]
    crit = list(dict.fromkeys(u["criterion"]))
    fig = plt.figure(figsize=(16.5, 13.2))
    ax = fig.add_axes([0.205, 0.40, 0.785, 0.52])
    ax.set_xlim(-0.55, width_units - 0.45)
    ax.set_ylim(n_rows - 0.45, -1.9)
    ax.axis("off")
    # group bands and column labels
    for g in dict.fromkeys(c[1] for c in COLUMNS):
        idx = [i for i, c in enumerate(COLUMNS) if c[1] == g]
        x0, x1 = xs[idx[0]] - 0.47, xs[idx[-1]] + 0.47
        ax.add_patch(FancyBboxPatch((x0, -1.85), x1 - x0, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="#efeeea", edgecolor="none"))
        ax.text((x0 + x1) / 2, -1.64, g.replace("\n", " ") if len(idx) > 1 else g, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=INK)
    for x, (key, _, label) in zip(xs, COLUMNS):
        ax.text(x, -0.72, label, ha="center", va="center", fontsize=6.7, color=INK, linespacing=1.1)
    # rows
    for j, (label, key, kind) in enumerate(ROWS):
        if j % 2 == 0:
            ax.add_patch(Rectangle((-0.55, j - 0.5), width_units + 0.1, 1.0, facecolor="#f5f4f1", edgecolor="none",
                                   zorder=0))
        ax.text(-0.75, j, label, ha="right", va="center", fontsize=8.2,
                fontweight="bold" if kind == "ref" else "normal", color=INK if kind != "test" else INK2)
        for x, (col, _, _) in zip(xs, COLUMNS):
            c = cells[(key, col)]
            if col == "tally":
                ax.text(x, j, c.text, ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.15)
                continue
            if c.verdict in ("na", "pending"):
                ax.text(x, j, c.text or "—", ha="center", va="center", fontsize=6.6, color=MUTED)
                continue
            face, ink, edge = {"better": (BETTER, "white", BETTER), "worse": (WORSE, "white", WORSE),
                               "similar": (SIMILAR, INK, SIMILAR), "ref": ("white", INK, INK2)}[c.verdict]
            ax.add_patch(FancyBboxPatch((x - 0.45, j - 0.4), 0.9, 0.8, boxstyle="round,pad=0,rounding_size=0.1",
                                        facecolor=face, edgecolor=edge, lw=0.9, zorder=2))
            mark = {"better": "▲ ", "worse": "▼ ", "similar": "", "ref": ""}[c.verdict]
            lines = c.text.split("\n")
            bold = "bold" if c.verdict in ("better", "worse") else "normal"
            first = mark + lines[0]
            fs1 = max(5.0, min(6.6, 6.6 * 12.5 / len(first)))          # shrink long text to fit the tile
            if len(lines) == 1:
                ax.text(x, j, first, ha="center", va="center", fontsize=fs1, color=ink, zorder=3, fontweight=bold)
            else:
                fs2 = max(4.6, min(5.7, 5.7 * 16 / len(lines[1])))
                ax.text(x, j - 0.13, first, ha="center", va="center", fontsize=fs1, color=ink, zorder=3,
                        fontweight=bold)
                ax.text(x, j + 0.17, lines[1], ha="center", va="center", fontsize=fs2, color=ink, zorder=3)
    ax.text(-0.75, -1.64, "a", fontsize=14, fontweight="bold", ha="right", va="center")
    # legend of panel a
    lx = 0.205
    for i, (face, ink, lab) in enumerate([(BETTER, "white", "▲ better than Threadfin"),
                                          (SIMILAR, INK, "no clear difference"),
                                          (WORSE, "white", "▼ worse than Threadfin"),
                                          ("white", INK, "Threadfin's own value")]):
        fig.text(lx + i * 0.135, 0.378, f"  {lab}  ", fontsize=7.6, color=ink, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=INK2 if face == "white" else face, lw=0.8))
    fig.text(lx + 4 * 0.135, 0.378, "—  the tool has no such output        …  still running", fontsize=7.4, color=MUTED,
             va="center")
    fig.text(lx, 0.355, "Each tile compares the tool with Threadfin on the same clones and datasets. Columns with "
             "many annotations or datasets use paired tests (P < 0.05); single measurements count as different when "
             "they differ by more than 25 %. Exact rules: README, section 0.", fontsize=7.2, color=INK2, va="center")

    # panel b: usability checklist
    bx = fig.add_axes([0.205, 0.045, 0.785, 0.25])
    order = list(dict.fromkeys(u["group"]))
    cx, x, crit_x = {}, 0.0, []
    for i, cname in enumerate(crit):
        g = u.loc[u["criterion"].eq(cname), "group"].iloc[0]
        if i and g != u.loc[u["criterion"].eq(crit[i - 1]), "group"].iloc[0]:
            x += 0.35
        cx[cname] = x
        crit_x.append(x)
        x += 1.0
    total_x = x + 0.35
    bx.set_xlim(-0.55, total_x + 0.8)
    bx.set_ylim(len(tools_u) - 0.45, -2.0)
    bx.axis("off")
    for g in order:
        names = [c for c in crit if u.loc[u["criterion"].eq(c), "group"].iloc[0] == g]
        x0, x1 = cx[names[0]] - 0.47, cx[names[-1]] + 0.47
        bx.add_patch(FancyBboxPatch((x0, -1.95), x1 - x0, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="#efeeea", edgecolor="none"))
        bx.text((x0 + x1) / 2, -1.74, g, ha="center", va="center", fontsize=7.6, fontweight="bold")
    for cname in crit:
        bx.text(cx[cname], -0.78, textwrap.fill(cname, 15), ha="center", va="center", fontsize=6.8, linespacing=1.1)
    bx.text(total_x, -0.78, "Score", ha="center", va="center", fontsize=7.2, fontweight="bold")
    shade = {1.0: "#55544f", 0.5: "#b7b6b0", 0.0: "white"}
    for j, tool in enumerate(tools_u):
        if j % 2 == 0:
            bx.add_patch(Rectangle((-0.55, j - 0.5), total_x + 1.35, 1.0, facecolor="#f5f4f1", edgecolor="none",
                                   zorder=0))
        bx.text(-0.75, j, tool, ha="right", va="center", fontsize=8.2, fontweight="bold" if tool == "Threadfin" else "normal")
        s = u[u["tool"].eq(tool)]
        for cname in crit:
            v = s.loc[s["criterion"].eq(cname), "score"]
            v = float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan
            if np.isnan(v):
                bx.text(cx[cname], j, "not\nassessed", ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.0)
                continue
            bx.add_patch(Rectangle((cx[cname] - 0.3, j - 0.3), 0.6, 0.6, facecolor=shade[v], edgecolor="#55544f",
                                   lw=0.8, zorder=2))
        got, possible = float(s["score"].sum(skipna=True)), int(s["score"].notna().sum())
        bx.text(total_x, j, f"{got:g} / {possible}", ha="center", va="center", fontsize=7.6)
    bx.text(-0.75, -1.74, "b", fontsize=14, fontweight="bold", ha="right", va="center")
    for i, (v, lab) in enumerate([(1.0, "yes"), (0.5, "partly"), (0.0, "no")]):
        bx.add_patch(Rectangle((i * 1.6, len(tools_u) - 0.1), 0.35, 0.35, facecolor=shade[v], edgecolor="#55544f",
                               lw=0.8, clip_on=False))
        bx.text(i * 1.6 + 0.5, len(tools_u) + 0.075, lab, va="center", fontsize=7.4)
    bx.text(4.9, len(tools_u) + 0.075, "Evidence for every score: scripts/usability_checklist.csv. Threadfin is not "
            "yet peer-reviewed; the other tools' rows describe their public releases.", va="center",
            fontsize=7.2, color=INK2)
    fig.text(0.012, 0.975, "The benchmark at a glance: each tool compared with Threadfin, and how easy each tool "
             "is to use", fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.012, 0.952, "Rows: tools at their own defaults and forced to the smallest clones, and simple baselines. "
             "11 real datasets (9 B-cell, 2 LARRY lineage-barcode), 12 simulated datasets.",
             fontsize=8.4, color=INK2, va="center")
    return fig


def main():
    ov = Overview()
    cells, rows = {}, []
    for label, key, kind in ROWS:
        tally = {"better": 0, "worse": 0}
        for col, group, clabel in COLUMNS[:-1]:
            c = ov.cell(key, col)
            cells[(key, col)] = c
            if c.verdict in tally and kind != "ref":
                tally[c.verdict] += 1
            rows.append({"row": label, "column": clabel.replace("\n", " "), "verdict": c.verdict,
                         "shown": c.text.replace("\n", " "), "detail": c.detail, "n": c.n, "p": c.p})
        cells[(key, "tally")] = Cell("na", "—" if kind == "ref" else f"{tally['better']} better\n{tally['worse']} worse")
    pd.DataFrame(rows).to_csv(RESULTS / "overview.csv", index=False)
    u = usability()
    u.to_csv(RESULTS / "usability.csv", index=False)
    (RESULTS / "overview_rules.json").write_text(json.dumps(RULES, indent=1, ensure_ascii=False) + "\n")
    fig = draw(cells, u)
    save(fig, "fig0_overview.png")


if __name__ == "__main__":
    main()
