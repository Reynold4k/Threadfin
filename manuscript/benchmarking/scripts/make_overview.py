#!/usr/bin/env python
"""Figure 1 of the benchmark: every tool compared with Threadfin at a glance, and how easy each tool is to use.

a  One column per question of the benchmark, in plain words, and one row per tool. A tile says whether the
   tool did better than Threadfin, about the same, or worse, and gives the number behind the verdict;
   Threadfin's own row gives its value. Failed and stopped runs are counted, not dropped. The rule for each
   column is in RULES (also printed in the README). The last row is a negative control: Threadfin against
   itself with another random seed, which shows how often the rules call a difference where there is none.
b  Usability checklist (scripts/usability_checklist.csv): 13 criteria in five groups, each scored 1 (yes),
   0.5 (partly) or 0 (no), with the evidence for every score in the same file. The two scores that this
   benchmark measured itself (same result with more cores) come from results/resources_reproducibility.csv.

Everything in panel a is computed from the tables in results/. Writes figures/fig1_overview.png (and .pdf),
results/overview.csv, results/overview_alpha_sensitivity.csv and results/usability.csv.
Usage: python make_overview.py
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
from make_figures import FIG, INK, INK2, MM, MUTED, ORDER, WIDTH, save  # noqa: E402

# verdict colours: a green / magenta pair that no tool uses (Threadfin is blue elsewhere), checked for colour-blind
# separation from each other and from Threadfin's blue; white text on both has contrast >= 5:1
BETTER, SIMILAR, WORSE = "#1a7f1a", "#e4e3df", "#b0306f"
TF = "threadfin_kernel"
ALPHA, TOL = 0.05, 0.25           # paired-test level; tolerance for single measurements (25 %)
MIN_UNITS = 6                     # fewest datasets or seeds with which a two-sided paired test can give P < 0.05
STOP_SECONDS = 72 * 3600
# fewest clones (of the tool's minimum size) a tool can represent at all: clone2vec's SVD initialisation needs more
# clones than its 10 dimensions, and the benchmark does not start Clonotrace on fewer than 3 clones
TOOL_MIN_CLONES = {"clone2vec": 11, "clonotrace": 3}

ROWS = [  # label, key, kind
    ("Threadfin\n(donor as context, as in\nthe case studies)", TF, "ref"),
    ("clone2vec, its default\n(clones ≥ 3 cells)", "clone2vec_min3", "tool"),
    ("clone2vec, forced to\nclones ≥ 2 cells", "clone2vec_min2", "tool"),
    ("Clonotrace, its default\n(clones ≥ 10 cells)", "clonotrace_revised_min10", "tool"),
    ("Clonotrace, forced to\nclones ≥ 2 cells", "clonotrace_revised_min2", "tool"),
    ("Clonotrace, exact OT\n(≥ 2 cells; ≥ 10 where\nonly that ran)", "clonotrace_exact", "tool"),
    ("Benisse", "benisse", "tool"),
    ("CoNGA's one-cell-\nper-clone rule", "conga_medoid", "baseline"),
    ("Mean of the embedding\n(baseline)", "mean_pca", "baseline"),
    ("Cluster composition\n(baseline)", "composition", "baseline"),
    ("Threadfin with another\nrandom seed\n(negative control)", "threadfin_kernel_seed1", "control"),
]
COLUMNS = [  # key, group, label
    ("coverage", "Which clones it can use", "Share of\nclones it\ndescribes"),
    ("two_cell", "Which clones it can use", "Keeps\ntwo-cell\nclones"),
    ("datasets", "Which clones it can use", "Datasets\nit could\nanalyse"),
    ("traits", "How well it describes clones", "Predicts\nmeasured\nclone traits"),
    ("traits_disjoint", "How well it describes clones", "… when\ntraits come\nfrom other\ncells"),
    ("reid", "How well it describes clones", "Recognises\na clone\nfrom other\ncells"),
    ("fate", "How well it describes clones", "Links\nday-4\nclones to\nday 6"),
    ("sim_groups", "How well it describes clones", "Finds\nplanted\nclone\ngroups"),
    ("sim_mixed", "How well it describes clones", "Tells\nmixed from\nin-between\nclones"),
    ("time", "Computer resources", "Run\ntime"),
    ("memory", "Computer resources", "Peak\nmemory"),
    ("cores", "Computer resources", "Gain from\n16 CPU\ncores"),
    ("growth", "Computer resources", "Time for\n10× the\nclones"),
    ("finished", "Computer resources", "Runs that\nended\nwithout\nhelp"),
    ("tally", "Count", "Tiles\nbetter /\nworse"),
]
RULES = {
    "coverage": "median over the datasets of the share of clones with ≥ 2 cells that the tool represents; a run that "
                "failed, stopped or could not start because too few clones reach the tool's minimum size counts as 0. "
                "Datasets outside a tool's scope (LARRY barcodes for Benisse, which needs receptor sequences) and runs "
                "the authors cancelled are left out. Better or worse: more than 5 percentage points from Threadfin.",
    "two_cell": "the same for clones of exactly two cells.",
    "datasets": "datasets on which the tool produced a result, of those within its scope (the same rule as above).",
    "traits": "measured annotations that no tool used (sorting gates, isotype, mutation, tissue, time, later fate; "
              "RNA-derived calls excluded), each scored on the clones the tool represents: the tool is better (worse) "
              "on an annotation when the 95 % paired bootstrap interval (10,000 resamples) of R² − R²(Threadfin) "
              "excludes 0; an interval end within the Monte-Carlo error of 0 is unresolved and not counted. Column "
              "verdict: two-sided sign test of better vs worse annotations, P < 0.05.",
    "traits_disjoint": "as above, but the representation and the annotation come from disjoint halves of each clone's "
                       "cells, and the tool's R² is compared with Threadfin's on the same halves and the same clones "
                       "(3 random splits, both directions, averaged; Clonotrace and Benisse: split 0 only); sign test "
                       "over annotations.",
    "reid": "share of clones whose representation from one half of their cells is nearest to their own other half, "
            "compared with Threadfin on the same halves and clones (median over datasets of the mean over splits); "
            "Wilcoxon signed-rank test over datasets. With fewer than 6 datasets no paired test can reach P < 0.05, "
            "so the tile is descriptive.",
    "fate": "LARRY: mean reciprocal rank (ties counted as mid-ranks) of each day-4 clone's own day-6 clone among all "
            "day-6 clones; 95 % paired bootstrap interval over clones (10,000 resamples).",
    "sim_groups": "simulated data with a known answer: share of each clone's 10 nearest clones that follow its planted "
                  "programme (mean over clones), compared with Threadfin on the same clones; the unit is the simulation seed (the two "
                  "donor-shift levels and the two clone-size regimes of a seed share clones or donors, so their "
                  "differences are averaged within the seed); Wilcoxon signed-rank test over seeds. 10 seeds for every "
                  "tool except Clonotrace (3 seeds: descriptive, since no paired test of 3 seeds can reach P < 0.05).",
    "sim_mixed": "AUROC for telling clones made of two kinds of cells from clones of cells halfway between them "
                 "(clones ≥ 10 cells), tested over seeds as above.",
    "time": "time to compute the clone representation, divided by Threadfin's on the same dataset (median); "
            "Clonotrace: only runs whose clone distances used several R sessions, a run stopped at 72 h counted as "
            "72 h. Better or worse: every ratio beyond 25 % in the same direction, otherwise a Wilcoxon test on "
            "the log ratios (descriptive with fewer than 6 datasets).",
    "memory": "peak memory, the same way. For Clonotrace /usr/bin/time sees only the main R session, so the job's "
              "peak from the scheduler is used where the run had a job of its own, and the main session otherwise "
              "(a lower bound, marked ≥).",
    "cores": "time with 1 core divided by time with 16 cores (GSE287123 RBD). The scopes differ: Threadfin's complete "
             "analysis, the other tools' clone representation. Better or worse: more than 25 % from Threadfin's gain.",
    "growth": "how much longer the clone representation takes for 10 times more clones, from 5-50 % subsets of "
              "GSE195673 (8 cores; exponent of a fit to 4 subsets). Better or worse: an exponent more than 0.25 "
              "from Threadfin's.",
    "finished": "runs on the full data and on the disjoint halves that finished with the tool's default settings; "
                "failed or stopped runs, and runs that needed a larger cell graph, are counted against the tool. Inputs "
                "with fewer clones than a tool can represent at all (clone2vec: 11 clones of ≥ 3 cells; Clonotrace: 3 "
                "of its minimum size) and runs the authors cancelled are not counted.",
    "tally": "number of tiles of the row that are better and worse than Threadfin. The columns are not independent "
             "questions, so the count is a summary of the row, not a score or a ranking.",
}


@dataclass
class Cell:
    verdict: str                  # ref, better, similar, worse, descriptive, unresolved, na, pending
    text: str = ""
    detail: str = ""
    n: int | None = None
    p: float | None = None


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def pct(v):
    return f"{100 * v:.0f} %" if v >= 0.095 or v == 0 else f"{100 * v:.1f} %"


class Rules:
    """The tests behind the verdicts, at a chosen significance level (ALPHA; re-run at 0.01 for the sensitivity)."""

    def __init__(self, alpha=ALPHA):
        self.alpha = alpha

    def sign_test(self, n_up, n_down):
        n = n_up + n_down
        if n == 0:
            return "similar", None
        p = float(stats.binomtest(n_up, n).pvalue)
        return ("better" if n_up > n_down else "worse") if p < self.alpha else "similar", p

    def paired(self, diffs, higher_better=True):
        """Verdict from paired differences (tool - Threadfin) over datasets or seeds."""
        d = np.asarray(diffs, float)
        d = d[np.isfinite(d)]
        if not higher_better:
            d = -d
        if d.size == 0:
            return "na", None
        if np.allclose(d, 0):
            return "similar", 1.0
        if d.size < MIN_UNITS:
            return "descriptive", None
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
        return ("better" if (d > 0).sum() > (d < 0).sum() else "worse") if p < self.alpha else "similar", p

    def ratio_verdict(self, ratios, lower_better=True):
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
        if r.size >= MIN_UNITS:
            p = float(stats.wilcoxon(lr).pvalue)
            if p < self.alpha:
                return ("better" if np.median(lr) > 0 else "worse"), p
            return "similar", p
        return "descriptive", None


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
    def __init__(self, alpha=ALPHA):
        self.rules = Rules(alpha)
        self.cov = read("coverage_runtime.csv")
        self.cov = self.cov[self.cov["dataset"].isin(ORDER)]
        self.pw = read("annotation_pairwise.csv")
        self.dis = read("annotation_recovery_disjoint.csv")
        self.rep = read("split_half_reproducibility.csv")
        self.larry = read("larry_clone_linking.csv")
        self.sim = read("simulation.csv")
        self.res = read("resources.csv")
        self.status = read("run_status.csv")
        self.hosts = read("run_hosts.csv")

    # helpers -----------------------------------------------------------------
    @staticmethod
    def method_names(key):
        return {"benisse": ["benisse_min2", "benisse_min10"],
                "clonotrace_exact": ["clonotrace_exact_min2", "clonotrace_exact_min10"]}.get(key, [key])

    @staticmethod
    def native(key):
        return key in ("clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10")

    def runs(self, key):
        """One coverage row per dataset for a row of the figure: the first completed run among the row's methods,
        else the most informative unfinished one (failed > stopped > cancelled > not run)."""
        rank = {"completed": 0, "failed": 1, "stopped": 2, "cancelled": 3}
        out = []
        for ds in ORDER:
            sub = self.cov[self.cov["dataset"] == ds].set_index("method")
            if sub.empty:
                continue
            cand = [sub.loc[m] for m in self.method_names(key) if m in sub.index]
            r = min(cand, key=lambda c: rank.get(c["status"], 4)).copy()
            r["method_used"] = r.name
            out.append(r)
        return pd.DataFrame(out)

    @staticmethod
    def scope(r):
        """not applicable: outside the tool's scope or cancelled by the authors (left out of every column)."""
        reason = str(r.get("reason", ""))
        return "no receptor" in reason or r.get("status") == "cancelled"

    def pick_per_dataset(self, frame, key):
        """Rows of the first of the row's methods that a dataset has (Benisse: >= 2 cells, else >= 10 cells)."""
        names = self.method_names(key)
        f = frame[frame["method"].isin(names)].copy()
        if f.empty or len(names) == 1:
            return f
        f["_rank"] = f["method"].map({m: i for i, m in enumerate(names)})
        keep = f.groupby("dataset")["_rank"].transform("min")
        return f[f["_rank"].eq(keep)].drop(columns="_rank")

    # which clones -------------------------------------------------------------
    def coverage(self, key, two=False):
        if key == "threadfin_kernel_seed1":
            return Cell("na", "—")
        r = self.runs(key)
        num, den = ("n_two_cell_clones_represented", "n_two_cell_clones") if two else ("n_clones_represented", "n_clones_ge2")
        r = r[~r.apply(self.scope, axis=1)]
        r = r[r[den] > 0]
        v = np.where(r["status"].eq("completed"), r[num] / r[den], 0.0)
        if key == TF:
            return Cell("ref", pct(np.median(v)), n=len(v))
        med = float(np.median(v))
        verdict = "worse" if med < 1 - 0.05 else "better" if med > 1 + 0.05 else "similar"
        return Cell(verdict, pct(med), f"per dataset: {', '.join(pct(x) for x in v)}", n=len(v))

    def datasets(self, key):
        if key == "threadfin_kernel_seed1":
            return Cell("na", "—")
        r = self.runs(key)
        out_of_scope = r.apply(self.scope, axis=1)
        notes_left_out = []
        if r["reason"].astype(str).str.contains("no receptor").any():
            notes_left_out.append(f"{int(r['reason'].astype(str).str.contains('no receptor').sum())} without BCR")
        if r["status"].eq("cancelled").any():
            notes_left_out.append(f"{int(r['status'].eq('cancelled').sum())} cancelled")
        r = r[~out_of_scope]
        ok = r["status"].eq("completed")
        n = int(ok.sum())
        if key == TF:
            return Cell("ref", f"{n} of {len(r)}", n=len(r))
        why = []
        reasons = r.loc[~ok, "reason"].astype(str)
        if reasons.str.contains("fewer than 3").any():
            why.append(f"{int(reasons.str.contains('fewer than 3').sum())} too small")
        if r.loc[~ok, "status"].eq("stopped").any():
            why.append("1 stopped")
        if r.loc[~ok, "status"].eq("failed").any():
            why.append(f"{int(r.loc[~ok, 'status'].eq('failed').sum())} failed")
        text = f"{n} of {len(r)}" + ("\n" + why[0].replace(" too small", "\ntoo small") if why else "")
        detail = "; ".join(why + [f"not counted: {x}" for x in notes_left_out])
        return Cell("worse" if n < len(r) else "similar", text, detail, n=len(r))

    # how well ----------------------------------------------------------------
    def traits(self, key):
        if key == TF:
            return Cell("ref", "reference")
        s = self.pick_per_dataset(self.pw[self.pw["tag"] != "rna"], key)
        if s.empty:
            return Cell("na", "—")
        up, down = int(s["call"].eq("better").sum()), int(s["call"].eq("worse").sum())
        unres = int(s["call"].eq("unresolved").sum())
        v, p = self.rules.sign_test(up, down)
        detail = f"{up} better, {down} worse, {unres} unresolved, {len(s)} annotations"
        return Cell(v, f"{up} better\n{down} worse", detail, len(s), p)

    def _set(self, key, forced, native, control):
        if key == "threadfin_kernel_seed1":
            return control
        return native if self.native(key) else forced

    def traits_disjoint(self, key):
        if key == TF:
            return Cell("ref", "reference")
        set_name = self._set(key, "forced: >= 2 cells per half", "native: >= 10 cells per half",
                             "sensitivity: >= 2 cells per half")
        d = self.dis[self.dis["set"].eq(set_name) & (self.dis["tag"] != "rna")]
        tool = self.pick_per_dataset(d, key)
        if tool.empty:
            return Cell("na", "not run on\nhalves" if key == "clonotrace_exact" else "—")
        keys = ["dataset", "target", "seed", "representation_half"]
        j = tool[keys + ["r2"]].merge(d[d["method"].eq(TF)][keys + ["r2"]], on=keys, suffixes=("_tool", "_tf"))
        g = j.groupby(["dataset", "target"])[["r2_tool", "r2_tf"]].mean()
        diff = (g["r2_tool"] - g["r2_tf"]).dropna()
        up, down = int((diff > 0).sum()), int((diff < 0).sum())
        v, p = self.rules.sign_test(up, down)
        return Cell(v, f"{up} better\n{down} worse", f"{len(diff)} annotations (matched halves)", len(diff), p)

    def reid(self, key):
        set_name = self._set(key, "forced: >= 2 cells per half", "native: >= 10 cells per half",
                             "sensitivity: >= 2 cells per half")
        r = self.rep[self.rep["set"].eq(set_name)]
        if key == TF:
            tf = r[r["method"].eq(TF)]
            return Cell("ref", pct(tf.groupby("dataset")["reid_top1"].mean().median()))
        tool = self.pick_per_dataset(r, key)
        if tool.empty:
            return Cell("na", "not run on\nhalves" if key == "clonotrace_exact" else "—")
        j = tool[["dataset", "seed", "reid_top1"]].merge(r[r["method"].eq(TF)][["dataset", "seed", "reid_top1"]],
                                                         on=["dataset", "seed"], suffixes=("_tool", "_tf"))
        g = j.groupby("dataset")[["reid_top1_tool", "reid_top1_tf"]].mean()
        v, p = self.rules.paired(g["reid_top1_tool"] - g["reid_top1_tf"])
        higher = int((g["reid_top1_tool"] > g["reid_top1_tf"]).sum())
        text = f"{pct(g['reid_top1_tool'].median())}\nvs {pct(g['reid_top1_tf'].median())}"
        detail = (f"{len(g)} datasets (matched halves); tool higher on {higher}"
                  + ("; too few datasets to test" if v == "descriptive" else ""))
        return Cell(v, text, detail, len(g), p)

    def fate(self, key):
        lk = self.larry[self.larry["set"].eq("native: clones >= 10 cells" if self.native(key) else "forced: clones >= 2 cells")]
        if key == TF:
            r = self.larry[self.larry["set"].eq("forced: clones >= 2 cells") & self.larry["method"].eq(TF)]
            return Cell("ref", f"{r['mrr'].iloc[0]:.3f}" if len(r) else "—")
        r = lk[lk["method"].isin(self.method_names(key))]
        tf = lk[lk["method"].eq(TF)]
        if r.empty or tf.empty:
            return Cell("na", "cancelled" if key == "clonotrace_exact" else "—")
        r = r.iloc[0]
        lo, hi = r.get("mrr_delta_ci_low", np.nan), r.get("mrr_delta_ci_high", np.nan)
        v = {"better": "better", "worse": "worse", "unresolved": "unresolved"}.get(r.get("call"), "similar")
        return Cell(v, f"{r['mrr']:.3f}\nvs {tf['mrr'].iloc[0]:.3f}", f"95 % CI of difference [{lo:.3f}, {hi:.3f}]",
                    int(r["n_query_clones"]))

    def sim_set(self, key):
        if key in ("clonotrace_revised_min10", "clonotrace_raw_min10"):
            return "native: clones >= 10 cells"
        if key in ("clonotrace_revised_min2", "clonotrace_raw_min2"):
            return "forced: clones >= 2 cells"
        if key == "clonotrace_exact":
            return "exact OT: clones >= 2 cells"
        if key == "threadfin_kernel_seed1":
            return "seeds sensitivity: clones >= 2 cells"
        return "seeds: clones >= 10 cells" if self.native(key) else "seeds: clones >= 2 cells"

    def simulation(self, key, what):
        s = self.sim
        if what == "mixture_vs_intermediate_auroc":
            s = s[s["regime"].eq("large")]
        if key == "benisse":
            return Cell("na", "no BCR")
        if key == TF:
            base = s[s["set"].eq("seeds: clones >= 2 cells") & s["method"].eq(TF)]
            return Cell("ref", f"{base.groupby('seed')[what].mean().mean():.2f}")
        s = s[s["set"].eq(self.sim_set(key))]
        names = ["clonotrace_exact_min2"] if key == "clonotrace_exact" else self.method_names(key)
        m = s.pivot_table(index="dataset", columns="method", values=what, aggfunc="first")
        cols = [c for c in names if c in m]
        if not cols or TF not in m:
            return Cell("na", "not run" if key == "clonotrace_exact" else "—")
        both = pd.concat([m[cols[0]], m[TF]], axis=1).dropna()
        both.columns = ["tool", "tf"]
        if both.empty:
            return Cell("na", "—")
        seed = s.drop_duplicates("dataset").set_index("dataset")["seed"]
        per_seed = both.groupby(seed.loc[both.index].to_numpy()).mean()
        v, p = self.rules.paired(per_seed["tool"] - per_seed["tf"])
        text = f"{per_seed['tool'].mean():.2f}\nvs {per_seed['tf'].mean():.2f}"
        detail = (f"{len(per_seed)} seeds ({len(both)} simulated datasets)"
                  + ("; too few seeds to test" if v == "descriptive" else ""))
        return Cell(v, text, detail, len(per_seed), p)

    # resources -----------------------------------------------------------------
    def _usable_runs(self, key):
        r = self.runs(key)
        tf = self.runs(TF).set_index("dataset")
        keep = r["status"].eq("completed")
        if key.startswith("clonotrace"):
            keep &= r["distance_workers"].fillna(1) > 1
            keep |= r["status"].eq("stopped")
        return r[keep], tf

    def job_peak(self, r):
        """The scheduler's peak memory of the job that ran this run alone (None if it shared its job)."""
        if self.hosts is None:
            return None
        h = self.hosts[self.hosts["dataset"].eq(r["dataset"]) & self.hosts["variant"].eq("full")
                       & self.hosts["tool_run"].eq(self._run_dir(r["method_used"]))]
        if h.empty or not bool(h["alone_in_job"].iloc[0]) or pd.isna(h["job_max_rss_gb"].iloc[0]):
            return None
        return float(h["job_max_rss_gb"].iloc[0])

    @staticmethod
    def _run_dir(method):
        return {"clonotrace_revised_min10": "clonotrace_min10", "clonotrace_raw_min10": "clonotrace_min10",
                "clonotrace_revised_min2": "clonotrace_min2", "clonotrace_raw_min2": "clonotrace_min2"}.get(method, method)

    def time(self, key):
        if key == TF:
            t = self.runs(TF)["seconds_method"]
            return Cell("ref", f"{t.median():.1f} s\n(median)")
        if key in ("conga_medoid", "mean_pca", "composition"):
            return Cell("na", "not timed")
        if key == "threadfin_kernel_seed1":
            return Cell("na", "—")
        r, tf = self._usable_runs(key)
        sec = np.where(r["status"].eq("stopped"), STOP_SECONDS,
                       r["seconds_method"].fillna(r["wall_seconds_task"]).to_numpy(float))
        ratios = sec / tf.loc[r["dataset"], "seconds_method"].to_numpy(float)
        v, p = self.rules.ratio_verdict(ratios)
        stopped = int(r["status"].eq("stopped").sum())
        text = times_text(float(np.median(ratios))).replace(" ", "\n", 1)
        if stopped:
            text = "≥ " + text + f", {stopped} stopped"
        return Cell(v, text, f"{len(ratios)} datasets" + ("; too few to test" if v == "descriptive" else ""),
                    len(ratios), p)

    def memory(self, key):
        if key == TF:
            return Cell("ref", f"{self.runs(TF)['peak_rss_gb_method'].median():.1f} GB\n(median)")
        if key in ("conga_medoid", "mean_pca", "composition"):
            return Cell("na", "not\nmeasured")
        if key == "threadfin_kernel_seed1":
            return Cell("na", "—")
        r, tf = self._usable_runs(key)
        r = r[r["status"].eq("completed")]
        peaks, lower = [], 0
        for _, row in r.iterrows():
            jp = self.job_peak(row) if key.startswith("clonotrace") else None
            if key.startswith("clonotrace") and jp is None:
                lower += 1
            peaks.append(max(jp or 0.0, float(row["peak_rss_gb_task"])))
        ratios = np.array(peaks) / tf.loc[r["dataset"], "peak_rss_gb_method"].to_numpy(float)
        v, p = self.rules.ratio_verdict(ratios)
        text = ("≥ " if lower else "") + times_text(float(np.median(ratios)), "more", "less").replace(" ", "\n", 1)
        detail = f"{len(ratios)} datasets" + (f"; {lower} main-session peaks (lower bounds)" if lower else "") \
            + ("; too few to test" if v == "descriptive" else "")
        return Cell(v, text, detail, len(ratios), p)

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
            return float(s[1] / s[16]) if {1, 16} <= set(s.index) else np.nan

        s_tf, s = speedup("threadfin_run"), speedup(tool)
        if not np.isfinite(s):
            return Cell("pending", "running")
        text = f"{s:.1f}×" if s >= 1 else f"{s:.2f}×"
        if key == TF:
            return Cell("ref", text + "\n(analysis)")
        text += f"\nvs {s_tf:.1f}×"
        v = "better" if s > s_tf * (1 + TOL) else "worse" if s < s_tf / (1 + TOL) else "similar"
        return Cell(v, text, f"Threadfin {s_tf:.1f}× (complete analysis); tool: clone representation")

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
        text = f"{10 ** b:,.0f}× time"
        if key == TF:
            return Cell("ref", text + "\n(profiles)")
        v = "better" if b < b_tf - TOL else "worse" if b > b_tf + TOL else "similar"
        return Cell(v, f"{10 ** b:,.0f}×\nvs {10 ** b_tf:,.0f}×", f"exponent {b:.2f} vs {b_tf:.2f}")

    def finished(self, key):
        st = self.status
        if st is None or key == "threadfin_kernel_seed1":
            return Cell("na", "—")
        runs = {"clonotrace_revised_min10": ["clonotrace_min10"], "clonotrace_revised_min2": ["clonotrace_min2"],
                "clonotrace_exact": ["clonotrace_exact_min2", "clonotrace_exact_min10"],
                "benisse": ["benisse_min2", "benisse_min10"]}.get(key, [key])
        s = st[st["tool_run"].isin(runs)]
        if s.empty:
            return Cell("na", "—")
        tool = key.split("_")[0]
        too_few = s["n_clones_eligible"] < TOOL_MIN_CLONES.get(tool, 0)
        cancelled = s["status"].eq("cancelled")
        n_na = int((too_few & ~cancelled).sum())
        na_inputs = s[too_few & ~cancelled].assign(half=lambda f: f["variant"].str[:5]).drop_duplicates(["dataset", "half"])
        s = s[~too_few & ~cancelled]
        big_graph = s["status"].eq("completed") & (s["graph_k_used"].fillna(10) > 10)
        unaided = s["status"].eq("completed") & ~big_graph
        n, u = len(s), int(unaided.sum())
        notes = []
        for what, mask in (("stopped", s["status"].eq("stopped")), ("failed", s["status"].eq("failed")),
                           ("re-run", big_graph)):
            if mask.any():
                notes.append(f"{int(mask.sum())} {what}")
        text = f"{u} of {n}" + ("\n" + ", ".join(notes[:2]) if notes else "")
        detail = "; ".join(notes)
        if n_na:
            detail += (f"; not counted: {n_na} runs on {len(na_inputs)} inputs with fewer than "
                       f"{TOOL_MIN_CLONES[tool]} clones the tool can represent")
        if cancelled.any():
            detail += f"; not counted: {int(cancelled.sum())} runs cancelled by the authors"
        if key == TF:
            return Cell("ref", text, detail.strip("; "))
        return Cell("worse" if u < n else "similar", text, detail.strip("; "), n)

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
        if key == "threadfin_kernel_seed1" and col in ("fate", "cores", "growth"):
            return Cell("na", "—")
        try:
            return f()
        except (KeyError, IndexError, ValueError) as e:      # a readout that this row does not have
            return Cell("na", "—", f"{type(e).__name__}: {e}")


def tiles(alpha=ALPHA):
    ov = Overview(alpha)
    cells, rows = {}, []
    for label, key, kind in ROWS:
        tally = {"better": 0, "worse": 0}
        for col, group, clabel in COLUMNS[:-1]:
            c = ov.cell(key, col)
            cells[(key, col)] = c
            if c.verdict in tally and kind != "ref":
                tally[c.verdict] += 1
            rows.append({"row": label.replace("\n", " ").replace("- ", "-"), "column": " ".join(clabel.replace("-\n", "").split()),
                         "verdict": c.verdict, "shown": c.text.replace("\n", " "), "detail": c.detail, "n": c.n, "p": c.p})
        cells[(key, "tally")] = Cell("na", "—" if kind == "ref" else f"{tally['better']} / {tally['worse']}")
        rows.append({"row": label.replace("\n", " ").replace("- ", "-"), "column": "Tiles better / worse",
                     "verdict": "count", "shown": cells[(key, "tally")].text, "detail": RULES["tally"], "n": None, "p": None})
    return cells, pd.DataFrame(rows)


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
            mask = u["tool"].eq(tool) & u["criterion"].eq("Same result with more cores")
            u.loc[mask, "score"] = 1.0 if same else 0.5
            u.loc[mask, "evidence"] = ("identical with 1-16 cores (this benchmark)" if same else
                                       f"slightly different with 1-16 cores: clone distances correlate {rho:.4f} "
                                       f"(Spearman), max difference {r['max_abs_difference'].max():.3g} (this benchmark)")
    return u


# --------------------------------------------------------------------------- drawing

STYLE = {"better": (BETTER, "white", BETTER, "-"), "worse": (WORSE, "white", WORSE, "-"),
         "similar": (SIMILAR, INK, SIMILAR, "-"), "ref": ("white", INK, INK2, "-"),
         "descriptive": ("white", INK, MUTED, (0, (2, 1.5))), "unresolved": ("white", INK, MUTED, (0, (0.6, 1.2)))}
MARK = {"better": "▲ ", "worse": "▼ "}


def draw(cells: dict, u: pd.DataFrame):
    xs, x = [], 0.0
    for i, (key, group, _) in enumerate(COLUMNS):
        if i and group != COLUMNS[i - 1][1]:
            x += 0.3
        xs.append(x)
        x += 1.0
    width_units = xs[-1] + 1
    n_rows = len(ROWS)
    tools_u = ["Threadfin", "Clonotrace", "clone2vec", "Benisse", "CoNGA", "scirpy"]
    crit = list(dict.fromkeys(u["criterion"]))
    fig = plt.figure(figsize=(WIDTH, 205 * MM))
    left, right = 0.155, 0.995
    ax = fig.add_axes([left, 0.37, right - left, 0.585])
    ax.set_xlim(-0.52, width_units - 0.48)
    ax.set_ylim(n_rows - 0.47, -2.35)
    ax.axis("off")
    for g in dict.fromkeys(c[1] for c in COLUMNS):
        idx = [i for i, c in enumerate(COLUMNS) if c[1] == g]
        x0, x1 = xs[idx[0]] - 0.48, xs[idx[-1]] + 0.48
        ax.add_patch(FancyBboxPatch((x0, -2.3), x1 - x0, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="#efeeea", edgecolor="none"))
        ax.text((x0 + x1) / 2, -2.09, g, ha="center", va="center", fontsize=6, fontweight="bold", color=INK)
    for x, (key, _, label) in zip(xs, COLUMNS):
        ax.text(x, -1.02, label, ha="center", va="center", fontsize=5, color=INK, linespacing=1.05)
    for j, (label, key, kind) in enumerate(ROWS):
        if j % 2 == 0:
            ax.add_patch(Rectangle((-0.52, j - 0.5), width_units + 0.04, 1.0, facecolor="#f5f4f1", edgecolor="none",
                                   zorder=0))
        if kind == "control":
            ax.plot([-0.52, width_units - 0.48], [j - 0.5, j - 0.5], color=INK2, lw=0.6, clip_on=False)
        ax.text(-0.62, j, label, ha="right", va="center", fontsize=5.5, linespacing=1.05,
                fontweight="bold" if kind == "ref" else "normal", color=INK if kind != "control" else INK2,
                style="italic" if kind == "control" else "normal")
        for x, (col, _, _) in zip(xs, COLUMNS):
            c = cells[(key, col)]
            if col == "tally":
                ax.text(x, j, c.text, ha="center", va="center", fontsize=5.5, color=INK)
                continue
            if c.verdict in ("na", "pending"):
                ax.text(x, j, c.text or "—", ha="center", va="center", fontsize=5, color=MUTED, linespacing=1.0)
                continue
            face, ink, edge, ls = STYLE[c.verdict]
            ax.add_patch(FancyBboxPatch((x - 0.46, j - 0.42), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.1",
                                        facecolor=face, edgecolor=edge, lw=0.7, ls=ls, zorder=2))
            lines = c.text.split("\n")[:3]
            bold = "bold" if c.verdict in ("better", "worse") else "normal"
            mark = MARK.get(c.verdict, "")
            if mark and len(lines[0]) <= 7:
                lines[0] = mark + lines[0]
            elif mark and len(lines) < 3:             # a long first line: the mark gets a line of its own
                lines = [mark.strip()] + lines
            offs = {1: [0.0], 2: [-0.15, 0.17], 3: [-0.25, 0.02, 0.27]}[len(lines)]
            for k, (line, dy) in enumerate(zip(lines, offs)):
                ax.text(x, j + dy, line, ha="center", va="center", fontsize=5, color=ink, zorder=3,
                        fontweight=bold if k == 0 else "normal")
    fig.text(0.004, 0.947, "a", fontsize=8, fontweight="bold", ha="left", va="center")
    # legend of panel a
    ly = 0.352
    items = [(BETTER, "white", BETTER, "-", "▲ better than Threadfin"), (SIMILAR, INK, SIMILAR, "-", "no clear difference"),
             (WORSE, "white", WORSE, "-", "▼ worse than Threadfin"), ("white", INK, INK2, "-", "Threadfin's own value"),
             ("white", INK, MUTED, "--", "too few datasets or seeds to test")]
    lx = left
    for face, ink, edge, ls, lab in items:
        t = fig.text(lx, ly, f" {lab} ", fontsize=5.5, color=ink, va="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor=edge, lw=0.7, ls=ls))
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        lx = bb.x1 + 0.012
    fig.text(left, 0.333, "Each tile compares the tool with Threadfin on the same clones; the numbers are the tool's and, "
             "after \"vs\", Threadfin's. Paired tests (P < 0.05) over annotations,\ndatasets or simulation seeds; single "
             "measurements count as different beyond 25 %. —: no such output. Rules: README, section 0. The last row "
             "shows\nhow often the rules call a difference between two runs of Threadfin that differ only in their "
             "random seed.", fontsize=5, color=INK2, va="top", linespacing=1.15)

    # panel b: usability checklist
    bx = fig.add_axes([left, 0.03, right - left, 0.24])
    order = list(dict.fromkeys(u["group"]))
    cx, x = {}, 0.0
    for i, cname in enumerate(crit):
        g = u.loc[u["criterion"].eq(cname), "group"].iloc[0]
        if i and g != u.loc[u["criterion"].eq(crit[i - 1]), "group"].iloc[0]:
            x += 0.3
        cx[cname] = x
        x += 1.0
    total_x = x + 0.3
    bx.set_xlim(-0.55, total_x + 0.7)
    bx.set_ylim(len(tools_u) - 0.45, -2.2)
    bx.axis("off")
    for g in order:
        names = [c for c in crit if u.loc[u["criterion"].eq(c), "group"].iloc[0] == g]
        x0, x1 = cx[names[0]] - 0.48, cx[names[-1]] + 0.48
        bx.add_patch(FancyBboxPatch((x0, -2.15), x1 - x0, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="#efeeea", edgecolor="none"))
        bx.text((x0 + x1) / 2, -1.94, g, ha="center", va="center", fontsize=6, fontweight="bold")
    short = {"Open-source licence": "Open-\nsource\nlicence", "Numbered releases": "Numbered\nreleases",
             "Installs with one command": "One-\ncommand\ninstall", "Dependencies listed with versions": "Depend-\nencies\nversioned",
             "Automated tests": "Automated\ntests", "Tests run on every change": "Tests on\nevery\nchange",
             "Same result with more cores": "Same\nresult,\nany cores", "Step-by-step tutorial": "Step-by-\nstep\ntutorial",
             "Every function documented": "Docu-\nmented\nfunctions", "Explains bad input and how to fix it": "Explains\nbad input",
             "Can use several CPU cores": "Uses\nseveral\ncores", "Works with AnnData or Seurat objects": "AnnData\nor Seurat",
             "Peer-reviewed paper": "Peer-\nreviewed"}
    for cname in crit:
        bx.text(cx[cname], -0.9, short.get(cname, textwrap.fill(cname, 11, break_long_words=False)), ha="center",
                va="center", fontsize=5, linespacing=1.0)
    bx.text(total_x, -0.9, "Score", ha="center", va="center", fontsize=5.5, fontweight="bold")
    shade = {1.0: "#55544f", 0.5: "#b7b6b0", 0.0: "white"}
    for j, tool in enumerate(tools_u):
        if j % 2 == 0:
            bx.add_patch(Rectangle((-0.55, j - 0.5), total_x + 1.25, 1.0, facecolor="#f5f4f1", edgecolor="none",
                                   zorder=0))
        bx.text(-0.75, j, tool, ha="right", va="center", fontsize=5.5, fontweight="bold" if tool == "Threadfin" else "normal")
        s = u[u["tool"].eq(tool)]
        for cname in crit:
            v = s.loc[s["criterion"].eq(cname), "score"]
            v = float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan
            if np.isnan(v):
                bx.text(cx[cname], j, "not\nassessed", ha="center", va="center", fontsize=5, color=MUTED, linespacing=0.95)
                continue
            bx.add_patch(Rectangle((cx[cname] - 0.28, j - 0.3), 0.56, 0.6, facecolor=shade[v], edgecolor="#55544f",
                                   lw=0.6, zorder=2))
        got, possible = float(s["score"].sum(skipna=True)), int(s["score"].notna().sum())
        bx.text(total_x, j, f"{got:g} / {possible}", ha="center", va="center", fontsize=5.5)
    fig.text(0.004, 0.262, "b", fontsize=8, fontweight="bold", ha="left", va="center")
    for i, (v, lab) in enumerate([(1.0, "yes"), (0.5, "partly"), (0.0, "no")]):
        bx.add_patch(Rectangle((i * 1.5, len(tools_u) - 0.05), 0.32, 0.4, facecolor=shade[v], edgecolor="#55544f",
                               lw=0.6, clip_on=False))
        bx.text(i * 1.5 + 0.45, len(tools_u) + 0.15, lab, va="center", fontsize=5.5)
    bx.text(4.6, len(tools_u) + 0.15, "The authors' assessment; evidence for every score in scripts/usability_checklist.csv. "
            "Threadfin is not yet peer-reviewed.", va="center", fontsize=5, color=INK2)
    return fig


def alpha_sensitivity(base: pd.DataFrame):
    """Which tiles change when the paired tests use P < 0.01 instead of 0.05."""
    _, strict = tiles(alpha=0.01)
    m = base.merge(strict, on=["row", "column"], suffixes=("_005", "_001"))
    m = m[m["verdict_005"].ne(m["verdict_001"]) & m["verdict_005"].ne("count")]
    return m[["row", "column", "verdict_005", "verdict_001", "shown_005", "p_005"]]


def main():
    cells, ov = tiles()
    ov.to_csv(RESULTS / "overview.csv", index=False)
    alpha_sensitivity(ov).to_csv(RESULTS / "overview_alpha_sensitivity.csv", index=False)
    u = usability()
    u.to_csv(RESULTS / "usability.csv", index=False)
    (RESULTS / "overview_rules.json").write_text(json.dumps(RULES, indent=1, ensure_ascii=False) + "\n")
    fig = draw(cells, u)
    save(fig, FIG["overview"])


if __name__ == "__main__":
    main()
