#!/usr/bin/env python
"""Draw the benchmark figures (``manuscript/benchmarking/figures/``) from the tables in ``results/``.

Every figure is drawn only from the CSV files that ``evaluate.py`` and ``run_tests.py`` write, so the figures
can be redrawn without re-running any method. A figure whose table is missing is skipped with a message.

Colours follow the tool, never its rank: Threadfin (all variants) blue, clone2vec orange, Clonotrace aqua,
Benisse yellow, the simple baselines grey. Every mark is also named on an axis or in a legend, so colour is
never the only cue.

Usage: python make_figures.py
"""
from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bench_paths import RESULTS  # noqa: E402

FIGURES = RESULTS.parent / "figures"

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
TF, C2V, CT, BN, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200, "lines.linewidth": 1.2,
})

METHOD = {
    "threadfin_kernel": ("Threadfin (default)", TF),
    "threadfin_mean": ("Threadfin, mean profile", TF),
    "threadfin_kernel_nocentre": ("Threadfin, no centring", TF),
    "threadfin_kernel_nosmooth": ("Threadfin, no smoothing", TF),
    "threadfin_kernel_noshrink": ("Threadfin, no shrinkage", TF),
    "clone2vec_min3": ("clone2vec (default, ≥ 3 cells)", C2V),
    "clone2vec_min2": ("clone2vec (forced to ≥ 2 cells)", C2V),
    "clonotrace_revised_min10": ("Clonotrace (default, ≥ 10 cells)", CT),
    "clonotrace_raw_min10": ("Clonotrace, published distance (≥ 10)", CT),
    "clonotrace_revised_min2": ("Clonotrace (forced to ≥ 2 cells)", CT),
    "clonotrace_raw_min2": ("Clonotrace, published distance (≥ 2)", CT),
    "benisse_min2": ("Benisse", BN),
    "benisse_min10": ("Benisse (≥ 10 cells)", BN),
    "mean_pca": ("Mean of the embedding (baseline)", BASE),
    "composition": ("Cluster composition (baseline)", BASE),
    "conga_medoid": ("Medoid cell, CoNGA's rule (baseline)", BASE),
}
DATASET = {
    "gut_gc_reporter": "Gut GC-fate reporter, mouse (GSE284622)",
    "mouse_rbd": "RBD division reporter, mouse (GSE287123)",
    "mouse_np": "NP division reporter, mouse (GSE287123)",
    "gc_np_pc": "GC zones + plasma cells, mouse (GSE246382)",
    "malaria_late": "Plasmodium days 10-42, mouse (GSE286215-2)",
    "malaria": "Plasmodium days 0-14, mouse (GSE286215-1)",
    "ln_vaccine": "Lymph node after mRNA vaccine, human (GSE195673)",
    "spont_gc": "Spontaneous GCs, mouse (GSE203132)",
    "covid_bcell": "B cells after SARS-CoV-2, human (GSE171703)",
    "larry_d2": "LARRY day 2 -> day-6 fate (GSE140802)",
    "larry_d46": "LARRY days 4 and 6 (GSE140802)",
}
ORDER = list(DATASET)


def name(m):
    return METHOD.get(m, (m, MUTED))[0]


def colour(m):
    return METHOD.get(m, (m, MUTED))[1]


def read(fname):
    p = RESULTS / fname
    if not p.exists():
        print(f"skip: {fname} not found")
        return None
    return pd.read_csv(p)


def save(fig, fname):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{fname}")


# --------------------------------------------------------------------------- 1. coverage

COVERAGE_COLUMNS = [("threadfin_kernel", "Threadfin\n(default)"), ("clone2vec_min3", "clone2vec\n(default, ≥ 3)"),
                    ("clone2vec_min2", "clone2vec\n(forced, ≥ 2)"), ("clonotrace_revised_min10", "Clonotrace\n(default, ≥ 10)"),
                    ("clonotrace_revised_min2", "Clonotrace\n(forced, ≥ 2)"), ("benisse", "Benisse")]


def coverage_figure():
    cov = read("coverage_runtime.csv")
    if cov is None:
        return
    cov = cov[cov["dataset"].isin(ORDER)]
    rows = [d for d in ORDER if d in set(cov["dataset"])]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd"] + BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(rows) + 1.6), constrained_layout=True)
    for ax, (what, total, title) in zip(axes, [
            ("n_clones_represented", "n_clones_ge2", "a  Clones with ≥ 2 cells that each tool represents"),
            ("n_two_cell_clones_represented", "n_two_cell_clones", "b  Clones of exactly 2 cells that each tool represents")]):
        ax.grid(False)
        for i, ds in enumerate(rows):
            sub = cov[cov["dataset"] == ds].set_index("method")
            for j, (m, _) in enumerate(COVERAGE_COLUMNS):
                if m == "benisse":
                    m = "benisse_min2" if sub.loc["benisse_min2", "status"] == "completed" else "benisse_min10"
                r = sub.loc[m]
                n_tot = int(r[total])
                if r["status"] != "completed":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f0efec", edgecolor=SURFACE, lw=2,
                                               hatch="////", alpha=0.6))
                    reason = str(r["reason"])
                    short = ("no receptor\nsequences" if "receptor" in reason else
                             "< 3 clones\n≥ 10 cells" if "fewer than 3" in reason else
                             reason.split(":")[0].replace(" after", "\nafter") if r["status"] == "stopped" else
                             "still running" if r["status"] in ("not run", "missing") else "failed")
                    ax.text(j, i, short, ha="center", va="center", fontsize=6.5, color=INK2)
                    continue
                n = int(r[what])
                frac = n / n_tot if n_tot else 0
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(frac), edgecolor=SURFACE, lw=2))
                label = f"{n:,} / {n_tot:,}\n{frac:.0%}" + (" (≥ 10)" if m == "benisse_min10" else "")
                ax.text(j, i, label, ha="center", va="center", fontsize=6.5, color="white" if frac > 0.55 else INK)
        ax.set_xlim(-0.5, len(COVERAGE_COLUMNS) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(COVERAGE_COLUMNS)), [c for _, c in COVERAGE_COLUMNS], fontsize=7)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), [DATASET[d] for d in rows] if ax is axes[0] else [""] * len(rows))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", pad=28)
    fig.suptitle("Coverage: how many of a dataset's clones each tool can represent (cell text: represented / all, share)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    save(fig, "fig1_coverage.png")


# --------------------------------------------------------------------------- 2-4. annotation recovery

def target_rows(df):
    rows = []
    for ds in ORDER:
        sub = df[df["dataset"] == ds]
        for t in sub["target"].drop_duplicates():
            rows.append((ds, t, sub.loc[sub["target"] == t, "tag"].iloc[0]))
    return rows


def delta_figure(pw, comparators, fname, title, note):
    pw = pw[pw["method"].isin(comparators)]
    if pw.empty:
        return
    comparators = [m for m in comparators if m in set(pw["method"])]
    rows = target_rows(pw)
    ypos = {}
    y, last = 0, None
    for ds, t, _ in rows:
        if last is not None and ds != last:
            y += 0.6
        ypos[(ds, t)] = y
        y += 1
        last = ds
    height = 0.19 * y + 1.8
    fig, axes = plt.subplots(1, len(comparators), figsize=(2.05 * len(comparators) + 3.2, height), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    # the axis covers 95 % of the interval ends; the few wider intervals run to the edge and end in an arrow
    ends = np.abs(pw[["delta_ci_low", "delta_ci_high"]].to_numpy()).ravel()
    lim = float(np.clip(np.ceil(np.nanquantile(ends, 0.95) * 11) / 10, 0.2, 1.2))
    for ax, m in zip(axes, comparators):
        sub = pw[pw["method"] == m]
        c = colour(m)
        for _, r in sub.iterrows():
            yy = ypos[(r["dataset"], r["target"])]
            lo, hi, d = r["delta_ci_low"], r["delta_ci_high"], r["delta_method_minus_reference"]
            ax.plot([max(lo, -lim), min(hi, lim)], [yy, yy], color=c, lw=1.2, solid_capstyle="round")
            for end, edge, mk in ((lo, -lim, "<"), (hi, lim, ">")):
                if (end < edge) if edge < 0 else (end > edge):
                    ax.plot([edge], [yy], mk, ms=3.5, color=c, clip_on=False)
            sig = lo > 0 or hi < 0
            ax.plot([np.clip(d, -lim, lim)], [yy], "o", ms=4.2, mfc=c if sig else SURFACE, mec=c, mew=1.1)
            if abs(d) > lim:                          # the estimate itself is off the axis: print it
                ax.text(np.sign(d) * lim * 0.9, yy - 0.45, f"{d:+.2f}", ha="right" if d > 0 else "left",
                        va="bottom", fontsize=6.3, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-lim, lim)
        n_better = int((sub["delta_ci_low"] > 0).sum())
        n_worse = int((sub["delta_ci_high"] < 0).sum())
        ax.set_title(f"{name(m)}", fontsize=7.5, loc="center")
        ax.text(0.5, -0.02, f"better than Threadfin: {n_better}\nworse: {n_worse}   (of {len(sub)})",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=INK2)
        ax.grid(axis="y", visible=False)
    labels = [f"{t}" for ds, t, _ in rows]
    axes[0].set_yticks([ypos[(ds, t)] for ds, t, _ in rows], labels, fontsize=7)
    axes[0].set_ylim(y - 0.4, -0.8)
    for ds in ORDER:
        ys = [ypos[(d, t)] for d, t, _ in rows if d == ds]
        if ys:
            axes[0].annotate(DATASET[ds], xy=(0, min(ys) - 0.75), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                             textcoords="offset points", ha="right", va="center", fontsize=7, fontweight="bold", color=INK)
    fig.supxlabel("R² of the tool minus R² of Threadfin on the same clones (dot: difference; line: 95 % paired bootstrap "
                  "interval, arrow: continues beyond the axis; filled: interval excludes 0)", fontsize=7.5, color=INK2, y=-0.04)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.01, -0.09, note, ha="left", va="top", fontsize=7, color=INK2, wrap=True)
    save(fig, fname)


def annotation_figures():
    pw = read("annotation_pairwise.csv")
    if pw is None:
        return
    delta_figure(pw, ["clone2vec_min2", "clonotrace_revised_min2", "clonotrace_raw_min2", "benisse_min2", "mean_pca",
                      "composition", "conga_medoid"], "fig2_annotation_forced.png",
                 "Recovering measured clone annotations: every tool forced to clones of ≥ 2 cells",
                 "Each row is one clone-level annotation that no method used (sorting gate, isotype, mutation, tissue, "
                 "measured later fate; '(RNA)' marks annotations derived from expression, which are easy by construction). "
                 "A clone's value is predicted from its 10 nearest clones in the tool's representation; values are centred "
                 "within donor. Each comparison uses the clones that the tool represents.")
    delta_figure(pw, ["clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10", "benisse_min10"],
                 "fig3_annotation_native.png",
                 "Recovering measured clone annotations: every tool at its own default (on the clones it represents)",
                 "clone2vec's default keeps clones of ≥ 3 cells, Clonotrace's ≥ 10; Threadfin is scored on the same clones. "
                 "Benisse was run at ≥ 10 cells only on the largest dataset (GSE195673).")
    delta_figure(pw, ["threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
                      "threadfin_kernel_noshrink", "mean_pca"], "fig9_ablation.png",
                 "Which parts of Threadfin's profile matter? Each variant against the default",
                 "Variants switch off one step of the default profile: the kernel mean embedding (mean profile), centring "
                 "on the donor, neighbour smoothing, shrinkage; 'mean of the embedding' switches off all of them.")
    two_cell_figure(pw)


def two_cell_figure(pw):
    comps = [m for m in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca", "composition"]
             if m in set(pw["method"])]
    sub = pw[pw["method"].isin(comps)].dropna(subset=["r2_two_cell_reference", "r2_two_cell_method"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(comps), figsize=(2.7 * len(comps), 3.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    lo = min(-0.3, float(np.floor(sub[["r2_two_cell_reference", "r2_two_cell_method"]].min().min() * 10) / 10))
    hi = max(0.6, float(np.ceil(sub[["r2_two_cell_reference", "r2_two_cell_method"]].max().max() * 10) / 10))
    for ax, m in zip(axes, comps):
        s = sub[sub["method"] == m]
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.scatter(s["r2_two_cell_reference"], s["r2_two_cell_method"], s=22, color=colour(m), edgecolor=SURFACE,
                   linewidth=1.0, zorder=3)
        above = int((s["r2_two_cell_method"] > s["r2_two_cell_reference"]).sum())
        ax.set_title(name(m), fontsize=7.5)
        ax.text(0.03, 0.97, f"tool higher: {above}\nThreadfin higher: {len(s) - above}", transform=ax.transAxes,
                va="top", fontsize=6.8, color=INK2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Threadfin: R² on two-cell clones")
    axes[0].set_ylabel("tool: R² on the same two-cell clones")
    fig.suptitle("Do two-cell clones carry information? R² of each annotation, computed on the two-cell clones only "
                 "(one dot per annotation)", x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, "fig4_two_cell_clones.png")


# --------------------------------------------------------------------------- 5. tests

def tests_figure():
    files = sorted(glob.glob(str(RESULTS / "tests" / "*.csv")))
    if not files:
        print("skip: tests/*.csv not found")
        return
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t["threadfin_reject"] = t["threadfin_p"] < 0.05
    t["scirpy_reject"] = t["scirpy_n_clones_fdr05"] > 0
    t["anova_reject"] = t["cell_anova_p"] < 0.05
    rows = [d for d in ORDER if d in set(t["dataset"])]
    tests = [("threadfin_reject", "Threadfin clonal coherence (one test per dataset)", TF, "o"),
             ("scirpy_reject", "scirpy clonotype modularity (≥ 1 clone at FDR 5 %)", INK2, "s"),
             ("anova_reject", "cell-level ANOVA (cells as replicates)", MUTED, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.38 * len(rows) + 2.1), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, scen, title in [(axes[0], "null_within_sample", "a  No clonal signal, labels shuffled within each sample"),
                            (axes[1], "null_global", "b  No clonal signal, labels shuffled across the dataset")]:
        sub = t[t["scenario"] == scen]
        for k, (col, lab, c, mk) in enumerate(tests):
            rate = sub.groupby("dataset")[col].mean()
            ys = [rows.index(d) + (k - 1) * 0.22 for d in rate.index if d in rows]
            ax.scatter([rate[d] for d in rate.index if d in rows], ys, color=c, marker=mk, s=26, label=lab,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.axvline(0.05, color=INK2, lw=0.8, ls="--")
        ax.text(0.06, -0.75, "5 %", fontsize=6.8, color=INK2, va="bottom")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(rows) - 0.5, -0.9)
        ax.set_yticks(range(len(rows)), [DATASET[d] for d in rows] if ax is axes[0] else [""] * len(rows))
        ax.set_xlabel("share of 50 replicates in which the test reports clonal structure")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5)
    # c: power of the Threadfin test when only a fraction of cells keep their real clone label
    ax = axes[2]
    ax.grid(False)
    cols = [("null_within_sample", 0.0, "0 %"), ("mix", 0.1, "10 %"), ("mix", 0.25, "25 %"), ("mix", 0.5, "50 %")]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd"] + BLUES)
    for i, d in enumerate(rows):
        for j, (scen, f, _) in enumerate(cols):
            s = t[(t["dataset"] == d) & (t["scenario"] == scen) & np.isclose(t["fraction_real_labels"], f)]
            if s.empty:
                continue
            v = float(s["threadfin_reject"].mean())
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(v), edgecolor=SURFACE, lw=2))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color="white" if v > 0.55 else INK)
        real = t[(t["dataset"] == d) & (t["scenario"] == "observed")]
        if not real.empty:                        # the test on the real labels (99 permutations: smallest P 0.01)
            ax.text(len(cols) - 0.3, i, f"P = {float(real['threadfin_p'].iloc[0]):.2f}", ha="left", va="center",
                    fontsize=6.8, color=INK)
    ax.text(len(cols) - 0.3, -0.75, "real labels", ha="left", va="center", fontsize=6.8, color=INK2)
    ax.set_xlim(-0.5, len(cols) + 0.6)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xticks(range(len(cols)), [c for *_, c in cols])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("cells that keep their real clone label\n(the rest shuffled within sample)")
    ax.set_title("c  Power of the Threadfin test\n(share of replicates that detect clonal structure)", loc="left",
                 fontsize=8.5)
    fig.suptitle("Is 'cells of one clone share a state' tested at the right error rate? Real data, clone labels shuffled",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    save(fig, "fig5_test_calibration.png")


# --------------------------------------------------------------------------- 6. simulation

SIM_METHODS = ["threadfin_kernel", "threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
               "threadfin_kernel_noshrink", "clone2vec", "clonotrace", "composition", "mean_pca", "conga_medoid"]


def simulation_figure():
    sim = read("simulation.csv")
    if sim is None:
        return
    small = sim["regime"] == "small"
    # set names as written by evaluate.py
    keep = (small & sim["set"].isin(["forced: clones >= 2 cells", "ablation: clones >= 2 cells"])) | \
           (~small & sim["set"].isin(["native: clones >= 10 cells", "ablation: clones >= 2 cells"]))
    sim = sim[keep].drop_duplicates(["dataset", "method"]).copy()
    sim["tool"] = sim["method"].str.replace(r"_min\d+$", "", regex=True).str.replace("clonotrace_revised", "clonotrace")
    sim = sim[sim["tool"].isin(SIM_METHODS)]
    label = {"clone2vec": "clone2vec", "clonotrace": "Clonotrace"}
    tool_colour = {"clone2vec": C2V, "clonotrace": CT}
    metrics = [("programme_purity", "programme purity of the 10 nearest clones", "purity_chance"),
               ("mixture_vs_intermediate_auroc", "mixture vs intermediate clones (AUROC)", None),
               ("same_donor_excess", "excess of same-donor neighbours", None)]
    regimes = [("small", 0.0, "small clones (half have 2 cells), no donor effect"),
               ("small", 1.5, "small clones, donor shift 1.5 SD"),
               ("large", 0.0, "large clones (≥ 10 cells), no donor effect"),
               ("large", 1.5, "large clones, donor shift 1.5 SD")]
    fig, axes = plt.subplots(len(regimes), len(metrics), figsize=(12.5, 2.35 * len(regimes) + 0.8), sharey=True,
                             sharex="col", constrained_layout=True)
    ypos = {m: i for i, m in enumerate(SIM_METHODS)}
    for r, (reg, b, rtitle) in enumerate(regimes):
        sub = sim[(sim["regime"] == reg) & np.isclose(sim["batch"], b)]
        for c, (col, mtitle, chance) in enumerate(metrics):
            ax = axes[r, c]
            for tool, g in sub.groupby("tool"):
                colr = tool_colour.get(tool, colour(tool))
                y = ypos[tool]
                ax.scatter(g[col], np.full(len(g), y), s=9, color=colr, alpha=0.45, linewidth=0, zorder=2)
                ax.scatter([g[col].mean()], [y], s=34, color=colr, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            if chance:
                ax.axvline(sub[chance].mean(), color=INK2, lw=0.8, ls="--")
            if col == "mixture_vs_intermediate_auroc":
                ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
            if col == "same_donor_excess":
                ax.axvline(0, color=INK2, lw=0.8, ls="--")
            ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(mtitle, fontsize=8)
            if c == 0:
                ax.set_ylabel(rtitle, fontsize=7.5)
            if r == len(regimes) - 1:
                ax.set_xlabel({"programme_purity": "share of neighbours from the same programme (dashed: chance)",
                               "mixture_vs_intermediate_auroc": "AUROC (dashed: chance)",
                               "same_donor_excess": "same-donor share minus expected (dashed: none)"}[col], fontsize=7)
    axes[0, 0].set_yticks(range(len(SIM_METHODS)), [label.get(m, name(m)) for m in SIM_METHODS])
    axes[0, 0].set_ylim(len(SIM_METHODS) - 0.5, -0.5)
    fig.suptitle("Simulated repertoires with a known answer (large dot: mean of 3 seeds; small dots: each seed)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    save(fig, "fig6_simulation.png")


# --------------------------------------------------------------------------- 7-8. split halves

def strip_by_method(df, metrics, methods, fname, title, xlabels, chance_cols=None):
    df = df[df["method"].isin(methods)]
    if df.empty:
        return
    methods = [m for m in methods if m in set(df["method"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics) + 1.5, 0.36 * len(methods) + 1.5),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, col, xl in zip(axes, metrics, xlabels):
        per = df.groupby(["method", "dataset"])[col].mean().reset_index()
        for i, m in enumerate(methods):
            v = per.loc[per["method"] == m, col]
            ax.scatter(v, np.full(len(v), i), s=11, color=colour(m), alpha=0.5, linewidth=0, zorder=2)
            if len(v):
                ax.scatter([v.median()], [i], s=40, marker="D", color=colour(m), edgecolor=SURFACE, linewidth=1, zorder=3)
        if chance_cols and chance_cols.get(col):
            ax.axvline(df[chance_cols[col]].median(), color=INK2, lw=0.8, ls="--")
        ax.set_xlabel(xl, fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(methods)), [name(m) for m in methods])
    axes[0].set_ylim(len(methods) - 0.5, -0.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    save(fig, fname)


def halves_figures():
    rep = read("split_half_reproducibility.csv")
    methods = ["threadfin_kernel", "threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
               "threadfin_kernel_noshrink", "clone2vec_min2", "clonotrace_revised_min2", "clonotrace_raw_min2",
               "benisse_min2", "mean_pca", "composition", "conga_medoid"]
    if rep is not None:
        rep = rep[rep["set"].str.startswith("forced") | rep["set"].str.startswith("ablation")]
        rep = rep.drop_duplicates(["dataset", "method", "seed"])
        strip_by_method(rep, ["distance_spearman", "knn_overlap", "reid_top1"], methods, "fig7_split_half_reproducibility.png",
                        "Same clones, disjoint cells: is a clone's representation reproducible? "
                        "(small dots: datasets; diamond: median)",
                        ["Spearman correlation of clone-clone distances\nbetween the two halves",
                         "overlap of the 10 nearest clones (dashed: chance)",
                         "clone re-identified as its own best match\nacross halves (dashed: chance)"],
                        {"knn_overlap": "knn_overlap_chance", "reid_top1": "reid_chance_top1"})
    dis = read("annotation_recovery_disjoint.csv")
    if dis is not None:
        dis = dis[dis["set"].str.startswith("forced") | dis["set"].str.startswith("ablation")]
        dis = dis.drop_duplicates(["dataset", "target", "method", "seed", "representation_half"])
        mean = dis.groupby(["dataset", "target", "method"])["r2"].mean().reset_index()
        ref = mean[mean["method"] == "threadfin_kernel"].set_index(["dataset", "target"])["r2"]
        mean["delta"] = mean["r2"] - ref.reindex(pd.MultiIndex.from_frame(mean[["dataset", "target"]])).to_numpy()
        mean = mean[mean["method"] != "threadfin_kernel"]
        strip_by_method(mean, ["delta"], [m for m in methods if m != "threadfin_kernel"],
                        "fig8_annotation_disjoint.png",
                        "Annotation of one half of the cells predicted from a representation of the other half: "
                        "each tool minus Threadfin",
                        ["R² of the tool minus R² of Threadfin (dots: annotations; diamond: median; >0: tool better)"])


# --------------------------------------------------------------------------- 10. LARRY linking

def larry_figure():
    lk = read("larry_clone_linking.csv")
    if lk is None or lk.empty:
        return
    lk = lk[lk["set"].str.startswith("forced")].copy()
    if lk.empty:
        return
    lk = lk.sort_values("mrr", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.34 * len(lk) + 1.5), sharey=True, constrained_layout=True)
    for ax, col, xl, chance in [(axes[0], "top1", "share of day-4 clones whose own day-6 clone is the nearest",
                                 "chance_top1"),
                                (axes[1], "mrr", "mean reciprocal rank of the own day-6 clone", None)]:
        for i, (_, r) in enumerate(lk.iterrows()):
            ax.plot([0, r[col]], [i, i], color=colour(r["method"]), lw=2.2, solid_capstyle="round")
            ax.plot([r[col]], [i], "o", ms=5, color=colour(r["method"]))
            ax.text(r[col], i, f"  {r[col]:.3f}", va="center", fontsize=7, color=INK)
        if chance:
            ax.axvline(lk[chance].iloc[0], color=INK2, ls="--", lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(lk)), [name(m) for m in lk["method"]])
    n = int(lk["n_query_clones"].iloc[0])
    k = int(lk["n_candidate_day6_clones"].iloc[0])
    fig.suptitle(f"LARRY: find each barcode's day-6 clone from its day-4 clone (clones of ≥ 2 cells kept by every tool: "
                 f"{n:,} queries among {k:,} day-6 clones; dashed: chance)", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold")
    save(fig, "fig10_larry_linking.png")


# --------------------------------------------------------------------------- 11. runtime and memory

RES_PANELS = [("clone2vec_min3", "clone2vec\n(default, ≥ 3 cells)", C2V), ("clone2vec_min2", "clone2vec\n(forced to ≥ 2 cells)", C2V),
              ("clonotrace_revised_min10", "Clonotrace\n(default, ≥ 10 cells)", CT),
              ("clonotrace_revised_min2", "Clonotrace\n(forced to ≥ 2 cells)", CT), ("benisse", "Benisse\n(≥ 2 cells)", BN)]
# the core-count and data-size experiments ran each program once; the default and forced settings share it
SAME_PROGRAM = {"clone2vec_min3": "clone2vec_min2", "clone2vec_min2": "clone2vec_min2",
                "clonotrace_revised_min10": "clonotrace_revised_min2", "clonotrace_revised_min2": "clonotrace_revised_min2",
                "benisse": "benisse_min2"}
STOP_SECONDS = 72 * 3600


def _run(sub, key):
    if key == "benisse":
        r = sub.loc["benisse_min2"]
        return sub.loc["benisse_min10"] if r["status"] != "completed" and sub.loc["benisse_min10", "status"] == "completed" else r
    return sub.loc[key]


def _not_run_note(r):
    reason = str(r["reason"])
    return ("stopped at 72 h" if r["status"] == "stopped" else "failed" if r["status"] == "failed" else
            "no BCR" if "receptor" in reason else "< 3 clones ≥ 10 cells" if "fewer than 3" in reason else "not run")


def resource_rows():
    """Rows of the computing-resource figure: (block, label, {panel: (ratio, note, hollow)})."""
    cov, res = read("coverage_runtime.csv"), read("resources.csv")
    cov = cov[cov["dataset"].isin(ORDER)]
    out = []
    for block, what in (("Run time on each dataset (4 cores)", "time"), ("Peak memory on each dataset", "memory")):
        for ds in ORDER:
            sub = cov[cov["dataset"] == ds].set_index("method")
            if sub.empty:
                continue
            tf = sub.loc["threadfin_kernel"]
            cells = {}
            for key, *_ in RES_PANELS:
                r = _run(sub, key)
                hollow = key.startswith("clonotrace") and r.get("distance_workers") == 1
                if r["status"] == "completed":
                    if what == "time":
                        v = (r["seconds_method"] if pd.notna(r["seconds_method"]) else r["wall_seconds_task"]) / tf["seconds_method"]
                    else:
                        v = r["peak_rss_gb_task"] / tf["peak_rss_gb_method"]
                    note = "≥ 10 cells" if r.name == "benisse_min10" else ""
                    cells[key] = (float(v), note, hollow)
                elif r["status"] == "stopped" and what == "time":
                    cells[key] = (STOP_SECONDS / tf["seconds_method"], "stopped at 72 h", False)
                else:
                    cells[key] = (np.nan, _not_run_note(r), False)
            out.append((block, DATASET[ds], cells))
    if res is not None:
        c = res[res["experiment"].eq("cores")]
        for n in sorted(c["cores"].unique()):
            at = c[c["cores"].eq(n)].set_index("tool")
            if "threadfin_run" not in at.index:
                continue
            cells = {}
            for key, *_ in RES_PANELS:
                t = SAME_PROGRAM[key]
                if t in at.index and at.loc[t, "status"] == "completed":
                    cells[key] = (float(at.loc[t, "seconds"] / at.loc["threadfin_run", "seconds"]), "", False)
                else:
                    planned = n in ((1, 4, 16) if t == "benisse_min2" else (1, 2, 4, 8, 16))
                    cells[key] = (np.nan, "running" if t in at.index else "pending" if planned else "not planned", False)
            out.append(("Run time with 1 to 16 cores (GSE287123 RBD; Threadfin: its complete analysis)",
                        f"{n} core" + ("s" if n > 1 else ""), cells))
        g = res[res["experiment"].eq("size")]
        for ds in sorted(g["dataset"].unique(), key=lambda d: float(d.split("_f")[1])):
            at = g[g["dataset"].eq(ds)].set_index("tool")
            if "threadfin_kernel" not in at.index:
                continue
            cells = {}
            for key, *_ in RES_PANELS:
                t = SAME_PROGRAM[key]
                if t in at.index and at.loc[t, "status"] == "completed":
                    cells[key] = (float(at.loc[t, "seconds"] / at.loc["threadfin_kernel", "seconds"]), "", False)
                else:
                    cells[key] = (np.nan, "running" if t in at.index else "not run", False)
            frac = float(ds.split("_f")[1])
            out.append(("Run time as the data grow (random subsets of GSE195673, 8 cores)",
                        f"{int(at['n_clones_ge2'].iloc[0]):,} clones ({100 * frac:g} %)", cells))
    return out


def runtime_figure():
    if read("coverage_runtime.csv") is None:
        return
    rows = resource_rows()
    blocks = list(dict.fromkeys(b for b, *_ in rows))
    ypos, labels, y = [], [], 0
    heads = []
    for b in blocks:
        heads.append((y, b))
        y += 1
        for blk, lab, cells in rows:
            if blk == b:
                ypos.append(y)
                labels.append(lab)
                y += 1
        y += 0.4
    fig, axes = plt.subplots(1, len(RES_PANELS), figsize=(14.5, 0.25 * y + 2.4), sharey=True, layout="constrained")
    lo, hi = 0.05, 3e5
    for ax, (key, title, col) in zip(axes, RES_PANELS):
        counts = {b: [0, 0, 0] for b in blocks}             # faster, slower, missing
        for (blk, lab, cells), yy in zip(rows, ypos):
            v, note, hollow = cells[key]
            if np.isfinite(v):
                ax.scatter([min(max(v, lo), hi)], [yy], s=30, zorder=3, color=SURFACE if hollow else col,
                           edgecolor=col, linewidth=1.2)
                if note:
                    ax.text(min(v, hi) * 1.6, yy, note, fontsize=6.2, color=INK2, va="center")
                if note.startswith("stopped"):
                    ax.annotate("", xy=(min(v, hi) * 1.5, yy), xytext=(min(v, hi), yy),
                                arrowprops=dict(arrowstyle="->", color=col, lw=1))
                counts[blk][0 if v < 1 / 1.25 else 1 if v > 1.25 else 2] += 0 if 1 / 1.25 <= v <= 1.25 else 1
            else:
                ax.text(np.sqrt(lo * hi), yy, note, fontsize=6.3,
                        color=MUTED if note in ("not run", "no BCR", "not planned", "pending", "running") else "#c2511f",
                        va="center", ha="center", style="italic")
                if note not in ("no BCR", "< 3 clones ≥ 10 cells", "not run", "running", "pending", "not planned"):
                    counts[blk][2] += 1
        ax.axvline(1, color=INK2, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=8.2)
        ax.grid(axis="y", visible=False)
        ax.text(0.75, -0.55, "← less", fontsize=6.6, color=INK2, ha="right", va="center")
        ax.text(1.35, -0.55, "more →", fontsize=6.6, color=INK2, ha="left", va="center")
        foot = []
        for b, short in zip(blocks, ("time", "memory", "1-16 cores", "data size")):
            f, sl, miss = counts[b]
            foot.append(f"{short}: {'less' if short == 'memory' else 'faster'} {f}, {'more' if short == 'memory' else 'slower'} {sl}"
                        + (f", failed or stopped {miss}" if miss else ""))
        ax.set_xlabel("times Threadfin (log scale)\n\n" + "\n".join(foot), fontsize=7)
    axes[0].set_yticks(ypos, labels, fontsize=7)
    for yy, b in heads:
        axes[0].text(-0.02, yy, b, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=7.6,
                     fontweight="bold")
    axes[0].set_ylim(y - 0.4, -1.4)
    fig.suptitle("Computing resources, each tool relative to Threadfin: time and memory per dataset, gain from more "
                 "cores and growth with the data (1 = same as Threadfin)", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold")
    stopped = any(note.startswith("stopped") for _, _, cells in rows for _, note, _ in cells.values())
    cov = read("coverage_runtime.csv")
    many = cov[cov["method"].str.startswith("clonotrace_revised") & cov["dataset"].isin(ORDER)
               & cov["status"].eq("completed") & cov["distance_workers"].gt(4)]
    many = "; ".join(f"{DATASET[r.dataset].split(' (')[0].split(',')[0]}, ≥ {10 if r.method.endswith('min10') else 2} "
                     f"cells ({int(r.distance_workers)})" for r in many.itertuples())
    fig.text(0.01, -0.01, "Hollow: Clonotrace runs whose clone distances used one R session (slower than its parallel "
             "plan). Counts at the bottom: differences of less than 25 % are not counted."
             + (" A run stopped at 72 h is drawn at 72 h; the arrow marks a lower bound." if stopped else "")
             + (f"\nClonotrace runs whose clone distances used more than 4 R sessions: {many}." if many else ""),
             fontsize=7, color=INK2, va="top")
    save(fig, "fig11_runtime_memory.png")


def scaling_figure():
    """Absolute numbers behind fig11: speed-up with cores, time and memory against the number of clones, and
    where the time of the complete Threadfin analysis goes."""
    res = read("resources.csv")
    if res is None:
        return
    tools = [("threadfin_run", "Threadfin, complete analysis", TF, "o", "-"),
             ("threadfin_kernel", "Threadfin, clone profiles only", TF, "o", "--"),
             ("clone2vec_min2", "clone2vec", C2V, "s", "-"), ("clonotrace_revised_min2", "Clonotrace", CT, "v", "-"),
             ("benisse_min2", "Benisse", BN, "D", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), layout="constrained")
    c = res[res["experiment"].eq("cores") & res["status"].eq("completed")]
    ax = axes[0]
    for t, lab, col, mk, ls in tools:
        s = c[c["tool"].eq(t)].sort_values("cores")
        if len(s) < 2:
            continue
        base = s.iloc[0]
        ax.plot(s["cores"], base["seconds"] / s["seconds"], marker=mk, color=col, ls=ls, label=lab + (
            f" (vs {int(base['cores'])} cores)" if base["cores"] != 1 else ""))
    ax.plot([1, 16], [1, 16], color=MUTED, lw=0.8, ls=":", label="perfect scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16], ["¼", "½", "1", "2", "4", "8", "16"])
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("speed-up over the fewest cores")
    ax.set_title("a  More cores (GSE287123 RBD, 1,887 clones)", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5)
    g = res[res["experiment"].eq("size") & res["status"].eq("completed")]
    for ax, col_name, ylab, title in ((axes[1], "seconds", "seconds (log scale)", "b  Time as the data grow"),
                                      (axes[2], "peak_rss_gb", "peak memory, GB (log scale)", "c  Memory as the data grow")):
        for t, lab, col, mk, ls in tools:
            s = g[g["tool"].eq(t)].sort_values("n_clones_ge2")
            if s.empty:
                continue
            ax.plot(s["n_clones_ge2"], s[col_name], marker=mk, color=col, ls=ls, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([500, 1000, 2000, 5000], ["500", "1,000", "2,000", "5,000"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("clones with ≥ 2 cells (random subsets of GSE195673, 8 cores)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[1].legend(fontsize=6.5)
    fig.suptitle("Scaling: how the tools use more cores and how their cost grows with the number of clones",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    save(fig, "fig12_scaling.png")


SENS = [("threadfin_kernel_rff128", "128 random features (default 256)"),
        ("threadfin_kernel_rff512", "512 random features"), ("threadfin_kernel_rff1024", "1,024 random features"),
        ("threadfin_kernel_bw0.5", "bandwidth × 0.5 (default: median distance)"), ("threadfin_kernel_bw2", "bandwidth × 2"),
        ("threadfin_kernel_smooth5", "smoothing over 5 cells (default 15)"),
        ("threadfin_kernel_smooth30", "smoothing over 30 cells"), ("threadfin_kernel_nosmooth", "no smoothing"),
        ("threadfin_kernel_pc10", "10 principal components (default 30)"), ("threadfin_kernel_pc50", "50 principal components"),
        ("threadfin_kernel_seed1", "random seed 1 (default 0)"), ("threadfin_kernel_seed2", "random seed 2")]


def sensitivity_figure():
    """How much do Threadfin's results move when its settings change?"""
    pw, rep, sim = read("annotation_pairwise.csv"), read("split_half_reproducibility.csv"), read("simulation.csv")
    if pw is None or rep is None or sim is None:
        return
    keys = [k for k, _ in SENS]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.36 * len(SENS) + 2.0), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    y = {k: i for i, k in enumerate(keys)}
    a = pw[pw["method"].isin(keys) & (pw["tag"] != "rna")]
    ax = axes[0]
    for k in keys:
        d = a[a["method"].eq(k)]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        sig = (d["delta_ci_low"] > 0) | (d["delta_ci_high"] < 0)
        ax.scatter(d["delta_method_minus_reference"], y[k] + jitter, s=9, color=np.where(sig, TF, "#b9c9de"), lw=0, zorder=2)
        ax.plot([d["delta_method_minus_reference"].median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.6, zorder=3)
        ax.text(0.205, y[k], f"{int((d['delta_ci_low'] > 0).sum())} ▲  {int((d['delta_ci_high'] < 0).sum())} ▼",
                fontsize=6.6, va="center", color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.2, 0.26)
    ax.set_xlabel("R² of the setting − R² of the default, per measured annotation\n(dark: 95 % interval excludes 0; "
                  "bar: median; ▲ ▼ annotations better / worse)")
    ax.set_title("a  Recovering measured annotations (35)", loc="left", fontsize=8.5)
    r = rep[rep["set"].str.startswith("sensitivity")].groupby(["dataset", "method"])["reid_top1"].mean().unstack()
    ax = axes[1]
    for k in keys:
        if k not in r:
            continue
        d = (r[k] - r["threadfin_kernel"]).dropna()
        ax.scatter(100 * d, [y[k]] * len(d), s=12, color=TF, alpha=0.7, lw=0)
        ax.plot([100 * d.median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("re-identified clones, percentage points\n(setting − default; one dot per dataset)")
    ax.set_title("b  Recognising a clone from other cells", loc="left", fontsize=8.5)
    ss = sim[sim["set"].str.startswith("sensitivity")]
    for ax, col, title in ((axes[2], "programme_purity", "c  Simulation: planted clone groups"),
                           (axes[3], "mixture_vs_intermediate_auroc", "d  Simulation: mixed vs in-between")):
        sub = ss if col == "programme_purity" else ss[ss["regime"].eq("large")]
        m = sub.pivot_table(index="dataset", columns="method", values=col, aggfunc="first")
        for k in keys:
            if k not in m:
                continue
            d = (m[k] - m["threadfin_kernel"]).dropna()
            ax.scatter(d, [y[k]] * len(d), s=12, color=TF, alpha=0.7, lw=0)
            ax.plot([d.median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(("purity" if col == "programme_purity" else "AUROC") + " of the setting − default\n"
                      "(one dot per simulated dataset" + ("" if col == "programme_purity" else ", clones ≥ 10 cells") + ")")
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].set_yticks(range(len(keys)), [lab for _, lab in SENS])
    axes[0].set_ylim(len(keys) - 0.5, -0.6)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Sensitivity of Threadfin to its settings: each setting changed alone, everything else at the default",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    save(fig, "fig13_sensitivity.png")


def main():
    coverage_figure()
    annotation_figures()
    tests_figure()
    simulation_figure()
    halves_figures()
    larry_figure()
    runtime_figure()
    scaling_figure()
    sensitivity_figure()


if __name__ == "__main__":
    main()
