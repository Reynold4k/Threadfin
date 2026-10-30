#!/usr/bin/env python
"""Draw the benchmark figures (``manuscript/benchmarking/figures/``) from the tables in ``results/``.

Every figure is drawn only from the CSV files that ``evaluate.py``, ``run_tests.py`` and ``record_run_hosts.py``
write, so the figures can be redrawn without re-running any method. A figure whose table is missing is skipped
with a message.

Print layout: every figure is at most 180 mm wide (one page width) with no text smaller than 5 pt at that size;
``save`` checks both and writes a 300-dpi PNG and a PDF with embedded fonts. Figures are numbered in the order
in which the README shows them (FIG). Their legends, with the numbers behind them, are in the README.

Colours follow the tool, never its rank: Threadfin (all variants) blue, clone2vec orange, Clonotrace aqua,
Benisse yellow, the simple baselines grey. Every mark is also named on an axis or in a legend, so colour is
never the only cue.

Usage: python make_figures.py
"""
from __future__ import annotations

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from bench_paths import FIGURES, RESULTS  # noqa: E402

MM = 1 / 25.4                     # inches per millimetre
WIDTH = 180 * MM                  # one page width
MIN_FONT = 5.0
# figure numbers in the order in which the README shows them
FIG = {"overview": "fig1_overview", "coverage": "fig2_coverage", "two_cell": "fig3_two_cell_clones",
       "forced": "fig4_annotation_forced", "native": "fig5_annotation_native", "disjoint": "fig6_annotation_disjoint",
       "reproducibility": "fig7_split_half_reproducibility", "larry": "fig8_larry_linking",
       "tests": "fig9_test_calibration", "simulation": "fig10_simulation", "ablation": "fig11_ablation",
       "resources": "fig12_runtime_memory", "scaling": "fig13_scaling", "sensitivity": "fig14_sensitivity"}

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
TF, C2V, CT, BN, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 6, "axes.titlesize": 6,
    "axes.titleweight": "bold", "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.linewidth": 0.5, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2, "axes.titlepad": 3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
    "lines.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

METHOD = {
    "threadfin_kernel": ("Threadfin (default)", TF),
    "threadfin_mean": ("Threadfin, mean profile", TF),
    "threadfin_kernel_nocentre": ("Threadfin, no centring", TF),
    "threadfin_kernel_nosmooth": ("Threadfin, no smoothing", TF),
    "threadfin_kernel_noshrink": ("Threadfin, no shrinkage", TF),
    "threadfin_kernel_seed1": ("Threadfin, another seed (control)", TF),
    "threadfin_kernel_seed2": ("Threadfin, seed 2 (control)", TF),
    "threadfin_kernel_seed1_ge3": ("Threadfin, another seed, ≥ 3 cells (control)", TF),
    "threadfin_kernel_seed1_ge10": ("Threadfin, another seed, ≥ 10 cells (control)", TF),
    "threadfin_kernel_seed2_ge3": ("Threadfin, seed 2, ≥ 3 cells (control)", TF),
    "threadfin_kernel_seed2_ge10": ("Threadfin, seed 2, ≥ 10 cells (control)", TF),
    "clone2vec_min3": ("clone2vec (default, ≥ 3 cells)", C2V),
    "clone2vec_min2": ("clone2vec (forced to ≥ 2 cells)", C2V),
    "clonotrace_revised_min10": ("Clonotrace (default, ≥ 10 cells)", CT),
    "clonotrace_raw_min10": ("Clonotrace, published distance (≥ 10)", CT),
    "clonotrace_revised_min2": ("Clonotrace (forced to ≥ 2 cells)", CT),
    "clonotrace_raw_min2": ("Clonotrace, published distance (≥ 2)", CT),
    "clonotrace_exact_min2": ("Clonotrace, exact OT (≥ 2)", CT),
    "clonotrace_exact_raw_min2": ("Clonotrace, exact OT, published (≥ 2)", CT),
    "clonotrace_exact_min10": ("Clonotrace, exact OT (≥ 10)", CT),
    "clonotrace_exact_raw_min10": ("Clonotrace, exact OT, published (≥ 10)", CT),
    "benisse_min2": ("Benisse", BN),
    "benisse_min10": ("Benisse (≥ 10 cells)", BN),
    "mean_pca": ("Mean of the embedding (baseline)", BASE),
    "composition": ("Cluster composition (baseline)", BASE),
    "conga_medoid": ("Medoid cell, CoNGA's rule (baseline)", BASE),
}
# short two-line names for panel titles
SHORT_METHOD = {
    "clone2vec_min2": "clone2vec\nforced to\n≥ 2 cells", "clone2vec_min3": "clone2vec\ndefault\n≥ 3 cells",
    "clonotrace_revised_min2": "Clonotrace\nforced to\n≥ 2 cells", "clonotrace_raw_min2": "Clonotrace,\npublished\ndistance",
    "clonotrace_revised_min10": "Clonotrace\ndefault\n≥ 10 cells", "clonotrace_raw_min10": "Clonotrace,\npublished,\n≥ 10 cells",
    "benisse_min2": "Benisse\n≥ 2 cells", "benisse_min10": "Benisse\n≥ 10 cells\n(GSE195673)",
    "mean_pca": "Mean of the\nembedding", "composition": "Cluster\ncomposition", "conga_medoid": "Medoid cell\n(CoNGA's\nrule)",
    "threadfin_mean": "Threadfin\nmean\nprofile", "threadfin_kernel_nocentre": "Threadfin\nno\ncentring",
    "threadfin_kernel_nosmooth": "Threadfin\nno\nsmoothing", "threadfin_kernel_noshrink": "Threadfin\nno\nshrinkage",
    "threadfin_kernel_seed1": "Control:\nThreadfin,\nother seeds", "threadfin_kernel_seed1_ge3": "Control:\nother seeds\n≥ 3 cells",
    "threadfin_kernel_seed1_ge10": "Control:\nother seeds\n≥ 10 cells",
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
SHORT_DS = {
    "gut_gc_reporter": "Gut GC reporter", "mouse_rbd": "RBD reporter", "mouse_np": "NP reporter",
    "gc_np_pc": "GC + plasma cells", "malaria_late": "Plasmodium d10-42", "malaria": "Plasmodium d0-14",
    "ln_vaccine": "Human LN vaccine", "spont_gc": "Spontaneous GC", "covid_bcell": "Human COVID-19",
    "larry_d2": "LARRY d2", "larry_d46": "LARRY d4/d6",
}
ORDER = list(DATASET)
CALLS = {"better": "tool better", "worse": "Threadfin better"}


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


def save(fig, stem):
    """Write <stem>.png (300 dpi) and <stem>.pdf; report the printed width and the smallest font."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    with Image.open(FIGURES / f"{stem}.png") as im:
        w_mm, h_mm = (v / 300 * 25.4 for v in im.size)
    flag = "" if w_mm <= 180.5 and min(sizes) >= MIN_FONT else "  <-- CHECK"
    print(f"wrote figures/{stem}.png: {w_mm:.0f} x {h_mm:.0f} mm, smallest text {min(sizes):g} pt{flag}")


def call_of(r):
    """better / worse / similar / unresolved for a row of the pairwise table."""
    c = r.get("call") if hasattr(r, "get") else None
    if isinstance(c, str):
        return c
    return "better" if r["delta_ci_low"] > 0 else "worse" if r["delta_ci_high"] < 0 else "similar"


def panel_letter(fig, ax, letter, dx=-6, dy=4):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, fontweight="bold")


# --------------------------------------------------------------------------- 2. coverage

COVERAGE_COLUMNS = [("threadfin_kernel", "Threadfin"), ("clone2vec_min3", "clone2vec\ndefault ≥ 3"),
                    ("clone2vec_min2", "clone2vec\nforced ≥ 2"), ("clonotrace_revised_min10", "Clonotrace\ndefault ≥ 10"),
                    ("clonotrace_revised_min2", "Clonotrace\nforced ≥ 2"), ("benisse", "Benisse")]


def coverage_figure():
    cov = read("coverage_runtime.csv")
    if cov is None:
        return
    cov = cov[cov["dataset"].isin(ORDER)]
    rows = [d for d in ORDER if d in set(cov["dataset"])]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd"] + BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, (5.2 * len(rows) + 16) * MM), layout="constrained",
                             gridspec_kw={"wspace": 0.02})
    for ax, (what, total, title, letter) in zip(axes, [
            ("n_clones_represented", "n_clones_ge2", "Clones with ≥ 2 cells", "a"),
            ("n_two_cell_clones_represented", "n_two_cell_clones", "Clones of exactly 2 cells", "b")]):
        ax.grid(False)
        for i, ds in enumerate(rows):
            sub = cov[cov["dataset"] == ds].set_index("method")
            for j, (m, _) in enumerate(COVERAGE_COLUMNS):
                if m == "benisse":
                    m = "benisse_min2" if sub.loc["benisse_min2", "status"] == "completed" else "benisse_min10"
                r = sub.loc[m]
                n_tot = int(r[total])
                if r["status"] != "completed":
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f0efec", edgecolor="white", lw=1.5,
                                               hatch="////", alpha=0.6))
                    reason = str(r["reason"])
                    short = ("no receptor\nsequences" if "receptor" in reason else
                             "< 3 clones\n≥ 10 cells" if "fewer than 3" in reason else "failed")
                    ax.text(j, i, short, ha="center", va="center", fontsize=5, color=INK2, linespacing=1.0)
                    continue
                n = int(r[what])
                frac = n / n_tot if n_tot else 0
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(frac), edgecolor="white", lw=1.5))
                label = f"{n:,}\n{frac:.0%}" + (" (≥ 10)" if m == "benisse_min10" else "")
                ax.text(j, i, label, ha="center", va="center", fontsize=5, color="white" if frac > 0.55 else INK,
                        linespacing=1.0)
        ax.set_xlim(-0.5, len(COVERAGE_COLUMNS) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(COVERAGE_COLUMNS)), [c for _, c in COVERAGE_COLUMNS], fontsize=5)
        ax.xaxis.tick_top()
        tot = cov.drop_duplicates("dataset").set_index("dataset")
        ax.set_yticks(range(len(rows)), [f"{SHORT_DS[d]} ({int(tot.loc[d, total]):,})" for d in rows])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=6.5)
    save(fig, FIG["coverage"])


# --------------------------------------------------------------------------- 3-5, 11. annotation recovery

def target_rows(df):
    rows = []
    for ds in ORDER:
        sub = df[df["dataset"] == ds]
        for t in sub["target"].drop_duplicates():
            rows.append((ds, t, sub.loc[sub["target"] == t, "tag"].iloc[0]))
    return rows


def short_target(t):
    return t.replace("day-6 neutrophil share (of neu + mono)", "day-6 neutrophil share").replace(
        "recently divided (mCherry-low)", "recently divided").replace("GC-fate sort (tdTomato+)", "GC-fate sort")


def delta_figure(pw, comparators, stem, controls=None):
    """One panel per comparator: R² of the tool minus R² of Threadfin per annotation, with its 95 % paired bootstrap
    interval. ``controls`` (id -> [(method, offset)]) adds a negative-control panel of Threadfin with other seeds."""
    controls = controls or {}
    want = list(comparators) + [m for ms in controls.values() for m, _ in ms]
    pw = pw[pw["method"].isin(want)]
    if pw.empty:
        return
    comparators = [m for m in comparators if m in set(pw["method"])] + list(controls)
    rows = target_rows(pw[pw["method"].isin([c for c in comparators if c not in controls])])
    ypos, heads = {}, []
    y, last = 0.0, None
    for ds, t, _ in rows:
        if ds != last:
            y += 0.4 if last is not None else 0
            heads.append((y, ds))
            y += 1.0
        ypos[(ds, t)] = y
        y += 1
        last = ds
    height = (2.05 * y + 30) * MM
    n = len(comparators)
    fig, axes = plt.subplots(1, n, figsize=(WIDTH, height), sharey=True,
                             gridspec_kw={"wspace": 0.12, "left": 0.21, "right": 0.995, "top": 1 - 14 * MM / height,
                                          "bottom": 13 * MM / height})
    axes = np.atleast_1d(axes)
    ends = np.abs(pw[["delta_ci_low", "delta_ci_high"]].to_numpy()).ravel()
    lim = float(np.clip(np.ceil(np.nanquantile(ends, 0.95) * 11) / 10, 0.2, 1.2))
    for ax, m in zip(axes, comparators):
        members = controls.get(m, [(m, 0.0)])
        n_b = n_w = n_u = n_all = 0
        for mm, off in members:
            sub = pw[pw["method"] == mm]
            c = colour(mm)
            for _, r in sub.iterrows():
                if (r["dataset"], r["target"]) not in ypos:
                    continue
                yy = ypos[(r["dataset"], r["target"])] + off
                lo, hi, d = r["delta_ci_low"], r["delta_ci_high"], r["delta_method_minus_reference"]
                ax.plot([max(lo, -lim), min(hi, lim)], [yy, yy], color=c, lw=0.7, solid_capstyle="round")
                for end, edge, mk in ((lo, -lim, "<"), (hi, lim, ">")):
                    if (end < edge) if edge < 0 else (end > edge):
                        ax.plot([edge], [yy], mk, ms=2.2, color=c, clip_on=False)
                cl = call_of(r)
                mk = "D" if cl == "unresolved" else "o"
                ax.plot([np.clip(d, -lim, lim)], [yy], mk, ms=2.6 if mk == "o" else 2.3,
                        mfc=c if cl in ("better", "worse") else "white", mec=c, mew=0.7)
                n_b += cl == "better"
                n_w += cl == "worse"
                n_u += cl == "unresolved"
                n_all += 1
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        ax.set_title(SHORT_METHOD.get(m, name(m)), fontsize=5, loc="center", linespacing=1.05,
                     color=INK2 if m in controls else INK, fontstyle="italic" if m in controls else "normal")
        ax.annotate(f"{n_b} better\n{n_w} worse\nof {n_all}" + (f"\n{n_u} unresolved" if n_u else ""),
                    xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=5, color=INK2, linespacing=1.1)
        ax.tick_params(axis="x", labelsize=5, pad=1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks([ypos[(ds, t)] for ds, t, _ in rows], [short_target(t) for _, t, _ in rows], fontsize=5)
    axes[0].set_ylim(y - 0.3, -0.7)
    for yy, ds in heads:
        axes[0].annotate(SHORT_DS[ds], xy=(0, yy), xycoords=("axes fraction", "data"), xytext=(-3, 0),
                         textcoords="offset points", ha="right", va="center", fontsize=5.5, fontweight="bold")
    axes[0].tick_params(axis="y", length=0)
    save(fig, stem)


def annotation_figures():
    pw = read("annotation_pairwise.csv")
    if pw is None:
        return
    control_all = {"threadfin_kernel_seed1": [("threadfin_kernel_seed1", -0.18), ("threadfin_kernel_seed2", 0.18)]}
    delta_figure(pw, ["clone2vec_min2", "clonotrace_revised_min2", "clonotrace_raw_min2", "benisse_min2", "mean_pca",
                      "composition", "conga_medoid"], FIG["forced"], control_all)
    delta_figure(pw, ["clone2vec_min3", "clonotrace_revised_min10", "clonotrace_raw_min10", "benisse_min10"],
                 FIG["native"],
                 {"threadfin_kernel_seed1_ge3": [("threadfin_kernel_seed1_ge3", -0.18), ("threadfin_kernel_seed2_ge3", 0.18)],
                  "threadfin_kernel_seed1_ge10": [("threadfin_kernel_seed1_ge10", -0.18),
                                                  ("threadfin_kernel_seed2_ge10", 0.18)]})
    delta_figure(pw, ["threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
                      "threadfin_kernel_noshrink", "mean_pca"], FIG["ablation"], control_all)
    two_cell_figure(pw)


def two_cell_figure(pw):
    comps = [m for m in ["clone2vec_min2", "clonotrace_revised_min2", "benisse_min2", "mean_pca", "composition"]
             if m in set(pw["method"])]
    sub = pw[pw["method"].isin(comps)].dropna(subset=["r2_two_cell_reference", "r2_two_cell_method"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, len(comps), figsize=(WIDTH, 50 * MM), layout="constrained")
    axes = np.atleast_1d(axes)
    cols = ["r2_two_cell_reference_ci_low", "r2_two_cell_reference_ci_high", "r2_two_cell_method_ci_low",
            "r2_two_cell_method_ci_high"]
    have_ci = all(c in sub for c in cols)
    vals = sub[["r2_two_cell_reference", "r2_two_cell_method"]].to_numpy()
    lo = min(-0.4, float(np.floor(np.nanmin(vals) * 10) / 10))
    hi = max(0.8, float(np.ceil(np.nanmax(vals) * 10) / 10))
    for i, (ax, m) in enumerate(zip(axes, comps)):
        s = sub[sub["method"] == m]
        c = colour(m)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.axvline(0, color=GRID, lw=0.6)
        if have_ci:
            ax.errorbar(s["r2_two_cell_reference"], s["r2_two_cell_method"],
                        xerr=[s["r2_two_cell_reference"] - s["r2_two_cell_reference_ci_low"].clip(lower=lo),
                              s["r2_two_cell_reference_ci_high"].clip(upper=hi) - s["r2_two_cell_reference"]],
                        yerr=[s["r2_two_cell_method"] - s["r2_two_cell_method_ci_low"].clip(lower=lo),
                              s["r2_two_cell_method_ci_high"].clip(upper=hi) - s["r2_two_cell_method"]],
                        fmt="none", ecolor=c, elinewidth=0.4, alpha=0.45, zorder=2)
        filled = s["call_two_cell"].isin(["better", "worse"]) if "call_two_cell" in s else np.zeros(len(s), bool)
        ax.scatter(s["r2_two_cell_reference"], s["r2_two_cell_method"], s=9, facecolor=np.where(filled, c, "white"),
                   edgecolor=c, linewidth=0.6, zorder=3)
        above = int((s["r2_two_cell_method"] > s["r2_two_cell_reference"]).sum())
        ax.set_title(SHORT_METHOD.get(m, name(m)).replace("\n", " "), fontsize=5.5)
        ax.text(0.03, 0.97, f"tool higher {above}\nThreadfin higher {len(s) - above}", transform=ax.transAxes,
                va="top", fontsize=5, color=INK2, linespacing=1.1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Threadfin, R² on two-cell clones", fontsize=5.5)
        if i == 0:
            ax.set_ylabel("tool, R² on the same clones", fontsize=5.5)
        ax.tick_params(labelsize=5)
    save(fig, FIG["two_cell"])


# --------------------------------------------------------------------------- 9. tests

def wilson(k, n, z=1.96):
    """Wilson score interval of a binomial proportion."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def tests_figure():
    files = sorted(glob.glob(str(RESULTS / "tests" / "*.csv")))
    if not files:
        print("skip: tests/*.csv not found")
        return
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t["threadfin_reject"] = t["threadfin_p"] < 0.05
    t["scirpy_reject"] = t["scirpy_n_clones_fdr05"] > 0
    t["anova_reject"] = t["cell_anova_p"] < 0.05
    pfiles = sorted(glob.glob(str(RESULTS / "tests" / "power" / "*.csv")))
    power = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True) if pfiles else None
    rows = [d for d in ORDER if d in set(t["dataset"])]
    tests = [("threadfin_reject", "Threadfin clonal coherence (one test per dataset)", TF, "o"),
             ("scirpy_reject", "scirpy clonotype modularity (≥ 1 clone at FDR 5 %)", INK2, "s"),
             ("anova_reject", "cell-level ANOVA (cells as replicates)", MUTED, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, (6.2 * len(rows) + 30) * MM), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 0.82, 1.15]})
    for ax, scen, title, letter in [(axes[0], "null_within_sample", "No clonal signal: labels shuffled\nwithin each sample", "a"),
                                    (axes[1], "null_global", "No clonal signal: labels shuffled\nacross the dataset", "b")]:
        sub = t[t["scenario"] == scen]
        for k, (col, lab, c, mk) in enumerate(tests):
            g = sub.groupby("dataset")[col].agg(["sum", "size"])
            g = g.loc[[d for d in rows if d in g.index]]
            lo, hi = wilson(g["sum"], g["size"])
            rate = g["sum"] / g["size"]
            ys = np.array([rows.index(d) for d in g.index]) + (k - 1) * 0.24
            ax.errorbar(rate, ys, xerr=[rate - lo, hi - rate], fmt="none", ecolor=c, elinewidth=0.6, capsize=0)
            ax.scatter(rate, ys, color=c, marker=mk, s=9, label=lab, edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(0.05, color=INK2, lw=0.6, ls="--")
        ax.text(0.07, -0.85, "5 %", fontsize=5, color=INK2, va="bottom")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(rows) - 0.5, -0.9)
        ax.set_yticks(range(len(rows)), [SHORT_DS[d] for d in rows] if ax is axes[0] else [""] * len(rows))
        ax.set_xlabel("share of 50 null replicates with a rejection\n(bar: Wilson 95 % interval)", fontsize=5.5)
        ax.set_title(title, loc="left", fontsize=6, linespacing=1.05)
        ax.grid(axis="y", visible=False)
        panel_letter(fig, ax, letter, dx=-2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=5.5, handletextpad=0.2, columnspacing=1.2)
    # c: power of the Threadfin test when only a fraction of cells keep their real clone label
    ax = axes[2]
    ax.grid(False)
    cols = [("null_within_sample", 0.0, "0 %"), ("mix", 0.1, "10 %"), ("mix", 0.25, "25 %"), ("mix", 0.5, "50 %")]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd"] + BLUES)
    n_rep = set()
    for i, d in enumerate(rows):
        for j, (scen, f, _) in enumerate(cols):
            if scen == "mix" and power is not None:
                s = power[(power["dataset"] == d) & np.isclose(power["fraction_real_labels"], f)]
            else:
                s = t[(t["dataset"] == d) & (t["scenario"] == scen) & np.isclose(t["fraction_real_labels"], f)]
            if s.empty:
                continue
            k = int((s["threadfin_p"] < 0.05).sum())
            n_rep.add(len(s))
            v = k / len(s)
            lo, hi = wilson(k, len(s))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(v), edgecolor="white", lw=1.5))
            ax.text(j, i, f"{v:.2f}\n{float(lo):.2f}–{float(hi):.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.55 else INK, linespacing=1.0)
        real = t[(t["dataset"] == d) & (t["scenario"] == "observed")]
        if not real.empty:                        # the test on the real labels (99 permutations: smallest P 0.01)
            ax.text(len(cols) - 0.38, i, f"P = {float(real['threadfin_p'].iloc[0]):.2f}", ha="left", va="center",
                    fontsize=5, color=INK)
    ax.text(len(cols) - 0.38, -0.8, "real\nlabels", ha="left", va="center", fontsize=5, color=INK2, linespacing=1.0)
    ax.set_xlim(-0.5, len(cols) + 0.55)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xticks(range(len(cols)), [c for *_, c in cols])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("cells that keep their real clone label\n(the rest shuffled within sample)", fontsize=5.5)
    reps = "/".join(str(x) for x in sorted(n_rep))
    ax.set_title(f"Power of the Threadfin test (share of\n{reps} replicates; Wilson 95 % interval)", loc="left",
                 fontsize=6, linespacing=1.05)
    panel_letter(fig, ax, "c", dx=-2)
    save(fig, FIG["tests"])


# --------------------------------------------------------------------------- 10. simulation

SIM_METHODS = ["threadfin_kernel", "threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
               "threadfin_kernel_noshrink", "clone2vec", "clonotrace", "composition", "mean_pca", "conga_medoid"]


def simulation_frame(sim):
    """Per simulated dataset and tool, the value the figure shows: every tool but Clonotrace from the seed-level set
    (10 seeds; clone2vec forced to >= 2 cells for small clones, at its default >= 3 for large ones), Clonotrace from
    the sets of seeds 1-3 (forced to >= 2 cells for small clones, its default >= 10 for large ones)."""
    small = sim["regime"].eq("small")
    s2 = sim["set"].eq("seeds: clones >= 2 cells")
    s10 = sim["set"].eq("seeds: clones >= 10 cells")
    keep = ((s2 & (small | ~sim["method"].eq("clone2vec_min2")))
            | (s10 & ~small & sim["method"].eq("clone2vec_min3"))
            | (small & sim["set"].eq("forced: clones >= 2 cells") & sim["method"].eq("clonotrace_revised_min2"))
            | (~small & sim["set"].eq("native: clones >= 10 cells") & sim["method"].eq("clonotrace_revised_min10")))
    out = sim[keep].drop_duplicates(["dataset", "method"]).copy()
    out["tool"] = out["method"].str.replace(r"_min\d+$", "", regex=True).str.replace("clonotrace_revised", "clonotrace")
    return out[out["tool"].isin(SIM_METHODS)]


def simulation_figure():
    sim = read("simulation.csv")
    if sim is None:
        return
    sim = simulation_frame(sim)
    label = {"clone2vec": "clone2vec", "clonotrace": "Clonotrace (3 seeds)"}
    tool_colour = {"clone2vec": C2V, "clonotrace": CT}
    metrics = [("programme_purity", "Programme purity of the\n10 nearest clones", "purity_chance"),
               ("mixture_vs_intermediate_auroc", "Mixture vs intermediate\nclones (AUROC)", None),
               ("same_donor_excess", "Excess of same-donor\nneighbours", None)]
    regimes = [("small", 0.0, "Small clones\n(half have 2 cells),\nno donor effect"),
               ("small", 1.5, "Small clones,\ndonor shift 1.5 SD"),
               ("large", 0.0, "Large clones\n(≥ 10 cells),\nno donor effect"),
               ("large", 1.5, "Large clones,\ndonor shift 1.5 SD")]
    fig, axes = plt.subplots(len(regimes), len(metrics), figsize=(WIDTH, 170 * MM), sharey=True, sharex="col",
                             gridspec_kw={"left": 0.3, "right": 0.99, "top": 0.94, "bottom": 0.07, "hspace": 0.12,
                                          "wspace": 0.08})
    ypos = {m: i for i, m in enumerate(SIM_METHODS)}
    for r, (reg, b, rtitle) in enumerate(regimes):
        sub = sim[(sim["regime"] == reg) & np.isclose(sim["batch"], b)]
        for c, (col, mtitle, chance) in enumerate(metrics):
            ax = axes[r, c]
            for tool, g in sub.groupby("tool"):
                colr = tool_colour.get(tool, colour(tool))
                y = ypos[tool]
                v = g[col].dropna()
                if v.empty:
                    continue
                ax.scatter(v, np.full(len(v), y), s=3, color=colr, alpha=0.5, linewidth=0, zorder=2)
                ax.scatter([v.mean()], [y], s=12, color=colr, edgecolor="white", linewidth=0.5, zorder=3)
            if chance:
                ax.axvline(sub[chance].mean(), color=INK2, lw=0.6, ls="--")
            if col == "mixture_vs_intermediate_auroc":
                ax.axvline(0.5, color=INK2, lw=0.6, ls="--")
            if col == "same_donor_excess":
                ax.axvline(0, color=INK2, lw=0.6, ls="--")
            ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(mtitle, fontsize=6, linespacing=1.05)
            if c == 0:
                ax.annotate(rtitle, xy=(0, 0.5), xycoords="axes fraction", xytext=(-92, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=5.5, fontweight="bold", linespacing=1.05)
            if r == len(regimes) - 1:
                ax.set_xlabel({"programme_purity": "share from the same programme\n(dashed: chance)",
                               "mixture_vs_intermediate_auroc": "AUROC (dashed: chance)",
                               "same_donor_excess": "same-donor share minus expected\n(dashed: none)"}[col], fontsize=5.5)
            ax.tick_params(labelsize=5)
    axes[0, 0].set_yticks(range(len(SIM_METHODS)), [label.get(m, name(m)) for m in SIM_METHODS], fontsize=5)
    axes[0, 0].set_ylim(len(SIM_METHODS) - 0.5, -0.5)
    save(fig, FIG["simulation"])


# --------------------------------------------------------------------------- 6-7. split halves

FORCED_HALF, ABL_HALF, SENS_HALF = "forced: >= 2 cells per half", "ablation: >= 2 cells per half", "sensitivity: >= 2 cells per half"
HALF_METHODS = [("clone2vec_min2", FORCED_HALF), ("clonotrace_revised_min2", FORCED_HALF),
                ("clonotrace_raw_min2", FORCED_HALF), ("benisse_min2", FORCED_HALF), ("mean_pca", FORCED_HALF),
                ("composition", FORCED_HALF), ("conga_medoid", FORCED_HALF), ("threadfin_mean", ABL_HALF),
                ("threadfin_kernel_nocentre", ABL_HALF), ("threadfin_kernel_nosmooth", ABL_HALF),
                ("threadfin_kernel_noshrink", ABL_HALF), ("threadfin_kernel_seed1", SENS_HALF)]


def matched_disjoint(dis):
    """Per (method, dataset, annotation): the method's R² and Threadfin's on the same halves and clones (the same set,
    split and direction), each averaged over the matched splits and directions."""
    keys = ["dataset", "target", "set", "seed", "representation_half"]
    tf = dis[dis["method"].eq("threadfin_kernel")][keys + ["r2"]]
    out = []
    for m, set_name in HALF_METHODS:
        s = dis[dis["method"].eq(m) & dis["set"].eq(set_name)][keys + ["tag", "r2"]]
        j = s.merge(tf, on=keys, suffixes=("", "_tf"))
        g = j.groupby(["dataset", "target", "tag"])[["r2", "r2_tf"]].mean().reset_index()
        g["method"], g["set"] = m, set_name
        g["delta"] = g["r2"] - g["r2_tf"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def matched_reproducibility(rep):
    """Per (method, dataset): the method's readouts and Threadfin's on the same halves and clones, averaged over the
    matched splits."""
    cols = ["distance_spearman", "knn_overlap", "knn_overlap_chance", "reid_top1", "reid_chance_top1"]
    tf = rep[rep["method"].eq("threadfin_kernel")][["dataset", "set", "seed"] + cols]
    out = []
    for m, set_name in [("threadfin_kernel", FORCED_HALF)] + HALF_METHODS:
        s = rep[rep["method"].eq(m) & rep["set"].eq(set_name)][["dataset", "set", "seed"] + cols]
        j = s.merge(tf, on=["dataset", "set", "seed"], suffixes=("", "_tf"))
        g = j.groupby("dataset")[[c for c in j.columns if c not in ("dataset", "set", "seed")]].mean().reset_index()
        g["method"], g["set"] = m, set_name
        out.append(g)
    return pd.concat(out, ignore_index=True)


def halves_figures():
    rep = read("split_half_reproducibility.csv")
    if rep is not None:
        mr = matched_reproducibility(rep)
        methods = ["threadfin_kernel"] + [m for m, _ in HALF_METHODS]
        methods = [m for m in methods if m in set(mr["method"])]
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, (4.6 * len(methods) + 22) * MM), sharey=True,
                                 gridspec_kw={"left": 0.235, "right": 0.99, "wspace": 0.1, "top": 0.95, "bottom": 0.2})
        for ax, col, xl, chance, letter in [
                (axes[0], "distance_spearman", "Spearman correlation of clone-clone\ndistances between the halves", None, "a"),
                (axes[1], "knn_overlap", "overlap of the 10 nearest clones\n(dashed: chance)", "knn_overlap_chance", "b"),
                (axes[2], "reid_top1", "clones re-identified as their own best\nmatch across halves (dashed: chance)",
                 "reid_chance_top1", "c")]:
            for i, m in enumerate(methods):
                g = mr[mr["method"] == m]
                ax.scatter(g[col], np.full(len(g), i), s=5, color=colour(m), alpha=0.55, linewidth=0, zorder=2)
                ax.scatter([g[col].median()], [i], s=16, marker="D", color=colour(m), edgecolor="white", linewidth=0.5,
                           zorder=3)
                if m != "threadfin_kernel":       # Threadfin on the same halves and clones
                    ax.plot([g[f"{col}_tf"].median()] * 2, [i - 0.32, i + 0.32], color=INK, lw=0.9, zorder=4)
            if chance:
                ax.axvline(mr[chance].median(), color=INK2, lw=0.6, ls="--")
            ax.set_xlabel(xl, fontsize=5.5)
            ax.grid(axis="y", visible=False)
            ax.tick_params(labelsize=5)
            panel_letter(fig, ax, letter, dx=-2)
        axes[0].set_yticks(range(len(methods)), [name(m) for m in methods], fontsize=5)
        axes[0].set_ylim(len(methods) - 0.5, -0.5)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.5, color=MUTED, alpha=0.6, label="dataset (mean over splits)"),
                            Line2D([], [], marker="D", ls="", ms=3.5, color=MUTED, label="median over datasets"),
                            Line2D([], [], marker="|", ls="", ms=6, mew=0.9, color=INK,
                                   label="Threadfin on the same halves and clones (median)")],
                   loc="lower center", ncol=3, fontsize=5.5, bbox_to_anchor=(0.6, 0.0))
        save(fig, FIG["reproducibility"])
    dis = read("annotation_recovery_disjoint.csv")
    if dis is not None:
        md = matched_disjoint(dis)
        methods = [m for m, _ in HALF_METHODS if m in set(md["method"])]
        fig, ax = plt.subplots(figsize=(WIDTH * 0.62, (4.6 * len(methods) + 22) * MM),
                               gridspec_kw={"left": 0.42, "right": 0.97, "top": 0.97, "bottom": 0.2})
        for i, m in enumerate(methods):
            v = md.loc[md["method"] == m, "delta"]
            ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), s=4,
                       color=colour(m), alpha=0.55, linewidth=0, zorder=2)
            ax.scatter([v.median()], [i], s=16, marker="D", color=colour(m), edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(1.01, i, f"{int((v > 0).sum())} / {int((v < 0).sum())}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=5, color=INK2)
        ax.text(1.01, -0.9, "higher /\nlower", transform=ax.get_yaxis_transform(), va="center", fontsize=5,
                color=INK2, linespacing=1.0)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(range(len(methods)), [name(m) for m in methods], fontsize=5)
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xlabel("R² of the tool minus R² of Threadfin on the same halves and clones\n"
                      "(dots: annotations, mean over splits and directions; diamond: median)", fontsize=5.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=5)
        save(fig, FIG["disjoint"])


# --------------------------------------------------------------------------- 8. LARRY linking

def larry_figure():
    lk = read("larry_clone_linking.csv")
    if lk is None or lk.empty:
        return
    sets = [("forced: clones >= 2 cells", "Clones of ≥ 2 cells kept by every tool"),
            ("native: clones >= 10 cells", "Clones of ≥ 10 cells kept by every tool")]
    n_rows = {s: int(lk["set"].eq(s).sum()) for s, _ in sets}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, (4.4 * sum(n_rows.values()) + 34) * MM),
                             gridspec_kw={"height_ratios": [n_rows[s] for s, _ in sets], "left": 0.27, "right": 0.98,
                                          "hspace": 0.55, "wspace": 0.12, "top": 0.93, "bottom": 0.1})
    for r, (set_name, title) in enumerate(sets):
        s = lk[lk["set"].eq(set_name)].sort_values("mrr", ascending=True)
        n, k = int(s["n_query_clones"].iloc[0]), int(s["n_candidate_day6_clones"].iloc[0])
        ax = axes[r, 0]
        for i, (_, row) in enumerate(s.iterrows()):
            ax.plot([0, row["mrr"]], [i, i], color=colour(row["method"]), lw=1.6, solid_capstyle="round")
            ax.text(row["mrr"], i, f"  {row['mrr']:.3f}", va="center", fontsize=5, color=INK)
        ax.axvline(float(np.mean([1 / j for j in range(1, k + 1)])), color=INK2, ls="--", lw=0.6)
        ax.set_xlim(0, s["mrr"].max() * 1.3)
        ax.set_yticks(range(len(s)), [name(m) for m in s["method"]], fontsize=5)
        ax.set_xlabel("mean reciprocal rank of the own day-6 clone (dashed: chance)", fontsize=5.5)
        ax.set_title(f"{title}: {n:,} day-4 queries among {k:,} day-6 clones", loc="left", fontsize=6, pad=12)
        ax.grid(axis="y", visible=False)
        panel_letter(fig, ax, "ac"[r], dx=-2)
        ax = axes[r, 1]
        panel_letter(fig, ax, "bd"[r], dx=-2)
        for i, (_, row) in enumerate(s.iterrows()):
            c = colour(row["method"])
            if row["method"] == "threadfin_kernel":
                ax.text(0, i, " reference", va="center", ha="left", fontsize=5, color=INK2)
                continue
            lo, hi = row["mrr_delta_ci_low"], row["mrr_delta_ci_high"]
            cl = row.get("call", "similar")
            ax.plot([lo, hi], [i, i], color=c, lw=0.8)
            ax.plot([row["mrr_delta_vs_threadfin"]], [i], "D" if cl == "unresolved" else "o", ms=2.6,
                    mfc=c if cl in ("better", "worse") else "white", mec=c, mew=0.7)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(range(len(s)), [""] * len(s))
        ax.set_xlabel("difference to Threadfin (filled: 95 % paired bootstrap\ninterval over queries excludes 0)",
                      fontsize=5.5)
        ax.grid(axis="y", visible=False)
        for a in axes[r]:
            a.set_ylim(len(s) - 0.5, -0.5)
            a.tick_params(labelsize=5)
    save(fig, FIG["larry"])


# --------------------------------------------------------------------------- 12. runtime and memory

RES_PANELS = [("clone2vec_min3", "clone2vec\ndefault, ≥ 3 cells", C2V), ("clone2vec_min2", "clone2vec\nforced, ≥ 2 cells", C2V),
              ("clonotrace_revised_min10", "Clonotrace\ndefault, ≥ 10 cells", CT),
              ("clonotrace_revised_min2", "Clonotrace\nforced, ≥ 2 cells", CT), ("benisse", "Benisse\n≥ 2 cells", BN)]
# the core-count and data-size experiments ran each program once; the default and forced settings share it
SAME_PROGRAM = {"clone2vec_min3": "clone2vec_min2", "clone2vec_min2": "clone2vec_min2",
                "clonotrace_revised_min10": "clonotrace_revised_min2", "clonotrace_revised_min2": "clonotrace_revised_min2",
                "benisse": "benisse_min2"}
STOP_SECONDS = 72 * 3600
RUN_DIR = {"clonotrace_revised_min10": "clonotrace_min10", "clonotrace_revised_min2": "clonotrace_min2"}


def _run(sub, key):
    if key == "benisse":
        r = sub.loc["benisse_min2"]
        return sub.loc["benisse_min10"] if r["status"] != "completed" and sub.loc["benisse_min10", "status"] == "completed" else r
    return sub.loc[key]


def _not_run_note(r):
    reason = str(r["reason"])
    return ("stopped at 72 h" if r["status"] == "stopped" else "failed" if r["status"] == "failed" else
            "no BCR" if "receptor" in reason else "< 3 clones ≥ 10" if "fewer than 3" in reason else "not run")


def job_peaks():
    """(dataset, run directory) -> the scheduler's peak memory of a job that ran that run alone."""
    h = read("run_hosts.csv")
    if h is None:
        return {}
    h = h[h["variant"].eq("full") & h["alone_in_job"].eq(True) & h["job_max_rss_gb"].notna()]
    return {(r.dataset, r.tool_run): float(r.job_max_rss_gb) for r in h.itertuples()}


def resource_rows():
    """Rows of the computing-resource figure: (block, label, {panel: (ratio, note, hollow)})."""
    cov, res = read("coverage_runtime.csv"), read("resources.csv")
    cov = cov[cov["dataset"].isin(ORDER)]
    peaks = job_peaks()
    out = []
    for block, what in (("Time per dataset", "time"), ("Peak memory per dataset", "memory")):
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
                    note = "≥ 10 cells" if r.name == "benisse_min10" else ""
                    if what == "time":
                        v = (r["seconds_method"] if pd.notna(r["seconds_method"]) else r["wall_seconds_task"]) / tf["seconds_method"]
                    else:
                        mem = r["peak_rss_gb_task"]
                        if key.startswith("clonotrace"):
                            jp = peaks.get((ds, RUN_DIR.get(key, key)))
                            mem = max(mem, jp) if jp else mem
                            note = "whole job" if jp else "≥ (main R session)"
                        v = mem / tf["peak_rss_gb_method"]
                    cells[key] = (float(v), note, hollow)
                elif r["status"] == "stopped" and what == "time":
                    cells[key] = (STOP_SECONDS / tf["seconds_method"], "stopped at 72 h", False)
                else:
                    cells[key] = (np.nan, _not_run_note(r), False)
            out.append((block, SHORT_DS[ds], cells))
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
            out.append(("Time with 1-16 cores (RBD)",
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
            out.append(("Time on GSE195673 subsets",
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
        y += 0.5
    height = (2.75 * y + 33) * MM
    fig, axes = plt.subplots(1, len(RES_PANELS), figsize=(WIDTH, height), sharey=True,
                             gridspec_kw={"left": 0.21, "right": 0.995, "wspace": 0.1, "top": 1 - 9 * MM / height,
                                          "bottom": 29 * MM / height})
    lo, hi = 0.05, 3e5
    for ax, (key, title, col) in zip(axes, RES_PANELS):
        counts = {b: [0, 0, 0] for b in blocks}             # less, more, failed or stopped
        for (blk, lab, cells), yy in zip(rows, ypos):
            v, note, hollow = cells[key]
            if np.isfinite(v):
                lower = note.startswith("≥ (main")          # Clonotrace's main R session: a lower bound
                ax.scatter([min(max(v, lo), hi)], [yy], s=10, zorder=3, color="white" if hollow else col,
                           edgecolor=col, linewidth=0.7, marker=">" if lower else "o")
                if note and not lower and note != "whole job":
                    ax.text(min(v, hi) * 1.7, yy, note, fontsize=5, color=INK2, va="center")
                if note.startswith("stopped"):
                    ax.annotate("", xy=(min(v, hi) * 1.6, yy), xytext=(min(v, hi), yy),
                                arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
                if not 1 / 1.25 <= v <= 1.25:
                    counts[blk][0 if v < 1 else 1] += 1
            else:
                ax.text(np.sqrt(lo * hi), yy, note, fontsize=5,
                        color=MUTED if note in ("not run", "no BCR", "not planned", "pending", "running",
                                                "< 3 clones ≥ 10") else "#b0306f",
                        va="center", ha="center", style="italic")
                if note in ("failed",) or note.startswith("stopped"):
                    counts[blk][2] += 1
        ax.axvline(1, color=INK2, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title, fontsize=5.5, linespacing=1.05)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=5)
        foot = []
        for b, short in zip(blocks, ("time", "memory", "cores", "data size")):
            f, sl, miss = counts[b]
            foot.append(f"{short}: {'less' if short == 'memory' else 'faster'} {f}, "
                        f"{'more' if short == 'memory' else 'slower'} {sl}" + (f", failed {miss}" if miss else ""))
        ax.set_xlabel("times Threadfin (log)\n" + "\n".join(foot), fontsize=5, linespacing=1.15)
    axes[0].set_yticks(ypos, labels, fontsize=5)
    for yy, b in heads:
        axes[0].text(-0.03, yy, b, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=5.5,
                     fontweight="bold")
    axes[0].set_ylim(y - 0.4, -1.2)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec=CT,
                               label="hollow: a Clonotrace run whose clone distances used one R session"),
                        Line2D([], [], marker=">", ls="", ms=3, color=CT,
                               label="triangle: Clonotrace memory of the main R session only (a lower bound)")],
               loc="lower left", ncol=2, fontsize=5, bbox_to_anchor=(0.21, -0.005), handletextpad=0.2, columnspacing=1.5)
    save(fig, FIG["resources"])


def scaling_figure():
    """Absolute numbers behind fig. 12: speed-up with cores, time and memory against the number of clones."""
    res = read("resources.csv")
    if res is None:
        return
    tools = [("threadfin_run", "Threadfin, complete analysis", TF, "o", "-"),
             ("threadfin_kernel", "Threadfin, clone profiles only", TF, "o", "--"),
             ("clone2vec_min2", "clone2vec", C2V, "s", "-"), ("clonotrace_revised_min2", "Clonotrace", CT, "v", "-"),
             ("benisse_min2", "Benisse", BN, "D", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 62 * MM), layout="constrained")
    c = res[res["experiment"].eq("cores") & res["status"].eq("completed")]
    ax = axes[0]
    for t, lab, col, mk, ls in tools:
        s = c[c["tool"].eq(t)].sort_values("cores")
        if len(s) < 2:
            continue
        base = s.iloc[0]
        ax.plot(s["cores"], base["seconds"] / s["seconds"], marker=mk, ms=2.5, color=col, ls=ls, lw=0.8,
                label=lab + (f" (vs {int(base['cores'])} cores)" if base["cores"] != 1 else ""))
    ax.plot([1, 16], [1, 16], color=MUTED, lw=0.6, ls=":", label="perfect scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16], ["¼", "½", "1", "2", "4", "8", "16"])
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("speed-up over the fewest cores")
    ax.set_title("More cores (GSE287123 RBD, 1,887 clones)", loc="left", fontsize=6)
    ax.legend(fontsize=5, handlelength=1.6)
    panel_letter(fig, ax, "a", dx=-2)
    g = res[res["experiment"].eq("size") & res["status"].eq("completed")]
    for ax, col_name, ylab, title, letter in ((axes[1], "seconds", "seconds (log)", "Time as the data grow", "b"),
                                              (axes[2], "peak_rss_gb", "peak memory, GB (log)", "Memory as the data grow", "c")):
        for t, lab, col, mk, ls in tools:
            s = g[g["tool"].eq(t)].sort_values("n_clones_ge2")
            if s.empty:
                continue
            ax.plot(s["n_clones_ge2"], s[col_name], marker=mk, ms=2.5, color=col, ls=ls, lw=0.8, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([500, 1000, 2000, 5000], ["500", "1,000", "2,000", "5,000"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("clones with ≥ 2 cells (subsets of GSE195673, 8 cores)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=6)
        panel_letter(fig, ax, letter, dx=-2)
    axes[1].legend(fontsize=5, handlelength=1.6)
    save(fig, FIG["scaling"])


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
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, (4.2 * len(SENS) + 26) * MM), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1, 1, 1], "left": 0.23, "right": 0.99, "wspace": 0.12,
                                          "top": 0.93, "bottom": 0.2})
    y = {k: i for i, k in enumerate(keys)}
    a = pw[pw["method"].isin(keys) & (pw["tag"] != "rna")]
    ax = axes[0]
    for k in keys:
        d = a[a["method"].eq(k)]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        calls = d.apply(call_of, axis=1)
        sig = calls.isin(["better", "worse"]).to_numpy()
        ax.scatter(d["delta_method_minus_reference"], y[k] + jitter, s=3, color=np.where(sig, TF, "#b9c9de"), lw=0, zorder=2)
        ax.plot([d["delta_method_minus_reference"].median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.0, zorder=3)
        ax.text(0.99, y[k], f"{int(calls.eq('better').sum())} ▲ {int(calls.eq('worse').sum())} ▼",
                transform=ax.get_yaxis_transform(), fontsize=5, va="center", ha="right", color=INK2,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xlabel("R² of the setting − R² of the default,\nper measured annotation (dark: 95 % interval\n"
                  "excludes 0; bar: median; ▲ ▼ annotations)", fontsize=5)
    ax.set_title(f"Measured annotations ({a['target'].groupby(a['dataset']).nunique().sum()})", loc="left", fontsize=6)
    panel_letter(fig, ax, "a", dx=-2)
    r = rep[rep["set"].str.startswith("sensitivity")].groupby(["dataset", "method"])["reid_top1"].mean().unstack()
    ax = axes[1]
    for k in keys:
        if k not in r:
            continue
        d = (r[k] - r["threadfin_kernel"]).dropna()
        ax.scatter(100 * d, [y[k]] * len(d), s=4, color=TF, alpha=0.7, lw=0)
        ax.plot([100 * d.median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("re-identified clones, percentage\npoints (setting − default; one\ndot per dataset)", fontsize=5)
    ax.set_title("Recognising a clone", loc="left", fontsize=6)
    panel_letter(fig, ax, "b", dx=-2)
    ss = sim[sim["set"].isin(["sensitivity: clones >= 2 cells", "seeds sensitivity: clones >= 2 cells"])]
    ss = ss.sort_values("set").drop_duplicates(["dataset", "method"], keep="last")
    for ax, col, title, letter in ((axes[2], "programme_purity", "Simulation: planted groups", "c"),
                                   (axes[3], "mixture_vs_intermediate_auroc", "Simulation: mixed vs\nin-between", "d")):
        sub = ss if col == "programme_purity" else ss[ss["regime"].eq("large")]
        m = sub.pivot_table(index="dataset", columns="method", values=col, aggfunc="first")
        seed = sub.drop_duplicates("dataset").set_index("dataset")["seed"]
        for k in keys:
            if k not in m:
                continue
            d = (m[k] - m["threadfin_kernel"]).dropna()
            d = d.groupby(seed.loc[d.index].to_numpy()).mean()       # one value per seed
            ax.scatter(d, [y[k]] * len(d), s=4, color=TF, alpha=0.7, lw=0)
            ax.plot([d.median()] * 2, [y[k] - 0.3, y[k] + 0.3], color=INK, lw=1.0)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlabel(("purity" if col == "programme_purity" else "AUROC") + " of the setting −\ndefault (one dot per "
                      "seed" + ("" if col == "programme_purity" else ",\nclones ≥ 10 cells") + ")", fontsize=5)
        ax.set_title(title, loc="left", fontsize=6, linespacing=1.05)
        panel_letter(fig, ax, letter, dx=-2)
    axes[0].set_yticks(range(len(keys)), [lab for _, lab in SENS], fontsize=5)
    axes[0].set_ylim(len(keys) - 0.5, -0.6)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=5)
    save(fig, FIG["sensitivity"])


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
