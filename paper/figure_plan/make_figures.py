#!/usr/bin/env python
"""Assemble the manuscript figures as A4 pages (Nature format).

Every main figure is one A4 page (210 x 297 mm). The figure area is 183 mm
wide (Nature double column) and laid out in millimetres from the top-left
corner of that area. Data panels are drawn from the committed benchmark
results; panels that still have to be drawn by hand (schematics, biological
models) are dashed placeholder boxes that say what they will show.

Usage:
    python make_figures.py            # writes Figure_1..5 and Extended_Data_* (.pdf and .png)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT.parent / "internal_validation" / "benchmarks" / "simulation" / "results"
DATA = ROOT / "case_studies" / "results"

# ------------------------------------------------------------------ style
MM = 1 / 25.4
A4_W, A4_H = 210 * MM, 297 * MM
AREA_W = 183.0                         # mm, Nature double column
LEFT = (210 - AREA_W) / 2              # mm
TOP = 18.0                             # mm, space for the figure title
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY, LIGHT, INK, INK2, MUTED = "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e", "#898781"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.0, "axes.titlesize": 6.5, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "lines.linewidth": 1.0, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def new_page(number: str, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.text(LEFT * MM / A4_W, 1 - 9 * MM / A4_H, f"{number} | {title}", fontsize=8.5, fontweight="bold",
             color=INK, va="top")
    if subtitle:
        fig.text(LEFT * MM / A4_W, 1 - 14 * MM / A4_H, subtitle, fontsize=6.5, color=INK2, va="top")
    fig.text(LEFT * MM / A4_W, 6 * MM / A4_H,
             "Threadfin figure plan. Grey dashed boxes are placeholders for panels still to be drawn; every other "
             "panel is generated from case_studies/results by paper/figure_plan/make_figures.py.",
             fontsize=5, color=MUTED, va="bottom")
    return fig


def panel(fig, x, y, w, h, letter=None, letter_dx=-4.0):
    """Axes at (x, y) mm from the top-left of the 183-mm figure area, size (w, h) mm."""
    left = (LEFT + x) * MM / A4_W
    bottom = 1 - (TOP + y + h) * MM / A4_H
    ax = fig.add_axes([left, bottom, w * MM / A4_W, h * MM / A4_H])
    if letter:
        fig.text((LEFT + x + letter_dx) * MM / A4_W, 1 - (TOP + y - 2.5) * MM / A4_H, letter,
                 fontsize=8.5, fontweight="bold", color=INK, va="bottom", ha="left")
    return ax


def placeholder(ax, text, fontsize=5.8):
    """Dashed box saying what a hand-drawn panel will show (text wrapped to the box width)."""
    import textwrap

    ax.set_axis_off()
    box = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.02",
                         transform=ax.transAxes, facecolor="#f6f6f3", edgecolor=MUTED, linewidth=0.6,
                         linestyle=(0, (4, 3)))
    ax.add_patch(box)
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72 * 0.9
    chars = max(int(width_pt / (fontsize * 0.5)), 12)
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, chars) or [""])
    ax.text(0.5, 0.5, "PLACEHOLDER\n" + "\n".join(lines), transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=INK2, linespacing=1.4)


def missing(ax, what):
    placeholder(ax, f"{what}\n(result not yet available - rerun make_figures.py after the benchmark)")


INTERNAL = ROOT.parent / "internal_validation" / "figures"   # kept out of the repository


def save(fig, name, folder=None):
    out = Path(folder or HERE)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", out / name)


def spread(values, gap):
    """Shift label positions apart so that neighbours are at least ``gap`` apart (keeps order)."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    out = v[order].copy()
    for i in range(1, out.size):
        out[i] = max(out[i], out[i - 1] + gap)
    shift = (out - v[order]).mean()          # re-centre the block on the data
    out -= shift
    for i in range(1, out.size):
        out[i] = max(out[i], out[i - 1] + gap)
    res = np.empty_like(out)
    res[order] = out
    return res


def log_ticks(ax, axis, ticks):
    """Plain-number major ticks on a log axis, no minor tick labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ loaders


def load_sim(task):
    files = sorted(SIM.glob(f"{task}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def load_summary(ds):
    path = DATA / ds / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_table(ds, name, **kw):
    path = DATA / ds / name
    return pd.read_csv(path, **kw) if path.exists() else None


# ------------------------------------------------------------------ simulation panels

METHODS = [  # (label in results, label in figure, Threadfin?)
    ("v3 centroid", "Threadfin v3 (centroid)", False),
    ("modal cell state", "Modal cell state", False),
    ("state composition", "State composition", False),
    ("clone2vec", "clone2vec", False),
    ("clone2vec (centred)", "clone2vec (context-centred)", False),
    ("Threadfin v4 (mean)", "Threadfin v4, mean profile", True),
    ("Threadfin v4 (kernel)", "Threadfin v4, kernel profile", True),
    ("true-state composition (reference)", "True-state composition*", None),
]
SCENARIOS = [("default", "Default"), ("strong_batch", "Strong sample shift"), ("no_batch", "No sample shift"),
             ("spread", "Clones across samples"), ("small_clones", "Mostly small clones"),
             ("bifurcation", "Equal-centroid programmes")]


def sim_recovery(fig, x, y, w, h, letter):
    rec = load_sim("recovery")
    if rec is None:
        missing(panel(fig, x, y, w, h, letter), "programme recovery (ARI) per scenario")
        return
    ncol, nrow = 3, 2
    gap_x, gap_y = 30.0, 9.0
    pw = (w - gap_x * (ncol - 1)) / ncol
    ph = (h - gap_y * (nrow - 1)) / nrow
    for k, (scen, title) in enumerate(SCENARIOS):
        r, c = divmod(k, ncol)
        ax = panel(fig, x + c * (pw + gap_x), y + r * (ph + gap_y), pw, ph, letter if k == 0 else None)
        sub = rec[rec["scenario"] == scen]
        rows = []
        for key, label, ours in METHODS:
            v = sub.loc[sub["method"] == key, "ari"].dropna()
            rows.append((label, v.mean() if v.size else np.nan, v.std(ddof=1) if v.size > 1 else 0, ours))
        ypos = np.arange(len(rows))[::-1]
        for yy, (label, m, sd, ours) in zip(ypos, rows):
            col = BLUE if ours else (LIGHT if ours is None else GREY)
            ax.barh(yy, m if np.isfinite(m) else 0, height=0.65, color=col, xerr=sd, error_kw=dict(lw=0.5, ecolor=INK2))
        ax.set_yticks(ypos, [r_[0] for r_ in rows] if c == 0 else [""] * len(rows))
        ax.set_xlim(0, 1)
        ax.set_title(title, color=INK, pad=2)
        ax.tick_params(axis="y", length=0)
        if r == nrow - 1:
            ax.set_xlabel("ARI vs true programmes")
    fig.text((LEFT + x - 40) * MM / A4_W, 1 - (TOP + y + h + 9) * MM / A4_H,
             "Bars: mean ± s.d. over 5 simulated repertoires (clones with >= 3 cells; best Leiden resolution for every "
             "method).\n*Composition of the noise-free true states of each clone's cells: a reference, not a ceiling "
             "(small clones are noisy even with true states).", fontsize=5, color=INK2, va="top")


def sim_by_size(ax):
    rec = load_sim("recovery")
    if rec is None:
        missing(ax, "ARI by clone size")
        return
    sub = rec[rec["scenario"] == "default"]
    bins = ["ari_3-4", "ari_5-9", "ari_>=10"]
    show = [("v3 centroid", "v3 centroid", GREY), ("clone2vec (centred)", "clone2vec (centred)", ORANGE),
            ("Threadfin v4 (kernel)", "Threadfin v4 (kernel)", BLUE),
            ("true-state composition (reference)", "true-state comp.", INK2)]
    ends = []
    for key, label, col in show:
        m = sub[sub["method"] == key][bins].mean()
        ax.plot(range(3), m.to_numpy(), "-o", color=col, ms=2.5, lw=1)
        ends.append(m.iloc[-1])
    for (key, label, col), yy in zip(show, spread(ends, 0.065)):
        ax.text(2.1, yy, label, color=INK2, fontsize=5, va="center")
    ax.set_xticks(range(3), ["3-4", "5-9", ">=10"])
    ax.set_xlim(-0.2, 2.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("cells per clone")
    ax.set_ylabel("ARI (default scenario)")


def sim_calibration(ax):
    cal = load_sim("calibration")
    if cal is None:
        missing(ax, "false-positive rates under the null")
        return
    order = [("coherence", "global shuffle", "Coherence:\nglobal shuffle"),
             ("coherence", "v3 null 1 (purity, within-donor)", "Coherence:\nv3 donor null"),
             ("coherence", "Threadfin v4 (within-sample null)", "Coherence:\nThreadfin v4"),
             ("association", "v3 (cell-level Fisher)", "Association:\ncell-level Fisher"),
             ("association", "Threadfin v4 (clone-level)", "Association:\nThreadfin v4")]
    rates, lo, hi, cols = [], [], [], []
    for test, method, _ in order:
        p = cal[(cal["test"] == test) & (cal["method"] == method)]["pvalue"].dropna()
        k, n = int((p < 0.05).sum()), int(p.size)
        rate = k / n if n else np.nan
        se = np.sqrt(max(rate * (1 - rate), 1e-9) / max(n, 1))
        rates.append(rate)
        lo.append(max(rate - 1.96 * se, 0))
        hi.append(min(rate + 1.96 * se, 1))
        cols.append(BLUE if "Threadfin" in method else GREY)
    xx = np.arange(len(order))
    ax.bar(xx, rates, width=0.65, color=cols)
    ax.errorbar(xx, rates, yerr=[np.array(rates) - lo, np.array(hi) - rates], fmt="none", ecolor=INK2, elinewidth=0.5)
    ax.axhline(0.05, color=MUTED, lw=0.5)
    ax.text(len(order) - 0.45, 0.06, "nominal 5%", fontsize=5, color=INK2, ha="right", va="bottom")
    ax.set_xticks(xx, [o[2] for o in order], fontsize=5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("false-positive rate (P < 0.05)")


def sim_memory(ax):
    mem = load_sim("memory")
    if mem is None:
        missing(ax, "memory index vs true keep-rate")
        return
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.8)
    ax.scatter(mem["true_keep_rate"], mem["v3_transition_diagonal"], s=8, color=GREY, label="v3 transition diagonal",
               edgecolors="white", linewidths=0.3)
    ax.scatter(mem["true_keep_rate"], mem["v4_memory_index"], s=8, color=BLUE, label="Threadfin v4 memory index",
               edgecolors="white", linewidths=0.3)
    ax.set_xlabel("true fraction of clones keeping their programme")
    ax.set_ylabel("estimate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(loc="lower right", handletextpad=0.2)


def sim_scaling(ax_t, ax_m):
    sc = load_sim("scaling")
    if sc is None:
        missing(ax_t, "runtime")
        missing(ax_m, "memory")
        return
    sc = sc.sort_values("n_cells")
    for col, label, c in (("profiles_kernel_s", "clone profiles", BLUE),
                          ("coherence_50perm_s", "coherence test (50 permutations)", ORANGE),
                          ("programmes_20boot_s", "programmes (20 bootstraps)", AQUA)):
        ax_t.plot(sc["n_cells"], sc[col] / 60, "-o", color=c, ms=2.5, label=label)
    ax_m.plot(sc["n_cells"], sc["peak_rss_gb"], "-o", color=BLUE, ms=2.5)
    xt = [1e4, 1e5, 1e6]
    for ax in (ax_t, ax_m):
        ax.set_xscale("log")
        log_ticks(ax, "x", xt)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlim(7e3, 1.5e6)
        ax.set_xlabel("cells in the data set")
    ax_t.set_yscale("log")
    log_ticks(ax_t, "y", [0.001, 0.01, 0.1, 1, 10, 100])
    ax_t.set_ylabel("wall time (min, 2 CPUs)")
    ax_t.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    ax_m.set_ylabel("peak memory (GB)")
    ax_m.set_ylim(0, None)


def sim_auto(ax):
    """Two-stage pipeline: core clones (precise) vs core + assigned (complete)."""
    rec = load_sim("recovery")
    if rec is None:
        missing(ax, "automatic pipeline: ARI vs coverage")
        return
    # sample shifts are pure translations in the simulation, so context centring makes these three identical
    labels = {"default": "default, strong shift,\nno shift (identical\nafter centring)", "spread": "clones across samples",
              "small_clones": "mostly small clones", "bifurcation": "equal centroids"}
    core = rec[rec["method"] == "Threadfin v4 (kernel) auto, core"].groupby("scenario")[["coverage", "ari"]].mean()
    full = rec[rec["method"] == "Threadfin v4 (kernel) auto, core + assigned"].groupby("scenario")[
        ["coverage", "ari"]].mean()
    core, full = core.loc[list(labels)], full.loc[list(labels)]
    for scen in labels:
        ax.plot([core.loc[scen, "coverage"], full.loc[scen, "coverage"]], [core.loc[scen, "ari"], full.loc[scen, "ari"]],
                color=LIGHT, lw=0.8, zorder=1)
    ax.scatter(core["coverage"], core["ari"], marker="o", s=14, color=BLUE, edgecolors="white", linewidths=0.3,
               label="core clones", zorder=3)
    ax.scatter(full["coverage"], full["ari"], marker="s", s=12, color=ORANGE, edgecolors="white", linewidths=0.3,
               label="core + assigned clones", zorder=3)
    for scen, yy in zip(labels, spread(full["ari"].to_numpy(), 0.06)):
        ax.text(1.03, yy, labels[scen], fontsize=4.8, color=INK2, va="center", linespacing=1.0)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, 0.75)
    ax.spines["bottom"].set_bounds(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("fraction of clones (>= 3 cells) labelled")
    ax.set_ylabel("ARI vs truth (automatic pipeline)")
    ax.legend(loc="lower left", handletextpad=0.2)


def sim_runtime(ax):
    rec = load_sim("recovery")
    if rec is None:
        missing(ax, "runtime per method")
        return
    sub = rec[rec["scenario"] == "default"]
    show = [("v3 centroid", "v3 centroid"), ("modal cell state", "modal state*"), ("state composition", "composition*"),
            ("clone2vec", "clone2vec"), ("clone2vec (centred)", "clone2vec (centred)"),
            ("Threadfin v4 (mean)", "Threadfin, mean"), ("Threadfin v4 (kernel)", "Threadfin, kernel"),
            ("Threadfin v4 (kernel) auto, core", "Threadfin, full\nprogramme step")]
    vals = [sub.loc[sub["method"] == k, "seconds"].mean() for k, _ in show]
    y = np.arange(len(show))[::-1]
    ax.barh(y, vals, height=0.6, color=[BLUE if "Threadfin" in lab else GREY for _, lab in show])
    ax.set_yticks(y, [lab for _, lab in show])
    ax.set_xscale("log")
    log_ticks(ax, "x", [0.1, 1, 10, 100])
    ax.set_xlim(0.1, 150)
    ax.set_xlabel("seconds per repertoire\n(~8,000 cells)")
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, -0.3, "*includes cell-level Leiden clustering", transform=ax.transAxes, fontsize=4.8,
            color=INK2, ha="right", va="top")


# ------------------------------------------------------------------ method panels (Figure 1)

DATASET_NAMES = {"ln_vaccine": "Lymph node, mRNA vaccine", "mouse_np": "Mouse GC, NP",
                 "mouse_rbd": "Mouse GC, RBD", "flu": "Blood, influenza vaccine", "ebv": "Tonsil organoids, EBV",
                 "tonsil": "Tonsil", "stephenson": "Blood, COVID-19"}
DATASET_ORDER = ["ln_vaccine", "mouse_np", "mouse_rbd", "flu", "ebv", "tonsil", "stephenson"]


def clone_threshold_panel(ax, ax_in=None, dataset="ln_vaccine"):
    """Distance of every junction to its nearest neighbour, with the automatic threshold."""
    hist_path, sens_path = DATA / "clone_threshold_histograms.csv", DATA / "clone_threshold_sensitivity.csv"
    thr_path = DATA / "clone_thresholds.json"
    if not hist_path.exists() or not thr_path.exists():
        missing(ax, "distance-to-nearest histogram")
        if ax_in is not None:
            ax_in.set_axis_off()
        return
    h = pd.read_csv(hist_path)
    h = h[(h["dataset"] == dataset) & (h["bin_left"] < 0.6)]
    thr = json.loads(thr_path.read_text())[dataset]["threshold"]
    width = (h["bin_right"] - h["bin_left"]).iloc[0]
    col = np.where(h["bin_left"] < thr, BLUE, GREY)
    ax.bar(h["bin_left"] + width / 2, h["count"] / h["count"].sum(), width=width * 0.9, color=col, linewidth=0)
    ax.axvline(thr, color=INK, lw=0.6, ls=(0, (2, 2)))
    ax.text(thr + 0.012, ax.get_ylim()[1] * 0.92, f"threshold {thr:.3f}\n(density valley)", fontsize=5, color=INK,
            va="top")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("junction distance to nearest other sequence\n(same donor, IGHV, IGHJ, length)")
    ax.set_ylabel("fraction of sequences")
    ax.text(0.02, 0.97, "clonal\nrelatives", transform=ax.transAxes, fontsize=5, color=BLUE, va="top")
    ax.text(0.75, 0.6, "unrelated", transform=ax.transAxes, fontsize=5, color=INK2, va="top")
    if ax_in is None:
        return
    if not sens_path.exists():
        ax_in.set_axis_off()
        return
    sens = pd.read_csv(sens_path)
    for ds, c in (("ln_vaccine", BLUE), ("flu", ORANGE)):
        t = sens[(sens["dataset"] == ds) & (sens.get("auto") != True)].sort_values("threshold")  # noqa: E712
        if t.empty:
            continue
        ax_in.plot(t["threshold"], t["ari"], "-o", color=c, ms=2, lw=0.8, label=DATASET_NAMES[ds].split(",")[0])
        auto = sens[(sens["dataset"] == ds) & (sens.get("auto") == True)]  # noqa: E712
        if not auto.empty:
            ax_in.scatter(auto["threshold"], auto["ari"], marker="*", s=28, color=c, edgecolors="white",
                          linewidths=0.3, zorder=3)
    ax_in.set_ylim(0, 1.02)
    ax_in.set_xlabel("threshold")
    ax_in.set_ylabel("ARI vs authors' clones")
    ax_in.legend(loc="lower left", handletextpad=0.2, fontsize=5)
    ax_in.text(0.98, 0.96, "* automatic", transform=ax_in.transAxes, fontsize=5, color=INK2, ha="right", va="top")


def reliability_panel(ax):
    """Spearman-Brown reliability of a clone profile against clone size, per dataset."""
    n = np.arange(1, 61)
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]
    ends = []
    for ds, c in zip(DATASET_ORDER, cols):
        s = load_summary(ds)
        if s is None:
            continue
        rho = s["coherence"]["main"]["icc"]
        r = n * rho / (1 + (n - 1) * rho)
        ax.plot(n, r, color=c, lw=1)
        ends.append((DATASET_NAMES[ds], r[-1], c, rho))
    if not ends:
        missing(ax, "reliability vs clone size")
        return
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.text(60, 0.47, "reliability 0.5 = core clone", fontsize=5, color=INK2, ha="right", va="top")
    for name, _, c, rho in ends:
        n_half = int(np.ceil((1 - rho) / rho))  # Spearman-Brown: n rho / (1 + (n - 1) rho) = 0.5
        ax.plot([n_half], [n_half * rho / (1 + (n_half - 1) * rho)], "o", color=c, ms=2.5)
    handles = [plt.Line2D([], [], color=c, lw=1) for _, _, c, _ in ends]
    ax.legend(handles, [f"{name} (ICC {rho:.2f})" for name, _, _, rho in ends], loc="lower right",
              fontsize=4.8, handlelength=1.2, borderaxespad=0.1, labelspacing=0.25)
    ax.set_xlim(1, 60)
    ax.set_ylim(0, 1)
    ax.spines["bottom"].set_bounds(1, 60)
    ax.set_xlabel("cells sampled from a clone")
    ax.set_ylabel("reliability of the clone profile")


def kernel_toy_panel(fig, x, y, w, h, letter):
    """Two clones with the same centroid but different cell distributions."""
    rng = np.random.default_rng(3)
    a = np.vstack([rng.normal([-1.6, 0], 0.35, size=(30, 2)), rng.normal([1.6, 0], 0.35, size=(30, 2))])
    b = rng.normal([0, 0], 0.45, size=(60, 2))
    bg = np.vstack([rng.normal(c_, 0.5, size=(150, 2)) for c_ in ([-1.6, 0], [0, 0], [1.6, 0], [0, 1.6])])
    pw = (w - 6) / 2
    for k, (cells, name, col) in enumerate(((a, "clone A: split between\ntwo states", BLUE),
                                            (b, "clone B: one\nintermediate state", ORANGE))):
        ax = panel(fig, x + k * (pw + 6), y, pw, h * 0.62, letter if k == 0 else None)
        ax.scatter(bg[:, 0], bg[:, 1], s=1.2, color=LIGHT, linewidths=0, rasterized=True)
        ax.scatter(cells[:, 0], cells[:, 1], s=3, color=col, linewidths=0)
        m = cells.mean(axis=0)
        ax.scatter([m[0]], [m[1]], marker="X", s=30, color=INK, edgecolors="white", linewidths=0.5, zorder=4)
        ax.set_xticks([]), ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(name, fontsize=5.5, color=INK, pad=1)
        ax.set_xlim(-3, 3), ax.set_ylim(-1.6, 2.6)
    # distances: centroid vs kernel mean embedding (MMD with a Gaussian kernel)
    def mmd(p, q, bw=1.0):
        def k(u, v):
            d = ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)
            return np.exp(-d / (2 * bw ** 2))
        return float(np.sqrt(max(k(p, p).mean() + k(q, q).mean() - 2 * k(p, q).mean(), 0)))
    ax = panel(fig, x, y + h * 0.72, w, h * 0.28)
    vals = [np.linalg.norm(a.mean(0) - b.mean(0)), mmd(a, b)]
    ax.barh([1, 0], vals, height=0.55, color=[GREY, BLUE])
    ax.set_yticks([1, 0], ["centroid distance\n(mean profile)", "distribution distance\n(kernel profile, MMD)"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("distance between clone A and clone B")
    ax.set_title("same centroid (x), different distributions", fontsize=5.5, color=INK, pad=2)


_SPLIT_CACHE = {}


def split_demo_data():
    """Root split test in a continuum repertoire and in a four-programme repertoire (simulated)."""
    if _SPLIT_CACHE:
        return _SPLIT_CACHE
    import warnings

    import threadfin as tf
    from threadfin.programmes import split_test

    warnings.filterwarnings("ignore")
    for name, kw in (("continuum (1 programme)", dict(n_programmes=1)), ("4 programmes", dict())):
        ad = tf.sim.simulate_repertoire(random_state=1, **kw)
        tf.tl.clone_profiles(ad, basis="X_pca", context_key="context", donor_key="donor",
                             representation="kernel", verbose=False)
        tf.tl.find_programmes(ad, n_boot=10, embed=False, test_splits=False, assign_remaining=False, verbose=False)
        table = ad.uns["threadfin"]["profiles"]["clone_table"]
        core = table.index[table["core"]]
        feats = ad.uns["threadfin"]["profiles"]["features"].loc[core].to_numpy()
        lab = table.loc[core, "clone_programme"].astype(str).to_numpy()
        names = sorted(set(lab))
        # root split of the community tree: the two halves of an average-linkage tree of centroids
        from scipy.cluster.hierarchy import linkage, to_tree
        cents = np.vstack([feats[lab == k].mean(0) for k in names])
        root = to_tree(linkage(cents, "average")) if len(names) > 1 else None
        left = [names[i] for i in root.left.pre_order()] if root is not None else names
        res = split_test(feats, np.isin(lab, left), n_null=300, random_state=0, return_null=True)
        _SPLIT_CACHE[name] = (res, len(names))
    return _SPLIT_CACHE


def split_demo_panel(fig, x, y, w, h, letter):
    data = split_demo_data()
    gap = 9
    ph = (h - gap) / 2
    for k, (name, (res, groups)) in enumerate(data.items()):
        ax = panel(fig, x, y + k * (ph + gap), w, ph, letter if k == 0 else None)
        col = GREY if name.startswith("continuum") else BLUE
        lo = min(res["null"].min(), res["cluster_index"]) - 0.02
        hi = max(res["null"].max(), res["cluster_index"]) + 0.02
        ax.hist(res["null"], bins=np.linspace(lo, hi, 40), color=col, linewidth=0, alpha=0.8)
        ax.axvline(res["cluster_index"], color=INK, lw=1)
        ax.set_xlim(0.45, 0.88)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        verdict = "split" if res["pvalue"] < 0.05 else "merge: one programme"
        ax.set_title(f"simulated {name}: Leiden finds {groups} groups -> {verdict} (p = {res['pvalue']:.1g})",
                     fontsize=5.3, color=INK, pad=2, loc="left")
        if k == 1:
            ax.set_xlabel("2-means cluster index of the clone profiles (lower = more clearly separated)")
        else:
            ax.text(res["cluster_index"] + 0.005, ax.get_ylim()[1] * 0.9, "observed", fontsize=5, color=INK,
                    va="top")
            ax.text(res["null"].min() - 0.006, ax.get_ylim()[1] * 0.45, "single-Gaussian\nnull samples",
                    fontsize=5, color=INK2, va="center", ha="right")


# ------------------------------------------------------------------ concept panels


def caption(ax, text, fontsize=5.2, y=0.0):
    """Explanatory line under a schematic, wrapped to the panel width."""
    import textwrap

    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    chars = max(int(width_pt / (fontsize * 0.49)), 20)
    ax.text(0.0, y, "\n".join(textwrap.wrap(text, chars)), transform=ax.transAxes, fontsize=fontsize,
            color=INK2, va="bottom", linespacing=1.35)


def _ring(ax, cx, cy, r, t0, t1, color, lw=1.4, arrow=True):
    """Arc of the germinal-centre cycle, optionally with an arrow head at its end."""
    t = np.linspace(np.radians(t0), np.radians(t1), 80)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=color, lw=lw, solid_capstyle="round", zorder=1)
    if arrow:
        ax.annotate("", xy=(cx + r * np.cos(t[-1]), cy + r * np.sin(t[-1])),
                    xytext=(cx + r * np.cos(t[-6]), cy + r * np.sin(t[-6])),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7), zorder=1)


def gc_cycle_panel(ax):
    """The germinal-centre reaction has no start and no end, so cells cannot be ordered."""
    ax.set_axis_off()
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    cx, cy, r = 0.40, 0.62, 0.23
    _ring(ax, cx, cy, r, 105, 345, BLUE)
    _ring(ax, cx, cy, r, 15, 75, BLUE)
    ax.add_patch(plt.Circle((cx, cy), r * 0.60, color="#f3f6fb", zorder=0))
    ax.text(cx, cy + 0.035, "germinal\ncentre", ha="center", va="center", fontsize=5.8, color=INK2,
            linespacing=1.1)
    ax.text(cx, cy - 0.065, "no start,\nno end", ha="center", va="center", fontsize=5, color=MUTED,
            style="italic", linespacing=1.1)
    ax.text(cx - r - 0.02, cy + 0.02, "dark zone\ndivide and\nmutate", ha="right", va="center", fontsize=5.2,
            color=INK, linespacing=1.15)
    ax.text(cx + r + 0.02, cy + 0.10, "light zone\ntest the\nreceptor", ha="left", va="center", fontsize=5.2,
            color=INK, linespacing=1.15)
    ax.text(cx, cy + r + 0.06, "another round of selection", ha="center", va="bottom", fontsize=5.2, color=BLUE)
    ax.annotate("", xy=(cx + r + 0.24, cy - r - 0.03), xytext=(cx + r * 0.72, cy - r * 0.72),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3, mutation_scale=7))
    ax.text(cx + r + 0.25, cy - r - 0.03, "exit:\nplasma cell\nor memory", ha="left", va="center", fontsize=5.2,
            color=ORANGE, linespacing=1.15)
    caption(ax, "A cell's state says where it is in the cycle, not how far it has come. Ordering cells along a "
                "pseudotime needs a beginning and an end that a germinal centre does not have.")


def clone_as_distribution_panel(ax):
    """A clone is a family sampled repeatedly around the cycle: a distribution over states."""
    ax.set_axis_off()
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    rng = np.random.default_rng(4)
    cx, cy, r = 0.5, 0.66, 0.23
    _ring(ax, cx, cy, r, 105, 345, LIGHT, lw=3.5, arrow=False)
    _ring(ax, cx, cy, r, 15, 75, LIGHT, lw=3.5, arrow=False)
    clones = [("clone 1", BLUE, 215, 40, "cells caught dividing: sent back for another round"),
              ("clone 2", ORANGE, 350, 35, "cells caught leaving: plasma-cell output"),
              ("clone 3", AQUA, 90, 150, "cells all around the cycle")]
    for name, col, centre, spread in [(c[0], c[1], c[2], c[3]) for c in clones]:
        a = np.radians(rng.normal(centre, spread, 11))
        rr = r + rng.normal(0, 0.016, a.size)
        ax.scatter(cx + rr * np.cos(a), cy + rr * np.sin(a), s=8, color=col, edgecolors="white",
                   linewidths=0.35, zorder=3)
    ax.text(cx, cy, "cells of\nthree clones", ha="center", va="center", fontsize=5.4, color=INK2, linespacing=1.1)
    for k, (name, col, _, _, note) in enumerate(clones):
        y = 0.30 - 0.07 * k
        ax.scatter([0.04], [y], s=9, color=col, edgecolors="white", linewidths=0.3)
        ax.text(0.09, y, f"{name}: {note}", fontsize=5.1, color=INK2, va="center")
    caption(ax, "Threadfin describes each clone by the whole distribution of its cells, with a reliability that "
                "grows with the number of cells sampled from that clone.")


def clone_relationship_panel(ax):
    """Clones placed by how their cells are distributed: a map of clones, not of cells."""
    from matplotlib.colors import LinearSegmentedColormap

    ax.set_axis_off()
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    rng = np.random.default_rng(11)
    n = 26
    angle, rad = rng.uniform(0, 2 * np.pi, n), rng.uniform(0.1, 0.32, n)
    x, y = 0.5 + rad * np.cos(angle), 0.66 + rad * np.sin(angle) * 0.7
    score = (x - x.min()) / (x.max() - x.min())
    cmap = LinearSegmentedColormap.from_list("sel", ["#cde2fb", BLUE, "#123a6b"])
    for i in range(n):
        d = (x - x[i]) ** 2 + (y - y[i]) ** 2
        for j in np.argsort(d)[1:3]:
            ax.plot([x[i], x[j]], [y[i], y[j]], color=LIGHT, lw=0.5, zorder=1)
    ax.scatter(x, y, s=12 + 70 * rng.beta(1.5, 4, n), c=score, cmap=cmap, edgecolors="white", linewidths=0.4,
               zorder=3)
    ax.text(0.5, 1.0, "one point = one clone;  size = cells sampled", fontsize=5.2, color=MUTED, ha="center",
            va="top")
    ax.annotate("", xy=(0.95, 0.33), xytext=(0.05, 0.33),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    ax.text(0.05, 0.30, "clones whose cells look\nrecently selected", fontsize=5.1, color=INK2, ha="left",
            va="top", linespacing=1.15)
    ax.text(0.95, 0.30, "clones whose cells look\nsent back to divide", fontsize=5.1, color=INK2, ha="right",
            va="top", linespacing=1.15)
    caption(ax, "Clones are compared with one another, and the measured labels - divisions, affinity mutations, "
                "sorted zone, plasma-cell output - are tested across clones rather than cells.")


# ------------------------------------------------------------------ germinal-centre panels

GC_SETS = [("mouse_np", "NP-OVA, division reporter"), ("mouse_rbd", "RBD vaccine, division + zone sorts"),
           ("gc_np_pc", "NP-OVA, sorted zones and plasma cells")]


def gc_coherence_panel(ax):
    """How much of B-cell state is explained by clone identity in each model-antigen experiment."""
    rows = []
    for ds, label in GC_SETS:
        s = load_summary(ds)
        if s is None:
            continue
        c = s["coherence"]
        rows.append((label, 100 * c["icc"], 100 * c["null_mean"], c["p_value"], c["n_clones"]))
    if not rows:
        missing(ax, "variance explained by clone identity")
        return
    y = np.arange(len(rows))[::-1].astype(float)
    ax.barh(y + 0.18, [r[1] for r in rows], height=0.33, color=BLUE, label="observed")
    ax.barh(y - 0.18, [r[2] for r in rows], height=0.33, color=GREY, label="clones shuffled within samples")
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.3, yy + 0.18, f"p = {r[3]:.3g},  {r[4]:,} clones", va="center", fontsize=4.8, color=INK2)
    ax.set_yticks(y, [r[0].replace(", ", ",\n") for r in rows], fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.75)
    ax.set_xlabel("B-cell state explained by clone identity (%)")
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)


def gc_clone_map_panel(ax, dataset="mouse_rbd", colour="division_gate:mCherry-low",
                       label="cells of the clone in the\nmost-divided gate"):
    """Map of clones, coloured by how much the clone's cells had divided."""
    from matplotlib.colors import LinearSegmentedColormap

    tab = load_table(dataset, "clone_table.csv")
    if tab is None or "x" not in tab or colour not in tab:
        missing(ax, "clone map")
        return
    t = tab.dropna(subset=["x", "y", colour])
    cmap = LinearSegmentedColormap.from_list("sel", ["#cde2fb", BLUE, "#123a6b"])
    sc = ax.scatter(t["x"], t["y"], c=t[colour], cmap=cmap, s=1 + 5 * np.sqrt(t["n_cells"]),
                    edgecolors="white", linewidths=0.2, alpha=0.9, rasterized=True)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{len(t):,} clones, each a point", fontsize=5.5, color=INK, pad=2)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(label, fontsize=4.8, color=INK2, linespacing=1.05)
    cb.ax.tick_params(labelsize=4.8, length=1.5)
    cb.outline.set_visible(False)
    caption(ax, "Clones sit on a continuum rather than in separate groups: no split between groups of clones was "
                "significant in any of these experiments.", y=-0.24)


def gc_label_effects_panel(ax):
    """What explains how germinal-centre clones differ from one another."""
    rows = []
    pretty = {"division_gate:mCherry-low": "divisions since labelling", "zone_gate:DZ": "dark- vs light-zone sort",
              "rbd_bait:RBD+": "antigen binding (bait)", "mutation_frequency": "somatic mutation load",
              "isotype": "isotype", "w33l": "high-affinity W33L mutation"}
    for ds, label in GC_SETS:
        t = load_table(ds, "label_effects.csv")
        if t is None:
            continue
        for _, r in t.iterrows():
            if r["label"] in pretty:
                rows.append({"dataset": label.split(",")[0], "label": pretty[r["label"]], "r2": 100 * r["r2"],
                             "null": 100 * r["null_mean"], "p": r["p_value"]})
    if not rows:
        missing(ax, "what explains clone differences")
        return
    d = pd.DataFrame(rows).sort_values("r2")
    y = np.arange(len(d)).astype(float)
    ax.barh(y, d["r2"], height=0.6, color=np.where(d["p"] < 0.05, BLUE, GREY))
    ax.barh(y, d["null"], height=0.6, color="none", edgecolor=INK2, linewidth=0.5, linestyle=(0, (2, 1)))
    ax.set_yticks(y, [f"{r.label}" for r in d.itertuples()], fontsize=5)
    for yy, r in zip(y, d.itertuples()):
        ax.text(r.r2 + 0.3, yy, f"{r.dataset};  p = {r.p:.3g}", va="center", fontsize=4.6, color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(d["r2"]) * 1.9)
    ax.set_xlabel("differences between clones explained (%)")
    ax.text(1.0, -0.22, "dashed outline: the same label shuffled among clones of the same mouse;  "
                        "grey: not significant", transform=ax.transAxes, fontsize=4.8, color=INK2, ha="right",
            va="top")


def gc_memory_panel(ax):
    """Does a clone keep its state across the sorted compartments of the germinal centre?"""
    rows = []
    pretty = {"division_gate": "most- vs least-divided cells", "zone_gate": "dark- vs light-zone cells",
              "rbd_bait": "antigen-binding vs non-binding cells", "plasma_cell": "plasma cells vs GC cells"}
    for ds, label in GC_SETS:
        s = load_summary(ds)
        for key, m in ((s or {}).get("memory") or {}).items():
            if key in pretty:
                rows.append({"label": pretty[key], "dataset": label.split(",")[0], "m": m["memory_index"],
                             "lo": m["memory_index_ci"][0], "hi": m["memory_index_ci"][1], "n": m["n_clones"],
                             "p": m["p_value"]})
    if not rows:
        missing(ax, "clonal memory")
        return
    d = pd.DataFrame(rows).sort_values("m")
    y = np.arange(len(d)).astype(float)
    ax.errorbar(d["m"], y, xerr=[d["m"] - d["lo"], d["hi"] - d["m"]], fmt="o", ms=3.5, color=BLUE,
                ecolor=INK2, elinewidth=0.6, capsize=1.5, markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_yticks(y, [f"{r.label}\n({r.dataset}, {r.n} clones)" for r in d.itertuples()], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.15, 1)
    ax.set_xlabel("clonal memory index")
    ax.text(1.0, -0.3, "0 = a clone's cells in one compartment say nothing about its cells in the other;\n"
                       "1 = they are as alike as two samples of the same thing", transform=ax.transAxes,
            fontsize=4.8, color=INK2, ha="right", va="top", linespacing=1.3)


# ------------------------------------------------------------------ figures


def figure_2():
    fig = new_page("Figure 2", "Germinal-centre clones differ along a continuum of selection",
                   "Model-antigen immunisation: NP-OVA and an RBD vaccine, with cells sorted by what they had "
                   "been doing.")
    placeholder(panel(fig, 0, 4, 183, 26, "a"),
                "Experimental design (to draw): NP-OVA or RBD immunisation; germinal-centre B cells sorted by a "
                "division reporter, by dark/light zone, by antigen-binding bait, and plasma cells; paired "
                "single-cell RNA + BCR sequencing of each sorted gate")
    gc_coherence_panel(panel(fig, 8, 40, 66, 34, "b"))
    gc_clone_map_panel(panel(fig, 108, 38, 48, 38, "c"))
    gc_label_effects_panel(panel(fig, 30, 90, 62, 44, "d"))
    gc_memory_panel(panel(fig, 118, 90, 58, 34, "e"))
    gc_lineage_panel(panel(fig, 30, 150, 60, 28, "f"))
    placeholder(panel(fig, 112, 146, 71, 36, "g"),
                "Biological interpretation (to draw): a clone's cells spread around the cycle; clones caught "
                "dividing look alike, clones caught in the light zone look alike, and about half of what "
                "distinguishes a clone is carried across the cycle")
    placeholder(panel(fig, 0, 202, 183, 28, "h"),
                "Validation to add: an experiment that follows or perturbs the same clones over time, which these "
                "snapshots cannot replace")
    save(fig, "Figure_2")


def figure_benchmarks_internal():
    fig = new_page("Benchmark figure", "Simulated repertoires with known clonal programmes",
                   "Programme recovery, calibration of every test under the null, clonal memory, speed and scaling.")
    placeholder(panel(fig, 0, 4, 52, 62, "a"),
                "Simulation design\n(schematic): donors and\nsamples with technical\nshifts, cell states,"
                "\nclonal programmes\n(mixtures of states),\nZipf clone sizes, time\npoints with a known\n"
                "switching rate")
    sim_recovery(fig, 88, 4, 95, 70, "b")
    sim_by_size(panel(fig, 8, 100, 50, 40, "c"))
    sim_calibration(panel(fig, 78, 100, 105, 40, "d"))
    sim_memory(panel(fig, 8, 158, 50, 42, "e"))
    sim_auto(panel(fig, 78, 158, 50, 42, "f"))
    sim_runtime(panel(fig, 158, 158, 25, 42, "g", letter_dx=-27))
    sim_scaling(panel(fig, 8, 220, 50, 38, "h"), panel(fig, 78, 220, 50, 38, "i"))
    save(fig, "Benchmarks_internal", INTERNAL)


def gc_lineage_panel(ax):
    """Where a cell sits in its clone's mutation-based family tree, by sorted compartment."""
    short = {"plasma cell vs light zone+Myc+ light zone+dark zone": "plasma cells vs their\ngerminal-centre sisters",
             "dark zone vs light zone+Myc+ light zone": "dark-zone vs\nlight-zone sisters",
             "Myc+ light zone vs light zone": "recently selected vs\nother light-zone sisters",
             "mCherry-low vs mCherry-high": "most- vs least-divided\nsisters"}
    rows = []
    for ds, name in (("gc_np_pc", "NP-OVA, sorted zones"), ("mouse_np", "NP-OVA, divisions")):
        path = DATA / ds / "lineage" / "within_clone_tree_comparisons.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for _, r in d[d["measure"] == "depth"].iterrows():
            if np.isfinite(r["wilcoxon_p"]):
                rows.append({"label": short.get(r["comparison"], r["comparison"]), "dataset": name,
                             "diff": r["mean_difference"], "n": r["n_clones"], "p": r["wilcoxon_p"]})
    if not rows:
        missing(ax, "position in the clone's family tree vs sorted compartment")
        return
    d = pd.DataFrame(rows).sort_values("diff")
    y = np.arange(len(d)).astype(float)
    ax.barh(y, d["diff"], height=0.55, color=np.where(d["p"] < 0.05, BLUE, GREY))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, d["label"], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    lim = max(abs(d["diff"])) * 2.6
    ax.set_xlim(-lim, lim)
    for yy, r in zip(y, d.itertuples()):
        right = r.diff >= 0
        ax.text(r.diff + (0.06 if right else -0.06) * lim, yy, f"{r.n} clones, p = {r.p:.2g}", va="center",
                fontsize=4.6, color=INK2, ha="left" if right else "right")
    ax.set_xlabel("mutations from the unmutated ancestor,\ndifference within a clone")
    caption(ax, "An independent check that uses only the receptor sequences: cells further from the unmutated "
                "ancestor arose later in their clone's history. Taken within clones.", y=-0.55)


def figure_1():
    fig = new_page("Figure 1", "Reading a germinal centre at the level of clones",
                   "A clone is a family of cells sampled around a cycle that has no beginning and no end.")
    gc_cycle_panel(panel(fig, 0, 8, 57, 54, "a"))
    clone_as_distribution_panel(panel(fig, 63, 8, 57, 54, "b"))
    clone_relationship_panel(panel(fig, 126, 8, 57, 54, "c"))
    placeholder(panel(fig, 0, 70, 183, 32, "d"),
                "Workflow schematic (to draw): paired scRNA-seq + BCR-seq -> clones within each donor (IGHV, IGHJ, "
                "junction length, data-driven distance threshold) -> expression embedding without immunoglobulin "
                "genes -> clone profiles relative to the cells sampled with them -> is state inherited within "
                "clones? -> which clones behave alike (or a continuum)? -> what explains the differences between "
                "clones? -> do clones keep their state?")
    kernel_toy_panel(fig, 4, 114, 78, 48, "e")
    reliability_panel(panel(fig, 108, 114, 55, 42, "f"))
    placeholder(panel(fig, 0, 174, 90, 36, "g"),
                "Clonal memory (schematic): the same clone sampled twice, compared with a random clone of the same "
                "donor after removing the noise expected from sampling few cells")
    placeholder(panel(fig, 96, 174, 87, 36, "h"),
                "Programmes or a continuum (schematic): clones are grouped only when the split between groups is "
                "real; otherwise they vary along a continuum")
    placeholder(panel(fig, 0, 220, 183, 38, "i"),
                "Outputs at a glance (to draw): is state inherited within clones | which clones behave alike, or a "
                "continuum | what explains the differences between clones | do clones keep their state | which "
                "genes are clonally inherited")
    save(fig, "Figure_1")


def figure_methods_internal():
    """Methodological panels, kept outside the repository (see internal_validation/figures)."""
    fig = new_page("Methods figure", "Clone definition, programme splits and benchmarks",
                   "Internal validation; not part of the manuscript's main figures.")
    clone_threshold_panel(panel(fig, 8, 6, 62, 40, "a"), panel(fig, 90, 8, 32, 30))
    split_demo_panel(fig, 8, 60, 92, 42, "b")
    save(fig, "Methods_internal", INTERNAL)


def main():
    figure_1()
    figure_2()
    figure_methods_internal()
    figure_benchmarks_internal()


if __name__ == "__main__":
    main()
