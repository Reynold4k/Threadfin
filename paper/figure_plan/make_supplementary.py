#!/usr/bin/env python
"""Supplementary figures: the evidence behind each main-figure claim.

Each page answers one sceptical question about the main figures - is the clone
definition right, is the null the right null, is the effect technical, is the
dataset powered - using the same committed results.

Usage:
    python make_supplementary.py          # writes Supplementary_1..N (.pdf and .png)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_figures import (AQUA, BLUE, DATA, GREEN, GREY, INK, INK2, LIGHT, MAGENTA, MUTED, ORANGE, VIOLET,
                          YELLOW, caption, load_summary,
                          load_table, log_ticks, missing, new_page, panel, placeholder, plt, save, spread)

HERE = Path(__file__).resolve().parent

MODEL = [("mouse_np", "NP-OVA, divisions"), ("mouse_rbd", "RBD vaccine, divisions + zones"),
         ("gc_np_pc", "NP-OVA, sorted fates")]
INFECTION = [("flu_lung", "influenza, lung and node"), ("malaria", "Plasmodium, time course"),
             ("ebv", "EBV organoids")]
NON_GC = [("bone_marrow_pc", "bone marrow plasma cells"), ("flu", "blood, influenza vaccine"),
          ("tonsil", "tonsil"), ("stephenson", "blood, COVID-19")]
ALL_SETS = MODEL + INFECTION + NON_GC + [("ln_vaccine", "lymph node, mRNA vaccine")]


def _datasets_with_results(sets):
    return [(d, n) for d, n in sets if (DATA / d / "summary.json").exists()]


# --------------------------------------------------------------------------- S1: clone definition


def clone_threshold_histograms(fig, x, y, w, h, letter):
    """Distance to the nearest other receptor, per dataset, with the threshold chosen."""
    hist = DATA / "clone_threshold_histograms.csv"
    thr = DATA / "clone_thresholds.json"
    if not hist.exists():
        missing(panel(fig, x, y, w, h, letter), "distance-to-nearest distributions")
        return
    hh = pd.read_csv(hist)
    thresholds = json.loads(thr.read_text()) if thr.exists() else {}
    names = [d for d in hh["dataset"].unique()]
    ncol = 3
    pw, ph = (w - 2 * 14) / ncol, (h - 12) / 2
    for k, ds in enumerate(names[:6]):
        r, c = divmod(k, ncol)
        ax = panel(fig, x + c * (pw + 14), y + r * (ph + 12), pw, ph, letter if k == 0 else None)
        sub = hh[(hh["dataset"] == ds) & (hh["bin_left"] < 0.6)]
        t = thresholds.get(ds, {}).get("threshold")
        width = (sub["bin_right"] - sub["bin_left"]).iloc[0]
        colour = np.where(sub["bin_left"] < (t or 0), BLUE, GREY)
        ax.bar(sub["bin_left"] + width / 2, sub["count"] / max(sub["count"].sum(), 1), width=width * 0.9,
               color=colour, linewidth=0)
        if t:
            ax.axvline(t, color=INK, lw=0.6, ls=(0, (2, 2)))
            ax.text(t + 0.015, ax.get_ylim()[1] * 0.9, f"{t:.3f}", fontsize=4.6, color=INK, va="top")
        ax.set_title(ds, fontsize=5.2, color=INK, pad=2)
        ax.set_xlim(0, 0.6)
        if r == 1:
            ax.set_xlabel("distance to the nearest\nother receptor", fontsize=5)
        if c == 0:
            ax.set_ylabel("fraction", fontsize=5)


def clone_threshold_sensitivity(ax):
    """Agreement with the authors' own clone calls, over a range of thresholds."""
    path = DATA / "clone_threshold_sensitivity.csv"
    if not path.exists():
        missing(ax, "agreement with the authors' clones")
        return
    d = pd.read_csv(path)
    for ds, col in zip(d["dataset"].unique(), (BLUE, ORANGE, AQUA)):
        sub = d[(d["dataset"] == ds) & (d.get("auto") != True)].sort_values("threshold")  # noqa: E712
        ax.plot(sub["threshold"], sub["ari"], "-o", color=col, ms=2.5, lw=1, label=ds)
        auto = d[(d["dataset"] == ds) & (d.get("auto") == True)]                          # noqa: E712
        if not auto.empty:
            ax.scatter(auto["threshold"], auto["ari"], marker="*", s=40, color=col, edgecolors="white",
                       linewidths=0.4, zorder=4)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("junction distance threshold")
    ax.set_ylabel("agreement with the authors' clones")
    ax.legend(loc="lower left", handletextpad=0.3, fontsize=5)
    caption(ax, "Stars mark the threshold chosen automatically from the data. Only two of these studies published "
                "their own clone assignments.", y=-0.3)


# --------------------------------------------------------------------------- S2: nulls and power


def coherence_all_nulls(ax):
    """The same statistic against three different nulls, per dataset."""
    rows = []
    for ds, name in _datasets_with_results(ALL_SETS):
        s = load_summary(ds)
        c = s["coherence"]
        rows.append({"dataset": name, "observed": 100 * c["icc"], "within sample": 100 * c["null_mean"],
                     "clones": c["n_clones"]})
    if not rows:
        missing(ax, "nulls")
        return
    d = pd.DataFrame(rows).sort_values("observed")
    y = np.arange(len(d)).astype(float)
    ax.barh(y + 0.2, d["observed"], height=0.36, color=BLUE, label="observed")
    ax.barh(y - 0.2, d["within sample"], height=0.36, color=GREY, label="clones shuffled within samples")
    ax.set_yticks(y, [f"{r.dataset}\n({r.clones:,} clones)" for r in d.itertuples()], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("B-cell state explained by clone identity (%)")
    ax.legend(loc="lower right", fontsize=5, handletextpad=0.3)


def power_panel(ax):
    """How many cells a clone needs before its profile is usable, and how many clones reach it."""
    rows = []
    for ds, name in _datasets_with_results(ALL_SETS):
        s = load_summary(ds)
        t = load_table(ds, "clone_table.csv")
        if t is None:
            continue
        rows.append({"dataset": name, "cells_needed": s["coherence"]["min_cells_reliability_0.5"],
                     "clones_profiled": len(t), "core": int((t["reliability"] >= 0.5).sum())})
    if not rows:
        missing(ax, "power")
        return
    d = pd.DataFrame(rows).sort_values("cells_needed")
    y = np.arange(len(d)).astype(float)
    ax.barh(y, d["cells_needed"], height=0.6, color=BLUE)
    for yy, r in zip(y, d.itertuples()):
        ax.text(r.cells_needed + 0.3, yy, f"{r.core:,} of {r.clones_profiled:,} clones reach it", va="center",
                fontsize=4.6, color=INK2)
    ax.set_yticks(y, d["dataset"], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(d["cells_needed"]) * 2.4)
    ax.set_xlabel("cells a clone needs for a reliable profile")


# --------------------------------------------------------------------------- S3: continuum vs groups


def split_test_panel(ax):
    """Why most datasets are reported as a continuum: the split test at the root."""
    rows = []
    for ds, name in _datasets_with_results(ALL_SETS):
        s = load_summary(ds)
        sp, pr = s.get("programme_splits", {}), s.get("programmes", {})
        if not sp:
            continue
        rows.append({"dataset": name, "p": sp.get("root_pvalue"), "groups": pr.get("n", 0),
                     "communities": sp.get("n_leiden_communities")})
    rows = [r for r in rows if r["p"] is not None]
    if not rows:
        missing(ax, "split test")
        return
    d = pd.DataFrame(rows).sort_values("p")
    y = np.arange(len(d))[::-1].astype(float)
    ax.scatter(d["p"].clip(lower=1e-60), y, s=18, color=np.where(d["p"] < 0.05, BLUE, GREY),
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.axvline(0.05, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{r.dataset}\nLeiden found {r.communities}, reported {r.groups}" for r in d.itertuples()],
                  fontsize=4.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("p for the first split between groups of clones")
    caption(ax, "Left of the dashed line: the split is real and groups are reported. Right of it: the clones vary "
                "along a continuum, and clustering them anyway would invent structure.", y=-0.3)


# --------------------------------------------------------------------------- S4: technical controls


def within_library_panel(ax):
    """Sort gates were sequenced separately: the same tests run inside single libraries."""
    path = DATA / "mouse_gc_deep_dive" / "summary.json"
    if not path.exists():
        missing(ax, "within-library controls")
        return
    s = json.loads(path.read_text())
    rows = []
    for key, val in s.items():
        if "within_library_profile" in key and isinstance(val, dict) and "r2" in val:
            label = key.split("[")[-1].rstrip("]") if "[" in key else key
            arm = key.split("][")[0].split("[")[-1] if "][" in key else ""
            rows.append({"what": f"{label} ({arm})" if arm else label, "r2": 100 * val["r2"],
                         "null": 100 * val["null_mean"], "p": val["p_value"], "n": val["n_clones"]})
    if not rows:
        missing(ax, "within-library controls")
        return
    d = pd.DataFrame(rows).sort_values("r2")
    y = np.arange(len(d)).astype(float)
    ax.barh(y, d["r2"], height=0.6, color=np.where(d["p"] < 0.05, BLUE, GREY))
    ax.barh(y, d["null"], height=0.6, color="none", edgecolor=INK2, linewidth=0.45, linestyle=(0, (2, 1)))
    for yy, r in zip(y, d.itertuples()):
        ax.text(r.r2 + 0.4, yy, f"{r.n} clones, p = {r.p:.2g}", va="center", fontsize=4.6, color=INK2)
    ax.set_yticks(y, d["what"], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(d["r2"]) * 1.7)
    ax.set_xlabel("differences between clones explained (%), within a single library")
    caption(ax, "Each clone is compared only with cells from its own library, so a difference between sort gates "
                "cannot come from the gates having been sequenced separately.", y=-0.26)


# --------------------------------------------------------------------------- S5: sequences


def sequence_selection_panel(ax):
    """Selection measured from the antibody sequence alone, against what the cells were doing."""
    rows = []
    for ds, name in MODEL:
        f = DATA / ds / "sequence_selection_vs_labels.csv"
        sel = DATA / ds / "sequence_selection.csv"
        if not f.exists() or not sel.exists():
            continue
        t = pd.read_csv(f)
        s = pd.read_csv(sel)
        for _, r in t.iterrows():
            rows.append({"what": f"{name}:\n{r['label'].split(':')[0]}", "rho": r["spearman_rho"],
                         "p": r["p_value"], "n": r["n_clones"],
                         "excess": s["observed_fraction"].mean() - s["expected_fraction"].mean()})
    if not rows:
        missing(ax, "sequence-level selection")
        return
    d = pd.DataFrame(rows).sort_values("rho")
    y = np.arange(len(d)).astype(float)
    ax.barh(y, d["rho"], height=0.6, color=np.where(d["p"] < 0.05, BLUE, GREY))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, d["what"], fontsize=4.6)
    ax.tick_params(axis="y", length=0)
    lim = max(abs(d["rho"])) * 1.9
    ax.set_xlim(-lim, lim)
    for yy, r in zip(y, d.itertuples()):
        ax.text(r.rho + (0.03 if r.rho >= 0 else -0.03) * lim, yy, f"{r.n} clones, p = {r.p:.2g}", va="center",
                fontsize=4.5, color=INK2, ha="left" if r.rho >= 0 else "right")
    ax.set_xlabel("correlation between sequence-level selection and what the cells were doing")
    caption(ax, "Sequence-level selection is the excess of amino-acid-changing mutations over what that clone's "
                "own germline gives by chance. It is largely unrelated to the clone's transcriptional behaviour, "
                "so the two measurements are not restatements of each other.", y=-0.3)


# --------------------------------------------------------------------------- benchmark

BENCH = HERE.parents[2] / "internal_validation" / "benchmark_results"

# What each published tool was built to do. "yes" means the tool provides it directly, not that a
# determined user could assemble it. Checked against each tool's paper and documentation.
CAPABILITIES = pd.DataFrame(
    [
        ["Threadfin", "yes", "yes", "yes", "yes", "yes", "yes"],
        ["scirpy\n(Bioinformatics 2020)", "yes", "no", "partly", "no", "no", "no"],
        ["Dandelion\n(Nat Biotechnol 2024)", "yes", "no", "no", "no", "no", "no"],
        ["CoNGA\n(Nat Biotechnol 2022)", "partly", "no", "partly", "no", "no", "no"],
        ["Benisse\n(Nat Mach Intell 2022)", "yes", "no", "partly", "no", "no", "no"],
        ["sciCSR\n(Nat Methods 2023)", "no", "no", "no", "no", "partly", "no"],
        ["BiGCN\n(Small Methods 2026)", "yes", "no", "partly", "no", "no", "no"],
    ],
    columns=["tool", "clones from\nhypermutated\nreceptors", "is state\ninherited\nwithin clones?",
             "a description\nof each clone", "are there\ngroups, or a\ncontinuum?",
             "do clones keep\ntheir state?", "a test with\nclones as\nreplicates"],
).set_index("tool")


def capability_matrix(ax):
    """Which of the questions each published tool can answer."""
    from matplotlib.colors import ListedColormap

    code = {"yes": 2, "partly": 1, "no": 0}
    m = CAPABILITIES.map(lambda v: code[v]).to_numpy()
    cmap = ListedColormap(["#f0f0ec", "#cde2fb", BLUE])
    ax.imshow(m, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(m.shape[1]), CAPABILITIES.columns, fontsize=4.6)
    ax.set_yticks(range(m.shape[0]), CAPABILITIES.index, fontsize=4.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = CAPABILITIES.iat[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=4.4,
                    color="white" if m[i, j] == 2 else INK2)
    ax.set_xticks(np.arange(-0.5, m.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    caption(ax, "Only tools that integrate single-cell BCR with gene expression and were published in journals "
                "with an impact factor of at least five are shown. \"Partly\" means the tool provides something "
                "related but not the quantity itself.", y=-0.42)


def benchmark_scores(ax):
    """How much of an experimentally measured clone property each tool's output explains."""
    files = sorted(BENCH.glob("*_benchmark.csv")) if BENCH.exists() else []
    if not files:
        missing(ax, "benchmark scores")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d["status"] == "ok") & (d["label"] != "random label (calibration)")]
    if d.empty:
        missing(ax, "benchmark scores")
        return
    d["what"] = d["dataset"] + ": " + d["label"]
    piv = d.pivot_table(index="method", columns="what", values="explained")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order]
    cols = [BLUE, ORANGE, AQUA, "#eda100", MUTED, GREEN, VIOLET]
    y = np.arange(len(piv)).astype(float)
    for k, what in enumerate(piv.columns):
        ax.scatter(100 * piv[what], y + (k - len(piv.columns) / 2) * 0.1, s=13,
                   color=cols[k % len(cols)], edgecolors="white", linewidths=0.3, label=what, zorder=3)
    ax.set_yticks(y, [i.replace(" (", "\n(") for i in piv.index], fontsize=4.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("measured clone property explained (%)")
    ax.legend(fontsize=4.2, loc="lower right", handletextpad=0.2, labelspacing=0.2, borderaxespad=0.2)
    caption(ax, "Each point is one dataset and one experimentally measured clone property. All methods are "
                "scored on the same clones (at least three cells each) with the same statistic, and every "
                "method was also run on a label shuffled among clones, where all of them correctly found "
                "nothing.", y=-0.3)


def benchmark_calibration(ax):
    """Under a label that means nothing, does the method still report something?"""
    files = sorted(BENCH.glob("*_benchmark.csv")) if BENCH.exists() else []
    if not files:
        missing(ax, "calibration")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d["status"] == "ok") & (d["label"] == "random label (calibration)")]
    if d.empty:
        missing(ax, "calibration")
        return
    g = d.groupby("method")["p_value"].apply(lambda p: (p < 0.05).mean())
    g = g.sort_values()
    y = np.arange(len(g)).astype(float)
    ax.barh(y, g.to_numpy(), height=0.6, color=np.where(g.to_numpy() > 0.1, ORANGE, BLUE))
    ax.axvline(0.05, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(y, [i.replace(" (", "\n(") for i in g.index], fontsize=4.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(0.2, g.max() * 1.2))
    ax.set_xlabel("fraction of random labels called significant")
    caption(ax, "A method that reports structure for a label flipped at random is not usable for discovery. "
                "Dashed line: the nominal 5%.", y=-0.3)


# --------------------------------------------------------------------------- pages


def supplementary_1():
    fig = new_page("Supplementary Figure 1", "Are the clones defined correctly?",
                   "Everything downstream rests on which cells are grouped into a clone.")
    clone_threshold_histograms(fig, 0, 4, 183, 72, "a")
    clone_threshold_sensitivity(panel(fig, 20, 96, 70, 44, "b"))
    placeholder(panel(fig, 110, 92, 73, 48, "c"),
                "Clone size distributions per dataset (to add): most clones hold one or two cells, which is why "
                "every clone profile carries a reliability")
    save(fig, "Supplementary_1")


def supplementary_2():
    fig = new_page("Supplementary Figure 2", "Is the comparison fair, and is there enough data?",
                   "The null model and the power of every dataset used in the main figures.")
    coherence_all_nulls(panel(fig, 24, 6, 70, 60, "a"))
    power_panel(panel(fig, 112, 6, 66, 60, "b"))
    split_test_panel(panel(fig, 24, 86, 80, 56, "c"))
    save(fig, "Supplementary_2")


def supplementary_3():
    fig = new_page("Supplementary Figure 3", "Could the germinal-centre result be technical?",
                   "Sort gates were sequenced as separate libraries; the effects are re-measured inside single "
                   "libraries, and against the receptor sequences.")
    within_library_panel(panel(fig, 24, 6, 78, 40, "a"))
    sequence_selection_panel(panel(fig, 24, 66, 78, 44, "b"))
    placeholder(panel(fig, 118, 6, 65, 40, "c"),
                "Lineage trees (to add): example clone trees with the sorted compartment of each cell marked")
    save(fig, "Supplementary_3")


def supplementary_4():
    fig = new_page("Supplementary Figure 4", "What other tools can and cannot do",
                   "Every published method that integrates single-cell BCR with gene expression, asked the same "
                   "question on the same clones.")
    capability_matrix(panel(fig, 24, 10, 100, 48, "a"))
    benchmark_scores(panel(fig, 24, 86, 80, 46, "b"))
    benchmark_calibration(panel(fig, 24, 156, 80, 40, "c"))
    save(fig, "Supplementary_4")


def main():
    supplementary_1()
    supplementary_2()
    supplementary_3()
    supplementary_4()


if __name__ == "__main__":
    main()
