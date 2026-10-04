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

import schematics as sk
from make_figures import (AQUA, BLUE, DATA, GREEN, GREY, INK, INK2, LIGHT, MAGENTA, MUTED, ORANGE,
                          VIOLET, YELLOW, caption, label_effects_panel, lineage_tree_gallery,
                          lineage_vs_state_panel, load_summary, load_table, log_ticks, missing,
                          new_page, panel, placeholder, plt, save, spread)

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
    pretty = {"mouse_np": "germinal centre, NP-OVA", "mouse_rbd": "germinal centre, RBD vaccine",
              "flu_lung": "influenza infection", "malaria": "Plasmodium, days 0-14",
              "bone_marrow_pc": "bone marrow and blood", "ln_vaccine": "lymph node, mRNA vaccine",
              "flu": "blood, influenza vaccine"}
    ncol = 4
    gap = 9.0
    pw, ph = (w - 24 - (ncol - 1) * gap) / ncol, (h - 14) / 2
    for k, ds in enumerate(names[:8]):
        r, c = divmod(k, ncol)
        ax = panel(fig, x + 14 + c * (pw + gap), y + r * (ph + 12), pw, ph, letter if k == 0 else None)
        sub = hh[(hh["dataset"] == ds) & (hh["bin_left"] < 0.6)]
        t = thresholds.get(ds, {}).get("threshold")
        width = (sub["bin_right"] - sub["bin_left"]).iloc[0]
        colour = np.where(sub["bin_left"] < (t or 0), BLUE, GREY)
        ax.bar(sub["bin_left"] + width / 2, sub["count"] / max(sub["count"].sum(), 1), width=width * 0.9,
               color=colour, linewidth=0)
        if t:
            ax.axvline(t, color=INK, lw=0.6, ls=(0, (2, 2)))
            ax.text(t + 0.015, ax.get_ylim()[1] * 0.9, f"{t:.3f}", fontsize=4.6, color=INK, va="top")
        ax.set_title(pretty.get(ds, ds), fontsize=5.0, color=INK, pad=2)
        ax.set_xlim(0, 0.6)
        ax.set_yticks([])
        if r == 1 or k + ncol >= len(names[:8]):
            ax.set_xlabel("distance to the nearest\nother receptor", fontsize=4.8)
        if c == 0:
            ax.set_ylabel("receptors", fontsize=5)


def clone_threshold_agreement(ax):
    """Agreement with the authors' own clone calls, over a range of thresholds."""
    path = DATA / "clone_threshold_sensitivity.csv"
    if not path.exists():
        missing(ax, "agreement with the authors' clones")
        return
    d = pd.read_csv(path)
    if "ari" not in d:
        missing(ax, "agreement with the authors' clones")
        return
    names = {"ln_vaccine": "lymph node, mRNA vaccine", "flu": "blood, influenza vaccine"}
    for (ds, label), col in zip(names.items(), (BLUE, ORANGE)):
        sub = d[(d["dataset"] == ds) & d["ari"].notna()].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["threshold"], sub["ari"], "-o", color=col, ms=2.5, lw=1, label=label)
        auto = sub[sub["auto"] == True]                                                    # noqa: E712
        if not auto.empty:
            ax.scatter(auto["threshold"], auto["ari"], marker="*", s=46, color=col, edgecolors="white",
                       linewidths=0.4, zorder=4)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("junction distance threshold")
    ax.set_ylabel("agreement with the authors' clones\n(adjusted Rand index)")
    ax.legend(loc="lower left", handletextpad=0.3, fontsize=4.8, borderaxespad=0.2)
    caption(ax, "Stars mark the threshold chosen automatically from each dataset. Only these two studies "
                "published their own clone assignments; at the automatic threshold Threadfin agrees with "
                "them at 0.91 and 0.95.", mm_below=9)


def clone_threshold_sensitivity(ax):
    """How the headline result moves when the clone threshold is moved."""
    path = DATA / "clone_threshold_sensitivity.csv"
    if not path.exists():
        missing(ax, "sensitivity of clone coherence to the threshold")
        return
    d = pd.read_csv(path)
    names = {"mouse_np": ("germinal centre, NP-OVA", BLUE), "mouse_rbd": ("germinal centre, RBD vaccine", AQUA),
             "flu_lung": ("influenza infection", GREEN), "malaria": ("Plasmodium, days 0-14", ORANGE),
             "bone_marrow_pc": ("bone marrow and blood", VIOLET), "ln_vaccine": ("lymph node, mRNA", YELLOW),
             "flu": ("blood, influenza vaccine", MAGENTA)}
    for ds, (label, col) in names.items():
        sub = d[d["dataset"] == ds].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["threshold"], 100 * sub["icc"], "-", color=col, lw=0.9, label=label)
        auto = sub[sub["auto"] == True]                                                    # noqa: E712
        if not auto.empty:
            ax.scatter(auto["threshold"], 100 * auto["icc"], marker="*", s=40, color=col, edgecolors="white",
                       linewidths=0.4, zorder=4)
    ax.set_ylim(0, None)
    ax.set_xlabel("junction distance threshold")
    ax.set_ylabel("B-cell state explained by\nclone identity (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.4, fontsize=4.6,
              borderaxespad=0.0, labelspacing=0.35)
    caption(ax, "Germinal-centre results are flat across a four-fold range of thresholds. Where clones are "
                "barely expanded a looser threshold merges unrelated receptors and the figure falls, yet stays "
                "far above the shuffled control.", mm_below=9)


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
                "along a continuum, and clustering them anyway would invent structure.", mm_below=9)


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
            pretty = {"division_gate": "divisions since labelling", "zone_gate": "dark- vs light-zone",
                      "rbd_bait": "antigen binding (bait)",
                      "np_within_library_profile_vs_division": "divisions since labelling"}
            nice = pretty.get(label, label.replace("_", " "))
            arm_nice = {"RBD protein": "RBD protein vaccine", "mRNA": "RBD mRNA vaccine"}.get(arm, arm)
            rows.append({"what": f"{nice}\n({arm_nice})" if arm_nice else f"{nice}\n(NP-OVA)",
                         "r2": 100 * val["r2"],
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
                "cannot come from the gates having been sequenced separately.", mm_below=9)


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
            nice = {"division_gate": "divisions", "zone_gate": "dark- vs light-zone",
                    "rbd_bait": "antigen binding", "fate": "sorted fate"}
            key = r["label"].split(":")[0]
            rows.append({"what": f"{name}:\n{nice.get(key, key)}", "rho": r["spearman_rho"],
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
    ax.set_xlabel("correlation between selection on the antibody\nand what the clone's cells were doing")
    caption(ax, "Sequence-level selection is the excess of amino-acid-changing mutations over what that clone's "
                "own germline gives by chance. It is largely unrelated to the clone's transcriptional behaviour, "
                "so the two measurements are not restatements of each other.", mm_below=12)


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
    names = {"mouse_np": "NP-OVA", "mouse_rbd": "RBD vaccine", "gc_np_pc": "NP-OVA sorted fates",
             "flu_lung": "influenza infection", "bone_marrow_pc": "bone marrow", "ln_vaccine": "lymph node"}
    d["what"] = d["dataset"].map(lambda v: names.get(v, v)) + ": " + d["label"]
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
    ax.legend(fontsize=4.2, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.2,
              labelspacing=0.3, borderaxespad=0.0)
    ax.set_xlim(left=-2)
    caption(ax, "Each point is one dataset and one experimentally measured clone property. All methods are "
                "scored on the same clones (at least three cells each) with the same statistic, and every "
                "method was also run on a label shuffled among clones.", mm_below=9)


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
                "Dashed line: the nominal 5%.", mm_below=9)



# --------------------------------------------------------------------------- clone sizes and reliability


def clone_size_panel(ax):
    """How big the clones actually are, which is what decides what can be said about them."""
    sets = [(d, n) for d, n in ALL_SETS + [("malaria_late", "Plasmodium, days 10-42")]
            if (DATA / d / "summary.json").exists()]
    rows = []
    for ds, name in sets:
        q = json.loads((DATA / ds / "summary.json").read_text()).get("qc")
        if not q:
            continue
        rows.append({"dataset": name, "clones": q["n_clones"], "ge2": q["clones_ge2"] / q["n_clones"],
                     "ge10": q["clones_ge10"] / q["n_clones"], "largest": q["largest_clone"]})
    if not rows:
        missing(ax, "clone sizes")
        return
    d = pd.DataFrame(rows).sort_values("ge2")
    y = np.arange(len(d)).astype(float)
    ax.barh(y + 0.2, 100 * d["ge2"], height=0.36, color=GREY, label="two or more cells")
    ax.barh(y - 0.2, 100 * d["ge10"], height=0.36, color=BLUE, label="ten or more cells")
    for yy, r in zip(y, d.itertuples()):
        ax.text(100 * r.ge2 + 0.6, yy + 0.2, f"{r.clones:,} clones, largest {r.largest:,}", va="center",
                fontsize=4.4, color=INK2)
    ax.set_yticks(y, d["dataset"], fontsize=4.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 40)
    ax.set_xlabel("share of all clones (%)")
    ax.legend(loc="lower right", fontsize=4.8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    caption(ax, "Between 76% and 98% of clones hold a single cell, and few reach ten. Threadfin states a "
                "reliability for every clone rather than imposing a size cut-off, and the tests weight clones "
                "by it.", mm_below=9)


def gc_detail_panel(fig, x, y, w, h, letter):
    """Every measured label of every model-antigen experiment, side by side."""
    head = panel(fig, x, y, w, 0.1, letter)
    head.set_axis_off()
    head.text(0, 1.0, "What explains the differences between germinal-centre clones, experiment by experiment",
              transform=head.transAxes, fontsize=6.0, color=INK, va="bottom")
    specs = [("mouse_np", "NP-OVA, division reporter",
              {"division_gate:mCherry-low": "divisions since labelling", "isotype": "isotype",
               "w33l": "high-affinity W33L mutation", "mutation_frequency": "somatic mutation load"}),
             ("mouse_rbd", "RBD vaccine, divisions, zones and bait",
              {"division_gate:mCherry-low": "divisions since labelling", "zone_gate:DZ": "dark- vs light-zone",
               "rbd_bait:RBD+": "antigen binding (bait)", "isotype": "isotype",
               "mutation_frequency": "somatic mutation load"})]
    have = [(ds, t, pretty) for ds, t, pretty in specs
            if (load_table(ds, "label_effects.csv") is not None)]
    left, gap = 26.0, 14.0
    gap = 20.0
    pw = (w - left - gap * len(have)) / max(len(have) + 1, 1)
    for i, (ds, title, pretty) in enumerate(have):
        label_effects_panel(panel(fig, x + left + i * (pw + gap), y + 6, pw, h - 16), ds, pretty, title=title)

    # the third experiment is reported as a refusal, which is the honest result for 388 cells
    note = panel(fig, x + left + len(have) * (pw + gap), y + 6, pw, h - 16)
    note.set_axis_off()
    note.text(0.5, 0.95, "NP-OVA, sorted zones and plasma cells", transform=note.transAxes, fontsize=5.5,
              color=INK, ha="center", va="top")
    note.text(0.5, 0.55, "no label is tested here.\n\n388 cells in 113 clones: only three\nclones reach a "
                         "reliability of 0.5,\nbelow the minimum of ten that the\ntest requires.\n\n"
                         "Threadfin declines rather than\nreporting an effect measured on\nthree clones.",
              transform=note.transAxes, fontsize=5.0, color=INK2, ha="center", va="center", linespacing=1.5)
    note.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.82, transform=note.transAxes, facecolor="#faf7f0",
                                 edgecolor=YELLOW, lw=0.6, zorder=0))


def malaria_detail_panel(ax):
    """Clone sizes through the Plasmodium time course, where almost nothing is expanded."""
    path = DATA / "malaria" / "timecourse.csv"
    if not path.exists():
        missing(ax, "clone sizes through the infection")
        return
    d = pd.read_csv(path).sort_values("day_number").groupby("day_number", as_index=False).first()
    ax.bar(d["day_number"], d["clones"], width=2.0, color=GREY, label="clones with two or more cells")
    ax.set_xlabel("day after infection")
    ax.set_ylabel("clones")
    ax2 = ax.twiny()
    ax2.set_axis_off()
    ax.plot(d["day_number"], d["cells_in_clones"], "-o", color=BLUE, ms=3,
            label="cells in those clones")
    ax.legend(loc="upper left", fontsize=4.8, handletextpad=0.3, borderaxespad=0.2)
    caption(ax, "Expansion stays small throughout: the largest clone of the first two weeks holds eight cells. "
                "Methods that need expanded clones have nothing to work with here.", mm_below=9)


# --------------------------------------------------------------------------- pages


def supplementary_1():
    fig = new_page("Supplementary Figure 1", "Are the clones defined correctly, and how big are they?",
                   "Everything downstream rests on which cells are grouped into a clone, and on how many cells "
                   "each clone has.", height=244)
    clone_threshold_histograms(fig, 0, 6, 183, 62, "a")
    clone_threshold_agreement(panel(fig, 22, 90, 60, 36, "b"))
    clone_threshold_sensitivity(panel(fig, 112, 90, 40, 36, "c"))
    clone_size_panel(panel(fig, 38, 160, 70, 42, "d"))
    save(fig, "Supplementary_1")


def supplementary_2():
    fig = new_page("Supplementary Figure 2", "Is the comparison fair, and is there enough data?",
                   "The null model and the power of every dataset used in the main figures.", height=230)
    coherence_all_nulls(panel(fig, 24, 6, 70, 60, "a"))
    power_panel(panel(fig, 112, 6, 66, 60, "b"))
    split_test_panel(panel(fig, 24, 88, 80, 56, "c"))
    malaria_detail_panel(panel(fig, 118, 88, 56, 40, "d"))
    save(fig, "Supplementary_2")


def supplementary_3():
    fig = new_page("Supplementary Figure 3", "The model-antigen germinal centre in detail",
                   "Every measured label of every experiment behind Figure 2, and the lineage trees that check "
                   "it against the receptor sequences alone.", height=262)
    gc_detail_panel(fig, 0, 4, 183, 56, "a")
    within_library_panel(panel(fig, 30, 74, 66, 38, "b"))
    sequence_selection_panel(panel(fig, 128, 74, 50, 38, "c"))
    lineage_tree_gallery(fig, 0, 134, 183, 70, "d", "mouse_np", "group",
                         {"mCherry-high": BLUE, "mCherry-low": ORANGE},
                         n_trees=6, title="Lineage trees of germinal-centre clones, NP-OVA division reporter",
                         note="Each tree is one clone, rooted at its unmutated ancestor (square); each dot is a "
                              "cell, coloured by its division gate. Cells of both gates are scattered through "
                              "the same trees, which is the picture behind the null result in Figure 2f.")
    lineage_vs_state_panel(panel(fig, 24, 212, 70, 32, "e"))
    save(fig, "Supplementary_3")


def supplementary_4():
    fig = new_page("Supplementary Figure 4", "What other tools can and cannot do",
                   "Every published method that integrates single-cell BCR with gene expression, asked the same "
                   "question on the same clones.", height=264)
    sk.draw_benchmark_logic(panel(fig, 0, 4, 183, 30, "a"))
    capability_matrix(panel(fig, 24, 44, 100, 48, "b"))
    benchmark_scores(panel(fig, 24, 120, 78, 46, "c"))
    benchmark_calibration(panel(fig, 24, 192, 78, 38, "d"))
    save(fig, "Supplementary_4")


def main():
    supplementary_1()
    supplementary_2()
    supplementary_3()
    supplementary_4()


if __name__ == "__main__":
    main()
