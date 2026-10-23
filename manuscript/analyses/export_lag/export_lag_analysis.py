#!/usr/bin/env python
"""How far behind its own germinal centre is a clone's antibody-secreting output?

Somatic hypermutation only accumulates. Inside one clone, an output cell (plasmablast or plasma cell) can
therefore carry at most the mutation load the clone's germinal-centre (GC) cells had when it left, and the
difference between the two compartments of the same clone is a one-directional clock
(``threadfin.tl.export_lag``). Converting the clock into days uses the GC compartment's own rate of
maturation, fitted on per-visit medians of class-switched GC cells (a Theil-Sen fit on every cell pair is
not needed for six to nine time points and does not fit in memory for 40,000 cells).

Datasets (cell-state annotations of the original studies; clones = paired-chain Dandelion clones):

* GSE286215 experiment 2 (*Plasmodium chabaudi*, days 10-42): GC B cells -> plasmablasts, same mouse and day;
* GSE286215 experiment 1 (days 0-14): the same comparison during the first two weeks;
* GSE195673 (human lymph node after SARS-CoV-2 mRNA vaccination): GC B cells -> plasmablasts and lymph-node
  plasma cells, same participant, visit and tissue.

Only class-switched cells are compared (class switching is irreversible and is not described by mutation
count). The analysis is descriptive: it does not establish that the output cells sampled at a visit
descend from the GC cells sampled at that visit, and mutation count is not affinity.

Usage: python export_lag_analysis.py      (writes results/ and figures/ next to this script)
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy import stats  # noqa: E402

import threadfin as tf  # noqa: E402

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1] / "case_studies" / "results"
OUT, FIG = HERE / "results", HERE / "figures"
DESIGNS = {
    "malaria_late": dict(title="Plasmodium infection, days 10-42 (GSE286215 exp. 2)", state="cell_state",
                         source=["GC"], output=["PB"], within=("timepoint",)),
    "malaria": dict(title="Plasmodium infection, days 0-14 (GSE286215 exp. 1)", state="cell_state",
                    source=["GC"], output=["PB"], within=("timepoint",)),
    "ln_vaccine": dict(title="Human lymph node after mRNA vaccination (GSE195673)", state="state",
                       source=["GC"], output=["PB", "LNPC"], within=("timepoint", "tissue")),
}


def day(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def switched(series: pd.Series) -> pd.Series:
    iso = series.astype(str).str.upper()
    return iso.str.startswith(("IGHG", "IGHA", "IGHE"))


def theil_sen(x, y):
    res = stats.theilslopes(y, x)
    return float(res.slope), float(res.low_slope), float(res.high_slope)


def analyse(name, d):
    cells = pd.read_csv(CASE / name / "cells.csv.gz", index_col=0, low_memory=False)
    cells = cells[cells["clone_id"].notna()].copy()
    cells["day"] = day(cells["timepoint"])
    a = ad.AnnData(X=sp.csr_matrix((len(cells), 0), dtype=np.float32), obs=cells)
    res = tf.tl.export_lag(a, state_key=d["state"], source_states=d["source"], output_states=d["output"],
                           within=d["within"], time_key=None)
    pairs = res.pop("pairs")
    # the GC compartment's own clock: per-visit medians of class-switched GC cells
    gc = cells[cells[d["state"]].isin(d["source"]) & switched(cells["isotype"]) & cells["mutation_frequency"].notna()]
    per_day = gc.groupby("day")["mutation_frequency"].agg(median="median", q25=lambda v: v.quantile(.25),
                                                          q75=lambda v: v.quantile(.75), n="size").reset_index()
    out_cells = cells[cells[d["state"]].isin(d["output"]) & cells["mutation_frequency"].notna()]
    out_day = out_cells.groupby("day")["mutation_frequency"].agg(
        median="median", mutated=lambda v: float((v > 0).mean()), n="size").reset_index()
    slope = lo = hi = np.nan
    if len(per_day) >= 3:
        slope, lo, hi = theil_sen(per_day["day"].to_numpy(float), per_day["median"].to_numpy(float))
    if np.isfinite(slope) and slope > 0:
        pairs["lag_days"] = pairs["lag"] / slope
    pairs["day"] = day(pairs["timepoint"])
    summary = {k: v for k, v in res.items() if k != "settings"}
    summary.update(dataset=name, title=d["title"], settings=res["settings"],
                   gc_clock_per_day=float(slope), gc_clock_ci=[float(lo), float(hi)],
                   median_lag_days=float(np.median(pairs["lag_days"])) if "lag_days" in pairs else None,
                   n_days=int(len(per_day)))
    if "lag_days" in pairs:
        per_visit = pairs.groupby("day").agg(n_clones=("lag", "size"), median_lag=("lag", "median"),
                                              output_earlier=("lag", lambda v: int((v < 0).sum())))
        summary["per_visit"] = per_visit.reset_index().to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / f"{name}_clone_pairs.csv", index=False)
    per_day.to_csv(OUT / f"{name}_gc_clock.csv", index=False)
    out_day.to_csv(OUT / f"{name}_output_by_day.csv", index=False)
    return summary, pairs, per_day, out_day


def figure(results):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(results), 3, figsize=(11, 3.3 * len(results)), constrained_layout=True)
    for row, (name, (s, pairs, per_day, out_day)) in zip(axes, results.items()):
        a, b, c = row
        a.fill_between(per_day["day"], per_day["q25"] * 100, per_day["q75"] * 100, color="#9ecae1", alpha=0.6, lw=0)
        a.plot(per_day["day"], per_day["median"] * 100, "o-", color="#2171b5", ms=4, label="GC B cells (switched)")
        a.plot(out_day["day"], out_day["median"] * 100, "s--", color="#cb181d", ms=4, label="output cells")
        a.set_xlabel("day"); a.set_ylabel("V-region mutation (%)")
        a.set_title(f"{s['title']}\nGC clock {s['gc_clock_per_day'] * 100:.3f} %/day", fontsize=8)
        a.legend(fontsize=7, frameon=False)
        lim = float(np.nanmax(pairs[["source", "output"]].to_numpy()) * 100) * 1.05
        b.plot([0, lim], [0, lim], color="0.6", lw=0.8)
        sc = b.scatter(pairs["source"] * 100, pairs["output"] * 100, c=pairs["day"], cmap="viridis", s=9, alpha=0.8, lw=0)
        b.set_xlabel("clone's GC cells (% mutation)"); b.set_ylabel("same clone's output cells (%)")
        b.set_title(f"{s['n_clones']} clones: output lower in {s['n_output_earlier']}, higher in {s['n_output_later']}",
                    fontsize=8)
        fig.colorbar(sc, ax=b, label="day", shrink=0.8)
        if "lag_days" in pairs:
            v = pairs["lag_days"].clip(-60, 60)
            c.hist(v, bins=40, color="#6a51a3", alpha=0.85)
            c.axvline(0, color="k", lw=0.8)
            c.axvline(np.median(pairs["lag_days"]), color="#cb181d", lw=1.2)
            c.set_xlabel("output lag behind own GC (days)")
            c.set_ylabel("clones")
            c.set_title(f"median {s['median_lag_days']:+.1f} days (axis clipped at ±60)", fontsize=8)
        else:
            c.axis("off")
    fig.savefig(FIG / "export_lag.png", dpi=200)
    fig.savefig(FIG / "export_lag.pdf")


def main():
    results, summaries = {}, {}
    for name, d in DESIGNS.items():
        try:
            s, pairs, per_day, out_day = analyse(name, d)
        except ValueError as exc:                       # e.g. no clone captured on both sides
            summaries[name] = {"dataset": name, "error": str(exc)}
            continue
        results[name] = (s, pairs, per_day, out_day)
        summaries[name] = s
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, default=float) + "\n")
    figure(results)
    for k, s in summaries.items():
        print(k, {x: s.get(x) for x in ("n_clones", "n_output_earlier", "n_output_later", "median_lag",
                                        "gc_clock_per_day", "median_lag_days", "error")})


if __name__ == "__main__":
    main()
