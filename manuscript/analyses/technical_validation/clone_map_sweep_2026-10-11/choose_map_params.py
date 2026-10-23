#!/usr/bin/env python
"""Pick one clone-map layout for the whole package, from the sweep over every dataset.

The rule is deliberately *global*. Choosing per dataset is what made the maps look unrelated to each
other, and a figure whose panels were each tuned separately is hard to defend; one setting that works
everywhere is both simpler and more honest.

Ranking: a setting is scored by its **mean rank across datasets** on

* ``continuity``    — do neighbours in profile space stay neighbours on the map (the owner's criterion)
* ``largest_piece`` — fraction of clones in the biggest visually connected piece, i.e. not shattered

with ``faithfulness`` and compartment ``agreement`` reported alongside so nothing is traded away
silently. Ranks rather than raw values, because the datasets differ in size by two orders of magnitude
and their absolute scores are not comparable.

Usage:  python choose_map_params.py [--top 12]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "results" / "clone_map_sweep" / "sweep.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    s = pd.read_csv(SWEEP)
    datasets = sorted(s["dataset"].unique())
    print(f"{len(s)} rows over {len(datasets)} datasets: {', '.join(datasets)}\n")

    # rank within each dataset (1 = best), then average across datasets
    for col in ("continuity", "largest_piece", "faithfulness", "agreement"):
        s[f"r_{col}"] = s.groupby("dataset")[col].rank(ascending=False, na_option="keep")
    g = s.groupby(["n_neighbors", "min_dist"])
    out = pd.DataFrame({
        "datasets": g.size(),
        "rank_continuity": g["r_continuity"].mean(),
        "rank_largest_piece": g["r_largest_piece"].mean(),
        "continuity": g["continuity"].mean(),
        "largest_piece": g["largest_piece"].mean(),
        "faithfulness": g["faithfulness"].mean(),
        "agreement": g["agreement"].mean(),
        "worst_largest_piece": g["largest_piece"].min(),
    })
    out["score"] = out[["rank_continuity", "rank_largest_piece"]].mean(axis=1)
    out = out.sort_values("score")

    print("Best layouts, by mean rank across every dataset (lower score = better):\n")
    print(out.head(a.top).round(3).to_string())

    best = out.index[0]
    print(f"\n--> n_neighbors={best[0]}, min_dist={best[1]}")
    print("\nPer dataset at that setting:")
    sel = s[(s["n_neighbors"] == best[0]) & (s["min_dist"] == best[1])]
    print(sel[["dataset", "n_clones", "continuity", "largest_piece", "n_pieces",
               "faithfulness", "agreement", "p_value"]].round(3).to_string(index=False))

    print("\nFor comparison, the package default (15, 0.1):")
    d = s[(s["n_neighbors"] == 15) & (s["min_dist"] == 0.1)]
    print(d[["dataset", "continuity", "largest_piece", "n_pieces", "faithfulness",
             "agreement"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
