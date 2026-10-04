#!/usr/bin/env python
"""How clonal structure develops through a Plasmodium infection.

Skinner, Asad et al. (Nature Immunology 2026) describe clones diversifying
internally as the infection runs: isotype switching in the first week, then
clones splitting between germinal-centre and extrafollicular plasmablast fates,
then somatic hypermutation. Those are statements about the spread of states
*within* a clone, which the variance decomposition measures directly.

For each sampling day this script reports, over clones of at least two cells:

* how much of B-cell state is explained by clone identity, against clones
  shuffled within the same mouse;
* how far apart the cells of one clone are, relative to unrelated cells of the
  same mouse (the within-clone spread that "diversification" should increase);
* the share of clones carrying more than one isotype, and more than one cell
  state - isotype variegation and fate bifurcation, counted directly.

Usage:
    python malaria_timecourse.py            # writes results/malaria/timecourse.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DAYS = ["D0", "D4", "D7", "D10", "D14", "D21", "D28", "D35", "D42"]


def day_number(label: str) -> int:
    return int(str(label).lstrip("D"))


def main(min_cells: int = 2, n_perm: int = 200):
    import threadfin as tf
    from run_case_study import prepare

    rows = []
    for experiment in ("malaria", "malaria_late"):
        adata, *_ = prepare(experiment, stamp=lambda m: None)
        adata.obs["day"] = adata.obs["timepoint"].astype(str)
        for day in [d for d in DAYS if (adata.obs["day"] == d).any()]:
            sub = adata[(adata.obs["day"] == day).to_numpy()].copy()
            sizes = sub.obs["clone_id"].value_counts()
            if (sizes >= min_cells).sum() < 15:
                continue
            tf.tl.clone_profiles(sub, basis="X_threadfin", context_key="donor", donor_key="donor",
                                 representation="mean", min_cells=min_cells, verbose=False)
            coh = tf.tl.clonal_coherence(sub, strata_key="donor", n_perm=n_perm, verbose=False)

            # how far apart are two cells of one clone, against two unrelated cells of the same mouse?
            x = pd.DataFrame(sub.obsm["X_threadfin"], index=sub.obs_names)
            obs = sub.obs.dropna(subset=["clone_id"])
            within, between = [], []
            rng = np.random.default_rng(0)
            for donor, g in obs.groupby("donor", observed=True):
                idx = g.index
                if len(idx) < 10:
                    continue
                v = x.loc[idx].to_numpy()
                clone = g["clone_id"].astype(str).to_numpy()
                same = clone[:, None] == clone[None, :]
                d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
                iu = np.triu_indices(len(idx), 1)
                within.extend(d[iu][same[iu]])
                other = np.flatnonzero(~same[iu])          # positions within the upper triangle
                pick = rng.choice(other, size=min(5000, other.size), replace=False)
                between.extend(d[iu][pick])
            spread = float(np.mean(within) / np.mean(between)) if within and between else np.nan

            # variegation: clones holding more than one isotype, or more than one cell state
            big = obs[obs["clone_id"].map(sizes) >= min_cells]
            iso = big.dropna(subset=["isotype"]).groupby("clone_id")["isotype"].nunique()
            state = big.groupby("clone_id")["cell_state"].nunique()
            rows.append({
                "experiment": experiment, "day": day, "day_number": day_number(day),
                "clones": int((sizes >= min_cells).sum()), "cells_in_clones": int(sizes[sizes >= min_cells].sum()),
                "explained_by_clone": coh["icc"], "shuffled": coh["null_mean"], "p_value": coh["p_value"],
                "within_clone_spread": spread,
                "clones_with_two_isotypes": float((iso > 1).mean()) if len(iso) else np.nan,
                "clones_with_two_states": float((state > 1).mean()) if len(state) else np.nan,
                "median_mutations": float(pd.to_numeric(big.get("mutation_frequency"), errors="coerce").median()
                                          * 300) if "mutation_frequency" in big else np.nan,
            })
            print(f"  {experiment} {day}: {rows[-1]['clones']} clones, clone explains "
                  f"{100 * coh['icc']:.1f}% vs {100 * coh['null_mean']:.1f}%, "
                  f"{100 * (rows[-1]['clones_with_two_states'] or 0):.0f}% of clones span two states", flush=True)

    out = pd.DataFrame(rows).sort_values(["day_number", "experiment"])
    (HERE / "results" / "malaria").mkdir(parents=True, exist_ok=True)
    out.to_csv(HERE / "results" / "malaria" / "timecourse.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
