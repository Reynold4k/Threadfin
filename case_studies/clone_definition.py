#!/usr/bin/env python
"""Is the clone definition right? Three checks, written once for all datasets.

Everything in this paper rests on which cells are grouped into a clone, so the
clone call has to be justified rather than assumed.

1. **The distance-to-nearest distribution.** For each receptor, the junction
   distance to the nearest other receptor of the same donor, V gene, J gene and
   junction length. If a clone definition is meaningful this distribution is
   bimodal: a near mode of clonal relatives and a far mode of unrelated
   receptors. The threshold Threadfin picks should fall in the valley.

2. **Agreement with the authors' own clone calls**, where a study published
   them, over a range of thresholds (adjusted Rand index). This asks whether
   the automatic threshold lands where an expert put it by hand.

3. **Sensitivity of the headline result** to the threshold: the share of B-cell
   state explained by clone identity, recomputed at each threshold.

Usage:
    python clone_definition.py                    # the datasets shown in the supplement
    python clone_definition.py mouse_rbd flu      # just these
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# the datasets shown in the supplementary histogram, in the order they appear there
SHOWN = ["mouse_np", "mouse_rbd", "flu_lung", "malaria", "bone_marrow_pc", "ln_vaccine", "flu"]
THRESHOLDS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24]


def nearest_distances(junction: pd.Series, v: pd.Series, j: pd.Series, donor: pd.Series) -> np.ndarray:
    """Junction distance from each receptor to the nearest other receptor it could be clonal with."""
    df = pd.DataFrame({"junction": junction.astype(str),
                       "v": v.astype(str).str.split(",").str[0].str.split("*").str[0],
                       "j": j.astype(str).str.split(",").str[0].str.split("*").str[0],
                       "donor": donor.astype(str)})
    df = df[df["junction"].str.len() > 2]
    out = []
    for _, g in df.groupby(["donor", "v", "j", df["junction"].str.len()], observed=True):
        if len(g) < 2 or len(g) > 20000:                  # the largest groups are quadratic and uninformative
            continue
        seqs = g["junction"].to_numpy()
        arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
        d = (arr[:, None, :] != arr[None, :, :]).mean(axis=2)
        np.fill_diagonal(d, np.inf)
        out.append(d.min(axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def main(datasets: list[str]):
    import threadfin as tf
    from run_case_study import CONFIGS, LOADERS, prepare

    OUT.mkdir(parents=True, exist_ok=True)
    hists, thresholds, sensitivity = [], {}, []

    for name in datasets:
        if name not in CONFIGS:
            print(f"  {name}: not a known case study, skipped", flush=True)
            continue
        try:
            adata, cfg, _, _ = prepare(name, stamp=lambda m: None)
        except Exception as e:                            # a dataset that is not downloaded here
            print(f"  {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue

        cols = {c: c[4:] for c in adata.obs.columns if c.startswith("bcr_")}
        bcr = adata.obs[list(cols)].rename(columns=cols)
        bcr = bcr[bcr["junction"].notna()] if "junction" in bcr.columns else bcr.iloc[:0]
        if bcr.empty or "v_call" not in bcr.columns:
            print(f"  {name}: no junctions in obs", flush=True)
            continue
        bcr["donor"] = adata.obs["donor"].reindex(bcr.index).astype(str)

        d = nearest_distances(bcr["junction"], bcr["v_call"], bcr["j_call"], bcr["donor"])
        if d.size:
            counts, edges = np.histogram(d[np.isfinite(d)], bins=np.linspace(0, 1, 101))
            hists.append(pd.DataFrame({"dataset": name, "bin_left": edges[:-1], "bin_right": edges[1:],
                                       "count": counts}))
        chosen = json.loads((OUT / name / "summary.json").read_text())["clone_definition"]
        thresholds[name] = {"threshold": chosen["threshold"], "method": chosen["threshold_method"]}
        print(f"  {name}: {d.size:,} receptors with a comparable neighbour, "
              f"threshold {chosen['threshold']:.3f} ({chosen['threshold_method']})", flush=True)

        # the authors' own clone call, where the study published one
        truth = None
        try:
            _, raw = LOADERS[name]()
            if "author_clone_id" in raw.columns:
                truth = raw["author_clone_id"].astype(str)
        except Exception:
            pass

        # how the clone call and the headline result move with the threshold
        ctx = cfg.get("context_key", "donor")
        for thr in sorted(set(THRESHOLDS) | {round(float(chosen["threshold"]), 3)}):
            auto = abs(thr - chosen["threshold"]) < 1e-9
            called = tf.define_clones(bcr.drop(columns=["clone_id"], errors="ignore"),
                                      donor_key="donor", threshold=thr, out_col="clone_id", verbose=False)
            ad = adata.copy()
            ad.obs["clone_id"] = called["clone_id"].reindex(ad.obs_names)
            row = {"dataset": name, "threshold": round(float(thr), 3), "auto": bool(auto),
                   "n_clones": int(called["clone_id"].nunique())}
            if truth is not None:
                from sklearn.metrics import adjusted_rand_score

                shared = called.index.intersection(truth.dropna().index)
                if len(shared) > 100:
                    row["ari"] = float(adjusted_rand_score(truth.loc[shared], called["clone_id"].loc[shared]))
            tf.tl.clone_profiles(ad, basis="X_threadfin", context_key=ctx, donor_key="donor", verbose=False)
            coh = tf.tl.clonal_coherence(ad, strata_key=ctx, n_perm=100, verbose=False)
            row.update(icc=coh["icc"], null=coh["null_mean"], n_clones_tested=coh["n_clones"])
            sensitivity.append(row)
            print(f"      {thr:.3f}{' (auto)' if auto else '      '}: {row['n_clones']:,} clones, "
                  f"clone explains {100 * coh['icc']:.1f}%"
                  + (f", agreement with the authors {row['ari']:.2f}" if "ari" in row else ""), flush=True)

        pd.DataFrame(sensitivity).to_csv(OUT / "clone_threshold_sensitivity.csv", index=False)
        if hists:
            pd.concat(hists).to_csv(OUT / "clone_threshold_histograms.csv", index=False)
        (OUT / "clone_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"wrote {OUT}/clone_threshold_*.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or SHOWN)
