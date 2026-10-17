"""When did a clone's output cells leave the germinal centre?

Question: a germinal centre and the plasma cells it exports are different compartments, but the receptor
says which of them belong to the same clone. Somatic hypermutation only ever accumulates, so inside one
clone the mutation load of an output cell cannot exceed the load the germinal centre had **when that cell
left**. The difference between a clone's germinal-centre cells and its own output cells is therefore a
clock, and it runs in only one direction.

    lag(clone) = mean V-region mutation in the clone's output cells
               - mean V-region mutation in the clone's germinal-centre cells

A negative lag means the output left while the clone was less mutated than it is now; zero means the export
is contemporaneous with the germinal centre that is being sampled.

Why this needs the clone as the unit
------------------------------------
At cell level, germinal-centre and plasma-cell populations differ in mutation for many reasons — different
donors, different visits, different sorts, and the simple fact that the two compartments are populated at
different times. None of those apply here, because every comparison is a clone against **itself**: the same
receptor, the same animal, and (with ``within``) the same visit and tissue. The clone is its own control,
which is what makes a one-directional reading of the difference defensible.

Turning the clock into a time
-----------------------------
Given a time column, the germinal centre's own maturation curve converts a mutation gap into days.
:func:`export_lag` fits mutation against time across the source compartment (Theil-Sen, so a few
hypermutated outliers cannot set the slope) and reports each clone's lag in days as well as in mutation
units. The conversion assumes the clone matured at the rate the population did, which is an assumption
about the average and is reported as such.

What it does not establish
--------------------------
No parent-offspring relationship between any two cells; no claim that the output cells seen at a visit
descend from the germinal-centre cells seen at that same visit; and nothing about affinity, which is not
measured by mutation count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import h5ad_safe, log, require_obs

__all__ = ["export_lag"]

SWITCHED = ("IGHG", "IGHA", "IGHE")
UNSWITCHED = ("IGHM", "IGHD")


def _switched(values: pd.Series) -> np.ndarray:
    iso = values.astype(str).str.upper()
    known = iso.str.startswith(SWITCHED + UNSWITCHED)
    return np.where(known, iso.str.startswith(SWITCHED).astype(float), np.nan)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Median pairwise slope; robust to the long right tail of mutation frequencies."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan")
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    keep = dx != 0
    if not keep.any():
        return float("nan")
    return float(np.median((y[j] - y[i])[keep] / dx[keep]))


def export_lag(
    adata,
    *,
    state_key: str,
    source_states,
    output_states,
    clone_key: str = "clone_id",
    mutation_key: str = "mutation_frequency",
    within=(),
    time_key: str | None = None,
    isotype_key: str | None = "isotype",
    switched_only: bool = True,
    min_cells: int = 1,
) -> dict:
    """Date each clone's output against its own germinal centre.

    Parameters
    ----------
    state_key
        ``obs`` column holding the captured cell state.
    source_states, output_states
        Which levels of ``state_key`` are the germinal centre and which are its output, for example
        ``source_states=["GC"]`` and ``output_states=["PB", "LNPC"]``.
    within
        ``obs`` columns that must match between the two cells being compared, on top of the clone itself.
        Pass the visit and the tissue here: a clone's germinal-centre cell from one visit should not be
        compared with its plasma cell from another.
    time_key
        Optional numeric (or ``d28``-style) column. When given, the source compartment's own maturation
        slope converts each clone's lag into days.
    switched_only
        Compare only class-switched cells. Class switching is irreversible, so an IgM cell and an IgG cell
        of one clone are separated by an event that mutation count does not describe.
    min_cells
        Cells of the clone required on each side of a comparison.

    Returns
    -------
    dict with

    ``pairs``
        one row per clone (and stratum): ``source``, ``output``, ``lag``, cell counts, and ``lag_days``
        when ``time_key`` is given.
    ``n_clones``, ``n_output_earlier``, ``n_output_later``
        how many clones were compared and how the lag fell.
    ``median_lag``
        median lag in mutation-frequency units, with a clone bootstrap interval.
    ``maturation_slope``
        mutation per unit time in the source compartment, when ``time_key`` is given.

    Examples
    --------
    >>> res = export_lag(adata, state_key="cell_state", source_states=["GC"],
    ...                  output_states=["PB"], within=("timepoint",), time_key="timepoint")
    >>> res["median_lag"], res["n_output_earlier"], res["n_clones"]
    """
    require_obs(adata, clone_key, state_key, mutation_key, context="export_lag")
    source_states, output_states = list(source_states), list(output_states)
    overlap = set(source_states) & set(output_states)
    if overlap:
        raise ValueError(f"source_states and output_states overlap: {sorted(overlap)}")

    obs = adata.obs
    keep = obs[clone_key].notna() & obs[state_key].isin(source_states + output_states)
    keep &= pd.to_numeric(obs[mutation_key], errors="coerce").notna()
    if switched_only:
        if isotype_key is None or isotype_key not in obs:
            raise ValueError("switched_only=True needs an isotype column; pass isotype_key=None to skip.")
        keep &= _switched(obs[isotype_key]) == 1
    d = obs.loc[keep, [clone_key, state_key, mutation_key, *within]].copy()
    if time_key is not None and time_key not in d.columns:
        d[time_key] = obs.loc[keep, time_key]
    if d.empty:
        raise ValueError("No cells left after filtering; check the state levels and the isotype column.")
    d["_side"] = np.where(d[state_key].isin(source_states), "source", "output")
    d[mutation_key] = pd.to_numeric(d[mutation_key], errors="coerce")

    keys = [clone_key, *within]
    g = d.groupby(keys + ["_side"])[mutation_key].agg(["mean", "size"])
    wide = g.unstack("_side")
    missing = [s for s in ("source", "output") if ("mean", s) not in wide.columns]
    if missing:
        # one side has no usable cell at all — say which, rather than failing on a missing column
        raise ValueError(
            f"No cell with a mutation frequency on the {' and '.join(missing)} side, so no clone can be "
            "captured on both sides. Check the state levels, the isotype filter and the mutation column.")
    wide = wide.dropna(subset=[("mean", "source"), ("mean", "output")])
    wide = wide[(wide[("size", "source")] >= min_cells) & (wide[("size", "output")] >= min_cells)]
    if wide.empty:
        raise ValueError("No clone was captured on both sides; widen `within` or lower `min_cells`.")

    pairs = pd.DataFrame({
        "source": wide[("mean", "source")].to_numpy(),
        "output": wide[("mean", "output")].to_numpy(),
        "n_source": wide[("size", "source")].to_numpy().astype(int),
        "n_output": wide[("size", "output")].to_numpy().astype(int),
    }, index=wide.index).reset_index()
    pairs["lag"] = pairs["output"] - pairs["source"]

    slope = np.nan
    if time_key is not None:
        t_all = pd.to_numeric(
            obs.loc[keep, time_key].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
        src = d["_side"].eq("source").to_numpy()
        slope = _theil_sen(t_all.to_numpy()[src], d[mutation_key].to_numpy()[src])
        if np.isfinite(slope) and slope > 0:
            pairs["lag_days"] = pairs["lag"] / slope

    rng = np.random.default_rng(0)
    lags = pairs["lag"].to_numpy()
    boot = [np.median(lags[rng.integers(0, lags.size, lags.size)]) for _ in range(1000)]
    out = {
        "pairs": pairs,
        "n_clones": int(len(pairs)),
        "n_output_earlier": int((pairs["lag"] < 0).sum()),
        "n_output_later": int((pairs["lag"] > 0).sum()),
        "n_equal": int((pairs["lag"] == 0).sum()),
        "median_lag": float(np.median(lags)),
        "median_lag_ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "maturation_slope": float(slope),
        "median_lag_days": float(np.median(pairs["lag_days"])) if "lag_days" in pairs else float("nan"),
        "settings": {"source_states": source_states, "output_states": output_states,
                     "within": list(within), "switched_only": bool(switched_only),
                     "min_cells": int(min_cells), "time_key": time_key},
    }
    log(f"export_lag: {out['n_clones']} clones captured on both sides; "
        f"{out['n_output_earlier']} with output earlier than their own source, "
        f"{out['n_output_later']} later; median lag {out['median_lag']:+.4f}"
        + (f" ({out['median_lag_days']:+.1f} days)" if np.isfinite(out["median_lag_days"]) else ""))
    adata.uns.setdefault("threadfin", {})["export_lag"] = h5ad_safe({
        k: v for k, v in out.items() if k != "pairs"})
    return out
