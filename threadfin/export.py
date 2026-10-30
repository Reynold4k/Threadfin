"""Mutation gap between a clone's germinal-centre cells and its own output cells.

Question: are the output cells of a clone (plasmablasts, plasma cells) less mutated than the clone's own
germinal-centre (GC) cells, sampled at the same visit and in the same tissue? The receptor says which cells of
the two compartments belong to the same clone, so each clone is compared with itself:

    lag(clone) = mean V-region mutation in the clone's output cells
               - mean V-region mutation in the clone's GC cells

The estimate is the median of this within-clone gap across clones, with a distribution-free 95 % interval and
a sign test. Individual clones fall on both sides of zero: output cells were more mutated than the clone's GC
cells in 8 of 92 mouse clones and in 821 of 2,564 human lymph-node clones of the case studies. A negative
median says that output cells are typically less mutated than GC cells of the same clone; it does not hold
clone by clone.

Why a gap can arise
-------------------
1. The output left the GC earlier, before the GC accumulated further mutations (the reading the name
   ``export_lag`` refers to).
2. Extrafollicular antibody-secreting cells of the same clone, which never went through the GC
   (Jacob & Kelsoe 1992).
3. Affinity-biased export: which GC cells become plasma cells depends on affinity, not on mutation count
   (Phan et al. 2006; Kräutler et al. 2017).
4. Regulated SHM per division: GC cells selected for high affinity divide more often with fewer mutations
   per division (Merkenschlager et al. 2025), so mutation count is not a clock with a constant rate.

The gap does not separate these.

Why the clone is the unit
-------------------------
At cell level, GC and plasma-cell populations differ in mutation for many reasons: different donors, visits
and sorts, and the times at which each compartment was populated. Comparing a clone with itself, and with
``within`` at the same visit and tissue, removes those differences. The interval treats clones as independent,
so it is conditional on the donors sampled and does not describe donor-to-donor variation; ``donor_key``
reports each donor separately.

Turning the gap into days
-------------------------
Given a time column, the GC compartment's maturation rate (mutation per unit time) converts the gap into days.
The rate is the Theil-Sen slope through the per-time-point medians of the GC cells (Theil 1950; Sen 1968):
each time point counts once, because cells of one visit are not independent observations of the trend. Its
interval is Sen's distribution-free one and needs at least five time points. The interval for the median gap
in days combines the 97.5 % intervals of the median gap and of the rate, so it covers at least 95 %. The
conversion assumes that every clone mutates at the population's average rate and that this rate is constant
per day, which regulated SHM per division (point 4) does not support; read the days as a scale.

What it does not establish
--------------------------
No parent-offspring relationship between any two cells; no claim that the output cells seen at a visit
descend from the GC cells seen at that visit; and nothing about affinity, which mutation count does not
measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import h5ad_safe, log, require_obs

__all__ = ["export_lag"]

SWITCHED = ("IGHG", "IGHA", "IGHE")
UNSWITCHED = ("IGHM", "IGHD")
CLOCKS = ("medians", "cells")


def _switched(values: pd.Series) -> np.ndarray:
    iso = values.astype(str).str.upper()
    known = iso.str.startswith(SWITCHED + UNSWITCHED)
    return np.where(known, iso.str.startswith(SWITCHED).astype(float), np.nan)


def _theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 5_000_000, random_state: int = 0) -> float:
    """Median pairwise slope; robust to the long right tail of mutation frequencies.

    All pairs are used up to ``max_pairs``; above that (about 3,200 cells) the median is taken over a
    fixed-seed random sample of ``max_pairs`` pairs, so memory stays bounded (all pairs of 40,000 cells
    would be 8 x 10^8).
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan")
    n_pairs = x.size * (x.size - 1) // 2
    if n_pairs <= max_pairs:
        i, j = np.triu_indices(x.size, 1)
    else:
        rng = np.random.default_rng(random_state)
        i = rng.integers(0, x.size, max_pairs)
        j = rng.integers(0, x.size, max_pairs)
        distinct = i != j
        i, j = i[distinct], j[distinct]
    dx = x[j] - x[i]
    keep = dx != 0
    if not keep.any():
        return float("nan")
    return float(np.median((y[j] - y[i])[keep] / dx[keep]))


def _median_interval(values, level: float = 0.95) -> tuple[float, float]:
    """Distribution-free interval for a median: the order statistics that invert the sign test.

    ``[x_(k), x_(n-k+1)]`` with the largest ``k`` such that ``P(Binomial(n, 1/2) <= k - 1) <= (1 - level) / 2``.
    It covers the median with probability at least ``level`` for any distribution, ties included, and needs
    no bootstrap (a bootstrap of a median is unreliable when values repeat, as mutation frequencies do).
    ``(nan, nan)`` when ``n`` is too small for ``level`` (n < 6 at 95 %).
    """
    from scipy.stats import binom

    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    j = int(binom.ppf(tail, n, 0.5))
    if binom.cdf(j, n, 0.5) > tail:
        j -= 1
    if j < 0:
        return float("nan"), float("nan")
    return float(x[j]), float(x[n - 1 - j])


def _sign_test(values) -> float:
    """Two-sided exact sign test of a zero median (zeros dropped)."""
    from scipy.stats import binomtest

    v = np.asarray(values, dtype=float)
    below, above = int((v < 0).sum()), int((v > 0).sum())
    if below + above == 0:
        return float("nan")
    return float(binomtest(below, below + above, 0.5).pvalue)


def _inversion_cdf(n: int) -> np.ndarray:
    """``P(D <= d)`` for the number of inversions ``D`` of a uniformly random permutation of ``n`` items.

    This is the null distribution of the number of discordant pairs behind Kendall's tau and hence of Sen's
    interval for a Theil-Sen slope.
    """
    p = np.array([1.0])
    for m in range(2, n + 1):
        p = np.convolve(p, np.full(m, 1.0 / m))
    return np.cumsum(p)


def _sen_interval(x, y, level: float = 0.95) -> tuple[float, float, float]:
    """Theil-Sen slope with Sen's distribution-free interval: ``(slope, low, high)``.

    The bounds are order statistics of the pairwise slopes, chosen from the exact null distribution of the
    number of discordant pairs (normal approximation above 50 points). ``low`` and ``high`` are nan when
    there are too few points for ``level`` (fewer than 5 at 95 %).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope = _theil_sen(x, y)
    if not np.isfinite(slope):
        return slope, float("nan"), float("nan")
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    keep = dx != 0
    slopes = np.sort((y[j] - y[i])[keep] / dx[keep])
    n_pairs, n = slopes.size, x.size
    tail = (1.0 - level) / 2.0
    if n <= 50:
        cdf = _inversion_cdf(n)
        allowed = np.flatnonzero(cdf <= tail + 1e-12)
        if allowed.size == 0:
            return slope, float("nan"), float("nan")
        k = int(allowed[-1]) + 1
    else:
        from scipy.stats import norm

        sd = np.sqrt(n * (n - 1) * (2 * n + 5) / 72.0)
        k = int(np.floor(n_pairs / 2.0 - norm.ppf(1 - tail) * sd))
        if k < 1:
            return slope, float("nan"), float("nan")
    k = min(k, n_pairs)
    return slope, float(slopes[k - 1]), float(slopes[n_pairs - k])


def _days_interval(lag_ci, rate_ci) -> tuple[float, float]:
    """Range of gap / rate over the rectangle of the two intervals; nan unless the rate is surely positive."""
    lo_lag, hi_lag = lag_ci
    lo_rate, hi_rate = rate_ci
    if not (np.isfinite([lo_lag, hi_lag, lo_rate, hi_rate]).all() and lo_rate > 0):
        return float("nan"), float("nan")
    corners = [a / b for a in (lo_lag, hi_lag) for b in (lo_rate, hi_rate)]
    return float(min(corners)), float(max(corners))


def _gap_summary(lag: np.ndarray) -> dict:
    lo, hi = _median_interval(lag)
    return {"n_clones": int(lag.size), "n_output_earlier": int((lag < 0).sum()),
            "n_output_later": int((lag > 0).sum()), "n_equal": int((lag == 0).sum()),
            "median_lag": float(np.median(lag)), "ci_low": lo, "ci_high": hi, "sign_test_pvalue": _sign_test(lag)}


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
    donor_key: str | None = None,
    clock: str = "medians",
) -> dict:
    """Within-clone mutation gap between output cells and the clone's own GC cells.

    For every clone captured in both compartments (and matched on ``within``), the lag is the mean
    mutation of its output cells minus that of its GC cells. The result is the median lag across clones
    with a distribution-free 95 % interval (order statistics inverting the sign test; clones as the unit,
    conditional on the donors sampled) and a sign test of a zero median. Several processes produce a
    gap; see the module documentation before reading it as time.

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
        Optional numeric (or ``d28``-style) column. When given, the source compartment's maturation rate
        converts each clone's lag into days.
    switched_only
        Compare only class-switched cells. Class switching is irreversible, so an IgM cell and an IgG cell
        of one clone are separated by an event that mutation count does not describe.
    min_cells
        Cells of the clone required on each side of a comparison.
    donor_key
        Optional ``obs`` column of the donor (animal). Adds ``per_donor``: each donor's median lag with
        its own interval and sign test. With at least six donors the median of the donor medians and its
        interval describe the population of donors.
    clock
        How the maturation rate is fitted when ``time_key`` is given. ``"medians"`` (default): Theil-Sen
        through the per-time-point medians of the source cells, with Sen's interval. ``"cells"``: the
        earlier fit over all source cells, which treats cells as independent observations of the trend
        and has no interval; kept for comparison.

    Returns
    -------
    dict with

    ``pairs``
        one row per clone (and stratum): ``source``, ``output``, ``lag``, cell counts, and ``lag_days``
        when ``time_key`` is given.
    ``n_clones``, ``n_output_earlier``, ``n_output_later``, ``n_equal``
        how many clones were compared and how many have a negative, positive or zero lag (output less,
        more or equally mutated than the clone's GC cells).
    ``median_lag``, ``median_lag_ci``, ``sign_test_pvalue``
        median lag in mutation-frequency units, its 95 % interval, and the two-sided sign test.
    ``maturation_slope``, ``maturation_slope_ci``, ``n_timepoints``, ``clock_points``
        mutation per unit time in the source compartment, its 95 % interval, and the per-time-point
        medians it was fitted on (``clock="medians"``), when ``time_key`` is given.
    ``median_lag_days``, ``median_lag_days_ci``
        the median lag divided by the rate, and an interval covering at least 95 % (nan when the rate's
        interval does not exclude zero or is not estimable).
    ``per_donor``, ``median_lag_across_donors``, ``median_lag_across_donors_ci``
        with ``donor_key``.

    Examples
    --------
    >>> res = export_lag(adata, state_key="cell_state", source_states=["GC"],
    ...                  output_states=["PB"], within=("timepoint",), time_key="timepoint")
    >>> res["median_lag"], res["median_lag_ci"], res["sign_test_pvalue"]
    """
    require_obs(adata, clone_key, state_key, mutation_key, context="export_lag")
    if clock not in CLOCKS:
        raise ValueError(f"clock must be one of {CLOCKS}.")
    if donor_key is not None:
        require_obs(adata, donor_key, context="export_lag")
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
    group_cols = list(within)
    d = obs.loc[keep, [clone_key, state_key, mutation_key, *group_cols]].copy()
    if time_key is not None and time_key not in d.columns:
        d[time_key] = obs.loc[keep, time_key]
    if d.empty:
        raise ValueError("No cells left after filtering; check the state levels and the isotype column.")
    d["_side"] = np.where(d[state_key].isin(source_states), "source", "output")
    d[mutation_key] = pd.to_numeric(d[mutation_key], errors="coerce")

    keys = [clone_key, *group_cols]
    g = d.groupby(keys + ["_side"], observed=True)[mutation_key].agg(["mean", "size"])
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
    if donor_key is not None and donor_key not in pairs.columns:
        # each clone's donor, for per_donor only: a clone without one stays in the overall result
        with_donor = obs.loc[keep & obs[donor_key].notna(), [clone_key, donor_key]]
        donor_of = with_donor.groupby(with_donor[clone_key].astype(str), observed=True)[donor_key].first()
        pairs.insert(1 + len(group_cols), donor_key, pairs[clone_key].astype(str).map(donor_of).to_numpy())
    lags = pairs["lag"].to_numpy()

    # the GC compartment's maturation rate: one point per time point, so cells of a visit count once
    slope, slope_ci, n_times, points = np.nan, (np.nan, np.nan), 0, None
    rate_ci_975 = (np.nan, np.nan)
    if time_key is not None:
        t_all = pd.to_numeric(
            obs.loc[keep, time_key].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
        src = d["_side"].eq("source").to_numpy()
        t_src, m_src = t_all.to_numpy()[src], d[mutation_key].to_numpy()[src]
        if clock == "cells":
            slope = _theil_sen(t_src, m_src)
            n_times = int(pd.Series(t_src).dropna().nunique())
        else:
            ok = np.isfinite(t_src) & np.isfinite(m_src)
            points = (pd.DataFrame({"time": t_src[ok], "mutation": m_src[ok]}).groupby("time")["mutation"]
                      .agg(median="median", n_cells="size").reset_index())
            n_times = len(points)
            slope, lo, hi = _sen_interval(points["time"], points["median"], 0.95)
            slope_ci = (lo, hi)
            rate_ci_975 = _sen_interval(points["time"], points["median"], 0.975)[1:]
        if np.isfinite(slope) and slope > 0:
            pairs["lag_days"] = pairs["lag"] / slope

    overall = _gap_summary(lags)
    days_ci = _days_interval(_median_interval(lags, 0.975), rate_ci_975)
    out = {
        "pairs": pairs,
        "n_clones": overall["n_clones"],
        "n_output_earlier": overall["n_output_earlier"],
        "n_output_later": overall["n_output_later"],
        "n_equal": overall["n_equal"],
        "median_lag": overall["median_lag"],
        "median_lag_ci": (overall["ci_low"], overall["ci_high"]),
        "sign_test_pvalue": overall["sign_test_pvalue"],
        "maturation_slope": float(slope),
        "maturation_slope_ci": (float(slope_ci[0]), float(slope_ci[1])),
        "n_timepoints": int(n_times),
        "median_lag_days": float(np.median(pairs["lag_days"])) if "lag_days" in pairs else float("nan"),
        "median_lag_days_ci": days_ci if "lag_days" in pairs else (float("nan"), float("nan")),
        "settings": {"source_states": source_states, "output_states": output_states,
                     "within": list(within), "switched_only": bool(switched_only),
                     "min_cells": int(min_cells), "time_key": time_key, "donor_key": donor_key,
                     "clock": clock if time_key is not None else None,
                     "interval": "order statistics inverting the sign test (clones as units, conditional on "
                                 "the donors sampled); days: Bonferroni rectangle of the 97.5 % gap and rate "
                                 "intervals"},
    }
    if points is not None:
        out["clock_points"] = points
    if donor_key is not None:
        per = []
        for donor, sub in pairs.groupby(donor_key, observed=True, sort=True):
            per.append({donor_key: donor, **_gap_summary(sub["lag"].to_numpy())})
        per_donor = pd.DataFrame(per)
        if "lag_days" in pairs:
            per_donor["median_lag_days"] = per_donor["median_lag"] / slope
        out["per_donor"] = per_donor
        medians = per_donor["median_lag"].to_numpy()
        out["median_lag_across_donors"] = float(np.median(medians))
        out["median_lag_across_donors_ci"] = _median_interval(medians)
    ci = out["median_lag_ci"]
    log(f"export_lag: {out['n_clones']} clones captured on both sides; output less mutated than the clone's "
        f"own source cells in {out['n_output_earlier']}, more in {out['n_output_later']}, equal in "
        f"{out['n_equal']} (sign test p = {out['sign_test_pvalue']:.2g}); median lag {out['median_lag']:+.4f} "
        f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        + (f" ({out['median_lag_days']:+.1f} days [{days_ci[0]:+.1f}, {days_ci[1]:+.1f}])"
           if np.isfinite(out["median_lag_days"]) else ""))
    adata.uns.setdefault("threadfin", {})["export_lag"] = h5ad_safe({
        k: v for k, v in out.items() if k != "pairs"})
    return out
