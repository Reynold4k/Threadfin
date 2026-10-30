"""Clonal memory across time points or tissues.

Question: when the same clone is sampled twice (two time points, or two
tissues), is it still in the same state?

Why not compare programme labels? Programmes are assigned per clone from
*all* of its cells, so every snapshot of a clone carries the same label by
construction and a label "transition matrix" is trivially diagonal.
:func:`clonal_memory` instead profiles every (clone, time) *snapshot*
separately and asks how similar a clone's consecutive snapshots are, compared
with snapshots of other clones.

The memory index
----------------
For consecutive snapshots ``a`` and ``b`` of the same clone, with ``n_a`` and
``n_b`` cells and context-centred mean profiles ``m_a`` and ``m_b``::

    D_same = ||m_a - m_b||^2 - v_a / n_a - v_b / n_b

where ``v`` is the variance of a snapshot's cells (summed over features), so
``D_same`` estimates how much the clone's *true* state changed (sampling noise
removed). Each snapshot uses its own ``v``. A one-cell snapshot has none; it
uses the mean ``v`` of the multi-cell snapshots of the same level from clones
of its kind (seen at one level, or at more), or, when there are fewer than
five of those, of its kind at every level. This assumes that one-cell
snapshots vary as much as larger ones; with ``min_cells=2`` no assumption is
needed.

``D_other`` is the same quantity when ``b`` is replaced by the later snapshot
of another clone of the same donor that was seen at the same two levels,
averaged over those clones. The memory index is::

    M = 1 - mean(D_same) / mean(D_other)

``M = 1``: clones keep everything that distinguished them from each other;
``M = 0``: a clone is no more similar to its former self than to another
clone. Without sampling noise, ``M`` estimates Lin's concordance correlation
(Lin 1989) between the clone profiles at the two levels, summed over features
and pooled over donors and pairs of levels, so a change that all these clones
share (for example a tissue effect that the context centring did not remove)
also lowers it; ``mean_change_shared`` reports its size.
Because the noise is estimated and subtracted, ``M`` is not bounded: sampling
error can put it below 0 or above 1, mostly when few clones are seen twice.

Why only clones seen twice? Clones found at two levels can share a state that
clones found at one level lack (for example, long-lived clones). Against all
clones, that shared state alone would look like memory. For the same reason a
pair is compared only with pairs between the same two levels. Threadfin 4.5
made the comparison with all clones (``reference="all"``) and used one
variance pooled over all snapshots (``noise="pooled"``), which over-corrects
when the compared snapshots vary less than the rest (pushing ``M`` above 1)
and under-corrects when they vary more.

The P value permutes the later snapshots among the clones of the same donor
seen at the same two levels. The 95% interval bootstraps clones, and
re-estimates the variance of one-cell snapshots in every resample.

As a descriptive companion, each snapshot is also assigned to the nearest
programme centroid and the programme transition table is reported together
with its expectation under random pairing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import codes, get_uns, h5ad_safe, log, perm_pvalue, permute_within, require_obs, require_positive_int
from .profiles import get_profiles, model_from_adata, variance_components

MEMORY_KEY = "memory"
REFERENCES = ("persistent", "all")
NOISE_MODELS = ("snapshot", "pooled")
# multi-cell snapshots a group needs before its mean variance is used for one-cell snapshots
_MIN_NOISE_SNAPSHOTS = 5


def _ordered_levels(values, order, time_key="time_key"):
    values = pd.Series(values)
    levels = list(pd.unique(values.dropna().astype(str)))
    if order is not None:
        order = [str(o) for o in order]
        missing = [lv for lv in levels if lv not in order]
        if missing:
            raise ValueError(f"order is missing levels: {missing}")
        return [o for o in order if o in levels]
    if isinstance(values.dtype, pd.CategoricalDtype) and values.cat.ordered:
        return [str(c) for c in values.cat.categories if str(c) in levels]
    num = pd.to_numeric(pd.Series(levels).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    if num.notna().all():
        return [levels[i] for i in np.argsort(num.to_numpy(), kind="stable")]
    warnings.warn(f"The levels of '{time_key}' ({', '.join(sorted(levels))}) have no order, so they are paired in "
                  "alphabetical order. If they have one (time points, or a path between tissues), pass order=[...] "
                  "or make the column an ordered pandas Categorical.", UserWarning, stacklevel=4)
    return sorted(levels)


def _snapshots(adata, model, clone_key, time_key, order, min_cells):
    """Unshrunk mean profile, size and within sum of squares of every (clone, time) snapshot."""
    obs = adata.obs
    clone_str = obs[clone_key].astype("object").map(lambda c: str(c) if pd.notna(c) else None)
    levels = _ordered_levels(obs.loc[clone_str.notna(), time_key], order, time_key)
    t_index = obs[time_key].astype(str).map({lv: i for i, lv in enumerate(levels)})
    valid = (clone_str.notna() & t_index.notna()).to_numpy()
    key = clone_str[valid] + "\x1f" + t_index[valid].astype(int).astype(str)
    snap_valid, snap_index = codes(key)
    snap = -np.ones(adata.n_obs, dtype=np.int64)
    snap[np.flatnonzero(valid)] = snap_valid
    n_snap = len(snap_index)

    ok = np.flatnonzero(snap >= 0)
    member = sp.csr_matrix((np.ones(ok.size), (snap[ok], ok)), shape=(n_snap, snap.size))
    n = np.asarray(member.sum(axis=1)).ravel()
    means = np.asarray(member @ model.residuals) / np.maximum(n, 1)[:, None]
    # sum of squared deviations of each snapshot's cells from its mean, over all features (0 for one cell)
    sq = np.asarray(member @ np.einsum("ij,ij->i", model.residuals, model.residuals)).ravel()
    within_ss = np.maximum(sq - n * np.einsum("ij,ij->i", means, means), 0.0)
    # within-snapshot noise variance pooled over all snapshots (pure sampling noise, no temporal change)
    sigma2 = variance_components(model.residuals, snap).sigma2
    info = pd.DataFrame({
        "clone": [s.split("\x1f")[0] for s in snap_index],
        "t": [int(s.split("\x1f")[1]) for s in snap_index],
        "n_cells": n.astype(int),
        "within_ss": within_ss,
    })
    keep = (info["n_cells"] >= min_cells).to_numpy()
    return info[keep].reset_index(drop=True), means[keep], sigma2, levels


def _snapshot_noise(snaps, seen_twice):
    """Sampling-noise term ``v / n`` of every snapshot, each with its own variance ``v`` (``noise="snapshot"``).

    A snapshot of >= 2 cells uses the unbiased variance of its own cells, summed over features. A one-cell
    snapshot has none and uses the mean of that variance over the multi-cell snapshots of its group: same level
    and same kind of clone (seen at one level, or at more); with fewer than ``_MIN_NOISE_SNAPSHOTS`` of those,
    the same kind at every level; else all snapshots. Returns ``terms(weights=None)``, which recomputes the group
    means with the snapshots weighted (bootstrap), and where each snapshot's variance came from ("own", "level",
    "kind" or "all").
    """
    n = snaps["n_cells"].to_numpy(dtype=float)
    multi = n >= 2
    own = np.zeros(n.size)
    own[multi] = snaps["within_ss"].to_numpy(dtype=float)[multi] / (n[multi] - 1)
    t = snaps["t"].to_numpy(dtype=np.int64)
    kind = np.asarray(seen_twice, dtype=np.int64)
    n_levels = int(t.max()) + 1 if t.size else 0
    # group ids, from the most specific; the id spaces do not overlap
    by_level, by_kind = 2 * t + kind, 2 * n_levels + kind
    everything = np.full(n.size, 2 * n_levels + 2)
    n_groups = 2 * n_levels + 3
    groupings = (by_level, by_kind, everything)

    def group_means(weights):
        num = sum(np.bincount(g[multi], weights=weights[multi] * own[multi], minlength=n_groups) for g in groupings)
        den = sum(np.bincount(g[multi], weights=weights[multi], minlength=n_groups) for g in groupings)
        return np.divide(num, den, out=np.full(n_groups, np.nan), where=den > 0)

    count = sum(np.bincount(g[multi], minlength=n_groups) for g in groupings)
    group = np.where(count[by_level] >= _MIN_NOISE_SNAPSHOTS, by_level,
                     np.where(count[by_kind] >= _MIN_NOISE_SNAPSHOTS, by_kind, everything))
    single = np.flatnonzero(~multi)
    source = np.select([multi, group == by_level, group == by_kind], ["own", "level", "kind"], default="all")

    full = group_means(np.ones(n.size))

    def terms(weights=None):
        g = full if weights is None else group_means(np.asarray(weights, dtype=float))
        g = np.where(np.isfinite(g), g, full)        # a group left empty by a resample keeps its full-data mean
        out = own / np.maximum(n, 1)
        out[single] = g[group[single]]               # one cell: v / 1
        return out

    return terms, source


def _mean_sqdist_to_others(a_means, b_means, stratum, k):
    """For every pair, the mean squared distance from its earlier profile to the later profiles of the *other*
    pairs of its stratum, computed exactly from stratum sums."""
    n_strata = k.size
    member = sp.csr_matrix((np.ones(stratum.size), (stratum, np.arange(stratum.size))),
                           shape=(n_strata, stratum.size))
    centre = np.asarray(member @ (a_means + b_means)) / (2.0 * k[:, None])   # distances do not depend on it
    a = a_means - centre[stratum]
    b = b_means - centre[stratum]
    b2 = np.einsum("ij,ij->i", b, b)
    sum_b = np.asarray(member @ b)
    sum_b2 = np.asarray(member @ b2).ravel()
    kk = k[stratum].astype(float)
    total = ((kk - 1.0) * np.einsum("ij,ij->i", a, a) + (sum_b2[stratum] - b2)
             - 2.0 * np.einsum("ij,ij->i", a, sum_b[stratum] - b))
    return total / (kk - 1.0)


def _shared_change(a_means, b_means, noise_terms, stratum, k):
    """Squared change of the mean profile of each stratum's clones between the two levels, noise removed, averaged
    over the pairs of strata with at least two pairs: the part of the change that the compared clones share."""
    member = sp.csr_matrix((np.ones(stratum.size), (stratum, np.arange(stratum.size))), shape=(k.size, stratum.size))
    shift = np.asarray(member @ (a_means - b_means)) / k[:, None]
    per_stratum = np.einsum("ij,ij->i", shift, shift) - np.asarray(member @ noise_terms).ravel() / k ** 2
    ok = k >= 2
    return float(np.sum(k[ok] * per_stratum[ok]) / np.sum(k[ok])) if ok.any() else float("nan")


def clonal_memory(
    adata,
    time_key: str,
    *,
    order=None,
    reference: str = "persistent",
    noise: str = "snapshot",
    programme_key: str = "clone_programme",
    min_cells: int = 1,
    min_pairs: int = 10,
    n_null: int = 500,
    n_boot: int = 500,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """Do clones keep their state between time points (or tissues)?

    Parameters
    ----------
    time_key
        ``obs`` column with the time point (or tissue) of each cell.
    order
        Order of the levels of ``time_key``. When omitted: the order of an
        ordered Categorical column, else numbers in the labels (``d7`` <
        ``d28``), else alphabetical, with a warning. Each clone's
        consecutive observed levels are paired.
    reference
        The clones that a clone's change is compared with. ``"persistent"``
        (default): other clones of the same donor seen at the same two
        levels. ``"all"``: any other clone of the same donor at the later
        level, including clones seen only there (Threadfin 4.5); a state that
        the clones seen twice share then counts as memory.
    noise
        The variance used to remove sampling noise. ``"snapshot"`` (default):
        each snapshot's own; a one-cell snapshot uses the mean of the
        multi-cell snapshots of its level from clones of its kind (module
        docstring). ``"pooled"``: one variance pooled over all snapshots
        (Threadfin 4.5). With ``min_cells=2`` every snapshot uses its own.
    programme_key
        Programme column of the clone table (for the descriptive transition
        table); skipped when absent.
    min_cells
        Minimum cells per snapshot.
    min_pairs
        Fewest snapshot pairs for which an index is reported. Below this the
        bootstrap interval collapses and the estimate says nothing, so the
        function raises instead of returning a number.
    n_null, n_boot
        Permutations (random re-pairings) for the p-value; bootstrap
        replicates for the CI.

    Returns
    -------
    dict with ``memory_index`` (module docstring; not bounded by 0 and 1),
    its bootstrap ``memory_index_ci``, ``p_value``, ``n_clones``,
    ``n_pairs``, the ``reference`` and ``noise`` used, ``mean_change`` and
    ``mean_change_random`` (mean ``D_same`` and ``D_other``),
    ``mean_change_shared`` (the part of ``mean_change`` that all compared
    clones of a donor share: the squared change of their mean profile;
    with the default reference it is also part of ``mean_change_random``),
    ``mean_noise_removed`` (mean noise term subtracted from ``D_same``), the
    per-pair table ``pairs`` and, when programmes exist, programme
    ``persistence`` with ``persistence_null`` and the ``transitions`` /
    ``transitions_expected`` tables. When the interval cannot be computed it
    is ``(nan, nan)`` and the index should not be interpreted. Stored in
    ``adata.uns['threadfin']['memory'][time_key]``.
    """
    for name, value in [("min_cells", min_cells), ("min_pairs", min_pairs), ("n_null", n_null), ("n_boot", n_boot)]:
        require_positive_int(value, name)
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, not {reference!r}.")
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise must be one of {NOISE_MODELS}, not {noise!r}.")
    prof = get_profiles(adata)
    clone_key = prof["params"]["clone_key"]
    require_obs(adata, time_key)
    rng = np.random.default_rng(random_state)
    model = model_from_adata(adata, unsmoothed=True)  # smoothing would leak between snapshots
    snaps, means, sigma2, levels = _snapshots(adata, model, clone_key, time_key, order, min_cells)
    table = prof["clone_table"]
    donor = table["donor"].astype(str) if "donor" in table.columns else pd.Series(dtype=str)
    snaps["donor"] = snaps["clone"].map(donor).fillna("all").to_numpy()

    # consecutive snapshot pairs of the same clone
    pa, pb = [], []
    for _, grp in snaps.groupby("clone", sort=False):
        if len(grp) >= 2:
            rows = grp.sort_values("t").index.to_numpy()
            pa.extend(rows[:-1])
            pb.extend(rows[1:])
    pa, pb = np.asarray(pa, dtype=np.int64), np.asarray(pb, dtype=np.int64)
    if pa.size == 0:
        raise ValueError(f"No clone has >= {min_cells} cells at two levels of '{time_key}'.")
    if pa.size < min_pairs:
        raise ValueError(f"Only {pa.size} clones have >= {min_cells} cells at two levels of '{time_key}' "
                         f"(need >= {min_pairs}); no memory index is reported.")

    n_cells = snaps["n_cells"].to_numpy(dtype=float)
    t_arr, d_arr, c_arr = snaps["t"].to_numpy(), snaps["donor"].to_numpy(), snaps["clone"].to_numpy()
    seen_twice = np.zeros(len(snaps), dtype=bool)
    seen_twice[pa] = seen_twice[pb] = True

    # sampling-noise term v / n of every snapshot
    if noise == "pooled":
        pooled = float(sigma2.sum())
        noise_terms, variance_source = (lambda weights=None: pooled / n_cells), None
    else:
        noise_terms, variance_source = _snapshot_noise(snaps, seen_twice)
    nt = noise_terms()

    def sqdist(a, b):
        diff = means[a] - means[b]
        return np.einsum("ij,ij->i", diff, diff)

    def corrected_sqdist(a, b):
        if noise == "pooled":                         # the 4.5 arithmetic, kept exactly
            return sqdist(a, b) - pooled * (1.0 / n_cells[a] + 1.0 / n_cells[b])
        return sqdist(a, b) - nt[a] - nt[b]

    # strata: pairs of the same donor between the same two levels
    pair_key = (pd.Series(d_arr[pa]).astype(str) + "\x1f" + pd.Series(t_arr[pa]).astype(str) + "\x1f"
                + pd.Series(t_arr[pb]).astype(str))
    if reference == "all":
        # candidate substitutes for every pair: other clones, same donor and time point
        pools: dict = {}
        for i in range(len(snaps)):
            pools.setdefault((d_arr[i], t_arr[i]), []).append(i)
        cands = []
        for a, b in zip(pa, pb):
            pool = np.asarray(pools[(d_arr[b], t_arr[b])])
            pool = pool[c_arr[pool] != c_arr[a]]
            cands.append(pool)
        usable = np.array([c.size > 0 for c in cands])
        cands = [c for c, u in zip(cands, usable) if u]
        unusable_reason = "another clone from the same donor and target time"
    else:
        # each pair needs another pair of its stratum to compare with
        all_strata = codes(pair_key)[0]
        usable = np.bincount(all_strata)[all_strata] >= 2
        unusable_reason = "another clone of the same donor seen at the same two levels"
    n_unmatched = int((~usable).sum())
    pa, pb = pa[usable], pb[usable]
    if pa.size < min_pairs:
        raise ValueError(f"Only {pa.size} snapshot pairs have {unusable_reason} "
                         f"(need >= {min_pairs}); no memory index is reported.")
    stratum = codes(pd.Series(pair_key[usable].to_numpy()))[0]
    k = np.bincount(stratum).astype(float)
    kk = k[stratum]

    raw_same = sqdist(pa, pb)
    d_same = corrected_sqdist(pa, pb)

    if reference == "all":
        def draw():
            return np.array([c[rng.integers(c.size)] for c in cands])

        null_stats = np.empty(n_null)
        d_random_all, kept = [], []
        for r in range(n_null):
            drawn = draw()
            dr = corrected_sqdist(pa, drawn)
            null_stats[r] = dr.mean()
            if r < 50:
                d_random_all.append(dr)
                kept.append(drawn)
        d_random = np.mean(d_random_all, axis=0)  # per-pair expected distance to a random clone
        kept = np.vstack(kept)
        raw_random = np.mean([sqdist(pa, drawn) for drawn in kept], axis=0)

        def random_noise(terms):
            return terms[kept].mean(axis=0)
    else:
        # per-pair expected distance to another clone of the stratum: an exact average, no sampling
        raw_random = _mean_sqdist_to_others(means[pa], means[pb], stratum, k)

        def random_noise(terms):
            return (np.bincount(stratum, weights=terms[pb], minlength=k.size)[stratum] - terms[pb]) / (kk - 1.0)

        d_random = raw_random - nt[pa] - random_noise(nt)
        # P value: later snapshots permuted among the pairs of each stratum. The noise terms and every squared
        # norm are the same in each permutation, so mean(D) only moves with sum_i m_a(i) . m_b(perm(i)).
        a_mean, b_mean = means[pa], means[pb]

        def cross(perm):
            return np.einsum("ij,ij->", a_mean, b_mean[perm])

        observed = cross(np.arange(pa.size))
        null_cross = np.array([cross(permute_within(np.arange(pa.size), stratum, rng)) for _ in range(n_null)])

    if not np.isfinite(d_random).all() or d_random.mean() <= 0:
        raise ValueError("Noise-corrected matched-clone distance is non-positive; "
                         "the memory ratio is not identifiable for these samples.")
    mem = 1.0 - d_same.mean() / d_random.mean()
    if reference == "all":
        p_value = (1 + int((null_stats <= d_same.mean()).sum())) / (n_null + 1)
    else:
        p_value = perm_pvalue(observed, null_cross, "greater")   # closer than permuted pairs = larger cross

    # bootstrap over clones; with noise="snapshot" the variance of one-cell snapshots is re-estimated each time
    clones_of_pairs = c_arr[pa]
    uniq, inv = np.unique(clones_of_pairs, return_inverse=True)
    snap_clone = pd.Index(uniq).get_indexer(c_arr)   # -1: a clone that is not resampled keeps weight 1
    boots = np.empty(n_boot)
    for r in range(n_boot):
        counts = np.bincount(rng.integers(0, uniq.size, size=uniq.size), minlength=uniq.size)
        w = counts[inv]
        ds, dr = d_same, d_random
        if noise == "snapshot":
            terms = noise_terms(np.where(snap_clone >= 0, counts[np.maximum(snap_clone, 0)], 1.0))
            ds = raw_same - terms[pa] - terms[pb]
            dr = raw_random - terms[pa] - random_noise(terms)
        denom = np.sum(w * dr)
        boots[r] = 1.0 - np.sum(w * ds) / denom if denom > 0 else np.nan
    identifiable = np.isfinite(boots)
    # Dropping undefined ratios would produce a falsely reassuring conditional CI.
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if identifiable.all() else (np.nan, np.nan)
    if not identifiable.all():
        warnings.warn("The confidence interval of the memory index could not be computed: in some bootstrap "
                      "resamples of clones no between-clone difference was left to compare with (too few clones "
                      "seen at two levels). The interval is reported as missing and the index should not be "
                      "interpreted; the P value (a permutation test) is still valid.", stacklevel=2)

    noise_removed = raw_same - d_same
    pairs = pd.DataFrame({
        "clone": clones_of_pairs, "earlier": [levels[t] for t in t_arr[pa]],
        "later": [levels[t] for t in t_arr[pb]], "n_cells_earlier": n_cells[pa].astype(int),
        "n_cells_later": n_cells[pb].astype(int), "change": d_same, "change_vs_random": d_random,
        "noise_removed": noise_removed,
    })
    res = {
        "time_key": time_key, "levels": levels, "reference": reference, "noise": noise,
        "n_clones": int(uniq.size), "n_pairs": int(pa.size), "n_unmatched_pairs": n_unmatched,
        "null_matching": ("other clones of the same donor seen at the same two levels; later snapshots "
                          "permuted among them" if reference == "persistent"
                          else "same donor and target time; no cross-donor fallback"),
        "memory_index": float(mem), "memory_index_ci": ci, "p_value": float(p_value),
        "bootstrap_identifiable_fraction": float(identifiable.mean()),
        "mean_change": float(d_same.mean()), "mean_change_random": float(d_random.mean()),
        "mean_change_shared": _shared_change(means[pa], means[pb], noise_removed, stratum, k),
        "mean_noise_removed": float(noise_removed.mean()),
        "pairs": pairs,
    }
    if variance_source is not None:
        # where the variance of each compared snapshot came from: its own cells, or the mean of a group
        compared = np.unique(np.concatenate([pa, pb]))
        res["variance_source"] = {s: int(np.sum(variance_source[compared] == s)) for s in ("own", "level", "kind", "all")}

    # descriptive: programme of each snapshot by nearest programme centroid (needs >= 2 programmes)
    if programme_key in table.columns and table[programme_key].dropna().nunique() >= 2:
        lab = table[programme_key].dropna().astype(str)
        centroids, names = [], sorted(lab.unique(), key=lambda s: (len(s), s))
        member_snaps = snaps["clone"].map(lab)
        for name in names:
            rows = np.flatnonzero((member_snaps == name).to_numpy())
            centroids.append(means[rows].mean(axis=0) if rows.size else np.full(means.shape[1], np.nan))
        centroids = np.vstack(centroids)
        d2 = ((means[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        nearest = np.nanargmin(np.where(np.isfinite(d2), d2, np.inf), axis=1)
        same = nearest[pa] == nearest[pb]
        trans = pd.crosstab(pd.Categorical(np.asarray(names)[nearest[pa]], categories=names),
                            pd.Categorical(np.asarray(names)[nearest[pb]], categories=names), dropna=False)
        trans.index.name, trans.columns.name = "earlier", "later"
        if reference == "all":
            null_same = np.mean([np.mean(nearest[pa] == nearest[draw()]) for _ in range(100)])
            exp = np.zeros_like(trans.to_numpy(), dtype=float)
            for _ in range(100):
                exp += pd.crosstab(pd.Categorical(np.asarray(names)[nearest[pa]], categories=names),
                                   pd.Categorical(np.asarray(names)[nearest[draw()]], categories=names),
                                   dropna=False).to_numpy()
            exp /= 100
        else:
            # exact expectation: the later programme of another pair of the same stratum
            onehot = np.eye(len(names))[nearest[pb]]
            per_stratum = np.zeros((k.size, len(names)))
            np.add.at(per_stratum, stratum, onehot)
            others = (per_stratum[stratum] - onehot) / (kk - 1.0)[:, None]
            null_same = float(np.mean(others[np.arange(pa.size), nearest[pa]]))
            exp = np.zeros((len(names), len(names)))
            np.add.at(exp, nearest[pa], others)
        res.update({
            "persistence": float(same.mean()), "persistence_null": float(null_same),
            "transitions": trans,
            "transitions_expected": pd.DataFrame(exp, index=trans.index, columns=trans.columns),
        })
        pairs["programme_earlier"] = np.asarray(names)[nearest[pa]]
        pairs["programme_later"] = np.asarray(names)[nearest[pb]]

    get_uns(adata).setdefault(MEMORY_KEY, {})[time_key] = h5ad_safe(res)
    extra = (f"; same programme {res['persistence']:.2f} vs {res['persistence_null']:.2f} for other clones"
             if "persistence" in res else "")
    interval = (f"[{ci[0]:.2f}, {ci[1]:.2f}]" if identifiable.all()
                else "(no interval: too few clones, so the index is not interpretable)")
    compared = ("other clones seen at the same two levels" if reference == "persistent"
                else "all clones at the later level")
    log(f"clonal memory over '{time_key}': {uniq.size} clones, {pa.size} snapshot pairs, compared with "
        f"{compared}; memory index {mem:.2f} {interval}, p = {p_value:.3g}{extra}.", verbose)
    return res
