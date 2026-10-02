"""Clonal memory across time points or tissues.

Question: when the same clone is sampled twice (two time points, or two
tissues), is it still in the same state?

Why not compare programme labels? Programmes are assigned per clone from
*all* of its cells, so every snapshot of a clone carries the same label by
construction and a label "transition matrix" is trivially diagonal.
:func:`clonal_memory` instead profiles every (clone, time) *snapshot*
separately and asks how similar a clone's consecutive snapshots are, compared
with snapshots of unrelated clones.

The memory index
----------------
For consecutive snapshots ``a`` and ``b`` of the same clone, with ``n_a`` and
``n_b`` cells and context-centred mean profiles ``m_a`` and ``m_b``::

    D_same = ||m_a - m_b||^2 - sum_j sigma2_j * (1/n_a + 1/n_b)

where ``sigma2`` is the within-snapshot variance, so ``D_same`` estimates how
much the clone's *true* state changed (sampling noise removed). ``D_random``
is the same quantity when ``b`` is replaced by a snapshot of a different clone
from the same donor and time point. The memory index is::

    M = 1 - mean(D_same) / mean(D_random)

``M = 1``: clones keep everything that distinguished them; ``M = 0``: a clone
is no more similar to its former self than to a random clone. ``M`` behaves
like a test-retest reliability of clone state. The p-value compares
``mean(D_same)`` with random re-pairings; the 95% interval bootstraps clones.

As a descriptive companion, each snapshot is also assigned to the nearest
programme centroid and the programme transition table is reported together
with its expectation under random pairing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import codes, get_uns, log, require_obs
from .profiles import get_profiles, model_from_adata, variance_components

MEMORY_KEY = "memory"


def _ordered_levels(values, order):
    levels = list(pd.unique(pd.Series(values).dropna().astype(str)))
    if order is not None:
        order = [str(o) for o in order]
        missing = [lv for lv in levels if lv not in order]
        if missing:
            raise ValueError(f"order is missing levels: {missing}")
        return [o for o in order if o in levels]
    num = pd.to_numeric(pd.Series(levels).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    if num.notna().all():
        return [levels[i] for i in np.argsort(num.to_numpy(), kind="stable")]
    return sorted(levels)


def _snapshots(adata, model, clone_key, time_key, order, min_cells):
    """Unshrunk mean profile, size and donor of every (clone, time) snapshot."""
    obs = adata.obs
    clone_str = obs[clone_key].astype("object").map(lambda c: str(c) if pd.notna(c) else None)
    levels = _ordered_levels(obs.loc[clone_str.notna(), time_key], order)
    t_index = obs[time_key].astype(str).map({lv: i for i, lv in enumerate(levels)})
    valid = (clone_str.notna() & t_index.notna()).to_numpy()
    key = clone_str[valid] + "\x1f" + t_index[valid].astype(int).astype(str)
    snap_valid, snap_index = codes(key)
    snap = -np.ones(adata.n_obs, dtype=np.int64)
    snap[np.flatnonzero(valid)] = snap_valid
    n_snap = len(snap_index)

    import scipy.sparse as sp

    ok = np.flatnonzero(snap >= 0)
    member = sp.csr_matrix((np.ones(ok.size), (snap[ok], ok)), shape=(n_snap, snap.size))
    n = np.asarray(member.sum(axis=1)).ravel()
    means = np.asarray(member @ model.residuals) / np.maximum(n, 1)[:, None]
    # within-snapshot noise variance (pure sampling noise, no temporal change)
    sigma2 = variance_components(model.residuals, snap).sigma2
    info = pd.DataFrame({
        "clone": [s.split("\x1f")[0] for s in snap_index],
        "t": [int(s.split("\x1f")[1]) for s in snap_index],
        "n_cells": n.astype(int),
    })
    keep = (info["n_cells"] >= min_cells).to_numpy()
    return info[keep].reset_index(drop=True), means[keep], sigma2, levels


def clonal_memory(
    adata,
    time_key: str,
    *,
    order=None,
    programme_key: str = "clone_programme",
    min_cells: int = 3,
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
        Order of the levels of ``time_key``; inferred from numbers in the
        labels (``d7`` < ``d28``) when omitted. Each clone's consecutive
        observed levels are paired.
    programme_key
        Programme column of the clone table (for the descriptive transition
        table); skipped when absent.
    min_cells
        Minimum cells per snapshot.
    n_null, n_boot
        Random re-pairings for the p-value; bootstrap replicates for the CI.

    Returns
    -------
    dict with ``memory_index`` (see module docstring), its bootstrap
    ``memory_index_ci``, ``p_value``, ``n_clones``, ``n_pairs``, the
    per-pair table ``pairs`` and, when programmes exist, programme
    ``persistence`` with ``persistence_null`` and the ``transitions`` /
    ``transitions_expected`` tables. Stored in
    ``adata.uns['threadfin']['memory'][time_key]``.
    """
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
    if pa.size < 10:
        warnings.warn(f"Only {pa.size} snapshot pairs across '{time_key}'; the estimate is noisy.", stacklevel=2)

    n_cells = snaps["n_cells"].to_numpy(dtype=float)
    noise = sigma2.sum()

    def corrected_sqdist(a, b):
        diff = means[a] - means[b]
        return np.einsum("ij,ij->i", diff, diff) - noise * (1.0 / n_cells[a] + 1.0 / n_cells[b])

    # candidate substitutes for every pair: other clones, same donor and time point
    t_arr, d_arr, c_arr = snaps["t"].to_numpy(), snaps["donor"].to_numpy(), snaps["clone"].to_numpy()
    pools: dict = {}
    for i in range(len(snaps)):
        pools.setdefault((d_arr[i], t_arr[i]), []).append(i)
    by_time: dict = {}
    for i in range(len(snaps)):
        by_time.setdefault(t_arr[i], []).append(i)
    cands = []
    for a, b in zip(pa, pb):
        pool = np.asarray(pools[(d_arr[b], t_arr[b])])
        pool = pool[c_arr[pool] != c_arr[a]]
        if pool.size == 0:
            pool = np.asarray(by_time[t_arr[b]])
            pool = pool[c_arr[pool] != c_arr[a]]
        cands.append(pool)
    usable = np.array([c.size > 0 for c in cands])
    pa, pb = pa[usable], pb[usable]
    cands = [c for c, u in zip(cands, usable) if u]

    d_same = corrected_sqdist(pa, pb)

    def draw():
        return np.array([c[rng.integers(c.size)] for c in cands])

    null_stats = np.empty(n_null)
    d_random_all = []
    for r in range(n_null):
        dr = corrected_sqdist(pa, draw())
        null_stats[r] = dr.mean()
        if r < 50:
            d_random_all.append(dr)
    d_random = np.mean(d_random_all, axis=0)  # per-pair expected distance to a random clone
    mem = 1.0 - d_same.mean() / d_random.mean()
    p_value = (1 + int((null_stats <= d_same.mean()).sum())) / (n_null + 1)

    # bootstrap over clones
    clones_of_pairs = c_arr[pa]
    uniq, inv = np.unique(clones_of_pairs, return_inverse=True)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, size=uniq.size), minlength=uniq.size)[inv]
        boots[r] = 1.0 - np.sum(w * d_same) / np.sum(w * d_random)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

    pairs = pd.DataFrame({
        "clone": clones_of_pairs, "earlier": [levels[t] for t in t_arr[pa]],
        "later": [levels[t] for t in t_arr[pb]], "n_cells_earlier": n_cells[pa].astype(int),
        "n_cells_later": n_cells[pb].astype(int), "change": d_same, "change_vs_random": d_random,
    })
    res = {
        "time_key": time_key, "levels": levels, "n_clones": int(uniq.size), "n_pairs": int(pa.size),
        "memory_index": float(mem), "memory_index_ci": ci, "p_value": float(p_value),
        "mean_change": float(d_same.mean()), "mean_change_random": float(d_random.mean()),
        "pairs": pairs,
    }

    # descriptive: programme of each snapshot by nearest programme centroid
    if programme_key in table.columns and table[programme_key].notna().any():
        lab = table[programme_key].dropna().astype(str)
        clone_means = pd.DataFrame(index=snaps.index)
        centroids, names = [], sorted(lab.unique(), key=lambda s: (len(s), s))
        member_snaps = snaps["clone"].map(lab)
        for name in names:
            rows = np.flatnonzero((member_snaps == name).to_numpy())
            centroids.append(means[rows].mean(axis=0) if rows.size else np.full(means.shape[1], np.nan))
        centroids = np.vstack(centroids)
        d2 = ((means[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        nearest = np.nanargmin(np.where(np.isfinite(d2), d2, np.inf), axis=1)
        same = nearest[pa] == nearest[pb]
        null_same = np.mean([np.mean(nearest[pa] == nearest[draw()]) for _ in range(100)])
        trans = pd.crosstab(pd.Categorical(np.asarray(names)[nearest[pa]], categories=names),
                            pd.Categorical(np.asarray(names)[nearest[pb]], categories=names), dropna=False)
        trans.index.name, trans.columns.name = "earlier", "later"
        exp = np.zeros_like(trans.to_numpy(), dtype=float)
        for _ in range(100):
            exp += pd.crosstab(pd.Categorical(np.asarray(names)[nearest[pa]], categories=names),
                               pd.Categorical(np.asarray(names)[nearest[draw()]], categories=names),
                               dropna=False).to_numpy()
        res.update({
            "persistence": float(same.mean()), "persistence_null": float(null_same),
            "transitions": trans,
            "transitions_expected": pd.DataFrame(exp / 100, index=trans.index, columns=trans.columns),
        })
        pairs["programme_earlier"] = np.asarray(names)[nearest[pa]]
        pairs["programme_later"] = np.asarray(names)[nearest[pb]]
        del clone_means

    get_uns(adata).setdefault(MEMORY_KEY, {})[time_key] = res
    extra = (f"; same programme {res['persistence']:.2f} vs {res['persistence_null']:.2f} for random clones"
             if "persistence" in res else "")
    log(f"clonal memory over '{time_key}': {uniq.size} clones, {pa.size} snapshot pairs; memory index "
        f"{mem:.2f} [{ci[0]:.2f}, {ci[1]:.2f}], p = {p_value:.3g}{extra}.", verbose)
    return res
