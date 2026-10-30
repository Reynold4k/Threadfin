"""Clonal programmes: groups of clones with similar state profiles.

A *clonal programme* is a set of genetically distinct clones whose cells,
relative to their sampling context, occupy similar transcriptional states
(for example ASC-rich clones, or clones whose cells are mostly germinal-centre
B cells). Programmes are found by Leiden community detection
on a k-nearest-neighbour graph of clone profiles
(:func:`threadfin.tl.clone_profiles`).

Community detection always returns *some* partition, even when clones vary
continuously without forming groups. Two safeguards address this:

1. **Split test.** The Leiden communities are arranged in a tree (average
   linkage of their centroids) and tested top-down: at every node, the clones
   below it are compared with a single multivariate Gaussian fitted to them
   (same mean and covariance). The statistic is the 2-means cluster index
   (within / total sum of squares, as in SigClust, Liu et al. 2008); the
   null distribution comes from re-clustering Gaussian samples, so the test
   repeats the 2-means search, but does not repeat the full data-dependent
   Leiden/tree/resolution selection. This is an approximate diagnostic. A
   node is split only if p <= alpha * (n_node - 1) / (n_all - 1), with
   n the number of clones. This allocation follows hierarchical-testing
   work (Meinshausen 2008; sc-SHC, Grabski et al. 2023), but this implementation
   does not establish family-wise error control for the selected tree. Communities below a non-significant
   node are merged. When even the root is not significant, the clones form
   a single programme. The test detects any departure from one Gaussian
   group, which graded variation along a path can also produce, so a split
   says that the clone profiles are not one Gaussian cloud, and one
   programme is not by itself evidence of a continuum (see :func:`split_test`).
2. **Bootstrap stability.** Cells are resampled within clones (Poisson
   bootstrap) with the fitted profile model held fixed (the set of clones,
   the smoothing neighbours and the variance components), profiles are
   re-estimated and the graph, Leiden and split test are repeated; each
   programme is matched to its best bootstrap counterpart by Jaccard overlap
   (as in Hennig 2007). This measures how robust a programme is to the
   sampling of cells. Programmes with mean Jaccard >= 0.75 are stable; below
   0.6 they should not be interpreted.

The Leiden resolution is chosen as the finest one whose size-weighted
stability reaches the threshold (otherwise the most stable one, with a
warning), instead of being fixed by hand.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import get_uns, h5ad_safe, log, require_obs, stable_categorical
from .profiles import get_profiles, model_from_adata, project_features

PROGRAMME_KEY = "programmes"
DEFAULT_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0)


# --------------------------------------------------------------------------- graph + Leiden


def knn_graph(features: np.ndarray, n_neighbors: int = 15, reliability: np.ndarray | None = None):
    """Weighted undirected kNN graph (igraph) with self-tuned Gaussian weights.

    ``reliability`` scales every edge by ``sqrt(rel_i * rel_j)``, so a clone seen in few cells still
    takes part in the clustering but pulls on it only as hard as its profile deserves. This is the
    alternative to a reliability threshold: nothing is removed, and a community of noisy profiles is
    held together by weak edges that Leiden will not keep as a separate module.
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    n = features.shape[0]
    k = int(min(n_neighbors, n - 1))
    if k < 1:
        raise ValueError("Need at least two clones to build a graph.")
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self
    sigma = np.maximum(dist[:, -1:], 1e-12)  # distance to the k-th neighbour
    w = np.exp(-(dist**2) / sigma**2)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
    # merge (i,j)/(j,i) duplicates, keeping the larger weight
    key = lo.astype(np.int64) * n + hi
    order = np.lexsort((-w.ravel(), key))
    key_sorted = key[order]
    first = np.ones(key_sorted.size, dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    sel = order[first]
    weights = w.ravel()[sel]
    if reliability is not None:
        r = np.clip(np.asarray(reliability, dtype=float), 1e-6, 1.0)
        weights = weights * np.sqrt(r[lo[sel]] * r[hi[sel]])
    g = ig.Graph(n=n, edges=list(zip(lo[sel].tolist(), hi[sel].tolist())), directed=False)
    g.es["weight"] = weights.tolist()
    return g


def leiden(graph, resolution: float, seed: int = 0) -> np.ndarray:
    """Leiden partition (RB configuration model) -> integer labels."""
    import leidenalg

    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return np.asarray(part.membership, dtype=np.int64)


def _leiden_many(tasks, n_jobs: int = 1) -> list:
    """Leiden partitions for ``(graph, resolution, seed)`` tasks, in parallel processes when ``n_jobs > 1``.

    Each partition depends only on its graph, resolution and seed, so the result is the same for any ``n_jobs``.
    """
    if n_jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=min(n_jobs, len(tasks)))(delayed(leiden)(g, r, s) for g, r, s in tasks)
    return [leiden(g, r, s) for g, r, s in tasks]


def jaccard_match(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-match Jaccard of every reference cluster, and per-item agreement.

    Returns ``(jaccard_per_ref_cluster, agrees)``; ``agrees[i]`` is True when
    item ``i`` sits in the bootstrap cluster that best matches its reference
    cluster.
    """
    import scipy.sparse as sp

    kr, ko = int(ref.max()) + 1, int(other.max()) + 1
    m = sp.coo_matrix((np.ones(ref.size), (ref, other)), shape=(kr, ko)).toarray()
    size_r, size_o = m.sum(axis=1), m.sum(axis=0)
    union = size_r[:, None] + size_o[None, :] - m
    jac = np.divide(m, union, out=np.zeros_like(m), where=union > 0)
    best = jac.argmax(axis=1)
    return jac.max(axis=1), other == best[ref]


# --------------------------------------------------------------------------- split test


def _two_means_index(x: np.ndarray, rng, extra_split=None, n_random: int = 2, n_iter: int = 30) -> float:
    """Smallest 2-means cluster index (within / total sum of squares) over a few starts.

    Starts: the sign of the first principal component, ``n_random`` random
    pairs of centres and, if given, ``extra_split`` (a boolean labelling).
    """
    xc = x - x.mean(axis=0)
    tss = float(np.einsum("ij,ij->", xc, xc))
    if xc.shape[0] < 3 or tss <= 0:
        return 1.0
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    starts = [xc @ vt[0] > 0]
    for _ in range(n_random):
        a, b = rng.choice(xc.shape[0], size=2, replace=False)
        starts.append(((xc - xc[a]) ** 2).sum(1) < ((xc - xc[b]) ** 2).sum(1))
    if extra_split is not None:
        starts.append(np.asarray(extra_split, dtype=bool))
    best = np.inf
    for lab in starts:
        for _ in range(n_iter):  # Lloyd iterations for two centres
            if lab.all() or not lab.any():
                break
            c1, c0 = xc[lab].mean(axis=0), xc[~lab].mean(axis=0)
            new = xc @ (c1 - c0) > 0.5 * (c1 @ c1 - c0 @ c0)
            if np.array_equal(new, lab):
                break
            lab = new
        if lab.all() or not lab.any():
            continue
        wss = tss - lab.sum() * np.sum(xc[lab].mean(0) ** 2) - (~lab).sum() * np.sum(xc[~lab].mean(0) ** 2)
        best = min(best, wss)
    return float(best / tss) if np.isfinite(best) else 1.0


def _two_means_labels(x: np.ndarray, rng, extra_split=None, n_random: int = 2, n_iter: int = 30):
    """The labelling attaining the smallest 2-means cluster index (same starts as :func:`_two_means_index`)."""
    xc = x - x.mean(axis=0)
    tss = float(np.einsum("ij,ij->", xc, xc))
    if xc.shape[0] < 3 or tss <= 0:
        return None
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    starts = [xc @ vt[0] > 0]
    for _ in range(n_random):
        a, b = rng.choice(xc.shape[0], size=2, replace=False)
        starts.append(((xc - xc[a]) ** 2).sum(1) < ((xc - xc[b]) ** 2).sum(1))
    if extra_split is not None:
        starts.append(np.asarray(extra_split, dtype=bool))
    best, best_lab = np.inf, None
    for lab in starts:
        for _ in range(n_iter):
            if lab.all() or not lab.any():
                break
            c1, c0 = xc[lab].mean(axis=0), xc[~lab].mean(axis=0)
            new = xc @ (c1 - c0) > 0.5 * (c1 @ c1 - c0 @ c0)
            if np.array_equal(new, lab):
                break
            lab = new
        if lab.all() or not lab.any():
            continue
        wss = tss - lab.sum() * np.sum(xc[lab].mean(0) ** 2) - (~lab).sum() * np.sum(xc[~lab].mean(0) ** 2)
        if wss < best:
            best, best_lab = wss, lab.copy()
    return best_lab


def split_test(x: np.ndarray, split=None, *, n_null: int = 100, random_state=0, return_null: bool = False,
               null: str = "total") -> dict:
    """Is ``x`` (clones x features) better described by two clusters than by one Gaussian?

    The 2-means cluster index of ``x`` is compared with its distribution in ``n_null`` Gaussian samples.
    ``null`` chooses that Gaussian's covariance. **Both options have been calibrated on real clone profiles
    with a planted two-group structure; the measured numbers are quoted below, and neither setting is satisfactory:**

    ``"total"`` (default)
        the sample covariance of ``x`` itself (SigClust-style). Conservative under its null: a single Gaussian
        group is rejected far less often than ``alpha`` (0 of 30 simulations on real clone profiles, 0 of 600
        on synthetic data). Its power depends on the direction of a separation relative to the covariance: two
        synthetic Gaussian groups four within-group standard deviations apart are detected, while a two-group
        structure planted in real mouse_rbd clone profiles was missed in 30 of 30 simulations, because a split
        inflates the covariance along the splitting direction and the null inherits that inflation. Graded,
        non-Gaussian variation (clones spread evenly along a path) is also rejected. **A significant split
        therefore says that the profiles are not one Gaussian group, not that they form discrete types; a
        non-significant one is not evidence of a continuum.**
    ``"within"``
        the covariance pooled *within* the two candidate groups. **Invalid — do not use for inference.** It was
        tried as a fix and measured: on mouse_rbd clone profiles it reports a split in 30 of 30 simulations with
        *no* planted structure (false-positive rate 1.0), because removing the candidate split from the
        covariance estimate is circular. Kept only so that the calibration can be reproduced.

    The p-value uses a normal fit to the null indices (so that it can be smaller than ``1 / n_null``); the
    empirical p-value is also returned (and, with ``return_null``, the null indices themselves).
    """
    from scipy.stats import norm

    if null not in ("within", "total"):
        raise ValueError("null must be 'within' or 'total'.")
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    observed = _two_means_index(x, rng, extra_split=split)
    if null == "within":
        lab = _two_means_labels(x, rng, extra_split=split)
        if lab is None or lab.all() or not lab.any():
            cov = np.atleast_2d(np.cov(x, rowvar=False))
        else:
            centred = x.copy()
            centred[lab] -= x[lab].mean(axis=0)
            centred[~lab] -= x[~lab].mean(axis=0)
            dof = max(x.shape[0] - 2, 1)
            cov = np.atleast_2d(centred.T @ centred / dof)
    else:
        cov = np.atleast_2d(np.cov(x, rowvar=False))
    w, v = np.linalg.eigh(cov)
    root = v * np.sqrt(np.clip(w, 0, None))
    null = np.array([_two_means_index(rng.standard_normal(x.shape) @ root.T, rng) for _ in range(n_null)])
    sd = float(null.std(ddof=1)) if n_null > 1 else np.nan
    p_norm = float(norm.cdf((observed - null.mean()) / sd)) if np.isfinite(sd) and sd > 0 else float("nan")
    p_emp = float((1 + np.sum(null <= observed)) / (n_null + 1))
    if not np.isfinite(p_norm):
        p_norm = p_emp  # A degenerate null must not turn identical data into p=0.
    out = {"cluster_index": observed, "null_mean": float(null.mean()), "null_sd": sd,
           "pvalue": p_norm, "pvalue_empirical": p_emp}
    if return_null:
        out["null"] = null
    return out


def _merge_by_split_test(feats, labels, alpha, n_null, rng, null="total"):
    """Test the community tree top-down and merge communities below non-significant nodes.

    Returns ``(merged_labels, tests)`` with merged labels ``0..K-1`` and one
    row per tested node.
    """
    from scipy.cluster.hierarchy import linkage, to_tree

    names = np.unique(labels)
    if names.size < 2:
        return np.zeros(labels.size, dtype=np.int64), pd.DataFrame()
    centroids = np.vstack([feats[labels == k].mean(axis=0) for k in names])
    root = to_tree(linkage(centroids, method="average"))
    total = labels.size
    rows, groups = [], []

    def visit(node):
        if node.is_leaf():
            groups.append([names[node.id]])
            return
        a = [names[i] for i in node.left.pre_order()]
        b = [names[i] for i in node.right.pre_order()]
        inside = np.isin(labels, a + b)
        res = split_test(feats[inside], np.isin(labels[inside], a), n_null=n_null, null=null,
                         random_state=int(rng.integers(2**31)))
        level = alpha * (inside.sum() - 1) / (total - 1)
        # communities as comma-separated text, so the table can be written to h5ad with the AnnData
        rows.append({"communities_left": ",".join(str(int(c)) for c in a),
                     "communities_right": ",".join(str(int(c)) for c in b),
                     "n_communities_left": len(a), "n_communities_right": len(b),
                     "n_clones": int(inside.sum()), **res,
                     "alpha_node": float(level), "split": bool(res["pvalue"] <= level)})
        if res["pvalue"] <= level:
            visit(node.left)
            visit(node.right)
        else:
            groups.append(a + b)

    visit(root)
    merged = np.empty(labels.size, dtype=np.int64)
    for g, members in enumerate(groups):
        merged[np.isin(labels, members)] = g
    tests = pd.DataFrame(rows)
    if not tests.empty:
        to_group = {int(c): g for g, members in enumerate(groups) for c in members}
        for side in ("left", "right"):
            tests[f"groups_{side}"] = [sorted({to_group[int(c)] for c in cs.split(",")})
                                       for cs in tests[f"communities_{side}"]]
    return merged, tests


# --------------------------------------------------------------------------- bootstrap


def _bootstrap_features(model, adata, eligible_codes, n_boot, rng):
    """Feature matrices of the eligible clones under ``n_boot`` Poisson bootstraps."""
    n_all = len(model.clone_index)
    remap = -np.ones(n_all, dtype=np.int64)
    remap[eligible_codes] = np.arange(eligible_codes.size)
    groups = np.where(model.clone_codes >= 0, remap[np.maximum(model.clone_codes, 0)], -1)
    in_group = groups >= 0
    out = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, size=groups.size).astype(float)
        # a clone that draws no cells keeps its original cells (avoid empty clones)
        tot = np.bincount(groups[in_group], weights=w[in_group], minlength=eligible_codes.size)
        empty = np.flatnonzero(tot == 0)
        if empty.size:
            w[np.isin(groups, empty)] = 1.0
        blup, _, _ = model.group_blups(groups, eligible_codes.size, weights=w)
        out.append(project_features(adata, blup))
    return out


# --------------------------------------------------------------------------- public API



def embed_clones(
    adata,
    *,
    n_neighbors: int | str = "auto",
    min_dist: float | str = "auto",
    spread: float = 1.0,
    min_reliability: float = 0.0,
    coherence_target: float | str = "auto",
    tune_kwargs: dict | None = None,
    random_state: int = 0,
    n_jobs: int | None = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Lay the clone profiles out in two dimensions — **every** profiled clone, by default.

    This is the clone map the rest of the package is built around, and it is deliberately separate from
    :func:`find_programmes`. Programme discovery is a clustering problem and needs reasonably estimated
    profiles, so it keeps its ``min_reliability`` gate. The map is a *display* of the clones that exist,
    and filtering it by reliability would throw away the data the map is meant to show (a floor of 0.5
    removes 75-100 % of the profiled clones in the germinal-centre datasets of the manuscript).

    The map does not change any statistic: profiles, coherence, programmes and tests are computed in
    profile space, never on the map.

    Parameters
    ----------
    n_neighbors, min_dist
        UMAP settings. ``"auto"`` (the default for both) chooses them per dataset with
        :func:`threadfin.tl.tune_clone_map`: a ladder of settings, each laid out with several seeds and
        smoothed over its neighbours, finds the most faithful layout that is no more fragmented than the
        clone profiles themselves, so a continuum of clones is drawn as one connected picture. Pass numbers to set them
        by hand (``n_neighbors=50, min_dist=0.8`` reproduces the fixed default of Threadfin 4.2); a
        number for one and ``"auto"`` for the other tunes only the second.
    spread
        UMAP ``spread``.
    min_reliability
        Reliability floor. ``0.0`` (the default) keeps every clone that has a profile; raise it only to
        reproduce an older filtered layout.
    coherence_target, tune_kwargs
        Passed to :func:`threadfin.tl.tune_clone_map` when a setting is ``"auto"``.
    random_state
        Seed for the tuner and for the final layout.
    n_jobs
        Processes for the layouts of the automatic search (``-1``: all available cores). The chosen
        settings and the map do not depend on it.

    Returns
    -------
    The clone table with ``x`` and ``y`` added. ``adata.uns['threadfin']['profiles']['map_params']``
    records the settings, how they were chosen (``"tuning"``, with the full optimisation history), the
    quality of the final map (``"quality"``: coherence, local and global fidelity) and how many clones
    were laid out.
    """
    from .maptune import layout_clone_map, tune_clone_map

    prof = get_profiles(adata)
    table, feats_all = prof["clone_table"], prof["features"]
    use = table.index[table["reliability"] >= min_reliability]
    if use.size < 15:
        raise ValueError(
            f"Only {use.size} clones to embed (need 15). Profile more clones, or lower min_cells in "
            "clone_profiles.")
    for name, value in (("n_neighbors", n_neighbors), ("min_dist", min_dist)):
        if isinstance(value, str) and value != "auto":
            raise ValueError(f"{name} must be 'auto' or a number, got {value!r}.")
    feats = feats_all.loc[use].to_numpy()
    tuning, repulsion = None, 1.0
    if n_neighbors == "auto" or min_dist == "auto":
        tuning = tune_clone_map(feats, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
                                coherence_target=coherence_target, random_state=random_state,
                                n_jobs=n_jobs, verbose=verbose, **(tune_kwargs or {}))
        n_neighbors, min_dist, repulsion = tuning.n_neighbors, tuning.min_dist, tuning.repulsion_strength

    xy, quality, setting, attempts = layout_clone_map(
        feats, n_neighbors=n_neighbors, min_dist=min_dist, repulsion_strength=repulsion, spread=spread,
        coherence_target=coherence_target, tuning=tuning, random_state=random_state)
    n_neighbors, min_dist, repulsion, layout_seed = setting
    table["x"], table["y"] = np.nan, np.nan
    table.loc[use, "x"], table.loc[use, "y"] = xy[:, 0], xy[:, 1]
    prof["clone_table"] = table
    prof["map_params"] = {"n_neighbors": int(n_neighbors), "min_dist": float(min_dist),
                          "repulsion_strength": float(repulsion), "spread": float(spread),
                          "min_reliability": float(min_reliability), "random_state": int(random_state),
                          "layout_seed": int(layout_seed), "n_clones": int(use.size),
                          "n_profiled": int(len(table)),
                          "chosen_by": "tune_clone_map" if tuning is not None else "user",
                          "quality": {k: float(v) for k, v in quality.items()},
                          "attempts": pd.DataFrame(attempts)}
    if tuning is not None:
        prof["map_params"]["tuning"] = tuning.to_uns()
    rep_txt = "" if repulsion == 1.0 else f", repulsion_strength {repulsion:g}"
    log(f"clone map: {use.size:,} of {len(table):,} profiled clones laid out (n_neighbors {int(n_neighbors)}, "
        f"min_dist {float(min_dist):.2f}{rep_txt}{', chosen automatically' if tuning is not None else ''}); "
        f"{100 * quality['coherence']:.1f}% of clones in one connected piece.", verbose)
    if tuning is None and quality["coherence"] < quality["coherence_target"] - 0.005:
        log(f"  the map is more fragmented than the clone profiles (which keep "
            f"{100 * quality['coherence_target']:.1f}% of clones connected); n_neighbors='auto', "
            "min_dist='auto' chooses settings that keep it together.", verbose)
    return table


def find_programmes(
    adata,
    *,
    resolution: float | str = "auto",
    resolutions=DEFAULT_RESOLUTIONS,
    n_neighbors: int = 15,
    min_reliability: float = 0.0,
    weight_by_reliability: bool = True,
    n_boot: int = 30,
    stability_threshold: float = 0.75,
    min_programme_size: int = 5,
    min_clones: int = 30,
    test_splits: bool = True,
    alpha: float = 0.05,
    n_null: int = 100,
    split_null: str = "total",
    assign_remaining: bool = True,
    min_posterior: float = 0.7,
    random_state: int = 0,
    n_jobs: int | None = 1,
    key_added: str = "clone_programme",
    verbose: bool = True,
) -> pd.DataFrame:
    """Group clones into clonal programmes, with bootstrap stability.

    Parameters
    ----------
    adata
        AnnData after :func:`threadfin.tl.clone_profiles`.
    resolution
        Leiden resolution, or ``"auto"`` to scan ``resolutions`` and keep the
        finest partition whose size-weighted stability is at least
        ``stability_threshold`` (falling back to the most stable partition).
    n_neighbors
        Neighbours per clone in the clone graph.
    weight_by_reliability
        Scale each graph edge by ``sqrt(rel_i * rel_j)`` so that every clone is clustered but a
        thinly observed one pulls weakly on its neighbours. Clones of a single cell are not profiled
        (``min_cells=2`` in :func:`threadfin.tl.clone_profiles`); that, rather than the weighting, keeps
        unfiltered runs from inventing programmes on a simulated one-programme repertoire.
    min_reliability
        Reliability floor for clustering; ``0.0`` (the default) clusters every profiled clone.
        Threadfin does not filter clones on reliability. Profiles are already shrunk: a clone seen in
        few cells is pulled towards the average profile, so its influence is down-weighted in
        proportion to how little was seen. A reliability floor on top of that discards the same
        information twice; in the germinal-centre datasets of the manuscript a floor of 0.5 removes
        75-100 % of the profiled clones.
    n_boot
        Bootstrap replicates for stability (0 disables stability, which is
        only allowed with a fixed ``resolution``).
    min_programme_size
        Groups with fewer clones are not reported as programmes (their clones
        stay unassigned); resolutions that leave more than 10% of clones in
        such groups are not selected.
    min_clones
        Minimum number of reliable clones; with fewer, programme detection is
        refused (report the coherence test only).
    test_splits, alpha, n_null
        Merge communities that are not significantly better described as two
        groups than as one Gaussian (see the module docstring); ``alpha`` is
        a nominal hierarchical threshold (not a validated family-wise error guarantee) and ``n_null`` the
        number of Gaussian null samples per test. The node tests are stored
        in ``adata.uns['threadfin']['programmes']['split_tests']``. A single
        programme means "no distinct programmes".
    assign_remaining
        Programmes are *defined* on reliable ("core") clones only. With this
        option every other profiled clone is then *assigned* to the most
        likely programme by a Gaussian classifier whose noise term shrinks
        with clone size (see ``_assign_remaining``); assignments with
        posterior >= ``min_posterior`` go to ``<key_added>_assigned``.
    n_jobs
        Processes for the Leiden runs on the bootstrap graphs (``-1``: all
        available cores). Programmes and stabilities are identical for any
        value; only the run time changes.
    key_added
        ``obs`` column receiving each cell's programme (cells of clones that
        were not clustered get NaN).

    Returns
    -------
    Programme summary: ``n_clones``, ``n_cells``, ``stability`` (mean
    bootstrap Jaccard) and ``mean_reliability`` per programme. Clone-level
    assignments (with per-clone bootstrap ``confidence``) are added to
    ``adata.uns['threadfin']['profiles']['clone_table']``.
    """
    prof = get_profiles(adata)
    table = prof["clone_table"]
    feats_all = prof["features"]
    eligible = table.index[table["reliability"] >= min_reliability]
    if eligible.size < min_clones:
        raise ValueError(
            (f"only {eligible.size} clones have two or more cells (need {min_clones}); too few expanded clones "
             "to define programmes. The coherence test above still applies." if min_reliability <= 0 else
             f"only {eligible.size} clones reach reliability {min_reliability} (need {min_clones}); too few "
             "expanded clones to define programmes. Report the coherence test only, or lower min_reliability.")
        )
    feats = feats_all.loc[eligible].to_numpy()
    rel = table.loc[eligible, "reliability"].to_numpy() if weight_by_reliability else None
    graph = knn_graph(feats, n_neighbors, rel)

    auto = isinstance(resolution, str)
    if auto and resolution != "auto":
        raise ValueError("resolution must be a number or 'auto'.")
    if auto and n_boot < 5:
        raise ValueError(f"choosing the resolution automatically needs n_boot >= 5 bootstrap replicates "
                         f"(n_boot={n_boot}).")
    res_grid = list(resolutions) if auto else [float(resolution)]

    rng = np.random.default_rng(random_state)
    boot_graphs, boot_feats = [], []
    if n_boot > 0:
        model = model_from_adata(adata)
        code_of = {str(c): i for i, c in enumerate(model.clone_index.astype(str))}
        eligible_codes = np.array([code_of[c] for c in eligible], dtype=np.int64)
        boot_feats = _bootstrap_features(model, adata, eligible_codes, n_boot, rng)
        boot_graphs = [knn_graph(f, n_neighbors, rel) for f in boot_feats]

    # every Leiden partition of the scan (reference and bootstrap graphs at every resolution), computed once
    from .maptune import resolve_n_jobs

    tasks = [(graph, res, random_state) for res in res_grid]
    tasks += [(bg, res, random_state + b + 1) for res in res_grid for b, bg in enumerate(boot_graphs)]
    parts = iter(_leiden_many(tasks, resolve_n_jobs(n_jobs)))
    ref_part = {res: next(parts) for res in res_grid}
    boot_part = {(res, b): next(parts) for res in res_grid for b in range(len(boot_graphs))}

    scan_rows, partitions = [], {}
    for res in res_grid:
        ref = ref_part[res]
        k = int(ref.max()) + 1
        jac_sum = np.zeros(k)
        agree = np.zeros(ref.size)
        for b in range(len(boot_graphs)):
            jac, ok = jaccard_match(ref, boot_part[(res, b)])
            jac_sum += jac
            agree += ok
        nb = max(len(boot_graphs), 1)
        jac_mean = jac_sum / nb if boot_graphs else np.full(k, np.nan)
        sizes = np.bincount(ref, minlength=k)
        big = sizes >= min_programme_size
        frac_small = float(sizes[~big].sum() / sizes.sum())
        weighted = (float(np.sum(jac_mean[big] * sizes[big]) / max(sizes[big].sum(), 1))
                    if boot_graphs and big.any() else np.nan)
        partitions[res] = (ref, jac_mean, agree / nb if boot_graphs else np.full(ref.size, np.nan))
        scan_rows.append({
            "resolution": res, "n_programmes": int(big.sum()), "n_groups": k,
            "frac_clones_in_small_groups": frac_small, "stability_weighted": weighted,
            "stability_min": float(np.min(jac_mean[big])) if boot_graphs and big.any() else np.nan,
        })
    scan = pd.DataFrame(scan_rows)

    caution = None          # raised only if programmes survive the split test (below)
    if auto:
        multi = scan[(scan["n_programmes"] >= 2) & (scan["frac_clones_in_small_groups"] <= 0.10)]
        stable = multi[multi["stability_weighted"] >= stability_threshold]
        if not stable.empty:
            best = stable.sort_values(["n_programmes", "stability_weighted"], ascending=False).iloc[0]
        elif not multi.empty:
            best = multi.sort_values("stability_weighted", ascending=False).iloc[0]
            if boot_graphs:     # without bootstrap replicates there is no stability to fall short of
                caution = (f"No resolution reached a clone-weighted mean stability of {stability_threshold} "
                           f"across programmes; using the most stable partition (mean stability "
                           f"{best['stability_weighted']:.2f}). Single programmes can still be stable: judge "
                           "each by its own stability (result.programmes['stability']).")
        else:
            best = scan.iloc[0]
        chosen = float(best["resolution"])
    else:
        chosen = res_grid[0]

    ref, jac_mean, confidence = partitions[chosen]
    # groups below the minimum size are not programmes: their clones stay unassigned
    sizes = np.bincount(ref)
    keep_mask = sizes[ref] >= min_programme_size
    eligible = eligible[keep_mask]
    feats = feats[keep_mask]
    confidence = confidence[keep_mask]
    ref = ref[keep_mask]
    split_tests = pd.DataFrame()
    if test_splits:
        ref, split_tests = _merge_by_split_test(feats, ref, alpha, n_null, rng, split_null)
        if boot_graphs:
            # stability of the final programmes: repeat Leiden *and* the split test in every bootstrap
            jac_mean = np.zeros(int(ref.max()) + 1)
            agree = np.zeros(ref.size)
            for b, bf in enumerate(boot_feats):
                bl = boot_part[(chosen, b)][keep_mask]
                bl, _ = _merge_by_split_test(bf[keep_mask], bl, alpha, max(n_null // 4, 20), rng, split_null)
                jac, ok = jaccard_match(ref, bl)
                jac_mean += jac
                agree += ok
            jac_mean /= len(boot_graphs)
            confidence = agree / len(boot_graphs)
        else:
            jac_mean = np.full(int(ref.max()) + 1, np.nan)
    labels = stable_categorical(ref, prefix="P")
    raw_to_label = dict(zip(ref, labels))

    # write clone-level results
    table = table.copy()
    table[key_added] = pd.Categorical([None] * len(table), categories=labels.categories)
    table.loc[eligible, key_added] = np.asarray(labels)
    table["confidence"] = np.nan
    table.loc[eligible, "confidence"] = confidence
    table["core"] = table.index.isin(eligible)
    if assign_remaining:
        model = model if n_boot > 0 else model_from_adata(adata)
        prof["clone_table"] = table
        best_lab, post = _assign_remaining(adata, model, eligible, np.asarray(labels, dtype=str))
        assigned = pd.Series(np.where(post >= min_posterior, best_lab, None), index=table.index, dtype=object)
        assigned.loc[eligible] = np.asarray(labels, dtype=str)  # core clones keep their own programme
        table[f"{key_added}_assigned"] = pd.Categorical(assigned, categories=labels.categories)
        table["assignment_posterior"] = post
    prof["clone_table"] = table

    # map to cells
    clone_key = prof["params"]["clone_key"]
    cell_clone = adata.obs[clone_key].astype("object")
    for col in [key_added] + ([f"{key_added}_assigned"] if assign_remaining else []):
        mapping = table[col].dropna().astype(str)
        adata.obs[col] = pd.Categorical(
            cell_clone.map(lambda c, m=mapping: m.get(str(c)) if pd.notna(c) else None),
            categories=labels.categories,
        )

    stab = {raw_to_label[r]: float(jac_mean[r]) for r in np.unique(ref)}
    elig_tab = table.loc[eligible]
    summary = (
        elig_tab.groupby(key_added, observed=True)
        .agg(n_clones=("n_cells", "size"), n_cells=("n_cells", "sum"),
             mean_reliability=("reliability", "mean"), mean_confidence=("confidence", "mean"))
    )
    summary.index = pd.Index(summary.index.astype(str), name="programme")
    summary["stability"] = [stab[p] for p in summary.index]
    if assign_remaining:
        n_assigned = table[f"{key_added}_assigned"].astype(str).value_counts()
        summary["n_clones_with_assigned"] = [int(n_assigned.get(p, 0)) for p in summary.index]
    if not split_tests.empty:  # name the two sides of every tested node by the final programmes
        split_tests = split_tests.copy()
        for side in ("left", "right"):
            split_tests[f"programmes_{side}"] = [", ".join(sorted({str(raw_to_label[g]) for g in gs if g in raw_to_label},
                                                                  key=lambda t: (len(t), t)))
                                                 for gs in split_tests.pop(f"groups_{side}")]
    get_uns(adata)[PROGRAMME_KEY] = {
        "summary": summary,
        "resolution_scan": h5ad_safe(scan),
        "split_tests": h5ad_safe(split_tests),
        "params": {
            "resolution": chosen, "auto": auto, "n_neighbors": int(n_neighbors),
            "weight_by_reliability": bool(weight_by_reliability),
            "test_splits": bool(test_splits), "alpha": float(alpha), "n_null": int(n_null),
            "split_null": str(split_null),
            "min_reliability": float(min_reliability), "n_boot": int(n_boot),
            "stability_threshold": float(stability_threshold), "key_added": key_added,
            "min_programme_size": int(min_programme_size),
            "assign_remaining": bool(assign_remaining), "min_posterior": float(min_posterior),
            "random_state": int(random_state),
        },
    }
    n_stable = int((summary["stability"] >= stability_threshold).sum()) if n_boot else 0
    if caution and summary.shape[0] >= 2 and n_stable < summary.shape[0]:
        # with a single programme left there is nothing to interpret, and the log says so
        warnings.warn(caution, stacklevel=2)
    extra = (f"; {int(table[f'{key_added}_assigned'].notna().sum())} of {len(table)} profiled clones "
             f"assigned with posterior >= {min_posterior}" if assign_remaining else "")
    if summary.shape[0] == 1:
        log(f"programmes: no significant split among {eligible.size} core clones (resolution {chosen}): one "
            "programme. The clones differ, but not as groups the split test can separate.", verbose)
    else:
        n_merged = int((~split_tests["split"]).sum()) if not split_tests.empty else 0
        merged_txt = f" after merging {n_merged} non-significant split(s)" if n_merged else ""
        log(
            f"programmes: {summary.shape[0]} programmes over {eligible.size} core clones "
            f"(resolution {chosen}{merged_txt}); {n_stable} with bootstrap stability >= {stability_threshold}{extra}.",
            verbose,
        )
    return summary


# --------------------------------------------------------------------------- posterior assignment


def _assign_remaining(adata, model, core_ids, core_labels):
    """Assign every profiled clone to a programme with a Gaussian classifier.

    Each programme ``P`` is a Gaussian over (unshrunk) clone mean profiles
    with centre ``mu_P`` and between-clone variance ``t_P`` estimated from its
    core clones (sampling noise removed). A clone with ``n`` cells is observed
    with extra noise ``sigma2 / n``, so small clones get flatter posteriors:

        p(P | clone) ~ pi_P * N(mean_clone; mu_P, t_P + sigma2 / n)

    Returns the most likely programme and its posterior for every row of the
    clone table.
    """
    import scipy.sparse as sp
    from scipy.special import logsumexp

    prof = get_profiles(adata)
    table = prof["clone_table"]
    code_of = {str(c): i for i, c in enumerate(model.clone_index.astype(str))}
    rows = np.array([code_of[c] for c in table.index], dtype=np.int64)
    g = model.clone_codes
    ok = np.flatnonzero(g >= 0)
    member = sp.csr_matrix((np.ones(ok.size), (g[ok], ok)), shape=(len(model.clone_index), g.size))
    n = np.asarray(member.sum(axis=1)).ravel()
    means = np.asarray(member @ model.residuals) / np.maximum(n, 1)[:, None] - model.vc.grand_mean
    means, n = means[rows], n[rows]
    red = prof.get("reduction")
    if red is None:
        x, noise = means, model.vc.sigma2
    else:
        comps = np.asarray(red["components"])
        x = (means - np.asarray(red["raw_mean"])) @ comps.T
        noise = (comps**2) @ model.vc.sigma2

    core_pos = table.index.get_indexer(core_ids)
    names = pd.Index(pd.unique(np.asarray(core_labels, dtype=str)))
    lab = names.get_indexer(np.asarray(core_labels, dtype=str))
    floor = 1e-6 * float(noise.mean()) + 1e-12
    mu, t, prior = [], [], []
    for k in range(len(names)):
        members = core_pos[lab == k]
        xm = x[members]
        mu.append(xm.mean(axis=0))
        t.append(np.maximum(xm.var(axis=0) - (noise[None, :] / n[members][:, None]).mean(axis=0), floor))
        prior.append(members.size)
    mu, t = np.vstack(mu), np.vstack(t)
    prior = np.log(np.asarray(prior, dtype=float) / np.sum(prior))
    var = t[None, :, :] + noise[None, None, :] / np.maximum(n, 1)[:, None, None]
    ll = -0.5 * (((x[:, None, :] - mu[None, :, :]) ** 2) / var + np.log(var)).sum(axis=2) + prior[None, :]
    post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    return np.asarray(names)[post.argmax(axis=1)], post.max(axis=1)


def get_programmes(adata) -> dict:
    """Stored programme results (raises if :func:`find_programmes` was not run)."""
    out = get_uns(adata).get(PROGRAMME_KEY)
    if out is None:
        raise ValueError("Run threadfin.tl.find_programmes() (or threadfin.run()) first.")
    return out


def programme_composition(adata, state_key: str, *, key: str | None = None, weight: str = "cells",
                          clone_key: str | None = None) -> pd.DataFrame:
    """Share of each cell state in each programme (rows sum to 1).

    ``weight="cells"`` (default) counts every cell once: the fraction of the programme's cells in each state,
    to which an expanded clone contributes in proportion to its sampled cells. ``weight="clones"`` counts every
    clone once: the share of each clone's cells in each state, averaged over the programme's clones; it describes
    a typical clone of the programme and is not dominated by a few large clones. Only cells with both a
    programme and a state are counted. ``clone_key`` (for ``weight="clones"``) defaults to the clone column of
    :func:`threadfin.tl.clone_profiles`.
    """
    if weight not in {"cells", "clones"}:
        raise ValueError(f"weight must be 'cells' or 'clones', not {weight!r}.")
    key = key or get_programmes(adata)["params"]["key_added"]
    if weight == "cells":
        df = adata.obs[[key, state_key]].dropna()
        tab = pd.crosstab(df[key], df[state_key])
        return tab.div(tab.sum(axis=1), axis=0)
    clone_key = clone_key or get_profiles(adata)["params"]["clone_key"]
    df = adata.obs[[key, clone_key, state_key]].dropna()
    per_clone = pd.crosstab([df[key], df[clone_key].astype(str)], df[state_key])
    shares = per_clone.div(per_clone.sum(axis=1), axis=0)
    return shares.groupby(level=0, observed=True).mean()


def _clone_expression(adata, *, clone_key, cell_mask, layer, context_key, state_key, state, min_cells_in_state,
                      min_frac_expressed, caller):
    """Mean context-centred expression of each clone's cells (optionally only its cells of one state).

    Returns ``(cells, clone_codes, clone_index, genes, clone_means)``: the cells used, their clone codes, the
    clone ids, the indices of the genes expressed in at least ``min_frac_expressed`` of those cells, and the
    clones x genes matrix of means, each minus the mean of the same-state cells of the clone's contexts.
    """
    import scipy.sparse as sp

    from ._utils import codes

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{caller}: adata.layers has no '{layer}' (layers: "
                         f"{', '.join(map(repr, adata.layers.keys())) or 'none'}). threadfin.pp.prepare_embedding "
                         "(and threadfin.run without basis=) writes 'log_norm'; otherwise pass the layer that holds "
                         "normalised, log-transformed counts, or layer=None to use adata.X.")
    x = adata.layers[layer] if layer is not None else adata.X
    x = sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr()
    in_state = np.ones(adata.n_obs, dtype=bool)
    if state_key is not None:
        require_obs(adata, state_key, context=f"{caller}(state_key=...)")
        levels = adata.obs[state_key].dropna().astype(str)
        if state is None or str(state) not in set(levels):
            raise ValueError(f"state={state!r} is not a level of '{state_key}'; levels: "
                             f"{', '.join(sorted(set(levels))[:12])}.")
        in_state = (adata.obs[state_key].astype(str) == str(state)).to_numpy()
    cells = np.flatnonzero(np.asarray(cell_mask) & adata.obs[clone_key].notna().to_numpy() & in_state)
    clone_codes, clone_index = codes(adata.obs[clone_key].iloc[cells].astype(str))
    if min_cells_in_state > 1:                    # keep clones with enough cells in the state
        counts = np.bincount(clone_codes, minlength=len(clone_index))
        cells = cells[(counts >= min_cells_in_state)[clone_codes]]
        clone_codes, clone_index = codes(adata.obs[clone_key].iloc[cells].astype(str))
    n_clones = len(clone_index)
    member = sp.csr_matrix((np.ones(cells.size), (clone_codes, np.arange(cells.size))), shape=(n_clones, cells.size))
    member = sp.diags(1.0 / np.maximum(np.asarray(member.sum(axis=1)).ravel(), 1)) @ member
    expressed = np.asarray((x[cells] > 0).mean(axis=0)).ravel() >= min_frac_expressed
    genes = np.flatnonzero(expressed)
    clone_means = np.asarray((member @ x[cells][:, genes]).toarray())
    if context_key is not None:
        # subtract, for every clone, the mean expression of the contexts its cells came from
        ctx, ctx_index = codes(adata.obs[context_key])
        ok = np.flatnonzero((ctx >= 0) & in_state)       # within a state, centre on that state's cells
        cmember = sp.csr_matrix((np.ones(ok.size), (ctx[ok], ok)), shape=(len(ctx_index), adata.n_obs))
        cmember = sp.diags(1.0 / np.maximum(np.asarray(cmember.sum(axis=1)).ravel(), 1)) @ cmember
        ctx_means = np.asarray((cmember @ x[:, genes]).toarray())
        cell_ctx = sp.csr_matrix((np.ones(cells.size), (np.arange(cells.size), np.maximum(ctx[cells], 0))),
                                 shape=(cells.size, len(ctx_index)))
        clone_means -= np.asarray((member @ cell_ctx).toarray()) @ ctx_means
    return cells, clone_codes, clone_index, genes, clone_means


def programme_markers(
    adata,
    *,
    programme_key: str = "clone_programme",
    layer: str | None = "log_norm",
    context_key: str | None = "auto",
    min_frac_expressed: float = 0.05,
    n_top: int = 25,
    state_key: str | None = None,
    state: str | None = None,
    min_cells_in_state: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Genes that distinguish programmes, tested with clones as replicates.

    Every clone is summarised by the mean context-centred log expression of
    its cells; for each programme, clones inside are compared with clones in
    other programmes by a Mann-Whitney test per gene (BH FDR). Testing across
    clones avoids the pseudoreplication of cell-level marker tests, where one
    expanded clone contributes hundreds of correlated cells.

    Programmes usually differ in which cell states their clones contain, so the genes found this way are
    largely the markers of those states. To ask what differs between programmes *within one state* — for
    example between the germinal-centre B cells of clones from two programmes — pass ``state_key`` and
    ``state``: each clone is then summarised by its cells in that state only (clones with at least
    ``min_cells_in_state`` such cells), centred on the cells of the same state in its context.

    Returns
    -------
    Tidy DataFrame ``[programme, gene, mean_diff, auc, pvalue, fdr, rank]`` (and ``state`` when given)
    with the top ``n_top`` up-regulated genes per programme.
    """
    import scipy.sparse as sp
    from scipy.stats import rankdata

    from ._utils import bh_fdr, codes

    prof = get_profiles(adata)
    clone_key = prof["params"]["clone_key"]
    if context_key == "auto":
        context_key = prof["params"].get("context_key")
    lab = adata.obs[programme_key]
    cells, clone_codes, clone_index, genes, clone_means = _clone_expression(
        adata, clone_key=clone_key, cell_mask=lab.notna().to_numpy(), layer=layer, context_key=context_key,
        state_key=state_key, state=state, min_cells_in_state=min_cells_in_state,
        min_frac_expressed=min_frac_expressed, caller="programme_markers")
    n_clones = len(clone_index)
    if n_clones < 6:
        raise ValueError(f"Only {n_clones} clones have cells{' in ' + repr(state) if state_key else ''}; "
                         "programme markers need at least 3 clones inside and 3 outside a programme.")
    clone_prog = (pd.Series(lab.iloc[cells].astype(str).to_numpy()).groupby(clone_codes).first()).to_numpy()

    ranks = np.apply_along_axis(rankdata, 0, clone_means)
    rows = []
    for prog in sorted(pd.unique(clone_prog), key=lambda s: (len(s), s)):
        inside = clone_prog == prog
        n1, n2 = int(inside.sum()), int((~inside).sum())
        if n1 < 3 or n2 < 3:
            continue
        u = ranks[inside].sum(axis=0) - n1 * (n1 + 1) / 2
        auc = u / (n1 * n2)
        # normal approximation to the Mann-Whitney U (ties ignored)
        z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        from scipy.stats import norm

        pvals = 2 * norm.sf(np.abs(z))
        diff = clone_means[inside].mean(axis=0) - clone_means[~inside].mean(axis=0)
        tab = pd.DataFrame({"programme": prog, "gene": adata.var_names[genes], "mean_diff": diff,
                            "auc": auc, "pvalue": pvals})
        if state_key is not None:
            tab.insert(1, "state", str(state))
        tab["fdr"] = bh_fdr(tab["pvalue"].to_numpy())
        tab = tab[tab["mean_diff"] > 0].sort_values(["fdr", "mean_diff"], ascending=[True, False]).head(n_top)
        tab["rank"] = np.arange(1, len(tab) + 1)
        rows.append(tab)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["programme", "gene", "mean_diff", "auc", "pvalue", "fdr", "rank"])
    get_uns(adata)["programme_markers"] = h5ad_safe(out)
    where = f" within '{state}'" if state_key is not None else ""
    log(f"programme markers{where}: {n_clones} clones as replicates, {genes.size} genes tested.", verbose)
    return out


def landscape_genes(
    adata,
    *,
    layer: str | None = "log_norm",
    context_key: str | None = "auto",
    n_neighbors: int = 15,
    min_frac_expressed: float = 0.05,
    state_key: str | None = None,
    state: str | None = None,
    min_cells_in_state: int = 1,
    n_top: int | None = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Genes whose expression changes smoothly across the clones, with no programmes needed.

    Programme markers need distinct programmes; when clones vary along a continuum (a single programme),
    this asks instead which genes follow the landscape of clone profiles. Every profiled clone is summarised
    by the mean context-centred log expression of its cells (as in :func:`programme_markers`; with
    ``state_key``/``state``, of its cells in that state only). Clones are linked to their ``n_neighbors``
    nearest clones in profile space, and each gene is scored by Moran's I of the clone means on that graph:
    I > 0 when neighbouring clones express the gene alike, about 0 when the gene ignores the landscape. P values
    use the normal approximation under randomisation (Cliff and Ord), corrected by Benjamini-Hochberg.
    ``corr_axis1`` is the Spearman correlation of the gene with the first component of the clone profiles,
    the main axis along which clones differ, and gives the direction of a gradient.

    This describes the landscape rather than testing an independent hypothesis: the profiles are built from
    the same expression, so the genes that make clones differ are expected to score. Use it to name what
    varies across clones; test annotations that the profiles did not use with :func:`profile_association`.

    With hundreds of clones most expressed genes reach FDR < 0.05, so rank genes by Moran's I (0 to 1 in
    practice) rather than counting them. The top genes often lie at one end of the main axis; ``n_top=None``
    returns every gene that passes, so both ends can be read from ``corr_axis1``.

    Returns
    -------
    DataFrame ``[gene, morans_i, z, pvalue, fdr, corr_axis1, rank]`` of the ``n_top`` genes with the largest
    Moran's I among those with FDR < 0.05 (fewer if fewer pass; all of them with ``n_top=None``), also stored
    in ``adata.uns['threadfin']['landscape_genes']``.
    """
    import scipy.sparse as sp
    from scipy.stats import norm, rankdata
    from sklearn.neighbors import NearestNeighbors

    from ._utils import bh_fdr, require_positive_int

    require_positive_int(n_neighbors, "n_neighbors")
    prof = get_profiles(adata)
    clone_key = prof["params"]["clone_key"]
    if context_key == "auto":
        context_key = prof["params"].get("context_key")
    feats = prof["features"]
    profiled = adata.obs[clone_key].astype(str).isin(set(feats.index.astype(str))).to_numpy()
    _, _, clone_index, genes, clone_means = _clone_expression(
        adata, clone_key=clone_key, cell_mask=profiled, layer=layer, context_key=context_key,
        state_key=state_key, state=state, min_cells_in_state=min_cells_in_state,
        min_frac_expressed=min_frac_expressed, caller="landscape_genes")
    n = len(clone_index)
    if n < n_neighbors + 2:
        raise ValueError(f"Only {n} profiled clones{' with cells in ' + repr(state) if state_key else ''}; "
                         f"landscape_genes needs more than n_neighbors + 1 = {n_neighbors + 1}.")
    f = feats.loc[clone_index.astype(str)].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(f)
    _, idx = nn.kneighbors(f)
    rows = np.repeat(np.arange(n), n_neighbors)
    w = sp.csr_matrix((np.ones(rows.size), (rows, idx[:, 1:].ravel())), shape=(n, n))
    s0 = float(w.sum())
    ws = w + w.T
    s1 = 0.5 * float(ws.multiply(ws).sum())
    s2 = float(((np.asarray(w.sum(axis=1)).ravel() + np.asarray(w.sum(axis=0)).ravel()) ** 2).sum())

    z = clone_means - clone_means.mean(axis=0)
    var = (z ** 2).sum(axis=0)
    ok = var > 0
    z, var, genes = z[:, ok], var[ok], genes[ok]
    moran = (n / s0) * np.asarray((z * (w @ z)).sum(axis=0)).ravel() / var
    expected = -1.0 / (n - 1)
    b2 = n * (z ** 4).sum(axis=0) / var ** 2                     # kurtosis of each gene's clone means
    var_i = ((n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
              - b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2))
             / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2) - expected ** 2)
    zscore = (moran - expected) / np.sqrt(np.maximum(var_i, 1e-12))
    pvals = norm.sf(zscore)                                       # one-sided: smoother than random
    # the main axis of the clone profiles (their first principal component; the sign is arbitrary, so it is
    # fixed to make the most extreme clone positive), and each gene's rank correlation with it
    fc = f - f.mean(axis=0)
    u, sv, _ = np.linalg.svd(fc, full_matrices=False)
    axis1 = u[:, 0] * sv[0]
    axis1 = axis1 if axis1[np.argmax(np.abs(axis1))] >= 0 else -axis1
    ra = rankdata(axis1)
    rz = rankdata(z, axis=0)                                      # columns of z are the genes kept above
    ra, rz = ra - ra.mean(), rz - rz.mean(axis=0)
    denom = np.sqrt((ra ** 2).sum() * (rz ** 2).sum(axis=0))
    corr = np.divide((ra[:, None] * rz).sum(axis=0), denom, out=np.full(rz.shape[1], np.nan), where=denom > 0)
    tab = pd.DataFrame({"gene": adata.var_names[genes], "morans_i": moran, "z": zscore, "pvalue": pvals,
                        "fdr": bh_fdr(pvals), "corr_axis1": corr})
    out = tab[tab["fdr"] < 0.05].sort_values("morans_i", ascending=False)
    out = (out if n_top is None else out.head(n_top)).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    get_uns(adata)["landscape_genes"] = h5ad_safe(out)
    where = f" within '{state}'" if state_key is not None else ""
    # with many clones most genes reach significance; Moran's I itself says how closely a gene follows the landscape
    top = ", ".join(f"{g} ({i:.2f})" for g, i in zip(out["gene"].head(5), out["morans_i"].head(5)))
    log(f"landscape genes{where}: Moran's I over {n} clones, FDR < 0.05 for {int((tab['fdr'] < 0.05).sum())} of "
        f"{len(tab)} genes; highest: {top}.", verbose)
    return out
