"""Clonal programmes: groups of clones with similar state profiles.

A *clonal programme* is a set of genetically distinct clones whose cells,
relative to their sampling context, occupy similar transcriptional states
(for example clones biased towards plasmablast output, or clones that stay
in the germinal centre). Programmes are found by Leiden community detection
on a k-nearest-neighbour graph of clone profiles
(:func:`threadfin.tl.clone_profiles`).

Because community detection always returns *some* partition, every
programme is reported with a bootstrap stability score: cells are resampled
within clones (Poisson bootstrap), profiles are re-estimated, the graph is
rebuilt and re-partitioned, and each original programme is matched to its
best bootstrap counterpart by Jaccard overlap (Hennig 2007, "clusterboot").
Following Hennig, programmes with mean Jaccard >= 0.75 are considered stable
and those below 0.6 should not be interpreted. The Leiden resolution is
chosen as the finest one whose size-weighted stability reaches the
threshold, instead of being fixed by hand.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import get_uns, log, stable_categorical
from .profiles import get_profiles, model_from_adata, project_features

PROGRAMME_KEY = "programmes"
DEFAULT_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0)


# --------------------------------------------------------------------------- graph + Leiden


def knn_graph(features: np.ndarray, n_neighbors: int = 15):
    """Weighted undirected kNN graph (igraph) with self-tuned Gaussian weights."""
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
    g = ig.Graph(n=n, edges=list(zip(lo[sel].tolist(), hi[sel].tolist())), directed=False)
    g.es["weight"] = w.ravel()[sel].tolist()
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


def find_programmes(
    adata,
    *,
    resolution: float | str = "auto",
    resolutions=DEFAULT_RESOLUTIONS,
    n_neighbors: int = 15,
    min_reliability: float = 0.5,
    n_boot: int = 30,
    stability_threshold: float = 0.75,
    embed: bool = True,
    random_state: int = 0,
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
    min_reliability
        Only clones whose profile reliability reaches this value are
        clustered (see :class:`threadfin.profiles.VarianceComponents`);
        other clones get no programme.
    n_boot
        Bootstrap replicates for stability (0 disables stability, which is
        only allowed with a fixed ``resolution``).
    embed
        Also compute a 2-D UMAP of clone profiles (columns ``x``/``y``) for
        the clone map.
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
    if eligible.size < 10:
        raise ValueError(
            f"Only {eligible.size} clones reach reliability {min_reliability}; lower "
            "min_reliability or use larger clones."
        )
    feats = feats_all.loc[eligible].to_numpy()
    graph = knn_graph(feats, n_neighbors)

    auto = isinstance(resolution, str)
    if auto and resolution != "auto":
        raise ValueError("resolution must be a number or 'auto'.")
    if auto and n_boot < 5:
        raise ValueError("resolution='auto' needs n_boot >= 5 bootstrap replicates.")
    res_grid = list(resolutions) if auto else [float(resolution)]

    rng = np.random.default_rng(random_state)
    boot_graphs = []
    if n_boot > 0:
        model = model_from_adata(adata)
        code_of = {str(c): i for i, c in enumerate(model.clone_index.astype(str))}
        eligible_codes = np.array([code_of[c] for c in eligible], dtype=np.int64)
        boot_feats = _bootstrap_features(model, adata, eligible_codes, n_boot, rng)
        boot_graphs = [knn_graph(f, n_neighbors) for f in boot_feats]

    scan_rows, partitions = [], {}
    for res in res_grid:
        ref = leiden(graph, res, seed=random_state)
        k = int(ref.max()) + 1
        jac_sum = np.zeros(k)
        agree = np.zeros(ref.size)
        for b, bg in enumerate(boot_graphs):
            jac, ok = jaccard_match(ref, leiden(bg, res, seed=random_state + b + 1))
            jac_sum += jac
            agree += ok
        nb = max(len(boot_graphs), 1)
        jac_mean = jac_sum / nb if boot_graphs else np.full(k, np.nan)
        sizes = np.bincount(ref, minlength=k)
        weighted = float(np.sum(jac_mean * sizes) / sizes.sum()) if boot_graphs else np.nan
        partitions[res] = (ref, jac_mean, agree / nb if boot_graphs else np.full(ref.size, np.nan))
        scan_rows.append({
            "resolution": res, "n_programmes": k, "stability_weighted": weighted,
            "stability_min": float(np.min(jac_mean)) if boot_graphs else np.nan,
        })
    scan = pd.DataFrame(scan_rows)

    if auto:
        multi = scan[scan["n_programmes"] >= 2]
        stable = multi[multi["stability_weighted"] >= stability_threshold]
        if not stable.empty:
            best = stable.sort_values(["n_programmes", "stability_weighted"], ascending=False).iloc[0]
        elif not multi.empty:
            best = multi.sort_values("stability_weighted", ascending=False).iloc[0]
            warnings.warn(
                f"No resolution reached stability {stability_threshold}; using the most stable "
                f"partition (stability {best['stability_weighted']:.2f}). Interpret programmes "
                "cautiously.", stacklevel=2,
            )
        else:
            best = scan.iloc[0]
            warnings.warn("No resolution split the clones into >= 2 programmes.", stacklevel=2)
        chosen = float(best["resolution"])
    else:
        chosen = res_grid[0]

    ref, jac_mean, confidence = partitions[chosen]
    labels = stable_categorical(ref, prefix="P")
    raw_to_label = dict(zip(ref, labels))

    # write clone-level results
    table = table.copy()
    table[key_added] = pd.Categorical([None] * len(table), categories=labels.categories)
    table.loc[eligible, key_added] = np.asarray(labels)
    table["confidence"] = np.nan
    table.loc[eligible, "confidence"] = confidence
    if embed and eligible.size >= 15:
        import umap

        xy = umap.UMAP(n_neighbors=int(min(n_neighbors, eligible.size - 1)),
                       random_state=random_state).fit_transform(feats)
        table["x"], table["y"] = np.nan, np.nan
        table.loc[eligible, "x"], table.loc[eligible, "y"] = xy[:, 0], xy[:, 1]
    prof["clone_table"] = table

    # map to cells
    clone_key = prof["params"]["clone_key"]
    mapping = table[key_added].dropna().astype(str)
    adata.obs[key_added] = pd.Categorical(
        adata.obs[clone_key].astype("object").map(lambda c: mapping.get(str(c)) if pd.notna(c) else None),
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
    get_uns(adata)[PROGRAMME_KEY] = {
        "summary": summary,
        "resolution_scan": scan,
        "params": {
            "resolution": chosen, "auto": auto, "n_neighbors": int(n_neighbors),
            "min_reliability": float(min_reliability), "n_boot": int(n_boot),
            "stability_threshold": float(stability_threshold), "key_added": key_added,
            "random_state": int(random_state),
        },
    }
    n_stable = int((summary["stability"] >= stability_threshold).sum()) if n_boot else 0
    log(
        f"programmes: {summary.shape[0]} programmes over {eligible.size} clones "
        f"(resolution {chosen}); {n_stable} with bootstrap stability >= {stability_threshold}.",
        verbose,
    )
    return summary


def get_programmes(adata) -> dict:
    """Stored programme results (raises if :func:`find_programmes` was not run)."""
    out = get_uns(adata).get(PROGRAMME_KEY)
    if out is None:
        raise ValueError("Run threadfin.tl.find_programmes() (or threadfin.run()) first.")
    return out


def programme_composition(adata, state_key: str, *, key: str | None = None) -> pd.DataFrame:
    """Fraction of each programme's cells in each cell state (rows sum to 1)."""
    key = key or get_programmes(adata)["params"]["key_added"]
    df = adata.obs[[key, state_key]].dropna()
    tab = pd.crosstab(df[key], df[state_key])
    return tab.div(tab.sum(axis=1), axis=0)
