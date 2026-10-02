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
    min_programme_size: int = 5,
    min_clones: int = 30,
    assign_remaining: bool = True,
    min_posterior: float = 0.7,
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
    min_programme_size
        Groups with fewer clones are not reported as programmes (their clones
        stay unassigned); resolutions that leave more than 10% of clones in
        such groups are not selected.
    min_clones
        Minimum number of reliable clones; with fewer, programme detection is
        refused (report the coherence test only).
    assign_remaining
        Programmes are *defined* on reliable ("core") clones only. With this
        option every other profiled clone is then *assigned* to the most
        likely programme by a Gaussian classifier whose noise term shrinks
        with clone size (see ``_assign_remaining``); assignments with
        posterior >= ``min_posterior`` go to ``<key_added>_assigned``.
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
    if eligible.size < min_clones:
        raise ValueError(
            f"Only {eligible.size} clones reach reliability {min_reliability} (need {min_clones}); "
            "too few expanded clones to define programmes. Report the coherence test only, or "
            "lower min_reliability."
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

    if auto:
        multi = scan[(scan["n_programmes"] >= 2) & (scan["frac_clones_in_small_groups"] <= 0.10)]
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
    # groups below the minimum size are not programmes: their clones stay unassigned
    sizes = np.bincount(ref)
    keep_mask = sizes[ref] >= min_programme_size
    eligible = eligible[keep_mask]
    feats = feats[keep_mask]
    confidence = confidence[keep_mask]
    ref = ref[keep_mask]
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
    if embed and eligible.size >= 15:
        import umap

        xy = umap.UMAP(n_neighbors=int(min(n_neighbors, eligible.size - 1)),
                       random_state=random_state).fit_transform(feats)
        table["x"], table["y"] = np.nan, np.nan
        table.loc[eligible, "x"], table.loc[eligible, "y"] = xy[:, 0], xy[:, 1]
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
    get_uns(adata)[PROGRAMME_KEY] = {
        "summary": summary,
        "resolution_scan": scan,
        "params": {
            "resolution": chosen, "auto": auto, "n_neighbors": int(n_neighbors),
            "min_reliability": float(min_reliability), "n_boot": int(n_boot),
            "stability_threshold": float(stability_threshold), "key_added": key_added,
            "min_programme_size": int(min_programme_size),
            "assign_remaining": bool(assign_remaining), "min_posterior": float(min_posterior),
            "random_state": int(random_state),
        },
    }
    n_stable = int((summary["stability"] >= stability_threshold).sum()) if n_boot else 0
    extra = (f"; {int(table[f'{key_added}_assigned'].notna().sum())} of {len(table)} profiled clones "
             f"assigned with posterior >= {min_posterior}" if assign_remaining else "")
    log(
        f"programmes: {summary.shape[0]} programmes over {eligible.size} core clones "
        f"(resolution {chosen}); {n_stable} with bootstrap stability >= {stability_threshold}{extra}.",
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


def programme_composition(adata, state_key: str, *, key: str | None = None) -> pd.DataFrame:
    """Fraction of each programme's cells in each cell state (rows sum to 1)."""
    key = key or get_programmes(adata)["params"]["key_added"]
    df = adata.obs[[key, state_key]].dropna()
    tab = pd.crosstab(df[key], df[state_key])
    return tab.div(tab.sum(axis=1), axis=0)


def programme_markers(
    adata,
    *,
    programme_key: str = "clone_programme",
    layer: str | None = "log_norm",
    context_key: str | None = "auto",
    min_frac_expressed: float = 0.05,
    n_top: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    """Genes that distinguish programmes, tested with clones as replicates.

    Every clone is summarised by the mean context-centred log expression of
    its cells; for each programme, clones inside are compared with clones in
    other programmes by a Mann-Whitney test per gene (BH FDR). Testing across
    clones avoids the pseudoreplication of cell-level marker tests, where one
    expanded clone contributes hundreds of correlated cells.

    Returns
    -------
    Tidy DataFrame ``[programme, gene, mean_diff, auc, pvalue, fdr, rank]``
    with the top ``n_top`` up-regulated genes per programme.
    """
    import scipy.sparse as sp
    from scipy.stats import rankdata

    from ._utils import bh_fdr, codes

    prof = get_profiles(adata)
    clone_key = prof["params"]["clone_key"]
    if context_key == "auto":
        context_key = prof["params"].get("context_key")
    x = adata.layers[layer] if layer is not None else adata.X
    x = sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr()
    lab = adata.obs[programme_key]
    cells = np.flatnonzero(lab.notna().to_numpy() & adata.obs[clone_key].notna().to_numpy())
    clone_codes, clone_index = codes(adata.obs[clone_key].iloc[cells].astype(str))
    n_clones = len(clone_index)
    member = sp.csr_matrix((np.ones(cells.size), (clone_codes, np.arange(cells.size))), shape=(n_clones, cells.size))
    member = sp.diags(1.0 / np.asarray(member.sum(axis=1)).ravel()) @ member
    expressed = np.asarray((x[cells] > 0).mean(axis=0)).ravel() >= min_frac_expressed
    genes = np.flatnonzero(expressed)
    clone_means = np.asarray((member @ x[cells][:, genes]).toarray())
    if context_key is not None:
        # subtract, for every clone, the mean expression of the contexts its cells came from
        ctx, ctx_index = codes(adata.obs[context_key])
        ok = np.flatnonzero(ctx >= 0)
        cmember = sp.csr_matrix((np.ones(ok.size), (ctx[ok], ok)), shape=(len(ctx_index), adata.n_obs))
        cmember = sp.diags(1.0 / np.maximum(np.asarray(cmember.sum(axis=1)).ravel(), 1)) @ cmember
        ctx_means = np.asarray((cmember @ x[:, genes]).toarray())
        cell_ctx = sp.csr_matrix((np.ones(cells.size), (np.arange(cells.size), np.maximum(ctx[cells], 0))),
                                 shape=(cells.size, len(ctx_index)))
        clone_means -= np.asarray((member @ cell_ctx).toarray()) @ ctx_means
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
        tab["fdr"] = bh_fdr(tab["pvalue"].to_numpy())
        tab = tab[tab["mean_diff"] > 0].sort_values(["fdr", "mean_diff"], ascending=[True, False]).head(n_top)
        tab["rank"] = np.arange(1, len(tab) + 1)
        rows.append(tab)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["programme", "gene", "mean_diff", "auc", "pvalue", "fdr", "rank"])
    get_uns(adata)["programme_markers"] = out
    log(f"programme markers: {n_clones} clones as replicates, {genes.size} genes tested.", verbose)
    return out
