"""Core Threadfin algorithm: transcriptional-state-aware clonotype reclustering.

Idea
----
Single-cell BCR sequencing defines *clonotypes* from rearranged V(D)J
sequences, while scRNA-seq defines *transcriptional states* from gene
expression. Threadfin links the two: it places every clonotype at the
centroid of its member cells in a transcriptional embedding and clusters
clonotypes in that space. The resulting ``clone_cluster`` labels group
clonotypes whose cells share a transcriptional state (e.g. germinal-center
light-zone vs dark-zone B cells, plasmablasts, memory), turning the
repertoire from a list of sequences into a map of clone behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import get_basis, require_positive_int
from .reclustering import _resolve_parameters

_CLONE_INFO_KEY = "threadfin_clones"
_CLONE_MAP_KEY = "threadfin"


def _get_matrix(adata, basis: str) -> np.ndarray:
    return get_basis(adata, basis)


def clone_centroids(
    adata,
    clone_key: str = "clone_id",
    basis: str = "X_umap",
    min_clone_size: int = 1,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute the centroid of every clonotype in a transcriptional embedding.

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[clone_key]`` and ``adata.obsm[basis]``.
    clone_key
        ``obs`` column holding the clonotype id. Cells with a missing value
        (no BCR) are ignored.
    basis
        ``obsm`` key of the embedding to average, or ``"X"`` for the full
        expression matrix.
    min_clone_size
        Clonotypes with fewer cells are dropped from the centroid table.
    weight_col
        Optional ``obs`` column of per-cell weights; centroids become
        weighted means (NaN weights count as 0).

    Returns
    -------
    DataFrame indexed by clone id with one column per embedding dimension and
    an ``n_cells`` column.
    """
    if clone_key not in adata.obs.columns:
        raise KeyError(
            f"'{clone_key}' not found in adata.obs. Attach BCR annotations "
            "first (threadfin.attach_bcr) or provide your own clone column."
        )

    require_positive_int(min_clone_size, "min_clone_size")
    coords = _get_matrix(adata, basis)
    clones = adata.obs[clone_key]
    placeholders = clones.astype("string").str.strip().str.lower().isin(["", "nan", "none", "<na>"])
    valid = clones.notna() & ~placeholders
    if valid.sum() == 0:
        raise ValueError(f"All values in adata.obs['{clone_key}'] are missing.")

    df = pd.DataFrame(coords[valid.to_numpy()], index=None)
    df["clone"] = clones[valid].to_numpy()

    sizes = df.groupby("clone").size().rename("n_cells")
    if weight_col is not None:
        if weight_col not in adata.obs.columns:
            raise KeyError(f"'{weight_col}' not found in adata.obs.")
        w = np.nan_to_num(
            pd.to_numeric(adata.obs[weight_col], errors="coerce").to_numpy(
                dtype=float
            )[valid.to_numpy()],
            nan=0.0,
        )
        coord_cols = [c for c in df.columns if c != "clone"]
        weighted = df[coord_cols].multiply(w, axis=0)
        weighted["clone"] = df["clone"]
        denom = pd.Series(w).groupby(df["clone"].to_numpy()).sum()
        centroids = weighted.groupby("clone").sum().div(
            denom.replace(0, np.nan), axis=0
        )
    else:
        centroids = df.groupby("clone").mean()
    centroids.columns = [f"{basis}_{i}" for i in range(centroids.shape[1])]
    centroids = centroids.join(sizes)
    centroids = centroids[centroids["n_cells"] >= min_clone_size]
    if len(centroids) == 0:
        raise ValueError(
            f"No clones with >= {min_clone_size} cells. Lower min_clone_size."
        )
    return centroids


def _leiden_on_distances(
    dist: np.ndarray,
    n_neighbors: int,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    """Leiden-cluster a precomputed distance matrix.

    Builds a kNN graph with self-scaled Gaussian edge weights (the standard
    UMAP/scanpy kernel) and partitions it with leidenalg directly, so the
    result does not depend on scanpy's (changing) neighbors API.
    """
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors

    n = dist.shape[0]
    k = int(min(n_neighbors, n - 1))
    if k < 1:
        raise ValueError("Need at least 2 clones to cluster.")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(dist)
    d_knn, idx = nn.kneighbors(dist)
    d_knn, idx = d_knn[:, 1:], idx[:, 1:]  # drop self

    # self-scaled Gaussian kernel (sigma = distance to k-th neighbor)
    sigma = np.maximum(d_knn[:, -1:], 1e-10)
    w = np.exp(-(d_knn**2) / (sigma**2))

    edges = [(i, j) for i in range(n) for j in idx[i]]
    weights = [float(x) for x in w.ravel()]
    g = ig.Graph(n=n, edges=edges, directed=True, edge_attrs={"weight": weights})
    g_und = g.to_undirected(combine_edges="max")  # igraph>=1.0 does this in place
    if g_und is not None:
        g = g_und

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )
    return np.array([str(m) for m in part.membership])


def clonotype_recluster(
    adata,
    clone_key: str = "clone_id",
    basis: str = "X_umap",
    min_clone_size: int = 3,
    n_neighbors: int | None = None,
    resolution: float | None = None,
    cdr3_weight: float = 0.0,
    distances: np.ndarray | None = None,
    embed_clones: bool = True,
    random_state: int = 0,
    key_added: str = "clone_cluster",
    copy: bool = False,
    *,
    preset: str = "cohesive",
    min_dist: float | None = None,
    spread: float | None = None,
    learning_rate: float | None = None,
    umap_n_neighbors: int | None = None,
    embedding_mode: str = "precomputed",
    cluster_on: str = "distances",
):
    """Cluster clonotypes by the transcriptional state of their member cells.

    Steps: (1) place each clonotype at the centroid of its cells in
    ``basis``; (2) optionally blend in heavy-chain CDR3 sequence distance
    (``cdr3_weight``); (3) build a kNN graph over clonotypes and Leiden-
    cluster it; (4) map the resulting ``clone_cluster`` label back to cells.

    Parameters
    ----------
    adata
        AnnData with ``obs[clone_key]`` and ``obsm[basis]``.
    clone_key
        ``obs`` column holding the clonotype id.
    basis
        Embedding used for centroids (``"X_umap"`` uses cell-map coordinates;
        ``"X_pca"`` avoids averaging a nonlinear display). This alone does
        not reproduce the historical notebook pipeline. ``"joint"``
        uses the coupled GEX+BCR embedding in
        ``adata.uns['threadfin']['joint']``, computing it with default
        parameters via :func:`threadfin.integrate.joint_embedding` if absent.
    min_clone_size
        Clonotypes with fewer cells are excluded from clustering (their cells
        receive ``NaN``). Singletons are mostly uninformative.
    n_neighbors
        k for the clonotype kNN graph and, unless overridden, UMAP.
        ``None`` uses the preset. Counts are capped at the number of clones
        minus one; requested and effective settings are saved.
    resolution
        Leiden resolution (higher = more clone clusters); ``None`` uses the preset.
    cdr3_weight
        In ``[0, 1]``. If > 0, the transcriptional distance between clones is
        blended with their consensus-CDR3 Hamming distance:
        ``D = (1 - w) * z(D_gex) + w * z(D_cdr3)``. Requires clone-level CDR3
        information stored by :func:`threadfin.attach_bcr`. Mutually exclusive
        with ``distances``.
    distances
        Optional precomputed clone-level distances (condensed pdist vector or
        square matrix, ordered like the centroid table), used instead of
        distances computed on ``basis``.
    embed_clones
        If ``True``, compute a UMAP of the clonotype centroids (stored in the
        clone map as ``x``/``y``) for visualization.
    random_state
        Seed for Leiden/UMAP reproducibility.
    key_added
        Name of the ``obs`` column receiving the clone cluster label.
    copy
        Return a copy of ``adata`` instead of modifying in place.
    preset
        ``"cohesive"`` (legacy defaults), ``"continuous"`` (report-style
        starting point) or ``"discrete"`` (finer local partitions).
        These are exploratory settings, not learned biological categories.
    min_dist, spread, learning_rate
        UMAP controls; explicit values override the preset. They change only
        the display when ``cluster_on="distances"`` (the default).
    umap_n_neighbors
        Optional UMAP neighbour count independent of the clustering graph.
    embedding_mode
        ``"precomputed"`` embeds clone distances directly (default).
        ``"distance_profiles"`` treats each distance-matrix row as Euclidean
        features, as in the old notebook.
    cluster_on
        ``"distances"`` builds the existing graph on original clone distances.
        ``"embedding"`` builds a Scanpy neighbour graph on the resulting
        two-dimensional clone UMAP, then applies Leiden, as in the historical
        notebook. In this explicit mode UMAP parameters can change clusters;
        ``embed_clones=True`` is required. Use ``n_neighbors=15`` and
        ``umap_n_neighbors=20`` for the notebook's separate neighbour counts.

    Returns
    -------
    ``adata`` with ``obs[key_added]`` (categorical) and a clone-level table in
    ``adata.uns['threadfin']['clone_map']`` (one row per clonotype: size,
    centroid, clone-cluster label, and embedding coordinates).
    """
    settings = _resolve_parameters(
        preset, n_neighbors=n_neighbors, resolution=resolution, min_dist=min_dist,
        spread=spread, learning_rate=learning_rate,
    )
    n_neighbors, resolution = settings["n_neighbors"], settings["resolution"]
    require_positive_int(min_clone_size, "min_clone_size")
    if umap_n_neighbors is None:
        umap_n_neighbors = n_neighbors
    require_positive_int(umap_n_neighbors, "umap_n_neighbors")
    if umap_n_neighbors < 2:
        raise ValueError("umap_n_neighbors must be at least 2.")
    if embedding_mode not in {"precomputed", "distance_profiles"}:
        raise ValueError("embedding_mode must be 'precomputed' or 'distance_profiles'.")
    if cluster_on not in {"distances", "embedding"}:
        raise ValueError("cluster_on must be 'distances' or 'embedding'.")
    if cluster_on == "embedding" and not embed_clones:
        raise ValueError("cluster_on='embedding' requires embed_clones=True.")
    if not np.isfinite(cdr3_weight) or not 0 <= cdr3_weight <= 1:
        raise ValueError("cdr3_weight must be in [0, 1].")
    if copy:
        adata = adata.copy()

    if basis == "joint":
        joint = adata.uns.get(_CLONE_MAP_KEY, {}).get("joint")
        if joint is None:
            from .integrate import joint_embedding

            joint = joint_embedding(adata, clone_key=clone_key)
        centroids = joint[joint["n_cells"] >= min_clone_size].copy()
    else:
        centroids = clone_centroids(adata, clone_key=clone_key, basis=basis,
                                    min_clone_size=min_clone_size)
    n_clones = centroids.shape[0]
    if n_clones < 5:
        raise ValueError(
            f"Only {n_clones} clones pass min_clone_size={min_clone_size}; "
            "too few to cluster."
        )

    from scipy.spatial.distance import pdist, squareform

    coord_cols = [c for c in centroids.columns if c != "n_cells"]
    d_gex = squareform(pdist(centroids[coord_cols].to_numpy(), metric="euclidean"))

    if distances is not None:
        if cdr3_weight > 0:
            raise ValueError("distances and cdr3_weight are mutually exclusive.")
        d = np.asarray(distances, dtype=float)
        dist = squareform(d) if d.ndim == 1 else d
        if dist.shape != (n_clones, n_clones):
            raise ValueError(
                f"distances has shape {dist.shape}; expected ({n_clones}, "
                f"{n_clones}) (or a condensed vector) matching the "
                f"{n_clones} centroid clones."
            )
    elif cdr3_weight > 0:
        if not 0 <= cdr3_weight <= 1:
            raise ValueError("cdr3_weight must be in [0, 1].")
        clone_info = adata.uns.get(_CLONE_INFO_KEY)
        if clone_info is None or "cdr3" not in clone_info.columns:
            raise ValueError(
                "cdr3_weight > 0 requires clone-level CDR3 info. Run "
                "threadfin.attach_bcr() first so adata.uns['threadfin_clones'] "
                "exists."
            )
        from .sequence import blend_distances, cdr3_distance_matrix

        cdr3 = clone_info["cdr3"].reindex(centroids.index)
        d_seq = cdr3_distance_matrix(cdr3.tolist())
        dist = blend_distances(d_gex, d_seq, weight=cdr3_weight)
    else:
        dist = d_gex

    if (not np.isfinite(dist).all() or (dist < 0).any()
            or not np.allclose(dist, dist.T) or not np.allclose(np.diag(dist), 0)):
        raise ValueError("Clone distances must be finite, non-negative, symmetric, with a zero diagonal.")

    graph_k = int(min(n_neighbors, n_clones - 1))
    umap_k = int(min(umap_n_neighbors, n_clones - 1))

    clone_map = centroids.copy()
    clone_map.index.name = clone_key

    if embed_clones:
        import umap

        reducer = umap.UMAP(
            n_neighbors=umap_k,
            metric="precomputed" if embedding_mode == "precomputed" else "euclidean",
            min_dist=settings["min_dist"], spread=settings["spread"],
            learning_rate=settings["learning_rate"], random_state=random_state,
        )
        clone_xy = reducer.fit_transform(dist)
        clone_map["x"] = clone_xy[:, 0]
        clone_map["y"] = clone_xy[:, 1]

    if cluster_on == "embedding":
        import scanpy as sc
        from anndata import AnnData

        clone_adata = AnnData(clone_xy)
        sc.pp.neighbors(clone_adata, n_neighbors=graph_k, use_rep="X",
                        metric="euclidean", random_state=random_state)
        sc.tl.leiden(clone_adata, resolution=resolution, random_state=random_state,
                     flavor="leidenalg", directed=True, n_iterations=-1)
        labels = clone_adata.obs["leiden"].astype(str).to_numpy()
    else:
        labels = _leiden_on_distances(
            dist, n_neighbors=graph_k, resolution=resolution, random_state=random_state
        )
    clone_map[key_added] = pd.Categorical(labels)

    # map back to cells (vectorized)
    mapping = clone_map[key_added]
    adata.obs[key_added] = adata.obs[clone_key].map(mapping).astype("category")

    adata.uns.setdefault(_CLONE_MAP_KEY, {})["clone_map"] = clone_map
    adata.uns[_CLONE_MAP_KEY]["clone_distances"] = dist
    adata.uns[_CLONE_MAP_KEY]["clonotype_recluster"] = {
        "basis": basis,
        "min_clone_size": min_clone_size,
        "n_neighbors": n_neighbors,
        "effective_n_neighbors": graph_k,
        "resolution": resolution,
        "preset": preset,
        "min_dist": settings["min_dist"],
        "spread": settings["spread"],
        "learning_rate": settings["learning_rate"],
        "umap_n_neighbors": umap_n_neighbors,
        "effective_umap_n_neighbors": umap_k,
        "embedding_mode": embedding_mode,
        "cluster_on": cluster_on,
        "graph_method": "scanpy_umap_connectivities" if cluster_on == "embedding" else "self_scaled_gaussian",
        "embed_clones": bool(embed_clones),
        "cdr3_weight": cdr3_weight,
        "random_state": random_state,
        "n_clones_clustered": int(n_clones),
        "n_clone_clusters": int(len(set(labels))),
    }
    print(
        f"[threadfin] {n_clones} clones -> {len(set(labels))} clone clusters "
        f"({key_added!r} added to obs, clone map in uns['threadfin'])."
    )
    return adata


def clonal_pseudotime(
    adata,
    clone_key: str = "clone_id",
    root: str | None = None,
    key_added: str = "clonal_pseudotime",
    random_state: int = 0,
    use_clone_graph: bool = False,
):
    """Diffusion pseudotime over the clonotype graph.

    Runs scanpy's diffusion pseudotime (Haghverdi et al., 2016) on the
    clonotype kNN graph built by :func:`clonotype_recluster`, then maps the
    value back to cells. Requires a prior :func:`clonotype_recluster` call.

    Parameters
    ----------
    adata
        AnnData after :func:`clonotype_recluster`.
    clone_key
        ``obs`` column holding the clonotype id.
    root
        Root clonotype id. If ``None``, the clone with the minimal first
        centroid coordinate is used (deterministic default; pass an explicit
        clone id of a known naive/early state for interpretable direction).
    key_added
        Name of the ``obs`` column receiving the pseudotime.
    random_state
        Seed.
    use_clone_graph
        If ``True``, run DPT on the kNN graph implied by the clone distances
        stored by :func:`clonotype_recluster` (adaptive-Gaussian kernel over
        the stored distance matrix) instead of recomputing neighbors on the
        centroid coordinates. Falls back to the default behavior with a
        warning when no stored distances are available.

    Returns
    -------
    ``adata`` with ``obs[key_added]``; clone-level values are added to the
    clone map in ``adata.uns['threadfin']['clone_map']``.
    """
    import scanpy as sc
    from anndata import AnnData

    clone_map = adata.uns.get(_CLONE_MAP_KEY, {}).get("clone_map")
    if clone_map is None:
        raise ValueError("Run threadfin.clonotype_recluster() first.")

    coord_cols = [c for c in clone_map.columns if c not in ("n_cells",) and clone_map[c].dtype != "category" and c not in ("x", "y")]
    coords = clone_map[coord_cols].to_numpy()

    ad = AnnData(coords)
    n_neighbors = int(min(15, ad.n_obs - 1))
    stored = adata.uns.get(_CLONE_MAP_KEY, {}).get("clone_distances")
    if use_clone_graph and stored is not None and stored.shape == (ad.n_obs, ad.n_obs):
        import scipy.sparse as sp
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1, metric="precomputed"
        ).fit(stored)
        d_knn, idx = nn.kneighbors(stored)
        d_knn, idx = d_knn[:, 1:], idx[:, 1:]  # drop self
        sigma = np.maximum(d_knn[:, -1:], 1e-10)
        w = np.exp(-(d_knn**2) / (sigma**2))
        rows = np.repeat(np.arange(ad.n_obs), n_neighbors)
        conn = sp.csr_matrix(
            (w.ravel(), (rows, idx.ravel())), shape=(ad.n_obs, ad.n_obs)
        )
        conn = conn.maximum(conn.T)
        dmat = sp.csr_matrix(
            (d_knn.ravel(), (rows, idx.ravel())), shape=(ad.n_obs, ad.n_obs)
        )
        ad.uns["neighbors"] = {
            "connectivities": conn,
            "distances": dmat.maximum(dmat.T),
            "params": {"n_neighbors": n_neighbors, "method": "umap",
                       "metric": "precomputed"},
        }
    else:
        if use_clone_graph:
            warnings.warn(
                "use_clone_graph=True but no stored clone distances found; "
                "falling back to centroid-coordinate neighbors.",
                stacklevel=2,
            )
        sc.pp.neighbors(ad, n_neighbors=n_neighbors)
    if root is None:
        ad.uns["iroot"] = int(np.argmin(coords[:, 0]))
    else:
        if root not in clone_map.index:
            raise KeyError(f"root clone {root!r} not found in clone map.")
        ad.uns["iroot"] = int(clone_map.index.get_loc(root))
    sc.tl.dpt(ad)

    pt = ad.obs["dpt_pseudotime"].replace([np.inf], np.nan)
    clone_map = clone_map.copy()
    clone_map[key_added] = pt.to_numpy()

    adata.obs[key_added] = pd.to_numeric(
        adata.obs[clone_key].map(clone_map[key_added]), errors="coerce"
    )
    adata.uns[_CLONE_MAP_KEY]["clone_map"] = clone_map
    print(f"[threadfin] clonal pseudotime added to obs[{key_added!r}].")
    return adata


def bcr_reclustering(adata, bcr_table=None, **kwargs):
    """Backward-compatible alias for the original Threadfin API.

    .. deprecated:: 1.0.0
        Use :func:`threadfin.attach_bcr` + :func:`threadfin.clonotype_recluster`.
    """
    warnings.warn(
        "bcr_reclustering() is deprecated; use threadfin.attach_bcr() followed "
        "by threadfin.clonotype_recluster().",
        DeprecationWarning,
        stacklevel=2,
    )
    from .io import attach_bcr, build_clone_key

    if bcr_table is not None and "clone_id" not in adata.obs.columns:
        if "clone_id" not in bcr_table.columns:
            bcr_table = build_clone_key(bcr_table)
        attach_bcr(adata, bcr_table)
    return clonotype_recluster(adata, **kwargs)
