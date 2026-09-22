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

_CLONE_INFO_KEY = "threadfin_clones"
_CLONE_MAP_KEY = "threadfin"


def _get_matrix(adata, basis: str) -> np.ndarray:
    if basis == "X":
        x = adata.X
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    if basis not in adata.obsm:
        raise KeyError(
            f"'{basis}' not found in adata.obsm. Run an embedding first "
            "(e.g. sc.tl.umap) or pass basis='X' / another key."
        )
    return np.asarray(adata.obsm[basis])


def clone_centroids(
    adata,
    clone_key: str = "clone_id",
    basis: str = "X_umap",
    min_clone_size: int = 1,
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

    coords = _get_matrix(adata, basis)
    clones = adata.obs[clone_key]
    valid = clones.notna()
    if valid.sum() == 0:
        raise ValueError(f"All values in adata.obs['{clone_key}'] are missing.")

    df = pd.DataFrame(coords[valid.to_numpy()], index=None)
    df["clone"] = clones[valid].to_numpy()

    sizes = df.groupby("clone").size().rename("n_cells")
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
    n_neighbors: int = 20,
    resolution: float = 0.3,
    cdr3_weight: float = 0.0,
    embed_clones: bool = True,
    random_state: int = 0,
    key_added: str = "clone_cluster",
    copy: bool = False,
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
        Embedding used for centroids (``"X_umap"`` reproduces the original
        manuscript; ``"X_pca"`` is more robust on noisy data).
    min_clone_size
        Clonotypes with fewer cells are excluded from clustering (their cells
        receive ``NaN``). Singletons are mostly uninformative.
    n_neighbors
        k for the clonotype kNN graph.
    resolution
        Leiden resolution (higher = more clone clusters).
    cdr3_weight
        In ``[0, 1]``. If > 0, the transcriptional distance between clones is
        blended with their consensus-CDR3 Hamming distance:
        ``D = (1 - w) * z(D_gex) + w * z(D_cdr3)``. Requires clone-level CDR3
        information stored by :func:`threadfin.attach_bcr`.
    embed_clones
        If ``True``, compute a UMAP of the clonotype centroids (stored in the
        clone map as ``x``/``y``) for visualization.
    random_state
        Seed for Leiden/UMAP reproducibility.
    key_added
        Name of the ``obs`` column receiving the clone cluster label.
    copy
        Return a copy of ``adata`` instead of modifying in place.

    Returns
    -------
    ``adata`` with ``obs[key_added]`` (categorical) and a clone-level table in
    ``adata.uns['threadfin']['clone_map']`` (one row per clonotype: size,
    centroid, clone-cluster label, and embedding coordinates).
    """
    if copy:
        adata = adata.copy()

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

    if cdr3_weight > 0:
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

    labels = _leiden_on_distances(
        dist, n_neighbors=n_neighbors, resolution=resolution, random_state=random_state
    )

    clone_map = centroids.copy()
    clone_map[key_added] = pd.Categorical(labels)
    clone_map.index.name = clone_key

    if embed_clones:
        import umap

        k = int(min(n_neighbors, n_clones - 1))
        reducer = umap.UMAP(
            n_neighbors=k, metric="precomputed", random_state=random_state
        )
        clone_xy = reducer.fit_transform(dist)
        clone_map["x"] = clone_xy[:, 0]
        clone_map["y"] = clone_xy[:, 1]

    # map back to cells (vectorized)
    mapping = clone_map[key_added]
    adata.obs[key_added] = adata.obs[clone_key].map(mapping).astype("category")

    adata.uns.setdefault(_CLONE_MAP_KEY, {})["clone_map"] = clone_map
    adata.uns[_CLONE_MAP_KEY]["clonotype_recluster"] = {
        "basis": basis,
        "min_clone_size": min_clone_size,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
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
