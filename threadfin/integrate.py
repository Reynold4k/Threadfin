"""Coupled GEX+BCR graph integration: joint clone embedding and diagnostics.

Implements the "Coupled Laplacian" principle of docs/DESIGN_V2.md: instead of
Benisse's ADMM on dense matrices, the coupled clone graph
``W = (1 - lam) * W_gex + lam * W_bcr`` (each row-normalized) is embedded
directly with normalized-Laplacian eigenmaps, giving (scaled) commute-time
geometry at O(n*k*m) cost on sparse graphs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import _CLONE_INFO_KEY, _CLONE_MAP_KEY, clone_centroids

_VJ_COLUMNS = ("v_call", "j_call", "cdr3")


def _adaptive_gaussian_knn(coords: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    """Sparse symmetric kNN graph with the self-scaled Gaussian kernel used by
    ``core._leiden_on_distances`` (sigma = distance to the k-th neighbor)."""
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = int(min(n_neighbors, n - 1))
    if k < 1:
        raise ValueError("Need at least 2 clones to build the GEX graph.")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    d_knn, idx = nn.kneighbors(coords)
    d_knn, idx = d_knn[:, 1:], idx[:, 1:]  # drop self
    sigma = np.maximum(d_knn[:, -1:], 1e-10)
    w = np.exp(-(d_knn**2) / (sigma**2))
    rows = np.repeat(np.arange(n), k)
    graph = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return graph.maximum(graph.T)


def _row_normalize(graph: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    return sp.diags(inv) @ graph


def _largest_component(graph: sp.csr_matrix) -> np.ndarray:
    n_comp, labels = sp.csgraph.connected_components(graph, directed=False)
    if n_comp == 1:
        return np.arange(graph.shape[0])
    sizes = np.bincount(labels)
    keep = labels == sizes.argmax()
    dropped = graph.shape[0] - int(keep.sum())
    if dropped / graph.shape[0] > 0.1:
        warnings.warn(
            f"Largest connected component drops {dropped}/{graph.shape[0]} "
            f"clones ({dropped / graph.shape[0]:.0%}); consider a lower lam, "
            "more neighbors, or a denser BCR graph.",
            stacklevel=2,
        )
    return np.flatnonzero(keep)


def _spectral_embedding(
    graph: sp.csr_matrix, n_components: int, random_state: int
) -> np.ndarray:
    """Normalized-Laplacian eigenmap: bottom nontrivial eigenvectors of
    ``L = I - D^-1/2 W D^-1/2``, rows renormalized to unit norm."""
    n = graph.shape[0]
    deg = np.asarray(graph.sum(axis=1)).ravel()
    d_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    norm = sp.diags(d_inv_sqrt)
    lap = sp.identity(n) - norm @ graph @ norm

    k = int(min(n_components + 1, n - 1))
    if k < 2:
        raise ValueError("Need at least 3 connected clones for an embedding.")
    if n <= k + 2:
        # dense fallback when too few clones for ARPACK
        vals, vecs = np.linalg.eigh(lap.toarray())
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        vals, vecs = vals[:k], vecs[:, :k]
    else:
        # L is PSD, so "SA" selects the same bottom eigenpairs "SM" would,
        # but ARPACK handles smallest-algebraic far more robustly.
        v0 = np.random.default_rng(random_state).standard_normal(n)
        try:
            vals, vecs = sp.linalg.eigsh(lap, k=k, which="SA", v0=v0, tol=1e-10)
        except Exception:
            vals, vecs = np.linalg.eigh(lap.toarray())
            order = np.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        vals, vecs = vals[:k], vecs[:, :k]

    emb = vecs[:, 1:]  # drop the trivial constant eigenvector
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)


def _get_bcr_graph(adata, clone_ids, bcr_graph, cdr3_sim_threshold):
    """Return W_bcr as a CSR matrix aligned to ``clone_ids`` order."""
    clone_info = adata.uns.get(_CLONE_INFO_KEY)
    n = len(clone_ids)
    if bcr_graph is not None:
        graph = sp.csr_matrix(bcr_graph)
        if clone_info is not None and graph.shape[0] == clone_info.shape[0]:
            pos = clone_info.index.get_indexer(clone_ids)
            if (pos < 0).any():
                raise ValueError(
                    "adata.uns['threadfin_clones'] does not cover all centroid "
                    "clones; cannot align the supplied bcr_graph."
                )
            return graph[pos][:, pos]
        if graph.shape[0] != n:
            raise ValueError(
                f"bcr_graph has {graph.shape[0]} rows but there are {n} "
                "clones; supply a graph aligned with "
                "adata.uns['threadfin_clones'] or with the centroid clones."
            )
        return graph

    if clone_info is None:
        raise KeyError(
            "bcr_graph=None requires clone-level BCR info in "
            "adata.uns['threadfin_clones']. Run threadfin.attach_bcr() first, "
            "or pass a precomputed bcr_graph."
        )
    missing = [c for c in _VJ_COLUMNS if c not in clone_info.columns]
    if missing:
        raise KeyError(
            f"adata.uns['threadfin_clones'] is missing required columns "
            f"{missing} (need v_call, j_call, cdr3)."
        )
    try:
        from .bcrgraph import bcr_similarity_graph
    except ImportError as e:
        raise ImportError(
            "bcr_graph=None requires the threadfin.bcrgraph module, which is "
            "not available in this install. Pass a precomputed bcr_graph "
            "(scipy sparse matrix aligned with adata.uns['threadfin_clones'])."
        ) from e
    shared = clone_ids.intersection(clone_info.index, sort=False)
    if len(shared) == 0:
        raise ValueError(
            "No overlap between centroid clone ids and "
            "adata.uns['threadfin_clones'] index."
        )
    graph = sp.csr_matrix(
        bcr_similarity_graph(
            clone_info.loc[shared], cdr3_sim_threshold=cdr3_sim_threshold
        )
    ).tocoo()
    if len(shared) == n:
        return sp.csr_matrix(graph)
    pos = clone_ids.get_indexer(shared)
    full = sp.csr_matrix(
        (graph.data, (pos[graph.row], pos[graph.col])), shape=(n, n)
    )
    return full


def joint_embedding(
    adata,
    *,
    clone_key: str = "clone_id",
    basis: str = "X_pca",
    min_clone_size: int = 3,
    n_neighbors: int = 20,
    lam: float = 0.5,
    n_components: int = 20,
    bcr_graph=None,
    cdr3_sim_threshold: float = 0.85,
    random_state: int = 0,
) -> pd.DataFrame:
    """Coupled-Laplacian eigenmap embedding of clones.

    Builds an adaptive-Gaussian kNN graph over GEX centroids and a BCR
    sequence-similarity graph, couples them as
    ``W = (1 - lam) * W_gex + lam * W_bcr`` (each row-normalized), and embeds
    clones with the bottom nontrivial eigenvectors of the symmetric normalized
    Laplacian.

    Returns a DataFrame indexed by clone id with columns
    ``joint_0..joint_{m-1}`` and ``n_cells``; also stored in
    ``adata.uns['threadfin']['joint']`` with the coupled graph alongside.
    """
    if not 0 <= lam <= 1:
        raise ValueError("lam must be in [0, 1].")

    centroids = clone_centroids(
        adata, clone_key=clone_key, basis=basis, min_clone_size=min_clone_size
    )
    coord_cols = [c for c in centroids.columns if c != "n_cells"]
    clone_ids = centroids.index

    w_bcr = sp.csr_matrix(
        _get_bcr_graph(adata, clone_ids, bcr_graph, cdr3_sim_threshold)
    )
    w_gex = _adaptive_gaussian_knn(centroids[coord_cols].to_numpy(), n_neighbors)

    w_gex = _row_normalize(w_gex)
    w_bcr = _row_normalize(w_bcr)
    coupled = (1.0 - lam) * w_gex + lam * w_bcr
    coupled = coupled.maximum(coupled.T).tocsr()

    keep = _largest_component(coupled)
    if len(keep) < coupled.shape[0]:
        coupled = coupled[keep][:, keep]
        w_bcr = w_bcr[keep][:, keep]
        centroids = centroids.iloc[keep]

    emb = _spectral_embedding(coupled, n_components, random_state)
    joint = pd.DataFrame(
        emb,
        index=centroids.index,
        columns=[f"joint_{i}" for i in range(emb.shape[1])],
    )
    joint["n_cells"] = centroids["n_cells"]

    uns = adata.uns.setdefault(_CLONE_MAP_KEY, {})
    uns["joint"] = joint
    uns["joint_graph"] = coupled
    uns["joint_graph_bcr"] = w_bcr
    uns["joint_params"] = {
        "clone_key": clone_key,
        "basis": basis,
        "min_clone_size": min_clone_size,
        "n_neighbors": n_neighbors,
        "lam": lam,
        "n_components": int(emb.shape[1]),
        "cdr3_sim_threshold": cdr3_sim_threshold,
        "random_state": random_state,
        "n_clones": int(joint.shape[0]),
    }
    print(
        f"[threadfin] joint embedding: {joint.shape[0]} clones x "
        f"{emb.shape[1]} components (lam={lam}) stored in "
        "uns['threadfin']['joint']."
    )
    return joint


def _pairwise_rows(mat: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.linalg.norm(mat[i] - mat[j], axis=1)


def _safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def integration_diagnostics(
    adata,
    joint: pd.DataFrame | None = None,
    *,
    basis: str = "X_pca",
    clone_key: str = "clone_id",
) -> dict:
    """Benisse testCor analogues and a heuristic modality-contribution split.

    On the edges of the coupled graph: Spearman correlation between latent
    distances (joint embedding) and (i) GEX centroid distances
    (``testcor_gex``), (ii) BCR dissimilarity ``1 - w_bcr`` over BCR edges
    only (``testcor_bcr``). ``modality_contribution`` is a simple heuristic:
    per joint dimension, the max absolute correlation with the top GEX
    centroid PCs versus the absolute correlation with each clone's mean
    BCR-edge-weight profile; the two means are normalized to sum to 1.
    """
    from scipy.stats import spearmanr

    uns = adata.uns.get(_CLONE_MAP_KEY, {})
    if joint is None:
        joint = uns.get("joint")
        if joint is None:
            raise ValueError("Run threadfin.integrate.joint_embedding() first.")
    coupled = uns.get("joint_graph")
    w_bcr = uns.get("joint_graph_bcr")
    if coupled is None or coupled.shape[0] != joint.shape[0]:
        raise ValueError(
            "Coupled graph not found in uns['threadfin']; run "
            "threadfin.integrate.joint_embedding() first."
        )

    triu = sp.triu(coupled, k=1).tocoo()
    i, j = triu.row, triu.col
    n_edges = len(i)

    joint_cols = [c for c in joint.columns if c != "n_cells"]
    emb = joint[joint_cols].to_numpy()
    latent_d = _pairwise_rows(emb, i, j)

    centroids = clone_centroids(
        adata, clone_key=clone_key, basis=basis, min_clone_size=1
    ).reindex(joint.index)
    coord_cols = [c for c in centroids.columns if c != "n_cells"]
    gex = centroids[coord_cols].to_numpy()
    gex_d = _pairwise_rows(gex, i, j)
    r_gex = float(spearmanr(latent_d, gex_d).statistic)

    if w_bcr is not None:
        bcr_triu = sp.triu(w_bcr, k=1).tocsr()
        bcr_w = np.asarray(bcr_triu[i, j]).ravel()
        mask = bcr_w > 0
        if mask.sum() >= 3:
            r_bcr = float(
                spearmanr(latent_d[mask], 1.0 - bcr_w[mask]).statistic
            )
        else:
            r_bcr = np.nan
    else:
        bcr_w = np.zeros(n_edges)
        r_bcr = np.nan

    # modality contribution (heuristic; see docstring)
    n_pcs = int(min(5, gex.shape[0] - 1, gex.shape[1]))
    xc = gex - gex.mean(axis=0)
    vt = np.linalg.svd(xc, full_matrices=False)[2]
    pcs = xc @ vt[:n_pcs].T
    bcr_profile = (
        np.asarray(w_bcr.mean(axis=1)).ravel()
        if w_bcr is not None
        else np.zeros(joint.shape[0])
    )
    gex_scores, bcr_scores = [], []
    for col in range(emb.shape[1]):
        v = emb[:, col]
        gex_scores.append(max(_safe_abs_corr(v, pcs[:, p]) for p in range(n_pcs)))
        bcr_scores.append(_safe_abs_corr(v, bcr_profile))
    gex_total = float(np.mean(gex_scores))
    bcr_total = float(np.mean(bcr_scores))
    total = gex_total + bcr_total
    if total > 0:
        contribution = {"gex": gex_total / total, "bcr": bcr_total / total}
    else:
        contribution = {"gex": 0.5, "bcr": 0.5}

    diagnostics = {
        "testcor_gex": r_gex,
        "testcor_bcr": r_bcr,
        "n_edges": int(n_edges),
        "modality_contribution": contribution,
    }
    adata.uns[_CLONE_MAP_KEY]["joint_diagnostics"] = diagnostics
    return diagnostics
