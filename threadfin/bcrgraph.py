"""Clone-level BCR similarity graph and sequence-based clone definition.

Candidate pairs are only evaluated within (V-gene, J-gene) blocks — both the
biologically correct support set and the scalability fix (no all-pairs CDR3
alignment). Edge weights come from CDR3 similarity, optionally blended with
light-chain CDR3 similarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sequence import cdr3_similarity_pairs

_BLOCK_ALLPAIRS_MAX = 2000
_SENTINEL = "__threadfin_na__"


def _first_call(x):
    """Normalize a V/J call: first call before ',', gene before allele '*'."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    s = str(x).split(",")[0].split("*")[0].strip()
    return s or None


def bcr_similarity_graph(
    clone_info: pd.DataFrame,
    *,
    cdr3_sim_threshold: float = 0.85,
    same_vj: bool = True,
    include_light: bool = False,
    max_pairs_per_clone: int = 200,
):
    """Sparse clone x clone BCR similarity adjacency.

    Parameters
    ----------
    clone_info
        DataFrame indexed by clone id with columns ``v_call, j_call, cdr3``
        (heavy chain; optionally ``cdr3_light, v_call_light``).
    cdr3_sim_threshold
        Minimum edge weight; weaker pairs are dropped.
    same_vj
        Restrict candidate pairs to clones sharing the (normalized) V and J
        gene calls.
    include_light
        If True and both clones have ``cdr3_light``, edge weight is
        ``0.7 * heavy + 0.3 * light`` CDR3 similarity.
    max_pairs_per_clone
        Random-subsample cap per clone for blocks larger than 2000 clones.

    Returns
    -------
    Symmetric ``(n_clones, n_clones)`` CSR matrix with zero diagonal, in the
    row order of ``clone_info``.
    """
    from scipy import sparse

    for col in ("v_call", "j_call", "cdr3"):
        if col not in clone_info.columns:
            raise ValueError(f"clone_info needs a '{col}' column.")

    n = len(clone_info.index)
    info = clone_info.copy()
    info["_v"] = info["v_call"].map(_first_call)
    info["_j"] = info["j_call"].map(_first_call)
    valid = info["_v"].notna() & info["_j"].notna() & info["cdr3"].notna()
    info = info[valid]
    info["_pos"] = pd.Series(np.arange(n), index=clone_info.index).reindex(info.index)

    rows_all, cols_all, data_all = [], [], []
    if n >= 2 and len(info) >= 2:
        if same_vj:
            blocks = list(info.groupby(["_v", "_j"], sort=False).indices.values())
        else:
            blocks = [np.arange(len(info))]

        heavy = info["cdr3"].astype(str).to_numpy()
        gpos = info["_pos"].to_numpy()
        light = None
        if include_light and "cdr3_light" in info.columns:
            light = info["cdr3_light"].to_numpy()

        for bidx in blocks:
            bidx = np.asarray(bidx)
            nb = len(bidx)
            if nb < 2:
                continue
            if nb <= _BLOCK_ALLPAIRS_MAX:
                ai, bi = np.triu_indices(nb, k=1)
            else:
                rng = np.random.default_rng(0)
                src = np.repeat(np.arange(nb), max_pairs_per_clone)
                dst = rng.integers(0, nb, size=nb * max_pairs_per_clone)
                keep = src != dst
                lo = np.minimum(src, dst)[keep]
                hi = np.maximum(src, dst)[keep]
                _, sel = np.unique(lo.astype(np.int64) * nb + hi, return_index=True)
                ai, bi = lo[sel], hi[sel]

            bseq = heavy[bidx]
            sims = np.asarray(
                cdr3_similarity_pairs(bseq[ai], bseq[bi]), dtype=np.float64
            )
            if light is not None:
                lseq = light[bidx]
                la, lb = lseq[ai], lseq[bi]
                both = pd.notna(la) & pd.notna(lb)
                if both.any():
                    lsim = np.asarray(
                        cdr3_similarity_pairs(
                            la[both].astype(str), lb[both].astype(str)
                        ),
                        dtype=np.float64,
                    )
                    sims[both] = 0.7 * sims[both] + 0.3 * lsim

            keep = sims >= cdr3_sim_threshold
            rows_all.append(gpos[bidx][ai[keep]])
            cols_all.append(gpos[bidx][bi[keep]])
            data_all.append(sims[keep])

    if data_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all)
        w = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
        w = w.maximum(w.T).tocsr()
        w.setdiag(0)
        w.eliminate_zeros()
    else:
        w = sparse.csr_matrix((n, n))

    print(
        f"[threadfin] bcr_similarity_graph: {n} clones, {w.nnz // 2} edges "
        f"(threshold={cdr3_sim_threshold}, same_vj={same_vj}).",
        flush=True,
    )
    return w


def _row_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    key = df[cols[0]].astype("string").fillna(_SENTINEL)
    for c in cols[1:]:
        key = key + "\x1f" + df[c].astype("string").fillna(_SENTINEL)
    return key


def define_clones(
    bcr: pd.DataFrame,
    *,
    cdr3_sim_threshold: float = 0.85,
    same_vj: bool = True,
    method: str = "leiden",
    resolution: float = 0.5,
    out_col: str = "clone_id_seq",
) -> pd.Series:
    """Sequence-similarity clone (re)definition over a per-cell BCR table.

    Unique ``(v_call, j_call, cdr3[, clone_id])`` rows are clustered via the
    BCR similarity graph (connected components or Leiden), and the labels are
    mapped back to cells. Cells lacking v/cdr3 get NA.

    Parameters
    ----------
    bcr
        Per-cell table indexed by barcode with ``v_call, j_call, cdr3`` and
        optionally ``clone_id``.
    cdr3_sim_threshold, same_vj
        Passed to :func:`bcr_similarity_graph`.
    method
        ``"connected"`` (scipy connected components) or ``"leiden"``
        (requires igraph + leidenalg).
    resolution
        Leiden resolution parameter.
    out_col
        Name of the returned Series.

    Returns
    -------
    Series indexed by barcode named ``out_col``.
    """
    for col in ("v_call", "j_call", "cdr3"):
        if col not in bcr.columns:
            raise ValueError(f"bcr needs a '{col}' column.")
    if method not in ("connected", "leiden"):
        raise ValueError(f"Unknown method: {method!r}")

    out = pd.Series(pd.NA, index=bcr.index, name=out_col, dtype="object")
    sub = bcr.dropna(subset=["v_call", "j_call", "cdr3"])
    if len(sub) == 0:
        return out

    key_cols = ["v_call", "j_call", "cdr3"]
    if "clone_id" in bcr.columns:
        key_cols = key_cols + ["clone_id"]
    uniq = sub[key_cols].drop_duplicates().reset_index(drop=True)
    uniq.index = pd.Index([f"__grp{i}" for i in range(len(uniq))], name="group")

    graph = bcr_similarity_graph(
        uniq, cdr3_sim_threshold=cdr3_sim_threshold, same_vj=same_vj
    )

    if method == "connected":
        from scipy.sparse import csgraph

        _, labels = csgraph.connected_components(graph, directed=False)
    else:
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise ImportError(
                "method='leiden' requires igraph and leidenalg "
                "(pip install igraph leidenalg); use method='connected' "
                "as a dependency-free fallback."
            ) from e
        from scipy import sparse

        tri = sparse.triu(graph).tocoo()
        g = ig.Graph(
            n=graph.shape[0],
            edges=list(zip(tri.row.tolist(), tri.col.tolist())),
            edge_attrs={"weight": tri.data.tolist()},
        )
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=0,
        )
        labels = np.asarray(part.membership)

    # deterministic labels: component 0 = largest
    sizes = pd.Series(labels).value_counts()
    remap = {lab: f"{out_col}_{i}" for i, lab in enumerate(sizes.index)}
    uniq_labels = pd.Series([remap[l] for l in labels], index=uniq.index)

    lut = pd.Series(uniq_labels.to_numpy(), index=_row_key(uniq, key_cols))
    out.loc[sub.index] = _row_key(sub, key_cols).map(lut).to_numpy()
    return out
