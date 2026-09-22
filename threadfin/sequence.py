"""CDR3 sequence-distance utilities (optional sequence-aware integration)."""

from __future__ import annotations

import numpy as np

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype="U1")
_AA_TO_INT = {aa: i for i, aa in enumerate(_AA)}


def _encode(seq: str) -> np.ndarray:
    return np.array([_AA_TO_INT.get(a, 20) for a in seq], dtype=np.int8)


def cdr3_distance_matrix(cdr3s: list) -> np.ndarray:
    """Pairwise amino-acid Hamming distance between CDR3 sequences.

    Sequences of equal length are compared position-wise (fraction of
    mismatches). Pairs with different lengths receive the maximal normalized
    distance of 1.0 plus a length-mismatch penalty, keeping the metric in a
    comparable range for downstream blending. Missing sequences (None/NaN)
    get distance 1.0 to everything.

    Parameters
    ----------
    cdr3s
        List of CDR3 amino-acid sequences (may contain None/NaN).

    Returns
    -------
    Symmetric ``(n, n)`` float matrix with zero diagonal.
    """
    n = len(cdr3s)
    dist = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    seqs = [s if isinstance(s, str) and len(s) > 0 else None for s in cdr3s]
    lengths = np.array([len(s) if s else -1 for s in seqs])

    for length in sorted({len(s) for s in seqs if s}):
        idx = np.flatnonzero(lengths == length)
        if len(idx) < 1:
            continue
        mat = np.stack([_encode(seqs[i]) for i in idx])
        # pairwise mismatch fraction via broadcasting
        sub = (mat[:, None, :] != mat[None, :, :]).mean(axis=2)
        dist[np.ix_(idx, idx)] = sub

    # length-mismatch penalty on top of the maximal base distance
    valid = lengths >= 0
    valid_pair = valid[:, None] & valid[None, :]
    len_diff = np.abs(lengths[:, None] - lengths[None, :]).astype(np.float64)
    len_diff[~valid_pair] = 0.0
    max_len = max(lengths.max(), 1)
    dist += (len_diff / max_len) * 0.5
    dist[~valid_pair] = 1.0
    np.fill_diagonal(dist, 0.0)
    return dist


def _zscore(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    vals = mat[iu]
    mu, sd = vals.mean(), vals.std()
    if sd == 0:
        return mat - mu
    return (mat - mu) / sd


def blend_distances(
    d_transcriptional: np.ndarray, d_sequence: np.ndarray, weight: float
) -> np.ndarray:
    """Convex blend of two distance matrices after z-scoring.

    ``D = (1 - weight) * z(D_transcriptional) + weight * z(D_sequence)``.
    The blended matrix is shifted so its minimum is 0 (kNN/UMAP expect
    non-negative distances).
    """
    if d_transcriptional.shape != d_sequence.shape:
        raise ValueError("Distance matrices must have the same shape.")
    out = (1.0 - weight) * _zscore(d_transcriptional) + weight * _zscore(d_sequence)
    out = out - out.min()
    np.fill_diagonal(out, 0.0)
    return out
