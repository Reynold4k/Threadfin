"""CDR3 sequence-distance utilities (optional sequence-aware integration)."""

from __future__ import annotations

import numpy as np

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype="U1")
_AA_TO_INT = {aa: i for i, aa in enumerate(_AA)}


def _encode(seq: str) -> np.ndarray:
    return np.array([_AA_TO_INT.get(a, 20) for a in seq], dtype=np.int8)


_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYVBZX*"
_BLOSUM62_ROWS = """\
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _build_blosum62() -> dict:
    rows = [list(map(int, line.split())) for line in _BLOSUM62_ROWS.strip().splitlines()]
    return {
        (a, b): rows[i][j]
        for i, a in enumerate(_BLOSUM62_ORDER)
        for j, b in enumerate(_BLOSUM62_ORDER)
    }


BLOSUM62 = _build_blosum62()

_GAP_OPEN = -8
_GAP_EXTEND = -1
_MAX_LEN_DIFF = 4

try:  # optional C-accelerated aligner; pure-Python fallback below
    import parasail as _parasail

    _PARASAIL_MATRIX = _parasail.blosum62
except ImportError:  # pragma: no cover
    _parasail = None
    _PARASAIL_MATRIX = None

_ATCHLEY_TABLE = {
    "A": (-0.591, -1.302, -0.733, 1.570, -0.146),
    "R": (1.538, -0.055, 1.502, 0.440, 2.897),
    "N": (0.945, 0.828, 1.299, -0.169, 0.933),
    "D": (1.050, 0.302, -3.656, -0.259, -3.242),
    "C": (-1.343, 0.465, -0.862, -1.020, -0.255),
    "Q": (0.931, -0.179, -3.005, -0.503, -1.853),
    "E": (1.357, -1.453, 1.477, 0.113, -0.837),
    "G": (-0.384, 1.652, 1.330, 1.045, 2.064),
    "H": (0.336, -0.417, -1.673, -1.474, -0.078),
    "I": (-1.239, -0.547, 2.131, 0.393, 0.816),
    "L": (-1.019, -0.987, -1.505, 1.266, -0.912),
    "K": (1.831, -0.561, 0.533, -0.277, 1.648),
    "M": (-1.329, -1.244, -0.663, 0.868, -0.500),
    "F": (-1.006, -0.590, 1.891, -0.397, 0.412),
    "P": (0.189, 2.081, -1.628, 0.421, -1.392),
    "S": (-0.228, 1.399, -4.760, 0.670, -2.647),
    "T": (-0.032, 0.326, 2.213, 0.908, 1.313),
    "W": (-0.595, 0.009, 0.672, -2.128, -0.184),
    "Y": (0.260, 0.830, 3.097, -0.838, 1.512),
    "V": (-1.337, -0.279, -0.544, 1.242, -1.262),
}
ATCHLEY = {a: np.array(v, dtype=np.float64) for a, v in _ATCHLEY_TABLE.items()}

_MODES = ("auto", "hamming", "blosum")

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_nt(seq) -> str | None:
    """Translate a nucleotide junction to amino acids (frame 0, stops at the
    first stop codon). Returns None for invalid/short input or unknown codons
    before any residue. Trailing partial codons are ignored."""
    if not isinstance(seq, str) or len(seq) < 3:
        return None
    s = seq.upper()
    aas = []
    for i in range(0, len(s) - 2, 3):
        aa = _CODON_TABLE.get(s[i:i + 3])
        if aa is None:
            return None if not aas else "".join(aas)
        if aa == "*":
            break
        aas.append(aa)
    return "".join(aas) if aas else None


def _sub(a: str, b: str) -> int:
    return BLOSUM62.get((a, b), -1)


from functools import lru_cache


@lru_cache(maxsize=200_000)
def _self_score(seq: str) -> int:
    return sum(_sub(c, c) for c in seq)


def _nw_score(a: str, b: str) -> float:
    """Gotoh global alignment score with BLOSUM62 and affine gaps.

    Uses parasail (C) when installed; falls back to a NumPy DP otherwise."""
    if _parasail is not None:
        return float(
            _parasail.nw(a, b, -_GAP_OPEN, -_GAP_EXTEND, _PARASAIL_MATRIX).score
        )
    n, m = len(a), len(b)
    M = np.zeros((n + 1, m + 1))
    Ix = np.full((n + 1, m + 1), -np.inf)
    Iy = np.full((n + 1, m + 1), -np.inf)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Ix[i, j] = max(M[i - 1, j] + _GAP_OPEN, Ix[i - 1, j] + _GAP_EXTEND)
            Iy[i, j] = max(M[i, j - 1] + _GAP_OPEN, Iy[i, j - 1] + _GAP_EXTEND)
            M[i, j] = max(M[i - 1, j - 1], Ix[i - 1, j - 1], Iy[i - 1, j - 1]) + _sub(
                a[i - 1], b[j - 1]
            )
    return max(M[n, m], Ix[n, m], Iy[n, m])


def _hamming_sim(a: str, b: str) -> float:
    return 1.0 - float((_encode(a) != _encode(b)).mean())


def _aligned_sim(a: str, b: str) -> float:
    denom = max(_self_score(a), _self_score(b))
    if denom <= 0:
        return 0.0
    return float(min(1.0, max(0.0, _nw_score(a, b) / denom)))


def _valid(seq) -> bool:
    return isinstance(seq, str) and len(seq) > 0


def cdr3_similarity(a: str, b: str, *, mode: str = "auto") -> float:
    """Similarity in [0, 1]. Equal length -> 1 - Hamming/len (fast path);
    otherwise BLOSUM62 global alignment normalized by self-scores.
    'auto': identical -> 1.0; len diff > 4 -> 0.0; equal len -> Hamming;
    else alignment. 'hamming': equal-length fast path only. 'blosum': always
    align."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}.")
    if not (_valid(a) and _valid(b)):
        return 0.0
    if a == b:
        return 1.0
    if mode == "hamming":
        return _hamming_sim(a, b) if len(a) == len(b) else 0.0
    if mode == "auto":
        if abs(len(a) - len(b)) > _MAX_LEN_DIFF:
            return 0.0
        if len(a) == len(b):
            return _hamming_sim(a, b)
    return _aligned_sim(a, b)


def cdr3_similarity_pairs(seqs_a, seqs_b, *, mode: str = "auto") -> np.ndarray:
    """Vectorized pairwise similarity for pre-filtered candidate pairs (1-D
    arrays of equal length). Used by bcrgraph; NOT an all-pairs API."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}.")
    a = np.asarray(seqs_a, dtype=object)
    b = np.asarray(seqs_b, dtype=object)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("seqs_a and seqs_b must be 1-D arrays of equal length.")
    n = a.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    len_a = np.array([len(s) if _valid(s) else -1 for s in a])
    len_b = np.array([len(s) if _valid(s) else -1 for s in b])
    valid = (len_a >= 0) & (len_b >= 0)
    equal_len = valid & (len_a == len_b)
    identical = equal_len & (a == b)
    out[identical] = 1.0

    if mode in ("auto", "hamming"):
        ham_idx = np.flatnonzero(equal_len & ~identical)
        for length in np.unique(len_a[ham_idx]):
            sel = ham_idx[len_a[ham_idx] == length]
            ea = np.stack([_encode(a[i]) for i in sel])
            eb = np.stack([_encode(b[i]) for i in sel])
            out[sel] = 1.0 - (ea != eb).mean(axis=1)
    if mode == "auto":
        rest = valid & ~equal_len & (np.abs(len_a - len_b) <= _MAX_LEN_DIFF)
    elif mode == "blosum":
        rest = valid & ~identical
    else:
        rest = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(rest):
        out[i] = _aligned_sim(a[i], b[i])
    return out


def atchley_embedding(seqs: list) -> np.ndarray:
    """(n, 21) deterministic embedding: mean/sd/min/max of the 5 Atchley
    factors over residues + normalized length. Unknown chars map to zeros."""
    seqs = list(seqs)
    n = len(seqs)
    out = np.zeros((n, 21), dtype=np.float64)
    lengths = np.array([len(s) if _valid(s) else 0 for s in seqs], dtype=np.float64)
    max_len = max(float(lengths.max()), 1.0) if n else 1.0
    zero = np.zeros(5, dtype=np.float64)
    for i, s in enumerate(seqs):
        if not _valid(s):
            continue
        feats = np.stack([ATCHLEY.get(c, zero) for c in s])
        out[i, :20] = np.concatenate(
            [feats.mean(axis=0), feats.std(axis=0), feats.min(axis=0), feats.max(axis=0)]
        )
    out[:, 20] = lengths / max_len
    return out


def _gene(call) -> str:
    if not isinstance(call, str) or not call:
        return ""
    return call.split(",")[0].split("*")[0].strip()


def _family(gene: str) -> str:
    return gene.split("-")[0]


def vj_distance(v1, j1, v2, j2, *, family_level: bool = False) -> float:
    """0.0 same V&J gene; 0.5 same V&J families; 1.0 otherwise.
    family_level=True collapses the 0.5 tier to 0.0."""
    gv1, gv2, gj1, gj2 = _gene(v1), _gene(v2), _gene(j1), _gene(j2)
    if not all([gv1, gv2, gj1, gj2]):
        return 1.0
    if gv1 == gv2 and gj1 == gj2:
        return 0.0
    if _family(gv1) == _family(gv2) and _family(gj1) == _family(gj2):
        return 0.0 if family_level else 0.5
    return 1.0


def cdr3_distance_matrix(cdr3s: list, *, mode: str = "hamming") -> np.ndarray:
    """Pairwise amino-acid distance between CDR3 sequences.

    ``mode="hamming"`` (default, v1 behavior): sequences of equal length are
    compared position-wise (fraction of mismatches). Pairs with different
    lengths receive the maximal normalized distance of 1.0 plus a
    length-mismatch penalty, keeping the metric in a comparable range for
    downstream blending. Missing sequences (None/NaN) get distance 1.0 to
    everything.

    ``mode="blosum"``: distance is ``1 - cdr3_similarity`` (BLOSUM62
    alignment for near-equal lengths), with distance 1.0 for missing
    sequences.

    Parameters
    ----------
    cdr3s
        List of CDR3 amino-acid sequences (may contain None/NaN).
    mode
        ``"hamming"`` or ``"blosum"``.

    Returns
    -------
    Symmetric ``(n, n)`` float matrix with zero diagonal.
    """
    if mode == "blosum":
        n = len(cdr3s)
        dist = np.ones((n, n), dtype=np.float64)
        seqs = [s if _valid(s) else None for s in cdr3s]
        for i in range(n):
            if seqs[i] is None:
                continue
            for j in range(i + 1, n):
                if seqs[j] is None:
                    continue
                dist[i, j] = dist[j, i] = 1.0 - cdr3_similarity(seqs[i], seqs[j])
        np.fill_diagonal(dist, 0.0)
        return dist
    if mode != "hamming":
        raise ValueError("mode must be 'hamming' or 'blosum'.")

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
