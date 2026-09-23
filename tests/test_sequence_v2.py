import numpy as np
import pytest

from threadfin.sequence import (
    ATCHLEY,
    BLOSUM62,
    atchley_embedding,
    blend_distances,
    cdr3_distance_matrix,
    cdr3_similarity,
    cdr3_similarity_pairs,
    vj_distance,
)


def test_blosum62_values():
    # standard NCBI BLOSUM62 spot checks
    assert BLOSUM62[("W", "W")] == 11
    assert BLOSUM62[("C", "C")] == 9
    assert BLOSUM62[("A", "R")] == -1 == BLOSUM62[("R", "A")]
    assert BLOSUM62[("*", "*")] == 1
    assert BLOSUM62[("A", "*")] == -4
    assert BLOSUM62[("D", "B")] == 4
    assert BLOSUM62[("Q", "Z")] == 3
    assert len(BLOSUM62) == 24 * 24


def test_atchley_table():
    assert set(ATCHLEY) == set("ACDEFGHIKLMNPQRSTVWY")
    assert all(v.shape == (5,) for v in ATCHLEY.values())
    assert np.allclose(ATCHLEY["A"], [-0.591, -1.302, -0.733, 1.570, -0.146])


def test_similarity_ordering():
    ref = "CASSIRSSYEQYF"
    similar = cdr3_similarity(ref, "CASSIRSSYEQFF")
    dissimilar = cdr3_similarity(ref, "AAAAAAAAAAA")
    assert 0.0 <= dissimilar < similar <= 1.0
    assert similar == pytest.approx(1.0 - 1.0 / len(ref))


def test_similarity_identical_and_invalid():
    assert cdr3_similarity("CASSIRSSYEQYF", "CASSIRSSYEQYF") == 1.0
    assert cdr3_similarity("", "CASSIRSSYEQYF") == 0.0
    assert cdr3_similarity(None, "CASSIRSSYEQYF") == 0.0
    assert cdr3_similarity(np.nan, "CASSIRSSYEQYF") == 0.0
    with pytest.raises(ValueError):
        cdr3_similarity("AAA", "AAA", mode="nope")


def test_similarity_unequal_length_alignment():
    # one-residue deletion within tolerance -> aligned, high but < 1
    sim = cdr3_similarity("CASSIRSSYEQYF", "CASSIRSSYEQY")
    assert 0.0 < sim < 1.0
    # alignment rewards similarity beyond raw mismatch counting
    far = cdr3_similarity("CASSIRSSYEQYF", "AAACASSIRSSYE")
    assert far < sim


def test_similarity_len_diff_shortcut():
    assert cdr3_similarity("AAAAA", "AAAAAAAAAA") == 0.0  # diff 5 > 4 in auto
    assert cdr3_similarity("AAAAA", "AAAAAAAAAA", mode="hamming") == 0.0
    # blosum mode always aligns, no shortcut
    assert cdr3_similarity("AAAAA", "AAAAAAAAAA", mode="blosum") > 0.0
    assert cdr3_similarity("AAAAA", "AAAAAAAA", mode="auto") > 0.0  # diff 3


def test_similarity_hamming_mode():
    assert cdr3_similarity("AAAA", "AAAT", mode="hamming") == pytest.approx(0.75)
    assert cdr3_similarity("AAAA", "AAAT", mode="auto") == pytest.approx(0.75)
    assert cdr3_similarity("AAAA", "AAAAT", mode="hamming") == 0.0


def test_similarity_pairs_consistency():
    a = ["CASSIRSSYEQYF", "CASSIRSSYEQYF", "AAAAA", "AAAA", None, "CASSIRSSYEQYF", "AAAAA"]
    b = ["CASSIRSSYEQYF", "CASSIRSSYEQFF", "AAAAAAAAAA", "AAAT", "AAAA", "CASSIRSSYEQY", "AAAAAAAA"]
    for mode in ("auto", "hamming", "blosum"):
        got = cdr3_similarity_pairs(a, b, mode=mode)
        assert isinstance(got, np.ndarray) and got.shape == (len(a),)
        expected = [cdr3_similarity(x, y, mode=mode) for x, y in zip(a, b)]
        assert np.allclose(got, expected)
        assert ((got >= 0) & (got <= 1)).all()
    with pytest.raises(ValueError):
        cdr3_similarity_pairs(a, b[:-1])
    assert cdr3_similarity_pairs([], []).shape == (0,)


def test_atchley_embedding():
    seqs = ["CASSIRSSYEQYF", "ARAAA", "ZZZ*", None, ""]
    emb = atchley_embedding(seqs)
    assert emb.shape == (5, 21)
    assert np.isfinite(emb).all()
    # deterministic
    assert np.array_equal(emb, atchley_embedding(seqs))
    # unknown chars contribute zeros; None/empty rows are zero except length 0
    assert np.allclose(emb[3], 0.0)
    assert np.allclose(emb[4], 0.0)
    # normalized length: longest sequence maps to 1.0
    assert emb[0, 20] == pytest.approx(1.0)
    assert emb[1, 20] == pytest.approx(5 / 13)
    # single-aa sequence: sd = 0, mean = min = max = factor values
    single = atchley_embedding(["A"])
    assert np.allclose(single[0, :5], ATCHLEY["A"])
    assert np.allclose(single[0, 5:10], 0.0)
    assert np.allclose(single[0, 10:15], ATCHLEY["A"])


def test_vj_distance_tiers():
    assert vj_distance("IGHV3-23*01", "IGHJ4*01", "IGHV3-23*02", "IGHJ4*02") == 0.0
    assert vj_distance("IGHV3-23*01", "IGHJ4*01", "IGHV3-48*01", "IGHJ4*03") == 0.5
    assert vj_distance("IGHV3-23*01", "IGHJ4*01", "IGHV3-48*01", "IGHJ4*03",
                       family_level=True) == 0.0
    assert vj_distance("IGHV3-23*01", "IGHJ4*01", "IGHV1-2*01", "IGHJ4*01") == 1.0
    assert vj_distance("IGHV3-23*01", "IGHJ4*01", "IGHV3-23*01", "IGHJ6*01") == 1.0
    # multi-call: first call wins
    assert vj_distance("IGHV3-23*01,IGHV3-23*02", "IGHJ4*01", "IGHV3-23*01", "IGHJ4*01") == 0.0
    assert vj_distance("IGHV3-23*01,IGHV1-2*01", "IGHJ4*01", "IGHV3-48*01", "IGHJ4*02") == 0.5
    # missing calls are maximally distant
    assert vj_distance(None, "IGHJ4*01", "IGHV3-23*01", "IGHJ4*01") == 1.0


def test_v1_functions_unchanged():
    d = cdr3_distance_matrix(["ARAAAA", "ARAAAA", "ARAAAT", None, "ARAA"])
    assert d[0, 1] == 0
    assert 0 < d[0, 2] < 0.2  # one mismatch in 6 aa
    assert d[0, 3] == 1.0     # missing sequence
    assert d[0, 4] > d[0, 2]  # length penalty
    assert np.array_equal(d, cdr3_distance_matrix(["ARAAAA", "ARAAAA", "ARAAAT", None, "ARAA"],
                                                  mode="hamming"))

    b = blend_distances(np.random.default_rng(0).random((5, 5)), np.eye(5), 0.5)
    assert b.shape == (5, 5)
    assert (np.diag(b) == 0).all()


def test_distance_matrix_blosum_mode():
    seqs = ["CASSIRSSYEQYF", "CASSIRSSYEQFF", "AAAAAAAAAAA", None]
    d = cdr3_distance_matrix(seqs, mode="blosum")
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T)
    assert (np.diag(d) == 0).all()
    assert d[0, 1] < d[0, 2]  # similar pair closer than dissimilar
    assert d[0, 3] == 1.0     # missing sequence
    assert ((d >= 0) & (d <= 1))[np.ix_([0, 1, 2], [0, 1, 2])].all()
    with pytest.raises(ValueError):
        cdr3_distance_matrix(seqs, mode="nope")
