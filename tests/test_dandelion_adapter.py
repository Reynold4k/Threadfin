"""The Dandelion adapter must never hand `find_clones` a constant grouping key.

`ddl.tl.find_clones` groups on `junction_aa` by default. An earlier version of the adapter filled every
absent optional AIRR column with an empty string so Dandelion's constructor would validate, and
`junction_aa` was in that list. GSE175522 ships it entirely null, so every cell received the same key and
merged on V gene, J gene and junction length alone: 84,040 cells collapsed into 1,968 "clones" with a
972-cell largest, and that dataset's clonal coherence fell from 0.558 to 0.050.

These tests cover the helper directly, so they run without the optional `sc-dandelion` dependency.
"""

import numpy as np
import pandas as pd
import pytest

from threadfin.dandelion import _fill_junction_aa

STEM = "TGTGCGAGAGATACCGGTGACTAC"                      # translates to CARDTGDY
#: one codon per amino acid, so every junction below gives a *different* peptide
CODONS = ["TGG", "TTT", "TAT", "CGG", "GGG", "AAA", "CCC", "ACA", "GTT", "ATG"]
AA = "CARDTGDYW"                                         # STEM + CODONS[0]


def junctions(n):
    """`n` distinct junctions, each translating to a distinct peptide."""
    out = []
    for i in range(n):
        a, b = CODONS[i % len(CODONS)], CODONS[i // len(CODONS)]
        out.append(STEM + a + ("" if i < len(CODONS) else b))
    return out


def contigs(n=6, **cols):
    t = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(n)],
        "locus": ["IGH"] * n,
        "v_call": ["IGHV1-2*02"] * n,
        "j_call": ["IGHJ4*02"] * n,
        "junction": junctions(n),
    })
    for k, v in cols.items():
        t[k] = v
    return t


def test_the_fixture_itself_gives_distinct_peptides():
    """Guards the tests below: if the fixture collided, they would pass for the wrong reason."""
    from threadfin.sequence import translate_nt
    assert len({translate_nt(j) for j in junctions(30)}) == 30


def test_absent_junction_aa_is_translated_not_blanked():
    out = _fill_junction_aa(contigs(), verbose=False)
    assert (out["junction_aa"] != "").all()
    assert out["junction_aa"].nunique() == len(out), "translation must preserve distinct junctions"
    assert out["junction_aa"].iloc[0] == AA


def test_all_null_junction_aa_is_translated():
    """GSE175522's exact shape: the column exists but every value is NaN."""
    out = _fill_junction_aa(contigs(junction_aa=np.nan), verbose=False)
    assert out["junction_aa"].nunique() == len(out)


@pytest.mark.parametrize("blank", ["", "  ", "nan", "None", "out_of_frame", "partial", "?", "-"])
def test_every_blank_spelling_is_refilled(blank):
    out = _fill_junction_aa(contigs(junction_aa=blank), verbose=False)
    assert out["junction_aa"].nunique() == len(out)


def test_partially_present_column_keeps_what_is_there_and_fills_the_rest():
    t = contigs()
    t["junction_aa"] = ["GIVEN1", "", "GIVEN3", np.nan, "out_of_frame", "GIVEN6"]
    out = _fill_junction_aa(t, verbose=False)
    assert list(out["junction_aa"].iloc[[0, 2, 5]]) == ["GIVEN1", "GIVEN3", "GIVEN6"]
    assert (out["junction_aa"] != "").all()
    assert out["junction_aa"].nunique() == len(out)


def test_a_few_untranslatable_contigs_are_dropped():
    t = contigs(n=40)                                # one bad row is 2.5%, under the 5% refusal
    t.loc[0, "junction"] = "NN"                      # too short to translate
    out = _fill_junction_aa(t, verbose=False)
    assert len(out) == 39 and "c0" not in set(out["cell_id"])


def test_mostly_untranslatable_input_is_refused_rather_than_merged():
    """The failure this guards against is silent, so it must raise rather than return one big clone."""
    t = contigs(n=6)
    t["junction"] = "NN"
    with pytest.raises(ValueError, match="junction_aa"):
        _fill_junction_aa(t, verbose=False)


# --------------------------------------------------------------------------- cells without a heavy chain
# Dandelion also groups cells whose heavy chain was not recovered, by their light chain alone (ids such as
# "B_VJ_..."). Light-chain rearrangements are far less diverse than heavy-chain ones, so such groups join
# unrelated cells; in GSE287123 they were 740 of 2,626 groups of two or more cells. The adapter must leave
# these cells without a clone unless asked otherwise.

LIGHT_STEM = "TGTCAGCAGTATAATAGC"                         # translates to CQQYNS


def paired_and_light_only():
    """Two cells with the same heavy + light chain, and two cells that only have (the same) light chain."""
    rows = []
    for cell in ("p1", "p2"):
        rows.append({"cell_id": cell, "locus": "IGH", "v_call": "IGHV1-2*02", "j_call": "IGHJ4*02",
                     "junction": STEM + "TGG", "productive": "T", "umi_count": 10})
        rows.append({"cell_id": cell, "locus": "IGK", "v_call": "IGKV1-5*01", "j_call": "IGKJ1*01",
                     "junction": LIGHT_STEM + "TGGACG", "productive": "T", "umi_count": 8})
    for cell in ("l1", "l2"):
        rows.append({"cell_id": cell, "locus": "IGK", "v_call": "IGKV3-20*01", "j_call": "IGKJ2*01",
                     "junction": LIGHT_STEM + "CCGTAC", "productive": "T", "umi_count": 8})
    return pd.DataFrame(rows)


def test_cells_with_heavy_chain_helper():
    from threadfin.dandelion import _cells_with_heavy_chain

    cells = set(_cells_with_heavy_chain(paired_and_light_only()))
    assert cells == {"p1", "p2"}


def test_light_only_cells_get_no_clone_by_default():
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion

    t = paired_and_light_only()
    ids = define_clones_dandelion(t, verbose=False)
    assert set(ids.index) == {"p1", "p2"}
    assert ids["p1"] == ids["p2"]
    assert ids.attrs["n_cells_without_heavy_chain"] == 2
    kept = define_clones_dandelion(t, require_heavy_chain=False, verbose=False)
    assert {"l1", "l2"} <= set(kept.index)


def test_hash_seed_in_use_is_recorded(monkeypatch):
    # Dandelion's clone calls depend on Python's string-hash seed; the result says which one was in use
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert define_clones_dandelion(paired_and_light_only(), verbose=False).attrs["python_hash_seed"] == "0"
    monkeypatch.delenv("PYTHONHASHSEED")
    assert define_clones_dandelion(paired_and_light_only(), verbose=False).attrs["python_hash_seed"] is None
