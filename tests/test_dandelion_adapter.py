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


def test_cell_key_replaces_cell_id_among_the_required_columns():
    """A table whose barcode column has another name is valid with cell_key; only truly absent columns count."""
    from threadfin.dandelion import define_clones_dandelion
    t = contigs().rename(columns={"cell_id": "barcode"}).drop(columns="v_call")
    with pytest.raises(ValueError, match=r"missing \['v_call'\]"):
        define_clones_dandelion(t, cell_key="barcode", verbose=False)


def test_contig_ids_reused_across_libraries_are_made_unique():
    """Cell Ranger contig ids repeat across libraries; two cells must never share one (GSE287123 did)."""
    from threadfin.dandelion import _unique_sequence_ids
    t = contigs(4, sequence_id=["AAAC-1_contig_1", "AAAC-1_contig_1", "GGGT-1_contig_1", "GGGT-1_contig_2"])
    out = _unique_sequence_ids(t, verbose=False)
    assert out["sequence_id"].is_unique
    assert list(out["sequence_id"].iloc[2:]) == ["GGGT-1_contig_1", "GGGT-1_contig_2"]   # unique ids are kept


def test_a_table_with_both_barcode_and_cell_id_keeps_the_column_that_names_the_cells():
    """A 10x table with an added AIRR cell_id (or the reverse) must not end up with two cell_id columns."""
    from threadfin.dandelion import read_contigs
    t = contigs(4).rename(columns={"cell_id": "barcode"})
    t["cell_id"] = ["x0", "x1", "x2", "x3"]                       # e.g. barcodes before prefixing
    t["barcode"] = ["s1_c0", "s1_c1", "s1_c2", "s1_c3"]           # harmonised as in Tutorial 2
    out = read_contigs(t, obs_names=["s1_c0", "s1_c1", "s1_c2", "s1_c3"])
    assert list(out.columns).count("cell_id") == 1
    assert out["cell_id"].tolist() == ["s1_c0", "s1_c1", "s1_c2", "s1_c3"]
    assert read_contigs(t)["cell_id"].tolist() == ["x0", "x1", "x2", "x3"]   # without obs_names: the AIRR column


def _tcr_contigs():
    """Two T-cell clones (identical beta and alpha junctions) and two cells whose beta differs by one residue."""
    rows = []
    beta = {"c1": "CASSLGQGYEQYF", "c2": "CASSLGQGYEQYF", "c3": "CASSPDRGNTEAFF", "c4": "CASSPDRGNTEAFF",
            "c5": "CASSIRSSYEQYF", "c6": "CASSIRSAYEQYF"}
    alpha = {"c1": "CAVRDSNYQLIW", "c2": "CAVRDSNYQLIW", "c3": "CAMSGGSYIPTF", "c4": "CAMSGGSYIPTF",
             "c5": "CALSEAGNNRKLIW", "c6": "CALSEAGNNRKLIW"}
    from threadfin.sequence import translate_nt  # noqa: F401  (junction_aa given directly)
    for cell in beta:
        rows.append({"cell_id": cell, "locus": "TRB", "v_call": "TRBV7-9*01", "j_call": "TRBJ2-7*01",
                     "junction": "TGT" * 13, "junction_aa": beta[cell], "productive": "T", "umi_count": 5})
        rows.append({"cell_id": cell, "locus": "TRA", "v_call": "TRAV1-2*01", "j_call": "TRAJ33*01",
                     "junction": "TGT" * 12, "junction_aa": alpha[cell], "productive": "T", "umi_count": 3})
    return pd.DataFrame(rows)


def test_t_cell_clones_are_called_on_identical_beta_and_alpha_junctions():
    """TCRs do not hypermutate: by default a T-cell clone needs identical junctions (Dandelion's default)."""
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion
    ids = define_clones_dandelion(_tcr_contigs(), verbose=False)
    assert ids.notna().all() and len(ids) == 6                     # no T cell is dropped for lacking an IGH chain
    assert ids["c1"] == ids["c2"] and ids["c3"] == ids["c4"]
    assert ids["c5"] != ids["c6"]                                  # one residue apart in beta: different clones
    assert ids.attrs["receptor"] == "TCR"
    assert define_clones_dandelion(_bcr_with_extra_chains(), verbose=False).attrs["receptor"] == "BCR"


def _bcr_with_extra_chains():
    """c1 and c2 share a heavy and a light chain, and c2 also carries a weak second heavy chain; c3 has two
    equally strong heavy chains (a likely doublet); c4 is another clone."""
    rows = [("c1", "IGH", "CARDTGDYW", 20), ("c1", "IGK", "CQQYNSYPLTF", 10),
            ("c2", "IGH", "CARDTGDYW", 20), ("c2", "IGK", "CQQYNSYPLTF", 10), ("c2", "IGH", "CAKGGYFDYW", 1),
            ("c3", "IGH", "CARDTGDYW", 10), ("c3", "IGH", "CTTEGGSWW", 10), ("c3", "IGK", "CQQYNSYPLTF", 5),
            ("c4", "IGH", "CAKGGYFDYW", 15), ("c4", "IGK", "CQQSYSTPWTF", 8)]
    t = pd.DataFrame(rows, columns=["cell_id", "locus", "junction_aa", "umi_count"])
    heavy = t["locus"].eq("IGH")
    t["v_call"] = np.where(heavy, "IGHV1-2*02", "IGKV1-39*01")
    t["j_call"] = np.where(heavy, "IGHJ4*02", "IGKJ1*01")
    t["c_call"] = np.where(heavy, "IGHG1", "IGKC")
    t["junction"] = ["TGT" * len(a) for a in t["junction_aa"]]
    t["productive"] = True
    return t


def test_contig_check_keeps_dominant_chains_and_sets_aside_cells_without_one():
    """check_contigs=True runs Dandelion's own check; a cell whose heavy chains were all set aside must not
    come back with a clone called on its light chain alone."""
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion
    checked = define_clones_dandelion(_bcr_with_extra_chains(), check_contigs=True, verbose=False)
    assert checked["c1"] == checked["c2"]                         # the weak extra heavy chain of c2 is dropped
    assert "c3" not in checked.index                               # no dominant heavy chain: set aside
    assert checked["c4"] != checked["c1"]
    assert checked.attrs["check_contigs"] and checked.attrs["n_contigs_ambiguous"] >= 2
    kept = define_clones_dandelion(_bcr_with_extra_chains(), check_contigs=True, require_heavy_chain=False,
                                   verbose=False)
    assert (kept.str.strip() != "").all()                          # an unplaced cell is missing, not clone ""
    # without the check, the cells with two heavy chains get ids that join both chains, which are counted
    plain = define_clones_dandelion(_bcr_with_extra_chains(), verbose=False)
    assert plain.attrs["n_cells_with_combined_id"] == 2 and checked.attrs["n_cells_with_combined_id"] == 0
    assert plain.attrs["n_cells_with_combined_id_two_heavy"] == 2


def test_productive_flags_written_as_numbers_are_understood():
    """A productive column read from CSV with gaps becomes 1.0/0.0; those contigs must not all be dropped."""
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion
    t = _bcr_with_extra_chains()
    t["productive"] = 1.0
    ids = define_clones_dandelion(t, verbose=False)
    assert {"c1", "c2", "c4"} <= set(ids.index)


def test_dandelion_prints_nothing(capsys):
    """Dandelion prints a sanitisation notice twice per donor for tables with boolean columns; it is hidden."""
    pytest.importorskip("dandelion")
    from threadfin.dandelion import define_clones_dandelion
    t = _bcr_with_extra_chains()
    t["rev_comp"] = np.nan                                         # not spelled T/F: Dandelion re-cleans and says so
    define_clones_dandelion(t, verbose=False)
    assert capsys.readouterr().out == ""


def test_receptor_data_can_come_from_a_scirpy_object():
    """Users of scirpy keep their chains in obsm['airr']; read_contigs turns them into one row per chain."""
    ir = pytest.importorskip("scirpy")
    from threadfin.dandelion import read_contigs

    cells = []
    for i, (heavy, light) in enumerate([("CARDTGDYW", "CQQYNSYPLTF"), ("CARDTGDYW", "CQQYNSYPLTF"), ("CAKGGYW", None)]):
        cell = ir.io.AirrCell(cell_id=f"cell{i}")
        for locus, junction_aa in (("IGH", heavy), ("IGK", light)):
            if junction_aa is None:
                continue
            chain = ir.io.AirrCell.empty_chain_dict()
            chain.update(locus=locus, v_call=f"{locus}V1-2*01", j_call=f"{locus}J1*01", junction="TGT" * 4,
                         junction_aa=junction_aa, productive=True, consensus_count=10, duplicate_count=3)
            cell.add_chain(chain)
        cells.append(cell)
    adata = ir.io.from_airr_cells(cells)
    t = read_contigs(adata)
    assert len(t) == 5 and set(t["cell_id"]) == {"cell0", "cell1", "cell2"}
    assert {"v_call", "j_call", "junction", "locus"} <= set(t.columns)


def test_dandelion_1x_is_flagged(monkeypatch):
    """sc-dandelion 1.0 calls clones differently from the validated 0.5.x; using it must not be silent."""
    pytest.importorskip("dandelion")
    import threadfin.dandelion as tfd
    monkeypatch.setattr(tfd, "_dandelion_version", lambda ddl: "1.0.0")
    with pytest.warns(UserWarning, match="validated with sc-dandelion 0.5.x"):
        tfd._require_dandelion()
