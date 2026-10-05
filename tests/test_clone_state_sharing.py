"""Numerical guards for the malaria clone/state co-observation analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "case_studies"))
import clone_state_sharing as css  # noqa: E402


def _cells(rows):
    x = pd.DataFrame(rows)
    for col, value in {"treatment": "T", "isotype": "IGHG", "mutation_frequency": 0.01,
                       "strict_igh": np.nan, "strict_igh_light": np.nan, "light_signature": np.nan}.items():
        if col not in x:
            x[col] = value
    return x


def test_pair_denominators_are_size_and_gc_qualified():
    # rows: GC/PB clone (size 2), GC singleton, and Memory/PB clone (size 2)
    counts = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]])
    assert css.pair_stat(counts, ("GC", "PB"), threshold=2) == (1, 1, 1.0)
    assert css.pair_stat(counts, ("GC", "Memory"), threshold=2) == (0, 1, 0.0)
    assert css.pair_stat(counts, ("Memory", "PB"), threshold=2) == (1, 2, 0.5)
    assert css.pair_stat(counts, ("GC", "PB"), threshold=3)[:2] == (0, 0)
    assert np.isnan(css.pair_stat(counts, ("GC", "PB"), threshold=3)[2])
    # The size cutoff is based on every clone cell, not only the three target states.
    assert css.pair_stat(np.array([[1, 0, 0]]), ("GC", "PB"), 2, np.array([2]))[:2] == (0, 1)
    # Primary joint occupancy fixes the all-size-eligible denominator, unlike
    # the GC-bearing conditional descriptive statistic.
    assert css.pair_stat(counts, ("GC", "PB"), 2, np.array([2, 2, 2]), gc_conditional=False)[:2] == (1, 3)


def test_mode_is_frequency_not_lexicographic_order():
    assert css.first_mode(pd.Series(["IGHA", "IGHG", "IGHG"])) == "IGHG"


def test_clone_ids_are_never_merged_across_donors():
    x = _cells([
        {"cohort": "exp", "donor": "mouse1", "day": "D10", "clone_id": "C1", "cell_state": "GC"},
        {"cohort": "exp", "donor": "mouse1", "day": "D10", "clone_id": "C1", "cell_state": "PB"},
        {"cohort": "exp", "donor": "mouse2", "day": "D10", "clone_id": "C1", "cell_state": "GC"},
        {"cohort": "exp", "donor": "mouse2", "day": "D10", "clone_id": "C1", "cell_state": "Memory"},
    ])
    occ = css.occupancy(x, "threadfin", "clone_id")
    assert len(occ) == 2
    rows = css.observed_rows(occ, "threadfin", 2)
    assert set(rows.donor) == {"mouse1", "mouse2"}
    assert rows.loc[rows.pair.eq("GC+PB"), "observed_shared"].sum() == 1
    assert rows.loc[rows.pair.eq("GC+Memory"), "observed_shared"].sum() == 1


def test_isotype_null_is_bounded_and_respects_isotype_strata():
    # One clone per isotype means an isotype-stratified label permutation cannot
    # move a label between clones.  Its expected rate must equal observation.
    x = _cells([
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "g", "cell_state": "GC", "isotype": "IGHG"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "g", "cell_state": "PB", "isotype": "IGHG"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "m", "cell_state": "GC", "isotype": "IGHM"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "m", "cell_state": "Memory", "isotype": "IGHM"},
    ])
    occ = css.occupancy(x, "threadfin", "clone_id")
    rows = css.permutation_rows(x, occ, n_perm=25, seed=3)
    for col in ("null_mean_rate",):
        assert np.isfinite(rows[col]).all()
        assert rows[col].between(0, 1).all()
    iso = rows[rows.null_type.eq("within_mouse_isotype")]
    assert np.allclose(iso.observed_rate, iso.null_mean_rate)
    joint = css.permutation_rows(x, occ, n_perm=9, seed=4, gc_conditional=False)
    assert set(joint.denominator_clones) == {2}


def test_strict_heavy_control_can_expose_merged_threadfin_sharing():
    x = _cells([
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "C", "strict_igh": "V1|J1|AAA", "cell_state": "GC"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "C", "strict_igh": "V1|J1|AAA", "cell_state": "GC"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "C", "strict_igh": "V2|J2|CCC", "cell_state": "PB"},
        {"cohort": "exp", "donor": "mouse", "day": "D10", "clone_id": "C", "strict_igh": "V2|J2|CCC", "cell_state": "PB"},
    ])
    threadfin = css.observed_rows(css.occupancy(x, "threadfin", "clone_id"), "threadfin", 2)
    strict = css.observed_rows(css.occupancy(x, "strict_igh", "strict_igh"), "strict_igh", 2)
    assert threadfin.loc[threadfin.pair.eq("GC+PB"), "observed_shared"].item() == 1
    assert strict.loc[strict.pair.eq("GC+PB"), "observed_shared"].item() == 0


def test_airr_barcode_suffix_and_absent_treatment_metadata(tmp_path):
    cells = pd.DataFrame({"Unnamed: 0": ["D10_AAAC-1"], "clone_id": ["D10_Hashtag1|C1"],
                          "cell_state": ["GC"], "donor": ["D10_Hashtag1"], "timepoint": ["D10"],
                          "isotype": ["IGHG"], "mutation_frequency": [0.02]})
    bcr = pd.DataFrame([
        {"cell_id": "D10_AAAC", "locus": "IGH", "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01", "junction": "TGTAAA", "c_call": "IGHG"},
        {"cell_id": "D10_AAAC", "locus": "IGK", "v_call": "IGKV1-1*01", "j_call": "IGKJ1*01", "junction": "TGTCCC", "c_call": "IGKC"},
    ])
    cell_path, bcr_path = tmp_path / "cells.csv.gz", tmp_path / "bcr.tsv.gz"
    cells.to_csv(cell_path, index=False, compression="gzip")
    bcr.to_csv(bcr_path, index=False, sep="\t", compression="gzip")
    got, evidence = css.read_cohort("early", cell_path, bcr_path, None)
    assert got.strict_igh.iloc[0] == "IGHV1-1*01|IGHJ1*01|TGTAAA"
    assert got.strict_igh_light.notna().all()
    assert got.treatment.isna().all()
    assert evidence == {"available": False, "metadata_file": None, "values": []}


def test_strict_control_filters_nonproductive_and_ambiguous_airr_calls(tmp_path):
    cells = pd.DataFrame({"Unnamed: 0": ["D10_GOOD-1", "D10_BAD-1", "D10_OFF-1"],
                          "clone_id": ["C1", "C2", "C3"], "cell_state": ["GC", "PB", "GC"],
                          "donor": ["m", "m", "m"], "timepoint": ["D10"] * 3,
                          "isotype": ["IGHG"] * 3, "mutation_frequency": [0.01] * 3})
    bcr = pd.DataFrame([
        {"cell_id": "D10_GOOD", "locus": "IGH", "productive": "T", "v_call": "IGHV1*01", "j_call": "IGHJ1*01", "junction": "TGT", "c_call": "IGHG"},
        {"cell_id": "D10_BAD", "locus": "IGH", "productive": "T", "v_call": "IGHV1*01,IGHV1*02", "j_call": "IGHJ1*01", "junction": "TGT", "c_call": "IGHG"},
        {"cell_id": "D10_OFF", "locus": "IGH", "productive": "F", "v_call": "IGHV1*01", "j_call": "IGHJ1*01", "junction": "TGT", "c_call": "IGHG"},
    ])
    cell_path, bcr_path = tmp_path / "cells.csv.gz", tmp_path / "bcr.tsv.gz"
    cells.to_csv(cell_path, index=False, compression="gzip")
    bcr.to_csv(bcr_path, index=False, sep="\t", compression="gzip")
    got, _ = css.read_cohort("early", cell_path, bcr_path, None)
    assert got.strict_igh.notna().sum() == 1
    assert got.loc[got.barcode.eq("D10_GOOD"), "strict_igh"].notna().all()


def test_paired_light_control_requires_unambiguous_light_calls(tmp_path):
    cells = pd.DataFrame({"Unnamed: 0": ["D10_A-1", "D10_B-1"], "clone_id": ["C1", "C2"],
                          "cell_state": ["GC", "PB"], "donor": ["m", "m"], "timepoint": ["D10", "D10"],
                          "isotype": ["IGHG", "IGHG"], "mutation_frequency": [0.01, 0.01]})
    bcr = pd.DataFrame([
        {"cell_id": "D10_A", "locus": "IGH", "productive": "T", "v_call": "IGHV1*01", "j_call": "IGHJ1*01", "junction": "TGT", "c_call": "IGHG"},
        {"cell_id": "D10_A", "locus": "IGK", "productive": "T", "v_call": "IGKV1*01", "j_call": "IGKJ1*01", "junction": "TGC", "c_call": "IGKC"},
        {"cell_id": "D10_B", "locus": "IGH", "productive": "T", "v_call": "IGHV1*01", "j_call": "IGHJ1*01", "junction": "TGT", "c_call": "IGHG"},
        {"cell_id": "D10_B", "locus": "IGK", "productive": "T", "v_call": "IGKV1*01,IGKV1*02", "j_call": "IGKJ1*01", "junction": "TGC", "c_call": "IGKC"},
    ])
    cell_path, bcr_path = tmp_path / "cells.csv.gz", tmp_path / "bcr.tsv.gz"
    cells.to_csv(cell_path, index=False, compression="gzip")
    bcr.to_csv(bcr_path, index=False, sep="\t", compression="gzip")
    got, _ = css.read_cohort("early", cell_path, bcr_path, None)
    assert got.strict_igh.notna().sum() == 2
    assert got.strict_igh_light.notna().sum() == 1
