"""Tests for threadfin.io: read_10x_vdj, read_airr and build_clone_key.

Current-behavior notes (io.py was NOT modified):

* Cells carrying only light-chain contigs still appear in the
  ``read_10x_vdj`` output (via index-union alignment), but with NaN
  heavy-chain columns (``v_call``/``d_call``/``j_call``/``cdr3``/``umis``/
  ``clonotype_id``).
* Cell Ranger writes ``productive``/``high_confidence`` as the strings
  "True"/"False"; ``pandas.read_csv`` infers bool for those, so the
  ``== True`` filtering in ``read_10x_vdj`` works on real files (pinned in
  ``test_read_10x_vdj_cellranger_string_flags``).
"""

import pandas as pd
import pytest

from threadfin.io import build_clone_key, read_10x_vdj, read_airr

_10X_COLUMNS = [
    "barcode", "contig_id", "chain", "v_gene", "d_gene", "j_gene", "c_gene",
    "cdr3", "cdr3_nt", "umis", "productive", "high_confidence",
    "raw_clonotype_id",
]


def _write_10x_csv(path):
    rows = [
        # cellA: two IGH contigs (UMI 30 vs 10) -> highest-UMI wins; plus an IGK.
        ("cellA", "cA_h1", "IGH", "IGHV1-2*01", "IGHD1-1*01", "IGHJ4*02",
         "IGHM*01", "CAKEDY", "TGTGCC", 30, True, True, "clonotypeA"),
        ("cellA", "cA_h2", "IGH", "IGHV3-9*01", "IGHD2-2*01", "IGHJ3*02",
         "IGHM*01", "CASSDY", "TGTAGC", 10, True, True, "clonotypeA"),
        ("cellA", "cA_k1", "IGK", "IGKV1-5*01", None, "IGKJ2*01",
         "IGKC*01", "CQQY", "TGTCAG", 5, True, True, "clonotypeA"),
        # cellB: one IGH + one IGL; a high-UMI TRA contig must be ignored.
        ("cellB", "cB_h1", "IGH", "IGHV4-34*01", "IGHD3-3*01", "IGHJ5*02",
         "IGHG1*01", "CARDY", "TGTGCC", 8, True, True, "clonotypeB"),
        ("cellB", "cB_l1", "IGL", "IGLV2-14*01", None, "IGLJ3*02",
         "IGLC2*01", "CAAWF", "TGTGCT", 20, True, True, "clonotypeB"),
        ("cellB", "cB_t1", "TRA", "TRAV1*01", None, "TRAJ4*01",
         "TRAC*01", "CAW", "TGTGC", 99, True, True, None),
        # cellC: IGK (UMI 25) and IGL (UMI 3) -> the IGK must win.
        ("cellC", "cC_h1", "IGH", "IGHV5-51*01", "IGHD4-4*01", "IGHJ6*02",
         "IGHA1*01", "CARGY", "TGTGCC", 12, True, True, "clonotypeC"),
        ("cellC", "cC_k1", "IGK", "IGKV3-11*01", None, "IGKJ4*01",
         "IGKC*01", "CQQSF", "TGTCAG", 25, True, True, "clonotypeC"),
        ("cellC", "cC_l1", "IGL", "IGLV1-40*01", None, "IGLJ2*01",
         "IGLC1*01", "CAAIY", "TGTGCT", 3, True, True, "clonotypeC"),
        # cellD: only heavy contig is non-productive -> cell dropped entirely.
        ("cellD", "cD_h1", "IGH", "IGHV2-5*01", "IGHD5-5*01", "IGHJ2*01",
         "IGHM*01", "CAMES", "TGTGCC", 15, False, True, "clonotypeD"),
        ("cellD", "cD_k1", "IGK", "IGKV2-24*01", None, "IGKJ1*01",
         "IGKC*01", "CQQAW", "TGTCAG", 4, True, True, "clonotypeD"),
        # cellE: only heavy contig is low-confidence -> cell dropped entirely.
        ("cellE", "cE_h1", "IGH", "IGHV6-1*01", "IGHD6-6*01", "IGHJ3*02",
         "IGHM*01", "CAGDY", "TGTGCC", 22, True, False, "clonotypeE"),
    ]
    pd.DataFrame(rows, columns=_10X_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture()
def vdj_csv(tmp_path):
    return _write_10x_csv(tmp_path / "filtered_contig_annotations.csv")


@pytest.fixture()
def vdj_csv_gz(tmp_path):
    return _write_10x_csv(tmp_path / "filtered_contig_annotations.csv.gz")


def test_read_10x_vdj_plain_and_gz(vdj_csv, vdj_csv_gz):
    plain = read_10x_vdj(str(vdj_csv))
    gz = read_10x_vdj(str(vdj_csv_gz))
    pd.testing.assert_frame_equal(plain, gz)

    assert plain.index.name == "barcode"
    # cellD's only heavy contig is non-productive and cellE's is low-confidence,
    # so both lose their heavy chain. cellD survives as a light-only row (NaN
    # heavy columns); cellE has no post-filter contigs at all and is dropped.
    assert list(plain.index) == ["cellA", "cellB", "cellC", "cellD"]
    d = plain.loc["cellD"]
    assert pd.isna(d["v_call"]) and pd.isna(d["cdr3"]) and pd.isna(d["clonotype_id"])
    assert d["v_call_light"] == "IGKV2-24*01"
    assert d["cdr3_light"] == "CQQAW"
    assert d["n_contigs"] == 1
    assert "cellE" not in plain.index

    # Highest-UMI heavy contig wins for cellA (UMI 30, not the UMI-10 one).
    a = plain.loc["cellA"]
    assert a["v_call"] == "IGHV1-2*01"
    assert a["d_call"] == "IGHD1-1*01"
    assert a["j_call"] == "IGHJ4*02"
    assert a["c_call"] == "IGHM*01"
    assert a["cdr3"] == "CAKEDY"
    assert a["cdr3_nt"] == "TGTGCC"
    assert a["umis"] == 30
    assert a["clonotype_id"] == "clonotypeA"

    # Light chain: cellA's IGK passes through; cellC's higher-UMI IGK (25)
    # beats its IGL (3); cellB's IGL passes through.
    assert a["v_call_light"] == "IGKV1-5*01"
    assert a["cdr3_light"] == "CQQY"
    assert plain.loc["cellC", "v_call_light"] == "IGKV3-11*01"
    assert plain.loc["cellC", "cdr3_light"] == "CQQSF"
    assert plain.loc["cellB", "v_call_light"] == "IGLV2-14*01"

    # n_contigs counts post-filter BCR contigs only (cellB's TRA excluded).
    assert plain.loc["cellA", "n_contigs"] == 3
    assert plain.loc["cellB", "n_contigs"] == 2
    assert plain.loc["cellC", "n_contigs"] == 3


def test_read_10x_vdj_missing_required_column(tmp_path):
    path = _write_10x_csv(tmp_path / "bad.csv")
    df = pd.read_csv(path).drop(columns=["chain"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        read_10x_vdj(str(path))


def test_read_10x_vdj_cellranger_string_flags(tmp_path):
    """Cell Ranger writes productive/high_confidence as the strings
    "True"/"False"; pandas infers bool on read, so the filtering works on
    real files."""
    path = tmp_path / "string_flags.csv"
    df = pd.read_csv(_write_10x_csv(path.with_name("src.csv")))
    df["productive"] = df["productive"].map({True: "True", False: "False"})
    df["high_confidence"] = df["high_confidence"].map({True: "True", False: "False"})
    df.to_csv(path, index=False)
    out = read_10x_vdj(str(path))
    # same result as with boolean flags: cellD loses its heavy chain, cellE
    # drops out entirely
    assert list(out.index) == ["cellA", "cellB", "cellC", "cellD"]
    assert out.loc["cellA", "v_call"] == "IGHV1-2*01"
    assert pd.isna(out.loc["cellD", "v_call"])
    assert "cellE" not in out.index


_AIRR_COLUMNS = [
    "cell_id", "sequence_id", "locus", "v_call", "d_call", "j_call",
    "c_call", "junction_aa", "clone_id",
]


def _write_airr_tsv(path, with_cell_id=True):
    rows = [
        ("cellA", "cellA_contig_1", "IGH", "IGHV1-2*01", "IGHD1-1*01",
         "IGHJ4*02", "IGHM*01", "CAKEDY", "CLONE1"),
        ("cellA", "cellA_contig_2", "IGL", "IGLV2-14*01", None,
         "IGLJ3*02", "IGLC2*01", "CAAWF", "CLONE1-light"),
        ("cellB", "cellB_contig_1", "IGH", "IGHV4-34*01", "IGHD3-3*01",
         "IGHJ5*02", "IGHG1*01", "CARDY", "CLONE2"),
        ("cellB", "cellB_contig_2", "IGL", "IGLV1-40*01", None,
         "IGLJ2*01", "IGLC1*01", "CAAIY", "CLONE2-light"),
    ]
    df = pd.DataFrame(rows, columns=_AIRR_COLUMNS)
    if not with_cell_id:
        df = df.drop(columns=["cell_id"])
    df.to_csv(path, sep="\t", index=False)
    return path


def test_read_airr_prefers_igh_and_renames_junction(tmp_path):
    path = _write_airr_tsv(tmp_path / "airr.tsv")
    per_cell = read_airr(str(path))

    assert per_cell.index.name == "barcode"
    assert list(per_cell.index) == ["cellA", "cellB"]

    # IGH rows win over IGL rows for the same cell.
    a = per_cell.loc["cellA"]
    assert a["v_call"] == "IGHV1-2*01"
    assert a["j_call"] == "IGHJ4*02"
    assert a["c_call"] == "IGHM*01"

    # junction_aa is renamed to cdr3 (no literal cdr3/cdr3_aa column present)...
    assert "cdr3" in per_cell.columns
    assert a["cdr3"] == "CAKEDY"
    assert per_cell.loc["cellB", "cdr3"] == "CARDY"
    # ...and the original junction_aa column does not survive.
    assert "junction_aa" not in per_cell.columns
    assert "locus" not in per_cell.columns

    # clone_id survives.
    assert per_cell.loc["cellA", "clone_id"] == "CLONE1"
    assert per_cell.loc["cellB", "clone_id"] == "CLONE2"


def test_read_airr_sequence_id_fallback(tmp_path):
    path = _write_airr_tsv(tmp_path / "airr_nocellid.tsv", with_cell_id=False)
    per_cell = read_airr(str(path))
    # the trailing _contig_<n> suffix is stripped to recover the barcode
    assert list(per_cell.index) == ["cellA", "cellB"]
    assert per_cell.loc["cellA", "clone_id"] == "CLONE1"


def test_build_clone_key_vdj():
    tab = pd.DataFrame(
        {"v_call": ["IGHV1-2*01", "IGHV3-9*01"],
         "d_call": ["IGHD1-1*01", "IGHD2-2*01"],
         "j_call": ["IGHJ4*02", "IGHJ3*02"],
         "cdr3": ["CAKEDY", "CASSDY"]},
        index=["cellA", "cellB"],
    )
    out = build_clone_key(tab, strategy="vdj")
    assert out.loc["cellA", "clone_id"] == "IGHV1-2*01_IGHD1-1*01_IGHJ4*02"
    assert out.loc["cellB", "clone_id"] == "IGHV3-9*01_IGHD2-2*01_IGHJ3*02"
    # default strategy is "vdj"
    assert build_clone_key(tab).equals(out)
    # input is not modified in place
    assert "clone_id" not in tab.columns
    # missing required column
    with pytest.raises(ValueError, match="requires a 'd_call' column"):
        build_clone_key(tab.drop(columns=["d_call"]), strategy="vdj")


def test_build_clone_key_clonotype_id():
    tab = pd.DataFrame(
        {"v_call": ["IGHV1-2*01"], "clonotype_id": ["clonotypeA"]},
        index=["cellA"],
    )
    out = build_clone_key(tab, strategy="clonotype_id")
    assert out.loc["cellA", "clone_id"] == "clonotypeA"
    with pytest.raises(ValueError, match="requires a 'clonotype_id' column"):
        build_clone_key(tab.drop(columns=["clonotype_id"]), strategy="clonotype_id")


def test_build_clone_key_cdr3():
    tab = pd.DataFrame(
        {"v_call": ["IGHV1-2*01", "IGHV3-9*01"], "cdr3": ["CAKEDY", "CASSDY"]},
        index=["cellA", "cellB"],
    )
    out = build_clone_key(tab, strategy="cdr3")
    assert out.loc["cellA", "clone_id"] == "CAKEDY"
    assert out.loc["cellB", "clone_id"] == "CASSDY"
    # custom output column name
    out = build_clone_key(tab, strategy="cdr3", out_col="my_key")
    assert out.loc["cellA", "my_key"] == "CAKEDY"


def test_build_clone_key_unknown_strategy():
    tab = pd.DataFrame({"v_call": ["IGHV1-2*01"]}, index=["cellA"])
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_clone_key(tab, strategy="nope")


def test_build_clone_key_missing_values_stay_na():
    """NaN / 'None' / '' clonotype or CDR3 values must not become literal
    'nan'/'None' clone keys (regression: per-donor NaN buckets once formed
    giant pseudo-clones)."""
    tab = pd.DataFrame(
        {
            "clonotype_id": ["clonotype1", None, "None", "clonotype2"],
            "cdr3": ["CAKEDY", "CASSDY", None, "CARQW"],
            "v_call": ["IGHV1-2*01", None, "IGHV3-9*01", "IGHV4-34*01"],
            "d_call": ["IGHD1-1*01", "IGHD2-2*01", None, "IGHD3-3*01"],
            "j_call": ["IGHJ4*01", "IGHJ6*01", "IGHJ4*01", "IGHJ5*01"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    out = build_clone_key(tab, strategy="clonotype_id")
    assert out.loc["c1", "clone_id"] == "clonotype1"
    assert pd.isna(out.loc["c2", "clone_id"])
    assert pd.isna(out.loc["c3", "clone_id"])
    out = build_clone_key(tab, strategy="cdr3")
    assert pd.isna(out.loc["c3", "clone_id"])
    assert out.loc["c2", "clone_id"] == "CASSDY"
    out = build_clone_key(tab, strategy="vdj")
    assert pd.isna(out.loc["c2", "clone_id"])  # missing v_call
    assert pd.isna(out.loc["c3", "clone_id"])  # missing d_call
    assert out.loc["c1", "clone_id"] == "IGHV1-2*01_IGHD1-1*01_IGHJ4*01"


def test_translate_nt():
    from threadfin.sequence import translate_nt

    assert translate_nt("TGTGCACGG") == "CAR"
    assert translate_nt("TGTGCACGGTAA") == "CAR"  # stop at stop codon
    assert translate_nt("TG") is None
    assert translate_nt(None) is None
    assert translate_nt("TGTNNNGGG") == "C"  # unknown codon truncates


def test_read_airr_cdr3_from_nt_junction(tmp_path):
    """AIRR files with empty junction_aa but populated junction (nt) get
    their CDR3 translated (GSE175522-style files)."""
    p = tmp_path / "airr.tsv"
    p.write_text(
        "cell_id\tlocus\tv_call\td_call\tj_call\tjunction_aa\tjunction\n"
        "cellA\tIGH\tIGHV1-2*01\tIGHD1-1*01\tIGHJ4*01\t\tTGTGCACGG\n"
        "cellB\tIGH\tIGHV3-9*01\tIGHD2-2*01\tIGHJ6*01\t\tTGTGCGTGA\n"
    )
    tab = read_airr(str(p), cell_col="cell_id")
    assert tab.loc["cellA", "cdr3"] == "CAR"
    assert tab.loc["cellB", "cdr3"] == "CA"  # TGA stop truncates
