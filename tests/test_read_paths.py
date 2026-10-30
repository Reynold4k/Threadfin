"""The README's main input path: a BCR file on disk, read by tf.read_bcr and by tf.run(bcr=<path>).

The files are written to a temporary folder: a Cell Ranger ``filtered_contig_annotations.csv`` for 26 B cells
of two donors (six expanded clones with paired heavy and light chains, two singletons, plus contigs that
must be ignored) and a small AIRR rearrangement table.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf

_CODON = {"A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT", "G": "GGT", "H": "CAT", "I": "ATT",
          "K": "AAA", "L": "CTG", "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAG", "R": "CGT", "S": "TCT",
          "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT"}


def _nt(aa: str) -> str:
    return "".join(_CODON[a] for a in aa)


# clone -> (donor, n cells, heavy V, heavy J, isotype, heavy CDR3, light chain, light V, light J, light CDR3).
# Clone B1 carries exactly the receptor of A1: clones are called within donors, so they must stay apart.
CLONES = {
    "A1": ("A", 5, "IGHV3-23", "IGHJ4", "IGHG1", "CAKDRGYSSGWYFDYW", "IGK", "IGKV1-5", "IGKJ1", "CQQYNSYPWTF"),
    "A2": ("A", 4, "IGHV1-69", "IGHJ6", "IGHM", "CARGGPLYYYGMDVW", "IGL", "IGLV2-14", "IGLJ2", "CSSYTSSSTLVF"),
    "A3": ("A", 3, "IGHV4-34", "IGHJ5", "IGHA1", "CARVNWFDPW", "IGK", "IGKV3-20", "IGKJ2", "CQQYGSSPYTF"),
    "A4": ("A", 1, "IGHV5-51", "IGHJ3", "IGHG2", "CARHEGWAFDIW", "IGK", "IGKV1-39", "IGKJ1", "CQQSYSTPRTF"),
    "B1": ("B", 5, "IGHV3-23", "IGHJ4", "IGHG1", "CAKDRGYSSGWYFDYW", "IGK", "IGKV1-5", "IGKJ1", "CQQYNSYPWTF"),
    "B2": ("B", 4, "IGHV3-30", "IGHJ4", "IGHG3", "CAKEGTTVVTPFDYW", "IGK", "IGKV4-1", "IGKJ4", "CQQYYSTPLTF"),
    "B3": ("B", 3, "IGHV1-2", "IGHJ4", "IGHM", "CARDLWSGYYPEDYW", "IGL", "IGLV1-44", "IGLJ3", "CAAWDDSLNGWVF"),
    "B4": ("B", 1, "IGHV4-39", "IGHJ6", "IGHM", "CARQTYYDFWSGYPNYGMDVW", "IGL", "IGLV3-21", "IGLJ2", "CQVWDSSSDHVVF"),
}
N_WITHOUT_CONTIGS = 4


def _cells():
    """Barcode, donor and planted clone of every cell (cells without contigs last)."""
    rows = []
    for clone, spec in CLONES.items():
        rows += [{"barcode": f"{clone}-{i}", "donor": spec[0], "clone": clone} for i in range(spec[1])]
    rows += [{"barcode": f"NOVDJ-{i}", "donor": "AB"[i % 2], "clone": None} for i in range(N_WITHOUT_CONTIGS)]
    return pd.DataFrame(rows).set_index("barcode")


def _write_10x_csv(path):
    rows = []

    def contig(barcode, k, chain, v, j, c, cdr3, umis, productive=True, high_confidence=True):
        rows.append({"barcode": barcode, "is_cell": True, "contig_id": f"{barcode}_contig_{k}",
                     "high_confidence": high_confidence, "length": 550, "chain": chain, "v_gene": v,
                     "d_gene": "None" if chain != "IGH" else "IGHD3-10", "j_gene": j, "c_gene": c,
                     "full_length": True, "productive": productive, "cdr3": cdr3, "cdr3_nt": _nt(cdr3),
                     "reads": 40 * umis, "umis": umis, "raw_clonotype_id": "clonotype1",
                     "raw_consensus_id": "clonotype1_consensus_1"})

    for barcode, cell in _cells().dropna(subset=["clone"]).iterrows():
        _, _, v, j, iso, cdr3, light, lv, lj, lcdr3 = CLONES[cell["clone"]]
        if barcode == "A1-1":                       # somatic hypermutation: one residue of 16 differs
            cdr3 = cdr3[:5] + "S" + cdr3[6:]
        contig(barcode, 1, "IGH", v, j, iso, cdr3, umis=12)
        contig(barcode, 2, light, lv, lj, light + "C" if light == "IGK" else "IGLC2", lcdr3, umis=8)
        if barcode == "A2-0":                       # a stronger, non-productive heavy contig must be ignored
            contig(barcode, 3, "IGH", "IGHV2-5", "IGHJ1", "IGHM", "CAHRQWLRYFDLW", umis=60, productive=False)
        if barcode == "B2-0":                       # a stronger, low-confidence heavy contig must be ignored
            contig(barcode, 3, "IGH", "IGHV2-70", "IGHJ2", "IGHE", "CARIRGYSYGYW", umis=60, high_confidence=False)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _adata(seed=0):
    cells = _cells()
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"donor": cells["donor"].to_numpy(), "sample": (cells["donor"] + "_s1").to_numpy()},
                       index=cells.index.to_list())
    adata = AnnData(rng.poisson(1.0, size=(len(obs), 20)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(len(obs), 6))
    return adata


def test_read_bcr_reads_a_cell_ranger_contig_file(tmp_path):
    """Guards the 10x branch of tf.read_bcr: one row per cell with its productive, high-confidence chains."""
    # stronger contigs that are non-productive or low-confidence are ignored; barcode prefixes/suffixes applied
    path = _write_10x_csv(tmp_path / "filtered_contig_annotations.csv")
    table = tf.read_bcr(path)
    cells = _cells().dropna(subset=["clone"])
    assert set(table.index) == set(cells.index) and table.index.is_unique
    for barcode, cell in cells.iterrows():
        _, _, v, j, iso, cdr3, _, lv, _, lcdr3 = CLONES[cell["clone"]]
        row = table.loc[barcode]
        assert (row["v_call"], row["j_call"], row["c_call"], row["v_call_light"]) == (v, j, iso, lv)
        assert row["cdr3_light"] == lcdr3
        assert row["cdr3"] == (cdr3 if barcode != "A1-1" else cdr3[:5] + "S" + cdr3[6:])
        assert row["n_contigs"] == 2
    prefixed = tf.read_bcr(str(path), barcode_prefix="s1_", barcode_suffix="-x")
    assert set(prefixed.index) == {f"s1_{b}-x" for b in cells.index}
    with pytest.raises(ValueError, match="Cannot tell the BCR format"):
        tf.read_bcr(tmp_path / "contigs.xlsx")


def test_read_bcr_reads_an_airr_table_with_mutation_frequency(tmp_path):
    """Guards the AIRR branch of tf.read_bcr: best heavy and light chain per cell, with V-region mutation frequency."""
    v = "CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTCACCGGCTAC"
    mutated = v[:40] + ("A" if v[40] != "A" else "G") + v[41:]
    rows = [
        # cell, locus, productive, v_call, junction_aa, umi, sequence_alignment, germline_alignment
        ("c1", "IGH", "T", "IGHV1-2*02", "CARDLWSGYYPEDYW", 9, mutated + "TGTGCG", v + "NNNNNN"),
        ("c1", "IGK", "T", "IGKV1-5*03", "CQQYNSYPWTF", 5, None, None),
        ("c2", "IGH", "T", "IGHV1-2*02", "CARDLWSGYYPEDYW", 7, v + "TGTGCG", v + "NNNNNN"),
        ("c2", "IGH", "F", "IGHV3-23*01", "CAKDRGYSSGWYFDYW", 30, v, v),       # stronger but non-productive
        ("c2", "IGL", "T", "IGLV2-14*01", "CSSYTSSSTLVF", 4, None, None),
    ]
    tab = pd.DataFrame(rows, columns=["cell_id", "locus", "productive", "v_call", "junction_aa", "umi_count",
                                      "sequence_alignment", "germline_alignment"])
    tab["sequence_id"] = [f"{c}_contig_{i}" for i, c in enumerate(tab["cell_id"])]
    tab["j_call"] = np.where(tab["locus"] == "IGH", "IGHJ4*02", "IGKJ1*01")
    tab["junction"] = [_nt(a) for a in tab["junction_aa"]]
    tab["v_sequence_start"] = np.where(tab["locus"] == "IGH", 1, np.nan)
    tab["v_sequence_end"] = np.where(tab["locus"] == "IGH", len(v), np.nan)
    path = tmp_path / "airr_rearrangement.tsv"
    tab.to_csv(path, sep="\t", index=False)
    table = tf.read_bcr(path)
    assert set(table.index) == {"c1", "c2"}
    assert table.loc["c2", "v_call"] == "IGHV1-2*02"
    assert table.loc["c1", "v_call_light"] == "IGKV1-5*03" and table.loc["c2", "v_call_light"] == "IGLV2-14*01"
    assert table.loc["c1", "cdr3"] == "CARDLWSGYYPEDYW"
    assert table.loc["c1", "mutation_frequency"] == pytest.approx(1 / len(v))
    assert table.loc["c2", "mutation_frequency"] == pytest.approx(0.0)


def test_run_calls_clones_from_a_contig_file_path(tmp_path):
    """Guards tf.run(bcr=<path>), the README's main path: Dandelion clones recover the planted clones."""
    # a hypermutated cell stays in its clone, identical receptors in two donors stay apart, cells without
    # contigs get no clone, and the per-cell chain annotations are attached beside the clone ids
    pytest.importorskip("dandelion")
    path = _write_10x_csv(tmp_path / "filtered_contig_annotations.csv")
    adata = _adata()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tf.run(adata, bcr=str(path), basis="X_pca", donor_key="donor", sample_key="sample",
                     n_perm=10, n_boot=5, verbose=False)
    cells = _cells()
    called = adata.obs["clone_id"]
    has = cells["clone"].notna().to_numpy()
    assert called[~has].isna().all()
    assert called[has].notna().all()
    # one called clone per planted clone and vice versa
    pairs = pd.crosstab(called[has].astype(str), cells.loc[has, "clone"])
    assert (pairs.gt(0).sum(axis=0) == 1).all() and (pairs.gt(0).sum(axis=1) == 1).all()
    assert called[cells["donor"].eq("A").to_numpy() & has].astype(str).str.startswith("A|").all()
    assert called["A1-0"] != called["B1-0"]
    # the per-cell annotations come from the same file
    for barcode in ("A1-1", "A2-0", "B2-0", "B3-2"):
        _, _, v, _, iso, _, _, lv, _, _ = CLONES[cells.loc[barcode, "clone"]]
        assert (adata.obs.loc[barcode, "bcr_v_call"], adata.obs.loc[barcode, "bcr_c_call"],
                adata.obs.loc[barcode, "bcr_v_call_light"]) == (v, iso, lv)
    assert res.settings["receptor"] == "BCR"
    assert res.settings["clone_calling"]["method"] == "dandelion"
    assert res.coherence["n_clones"] == 6                     # the six clones with two or more cells
