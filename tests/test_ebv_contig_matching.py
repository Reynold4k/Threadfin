"""Regression guard: preserve EBV GFP-positive sample tags before clone calling."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pandas as pd

_SPEC = spec_from_file_location("case_dandelion_clones", Path(__file__).resolve().parents[1] / "manuscript/case_studies/dandelion_clones.py")
ddl = module_from_spec(_SPEC)
_SPEC.loader.exec_module(ddl)

def test_all_ebv_gex_sample_tags_survive_contig_loading(tmp_path, monkeypatch):
    tags = {
        "GSM9473009": "d0", "GSM9473013": "d4", "GSM9473017": "d7",
        "GSM9473021": "d14-gfp-neg-b", "GSM9473024": "d14_gfp-pos-b",
        "GSM9473028": "d21-gfp-neg-b", "GSM9473031": "d21_gfp-pos-b",
    }
    d = tmp_path / "gse317492_ebv"
    d.mkdir()
    for gsm, tag in tags.items():
        pd.DataFrame({"barcode": ["AAAC-1", "AAAC-1"], "chain": ["IGH", "IGK"],
                      "is_cell": [True, True], "high_confidence": [True, True]}).to_csv(
            d / f"{gsm}_{tag}_vdjB_filtered_contig_annotations.csv.gz", index=False)
    monkeypatch.setattr(ddl, "DATA", tmp_path)
    contigs = ddl.contigs_for("ebv")
    expected = {f"{tag}|AAAC-1" for tag in tags.values()}
    assert set(contigs.cell) == expected
    assert contigs.groupby("cell").size().eq(2).all()
    assert len(contigs) == 14
