"""Design guards for GSE253857; these do not require the downloaded matrices."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = spec_from_file_location("case_study_datasets", _ROOT / "case_studies" / "datasets.py")
datasets = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(datasets)


def test_bone_marrow_manifest_covers_all_paired_library_designs():
    manifest = datasets._bone_marrow_sample_manifest()
    assert set(manifest.index) == {
        "BM_6_PC_Bmem", "BM_Bsm_2", "BM_Bsm_3", "BM_PCs-Bmem_2",
        "BM_PCs-Bmem_3_555", "BM_PCs-Bmem_3_556-blood555", "BM_PCs_2",
        "BM_PCs_3", "BM_PCs_Bmem_4", "BM_PCs_Bmem_5", "BM_Spike_2",
        "BM_Spike_3", "BM_Spike_4", "BM_Spike_Pool1", "BM_Tetanus_3",
        "BM_Tetanus_4", "BM_blood_Spike_5", "Blood6_PCs_Bmem",
        "Blood_Bsm_2", "Blood_Bsm_3",
    }
    assert manifest.loc["BM_6_PC_Bmem", "donor"] == "561"
    assert manifest.loc["BM_Spike_Pool1", "design_status"] == "pooled_donors"
    assert manifest.loc["BM_PCs-Bmem_3_556-blood555", "design_status"] == "mixed_donors"
    assert manifest.loc["BM_blood_Spike_5", "design_status"] == "mixed_tissues"
    assert manifest.loc["BM_Tetanus_4", "antigen"] == "tetanus toxoid"


def test_bone_marrow_donor_labels_are_manifest_backed_and_never_pseudo_donors():
    manifest = datasets._bone_marrow_sample_manifest()
    assert datasets._bmpc_donor("BM_6_PC_Bmem", manifest) == "donor 561"
    assert datasets._bmpc_donor("BM_Spike_Pool1", manifest) == "pool:BM_Spike_Pool1"
    assert datasets._bmpc_donor("BM_PCs-Bmem_3_556-blood555", manifest).startswith("excluded:")
    with pytest.raises(KeyError, match="exact GEO SOFT mapping"):
        datasets._bmpc_donor("BM_unknown", manifest)
