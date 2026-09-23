import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from threadfin.specificity import annotate_specificity, specificity_enrichment

# Reference CDR3 seeds (13 aa). Clones are built around them:
#  CL1 exact hit, CL2 one mismatch (12/13 ~ 0.923 >= 0.85),
#  CL3 same CDR3 as CL1 but a different V gene, CL4 three mismatches
#  (10/13 ~ 0.769 < 0.85), CL5 a V gene absent from the reference.
REF_CDR3_A = "CARDYTGNYFDFW"
REF_CDR3_B = "CSARGGYYGMDVW"

REF = pd.DataFrame(
    {
        "Name": ["mAb-A", "mAb-B", "mAb-C", "mAb-D"],
        "Heavy V Gene": ["IGHV3-23*01", "IGHV3-23", "IGHV1-69*01", "IGHV5-51*01"],
        "Heavy J Gene": ["IGHJ6*01", "IGHJ4*01", "IGHJ4*01", "IGHJ2*01"],
        "CDRH3": [REF_CDR3_A, REF_CDR3_B, "CQQQHHHHHHHW", "AKRVVAGTTTVV"],
        "Binds to": ["SARS-CoV-2 S", "SARS-CoV-2 S; SARS-CoV-2 RBD",
                     "Influenza HA", "EBV gp350"],
    }
)

CLONES = {
    # clone: (v_call, j_call, cdr3, expected specificity or None)
    "CL1": ("IGHV3-23*01", "IGHJ6*01", REF_CDR3_A, "SARS-CoV-2 S"),
    "CL2": ("IGHV3-23*01", "IGHJ4*01", "CSARGGYYGMDVY", "SARS-CoV-2 S"),
    "CL3": ("IGHV1-69*01", "IGHJ6*01", REF_CDR3_A, None),   # right CDR3, wrong V
    "CL4": ("IGHV3-23*01", "IGHJ6*01", "CARDATGNCFDFW", None),  # below identity
    "CL5": ("IGHV4-34*01", "IGHJ5*01", "AKRVVAGTTTVVV", None),  # V not in ref
}


def _spec_adata(cluster_of=None):
    rows = []
    i = 0
    for cid, (v, j, cdr3, _) in CLONES.items():
        cl = (cluster_of or {}).get(cid, "0")
        for _ in range(3):
            rows.append({"clone_id": cid, "clone_cluster": cl,
                         "bcr_v_call": v, "bcr_j_call": j, "bcr_cdr3": cdr3})
            i += 1
    # cells without BCR / without clone
    rows.append({"clone_id": None, "clone_cluster": "0", "bcr_v_call": None,
                 "bcr_j_call": None, "bcr_cdr3": None})
    obs = pd.DataFrame(rows, index=[f"cell{k}" for k in range(len(rows))])
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(len(rows), 5)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{k}" for k in range(5)])
    return AnnData(X=x, obs=obs, var=var)


# ------------------------------ label mode ---------------------------------

def test_label_mode_maps_by_clone():
    adata = _spec_adata()
    labels = pd.DataFrame(
        {"clone_id": ["CL1", "CL2", "CL9"], "s_pos": ["SARS-CoV-2 S", "SARS-CoV-2 S", "x"]}
    )
    out = annotate_specificity(adata, labels=labels, label_col="s_pos")
    assert out is adata
    assert "specificity" in adata.obs.columns

    got = adata.obs.groupby("clone_id", observed=True)["specificity"].first()
    assert got.loc["CL1"] == "SARS-CoV-2 S"
    assert got.loc["CL2"] == "SARS-CoV-2 S"
    assert got.loc[["CL3", "CL4", "CL5"]].isna().all()  # unlabelled clones
    # cells without clone stay NA; CL9 absent from adata is simply unused
    assert adata.obs["specificity"].isna().sum() == 3 * 3 + 1


def test_label_mode_errors():
    adata = _spec_adata()
    labels = pd.DataFrame({"clone_id": ["CL1"], "s_pos": ["S"]})
    with pytest.raises(ValueError, match="label_col"):
        annotate_specificity(adata, labels=labels)
    with pytest.raises(KeyError, match="s_pos"):
        annotate_specificity(adata, labels=labels, label_col="s_pos_wrong")
    with pytest.raises(KeyError, match="clone_id"):
        annotate_specificity(adata, labels=labels.rename(columns={"clone_id": "cl"}),
                             label_col="s_pos")
    with pytest.raises(ValueError, match="exactly one"):
        annotate_specificity(adata)
    with pytest.raises(ValueError, match="exactly one"):
        annotate_specificity(adata, labels=labels, label_col="s_pos", reference=REF)
    with pytest.raises(KeyError, match="nope"):
        annotate_specificity(adata, labels=labels, label_col="s_pos",
                             clone_key="nope")


# ---------------------------- reference mode -------------------------------

def test_reference_mode_matching():
    adata = _spec_adata()
    annotate_specificity(adata, reference=REF, identity=0.85)

    got = adata.obs.groupby("clone_id", observed=True)["specificity"].first()
    for cid, (_, _, _, expected) in CLONES.items():
        if expected is None:
            assert pd.isna(got.loc[cid]), cid
        else:
            assert got.loc[cid] == expected, cid

    tab = adata.uns["specificity_clones"]
    assert list(tab.columns) == ["clone_id", "n_cells", "best_score",
                                 "hit_name", "specificity"]
    assert set(tab["clone_id"]) == {"CL1", "CL2"}
    r1 = tab.set_index("clone_id").loc["CL1"]
    assert r1["hit_name"] == "mAb-A"
    assert r1["best_score"] == pytest.approx(1.0)
    assert r1["n_cells"] == 3
    r2 = tab.set_index("clone_id").loc["CL2"]
    assert r2["hit_name"] == "mAb-B"  # 1 mismatch from REF_CDR3_B
    assert r2["best_score"] == pytest.approx(12 / 13)
    # multi-antigen "Binds to" collapses to the first entry
    assert r2["specificity"] == "SARS-CoV-2 S"


def test_reference_mode_identity_threshold():
    adata = _spec_adata()
    annotate_specificity(adata, reference=REF, identity=0.70)
    got = adata.obs.groupby("clone_id", observed=True)["specificity"].first()
    assert got.loc["CL4"] == "SARS-CoV-2 S"  # 10/13 ~ 0.769 passes now
    assert pd.isna(got.loc["CL3"])           # wrong V still never matches
    with pytest.raises(ValueError, match="identity"):
        annotate_specificity(_spec_adata(), reference=REF, identity=1.5)


def test_reference_mode_from_csv_path(tmp_path):
    adata = _spec_adata()
    path = tmp_path / "covabdab_mini.csv"
    REF.to_csv(path, index=False)
    annotate_specificity(adata, reference=path)
    assert set(adata.uns["specificity_clones"]["clone_id"]) == {"CL1", "CL2"}


def test_reference_mode_ref_cols_override():
    adata = _spec_adata()
    alt = REF.rename(columns={"Heavy V Gene": "v", "CDRH3": "cdr3",
                              "Binds to": "antigen", "Name": "id"})
    alt = alt.drop(columns=["Heavy J Gene"])
    with pytest.raises(KeyError, match="ref_cols"):
        annotate_specificity(adata, reference=alt)
    annotate_specificity(
        adata, reference=alt,
        ref_cols={"v": "v", "cdr3": "cdr3", "specificity": "antigen", "name": "id"},
    )
    tab = adata.uns["specificity_clones"]
    assert set(tab["clone_id"]) == {"CL1", "CL2"}
    assert tab.set_index("clone_id").loc["CL1", "hit_name"] == "mAb-A"


def test_reference_mode_missing_obs_columns():
    adata = _spec_adata()
    del adata.obs["bcr_cdr3"]
    with pytest.raises(KeyError, match="bcr_cdr3"):
        annotate_specificity(adata, reference=REF)
    with pytest.raises(KeyError, match="my_v"):
        annotate_specificity(_spec_adata(), reference=REF, v_col="my_v")


def test_reference_mode_key_added():
    adata = _spec_adata()
    annotate_specificity(adata, reference=REF, key_added="covid_spec")
    assert "covid_spec" in adata.obs.columns
    assert "covid_spec_clones" in adata.uns


# --------------------------- specificity_enrichment ------------------------

def _enrichment_adata():
    """cluster A: 9 SARS-CoV-2 S + 1 other; cluster B: 1 SARS-CoV-2 S + 9 other."""
    rows = []
    for cl, sp, n in [("A", "SARS-CoV-2 S", 9), ("A", "Influenza HA", 1),
                      ("B", "SARS-CoV-2 S", 1), ("B", "Influenza HA", 9)]:
        rows += [{"clone_cluster": cl, "specificity": sp}] * n
    rows.append({"clone_cluster": "A", "specificity": None})
    obs = pd.DataFrame(rows, index=[f"cell{k}" for k in range(len(rows))])
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(len(rows), 5)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{k}" for k in range(5)])
    return AnnData(X=x, obs=obs, var=var)


def test_specificity_enrichment_direction_and_fdr():
    adata = _enrichment_adata()
    out = specificity_enrichment(adata)
    assert list(out.columns) == ["clone_cluster", "specificity", "n_cells",
                                 "fraction", "odds_ratio", "pvalue", "fdr"]

    top = out.iloc[0]
    assert top["clone_cluster"] == "A"
    assert top["specificity"] == "SARS-CoV-2 S"
    assert top["n_cells"] == 9
    assert top["fraction"] == pytest.approx(0.9)
    assert top["odds_ratio"] > 1
    assert top["pvalue"] < 0.05

    depleted = out[(out["clone_cluster"] == "A") &
                   (out["specificity"] == "Influenza HA")].iloc[0]
    assert depleted["odds_ratio"] < 1
    assert depleted["pvalue"] > 0.99

    # BH FDR present, in [0, 1], monotone non-decreasing in sorted p-values
    assert out["pvalue"].is_monotonic_increasing
    assert out["fdr"].is_monotonic_increasing
    assert ((out["fdr"] >= 0) & (out["fdr"] <= 1)).all()


def test_specificity_enrichment_errors():
    adata = _enrichment_adata()
    with pytest.raises(KeyError, match="nope"):
        specificity_enrichment(adata, cluster_key="nope")
    with pytest.raises(KeyError, match="nope"):
        specificity_enrichment(adata, specificity_key="nope")
    empty = adata.copy()
    empty.obs["specificity"] = None
    with pytest.raises(ValueError, match="No cells"):
        specificity_enrichment(empty)
