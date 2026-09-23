import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from threadfin import clones as tf_clones
from threadfin.bcrgraph import bcr_similarity_graph, define_clones

# Planted sequence families: famA (IGHV3-23*01 / IGHJ6*01, <=1 mismatch from a
# 13-aa seed) and famB (IGHV1-69*01 / IGHJ4*01), plus unrelated clones and one
# distant CDR3 inside famA's VJ block.

FAMA = {
    "A0": "CARDYTGNYFDFW",
    "A1": "CARDYTGNYFDFS",
    "A2": "CARDYTGNFFDFW",
    "A3": "CARDYTGNCFDFW",
    "A4": "CARDATGNYFDFW",
    "A5": "CARSYTGNYFDFW",
}
FAMB = {
    "B0": "CSARGGYYGMDVW",
    "B1": "CSARGGYYGMDVY",
    "B2": "CSARGGYFGMDVW",
    "B3": "CSARGGYYGMDVW",
}
UNRELATED = {
    "U0": ("IGHV4-34*01", "IGHJ5*01", "AKRVVAGTTTVVV"),
    "U1": ("IGHV5-51*01", "IGHJ2*01", "CQQQHHHHHHHW"),
    "U2": ("IGHV1-2*01", "IGHJ3*01", "AWWWTTTYYYW"),
    "U3": ("IGHV3-11*01", "IGHJ6*02", "CLLLGGGPPPGGGW"),
}
# inside famA's VJ block but sequence-distant
AX = ("AX", "IGHV3-23*01", "IGHJ6*01", "YYYYYYYYYYYYY")


def _clone_info():
    rows = {}
    for cid, cdr3 in FAMA.items():
        rows[cid] = ("IGHV3-23*01", "IGHJ6*01", cdr3)
    for cid, cdr3 in FAMB.items():
        rows[cid] = ("IGHV1-69*01", "IGHJ4*01", cdr3)
    for cid, (v, j, cdr3) in UNRELATED.items():
        rows[cid] = (v, j, cdr3)
    rows[AX[0]] = AX[1:]
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["v_call", "j_call", "cdr3"]
    )


def _edges(graph, clone_info):
    ids = list(clone_info.index)
    coo = graph.tocoo()
    return {(ids[i], ids[j]) for i, j in zip(coo.row, coo.col)}


def test_graph_edges_only_within_vj_blocks():
    clone_info = _clone_info()
    g = bcr_similarity_graph(clone_info, cdr3_sim_threshold=0.85)
    edges = _edges(g, clone_info)

    assert (g != g.T).nnz == 0  # symmetric
    assert np.allclose(g.diagonal(), 0)

    famA = set(FAMA)
    famB = set(FAMB)
    for i, j in edges:
        same_a = i in famA and j in famA
        same_b = i in famB and j in famB
        assert same_a or same_b, f"cross-block edge {i}-{j}"
    assert ("A0", "A1") in edges
    assert not any(AX[0] in e for e in edges)  # distant CDR3: no edges
    assert not any(u in e for e in edges for u in UNRELATED)


def test_threshold_behavior():
    clone_info = _clone_info()
    # A0 vs A1: 1 mismatch in 13 aa -> sim 12/13 ~ 0.923
    g85 = _edges(bcr_similarity_graph(clone_info, cdr3_sim_threshold=0.85), clone_info)
    g95 = _edges(bcr_similarity_graph(clone_info, cdr3_sim_threshold=0.95), clone_info)
    assert ("A0", "A1") in g85
    assert ("A0", "A1") not in g95


def test_light_chain_weighting():
    ci = pd.DataFrame(
        {
            "v_call": ["IGHV3-23*01"] * 3,
            "j_call": ["IGHJ6*01"] * 3,
            "cdr3": ["CARDYTGNYFDFW"] * 3,
            "cdr3_light": ["AAAAA", "CCCCC", None],
        },
        index=["L0", "L1", "L2"],
    )
    heavy_only = _edges(bcr_similarity_graph(ci), ci)
    assert ("L0", "L1") in heavy_only
    blended = _edges(bcr_similarity_graph(ci, include_light=True), ci)
    # L0/L1 light sim is 0 -> 0.7 < 0.85, edge dropped; L2 lacks light -> kept
    assert ("L0", "L1") not in blended
    assert ("L0", "L2") in blended


def _cell_bcr():
    rows = []
    truth = {}
    for fam, clones in (("A", FAMA), ("B", FAMB)):
        for cid, cdr3 in clones.items():
            v, j = ("IGHV3-23*01", "IGHJ6*01") if fam == "A" else ("IGHV1-69*01", "IGHJ4*01")
            for k in range(3):
                bc = f"{cid}_cell{k}"
                rows.append({"barcode": bc, "clone_id": cid, "v_call": v,
                             "j_call": j, "cdr3": cdr3})
                truth[bc] = fam
    for cid, (v, j, cdr3) in UNRELATED.items():
        rows.append({"barcode": f"{cid}_cell0", "clone_id": cid,
                     "v_call": v, "j_call": j, "cdr3": cdr3})
        truth[f"{cid}_cell0"] = cid
    # cells lacking cdr3 -> NA
    rows.append({"barcode": "nobcr_cell0", "clone_id": "NB", "v_call": "IGHV3-23*01",
                 "j_call": "IGHJ6*01", "cdr3": None})
    rows.append({"barcode": "nobcr_cell1", "clone_id": "NB", "v_call": None,
                 "j_call": None, "cdr3": None})
    return pd.DataFrame(rows).set_index("barcode"), truth


def _check_recovered(labels, truth):
    used = labels.dropna()
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score([truth[c] for c in used.index], used.astype(str))
    assert ari == 1.0


def test_define_clones_connected():
    bcr, truth = _cell_bcr()
    labels = define_clones(bcr, method="connected", cdr3_sim_threshold=0.85)
    assert labels.name == "clone_id_seq"
    assert len(labels) == len(bcr)
    assert labels.loc[["nobcr_cell0", "nobcr_cell1"]].isna().all()
    _check_recovered(labels, truth)


def test_define_clones_leiden():
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    bcr, truth = _cell_bcr()
    labels = define_clones(bcr, method="leiden", resolution=0.5,
                           cdr3_sim_threshold=0.85)
    assert labels.loc[["nobcr_cell0", "nobcr_cell1"]].isna().all()
    _check_recovered(labels, truth)


def _mini_adata():
    rows = [
        # clone, cluster, c_call, mu, timepoint, state
        ("C1", "0", "IGHM", 2, "d4", "GC"),
        ("C1", "0", "IGHM", 3, "d4", "GC"),
        ("C1", "0", "IGHG1", 8, "d7", "GC"),
        ("C1", "0", "IGHG1", 9, "d7", "PB"),
        ("C2", "0", "IGHM", 1, "d4", "GC"),
        ("C2", "0", "IGHM", 4, "d7", "GC"),
        ("C2", "1", "IGHA1", 12, "d14", "PB"),
        ("C3", "1", "IGHG1,IGHG2", 7, "d7", "PB"),
        ("C3", "1", "IGHG1", 6, "d7", "PB"),
        ("C3", "1", "IGHA2", 10, "d14", "PB"),
        ("C4", "1", "IGHD", 0, "d4", "mem"),
        ("C4", "1", "IGHM", 1, "d7", "mem"),
        ("C5", "1", "IGHG1", 5, "d4", "PB"),  # singleton clone
        (None, None, None, np.nan, "d4", "GC"),  # no BCR
    ]
    obs = pd.DataFrame(
        rows, columns=["clone_id", "clone_cluster", "bcr_c_call",
                       "bcr_mu_count", "timepoint", "state"],
        index=[f"cell{i}" for i in range(len(rows))],
    )
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(len(rows), 10)).astype(np.float32)
    var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
    return AnnData(X=x, obs=obs, var=var)


def test_clone_isotype_summary():
    adata = _mini_adata()
    s = tf_clones.clone_isotype_summary(adata)
    assert list(s.columns) == ["clone_cluster", "isotype", "fraction", "n_cells"]

    c0 = s[s["clone_cluster"] == "0"].set_index("isotype")
    assert c0.loc["IGHM", "fraction"] == pytest.approx(4 / 6)
    assert c0.loc["IGHG", "fraction"] == pytest.approx(2 / 6)  # IGHG1 collapsed
    assert "IGHA" not in c0.index  # C2's IGHA cell sits in cluster "1"

    c1 = s[s["clone_cluster"] == "1"].set_index("isotype")
    assert c1.loc["IGHG", "n_cells"] == 3  # incl. "IGHG1,IGHG2" first call
    assert c1.loc["IGHD", "fraction"] == pytest.approx(1 / 7)

    per_clone = tf_clones.clone_isotype_summary(adata, cluster_key=None)
    c3 = per_clone[per_clone["clone_id"] == "C3"].set_index("isotype")
    assert c3.loc["IGHG", "fraction"] == pytest.approx(2 / 3)


def test_clone_shm_summary():
    adata = _mini_adata()
    s = tf_clones.clone_shm_summary(adata)
    assert set(s["level"]) == {"clone"}

    c1 = s[(s["level"] == "clone") & (s["group"] == "C1")].iloc[0]
    assert c1["shm_mean"] == pytest.approx(5.5)
    assert c1["shm_median"] == pytest.approx(5.5)
    assert c1["n_cells"] == 4

    sc = tf_clones.clone_shm_summary(adata, cluster_key="clone_cluster")
    cl0 = sc[(sc["level"] == "cluster") & (sc["group"] == "0")].iloc[0]
    assert cl0["shm_mean"] == pytest.approx((2 + 3 + 8 + 9 + 1 + 4) / 6)
    assert cl0["shm_median"] == pytest.approx((3 + 4) / 2)

    # v_identity fallback: 100 - identity as mutation proxy
    ad2 = _mini_adata()
    ad2.obs["bcr_v_identity"] = 100.0 - ad2.obs["bcr_mu_count"]
    del ad2.obs["bcr_mu_count"]
    s2 = tf_clones.clone_shm_summary(ad2)
    c1b = s2[(s2["level"] == "clone") & (s2["group"] == "C1")].iloc[0]
    assert c1b["shm_mean"] == pytest.approx(5.5)

    ad3 = _mini_adata()
    del ad3.obs["bcr_mu_count"]
    with pytest.raises(ValueError):
        tf_clones.clone_shm_summary(ad3)


def test_clone_fate_table():
    adata = _mini_adata()
    f = tf_clones.clone_fate_table(adata)
    assert list(f.columns) == ["clone_id", "timepoint", "state", "n_cells"]
    assert "C5" not in set(f["clone_id"])  # <2 cells excluded

    got = f.set_index(["clone_id", "timepoint", "state"])["n_cells"]
    assert got.loc[("C1", "d4", "GC")] == 2
    assert got.loc[("C1", "d7", "GC")] == 1
    assert got.loc[("C1", "d7", "PB")] == 1
    assert got.loc[("C2", "d14", "PB")] == 1


def test_community_transition():
    adata = _mini_adata()
    m = tf_clones.community_transition(adata, time_key="timepoint")
    assert m.index.tolist() == ["0", "1"]
    assert m.columns.tolist() == ["0", "1"]
    # d4->d7: C1 0->0, C2 0->0, C4 1->1; d7->d14: C2 0->1, C3 1->1
    assert m.loc["0", "0"] == 2
    assert m.loc["0", "1"] == 1
    assert m.loc["1", "1"] == 2
    assert m.loc["1", "0"] == 0  # C5 has one timepoint -> excluded
