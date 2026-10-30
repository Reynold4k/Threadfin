"""Label tests: clone-size strata, sample-level labels, mirrored two-level rows and missing-value levels.

The tests use a small hand-made clone table (no embedding, no programme search), so they run in seconds.
In it, programme P1 holds mostly two-cell clones and P2 mostly larger ones, and the profiles of small clones
are less spread out, as shrinkage makes them in real data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf
from threadfin._utils import bh_fdr
from threadfin.stats import SIZE_BINS, SIZE_STRATA_DEFAULT


def _toy(seed: int = 0, n_per_donor: int = 150) -> AnnData:
    """Two donors x two samples; clones nested in samples; programmes and profile spread follow clone size."""
    rng = np.random.default_rng(seed)
    rows = []
    for donor in ("d1", "d2"):
        for c in range(n_per_donor):
            size = int(rng.choice([2, 3, 4, 6, 12], p=[0.5, 0.15, 0.1, 0.15, 0.1]))
            prog = ("P1" if rng.random() < 0.85 else "P2") if size == 2 else ("P2" if rng.random() < 0.85 else "P1")
            rows.append({"clone": f"{donor}|c{c:03d}", "donor": donor, "sample": f"{donor}_s{c % 2}",
                         "n_cells": size, "programme": prog})
    clones = pd.DataFrame(rows).set_index("clone")
    n = clones["n_cells"].to_numpy(dtype=float)
    cell_clone = np.repeat(clones.index.to_numpy(), clones["n_cells"].to_numpy())
    obs = pd.DataFrame({"clone_id": cell_clone, "donor": clones.loc[cell_clone, "donor"].to_numpy(),
                        "sample": clones.loc[cell_clone, "sample"].to_numpy()},
                       index=[f"cell{i}" for i in range(cell_clone.size)])
    ad = AnnData(X=np.zeros((obs.shape[0], 1), dtype=np.float32), obs=obs)
    table = pd.DataFrame({"n_cells": clones["n_cells"].to_numpy(), "reliability": n / (n + 20.0),
                          "donor": clones["donor"].to_numpy(), "clone_programme": clones["programme"].to_numpy()},
                         index=pd.Index(clones.index.astype(str), name="clone_id"))
    ad.uns["threadfin"] = {"profiles": {"clone_table": table,
                                        "params": {"clone_key": "clone_id", "context_key": "sample",
                                                   "donor_key": "donor"}}}
    _draw_profiles(ad, rng)
    return ad


def _draw_profiles(ad: AnnData, rng) -> None:
    """Profiles with no clonal signal, shrunk as with little clonal variance (reliability n / (n + 20)):
    the profiles of small clones are less spread out."""
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    n = table["n_cells"].to_numpy(dtype=float)
    feats = rng.normal(size=(n.size, 20)) * np.sqrt(n / (n + 20.0))[:, None]
    ad.uns["threadfin"]["profiles"]["features"] = pd.DataFrame(
        feats, index=table.index, columns=[f"kpc{j}" for j in range(feats.shape[1])])


def _set_clone_label(ad: AnnData, values, name: str = "lab") -> None:
    """Give every cell its clone's value of a clone-level label."""
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    ad.obs[name] = ad.obs["clone_id"].astype(str).map(pd.Series(values, index=table.index)).to_numpy()
    if not pd.api.types.is_numeric_dtype(ad.obs[name]):
        ad.obs[name] = ad.obs[name].astype(object)


def test_defaults_are_named_constants():
    assert SIZE_BINS == (1, 2, 3, 5, 10)
    assert set(SIZE_STRATA_DEFAULT) == {"association_test", "profile_association"}
    ad = _toy()
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    _set_clone_label(ad, np.where(np.random.default_rng(1).random(n.size) < 0.3, "B", "A"))
    res = tf.tl.association_test(ad, "lab", n_perm=19, verbose=False)
    prof = tf.tl.profile_association(ad, "lab", n_perm=19, verbose=False)
    assert (res.attrs["size_bins"] is not None) == SIZE_STRATA_DEFAULT["association_test"]
    assert (prof["size_bins"] is not None) == SIZE_STRATA_DEFAULT["profile_association"]


def test_size_strata_restore_calibration_for_a_label_that_follows_clone_size():
    """The label depends on clone size only; programmes depend on clone size: shuffling within donors reads the
    label as a programme effect, shuffling within donor x clone-size bins does not."""
    ad = _toy()
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    rng = np.random.default_rng(2)
    rejected = {False: 0, True: 0}
    for k in range(30):
        _set_clone_label(ad, np.where(rng.random(n.size) < np.where(n == 2, 0.15, 0.55), "B", "A"))
        for size_strata in (False, True):
            res = tf.tl.association_test(ad, "lab", size_strata=size_strata, n_perm=199, random_state=k,
                                         verbose=False)
            rejected[size_strata] += bool(res["fdr"].min() < 0.05)
    assert rejected[False] >= 25
    assert rejected[True] <= 5


def test_size_strata_restore_calibration_of_the_profile_test():
    """Profile spread grows with clone size, so a numeric label that grows with clone size 'explains' profile
    variance when shuffled within donors only (Anderson & Walsh 2013); not within clone-size bins."""
    ad = _toy()
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    rng = np.random.default_rng(3)
    rejected = {False: 0, True: 0}
    for k in range(40):                                       # new profiles and a new label every time
        _draw_profiles(ad, rng)
        _set_clone_label(ad, np.log2(n) + rng.normal(0, 0.5, size=n.size))
        for size_strata in (False, True):
            res = tf.tl.profile_association(ad, "lab", size_strata=size_strata, n_perm=99, random_state=k,
                                            verbose=False)
            rejected[size_strata] += bool(res["p_value"] <= 0.05)
    assert rejected[False] >= 8                               # 12 of 40 with these seeds
    assert rejected[True] <= 5                                # 2 of 40


def test_size_strata_keep_the_power_for_a_programme_effect():
    ad = _toy()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    in_p1 = table["clone_programme"].eq("P1").to_numpy()
    rng = np.random.default_rng(4)
    _set_clone_label(ad, np.where(rng.random(in_p1.size) < np.where(in_p1, 0.55, 0.2), "B", "A"))
    feats = ad.uns["threadfin"]["profiles"]["features"]
    for size_strata in (False, True):
        res = tf.tl.association_test(ad, "lab", size_strata=size_strata, n_perm=199, verbose=False)
        assert res["fdr"].max() < 0.01
        _set_clone_label(ad, feats["kpc0"].to_numpy() + rng.normal(0, 0.5, size=in_p1.size), "state")
        prof = tf.tl.profile_association(ad, "state", size_strata=size_strata, n_perm=199, verbose=False)
        assert prof["p_value"] < 0.01


def test_clone_sizes_are_reported():
    ad = _toy()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    rng = np.random.default_rng(5)
    _set_clone_label(ad, np.where(rng.random(len(table)) < 0.4, "B", "A"))
    res = tf.tl.association_test(ad, "lab", n_perm=19, verbose=False)
    for prog, g in table.groupby("clone_programme"):
        row = res[res["programme"] == prog].iloc[0]
        assert row["median_cells_programme"] == g["n_cells"].median()
        assert row["frac_2cell_programme"] == pytest.approx((g["n_cells"] == 2).mean())
    prof = tf.tl.profile_association(ad, "lab", n_perm=19, verbose=False)
    sizes = pd.DataFrame(prof["clone_size"]).set_index("level")
    assert sizes["n_clones"].sum() == prof["n_clones"]
    assert {"median_cells", "frac_2cell"} <= set(sizes.columns)


def test_a_label_set_by_clone_size_is_refused_with_size_strata_and_tested_without():
    ad = _toy()
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    _set_clone_label(ad, np.where(n >= 5, "expanded", "small"))
    with pytest.raises(ValueError, match="size_strata=False"):
        tf.tl.association_test(ad, "lab", size_strata=True, n_perm=19, verbose=False)
    with pytest.raises(ValueError, match="size_strata=False"):
        tf.tl.profile_association(ad, "lab", size_strata=True, n_perm=19, verbose=False)
    res = tf.tl.association_test(ad, "lab", size_strata=False, n_perm=19, verbose=False)
    assert (~res["mirror"]).sum() == 2                        # one row per programme for the rarer level


def test_a_donor_level_label_is_refused_with_the_same_message_with_or_without_size_strata():
    ad = _toy()
    ad.obs["age"] = ad.obs["donor"].map({"d1": "old", "d2": "young"}).astype(object)
    for size_strata in (False, True):
        with pytest.raises(ValueError, match="does not vary among the clones of any single 'donor'"):
            tf.tl.association_test(ad, "age", size_strata=size_strata, n_perm=19, verbose=False)
        with pytest.raises(ValueError, match="does not vary among the clones of any single 'donor'"):
            tf.tl.profile_association(ad, "age", size_strata=size_strata, n_perm=19, verbose=False)


def test_size_strata_accept_bins_and_refuse_nonsense():
    ad = _toy()
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    _set_clone_label(ad, np.where(np.random.default_rng(6).random(n.size) < 0.3, "B", "A"))
    res = tf.tl.association_test(ad, "lab", size_strata=(2, 5), n_perm=19, verbose=False)
    assert res.attrs["size_bins"] == [1, 2, 5]
    for bad in [(), (0, 2), (5, 2), (2.5,), "2", (True, 3)]:
        with pytest.raises(ValueError, match="size_strata must be"):
            tf.tl.association_test(ad, "lab", size_strata=bad, n_perm=19, verbose=False)


def test_a_sample_level_label_is_tested_with_a_warning_and_a_flag():
    ad = _toy()
    ad.obs["tissue"] = np.where(ad.obs["sample"].str.endswith("s0"), "LN", "blood").astype(object)
    with pytest.warns(UserWarning, match="compares 4 samples, not independent clones"):
        res = tf.tl.association_test(ad, "tissue", n_perm=19, verbose=False)
    assert res.attrs["sample_level"] and res.attrs["n_samples"] == 4 and res.attrs["sample_key"] == "sample"
    with pytest.warns(UserWarning, match="compares 4 samples"):
        prof = tf.tl.profile_association(ad, "tissue", n_perm=19, verbose=False)
    assert prof["sample_level"] and prof["n_samples"] == 4
    assert prof["samples_per_level"] == [{"level": "LN", "n_samples": 2}, {"level": "blood", "n_samples": 2}]
    # a label that varies within samples is not flagged; sample_key=None skips the check
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    _set_clone_label(ad, np.where(np.random.default_rng(7).random(n.size) < 0.3, "B", "A"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prof = tf.tl.profile_association(ad, "lab", n_perm=19, verbose=False)
        assert not prof["sample_level"] and prof["n_samples"] == 4 and prof["samples_per_level"] is None
        prof = tf.tl.profile_association(ad, "tissue", sample_key=None, n_perm=19, verbose=False)
        assert not prof["sample_level"] and prof["n_samples"] is None


def test_results_with_the_new_fields_can_be_concatenated_and_written(tmp_path):
    ad = _toy()
    ad.obs["tissue"] = np.where(ad.obs["sample"].str.endswith("s0"), "LN", "blood").astype(object)
    n = ad.uns["threadfin"]["profiles"]["clone_table"]["n_cells"].to_numpy()
    _set_clone_label(ad, np.where(np.random.default_rng(8).random(n.size) < 0.3, "B", "A"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = [tf.tl.association_test(ad, lab, n_perm=19, verbose=False) for lab in ("tissue", "lab")]
        for lab in ("tissue", "lab"):
            tf.tl.profile_association(ad, lab, n_perm=19, verbose=False)
    assert len(pd.concat(tables, ignore_index=True)) == sum(len(t) for t in tables)
    ad.write_h5ad(tmp_path / "toy.h5ad")


def test_a_two_level_label_marks_its_mirrored_rows():
    """The rows of the more common level of a two-level label mirror those of the rarer level (same p-value,
    inverse odds ratio): they are flagged, and the FDR is the same as over one row per programme."""
    ad = _toy()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    n_prog = table["clone_programme"].nunique()
    rng = np.random.default_rng(9)
    for rare, common in (("pos", "neg"), ("A", "B")):
        _set_clone_label(ad, np.where(rng.random(len(table)) < 0.3, rare, common))
        res = tf.tl.association_test(ad, "lab", n_perm=99, verbose=False)
        assert len(res) == 2 * n_prog
        assert set(res.loc[res["mirror"], "level"]) == {common} and set(res.loc[~res["mirror"], "level"]) == {rare}
        kept = res[~res["mirror"]].set_index("programme")
        mirrored = res[res["mirror"]].set_index("programme").loc[kept.index]
        assert np.array_equal(kept["pvalue"].to_numpy(), mirrored["pvalue"].to_numpy())
        assert np.allclose(kept["odds_ratio"].to_numpy() * mirrored["odds_ratio"].to_numpy(), 1.0)
        assert np.allclose(kept["fdr"], bh_fdr(kept["pvalue"].to_numpy()))
    _set_clone_label(ad, rng.choice(["a", "b", "c"], size=len(table)))
    res = tf.tl.association_test(ad, "lab", n_perm=19, verbose=False)
    assert len(res) == 3 * n_prog and not res["mirror"].any()


def test_numeric_effects_are_compared_with_their_expectation_within_strata():
    from scipy.stats import kruskal

    ad = _toy()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    values = np.random.default_rng(10).normal(size=len(table))
    _set_clone_label(ad, values)
    res = tf.tl.association_test(ad, "lab", strata_key=None, size_strata=False, n_perm=99, verbose=False)
    assert np.allclose(res["expected_rank_effect"], 0.0)            # one stratum: nothing expected
    groups = [values[table["clone_programme"].eq(p).to_numpy()] for p in sorted(table["clone_programme"].unique())]
    assert res.attrs["kruskal_h"] == pytest.approx(kruskal(*groups).statistic)
    # within donor x clone-size bins, the programme mostly of two-cell clones is expected to rank low for a
    # label that grows with clone size
    _set_clone_label(ad, np.log2(table["n_cells"].to_numpy()) + values)
    res = tf.tl.association_test(ad, "lab", n_perm=99, verbose=False).set_index("programme")
    assert res.loc["P1", "expected_rank_effect"] < -0.1 < 0.1 < res.loc["P2", "expected_rank_effect"]


def test_levels_that_look_like_missing_values_are_named_once():
    ad = _toy()
    table = ad.uns["threadfin"]["profiles"]["clone_table"]
    values = np.random.default_rng(11).choice(["IgM", "IgG", "nan", "."], size=len(table), p=[0.5, 0.3, 0.1, 0.1])
    _set_clone_label(ad, values)
    with pytest.warns(UserWarning, match="look like a missing value") as record:
        labels = tf.tl.clone_labels(ad, "lab")
    hits = [w for w in record if "missing value" in str(w.message)]
    assert len(hits) == 1
    message = str(hits[0].message)
    n_nan = int((ad.obs["lab"] == "nan").sum())
    assert f"'nan' ({n_nan:,} cells)" in message and "'.' (" in message and "to NaN" in message
    assert {"nan", "."} <= set(labels.dropna())                      # the result is unchanged
    ad.obs["clean"] = ad.obs["lab"].replace({"nan": "IgA", ".": "IgA"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tf.tl.clone_labels(ad, "clean")
