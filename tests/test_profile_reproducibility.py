"""Regressions for feature-map provenance, AIRR alignment and matched inference."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf
from threadfin.api import _read_airr_cells
from threadfin.profiles import VarianceComponents, model_from_adata, project_features
from threadfin.programmes import split_test

def repertoire():
    rng=np.random.default_rng(17)
    obs=pd.DataFrame({"clone_id":np.repeat([f"c{k}" for k in range(8)],10)},
                     index=[f"cell{k}" for k in range(80)])
    a=AnnData(np.zeros((80,1)),obs=obs)
    a.obsm["X_pca"]=rng.normal(size=(80,5))+np.repeat(rng.normal(size=(8,5)),10,axis=0)
    return a

@pytest.mark.parametrize("bandwidth",["median","local",1.7])
@pytest.mark.parametrize("smooth",[0,5])
def test_kernel_reconstruction_and_legacy_recovery_share_original_coordinates(bandwidth,smooth):
    a=repertoire()
    tf.tl.clone_profiles(a,representation="kernel",n_features=32,n_components=5,
                         bandwidth=bandwidth,smooth=smooth,verbose=False)
    original=a.uns["threadfin"]["profiles"]["features"].to_numpy().copy()
    for legacy in [False,True]:
        if legacy:
            del a.uns["threadfin"]["profiles"]["params"]["feature_rng_state"]
        m=model_from_adata(a)
        blup,_,_=m.group_blups(m.clone_codes,len(m.clone_index))
        np.testing.assert_allclose(project_features(a,blup),original,rtol=1e-10,atol=1e-10)
        unsmoothed=model_from_adata(a,unsmoothed=True)
        assert unsmoothed.params["feature_rng_state"]==m.params["feature_rng_state"]

@pytest.mark.parametrize("legacy_omitted_none", [False, True])
def test_kernel_provenance_survives_anndata_roundtrip(tmp_path, legacy_omitted_none):
    import anndata
    a=repertoire()
    tf.tl.clone_profiles(a,representation="kernel",n_features=24,n_components=4,smooth=0,verbose=False)
    path=tmp_path/"model.h5ad"
    a.write_h5ad(path)
    b=anndata.read_h5ad(path)
    if legacy_omitted_none:
        # Older AnnData writers drop None-valued parameters from HDF5.
        b.uns["threadfin"]["profiles"]["params"].pop("context_key", None)
        b.uns["threadfin"]["profiles"]["params"].pop("feature_rng_state", None)
    m=model_from_adata(b)
    blup,_,_=m.group_blups(m.clone_codes,len(m.clone_index))
    np.testing.assert_allclose(project_features(b,blup),b.uns["threadfin"]["profiles"]["features"],atol=1e-10)

def test_legacy_model_refuses_changed_embedding():
    a=repertoire()
    tf.tl.clone_profiles(a,representation="kernel",n_features=24,smooth=0,verbose=False)
    del a.uns["threadfin"]["profiles"]["params"]["feature_rng_state"]
    del a.uns["threadfin"]["profiles"]["params"]["input_sha256"]
    a.obsm["X_pca"]*=2
    with pytest.raises(ValueError,match="cannot be reproduced"):
        model_from_adata(a)

def test_reported_capture_requirement_matches_actual_weighted_reliability():
    vc=VarianceComponents(np.array([100.,.1]),np.array([1.,9.]),1.,2,4,np.zeros(2))
    assert vc.min_cells_for(.5)==1
    for target in [.5,.8,.9,.99]:
        n=vc.min_cells_for(target)
        assert vc.reliability([n])[0]>=target
        if n>1:assert vc.reliability([n-1])[0]<target
    with pytest.raises(ValueError,match="target"):
        vc.min_cells_for(1.)

def test_airr_umi_sort_keeps_chain_identity_aligned(tmp_path):
    f=tmp_path/"input.tsv"
    pd.DataFrame({"cell_id":["A","A","B","B"],"locus":["IGH","IGK","IGH","IGL"],
                  "umi_count":[1,100,50,2],"v_call":["IGHV1","IGKV1","IGHV2","IGLV1"],
                  "j_call":["IGHJ1","IGKJ1","IGHJ2","IGLJ1"],
                  "junction":["AAA","CCC","GGG","TTT"]}).to_csv(f,sep="\t",index=False)
    got=_read_airr_cells(f)
    assert got.v_call.to_dict()=={"A":"IGHV1","B":"IGHV2"}
    assert got.v_call_light.to_dict()=={"A":"IGKV1","B":"IGLV1"}

def test_profiles_reject_shared_clone_ids_across_donors():
    a=repertoire()
    a.obs["donor"]=np.tile(["A","B"],40)
    with pytest.raises(ValueError,match="multiple donors"):
        tf.tl.clone_profiles(a,donor_key="donor",verbose=False)

def test_memory_does_not_borrow_other_donors_when_matching_is_impossible():
    rng=np.random.default_rng(3)
    obs=pd.DataFrame({"clone_id":np.repeat(["a","b","c"],6),
                     "donor":np.repeat(["A","B","C"],6),
                     "time":np.tile(np.repeat(["d0","d1"],3),3)},
                     index=[f"x{i}" for i in range(18)])
    a=AnnData(np.zeros((18,1)),obs=obs)
    a.obsm["X_pca"]=rng.normal(size=(18,3))+np.repeat(np.arange(3)*5,6)[:,None]
    tf.tl.clone_profiles(a,donor_key="donor",verbose=False)
    with pytest.raises(ValueError,match="same donor"):
        tf.tl.clonal_memory(a,"time",min_pairs=2,n_null=5,n_boot=5,verbose=False)

def test_identical_profiles_are_not_a_significant_split():
    got=split_test(np.zeros((12,3)),n_null=10,random_state=4)
    assert got["pvalue"]==got["pvalue_empirical"]==1.

def test_numpy_alignment_penalises_terminal_gaps(monkeypatch):
    import threadfin.sequence as sequence
    monkeypatch.setattr(sequence, "_parasail", None)
    assert sequence._nw_score("ACCCCC", "CCCCC") == 37
    assert sequence._nw_score("CCCCC", "ACCCCC") == 37
    assert sequence._nw_score("AACCCCC", "CCCCC") == 36
    assert sequence._nw_score("CCCCC", "CCCCC") == 45


def test_negative_centroid_weights_are_rejected():
    import anndata as ad
    import numpy as np
    import pytest
    import threadfin as tf
    a = ad.AnnData(np.ones((4, 2)))
    a.obsm["X_pca"] = np.arange(8).reshape(4, 2)
    a.obs["clone_id"] = ["a", "a", "b", "b"]
    a.obs["weight"] = [-1., 2., 1., 1.]
    with pytest.raises(ValueError, match="weights"):
        tf.clone_centroids(a, basis="X_pca", weight_col="weight")


@pytest.mark.parametrize("change", ["embedding", "clone", "subset", "donor"])
def test_saved_profile_rejects_changed_input(change):
    from threadfin.profiles import clone_profiles, model_from_adata
    a = repertoire()
    a.obs["donor"] = "d0"
    clone_profiles(a, representation="kernel", smooth=0, donor_key="donor", verbose=False)
    if change == "embedding":
        a.obsm["X_pca"][0, 0] += 1
    elif change == "clone":
        a.obs["clone_id"] = a.obs["clone_id"].astype(object)
        a.obs.iloc[0, a.obs.columns.get_loc("clone_id")] = "changed"
    elif change == "donor":
        a.obs["donor"] = "changed"
    else:
        a = a[:-1].copy()
    with pytest.raises(ValueError, match="changed since"):
        model_from_adata(a)
