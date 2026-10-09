#!/usr/bin/env python
"""Reanalyse public LARRY lineage barcodes with Threadfin.

All-day maps are descriptive. Day-2 -> day-6 prediction uses a separate
day-2-only expression model; no future cells or annotations enter that model.
"""
from __future__ import annotations
import gc
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import threadfin as tf
from threadfin.profiles import build_model, variance_components, median_bandwidth, _random_fourier_features

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/scratch/projects/punim1236/threadfin_data/gse140802_larry")
OUT = ROOT/"case_studies/results/clonotrace_larry"
CACHE = ROOT.parent/"internal_validation/paper_673503/larry"
STATES = ["Monocyte", "Neutrophil", "Erythroid", "Megakaryocyte", "Basophil", "Eosinophil",
          "Lymphoid", "Mast", "Dendritic", "Undifferentiated"]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v,"item") else str(v))+"\n")

def load_inputs():
    meta = pd.read_csv(DATA/"stateFate_inVitro_metadata.txt.gz", sep="\t")
    meta.columns = ["library","cell_barcode","day","population","state","well","spring_x","spring_y"]
    meta.index = pd.Index(["larry_"+str(i) for i in range(len(meta))], name="cell_id")
    meta["day"] = meta.day.astype(int)
    meta["well"] = meta.well.astype(str)
    clones = scipy.io.mmread(DATA/"stateFate_inVitro_clone_matrix.mtx.gz").tocsr()
    assert clones.shape[0] == len(meta) and np.isin(clones.data, [0,1]).all()
    nlabels = np.asarray((clones > 0).sum(axis=1)).ravel()
    unique = nlabels == 1
    meta["barcode"] = pd.Series(pd.NA, index=meta.index, dtype="object")
    meta.loc[unique,"barcode"] = ["LARRY_"+str(i) for i in np.asarray(clones[unique].argmax(axis=1)).ravel()]
    meta["clone_day"] = meta.barcode+"|d"+meta.day.astype(str)
    meta["clone_day_well"] = meta.clone_day+"|w"+meta.well
    meta["n_lineage_labels"] = nlabels
    meta.to_csv(OUT/"cell_metadata.csv.gz")
    summary = {"n_input_cells":len(meta), "n_barcodes":clones.shape[1],
               "n_unique_barcode_cells":int(unique.sum()), "n_ambiguous_barcode_cells":int((nlabels>1).sum()),
               "n_unlabelled_cells":int((nlabels==0).sum()), "state_counts":meta.state.value_counts().to_dict(),
               "time_counts":meta.day.value_counts().to_dict(),
               "input_expression":"Author total-count-normalised UMI values; not raw integer counts",
               "source_manifest":str(DATA/"sources.json")}
    dump(OUT/"input_audit.json",summary)
    xp = CACHE/"expression.npz"
    if xp.exists():
        x = sp.load_npz(xp)
    else:
        print("Reading 176.9 million nonzero expression entries", flush=True)
        x = scipy.io.mmread(DATA/"stateFate_inVitro_normed_counts.mtx.gz").tocsr().astype(np.float32)
        sp.save_npz(xp,x)
    genes = pd.read_csv(DATA/"stateFate_inVitro_gene_names.txt.gz",header=None)[0].astype(str).to_numpy()
    assert x.shape == (len(meta),len(genes)) and np.isfinite(x.data).all()
    # Names are author symbols and an exact common order is retained.
    return x,meta,genes,summary

def make_adata(x,obs,genes):
    a=ad.AnnData(x,obs=obs.copy(),var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    return a

def run_profiles(a, key, suffix):
    import umap
    members=a.obs.dropna(subset=[key]).groupby(key,observed=True).agg(
        barcode=("barcode","first"),day=("day","first"),population=("population","first"))
    fractions=pd.crosstab(a.obs[key],a.obs.state,normalize="index").add_prefix("state:")
    records={}
    for rep in ("mean","kernel"):
        tf.tl.clone_profiles(a, clone_key=key, basis="X_threadfin",context_key=None,
                             representation=rep,min_cells=2,random_state=0,verbose=True)
        prof=a.uns["threadfin"]["profiles"]
        tab=prof["clone_table"].join(members).join(fractions)
        feats=prof["features"]
        tab.to_csv(OUT/f"profiles_{suffix}_{rep}.csv.gz")
        feats.to_csv(OUT/f"features_{suffix}_{rep}.csv.gz")
        records[rep]={"icc":float(prof["variance_components"]["icc"]),"n_profiles":len(tab),
                      "n_reliable":int(tab.reliability.ge(.5).sum()),"parameters":prof["params"]}
        if suffix=="all":
            good=tab.index[tab.reliability.ge(.5)]
            xy=umap.UMAP(n_neighbors=15,min_dist=.3,random_state=0,n_jobs=1).fit_transform(feats.loc[good])
            view=tab.loc[good].copy(); view[["x","y"]]=xy
            view.to_csv(OUT/f"clone_umap_{rep}.csv.gz")
    dump(OUT/f"profile_audit_{suffix}.json",records)

def predict_future(early,meta):
    """Identical clone folds and readout model for all representations."""
    early_tab=pd.read_csv(OUT/"profiles_early_mean.csv.gz",index_col=0)
    # Each outcome uses only day-6 cells; an annotation is an observed state,
    # not a complete census of a barcode's latent developmental potential.
    late=meta[(meta.day==6)&meta.barcode.notna()]
    counts=pd.crosstab(late.barcode,late.state)
    ids=early_tab.index.intersection(counts.index[counts.sum(axis=1)>=4])
    if len(ids)<25: raise ValueError(f"Only {len(ids)} early-to-late barcodes")
    counts=counts.loc[ids]
    y=counts.div(counts.sum(axis=1),axis=0)
    y.to_csv(OUT/"day6_observed_state_fractions.csv")
    early_tab.loc[ids].to_csv(OUT/"day2_predictor_coverage.csv")
    un=early.obs.dropna(subset=["barcode"])
    pca=pd.DataFrame(early.obsm["X_threadfin"],index=early.obs_names)
    raw=pca.loc[un.index].groupby(un.barcode).mean().loc[ids]
    # A deliberately simple higher-moment baseline: no smoothing or shrinkage.
    variance=pca.loc[un.index].groupby(un.barcode).var(ddof=0).loc[ids]
    representations={"RNA_mean":raw.to_numpy(),
                     "RNA_mean_variance":np.c_[raw.to_numpy(),variance.to_numpy()],
                     "Size_only":np.log1p(early_tab.loc[ids,["n_cells"]].to_numpy()),
                     "Threadfin_mean":pd.read_csv(OUT/"features_early_mean.csv.gz",index_col=0).loc[ids].to_numpy(),
                     "Threadfin_kernel":pd.read_csv(OUT/"features_early_kernel.csv.gz",index_col=0).loc[ids].to_numpy()}
    scores=[]; predictions=[]; assignments=[]
    for repeat in range(3):
        folds=KFold(5,shuffle=True,random_state=100+repeat)
        for fold,(train,test) in enumerate(folds.split(ids)):
            assignments.extend({"repeat":repeat,"fold":fold,"barcode":ids[j],"role":"test"} for j in test)
            yy=y.to_numpy()
            for method in ["Training_mean"]+list(representations):
                if method=="Training_mean":
                    pred=np.tile(yy[train].mean(axis=0),(len(test),1)); alpha=None
                else:
                    model=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),
                         {"ridge__alpha":[.1,1.,10.,100.]},cv=KFold(3,shuffle=True,random_state=9),
                         scoring="neg_mean_squared_error",n_jobs=1)
                    xx=representations[method]
                    model.fit(xx[train],yy[train])
                    pred=np.clip(model.predict(xx[test]),0,1)
                    row_sum=pred.sum(axis=1)
                    pred[row_sum==0]=yy[train].mean(axis=0)
                    pred/=pred.sum(axis=1,keepdims=True)
                    alpha=model.best_params_["ridge__alpha"]
                for k,state in enumerate(y.columns):
                    scores.append({"repeat":repeat,"fold":fold,"method":method,"target":state,
                                   "n_test":len(test),"mae":mean_absolute_error(yy[test,k],pred[:,k]),
                                   "r2":r2_score(yy[test,k],pred[:,k]) if np.var(yy[test,k])>0 else np.nan,
                                   "alpha":alpha})
                    predictions.extend({"repeat":repeat,"fold":fold,"method":method,"barcode":ids[j],
                                        "target":state,"observed":float(yy[j,k]),"predicted":float(pred[l,k])}
                                       for l,j in enumerate(test))
    pd.DataFrame(scores).to_csv(OUT/"future_scores.csv",index=False)
    pd.DataFrame(predictions).to_csv(OUT/"future_predictions.csv.gz",index=False)
    pd.DataFrame(assignments).to_csv(OUT/"future_folds.csv",index=False)
    dump(OUT/"future_audit.json",{"n_barcodes":len(ids),"n_early_cells":int(early_tab.loc[ids,"n_cells"].sum()),
          "n_late_cells":int(counts.sum().sum()),"early_day":2,"outcome_day":6,
          "min_early_cells":2,"min_late_cells":4,"fold_unit":"lineage barcode, 3 repeats x 5 folds",
          "preprocessing":"Day-2 cells only; transductive unsupervised PCA/profiles; no day-6 cells",
          "readout":"nested 3-fold ridge tuning within each outer training set; same folds for every method",
          "methods":["Training_mean"]+list(representations),
          "RNA_mean_variance":"Concatenated day-2 PC means and per-PC population variances (ddof=0); same ridge grid.",
          "limits":["Replicate folds are correlated, not independent biological replicates.",
                    "Later RNA-defined states are observed clone composition, not experimentally assayed fate potency.",
                    "No independent-donor validation is available in this in-vitro experiment."]})

def split_sample(a):
    """Evaluate shrinkage against disjoint cells, with disjoint calibration clones."""
    obs=a.obs.dropna(subset=["clone_day_well"])
    groups=obs.groupby("clone_day_well",observed=True).indices
    keys=sorted(groups)
    train_keys={k for k in keys if int(hashlib.sha256(k.split("|")[0].encode()).hexdigest()[:8],16)%2==0}
    train=obs.clone_day_well.isin(train_keys).to_numpy()
    basis=np.asarray(a.obsm["X_threadfin"])[a.obs.index.get_indexer(obs.index)]
    center=basis[train].mean(axis=0)
    residual=basis-center
    rng=np.random.default_rng(712)
    bw=median_bandwidth(residual[train],rng=rng)
    z,_=_random_fourier_features(residual,256,bw,np.random.default_rng(713))
    z-=z[train].mean(axis=0)
    codes=pd.factorize(obs.loc[train,"clone_day_well"],sort=True)[0]
    vc=variance_components(z[train],codes)
    evaluations=[(k,idx) for k,idx in groups.items() if k not in train_keys and len(idx)>=16]
    rows=[]
    for k,idx in evaluations:
        for repeat in range(20):
            perm=rng.permutation(idx); half=len(perm)//2
            ref=z[perm[half:]].mean(axis=0)
            for n in (2,4,8):
                raw=z[perm[:n]].mean(axis=0)
                blup=vc.shrinkage(np.array([n]))[0]*raw
                rows.append({"clone_day_well":k,"day":int(obs.iloc[idx[0]].day),
                             "repeat":repeat,"n_sample":n,"n_reference":len(perm)-half,
                             "raw_mse":float(np.mean((raw-ref)**2)),
                             "threadfin_mse":float(np.mean((blup-ref)**2)),
                             "model_reliability":float(vc.reliability([n])[0])})
    pd.DataFrame(rows).to_csv(OUT/"split_sample_errors.csv.gz",index=False)
    dump(OUT/"split_sample_audit.json",{"n_calibration_profiles":len(train_keys),
         "n_evaluation_profiles":len(evaluations),"n_evaluation_rows":len(rows),"kernel_bandwidth":bw,
         "n_features":256,"smoothing":0,"variance_components_from":"disjoint calibration profiles",
         "reference":"disjoint half of each evaluation clone-day-well",
         "limits":"Same-experiment split-cell denoising check, not a calibration of absolute reliability or future fate."})

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    x,meta,genes,summary=load_inputs()
    print("Input audit",json.dumps(summary),flush=True)
    mask=meta.n_lineage_labels.eq(1).to_numpy()
    all_cache=CACHE/"all_embedding.h5ad"
    if all_cache.exists():
        a=ad.read_h5ad(all_cache)
    else:
        a=make_adata(x[mask],meta.loc[mask],genes)
        tf.pp.prepare_embedding(a,batch_key=None,integrate=None,random_state=0,verbose=True)
        small=ad.AnnData(sp.csr_matrix((a.n_obs,1)),obs=a.obs.copy())
        small.obsm["X_threadfin"]=a.obsm["X_threadfin"]
        small.write_h5ad(all_cache);a=small
    run_profiles(a,"clone_day","all")
    split_sample(a)
    early_cache=CACHE/"early_embedding.h5ad"
    if early_cache.exists():
        early=ad.read_h5ad(early_cache)
    else:
        mask=meta.day.eq(2).to_numpy()
        early=make_adata(x[mask],meta.loc[mask],genes)
        tf.pp.prepare_embedding(early,batch_key=None,integrate=None,random_state=0,verbose=True)
        small=ad.AnnData(sp.csr_matrix((early.n_obs,1)),obs=early.obs.copy())
        small.obsm["X_threadfin"]=early.obsm["X_threadfin"]
        small.write_h5ad(early_cache);early=small
    del x;gc.collect()
    run_profiles(early,"barcode","early")
    predict_future(early,meta)
    dump(OUT/"summary.json",{**summary,"status":"complete","all_time_profiles":json.loads((OUT/"profile_audit_all.json").read_text()),
          "future_prediction":json.loads((OUT/"future_audit.json").read_text()),
          "split_sample":json.loads((OUT/"split_sample_audit.json").read_text())})
    print("LARRY analysis complete",flush=True)

if __name__=="__main__": main()
