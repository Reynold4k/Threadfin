#!/usr/bin/env python
"""Descriptive and matched-null analysis of longitudinal exact paired TCRs."""
from __future__ import annotations
import json
import re
from itertools import combinations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import threadfin as tf
from prepare_clonotrace_nsclc import OUT, PRIVATE, read_export, export_r, paired_tcr

MODULES = {"cytotoxic":["NKG7","PRF1","GZMB","GNLY","CTSW"],
           "memory":["IL7R","CCR7","TCF7","LEF1","LTB"],
           "activation":["IFNG","FOS","EGR1","CD69","TNFRSF9"],
           "cycling":["MKI67","TOP2A","TYMS","STMN1"]}

def cycle_number(value):
    text=str(value)
    m=re.search(r"(?:cycle|cy|c)[_ -]*([0-9]+)",text,re.I)
    if m: return int(m.group(1))
    m=re.search(r"[_ -]([0-9]+)$",text)
    if m: return int(m.group(1))
    if text.isdigit(): return int(text)
    raise ValueError("Unrecognised cycle label: "+text)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v))+"\n")

def prepare():
    cache=PRIVATE/"analysed_embedding.h5ad"
    if cache.exists():
        return ad.read_h5ad(cache),json.loads((OUT/"input_audit.json").read_text())
    x,cells,genes,meta=read_export(export_r())
    assert x.shape == (len(meta),len(genes)) and meta.index.is_unique
    meta["tcr_pair"]=meta.cdr3s_aa.map(paired_tcr)
    meta["cycle_raw"]=meta.PtCycle.astype(str)
    meta["cycle"]=meta.PtCycle.map(cycle_number)
    meta["clone_biological"]=pd.Series(pd.NA,index=meta.index,dtype="object")
    matched=meta.tcr_pair.notna()
    meta.loc[matched,"clone_biological"]=meta.loc[matched,"patient"].astype(str)+"|"+meta.loc[matched,"tcr_pair"]
    meta["clone_cycle"]=meta.clone_biological+"|c"+meta.cycle.astype(str)
    key_columns=["patient","clonotype_id"]
    audit=meta[matched].groupby(key_columns).tcr_pair.nunique().rename("n_exact_pairs").reset_index()
    audit.to_csv(OUT/"author_clonotype_exact_pair_audit.csv",index=False)
    if "scRXN" in meta:
        meta[matched].groupby(["patient","scRXN","clonotype_id"]).tcr_pair.nunique().rename("n_exact_pairs").to_csv(
            OUT/"author_clonotype_within_library_audit.csv")
    meta.to_csv(OUT/"exported_cell_metadata.csv.gz")
    counts=meta.groupby(["patient","cycle"]).agg(exported_cells=("patient","size"),paired_cells=("tcr_pair","count"))
    counts.to_csv(OUT/"patient_cycle_cell_counts.csv")
    keep=matched.to_numpy()
    a=ad.AnnData(x[keep],obs=meta.loc[keep].copy(),var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    tf.pp.prepare_embedding(a,batch_key=None,integrate=None,random_state=0,verbose=True)
    module_audit={}
    for label,names in MODULES.items():
        found=[g for g in names if g in a.var_names]
        if len(found)<3: raise ValueError(f"Insufficient {label} genes: {found}")
        z=a.layers["log_norm"][:,a.var_names.get_indexer(found)]
        z=z.toarray() if sp.issparse(z) else np.asarray(z)
        z=(z-z.mean(axis=0))/np.maximum(z.std(axis=0),1e-8)
        a.obs["module:"+label]=z.mean(axis=1)
        module_audit[label]=found
    small=ad.AnnData(sp.csr_matrix((a.n_obs,1)),obs=a.obs.copy())
    small.obsm["X_threadfin"]=a.obsm["X_threadfin"]
    small.write_h5ad(cache)
    audit={"n_exported_tcr_annotated_cells":len(meta),"n_cells_exact_paired_tcr":int(matched.sum()),
           "n_patients":int(meta.patient.nunique()),"n_common_genes":len(genes),
           "cycle_labels":meta[["cycle_raw","cycle"]].drop_duplicates().to_dict("records"),
           "modules":module_audit,"embedding":"Global 30-PC receptor-excluded RNA; no patient/cycle integration",
           "profile_context":"patient; not patient x cycle, so longitudinal changes are retained",
           "cohort_scope":"All 10 public patient objects; not the paper's selected 8-patient response comparison",
           "limits":["No audited clinical-response mapping; no efficacy or responder claims.",
                     "Exact paired TRA/TRB amino-acid identity is used within patient.",
                     "Module scores reuse expression and are not orthogonal validation.",
                     "Longitudinal distribution changes do not establish cellular differentiation."]}
    dump("input_audit.json",audit)
    return small,audit

def profile(a):
    import umap
    agg={"patient":("patient","first"),"cycle":("cycle","first"),
         "biological_clone":("clone_biological","first"),"author_clonotype_id":("clonotype_id","first")}
    agg.update({key:("module:"+key,"mean") for key in MODULES})
    members=a.obs.groupby("clone_cycle",observed=True).agg(**agg)
    raw=pd.DataFrame(a.obsm["X_threadfin"],index=a.obs_names).groupby(a.obs.clone_cycle).mean()
    raw.to_csv(OUT/"features_RNA_mean.csv.gz")
    records={}
    for rep in ("mean","kernel"):
        tf.tl.clone_profiles(a,clone_key="clone_cycle",basis="X_threadfin",context_key="patient",
                            donor_key="patient",representation=rep,min_cells=2,random_state=0,verbose=True)
        saved=a.uns["threadfin"]["profiles"]
        table=saved["clone_table"].join(members)
        features=saved["features"]
        table.to_csv(OUT/f"profiles_{rep}.csv.gz")
        features.to_csv(OUT/f"features_{rep}.csv.gz")
        ids=table.index[table.reliability.ge(.5)]
        xy=umap.UMAP(n_neighbors=15,min_dist=.3,random_state=0,n_jobs=1).fit_transform(features.loc[ids])
        coords=table.loc[ids].copy();coords[["x","y"]]=xy
        coords.to_csv(OUT/f"clone_umap_{rep}.csv.gz")
        records[rep]={"n_profiles":len(table),"n_reliable":len(ids),
                      "icc":float(saved["variance_components"]["icc"]),"parameters":saved["params"]}
    dump("profile_audit.json",records)
    # A deterministic cell subsample only affects the descriptive cell display.
    ids=np.random.default_rng(19).choice(a.n_obs,min(30000,a.n_obs),replace=False)
    xy=umap.UMAP(n_neighbors=15,min_dist=.3,random_state=0,n_jobs=1).fit_transform(a.obsm["X_threadfin"][ids])
    cell=a.obs.iloc[ids].copy();cell[["x","y"]]=xy
    cell.to_csv(OUT/"cell_umap.csv.gz")
    return table,records

def paired_comparison(table):
    features={"RNA_mean":pd.read_csv(OUT/"features_RNA_mean.csv.gz",index_col=0),
              "Threadfin_mean":pd.read_csv(OUT/"features_mean.csv.gz",index_col=0),
              "Threadfin_kernel":pd.read_csv(OUT/"features_kernel.csv.gz",index_col=0)}
    # Identical coverage for all methods. Reliability is not used to select pairs.
    t=table[table.n_cells.ge(4)].copy()
    linked=t.groupby(["patient","biological_clone"]).filter(lambda z:z.cycle.nunique()>=2)
    linked.to_csv(OUT/"longitudinal_clone_cycle_links.csv.gz")
    rows=[]; nulls=[]; changes=[]
    rng=np.random.default_rng(30)
    for patient,pt in t.groupby("patient"):
        cycles=sorted(pt.cycle.unique())
        for c1,c2 in zip(cycles[:-1],cycles[1:]):
            x=pt[pt.cycle.eq(c1)].set_index("biological_clone",drop=False)
            y=pt[pt.cycle.eq(c2)].set_index("biological_clone",drop=False)
            common=x.index.intersection(y.index)
            if len(common)<3: continue
            x=x.loc[common]; y=y.loc[common]
            # Preserve exact observed source and target pools, stratify target
            # permutations by log2 capture-count bin, and exclude fixed points.
            bins=np.floor(np.log2(y.n_cells)).astype(int).to_numpy()
            eligible=np.array([np.sum(bins==b)>=2 for b in bins])
            x=x.iloc[eligible]; y=y.iloc[eligible]; bins=bins[eligible]
            if len(x)<3: continue
            ix=x["clone_cycle"].to_numpy() if "clone_cycle" in x else pt.index[pt.cycle.eq(c1)]
            # Original profile IDs are deterministic controlled keys.
            ix=(x.biological_clone.astype(str)+"|c"+str(int(c1))).to_numpy()
            iy=(y.biological_clone.astype(str)+"|c"+str(int(c2))).to_numpy()
            permutations=[]
            for it in range(200):
                perm=np.arange(len(x))
                for b in np.unique(bins):
                    ids=np.flatnonzero(bins==b)
                    order=rng.permutation(ids)
                    perm[order]=np.roll(order,int(rng.integers(1,len(order))))
                assert np.all(perm!=np.arange(len(x)))
                permutations.append(perm)
            for method,f in features.items():
                d=cdist(f.loc[ix],f.loc[iy],metric="euclidean")
                same=np.diag(d)
                shuffled=np.array([d[np.arange(len(x)),perm] for perm in permutations])
                null_med=np.median(shuffled,axis=1)
                rows.append({"patient":patient,"cycle_from":int(c1),"cycle_to":int(c2),"method":method,
                             "n_clones":len(x),"same_median":float(np.median(same)),
                             "null_median":float(np.median(null_med)),
                             "distance_ratio":float(np.median(same)/np.median(null_med)),
                             "p_lower":float((1+np.sum(null_med<=np.median(same)))/(1+len(null_med)))})
                nulls.extend({"patient":patient,"cycle_from":int(c1),"cycle_to":int(c2),"method":method,
                              "permutation":i,"median_distance":float(v)} for i,v in enumerate(null_med))
            for j,barcode in enumerate(x.index):
                row={"patient":patient,"cycle_from":int(c1),"cycle_to":int(c2),"biological_clone":barcode,
                     "n_from":int(x.iloc[j].n_cells),"n_to":int(y.iloc[j].n_cells)}
                for key in MODULES:
                    row[key+"_from"]=float(x.iloc[j][key]);row[key+"_to"]=float(y.iloc[j][key])
                    row[key+"_delta"]=float(y.iloc[j][key]-x.iloc[j][key])
                changes.append(row)
    if not rows: raise ValueError("No matched longitudinal comparisons passed coverage criteria")
    pd.DataFrame(rows).to_csv(OUT/"longitudinal_distance_comparison.csv",index=False)
    pd.DataFrame(nulls).to_csv(OUT/"longitudinal_distance_null.csv.gz",index=False)
    pd.DataFrame(changes).to_csv(OUT/"longitudinal_module_changes.csv.gz",index=False)
    per=table.groupby("patient").agg(clone_cycles=("cycle","size"),biological_clones=("biological_clone","nunique"),
             reliable_clone_cycles=("reliability",lambda v:int(v.ge(.5).sum())))
    per["linked_biological_clones_min4"]=linked.groupby("patient").biological_clone.nunique()
    per.fillna(0).to_csv(OUT/"per_patient_summary.csv")
    return {"n_patient_intervals":len(rows)//len(features),"min_cells_per_cycle":4,
            "null":"200 no-fixed-point permutations within patient, cycle pair and target log2 size bin",
            "cohort_selection":"Same clone pairs across all three methods; no reliability filtering"}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a,audit=prepare()
    if all((OUT/name).exists() for name in ["profiles_kernel.csv.gz","features_mean.csv.gz","features_kernel.csv.gz","features_RNA_mean.csv.gz","cell_umap.csv.gz","profile_audit.json"]):
        table=pd.read_csv(OUT/"profiles_kernel.csv.gz",index_col=0)
        profiles=json.loads((OUT/"profile_audit.json").read_text())
        print("Reusing completed, unchanged profile and UMAP results",flush=True)
    else:
        table,profiles=profile(a)
    longitudinal=paired_comparison(table)
    dump("summary.json",{**audit,"status":"complete","profiles":profiles,"longitudinal":longitudinal})
    print("NSCLC analysis complete",flush=True)

if __name__=="__main__": main()
