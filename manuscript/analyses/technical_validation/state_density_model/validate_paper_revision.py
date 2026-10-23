#!/usr/bin/env python3
"""Validate identities, common coverage and saved readouts of the paper revision."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"case_studies/results"
def csv(folder,name):return pd.read_csv(folder/name,index_col=0)
def main():
    checks={}
    l=DATA/"clonotrace_larry"; n=DATA/"clonotrace_nsclc"
    meta=csv(l,"cell_metadata.csv.gz")
    assert meta.index.is_unique and len(meta)==130887
    assert meta.n_lineage_labels.eq(1).sum()==49302
    for suffix,key in [("all","clone_day"),("early","barcode")]:
        obs=meta if suffix=="all" else meta[meta.day.eq(2)]
        count=obs.groupby(key).size();count=count[count.ge(2)]
        for rep in ["mean","kernel"]:
            p=csv(l,f"profiles_{suffix}_{rep}.csv.gz");f=csv(l,f"features_{suffix}_{rep}.csv.gz")
            assert p.index.is_unique and p.index.equals(f.index)
            assert np.isfinite(f.to_numpy()).all()
            np.testing.assert_array_equal(p.n_cells,count.reindex(p.index))
            assert set(p.index)==set(count.index)
            np.testing.assert_allclose(p.filter(regex="^state:").sum(axis=1),1)
    checks["larry_profile_identity_and_coverage"]=True
    y=csv(l,"day6_observed_state_fractions.csv")
    assert len(y)==172 and np.isfinite(y.to_numpy()).all()
    np.testing.assert_allclose(y.sum(axis=1),1)
    folds=pd.read_csv(l/"future_folds.csv")
    assert not folds.duplicated(["repeat","barcode"]).any()
    assert folds.groupby("repeat").barcode.nunique().eq(172).all()
    pred=pd.read_csv(l/"future_predictions.csv.gz")
    assert not pred.duplicated(["repeat","method","barcode","target"]).any()
    assert np.isfinite(pred[["observed","predicted"]]).all().all()
    assert pred.groupby(["repeat","method"]).barcode.nunique().eq(172).all()
    assert pred.groupby(["repeat","barcode","target"])["observed"].nunique().eq(1).all()
    assert pred.groupby(["repeat","barcode"]).fold.nunique().eq(1).all()
    np.testing.assert_allclose(pred.groupby(["repeat","method","barcode"]).predicted.sum(),1,atol=1e-6)
    for (repeat,fold),q in pred[pred.method.eq("Training_mean")].groupby(["repeat","fold"]):
        test=folds[(folds["repeat"]==repeat)&(folds.fold==fold)].barcode
        train=y.index.difference(test)
        for target,z in q.groupby("target"):
            np.testing.assert_allclose(z.predicted,y.loc[train,target].mean())
    checks["larry_common_barcode_folds_and_training_only_baseline"]=True
    err=pd.read_csv(l/"split_sample_errors.csv.gz")
    assert len(err)==15000 and err.clone_day_well.nunique()==250
    assert not err.duplicated(["clone_day_well","repeat","n_sample"]).any()
    assert set(err.n_sample)=={2,4,8} and err.n_reference.min()>=8
    assert (err[["raw_mse","threadfin_mse"]]>=0).all().all()
    assert all(int(hashlib.sha256(k.split("|")[0].encode()).hexdigest()[:8],16)%2==1
               for k in err.clone_day_well.unique())
    # The source partitions a permutation into query and the disjoint second half.
    source=(ROOT/"case_studies/reanalyse_clonotrace_larry.py").read_text()
    assert "ref=z[perm[half:]].mean(axis=0)" in source
    assert "raw=z[perm[:n]].mean(axis=0)" in source
    checks["larry_split_coverage_and_disjoint_barcode_allocation"]=True
    meta=csv(n,"exported_cell_metadata.csv.gz")
    assert len(meta)==195685 and meta.index.is_unique
    assert meta.tcr_pair.notna().sum()==124534 and meta.patient.nunique()==10
    paired=meta[meta.tcr_pair.notna()]
    assert paired.groupby("clone_biological").patient.nunique().eq(1).all()
    assert paired.groupby("clone_cycle").cycle.nunique().eq(1).all()
    expected=paired.groupby("clone_cycle").size()
    for rep in ["mean","kernel"]:
        p=csv(n,f"profiles_{rep}.csv.gz");f=csv(n,f"features_{rep}.csv.gz")
        assert len(p)==10428 and p.index.is_unique and p.index.equals(f.index)
        assert set(p.index)==set(expected[expected.ge(2)].index)
        np.testing.assert_array_equal(p.n_cells,expected.reindex(p.index))
        assert np.isfinite(f.to_numpy()).all()
    d=pd.read_csv(n/"longitudinal_distance_comparison.csv")
    assert len(d)==87 and d.patient.nunique()==10
    assert set(d.method)=={"RNA_mean","Threadfin_mean","Threadfin_kernel"}
    assert d.groupby(["patient","cycle_from","cycle_to"]).n_clones.nunique().eq(1).all()
    assert (d.cycle_to>d.cycle_from).all() and np.isfinite(d.distance_ratio).all()
    null=pd.read_csv(n/"longitudinal_distance_null.csv.gz")
    assert len(null)==87*200
    changes=pd.read_csv(n/"longitudinal_module_changes.csv.gz")
    assert (changes[["n_from","n_to"]]>=4).all().all()
    assert not changes.duplicated(["patient","cycle_from","cycle_to","biological_clone"]).any()
    checks["nsclc_patient_identity_cycles_and_common_method_pairs"]=True
    fig2=json.loads((DATA/"figure2_biology/summary.json").read_text())
    assert fig2["n_expanded_families"]==1414 and fig2["n_plotted_families"]==381
    assert fig2["paired_same_family_other_gate"]["n_families"]==218
    checks["figure2_audit_denominators"]=True
    sources=[ROOT/"case_studies/reanalyse_clonotrace_larry.py",
             ROOT/"case_studies/reanalyse_clonotrace_nsclc.py",
             ROOT/"case_studies/review_figure2_biology.py",
             ROOT/"paper/figure_plan/external_validation_figures.py",
             ROOT/"paper/export_manuscript.py"]
    for p in sources:ast.parse(p.read_text())
    checks["analysis_and_export_syntax"]=True
    result={"status":"passed","checks":checks,
            "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "limits":["Checks validate saved identities/coverage and readout consistency, not clinical causality.",
                      "Unsupervised PCA/profiles remain transductive within the stated input time window.",
                      "Visual source figures and DOCX media are checked separately."]}
    (DATA/"clonotrace_revision_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
