#!/usr/bin/env python3
"""Finalize complete native runs; refuse partial comparisons and zero filling."""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
from score_native_benchmark import OUT, ROOT, build_representations, score

INTERNAL = ROOT.parent / 'internal_validation'
DATASETS = ['mouse_np','mouse_rbd']


def resource_log(name):
    path=INTERNAL/name
    text=path.read_text() if path.exists() else ''
    peaks=[int(v) for v in re.findall(r'Maximum resident set size \(kbytes\): (\d+)',text)]
    return peaks


def finalize():
    rows=[]
    runs=OUT/'model_runs';runs.mkdir(exist_ok=True)
    for i,ds in enumerate(DATASETS,1):
        source=json.loads((OUT/ds/'run.json').read_text())
        assert source.get('benisse_r_status')=='completed',f'{ds}: full Benisse model is incomplete'
        assert source.get('bigcn_status')=='completed',f'{ds}: BiGCN model is incomplete'
        (runs/f'{ds}.json').write_text(json.dumps(source,indent=2)+'\n')
        build_representations(ds,'native')
        score(ds)
        a=json.loads((OUT/'representations'/ds/'representation_audit.json').read_text())
        peaks={key:resource_log(f'{prefix}_{job}_{i}.err') for key,prefix,job in [
            ('Benisse','native_benchmark',32294672),('BiGCN','native_benchmark',32288767),
            ('profiles','clone_profiles_benchmark',32295543)]}
        if source.get('bigcn_resource_log'):
            peaks['BiGCN']=resource_log(source['bigcn_resource_log'])
        base={'dataset':ds,'input_cells':source['n_cells'],'status':'completed',
              'hardware_scope':'2 CPU threads; separate allocations; not an end-to-end speed ranking'}
        rows.append({**base,'method':'Benisse','stage':'pretrained_encoder',
                     'runtime_seconds':source['benisse_encoder_seconds'],'max_rss_kb':None,
                     'version':source['benisse_commit'],'notes':'CDR3 encoder only; unique sequences are mapped back to cells by full R stage.'})
        rows.append({**base,'method':'Benisse','stage':'official_R','runtime_seconds':source['benisse_r_seconds'],
                     'max_rss_kb':peaks['Benisse'][0] if peaks['Benisse'] else None,
                     'version':source['benisse_commit'],'notes':'Full native R graph model with reversible cell aliases; RSS is whole command peak.'})
        rows.append({**base,'method':'BiGCN','stage':'official_graph_and_training','runtime_seconds':source['bigcn_seconds'],
                     'max_rss_kb':peaks['BiGCN'][0] if peaks['BiGCN'] else None,
                     'version':source['bigcn_commit'],'notes':'Native data_process + graphStructure + main; Benisse encoder and input adapter prep excluded; single successful upstream run with no fixed seed.' +
                     (' Initial allocation timed out; same configuration retried with a longer allocation. Failed attempt is recorded separately in the model manifest.' if source.get('bigcn_retry_of') else '')})
        for method in ['Threadfin_mean','Threadfin_kernel','clone2vec']:
            rows.append({**base,'method':method,'stage':method,'runtime_seconds':a[method]['seconds'],
                         'max_rss_kb':peaks['profiles'][1] if method=='clone2vec' and len(peaks['profiles'])>1 else None,
                         'version':a[method].get('version','repository v4'),
                         'notes':'RNA PCA100 preprocessing and downstream readout excluded. Profile-stage RSS was measured jointly, not per representation.' if method.startswith('Threadfin') else 'Native neighbour graph plus Skip-Gram; RSS is whole command peak.'})
        if peaks['profiles']:
            rows.append({**base,'method':'Threadfin profiles','stage':'combined_profile_process','runtime_seconds':None,
                         'max_rss_kb':peaks['profiles'][0],'version':'repository v4',
                         'notes':'Joint RNA-baseline/mean/kernel process peak; cannot be assigned separately to either model.'})
    for filename in ['heldout_scores.csv','heldout_predictions.csv','coverage.csv']:
        pd.concat([pd.read_csv(OUT/'scores'/ds/filename) for ds in DATASETS],ignore_index=True).to_csv(OUT/filename,index=False)
    pd.DataFrame(rows).to_csv(OUT/'execution.csv',index=False)
    scores=pd.read_csv(OUT/'heldout_scores.csv')
    summary=scores.groupby(['dataset','target','min_cells','method'],sort=False).agg(
        scored_mice=('held_out_donor','nunique'),median_mae=('mae','median'),
        q25_mae=('mae',lambda x:x.quantile(.25)),q75_mae=('mae',lambda x:x.quantile(.75)),
        median_r2=('r2','median'),n_scored_families=('n_test','sum')).reset_index()
    summary.to_csv(OUT/'readout_summary.csv',index=False)
    (OUT/'pipeline_status.json').write_text(json.dumps({'status':'completed','datasets':DATASETS,
        'no_partial_comparisons':True,'models':7,'comparators_including_intercept':8,
        'primary_threshold':2,'sensitivity_threshold':5,'readout_scope':'label-held-out transductive; whole-mouse folds',
        'source_scripts':['native_benchmark.py','score_native_benchmark.py','finalize_native_benchmark.py']},indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'paper/figure_plan/make_biology_figures.py')],cwd=ROOT,check=True)
    print('Complete native benchmark source tables and publication figures regenerated.',flush=True)


if __name__=='__main__':finalize()
