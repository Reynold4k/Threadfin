#!/usr/bin/env python3
"""Descriptive GC SHM trends, with donors weighted equally and labels audited.

FALSE s_pos_clone is not a uniformly assayed negative. This does not compute
affinity, a prediction score or a change within the same persistent family.
"""
from pathlib import Path
import json
import pandas as pd

RESULTS=Path(__file__).resolve().parent/'results'

def main():
    out=RESULTS/'spike_binding_audit'
    out.mkdir(exist_ok=True)
    columns=['clone_id','donor','timepoint','state','spike_binding','mutation_frequency']
    d=pd.read_csv(RESULTS/'ln_vaccine/cells.csv.gz',usecols=columns)
    dates=['d28','d35','d60','d110','d201']
    d=d[d.state.eq('GC') & d.timepoint.isin(dates)].dropna(subset=['clone_id','mutation_frequency'])
    d=d[d.spike_binding.isin(['S+','S-'])]
    f=d.groupby(['donor','timepoint','spike_binding','clone_id'],observed=True).agg(
        gc_cells=('state','size'), mean_family_shm=('mutation_frequency','mean')).reset_index()
    f.to_csv(out/'gc_family_timepoint_source.csv',index=False)
    t=f.groupby(['donor','timepoint','spike_binding'],observed=True).agg(
        n_families=('clone_id','size'),n_gc_cells=('gc_cells','sum'),
        median_family_shm=('mean_family_shm','median')).reset_index()
    # Each plotted date uses the same available donors for its two labels.
    paired=t.pivot(index=['donor','timepoint'],columns='spike_binding',values='median_family_shm').dropna()
    good=pd.MultiIndex.from_frame(t[['donor','timepoint']]).isin(paired.index)
    t['included_in_paired_donor_summary']=good
    t.to_csv(out/'gc_donor_timepoint_shm.csv',index=False)
    s=t[good].groupby(['timepoint','spike_binding'],observed=True).agg(
        n_donors=('donor','nunique'),n_families=('n_families','sum'),n_gc_cells=('n_gc_cells','sum'),
        equal_donor_mean=('median_family_shm','mean'),
        donor_q25=('median_family_shm',lambda x:x.quantile(.25)),
        donor_q75=('median_family_shm',lambda x:x.quantile(.75))).reset_index()
    s.to_csv(out/'gc_shm_curve_summary.csv',index=False)
    (out/'gc_shm_curve_definition.json').write_text(json.dumps({
        'population':'GC-annotated cells at nonpooled d28/35/60/110/201',
        'unit':'family mean SHM within donor/date, then donor median; equal-weight donor mean at each date',
        'comparison':'S+ author-identified family versus not identified S+ (source FALSE)',
        'matching':'At each date require both labels in the donor; donors can differ between dates.',
        'interval':'Interquartile range across donor medians, not a confidence interval.',
        'limits':'Descriptive changing captured repertoire; no test of quantitative affinity, matched-family maturation or future fate.'
    },indent=2))

if __name__=='__main__':main()
