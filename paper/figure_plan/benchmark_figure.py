"""Task-aware benchmark layout; never substitute missing scores with zero.

Visual organisation follows the task/performance/usage separation of Yan et al.
(2026); all artwork and plots are generated independently from our own results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import schematics as sk
from concept_figure import bcell, family_profile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BLUE='#6666b2'; GREEN='#93bd8b'; RED='#a67cb9'; INK='#26313b'; GREY='#b9bec5'
METHODS=['RNA_centroid','RNA_context_centroid','Threadfin_mean','Threadfin_kernel','Benisse','BiGCN','clone2vec','Training_mean']
NAMES={'RNA_centroid':'RNA mean','RNA_context_centroid':'RNA mean + context','Threadfin_mean':'Threadfin mean',
       'Threadfin_kernel':'Threadfin kernel','Training_mean':'Training mean','clone2vec':'clone2vec','Benisse':'Benisse','BiGCN':'BiGCN'}

def draw(new_page,p,save):
    folder=ROOT/'case_studies/results/native_benchmark'
    scores=pd.read_csv(folder/'heldout_scores.csv')
    if scores.empty:
        raise ValueError('Figure 6 requires completed native model scores, not an empty schema.')
    scores=scores[scores.min_cells.eq(2) & scores.status.eq('completed')]
    for ds in ['mouse_np','mouse_rbd']:
        present=set(scores.loc[scores.dataset.eq(ds),'method'])
        if present != set(METHODS):
            raise ValueError(f'{ds}: incomplete native comparison: {set(METHODS)-present}')
    cap=pd.read_csv(HERE/'method_capabilities.csv')
    fig=new_page('Figure 6','Comparing methods at the task they actually perform',
                 'Published capabilities and measured reporter interpretation are shown separately.',height=334)
    a=p(fig,0,7,183,42,'A','One common biological readout; native methods keep their own representation')
    sk.canvas(a,183,42)
    stages=[(0,28,'GC reporters','NP-OVA / RBD',BLUE),(37,29,'Paired input','RNA + BCR',GREEN),
            (76,29,'Representation','Native output',RED),(115,29,'Fixed families','Same mice',BLUE),
            (154,29,'Mouse readout','Labels held out',GREEN)]
    for x,w,t,sub,col in stages:
        a.add_patch(FancyBboxPatch((x,5),w,32,boxstyle='round,pad=.2,rounding_size=1.5',fc=col+'0c',ec=col,lw=.7))
        if x==0:
            sk.mouse(a,x+w/2-3,29,.8,color=col)
            sk.syringe(a,x+w/2+8,30,.6,color=col)
        elif x==37:
            # A count/PC matrix and receptor records: no gate input to models.
            for row in range(3):
                for column in range(5):
                    a.add_patch(Rectangle((x+6+column*1.8,26+row*1.6),1.5,1.3,fc=BLUE if (row+column)%3 else '#ddd9ef',ec='none'))
            for row in range(3):
                a.plot([x+18,x+24],[27+row*1.6]*2,color=RED,lw=.7)
        elif x==76:
            coords=np.array([[8,27],[13,32],[19,29],[22,33],[16,25]])+[x,0]
            for i,j in [(0,1),(1,2),(2,3),(2,4)]:a.plot(coords[[i,j],0],coords[[i,j],1],color=GREY,lw=.7)
            a.scatter(coords[:,0],coords[:,1],s=15,c=[BLUE,GREEN,RED,BLUE,GREEN],linewidths=0)
        elif x==115:
            family_profile(a,x+9,29,[.6,.2,.1,.1],r=3.1,label='A')
            family_profile(a,x+21,29,[.1,.1,.7,.1],r=3.1,label='B')
        else:
            for i in range(3):sk.mouse(a,x+6+i*7,29,.5,color=BLUE if i==2 else GREY)
        a.text(x+w/2,19,t,ha='center',fontsize=6.2,color=INK)
        a.text(x+w/2,11,sub,ha='center',fontsize=5.4,color=INK)
        if x<154:sk.arrow(a,x+w+1,21,x+w+7,21,color=GREY,head=1.7)
    b=p(fig,23,83,160,81,'B','Capabilities from papers and official implementations')
    b.set_xlim(0,4.4);b.set_ylim(len(cap)-.5,-1.5);b.set_axis_off()
    columns=['repertoire_workflow','joint_representation','clone_state_analysis']
    for j,label in enumerate(['Repertoire','Joint representation','Clone/state analysis','Specialised strength']):
        b.text(j*.93,-1.1,label,fontsize=6.1,color=INK)
    for i,r in cap.iterrows():
        b.text(-.60,i,r.method,fontsize=6,va='center',color=BLUE if r.method=='Threadfin' else INK)
        for j,key in enumerate(columns):
            val=r[key];col=BLUE if val=='Native' else GREEN if val.startswith('Via') else GREY
            b.add_patch(FancyBboxPatch((j*.93,i-.35),.84,.68,boxstyle='round,pad=.02',fc=col+'18',ec='none'))
            b.text(j*.93+.42,i,val,ha='center',va='center',fontsize=5.5,color=INK)
        b.text(2.8,i,r.special_focus,va='center',fontsize=5.4,color=INK)
    b.text(-.6,len(cap)+.35,'Native and workflow outputs are different tasks; this table is not a performance ranking.',fontsize=5.7,color=INK)
    c=p(fig,31,201,71,91,'C','How accurately can each representation\nread captured division-gate occupancy?')
    d=p(fig,121,201,62,81,'D','Which families enter the comparison?')
    c.set_axis_off();d.set_axis_off()
    # Two dataset-specific readouts; no pooled cell replicate or omitted zero.
    targets={'mouse_np':'division_gate:mCherry-low','mouse_rbd':'division_gate:mCherry-low'}
    for k,(ds,target) in enumerate(targets.items()):
        q=scores[(scores.dataset.eq(ds)) & scores.target.eq(target)].copy()
        if 'status' in q:q=q[q.status.eq('completed')]
        if q.empty:raise ValueError(f'Missing completed reporter scores for {ds}')
        ax=c.inset_axes([0,.57 if k==0 else .04,1,.36])
        rng=np.random.default_rng(0)
        for i,method in enumerate(METHODS):
            v=100*q.loc[q.method.eq(method),'mae'].dropna()
            ax.scatter(v,np.full(len(v),i)+rng.uniform(-.12,.12,len(v)),s=10,
                       color=BLUE if method.startswith('Threadfin') else GREY if method=='Training_mean' else RED,alpha=.7)
            if len(v):ax.plot([v.median()],[i],marker='|',ms=8,color=INK)
        ax.set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS],fontsize=5.5);ax.invert_yaxis()
        ax.set_title('NP-OVA' if ds=='mouse_np' else 'RBD',loc='right',fontsize=5.9)
        ax.tick_params(labelsize=5.2);ax.set_xlabel('Gate-fraction MAE (percentage points; lower is better)',fontsize=5.5)
    # Coverage must be actual family coverage, never unique CDR3 / cell count.
    d.text(0,.98,'Expanded-family feature coverage',fontsize=5.8,color=INK)
    for x,ds,lab in [(.69,'mouse_np','NP-OVA'),(.96,'mouse_rbd','RBD')]:
        d.text(x,.89,lab,ha='right',fontsize=5.5,color=INK)
        for i,method in enumerate(METHODS):
            q=scores[scores.dataset.eq(ds) & scores.method.eq(method)]
            d.text(x,.82-i*.045,f'{100*q.coverage.iloc[0]:.0f}%',ha='right',fontsize=5.6,color=INK)
    for i,method in enumerate(METHODS):d.text(0,.82-i*.045,NAMES[method],fontsize=5.6,color=INK)
    counts=scores.groupby('dataset').n_common_families.first()
    d.text(0,.38,f'Common expanded families:\nNP-OVA: {counts.mouse_np:,}; RBD: {counts.mouse_rbd:,}',fontsize=6.1,color=INK,linespacing=1.5)
    d.text(0,.22,'Same family membership and label denominators.\nUnmeasured gates stay missing.\nNo 2D UMAP coordinates enter the readout.',fontsize=5.7,color=INK,linespacing=1.6)
    d.text(0,.01,'Biological scope\nCaptured division-gate composition\nfrom label-blind, transductive representations.\nThe comparison does not measure future fate.',fontsize=5.7,color=INK,linespacing=1.6)
    c.text(0,-.14,'Dots: label-held-out mice; bars: medians. Native model input excludes gates.\nTraining-only kernel centring/scaling and nested mouse-wise ridge selection.',transform=c.transAxes,fontsize=5.4,color=INK)
    d.text(0,-.15,'Size sensitivity, additional RBD gates and measured costs: Supplementary 7.\nCapabilities without a comparable output are not assigned zero scores.',transform=d.transAxes,fontsize=5.4,color=INK)
    save(fig,'Figure_6')


def supplementary(new_page,p,save):
    folder=ROOT/'case_studies/results/native_benchmark'
    scores=pd.read_csv(folder/'heldout_scores.csv')
    fig=new_page('Supplementary Figure 7','Native benchmark sensitivity and execution scope',
                 'Independent gate fractions, family-size sensitivity and model-stage costs are audited separately.',height=305)
    for j,ds in enumerate(['mouse_np','mouse_rbd']):
        a=p(fig,32+j*93,10,58,69,'A' if j==0 else 'B',('NP-OVA' if j==0 else 'RBD')+': family-size sensitivity')
        q=scores[scores.dataset.eq(ds) & scores.target.eq('division_gate:mCherry-low')]
        for i,m in enumerate(METHODS):
            for minimum,offset,col in [(2,-.1,BLUE),(5,.1,GREEN)]:
                v=100*q.loc[q.method.eq(m) & q.min_cells.eq(minimum),'mae']
                if len(v):a.errorbar(v.median(),i+offset,xerr=[[v.median()-v.quantile(.25)],[v.quantile(.75)-v.median()]],fmt='o',ms=2.8,color=col,lw=.65)
        a.set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS],fontsize=5.6);a.invert_yaxis();a.set_xlabel('Gate-fraction MAE (percentage points)')
        a.text(0,-.20,'Median / mouse IQR; violet: ≥2, green: ≥5 measured cells.\nDifferent size thresholds can retain different mice.',transform=a.transAxes,fontsize=5.3,color=INK)
    for j,target in enumerate(['rbd_bait:RBD+','zone_gate:DZ']):
        a=p(fig,32+j*93,125,58,59,'C' if j==0 else 'D','RBD: '+('antigen-probe gate' if j==0 else 'dark-zone sort gate'))
        q=scores[scores.dataset.eq('mouse_rbd') & scores.target.eq(target) & scores.min_cells.eq(2)]
        rng=np.random.default_rng(0)
        for i,m in enumerate(METHODS):
            v=100*q.loc[q.method.eq(m),'mae']
            a.scatter(v,np.full(len(v),i)+rng.uniform(-.12,.12,len(v)),s=9,color=BLUE if m.startswith('Threadfin') else RED,alpha=.65)
            if len(v):a.plot([v.median()],[i],marker='|',ms=7,color=INK)
        a.set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS],fontsize=5.6);a.invert_yaxis();a.set_xlabel('Gate-fraction MAE (percentage points)')
        n=int(q.n_selected_families.iloc[0]) if len(q) else 0
        nmouse=q.held_out_donor.nunique()
        a.text(0,-.21,f'{n:,} eligible families; {nmouse} scored mice.\nUnmeasured gates are excluded, not treated as negatives.',transform=a.transAxes,fontsize=5.3,color=INK)
    a=p(fig,0,236,183,37,'E','Measured execution uses the published model stages')
    a.set_axis_off();execution=pd.read_csv(folder/'execution.csv')
    # Keep scopes explicit; preprocessing, encoders and model stages are not interchangeable.
    main=execution[execution.stage.isin(['official_R','official_graph_and_training','clone2vec','Threadfin_mean','Threadfin_kernel'])]
    a.text(0,1,'Model / stage',fontsize=5.8);a.text(.39,1,'NP-OVA seconds',fontsize=5.8);a.text(.62,1,'RBD seconds',fontsize=5.8);a.text(.83,1,'Resource scope',fontsize=5.8)
    for i,stage in enumerate(['official_R','official_graph_and_training','clone2vec','Threadfin_mean','Threadfin_kernel']):
        q=main[main.stage.eq(stage)];y=.83-i*.14
        method=q.method.iloc[0] if len(q) else stage
        a.text(0,y,method+' · '+stage.replace('official_',''),fontsize=5.6,color=INK)
        for x,ds in [(.49,'mouse_np'),(.70,'mouse_rbd')]:
            z=q[q.dataset.eq(ds)]
            a.text(x,y,f'{z.runtime_seconds.iloc[0]:,.1f}' if len(z) and pd.notna(z.runtime_seconds.iloc[0]) else 'Not completed',ha='right',fontsize=5.6,color=INK)
        a.text(.83,y,'CPU 2 threads; model stage',fontsize=5.4,color=INK)
    a.text(0,-.13,'Counts/IDs, dependency versions, peak RSS and failed attempts are in the audit tables. PCA preprocessing and downstream readout are outside these timings.\nBenisse additionally needs its pretrained encoder; BiGCN consumes that encoding. These timings are not an end-to-end package speed ranking.',fontsize=5.4,color=INK)
    save(fig,'Supplementary_7')
