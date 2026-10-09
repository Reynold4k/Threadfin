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
    readouts=pd.read_csv(folder/'readout_summary.csv')
    cap=pd.read_csv(HERE/'method_capabilities.csv').set_index('method')
    fig=new_page('Figure 6','Comparing methods at the task they actually perform',
                 'Published capabilities and measured reporter interpretation are shown separately.',height=288)
    a=p(fig,0,7,183,46,'A','What Threadfin does')
    sk.canvas(a,183,46)
    stages=[(0,44,'Paired input','RNA + BCR per cell',BLUE),
            (51,40,'Donor-private families','defined by receptor sequence',GREEN),
            (98,42,'Profiles + reliability','context-adjusted, sampling-aware',RED),
            (147,36,'Family-level readouts','maps · gates · sharing',BLUE)]
    for x,w,t,sub,col in stages:
        a.add_patch(FancyBboxPatch((x,4),w,34,boxstyle='round,pad=.2,rounding_size=1.5',fc=col+'0c',ec=col,lw=.7))
        a.text(x+w/2,12.5,t,ha='center',fontsize=6.2,color=INK,fontweight='bold')
        a.text(x+w/2,7.5,sub,ha='center',fontsize=5.2,color=INK)
    # stage 1 icon: count matrix + receptor record
    for row in range(3):
        for column in range(4):
            a.add_patch(Rectangle((8+column*2.0,17+row*2.0),1.6,1.5,fc=BLUE if (row+column)%3 else '#ddd9ef',ec='none'))
    for row in range(3):
        a.plot([30,36],[18+row*2.0]*2,color=RED,lw=.8)
    a.text(33,24.5,'VDJ',ha='center',fontsize=4.6,color=RED)
    # stage 2 icon: three sequence-defined families
    family_profile(a,60,25,[.6,.2,.1,.1],r=3.2,label='A')
    family_profile(a,71,25,[.1,.1,.7,.1],r=3.2,label='B')
    family_profile(a,82,25,[.2,.5,.1,.2],r=3.2,label='C')
    # stage 3 icon: one family profile + context neighbourhood + reliability gauge
    family_profile(a,108,25,[.3,.4,.2,.1],r=3.4,label='')
    rng=np.random.default_rng(2)
    a.scatter(116+rng.uniform(0,9,10),20+rng.uniform(0,10,10),s=2.5,color=GREY,linewidths=0)
    a.add_patch(Rectangle((128,19),2.2,12,fc='white',ec=INK,lw=.5))
    a.add_patch(Rectangle((128,19),2.2,9,fc=RED,ec='none'))
    a.text(129.1,32.5,'rel.',fontsize=4.4,color=INK)
    # stage 4 icon: mini clone map + gate bars
    pts=np.array([[151,22],[154,27],[157,21],[160,26],[163,22],[166,27],[169,23]])
    a.scatter(pts[:,0],pts[:,1],s=6,c=[BLUE,BLUE,RED,RED,RED,GREEN,GREEN],linewidths=0)
    a.bar([174,177],[5,8],width=2.2,bottom=19,color=[GREY,RED])
    for x in [44,91,140]:
        sk.arrow(a,x+2,21,x+7,21,color=GREY,head=1.7)
    # B: capability matrix in three tiers — shared, subset-native, Threadfin-only.
    b=p(fig,0,56,183,119,'B','Which capabilities are shared, native in a subset, or unique to Threadfin')
    b.set_xlim(0,183);b.set_ylim(119,0);b.set_axis_off()
    order=['Threadfin','Benisse','BiGCN','CoNGA','sciCSR','clone2vec','Ibex','Dandelion','Scirpy','scRepertoire','Platypus']
    N,V,X='native','via','none'
    def mrep(m):return N if cap.loc[m,'repertoire_workflow']=='Native' else V
    def mjoint(m):return N if cap.loc[m,'joint_representation'] in ('Native','Expression context') else V
    def mstate(m):return N if cap.loc[m,'clone_state_analysis'] in ('Native','CSR output') else V
    def tf_only(m):return N if m=='Threadfin' else X
    bands=[('Shared by all compared tools',
            [('VDJ import and clonotype grouping',mrep),
             ('Joint receptor–expression representation',mjoint),
             ('Clone-state or trajectory analysis',mstate)]),
           ('Native in a subset of tools',
            [('Pretrained deep receptor-sequence encoder',lambda m:N if m in ('Benisse','BiGCN','Ibex') else X),
             ('Native joint receptor–expression model',lambda m:mjoint(m) if mjoint(m)==N else X)]),
           ('Unique to Threadfin',
            [('Donor-private family definitions',tf_only),
             ('Sampling-aware reliability per family',tf_only),
             ('Context-adjusted profiles, receptor genes excluded',tf_only),
             ('Held-out prediction of measured family biology ‡',tf_only)])]
    x0=69.;dx=10.1;centres=[x0+j*dx for j in range(len(order))]
    b.add_patch(Rectangle((centres[0]-dx/2,22),dx,90,fc=BLUE+'14',ec='none',zorder=0))
    for j,m in enumerate(order):
        b.text(centres[j],21,m,rotation=38,ha='left',va='top',fontsize=5.2,
               color=BLUE if m=='Threadfin' else INK,fontweight='bold' if m=='Threadfin' else 'normal')
    y=25.
    for band,rows in bands:
        b.text(2,y,band,fontsize=6.0,color=INK,fontweight='bold',va='center')
        b.plot([2,181],[y+2.8]*2,color='#e3e6ea',lw=.6,zorder=0)
        y+=7.5
        for label,state in rows:
            b.text(63,y,label,fontsize=5.6,color=INK,ha='right',va='center')
            for j,m in enumerate(order):
                st=state(m)
                if st==N:b.scatter(centres[j],y,marker='o',s=22,color=BLUE,zorder=3)
                elif st==V:b.scatter(centres[j],y,marker='o',s=22,facecolors='none',edgecolors=GREEN,linewidths=1.1,zorder=3)
                else:b.text(centres[j],y,'—',ha='center',va='center',fontsize=6,color=GREY)
            y+=6.8
        y+=2.0
    b.scatter([3],[114.5],marker='o',s=20,color=BLUE)
    b.text(5.5,114.5,'native',fontsize=5.4,va='center',color=INK)
    b.scatter([17],[114.5],marker='o',s=20,facecolors='none',edgecolors=GREEN,linewidths=1.1)
    b.text(19.5,114.5,'via workflow',fontsize=5.4,va='center',color=INK)
    b.text(34,114.5,'— none reported',fontsize=5.4,va='center',color=GREY)
    b.text(181,114.5,'‡ quantified in panels C and D',fontsize=5.4,va='center',ha='right',color=INK)
    # C: measured reporter interpretation, held-out mice.
    c=p(fig,10,183,88,95,'C','Held-out gate fractions predicted from family profiles')
    c.set_axis_off()
    targets={'mouse_np':'division_gate:mCherry-low','mouse_rbd':'division_gate:mCherry-low'}
    stats={}
    for ds,target in targets.items():
        q=scores[(scores.dataset.eq(ds)) & scores.target.eq(target)].copy()
        if 'status' in q:q=q[q.status.eq('completed')]
        if q.empty:raise ValueError(f'Missing completed reporter scores for {ds}')
        stats[ds]=q
    order_m=sorted(METHODS,key=lambda m:100*np.median([stats[ds].loc[stats[ds].method.eq(m),'mae'].median() for ds in targets]))
    ax=c.inset_axes([.27,.13,.71,.74])
    dcols={'mouse_np':BLUE,'mouse_rbd':RED}
    for i,m in enumerate(order_m):
        for ds,marker in [('mouse_np','o'),('mouse_rbd','s')]:
            v=100*stats[ds].loc[stats[ds].method.eq(m),'mae'].dropna()
            col=INK if m.startswith('Threadfin') else GREY if m=='Training_mean' else dcols[ds]
            ax.errorbar(v.median(),i+( -.12 if ds=='mouse_np' else .12),
                        xerr=[[v.median()-v.quantile(.25)],[v.quantile(.75)-v.median()]],
                        fmt=marker,ms=3.6,color=col,lw=.9,capsize=1.8,zorder=3)
    for ds,ls,ytxt in [('mouse_np','-',-.9),('mouse_rbd','--',-2.1)]:
        chance=100*stats[ds].loc[stats[ds].method.eq('Training_mean'),'mae'].median()
        ax.axvline(chance,color=GREY,lw=.9,ls=ls,zorder=1)
        ax.text(chance,ytxt,'chance\nNP-OVA' if ds=='mouse_np' else 'chance\nRBD',fontsize=5.0,color=INK,ha='center',va='bottom')
    ax.set_yticks(range(len(order_m)),[NAMES[m] for m in order_m],fontsize=5.8)
    for i,m in enumerate(order_m):
        if m.startswith('Threadfin'):
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.invert_yaxis();ax.tick_params(labelsize=5.4)
    ax.set_xticks([15,20,25,30,35])
    ax.set_xlabel('Gate-fraction MAE (percentage points; lower is better)',fontsize=5.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=dcols['mouse_np'],label='NP-OVA'),
                       Line2D([],[],marker='s',ls='',color=dcols['mouse_rbd'],label='RBD'),
                       Line2D([],[],color=GREY,lw=.9,ls='-',label='Chance = training mean')],
              loc='lower left',bbox_to_anchor=(.01,.02),fontsize=5.4,frameon=False)
    # D: held-out R² across every readout; same method order and row alignment as C.
    d=p(fig,106,183,77,95,'D','What carries the held-out signal')
    d.set_axis_off()
    ax2=d.inset_axes([.06,.13,.90,.74])
    for i,m in enumerate(order_m):
        vals=readouts.loc[readouts.method.eq(m),'median_r2'].to_numpy()
        col=INK if m.startswith('Threadfin') else GREY
        ax2.scatter(vals,i+np.linspace(-.22,.22,len(vals)),s=7,color=col,alpha=.75,linewidths=0,zorder=3)
        ax2.plot([np.median(vals)],[i],marker='|',ms=8,mew=1.3,color=INK if m.startswith('Threadfin') else '#6b7480',zorder=4)
    ax2.axvline(0,color=GREY,lw=.9,ls='--',zorder=1)
    ax2.text(0,-.9,'chance',fontsize=5.0,color=INK,ha='center',va='bottom')
    ax2.set_yticks([]);ax2.invert_yaxis();ax2.set_xlim(-.32,.85)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlabel('Held-out R² across 8 readouts (tick: median)',fontsize=5.8)
    ax2.tick_params(labelsize=5.4)
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
