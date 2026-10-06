#!/usr/bin/env python3
"""GC-focused publication figures from committed tables, with auditable selection.

Run from any directory. Every map reuses saved coordinates; no UMAP inference.
All diagrams are vector; cell clouds are rasterized inside font-embedded PDFs.
"""
from __future__ import annotations
import json
import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import schematics as sk
from experimental_designs import model_antigen, plasmodium
from legacy_panels import new_page, panel, save, lineage_tree_gallery

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'case_studies/results'
SHARE = DATA / 'clone_state_sharing'
INK = '#26313b'; GREY = '#b9bec5'; PALE = '#f3f5f7'
BLUE = '#3478ad'; RED = '#c65359'; GREEN = '#4b9a8b'; GOLD = '#d9a643'; VIOLET = '#8b6da5'
STATE_COLORS = {'GC': BLUE, 'PB': RED, 'Memory': GREEN, 'RMB': GREEN, 'LNPC': GOLD,
                'Naive': '#b9bec5', 'Other': '#d5d8dd', 'dark zone': BLUE,
                'light zone': GOLD, 'Myc+ light zone': GREEN, 'plasma cell': RED}
DATASET_LABELS={'mouse_np':'NP-OVA','mouse_rbd':'RBD','gc_np_pc':'NP PC/GC',
                'malaria':'PcAS early','malaria_late':'PcAS late','ln_vaccine':'Human GC',
                'bone_marrow_pc':'Marrow'}
AUDIT = {'map_note': 'Saved UMAP coordinates; distances are visualization, not lineage evidence.',
         'selected_clones': {}, 'sources': {}, 'outputs': []}
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})

@lru_cache(None)
def cells(ds):
    p = DATA / ds / 'cells.csv.gz'
    AUDIT['sources'][str(p.relative_to(ROOT))] = int(pd.read_csv(p, usecols=[0]).shape[0])
    return pd.read_csv(p, index_col=0)

@lru_cache(None)
def clones(ds):
    return pd.read_csv(DATA / ds / 'clone_table.csv', index_col=0)

@lru_cache(None)
def summary(ds):
    return json.loads((DATA / ds / 'summary.json').read_text())


def p(fig, x, y, w, h, letter, title):
    a = panel(fig, x, y, w, h, letter, letter_dx=-3)
    a.set_title(title, loc='left', pad=7, fontweight='normal')
    return a


def note(ax, text, y=-.10, size=6):
    ax.text(0, y, text, transform=ax.transAxes, ha='left', va='top', fontsize=size,
            color=INK, linespacing=1.3)


def legend(ax, palette, ncol=3, y=-.04):
    handles = [Line2D([], [], marker='o', ls='', ms=4, color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, y), borderaxespad=0,
              ncol=ncol, columnspacing=.9, handletextpad=.3)


def map_axis(ax):
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def cell_map(ax, ds, col, palette, title=None, subset=None):
    d = cells(ds) if subset is None else subset
    d = d.dropna(subset=['umap_1', 'umap_2'])
    vals = d[col].fillna('Other').astype(str)
    vals = vals.where(vals.isin(palette), 'Other')
    pal = {'Other': '#d5d8dd', **palette}
    for key in pal:
        q = d[vals.eq(key)]
        if len(q): ax.scatter(q.umap_1, q.umap_2, s=.8, color=pal[key], alpha=.55,
                             linewidths=0, rasterized=True)
    map_axis(ax)
    note(ax, f'{len(d):,} cells; cell UMAP', -.01)


def color_map(ax, ds, column, label, vmin=0, vmax=1, cmap='viridis', note_y=-.30):
    d = clones(ds).dropna(subset=['x', 'y', column])
    q = d[d.reliability >= .5]
    im = ax.scatter(q.x, q.y, c=q[column], s=1+5*np.sqrt(q.n_cells), cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='white', linewidths=.18, alpha=.9, rasterized=True)
    map_axis(ax)
    cb = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=.045, pad=.055, aspect=24)
    cb.set_label(label, fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    note(ax, f'{len(q):,} reliable clones; clone UMAP', note_y)
    return q


def annotated_clone_map(ax, ds, column, palette):
    """Same saved clone coordinates; labels summarise observed states, not ancestry."""
    t=clones(ds).dropna(subset=['x','y']).copy();t=t[t.reliability>=.5]
    d=cells(ds).dropna(subset=['clone_id']).copy()
    label=d[column].fillna('Other').where(d[column].isin(palette),'Other')
    c=pd.crosstab(d.clone_id,label).reindex(t.index,fill_value=0)
    dominant=c.idxmax(axis=1);pal={'Other':'#d1d5db',**palette}
    for key,color in pal.items():
        q=t[dominant.eq(key)];area=np.clip(2+3*np.sqrt(q.n_cells),5,55)
        ax.scatter(q.x,q.y,s=area,color=color,edgecolors='white',linewidths=.1,alpha=.82,rasterized=True)
    if ds=='ln_vaccine':
        q=t[t.spike_binding.eq('S+')].sort_values('n_cells',ascending=False,kind='stable').head(12)
        ax.scatter(q.x,q.y,s=np.clip(2+3*np.sqrt(q.n_cells),5,55)+7,facecolors='none',edgecolors=INK,linewidths=.6)
        AUDIT['selected_clones']['human_spike_rings']={'rule':'12 largest reliable spike-positive families','ids':q.index.tolist()}
    # Label the median of each observed major region; a label is not a tested split.
    for key,color in palette.items():
        q=t[dominant.eq(key)]
        if len(q)<5:continue
        text={'GC':'GC-rich','PB':'PB-rich','LNPC':'LN-PC-rich','RMB':'RMB-rich',
              'plasma cells':'PC sort','memory B cells':'Memory sort'}.get(key,key)
        ax.text(q.x.median(),q.y.median(),text,fontsize=5.4,color=INK,ha='center',va='center',
                bbox=dict(boxstyle='round,pad=.2',facecolor='white',edgecolor=color,alpha=.9,lw=.6))
    map_axis(ax)
    labels={('PC sort' if k=='plasma cells' else 'Memory sort' if k=='memory B cells' else k):v for k,v in palette.items()}
    legend(ax,labels,ncol=2,y=-.08)
    text=f'{len(t):,} reliable clones; size reflects capture count.'
    if ds=='ln_vaccine':text+='\nBlack rings: 12 largest spike-positive families.'
    else:text+='\nGrey: combined or antigen-first sort majority.'
    note(ax,text,-.35,5.8)


def card(ax, x, y, w, h, title, body, color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1',
                              facecolor='white',edgecolor=color,lw=.7))
    ax.text(x+2,y+h-2,title,ha='left',va='top',fontsize=7,color=color,fontweight='bold')
    ax.text(x+2,y+h-7,body,ha='left',va='top',fontsize=6.3,color=INK,linespacing=1.35)


def design(ax, kind):
    if kind == 'model':
        return model_antigen(ax)
    if kind == 'malaria':
        return plasmodium(ax)
    sk.canvas(ax,183,52)
    if kind == 'human':
        sk.person(ax,7,37,.9,color=BLUE)
        ax.text(19,43,'BNT162b2 vaccination',fontsize=7,fontweight='bold',color=BLUE)
        ax.text(19,35,'Repeated samples from the same participants',fontsize=6.4)
        ax.plot([86,181],[40,40],color=INK,lw=.7)
        for d in [28,35,60,110,201]:
            x=88+90*(d-28)/173
            ax.plot([x,x],[38,42],color=INK,lw=.7)
            ax.text(x,45,str(d),ha='center',fontsize=6)
        ax.text(181,32,'day after first dose (deposited scRNA/BCR)',ha='right',fontsize=6)
        card(ax,0,1,56,25,'Axillary LN FNA','GC / LN plasma cells / RMB\nRepeated GC snapshots',BLUE)
        card(ax,63,1,56,25,'Paired blood','Plasmablast / memory states\nSame-donor clone linkage',RED)
        card(ax,126,1,57,25,'External receptor labels','Spike probe + expressed mAbs\nSHM + heavy-chain isotype',GREEN)
        ax.text(0,-3,'Longitudinal clone sampling supports persistence; shared ancestry does not identify a parent cell.',fontsize=6.2)
    elif kind == 'marrow':
        sk.person(ax,7,37,.9,color=BLUE)
        ax.text(19,43,'Human marrow + blood',fontsize=7,fontweight='bold',color=BLUE)
        ax.text(19,35,'GSE253857 • verified GEO donor identities',fontsize=6.4)
        card(ax,85,29,98,22,'Primary analysis: 7 known donors',
             '15 single-donor / single-tissue paired libraries\n5 pooled or mixed libraries excluded',GREEN)
        card(ax,0,1,56,25,'Bone marrow','FACS PC / memory / combined\nCD38-high or CD138+ PC gates',BLUE)
        card(ax,63,1,56,25,'Blood','Memory ± plasmablast gates\n3 donors with both tissues',RED)
        card(ax,126,1,57,25,'Antigen-labelled libraries','Spike / tetanus probe sorting\nRecorded sampling gate',GREEN)
        ax.text(0,-3,'Clone membership is constrained to the actual donor; mixed sort gates are not pure cell fates.',fontsize=6.2)


def composition(ax, ds, cols=None):
    d = pd.read_csv(DATA/ds/'programme_composition.csv',index_col=0)
    if cols: d=d.reindex(columns=cols,fill_value=0)
    left=np.zeros(len(d))
    for col in d:
        ax.barh(np.arange(len(d)),d[col],left=left,height=.7,
                color=STATE_COLORS.get(col,'#ccd1d7'),label=col)
        left += d[col].to_numpy()
    ax.set_yticks(np.arange(len(d)),d.index);ax.set_xlim(0,1);ax.invert_yaxis()
    ax.set_xlabel('Fraction of programme cells (descriptive)')


def signatures(ax, ds, names=('germinal centre','dark zone / cycling','light zone','plasma cell','memory')):
    d = pd.read_csv(DATA/ds/'programme_signatures.csv',index_col=0).reindex(columns=names)
    d = d.dropna(axis=1,how='all')
    if d.empty: raise RuntimeError(f'No signatures for {ds}; correct species gene symbols and rerun')
    im=ax.imshow(d.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-1.5,vmax=1.5)
    ax.set_xticks(range(len(d.columns)),[x.replace('germinal centre','GC').replace('dark zone / cycling','DZ / cycling') for x in d.columns],rotation=35,ha='right')
    stability=summary(ds).get('programmes',{}).get('stability',{})
    labels=[x+'*' if stability.get(x,0)<.75 else x for x in d.index]
    ax.set_yticks(range(len(d)),labels)
    cb=ax.figure.colorbar(im,ax=ax,orientation='horizontal',fraction=.05,pad=.37,aspect=25)
    cb.set_label('Expression score; * resampling stability <0.75',fontsize=5.8)


def clone_gallery(ax, ds, col, palette, key, n=4, require=None):
    d=cells(ds).dropna(subset=['clone_id'])
    c=pd.crosstab(d.clone_id,d[col]).reindex(columns=list(palette),fill_value=0)
    eligible=c[(c>0).sum(axis=1)>=2]
    if require: eligible=eligible[(eligible[require]>0).all(axis=1)]
    # Fixed selection rule, independent of embeddings: greatest captured size, then clone ID.
    ranked=eligible.assign(_size=eligible.sum(axis=1)).sort_values(['_size'],ascending=False,kind='stable')
    ids=ranked.head(n).index.tolist()
    AUDIT['selected_clones'][key]={'rule':'>=2 displayed labels; descending captured size; clone-ID tie order','ids':ids}
    ax.set_axis_off()
    for i,cid in enumerate(ids):
        q=d[d.clone_id.eq(cid)]
        sx=(i % 2)*.51; sy=.53 if i<2 else 0
        a=ax.inset_axes([sx,sy,.47,.41])
        a.scatter(d.umap_1,d.umap_2,s=.15,color='#e1e4e8',rasterized=True)
        for label,color in palette.items():
            z=q[q[col].eq(label)];a.scatter(z.umap_1,z.umap_2,s=9,color=color,edgecolors='white',linewidths=.25,rasterized=True)
        map_axis(a)
        a.set_title(f'{cid}\nn={len(q)}',fontsize=5.5,loc='left',pad=0)
    note(ax,'Each inset shows one BCR-defined clone on the same cell UMAP.',-.10)


def memory_dots(ax, ds, keys, labels):
    mem=summary(ds)['memory']
    for i,(key,lab) in enumerate(zip(keys,labels)):
        m=mem[key];lo,hi=m['memory_index_ci'];v=m['memory_index']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,capsize=2,ms=4)
        ax.text(.99,i,f"n={m['n_clones']}",ha='right',va='center',fontsize=6)
    ax.axvline(0,color=GREY,lw=.8,ls='--');ax.set_xlim(-.08,1);ax.set_yticks(range(len(keys)),labels)
    ax.set_xlabel('Clone-state retention (95% interval)');ax.invert_yaxis()


def figure1():
    from clone_concept import integration_story, capabilities_story
    fig=new_page('Figure 1','Threadfin links BCR families to captured germinal-centre states',height=304)
    a=panel(fig,0,2,183,85,'A',letter_dx=-3)
    asset=HERE/'assets/GC_structure_supplied.png'
    a.imshow(plt.imread(asset));a.set_axis_off()
    AUDIT['sources'][str(asset.relative_to(ROOT))]='Author-original GC schematic by Chen Satoshi; ownership confirmed; inserted without pixel editing.'
    AUDIT['figure1_concept']={'panels':['original GC art','traceable paired-cell/family/profile/map example','four visual evidence cards'],
                             'illustrative_cells':18,'families':3,'cells_per_family':6,
                             'ring_compositions':'illustrative, not the algorithmic kernel features',
                             'A_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}
    b=panel(fig,0,94,183,112,'B',letter_dx=-3);integration_story(b)
    c=panel(fig,0,227,183,55,'C',letter_dx=-3);capabilities_story(c)
    save(fig,'Figure_1');AUDIT['outputs'].append('Figure_1')


def figure2():
    from gc_reclustering_panels import main_node_panel, DATA as GC_DATA
    fig=new_page('Figure 2','Model-antigen GC state nodes and independent reporter measurements',
                 'Clone embedding is coloured by unsupervised Leiden clusters; measured gates and RNA provide GC-state interpretation.',height=305)
    design(p(fig,0,6,183,54,'A','Independent NP-OVA and RBD reporter designs'),'model')
    b=p(fig,0,78,126,68,'B','NP-OVA day 14: clone embedding and Leiden reclustering')
    cm=main_node_panel(b)
    AUDIT['sources']['case_studies/results/gc_np_pc_clone_embedding/notebook_seed123_clone_map.csv']=len(cm)
    AUDIT['gc_reclustering']={'primary':'notebook_seed123','n_clones':len(cm),
        'job_id':json.loads((GC_DATA/'summary.json').read_text())['job_id'],
        'graph_uses_gate_labels':False,'plot_colors':'Unsupervised Leiden clone_cluster',
        'parameters':'case_studies/results/gc_np_pc_clone_embedding/notebook_seed123_parameters.json'}
    state=p(fig,134,78,49,68,None,'Measured compartments\nand marker support (S8B–D)')
    state.set_axis_off()
    state.text(0,.95,'Cluster 1: GC selection-associated\n42% Myc+ LZ; 21% DZ\nHigher mean Myc RNA',va='top',fontsize=7,color='#ff7f0e',linespacing=1.35)
    state.text(0,.58,'Cluster 0: LZ / output-enriched\n47% PC; 48% LZ\nHigher plasma-cell module',va='top',fontsize=7,color='#1f77b4',linespacing=1.35)
    state.text(0,.21,'Cluster 2: mixed GC / output.\nMean compartment fractions\nover families; no direction\nor future fate assigned.',va='top',fontsize=6.5,color=INK,linespacing=1.3)
    note(b,'GSE246382: 49 same-mouse sequence-defined families / 260 cells; ≥3 cells per family; area follows capture count.',-.10,6)
    np_note=p(fig,0,176,55,33,None,'Separate reporter evidence')
    np_note.set_axis_off()
    np_note.text(0,.95,'NP-OVA division reporter\n36-hour H2B-mCherry window\n\nC: measured division bias\nD: receptor mutation history\n\nIndependent of the B cohort.',va='top',fontsize=6.8,linespacing=1.4)
    for y,ds,title in [(176,'mouse_np','NP-OVA'),(235,'mouse_rbd','RBD')]:
        col='division_gate:mCherry-low'
        if ds=='mouse_rbd':
            a=p(fig,0,y,55,31,'E',f'{title}: cell division gate')
            cell_map(a,ds,'division_gate',{'mCherry-high':GOLD,'mCherry-low':BLUE})
        b=p(fig,64,y,53,31,'C' if ds=='mouse_np' else 'F',f'{title}: division-associated\nclone-state bias')
        color_map(b,ds,col,'Fraction mCherry-low (≥6 divisions)',note_y=-.50)
        c=p(fig,129,y,54,31,'D' if ds=='mouse_np' else 'G',f'{title}: mutation history\non the same clone map')
        color_map(c,ds,'mutation_frequency','Mean V mutation frequency (clipped)',vmax=.03,cmap='magma',note_y=-.50)
    save(fig,'Figure_2');AUDIT['outputs'].append('Figure_2')


def archived_panel(ax,name):
    """Embed an unmodified author-supplied scientific panel and record its hash."""
    asset=HERE/'assets'/name
    ax.imshow(plt.imread(asset));ax.set_axis_off()
    AUDIT.setdefault('archived_panels',{})[name]={'sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
        'source_manifest':'assets/legacy_gc_provenance.json','new_analysis':False}


def share_plot(ax):
    d=pd.read_csv(SHARE/'primary_joint_sharing.csv')
    # Plot per-mouse absolute excess sharing, never cell-level replicate tests.
    if 'null_type' in d: nk='null_type'
    elif 'null_model' in d: nk='null_model'
    else: nk='null'
    d=d[(d[nk]=='within_mouse_isotype') & (d.threshold==2)]
    d=d[d.cohort.eq('malaria_early') | d.treatment.eq('Saline')]
    for i,pairname in enumerate(['GC–PB','GC–Memory','Memory–PB']):
        a,b=pairname.split('–')
        q=d[(d.state_a==a)&(d.state_b==b)]
        for cohort,dx,col in [('malaria_early',-.13,GOLD),('malaria_late',.13,BLUE)]:
            z=q[q.cohort.eq(cohort)]
            val=z['observed_rate']-z['null_mean_rate']
            val=val.dropna()
            ax.scatter(i+dx+np.linspace(-.045,.045,len(val)),val,s=10,color=col,alpha=.7)
            if len(val):ax.plot([i+dx-.09,i+dx+.09],[val.median()]*2,color=col,lw=1.7)
    ax.axhline(0,color=GREY,lw=.8,ls='--');ax.set_xticks(range(3),['GC–PB','GC–memory','Memory–PB'])
    ax.set_ylabel('Observed − expected joint occupancy')
    legend(ax,{'d4–14 infected':GOLD,'d10–42 saline':BLUE},ncol=2,y=-.22)
    note(ax,'Fixed denominator: all clones with ≥2 captured cells.\nNull preserves mouse, clone size and per-cell isotype.',-.46)


def treatment_plot(ax):
    occ=pd.read_csv(SHARE/'clone_occupancy.csv.gz')
    occ=occ[(occ.cohort=='malaria_late')&(occ.clone_definition=='threadfin')&(occ.n_cells>=2)]
    rows=[]
    for (day,treat,donor),g in occ.groupby(['day','treatment','donor']):
        if treat not in ['Artesunate','Saline']:continue
        q=g[g.GC>0]
        rows.append({'day':int(str(day).lstrip('D')),'treatment':treat,'donor':donor,
                     'fraction':((q.PB>0)|(q.Memory>0)).mean() if len(q) else np.nan,'n_gc_clones':len(q)})
    d=pd.DataFrame(rows)
    for label,dx,col in [('Saline',-.55,BLUE),('Artesunate',.55,GREEN)]:
        q=d[d.treatment.eq(label)]
        ax.scatter(q.day+dx,q.fraction,s=14,color=col,alpha=.8,label=label)
        m=q.groupby('day').fraction.mean();ax.plot(m.index,m.values,color=col,lw=.9)
    ax.set_xticks([10,14,21,28,35,42]);ax.set_xlabel('Day after infection');ax.set_ylabel('GC-bearing clones shared with PB / memory')
    legend(ax,{'Saline':BLUE,'Anti-malarial':GREEN},ncol=2,y=-.24)
    note(ax,'Each dot is one mouse; 3 mice per infected arm/day.\nSparse GC-bearing clones limit treatment interpretation.',-.46)


def figure3():
    fig=new_page('Figure 3','Clonal co-occupancy of GC and output-like states in Plasmodium',
                 'Same-mouse clone membership tests shared ancestry; independent terminal samples define the time course.',height=297)
    design(p(fig,0,6,183,68,'A','Infection, sample preparation and treatment are represented explicitly'),'malaria')
    b=p(fig,0,104,54,47,'B','Early infection:\nPB-biased clone profiles');color_map(b,'malaria','cell_state:PB','Fraction PB cells')
    c=p(fig,65,104,53,47,'C','Later infection:\nGC-biased clone profiles');color_map(c,'malaria_late','cell_state:GC','Fraction GC cells')
    d=p(fig,129,104,54,47,'D','Cell states underlying\nthe later clone map');cell_map(d,'malaria_late','cell_state',{'GC':BLUE,'PB':RED,'Memory':GREEN})
    legend(d,{'GC':BLUE,'PB':RED,'Memory-like':GREEN,'Other':GREY},ncol=2,y=-.14)
    e=p(fig,0,191,86,58,'E','Co-observed cells of individual GC–PB clones')
    clone_gallery(e,'malaria_late','cell_state',{'GC':BLUE,'PB':RED,'Memory':GREEN},'malaria_gc_pb',require=['GC','PB'])
    f=p(fig,103,191,80,45,'F','Is shared occupancy more than\nexpected from isotype composition?');share_plot(f)
    save(fig,'Figure_3');AUDIT['outputs'].append('Figure_3')


def lineage_presence(ax):
    d=cells('ln_vaccine').dropna(subset=['clone_id'])
    d=d[d.timepoint.isin(['d28','d35','d60','d110','d201'])]
    gc=d[d.state.eq('GC')].groupby('clone_id').timepoint.nunique()
    sizes=d.groupby('clone_id').size()
    ids=gc[gc>=3].index
    ids=sizes.reindex(ids).sort_values(ascending=False,kind='stable').head(4).index.tolist()
    AUDIT['selected_clones']['human_repeated_gc']={'rule':'GC captured at >=3 distinct nonpooled dates; largest total captured sizes','ids':ids}
    dates=['d28','d35','d60','d110','d201'];states=['GC','LNPC','PB','RMB']
    ax.set_axis_off()
    for i,cid in enumerate(ids):
        a=ax.inset_axes([0,1-(i+1)*.25,.80,.17]); q=d[d.clone_id.eq(cid)]
        c=pd.crosstab(q.timepoint,q.state).reindex(index=dates,columns=states,fill_value=0)
        den=c.sum(axis=1);fr=c.div(den.replace(0,np.nan),axis=0).fillna(0);bottom=np.zeros(5)
        for s in states:a.bar(range(5),fr[s],bottom=bottom,color=STATE_COLORS[s],width=.65);bottom+=fr[s]
        a.set_ylim(0,1);a.set_yticks([]);a.set_xticks(range(5),[x[1:] for x in dates] if i==len(ids)-1 else [])
        a.text(1.02,.5,cid,transform=a.transAxes,fontsize=5.1,va='center')
        for j,n in enumerate(den):a.text(j,1.04,str(int(n)) if n else 'NA',ha='center',fontsize=5)
    note(ax,'Captured state fractions per date; counts above bars; NA = no captured family cells.\nBars show repeated membership, without assigning parent–offspring direction.',-.14)


def figure4():
    fig=new_page('Figure 4','Testing GC clone persistence and compartment bias in humans',
                 'Repeat sampling, author-identified binding labels and mutation history provide different measurements.',height=367)
    design(p(fig,0,6,183,52,'A','Repeated lymph-node aspiration and blood sampling after mRNA vaccination'),'human')
    b=p(fig,0,86,54,53,'B','GC and output compartments');cell_map(b,'ln_vaccine','state',{'GC':BLUE,'LNPC':GOLD,'PB':RED,'RMB':GREEN})
    legend(b,{'GC':BLUE,'LNPC':GOLD,'PB':RED,'RMB':GREEN},ncol=2,y=-.12)
    c=p(fig,65,86,53,53,'C','Clone families by captured state')
    annotated_clone_map(c,'ln_vaccine','state',{'GC':BLUE,'LNPC':GOLD,'PB':RED,'RMB':GREEN})
    d=p(fig,135,86,48,53,'D','Are expression programmes\nenriched for S+ families?');binding_enrichment(d)
    e=p(fig,0,188,88,57,'E','GC-containing clones sampled repeatedly');lineage_presence(e)
    f=p(fig,111,188,72,43,'F','What does a clone retain?')
    memory_dots(f,'ln_vaccine',['timepoint','tissue'],['Across dates','LN vs blood'])
    note(f,'Retained state is observational persistence.\nIt does not demonstrate memory GC re-entry.',-.31)
    g=p(fig,8,280,96,43,'G','SHM of captured GC families: binding labels are compared explicitly')
    gc_shm_curve(g)
    h=p(fig,129,280,54,43,'H','What does this comparison measure?')
    h.set_axis_off();h.text(0,.98,'Threadfin: sequence-defined families\nRNA: captured family state profiles\nS+: authors’ binding classification\nSHM: V-region mutation frequency\n\nThese are associated measurements.\nSHM and binary binding are not affinity.\nEarly dates: one nonpooled donor.',va='top',fontsize=6,linespacing=1.5)
    save(fig,'Figure_4');AUDIT['outputs'].append('Figure_4')


def binding_enrichment(ax):
    t=pd.read_csv(DATA/'ln_vaccine/programme_associations.csv')
    d=t[t.label.eq('spike_binding') & t.level.eq('S+')].sort_values('programme')
    y=np.arange(len(d))
    ax.errorbar(d.odds_ratio,y,xerr=[d.odds_ratio-d.ci_low,d.ci_high-d.odds_ratio],
                fmt='o',ms=3,color=VIOLET,elinewidth=.7,capsize=1.4)
    ax.axvline(1,color=GREY,lw=.7);ax.set_xscale('log');ax.set_xlim(.001,8)
    stability=summary('ln_vaccine').get('programmes',{}).get('stability',{})
    ax.set_yticks(y,[v+'*' if stability.get(v,0)<.75 else v for v in d.programme]);ax.invert_yaxis();ax.set_xticks([.01,.1,1],[.01,.1,1])
    ax.set_xlabel('S+ odds within donors\n(programme vs elsewhere)',fontsize=5.8)
    note(ax,'* Programme stability <0.75.\nIntervals resample families, not donors.\nNo quantitative affinity is measured.',-.37,5.5)


def gc_shm_curve(ax, donor_lines=False):
    folder=DATA/'spike_binding_audit'
    t=pd.read_csv(folder/'gc_shm_curve_summary.csv')
    ds=pd.read_csv(folder/'gc_donor_timepoint_shm.csv')
    dates=['d28','d35','d60','d110','d201'];days=[28,35,60,110,201]
    for key,col,lab in [('S+',VIOLET,'Author-identified S+'),('S-',GREY,'Not identified S+')]:
        q=t[t.spike_binding.eq(key)].set_index('timepoint').reindex(dates)
        x=np.asarray(days);m=100*q.equal_donor_mean.to_numpy()
        if donor_lines:
            for _,z in ds[ds.spike_binding.eq(key) & ds.included_in_paired_donor_summary].groupby('donor'):
                z=z.set_index('timepoint').reindex(dates)
                ax.plot(x,100*z.median_family_shm,color=col,lw=.5,alpha=.32)
        ax.plot(x,m,'o-',color=col,ms=2.5,lw=1,label=lab)
        ax.fill_between(x,100*q.donor_q25,100*q.donor_q75,color=col,alpha=.16,lw=0)
    ax.set_xticks(days,[28,35,60,110,201]);ax.set_xlabel('Days after first vaccination')
    ax.set_ylabel('V-region SHM (%)');ax.legend(frameon=False,fontsize=5.6,loc='upper left')
    counts=t[t.spike_binding.eq('S+')].set_index('timepoint').reindex(dates).n_donors.fillna(0)
    note(ax,'Equal-donor mean of family-SHM medians; shading = donor IQR.\nDonors at each date: '+', '.join(str(int(n)) for n in counts)+'. Different families can contribute at different dates.',-.29,5.7)


def supplementary6():
    fig=new_page('Supplementary Figure 6','Binding-label provenance and donor-level GC mutation trends',
                 'Sequence identity, author clone annotation and measured SHM are audited separately.',height=244)
    a=p(fig,9,10,83,53,'A','How author clones map to Threadfin families')
    z=pd.read_csv(DATA/'spike_binding_audit/author_clone_crosswalk.csv')
    counts=z.n_threadfin_families.value_counts().sort_index()
    a.bar(counts.index,counts.values,color=VIOLET);a.set_yscale('log')
    a.set_xlabel('Threadfin families per author clone');a.set_ylabel('Author clones (log scale)')
    note(a,'623 author clones are split; each Threadfin family maps to one author clone.\nA propagated binding label is not a newly measured biological replicate.',-.32,5.7)
    b=p(fig,113,10,70,53,'B','Donors contribute different programme composition')
    q=pd.read_csv(DATA/'spike_binding_audit/donor_programme_binding_coverage.csv')
    labels=sorted(q.programme.unique());rng=np.random.default_rng(0)
    for i,prog in enumerate(labels):
        v=q[q.programme.eq(prog)].frac_Splus_among_known
        b.scatter(i+rng.uniform(-.10,.10,len(v)),100*v,s=11,color=VIOLET,alpha=.65)
    b.set_xticks(range(len(labels)),labels);b.set_ylabel('Author-identified S+ families (%)')
    note(b,'One dot per donor/programme; family-level labels.\nFALSE is not a uniformly assayed negative.',-.32,5.7)
    c=p(fig,9,118,83,56,'C','GC SHM trends with individual donors visible');gc_shm_curve(c,donor_lines=True)
    d=p(fig,119,118,64,56,'D','Expression signatures describe the same captured states');signatures(d,'ln_vaccine')
    footer=panel(fig,119,201,64,10);footer.set_axis_off()
    footer.text(0,1,'Modules reuse profile-building expression.\nAgreement is descriptive, not independent validation.',
                transform=footer.transAxes,va='top',fontsize=5.7,color=INK)
    save(fig,'Supplementary_6');AUDIT['outputs'].append('Supplementary_6')


def marrow_examples(ax):
    d=cells('bone_marrow_pc').dropna(subset=['clone_id'])
    d=d[d.sorted_as.isin(['plasma cells','memory B cells'])]
    c=pd.crosstab(d.clone_id,d.sorted_as).reindex(columns=['plasma cells','memory B cells'],fill_value=0)
    q=c[(c>0).all(axis=1)];ids=q.sum(axis=1).sort_values(ascending=False,kind='stable').head(5).index
    AUDIT['selected_clones']['marrow_pure_sort_shared']={'rule':'captured in both pure PC and pure memory sorts; descending combined size','ids':ids.tolist()}
    t=q.loc[ids].div(q.loc[ids].sum(axis=1),axis=0)
    ax.barh(range(len(t)),t['plasma cells'],color=RED,label='PC sort')
    ax.barh(range(len(t)),t['memory B cells'],left=t['plasma cells'],color=GREEN,label='Memory sort')
    labels=[cid.replace('donor ','').replace('|',' · ') for cid in ids]
    ax.set_yticks(range(len(t)),labels,fontsize=5.5);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Captured cells within each clone');legend(ax,{'PC sort':RED,'Memory sort':GREEN},ncol=2,y=-.26)


def supplementary5():
    fig=new_page('Supplementary Figure 5','Non-GC validation across marrow and blood',
                 'Verified single-donor libraries test whether ancestry and terminal state remain distinguishable.',height=276)
    design(p(fig,0,6,183,52,'A','Compartment and antigen sorting are represented explicitly'),'marrow')
    b=p(fig,0,86,54,52,'B','Cell compartments');cell_map(b,'bone_marrow_pc','sorted_as',{'plasma cells':RED,'memory B cells':GREEN,'plasma cells and memory B cells':GOLD,'plasmablasts and memory B cells':VIOLET})
    legend(b,{'PC sort':RED,'Memory sort':GREEN,'Combined BM':GOLD,'Combined blood':VIOLET},ncol=2,y=-.12)
    c=p(fig,65,86,53,52,'C','Clone families by measured sort gate')
    annotated_clone_map(c,'bone_marrow_pc','sorted_as',{'plasma cells':RED,'memory B cells':GREEN})
    d=p(fig,130,86,53,52,'D','Descriptive expression signatures');signatures(d,'bone_marrow_pc',('plasma cell','memory','naive','light zone'))
    e=p(fig,13,180,75,43,'E','Shared clones occupy\ndistinct measured gates');marrow_examples(e)
    f=p(fig,112,180,71,43,'F','Identical heavy + light receptors\nacross pure gates')
    t=pd.read_csv(DATA/'bone_marrow_pure_gate_check/donor_validation.csv')
    q=t[t.receptor_definition.eq('exact_IGH_plus_light')].copy()
    q['comparison']=q.left_source+'_vs_'+q.right_source
    for donor,dx,col in [('1681',-.16,BLUE),('1684',.16,GREEN)]:
        z=q[q.donor.astype(str).eq(donor)].set_index('comparison')
        vals=[z.loc[key,'shared_groups'] for key in ['PC_BM_vs_Memory_BM','PC_BM_vs_Memory_blood']]
        f.barh(np.arange(2)+dx,vals,height=.28,color=col)
        for i,v in enumerate(vals):f.text(v+3,i+dx,str(int(v)),va='center',fontsize=6)
    f.set_yticks(range(2),['PC + BM memory','PC + blood memory']);f.set_xlim(0,260)
    f.set_xlabel('Exact H+L groups captured in both gates')
    legend(f,{'Donor 1681':BLUE,'Donor 1684':GREEN},ncol=2,y=-.27)
    note(f,'Unique productive heavy + light chain per cell.\nThis supports common receptor identity, not fate direction.',-.51)
    save(fig,'Supplementary_5');AUDIT['outputs'].append('Supplementary_5')


def figure5():
    """Separate multi-study clonal expression signal from biological validation."""
    folder=DATA/'clonal_information_summary'
    t=pd.read_csv(folder/'dataset_evidence.csv')
    g=pd.read_csv(folder/'module_evidence.csv')
    fig=new_page('Figure 5','What receptor-defined families add to the captured expression landscape',
                 'Twelve dataset analyses contribute different evidence; expression resemblance alone does not validate fate.',height=312)
    a=p(fig,0,6,183,51,'A','Biological anchors determine the question each model can answer')
    sk.canvas(a,183,51)
    from concept_figure import bcell, BLUE as CB, GREEN as CG, RED as CR
    entries=[(0,36,'Model antigens','NP-OVA / RBD','Division / GC gates',CB),
             (39,36,'Infection','PcAS early / late','Same-mouse states',CR),
             (78,33,'Human vaccine','Repeated LN samples','Persistence / binding',CG),
             (114,32,'Non-GC anchor','Marrow / blood','Pure PC / memory',VIOLET),
             (149,34,'Tested coverage','Five further datasets','Exploratory states',GREY)]
    for x,w,title,body,evidence,col in entries:
        a.add_patch(FancyBboxPatch((x,5),w,42,boxstyle='round,pad=.25,rounding_size=1.4',fc=col+'0d',ec=col,lw=.7))
        bcell(a,x+w/2,37,3,col)
        a.text(x+w/2,27,title,ha='center',fontsize=6.4,fontweight='bold',color=INK)
        a.text(x+w/2,18,body,ha='center',fontsize=5.7,color=INK)
        a.text(x+w/2,10,evidence,ha='center',fontsize=5.4,color=INK)
    b=p(fig,29,78,91,87,'B','Do related cells resemble each other beyond the library baseline?')
    y=np.arange(len(t));cols=[BLUE if x=='GC anchors' else VIOLET if x=='Non-GC anchor' else GREY for x in t.evidence_tier]
    for i,r in t.iterrows():
        b.plot([100*r.shuffled,100*r.observed],[i,i],color=cols[i],lw=1.2)
        b.plot([100*r.shuffled,100*r.shuffled_q95],[i,i],color=INK,lw=.7)
    b.scatter(100*t.shuffled,y,marker='o',s=19,facecolors='white',edgecolors=INK,lw=.7,zorder=3,label='Shuffled mean')
    b.scatter(100*t.observed,y,s=23,c=cols,edgecolors='white',lw=.3,zorder=4,label='Observed')
    b.set_yticks(y,t.label);b.invert_yaxis();b.set_xlim(0,65)
    b.set_xlabel('Expression variation associated with clone identity (%)')
    b.axhline(5.5,color='#dde1e5',lw=.6);b.axhline(6.5,color='#dde1e5',lw=.6)
    b.legend(loc='upper left',bbox_to_anchor=(0,-.26),ncol=2,frameon=False)
    note(b,'500 within-library shuffles preserve clone sizes and library composition.\nShort black segment: shuffled mean to 95th percentile. NP PC/GC: p = 0.224.',-.44,5.8)
    c=p(fig,139,78,44,87,'C','Captured profiles')
    c.set_title('Captured profiles',loc='left',pad=21)
    c.set_xlim(0,1);c.set_ylim(len(t)-.5,-.5);c.set_axis_off()
    c.text(.03,-.95,'Expanded',fontsize=5.8);c.text(.60,-.95,'Reliable',fontsize=5.8)
    for i,r in t.iterrows():
        c.text(.17,i,f'{r.expanded_clones:,}',ha='center',va='center',fontsize=6.2,color=cols[i])
        c.text(.79,i,f'{r.reliable_profiles:,}',ha='center',va='center',fontsize=6.2,color=cols[i])
    note(c,'Expanded: ≥2 captured cells.\nReliable: profile reliability ≥0.5.\nSmall profiles can restrict interpretation.',-.19,5.7)
    ds=t.dataset.iloc[:7].tolist();sets=['germinal centre','dark zone / cycling','light zone','plasma cell','memory','interferon']
    q=g[g.dataset.isin(ds)].copy()
    q['contrast']=100*(q.median_icc-q.matched_background_median_icc)
    m=q.pivot(index='dataset',columns='gene_set',values='contrast').reindex(index=ds,columns=sets)
    d=p(fig,29,223,91,43,'D','Which expression programmes are more similar within families?')
    im=d.imshow(m.to_numpy(),cmap='RdBu_r',vmin=-30,vmax=30,aspect='auto',interpolation='nearest')
    d.set_yticks(range(7),t.label.iloc[:7],fontsize=5.8)
    d.set_xticks(range(6),['GC','Cycling','LZ','PC','Memory','IFN'],fontsize=5.8)
    cb=fig.colorbar(im,ax=d,orientation='horizontal',fraction=.07,pad=.24,aspect=28)
    cb.set_label('Clonal resemblance above expression-matched genes (percentage points)',fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5,length=2)
    e=p(fig,138,223,45,43,'E','Read the evidence at its\nexperimental resolution')
    e.set_axis_off()
    e.text(0,.98,'Reporter → recent divisions\n\nSame mouse → state co-occupancy\n\nSame donor, repeat dates → persistence\n\nPure gates + H/L → receptor identity',va='top',fontsize=5.9,linespacing=1.25,color=INK)
    note(e,'No panel measures future fate,\nGC re-entry or binding affinity.',-.28,5.7)
    AUDIT['sources'][str((folder/'dataset_evidence.csv').relative_to(ROOT))]=len(t)
    save(fig,'Figure_5');AUDIT['outputs'].append('Figure_5')


def supplementary1():
    fig=new_page('Supplementary Figure 1','Clone evidence and captured repertoire coverage',height=238)
    dslist=['mouse_np','mouse_rbd','gc_np_pc','malaria','malaria_late','ln_vaccine','bone_marrow_pc']
    a=p(fig,0,9,87,73,'A','Expansion determines what can be interpreted')
    labels=[DATASET_LABELS[x] for x in dslist]
    a.boxplot([np.log10(clones(ds).n_cells) for ds in dslist],vert=False,tick_labels=labels,showfliers=False)
    a.set_xlabel('log10 captured cells per analysed clone')
    b=p(fig,108,9,75,73,'B','Reliable profile coverage')
    vals=[(clones(ds).reliability>=.5).sum() for ds in dslist]
    b.barh(range(len(vals)),vals,color=BLUE);b.set_yticks(range(7),labels);b.set_xscale('log');b.set_xlabel('Clones with reliability ≥0.5')
    c=p(fig,0,111,87,69,'C','Clone coherence with within-library null')
    obs=[summary(ds)['coherence']['icc'] for ds in dslist];nul=[summary(ds)['coherence']['null_mean'] for ds in dslist]
    c.barh(np.arange(7)-.15,obs,height=.28,color=BLUE);c.barh(np.arange(7)+.15,nul,height=.28,color=GREY)
    c.set_yticks(range(7),labels);c.set_xlabel('Expression variance explained by clone identity')
    legend(c,{'Observed':BLUE,'Within-library null':GREY},ncol=2,y=-.25)
    d=p(fig,111,111,72,69,'D','Known limits of the direct fate-sort dataset')
    d.set_axis_off();s=summary('gc_np_pc')
    d.text(0,.95,f"NP-OVA day 14, Smart-seq2\n884 captured cells; 388 in expanded clones\n113 clones with ≥2 cells\nOnly 3 reliable profiles\n\nCoherence p={s['coherence']['p_value']:.2f}\nProgramme inference declined\n\nFACS PC / Myc+ LZ / DZ labels support\ndescriptive membership examples;\nthey do not validate a fate predictor.",va='top',fontsize=7,linespacing=1.6)
    save(fig,'Supplementary_1')


def supplementary2():
    fig=new_page('Supplementary Figure 2','Model-antigen clone relationships and orthogonal GC labels',height=293)
    a=p(fig,0,8,87,62,'A','NP-OVA clones spanning division gates')
    clone_gallery(a,'mouse_np','division_gate',{'mCherry-high':GOLD,'mCherry-low':BLUE},'np_division')
    b=p(fig,103,8,80,62,'B','RBD clones spanning division gates')
    clone_gallery(b,'mouse_rbd','division_gate',{'mCherry-high':GOLD,'mCherry-low':BLUE},'rbd_division')
    c=p(fig,0,102,55,50,'C','RBD protein: antigen binding');color_map(c,'mouse_rbd','rbd_bait:RBD+','Fraction RBD probe-positive')
    d=p(fig,65,102,53,50,'D','RBD mRNA: DZ occupancy');color_map(d,'mouse_rbd','zone_gate:DZ','Fraction sorted dark zone')
    e=p(fig,133,102,50,50,'E','State across GC gates');memory_dots(e,'mouse_rbd',['division_gate','rbd_bait','zone_gate'],['Division','Binding','LZ/DZ'])
    f=p(fig,0,200,87,53,'F','Directly measured NP-OVA output compartments')
    clone_gallery(f,'gc_np_pc','fate',{'dark zone':BLUE,'light zone':GOLD,'Myc+ light zone':GREEN,'plasma cell':RED},'np_direct_fates')
    g=p(fig,106,200,77,53,'G','Library-stratified reporter associations')
    dd=json.loads((DATA/'mouse_gc_deep_dive/summary.json').read_text())
    vals=[dd['np_within_library_profile_vs_division']['r2'],dd['rbd_within_library_profile[RBD protein][division_gate]']['r2'],dd['rbd_within_library_profile[mRNA][division_gate]']['r2']]
    g.barh(range(3),vals,color=[GOLD,GREEN,VIOLET]);g.set_yticks(range(3),['NP division','RBD protein division','RBD mRNA division']);g.set_xlabel('Clone-profile variance explained within libraries')
    save(fig,'Supplementary_2')


def supplementary3():
    fig=new_page('Supplementary Figure 3','Plasmodium sharing controls and candidate selection',height=280)
    a=p(fig,0,9,87,62,'A','Treatment and GC–output sharing');treatment_plot(a)
    b=p(fig,104,9,79,62,'B','GC–memory examples')
    clone_gallery(b,'malaria_late','cell_state',{'GC':BLUE,'PB':RED,'Memory':GREEN},'malaria_gc_memory',require=['GC','Memory'])
    c=p(fig,0,120,87,60,'C','Receptor identity sensitivity (no sampling test)')
    t=pd.read_csv(SHARE/'exact_receptor_sharing.csv')
    # Raw observed counts with all definitions; conservative controls lose SHM relatives.
    q=t[t.pair.eq('GC+PB')]
    defs=sorted(q.clone_definition.unique())
    for i,definition in enumerate(defs):
        z=q[q.clone_definition.eq(definition)]
        for cohort,dx,color in [('malaria_early',-.12,GOLD),('malaria_late',.12,BLUE)]:
            val=z[z.cohort.eq(cohort)]['shared_clones'].sum()
            c.bar(i+dx,val,width=.23,color=color)
    c.set_xticks(range(len(defs)),[{'strict_igh':'Exact H','strict_igh_light':'Exact H+L','threadfin':'Threadfin family'}[x] for x in defs],rotation=25,ha='right');c.set_ylabel('GC–PB shared receptor groups observed')
    legend(c,{'Early':GOLD,'Later':BLUE},ncol=2,y=-.33)
    d=p(fig,110,120,73,60,'D','What the experiment can and cannot establish')
    d.set_axis_off();d.text(0,.97,'Supported\nSame-mouse clone membership\nGC / output state occupancy\nDay- and treatment-associated differences\n\nRequires additional evidence\nMemory re-entry into GC\nParent–offspring direction\nFuture PC or memory fate\nAntigen affinity / functional protection',va='top',fontsize=7,linespacing=1.45)
    e=p(fig,0,226,183,22,'E','Predeclared candidate rule')
    e.set_axis_off();e.text(0,.95,'Examples require both displayed states within one mouse and one time point, then rank by captured size.\nExact IGH and paired-light controls test whether sharing survives conservative sequence matching.\nThe supplement reports clone-size sensitivity and conditional nulls; illustrative examples are not independent validation.',va='top',fontsize=6.8,linespacing=1.6)
    save(fig,'Supplementary_3')


def supplementary4():
    fig=new_page('Supplementary Figure 4','GC and terminal-state biology are complementary records',height=269)
    a=p(fig,0,9,83,69,'A','Longitudinal GC programme composition');composition(a,'ln_vaccine',['GC','PB','LNPC','RMB','Naive'])
    legend(a,{k:STATE_COLORS[k] for k in ['GC','PB','LNPC','RMB']},ncol=4,y=-.24)
    b=p(fig,109,9,74,69,'B','Marrow sampling: explicit donor and sort identity')
    m=pd.read_csv(ROOT/'case_studies/bone_marrow_sample_manifest.tsv',sep='\t',dtype=str)
    q=m[m.primary_analysis.eq('TRUE')]
    counts=pd.crosstab(q.donor,q.tissue).reindex(columns=['bone marrow','blood'],fill_value=0)
    counts.plot.barh(ax=b,color=[BLUE,RED],width=.7,legend=False)
    b.set_ylabel('Verified donor');b.set_xlabel('Single-donor paired libraries');legend(b,{'Bone marrow':BLUE,'Blood':RED},ncol=2,y=-.25)
    c=p(fig,0,117,83,57,'C','Clone-associated expression modules')
    names=['germinal centre','dark zone / cycling','light zone','plasma cell','memory']
    sets=['mouse_np','mouse_rbd','malaria_late','ln_vaccine']
    mat=[]
    for ds in sets:
        tab=pd.read_csv(DATA/ds/'geneset_heritability.csv').set_index('gene_set')
        mat.append(tab.reindex(names).median_icc.to_numpy())
    im=c.imshow(mat,aspect='auto',cmap='Blues');c.set_yticks(range(4),[DATASET_LABELS[x] for x in sets]);c.set_xticks(range(5),['GC','DZ/cycling','LZ','PC','Memory'],rotation=35,ha='right')
    cb=fig.colorbar(im,ax=c,orientation='horizontal',fraction=.05,pad=.3);cb.set_label('Median excess clone-associated gene variance',fontsize=6)
    d=p(fig,106,117,77,57,'D','Lineage and current state remain distinct')
    d.set_axis_off();d.text(0,.95,'Reporter experiments: 923 tested clones\nWithin-clone SHM distance did not predict\ncurrent expression distance.\n\nPower calibration applies to the two reporter\nexperiments, not all GC/output datasets.\n\nA clone profile describes a family-wide bias.\nA sequence tree describes mutation history.\nNeither alone assigns a future cell fate.',va='top',fontsize=7,linespacing=1.5)
    e=p(fig,0,220,183,20,'E','Validation hierarchy')
    e.set_axis_off();e.text(0,.95,'Measured reporter / FACS labels → same-donor sequence membership → conditional controls → external functional test.\nExpression signatures and annotation fractions are interpretations of the measured transcriptome, not held-out outcomes.',va='top',fontsize=6.8,linespacing=1.6)
    save(fig,'Supplementary_4')


def tested_datasets():
    out=HERE/'tested_datasets';out.mkdir(exist_ok=True)
    for ds in ['flu','flu_lung','ebv','tonsil','stephenson']:
        d=cells(ds);s=summary(ds)
        fig=new_page(ds,f'Tested dataset: {ds}','Exploratory coverage; no claim of a demonstrated GC fate mechanism.',height=132)
        a=p(fig,0,10,80,68,'A','Captured cell states')
        key=next(x for x in ['state','leiden_state'] if x in d)
        vals=sorted(d[key].dropna().astype(str).unique());pal={x:plt.get_cmap('tab20')(i%20) for i,x in enumerate(vals)}
        cell_map(a,ds,key,pal)
        b=p(fig,106,10,77,68,'B','Captured clone coverage')
        b.set_axis_off();b.text(0,.95,f"Cells: {len(d):,}\nCells with a clone: {d.clone_id.notna().sum():,}\nExpanded clones: {len(clones(ds)):,}\nInferred programmes: {s.get('programmes',{}).get('n','NA')}\n\nThese are datasets already tested.\nThey are excluded from the main biological\nargument and GC-focused supplements.",va='top',fontsize=7,linespacing=1.6)
        save(fig,ds,folder=out)


def figure6():
    from benchmark_figure import draw
    draw(new_page,p,save)
    AUDIT['outputs'].append('Figure_6')


def supplementary7():
    from benchmark_figure import supplementary
    supplementary(new_page,p,save)
    AUDIT['outputs'].append('Supplementary_7')


def supplementary8():
    from gc_reclustering_panels import cell_gates, myc_panel, markers, DATA as GC_DATA
    fig=new_page('Supplementary Figure 8','Historical Top2a display and current model-antigen GC state evidence',
                 'A is an archived output; B–D are audited real-data GSE246382 views with measured gates and descriptive RNA.',height=297)
    a=p(fig,0,8,96,79,'A','Top2a on the historical clone map')
    archived_panel(a,'legacy_notebook_clone_Top2a.png')
    note(a,'Notebook cell 33 output; cohort and sampling day unresolved.',-.04,6)
    b=p(fig,106,8,77,58,'B','GSE246382: measured cell compartments')
    cell_gates(b)
    note(b,'884 cells; saved cell UMAP, measured FACS gates.\nThese cells supply the frozen same-mouse families in Figure 2B.',-.18,6)
    c=p(fig,0,116,96,55,'C','GSE246382: mean Myc expression per clone')
    myc_panel(c)
    note(c,'New clone coordinates, same 49 families as Figure 2B.\nMyc is averaged over captured members of each family.',-.25,6)
    d=p(fig,111,113,72,77,'D','GSE246382: marker expression by clone cluster')
    source=markers(d)
    AUDIT.setdefault('gc_reclustering_tables',{})[str(source.relative_to(ROOT))]={
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'values':'Mean RNA over clones grouped by unsupervised Leiden clone_cluster'}
    e=p(fig,0,222,183,36,'E','Interpret the model-antigen GC nodes within their evidence')
    e.set_axis_off()
    e.text(0,.95,'Cell-UMAP clone centroids → distance-row UMAP (20 neighbours, min_dist 0.4) → Scanpy graph (15) → Leiden (0.3).\nFigure 2B colours three unsupervised clone clusters; gate labels and marker values do not enter the graph.\nCluster 1 is enriched for Myc+ LZ / DZ capture; cluster 0 has greater PC capture and plasma-cell-module expression.\nCluster 2 contains mixed GC/output states. Cluster annotations describe captured state biases, not future fates.\nA is a separate historical display. B–D do not assign a differentiation direction or extend GC interpretation to non-GC.',
               va='top',fontsize=6.7,linespacing=1.6)
    save(fig,'Supplementary_8');AUDIT['outputs'].append('Supplementary_8')


def main(biological_only=False):
    main_figures=[figure1,figure2,figure3,figure4,figure5]+([] if biological_only else [figure6])
    supplements=[supplementary1,supplementary2,supplementary3,supplementary4,supplementary5,supplementary6]+([] if biological_only else [supplementary7])+[supplementary8]
    for f in main_figures+supplements+[tested_datasets]:f()
    AUDIT['outputs']=[f'Figure_{i}' for i in range(1,6 if biological_only else 7)]+[f'Supplementary_{i}' for i in range(1,7 if biological_only else 8)]+['Supplementary_8']
    AUDIT['benchmark_included']=not biological_only
    (HERE/'figure_audit.json').write_text(json.dumps(AUDIT,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--biological-only',action='store_true',help='Reproduce Figure1-5/S1-6/S8 while native benchmark jobs are pending.')
    main(parser.parse_args().biological_only)
