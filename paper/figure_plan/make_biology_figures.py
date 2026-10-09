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


def clone_axis(ax):
    """Corner axes marking a clone-profile map, to distinguish it from cell UMAPs."""
    ax.annotate('', xy=(.15, .03), xytext=(.02, .03), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color=INK, lw=.7, shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(.02, .15), xytext=(.02, .03), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color=INK, lw=.7, shrinkA=0, shrinkB=0))
    box = dict(boxstyle='round,pad=.12', facecolor='white', edgecolor='none', alpha=.75)
    ax.text(.16, .03, 'Clone UMAP 1', transform=ax.transAxes, fontsize=5.2, va='center', color=INK, bbox=box)
    ax.text(.035, .16, 'Clone UMAP 2', transform=ax.transAxes, fontsize=5.2, va='bottom',
            rotation=90, color=INK, bbox=box)


def arm_labels(ax, t, xcol, ycol, mem_col, push=.34, pos=None):
    """State-arm labels pulled outside the map with leader lines."""
    specs = [('cell_state:GC', 'GC arm'), ('cell_state:PB', 'PB arm'), (mem_col, 'Memory arm')]
    cx0, cy0 = t[xcol].mean(), t[ycol].mean()
    span = max(t[xcol].max() - t[xcol].min(), t[ycol].max() - t[ycol].min())
    for col, lab_ in specs:
        v = t[col]
        hi = t[v >= v.quantile(.9)]
        cx, cy = hi[xcol].mean(), hi[ycol].mean()
        if pos and lab_ in pos:
            tx, ty = pos[lab_]
        else:
            dx, dy = cx - cx0, cy - cy0
            n = np.hypot(dx, dy) or 1
            tx, ty = cx + dx / n * span * push, cy + dy / n * span * push
        ax.annotate(lab_, xy=(cx, cy), xytext=(tx, ty), fontsize=5.6, color=INK, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=GREY, lw=.7, shrinkA=2, shrinkB=2),
                    bbox=dict(boxstyle='round,pad=.2', facecolor='white', edgecolor=GREY, alpha=.9, lw=.6))


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


def color_map(ax, ds, column, label, vmin=0, vmax=1, cmap='viridis', note_y=-.30, coords=None):
    d = clones(ds).dropna(subset=['x', 'y', column])
    q = d[d.reliability >= .5].copy()
    if coords is not None:
        co = coords.reindex(q.index)
        q['x'], q['y'] = co['x'], co['y']
    im = ax.scatter(q.x, q.y, c=q[column], s=1+5*np.sqrt(q.n_cells), cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='white', linewidths=.18, alpha=.9, rasterized=True)
    map_axis(ax); clone_axis(ax)
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
    map_axis(ax); clone_axis(ax)
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
    from figure1_overview import draw, manifest
    fig=new_page('Figure 1','Threadfin connects receptor identity to B-cell state relationships',
                 'Integrating paired scRNA-seq and scBCR-seq to resolve clone-state organisation.',height=212)
    draw(panel(fig,0,8,183,183))
    AUDIT['sources'].pop('paper/figure_plan/assets/GC_structure_supplied.png',None)
    AUDIT['figure1_concept']=manifest()
    AUDIT['figure1_concept']['source']='paper/figure_plan/figure1_overview.py'
    AUDIT['figure1_concept']['source_sha256']=hashlib.sha256((HERE/'figure1_overview.py').read_bytes()).hexdigest()
    save(fig,'Figure_1',dpi=300);AUDIT['outputs'].append('Figure_1')
    AUDIT['figure1_concept']['outputs']={suffix:hashlib.sha256((HERE/('Figure_1.'+suffix)).read_bytes()).hexdigest()
                                       for suffix in ('png','pdf')}


def figure2():
    from gc_reclustering_panels import main_node_panel, selection, AUDIT_DATA as GC_DATA
    fig=new_page('Figure 2','Model-antigen GC state nodes and independent reporter measurements',
                 'Reconstructed clone embedding; Leiden colours, captured-cell sizes and measured GC-state context.',height=294)
    design(p(fig,0,6,183,54,'A','Independent NP-OVA and RBD reporter designs'),'model')
    b=p(fig,0,78,141,80,'B','NP-OVA day 14: clone embedding and Leiden reclustering')
    legend_ax=p(fig,151,80,32,78,None,'')
    cm=main_node_panel(b,legend_ax)
    chosen=selection()
    source=GC_DATA/'maps'/(chosen['name']+'.csv')
    AUDIT['sources'][str(source.relative_to(ROOT))]=len(cm)
    AUDIT['gc_reclustering']={'primary':chosen['name'],'n_clones':len(cm),
        'n_cells':int(cm.n_cells.sum()),'coordinates_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'job_id':chosen['job_id'],'selection_manifest':str((GC_DATA/'selected.json').relative_to(ROOT)),
        'graph_uses_gate_labels':False,'plot_colors':'Unsupervised Leiden clone_cluster',
        'size_encoding':'Area = 6 square points per captured cell',
        'parameters':str(source.with_suffix('.json').relative_to(ROOT))}
    footer=p(fig,0,166,183,6,None,'');footer.set_axis_off()
    footer.text(0,.8,f'GSE246382: {len(cm)} same-mouse V–D–J receptor groups / {int(cm.n_cells.sum())} cells; ≥1 cell/group. '
                'Measured context: S8B–E.',fontsize=6,va='top')
    # Featured row: measured division and mutation history in the RBD model.
    a=p(fig,0,182,57,62,'C','RBD: measured division history per cell\n(mCherry-low = ≥6 divisions / 36 h)')
    cell_map(a,'mouse_rbd','division_gate',{'mCherry-high':GOLD,'mCherry-low':BLUE})
    legend(a,{'mCherry-high':GOLD,'mCherry-low':BLUE},ncol=2,y=-.12)
    b=p(fig,63,182,57,62,'D','RBD: division history per clone family\n(fraction of family mCherry-low)')
    q_div=color_map(b,'mouse_rbd','division_gate:mCherry-low','Fraction mCherry-low (≥6 divisions)',note_y=-.30)
    c=p(fig,126,182,57,62,'E','RBD: V-region mutation history\non the same clone map')
    q_mut=color_map(c,'mouse_rbd','mutation_frequency','Mean V mutation frequency (clipped)',vmax=.03,cmap='magma',note_y=-.30)
    assert q_div.index.equals(q_mut.index) and len(q_div)==381
    assert np.array_equal(q_div[['x','y']].to_numpy(),q_mut[['x','y']].to_numpy())
    reporter_source=DATA/'mouse_rbd/clone_table.csv'
    AUDIT['figure2_reporter_review']={
        'accession':'GSE287123','n_reliable_families':len(q_div),
        'family_definition':'Donor-private IGH V/J and nucleotide-junction sequence families',
        'coordinates_source':str(reporter_source.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(reporter_source.read_bytes()).hexdigest(),
        'same_coordinates_in_D_and_E':True,
        'parameters':{'n_neighbors':15,'min_dist':.1,'spread':1.,'random_state':0},
        'size_encoding':'Area in square points = 1 + 5 * sqrt(captured cells)',
        'parameter_review':'case_studies/results/mouse_rbd_embedding_audit/review/review_summary.json',
        'inference_limit':'Measured-gate association; no temporal fate or comparative method superiority'}
    footer2=p(fig,0,268,183,6,None,'');footer2.set_axis_off()
    footer2.text(0,.8,'GSE287123 RBD arms: 10 H2B-mCherry mice (5 protein, 5 mRNA), 36,188 cells; LZ/DZ sorts in the mRNA arm only.\n'
                 'D/E: 381 reliable sequence-defined IGH families; UMAP k=15, min_dist=0.1, spread=1, seed=0. Parameter audit: S11.',
                 fontsize=6,va='top')
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
    from types import SimpleNamespace
    from figure3_infection import draw
    fig=new_page('Figure 3','Clone-state variation across Plasmodium infection',
                 'Receptor-defined families resolve captured state mixtures and variation within a shared cell annotation.',
                 height=354)
    draw(fig, SimpleNamespace(**globals()))
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


def supplementary9():
    fig=new_page('Supplementary Figure 9','Human GC clone persistence and compartment bias',
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
    save(fig,'Supplementary_9');AUDIT['outputs'].append('Supplementary_9')


def supplementary10():
    fig=new_page('Supplementary Figure 10','Cell-level context of the exploratory non-GC datasets',
                 'Family-level maps carry cell-level maps beside them; the exploratory cohorts quantified in S14 are shown here.',height=190)
    specs=[('ebv','gfp',{'GFP+':GREEN,'GFP-':GREY},'A','EBV organoids: measured GFP per cell'),
           ('flu','timepoint',{'d0':GREY,'d7':BLUE},'B','Influenza blood: sampling day'),
           ('flu_lung','tissue',{'medLN':BLUE,'Lung':GOLD},'C','Influenza lung: tissue of capture'),
           ('tonsil','state',{'GC':BLUE,'DZ GC':VIOLET,'Cycling B':GOLD,'MBC':GREEN,'MBC FCRL4+':GREEN,'Plasmablast':RED},'D','Tonsil: author cell states'),
           ('stephenson','state',{'Plasmablast':RED,'Plasma_cell_IgG':RED,'Plasma_cell_IgA':RED,'B_switched_memory':GREEN,'B_non-switched_memory':GREEN},'E','COVID blood: author cell states')]
    for (ds,col,pal,letter,title),(x,y) in zip(specs,[(0,8),(65,8),(129,8),(0,86),(65,86)]):
        a=p(fig,x,y,54,52,letter,title)
        cell_map(a,ds,col,pal)
        legend(a,dict(pal),ncol=2,y=-.28)
    save(fig,'Supplementary_10');AUDIT['outputs'].append('Supplementary_10')


def supplementary11():
    audit_dir=DATA/'mouse_rbd_embedding_audit'
    ver=json.loads((audit_dir/'verify.json').read_text())
    fig=new_page('Supplementary Figure 11','RBD clone map: measured-anchored zone axis and parameter robustness',
                 'Programme scores reuse expression and are descriptive; sort gates and division history are measured.',height=214)
    t=clones('mouse_rbd');t=t[t.reliability>=.5]
    g=pd.read_csv(DATA/'mouse_rbd/clone_gene_scores.csv',index_col=0)
    d=t.join(g)
    # A/B: programme-score axis on the clone map (expression-derived, descriptive)
    for x,col,letter,title in [(0,'dark zone / cycling','A','Dark-zone / cycling\nprogramme score'),
                               (65,'light zone','B','Light-zone programme score\n(incl. Myc)')]:
        a=p(fig,x,8,55,55,letter,title)
        im=a.scatter(d.x,d.y,c=d[col],s=1+5*np.sqrt(d.n_cells),cmap='viridis',
                     edgecolors='white',linewidths=.18,alpha=.9,rasterized=True)
        map_axis(a); clone_axis(a)
        cb=fig.colorbar(im,ax=a,orientation='horizontal',fraction=.045,pad=.055,aspect=24)
        cb.set_label('Family programme score',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=6,length=2)
        note(a,f'{len(d):,} reliable clones; clone UMAP',-.30)
    # C: measured zone gate over the same map
    c=p(fig,129,8,54,55,'C','Measured DZ sort fraction\n(mRNA arm only)')
    c.scatter(d.x,d.y,s=1+5*np.sqrt(d.n_cells),color='#e1e4e8',linewidths=0,rasterized=True)
    z=d.dropna(subset=['zone_gate:DZ'])
    im=c.scatter(z.x,z.y,c=z['zone_gate:DZ'],s=1+5*np.sqrt(z.n_cells),cmap='plasma',vmin=0,vmax=1,
                 edgecolors='white',linewidths=.18,alpha=.95,rasterized=True)
    map_axis(c); clone_axis(c)
    cb=fig.colorbar(im,ax=c,orientation='horizontal',fraction=.045,pad=.055,aspect=24)
    cb.set_label('Fraction of family sorted DZ',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=6,length=2)
    note(c,f'Grey: all {len(d):,} reliable clones; colour: {len(z)} clones\nwith zone-gated cells (5 mRNA-arm mice).',-.30)
    # D: parameter audit of map association
    sw=pd.read_csv(audit_dir/'sweep.csv')
    dd=p(fig,0,100,104,58,'D','Map associations across 85 embedding\nparameter sets')
    order=[('division_gate:mCherry-low','Division\n(measured)'),('zone_gate:DZ','DZ sort\n(measured)'),
           ('LZ minus DZ','LZ–DZ axis\n(RNA)'),('rbd_bait:RBD+','RBD bait\n(measured)'),('mutation_frequency','SHM\n(measured)')]
    for i,(var,lab_) in enumerate(order):
        zz=sw[sw.variable.eq(var)]
        dd.scatter(i-.17+np.linspace(-.09,.09,len(zz)),zz.excess,s=5,color=BLUE,alpha=.65,linewidths=0)
        dd.scatter(i+.17+np.linspace(-.09,.09,len(zz)),zz.lin_r2,s=5,color=GOLD,alpha=.65,linewidths=0)
        dd.plot([i-.3,i-.04],[zz.excess.median()]*2,color=BLUE,lw=1.6)
        dd.plot([i+.04,i+.3],[zz.lin_r2.median()]*2,color=GOLD,lw=1.6)
    dd.axhline(0,color=GREY,lw=.8,ls='--');dd.set_ylim(-.05,.9)
    dd.set_xticks(range(len(order)),[l for _,l in order],fontsize=5.8)
    dd.set_ylabel('Association with\nmap position',fontsize=5.8)
    legend(dd,{'Local structure (excess kNN EV vs donor null)':BLUE,'Linear gradient (R² of value ~ x+y)':GOLD},ncol=1,y=1.0)
    nsets=sw.groupby(['n_neighbors','min_dist','spread','random_state']).ngroups
    note(dd,f'{nsets} UMAP parameter sets (n_neighbors/min_dist/spread/seed), map rebuilt from committed\nfamily features each time; bars = median. SHM: linear R² ≤ 0.04 in all sets.',-.30,5.6)
    # E: the measured division gradient tracks the programme axis
    at=ver['axis_alignment']
    e=p(fig,116,100,67,58,'E','The measured division gradient tracks\nthe DZ–LZ programme axis')
    arm=np.where(d.donor.str[1:].astype(int)<=5,'RBD protein arm','mRNA arm')
    axv=d['light zone']-d['dark zone / cycling']
    for label,col in [('RBD protein arm',GOLD),('mRNA arm',BLUE)]:
        m=arm==label
        e.scatter(axv[m],d['division_gate:mCherry-low'][m],s=6,color=col,alpha=.6,linewidths=0,rasterized=True)
    e.set_xlabel('LZ − DZ programme score (family)',fontsize=5.8)
    e.set_ylabel('Fraction mCherry-low\n(≥6 divisions)',fontsize=5.8)
    legend(e,{'RBD protein arm':GOLD,'mRNA arm':BLUE},ncol=2,y=-.30)
    note(e,f"r = {at['corr_lz_minus_dz_vs_mcherry_low']:.2f}; donor-stratified permutation "
           f"p = {at['p_value_two_sided']:.3f}; n = {at['n_clones']} families.",-.52,5.6)
    save(fig,'Supplementary_11');AUDIT['outputs'].append('Supplementary_11')


def supplementary12():
    audit_dir=DATA/'malaria_late_embedding_audit'
    fig=new_page('Supplementary Figure 12','The later-infection clone map encodes the infection time course',
                 'Terminal samples; day is a donor-level label (unstratified null); LZ score is expression-derived (descriptive). '
                 'Clone map: UMAP k=50, min_dist 0.5 (see D).',height=318)
    t=clones('malaria_late');t=t[t.reliability>=.5].copy()
    mal_sel=json.loads((DATA/'malaria_late_embedding_audit'/'selected.json').read_text())
    mal_map=pd.read_csv(DATA/'malaria_late_embedding_audit'/'maps'/(mal_sel['name']+'.csv'),index_col=0)
    t['x'],t['y']=mal_map.reindex(t.index).x,mal_map.reindex(t.index).y
    t['day']=pd.to_numeric(t.donor.astype(str).str.extract(r'D(\d+)')[0])
    c=cells('malaria_late')
    mem=pd.crosstab(c.clone_id,c.cell_state,normalize='index')['Memory']
    t['cell_state:Memory']=mem.reindex(t.index)
    days=[10,14,21,28,35,42]
    # A: clone map coloured by sampling day
    a=p(fig,0,8,57,55,'A','Later-infection clone map\ncoloured by sampling day')
    im=a.scatter(t.x,t.y,c=t.day,s=1+5*np.sqrt(t.n_cells),cmap='plasma',vmin=10,vmax=42,
                 edgecolors='white',linewidths=.18,alpha=.9,rasterized=True)
    map_axis(a); clone_axis(a)
    cb=fig.colorbar(im,ax=a,orientation='horizontal',fraction=.045,pad=.055,aspect=24)
    cb.set_label('Sampling day',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=6,length=2)
    note(a,f'{len(t):,} reliable clones; clone UMAP',-.30)
    arm_labels(a,t,'x','y','cell_state:Memory',pos={'GC arm':(-1.2,9.8),'PB arm':(16.3,4.1),'Memory arm':(-1.2,-1.6)})
    # B: composition turnover across the time course
    b=p(fig,65,8,53,55,'B','Reliable-clone composition\nacross the time course')
    gday=t.groupby('day')[['cell_state:GC','cell_state:PB','cell_state:Memory']].mean().reindex(days)
    for col,lab_,colr in [('cell_state:GC','GC',BLUE),('cell_state:PB','PB',RED),('cell_state:Memory','Memory',GREEN)]:
        b.plot(days,gday[col],marker='o',ms=3.4,color=colr,lw=1.4,label=lab_)
    b.set_xticks(days);b.set_xlabel('Day after infection',fontsize=5.8);b.set_ylabel('Mean fraction of family cells',fontsize=5.8)
    b.set_ylim(-.03,.95)
    legend(b,{'GC':BLUE,'PB':RED,'Memory':GREEN},ncol=3,y=-.30)
    note(b,'Means over reliable clones sampled per day, including controls.\nTerminal samples; per-mouse infected summaries in S13.',-.52,5.6)
    # C: day structure within state strata
    from scipy.spatial import cKDTree
    cc=p(fig,129,8,54,55,'C','Day structure persists\nwithin state strata')
    xy=t[['x','y']].to_numpy();v=t.day.to_numpy(float)
    rng=np.random.default_rng(0)
    def day_assoc(mask):
        q=np.flatnonzero(mask)
        idx=cKDTree(xy[q]).query(xy[q],k=11)[1][:,1:]
        vv=v[q]
        obs=1-((vv-vv[idx].mean(axis=1)).var()/vv.var())
        null=np.empty(300)
        for bi in range(300):
            vp=rng.permutation(vv);null[bi]=1-((vp-vp[idx].mean(axis=1)).var()/vp.var())
        return obs-null.mean(),(1+(null>=obs).sum())/301,len(q)
    strata=[('All',np.ones(len(t),bool)),('GC-rich',(t['cell_state:GC']>=.5).to_numpy()),
            ('PB-rich',(t['cell_state:PB']>=.5).to_numpy()),
            ('Neither',((t['cell_state:GC']<.2)&(t['cell_state:PB']<.2)).to_numpy())]
    vals=[day_assoc(m) for _,m in strata]
    cc.bar(range(4),[x[0] for x in vals],color=[INK,BLUE,RED,GREY],width=.62)
    for i,(excess,pv,n) in enumerate(vals):
        cc.text(i,vals[i][0]+.02,f'{excess:.2f}',ha='center',fontsize=5.6)
        cc.text(i,-.13,f'n={n}',ha='center',fontsize=5.2,color=INK)
    cc.set_xticks(range(4),[s for s,_ in strata],fontsize=5.8);cc.set_ylim(-.16,.85)
    cc.axhline(0,color=GREY,lw=.8,ls='--')
    cc.set_ylabel('Day association\n(excess kNN EV)',fontsize=5.8)
    note(cc,'All strata p=0.003 (300 permutations). Donor-level label;\nunstratified null; state strata defined by family fractions.',-.34,5.6)
    # D: parameter audit
    sw=pd.read_csv(audit_dir/'sweep.csv')
    dd=p(fig,0,108,104,58,'D','Map associations across 85 embedding\nparameter sets')
    order=[('cell_state:GC','GC\nfraction'),('cell_state:PB','PB\nfraction'),('cell_state:Memory','Memory\nfraction'),
           ('mutation_frequency','SHM'),('day','Day\n(donor-level)'),('treated','Treatment\n(donor-level)')]
    for i,(var,lab_) in enumerate(order):
        zz=sw[sw.variable.eq(var)]
        dd.scatter(i-.17+np.linspace(-.09,.09,len(zz)),zz.excess,s=5,color=BLUE,alpha=.65,linewidths=0)
        dd.scatter(i+.17+np.linspace(-.09,.09,len(zz)),zz.lin_r2,s=5,color=GOLD,alpha=.65,linewidths=0)
        dd.plot([i-.3,i-.04],[zz.excess.median()]*2,color=BLUE,lw=1.6)
        dd.plot([i+.04,i+.3],[zz.lin_r2.median()]*2,color=GOLD,lw=1.6)
    dd.axhline(0,color=GREY,lw=.8,ls='--');dd.set_ylim(-.05,.95)
    dd.set_xticks(range(len(order)),[l for _,l in order],fontsize=5.6)
    dd.set_ylabel('Association with\nmap position',fontsize=5.8)
    legend(dd,{'Local structure (excess kNN EV vs null)':BLUE,'Linear gradient (R² of value ~ x+y)':GOLD},ncol=1,y=1.0)
    note(dd,'85 UMAP parameter sets (n_neighbors/min_dist/spread/seed), map rebuilt from committed\nfamily features each time; bars = median.',-.30,5.6)
    # E: SHM accumulates over the chronic infection
    e=p(fig,116,108,67,58,'E','V-region mutation burden\nacross sampled infection dates')
    shm=t.mutation_frequency*100
    e.scatter(t.day+np.linspace(-1.4,1.4,len(t)),shm,s=5,color=GREY,alpha=.5,linewidths=0,rasterized=True)
    med=shm.groupby(t.day).median().reindex(days)
    e.plot(days,med,color=RED,lw=1.6,marker='o',ms=3.6)
    e.set_xticks(days);e.set_xlabel('Day after infection',fontsize=5.8)
    e.set_ylabel('Family SHM (%)',fontsize=5.8);e.set_ylim(-.08,4.2)
    note(e,'Red: median per day. Median family SHM rises 0.07% (d10) to\n2.50% (d42); map gradient linear R² = 0.50 (panel D, SHM gold).',-.30,5.6)
    # F: LZ-programme families at the base of the GC arm (Figure 3C selection zone)
    gsc=pd.read_csv(DATA/'malaria_late'/'clone_gene_scores.csv',index_col=0)
    lz=gsc['light zone'].reindex(t.index)
    g2=p(fig,0,202,90,58,'F','LZ-programme families sit at the\nbase of the GC arm')
    im=g2.scatter(t.x,t.y,c=lz,s=1+5*np.sqrt(t.n_cells),cmap='viridis',edgecolors='white',linewidths=.18,alpha=.9,rasterized=True)
    hi=lz>=lz.quantile(.95)
    g2.scatter(t.x[hi],t.y[hi],s=16+5*np.sqrt(t.n_cells[hi]),facecolors='none',edgecolors=RED,linewidths=.6,rasterized=True)
    map_axis(g2); clone_axis(g2)
    arm_labels(g2,t,'x','y','cell_state:Memory',pos={'GC arm':(-1.2,9.8),'PB arm':(16.3,4.1),'Memory arm':(-1.2,-1.6)})
    cb=fig.colorbar(im,ax=g2,orientation='horizontal',fraction=.045,pad=.055,aspect=24)
    cb.set_label('Family LZ programme score (incl. Myc)',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=6,length=2)
    note(g2,'Expression-derived, descriptive. Red rings: top 5% LZ families.',-.30,5.6)
    # G: programme coherence above matched background (moved from the main figure)
    g3=p(fig,100,202,83,58,'G','GC and output programmes are\nclonally coherent above background')
    gg=pd.read_csv(DATA/'clonal_information_summary'/'module_evidence.csv')
    sets=['germinal centre','dark zone / cycling','plasma cell','memory']
    zz=gg[gg.dataset.isin(['malaria','malaria_late'])&gg.gene_set.isin(sets)].copy()
    zz['excess']=100*(zz.median_icc-zz.matched_background_median_icc)
    mm=zz.pivot(index='gene_set',columns='dataset',values='excess').reindex(sets)
    x=np.arange(len(sets))
    g3.bar(x-.17,mm.malaria,width=.34,color=GOLD,label='Early')
    g3.bar(x+.17,mm.malaria_late,width=.34,color=BLUE,label='Later')
    for i,s in enumerate(sets):
        for dx,ds_ in [(-.17,'malaria'),(.17,'malaria_late')]:
            g3.text(i+dx,mm.loc[s,ds_]+.6,f'{mm.loc[s,ds_]:.0f}',ha='center',fontsize=5.4)
    g3.set_xticks(x,['GC','Cycling','PC','Memory'],fontsize=5.8)
    g3.set_ylabel('Clonal resemblance above\nmatched genes (pp)',fontsize=5.8)
    g3.legend(frameon=False,fontsize=5.6,loc='upper right')
    note(g3,'Family-level expression resemblance of each programme versus\nexpression-matched background genes (both infection cohorts).',-.34,5.6)
    save(fig,'Supplementary_12');AUDIT['outputs'].append('Supplementary_12')


def supplementary13():
    from types import SimpleNamespace
    from malaria_three_arms import draw
    fig=new_page('Supplementary Figure 13','Three state-enriched regions in the later-infection family map',
                 'GC, plasmablast and memory occupancy shown separately on identical clone coordinates.',height=250)
    draw(fig,SimpleNamespace(**globals()))
    save(fig,'Supplementary_13');AUDIT['outputs'].append('Supplementary_13')


def supplementary14():
    """Non-GC validation: pure-gate receptor identity and explanatory power outside the GC."""
    fig=new_page('Supplementary Figure 14','Receptor identity and expression across additional B-cell systems',
                 'Pure marrow/blood gates, exact receptor identity and additional B-cell datasets test the same framework.',height=320)
    summary_t=pd.read_csv(DATA/'clonal_information_summary/dataset_evidence.csv')
    nongc=summary_t[summary_t.evidence_tier.ne('GC anchors')].reset_index(drop=True)
    # A: quantitative overview of the six non-GC analyses
    a=p(fig,0,6,183,42,'A','Marrow/blood validation and five additional datasets')
    a.set_xlim(0,183);a.set_ylim(len(nongc)-.3,-1.3);a.set_axis_off()
    maxfam=nongc.expanded_clones.max()
    for i,r in nongc.iterrows():
        col=VIOLET if r.evidence_tier=='Non-GC anchor' else GREY
        a.text(0,i,r.label,fontsize=6,va='center',color=INK)
        w=95*np.log10(r.expanded_clones)/np.log10(maxfam)
        a.add_patch(FancyBboxPatch((40,i-.30),w,.60,boxstyle='round,pad=.02',fc=col+'33',ec='none'))
        a.text(40+w+2,i,f'{r.expanded_clones:,} families · {r.n_donors} donors · {r.n_cells:,} cells',fontsize=5.6,va='center',color=INK)
    a.text(40,-1.05,'Expanded families (≥2 captured cells; log-scaled bar)',fontsize=5.6,color=INK)
    # B: marrow cell map by measured sort gate
    b=p(fig,0,56,88,62,'B','Marrow/blood cells by measured sort gate')
    cell_map(b,'bone_marrow_pc','sorted_as',{'plasma cells':RED,'memory B cells':GREEN,'plasma cells and memory B cells':GOLD,'plasmablasts and memory B cells':VIOLET})
    legend(b,{'PC':RED,'Memory':GREEN,'Comb. BM':GOLD,'Comb. blood':VIOLET},ncol=4,y=-.07)
    note(b,'Pure gates are measured, not inferred.',-.14,5.8)
    # C: marrow family map by plasma-cell capture fraction
    c=p(fig,95,56,88,62,'C','Families by plasma-cell capture')
    color_map(c,'bone_marrow_pc','sorted_as:plasma cells','Fraction of family in pure PC sort',note_y=-.22)
    # D: exact heavy+light receptor identity across pure gates
    d=p(fig,30,150,58,56,'D','Identical H+L receptors across pure gates')
    t=pd.read_csv(DATA/'bone_marrow_pure_gate_check/donor_validation.csv')
    q=t[t.receptor_definition.eq('exact_IGH_plus_light')].copy()
    q['comparison']=q.left_source+'_vs_'+q.right_source
    comps=['PC_BM_vs_Memory_BM','PC_BM_vs_Memory_blood','Memory_BM_vs_Memory_blood']
    labels=['PC + memory (BM)','PC (BM) + memory (blood)','Memory (BM) + memory (blood)']
    for donor,dx,col in [('1681',-.17,BLUE),('1684',.17,GOLD)]:
        z=q[q.donor.astype(str).eq(donor)].set_index('comparison')
        vals=[z.loc[key,'shared_groups'] for key in comps]
        frac=[z.loc[key,'shared_groups']/z.loc[key,'eligible_groups_ge2_cells'] for key in comps]
        d.barh(np.arange(3)+dx,vals,height=.30,color=col)
        for i,(v,fr) in enumerate(zip(vals,frac)):
            if v>=80:
                d.text(v-4,i+dx,f'{int(v)} ({100*fr:.0f}%)',va='center',ha='right',fontsize=5.5,color='white',fontweight='bold')
            else:
                d.text(v+3,i+dx,f'{int(v)} ({100*fr:.0f}%)',va='center',fontsize=5.5)
            d.text(3,i+dx,donor,va='center',fontsize=4.8,color='white',fontweight='bold')
    d.set_yticks(range(3),labels,fontsize=5.4);d.set_xlim(0,315)
    d.set_xlabel('Exact H+L receptor groups shared across gates',fontsize=5.8)
    # E: non-GC explanatory power vs shuffle
    e=p(fig,95,150,88,56,'E','Clonal expression signal\nexceeds the library baseline')
    y=np.arange(len(nongc));cols=[VIOLET if x=='Non-GC anchor' else GREY for x in nongc.evidence_tier]
    for i,r in nongc.iterrows():
        e.plot([100*r.shuffled,100*r.observed],[i,i],color=cols[i],lw=1.2)
        e.plot([100*r.shuffled,100*r.shuffled_q95],[i,i],color=INK,lw=.7)
        e.text(100*r.observed+1.2,i,f'{100*r.observed:.0f}%',va='center',fontsize=5.4,color=cols[i])
    e.scatter(100*nongc.shuffled,y,marker='o',s=17,facecolors='white',edgecolors=INK,lw=.7,zorder=3,label='Shuffled mean')
    e.scatter(100*nongc.observed,y,s=21,c=cols,edgecolors='white',lw=.3,zorder=4,label='Observed')
    e.set_yticks(y,nongc.label,fontsize=5.8);e.invert_yaxis();e.set_xlim(0,68)
    e.set_xlabel('Expression variation associated with clone identity (%)',fontsize=5.8)
    e.legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=2,frameon=False,fontsize=5.6)
    # F: non-GC module resemblance
    gdf=pd.read_csv(DATA/'clonal_information_summary/module_evidence.csv')
    nds=['bone_marrow_pc','flu','ebv','tonsil','stephenson']
    sets=['plasma cell','memory','interferon']
    z=gdf[gdf.dataset.isin(nds)&gdf.gene_set.isin(sets)].copy()
    z['contrast']=100*(z.median_icc-z.matched_background_median_icc)
    mm=z.pivot(index='dataset',columns='gene_set',values='contrast').reindex(index=nds,columns=sets)
    f=p(fig,14,228,76,50,'F','Gene modules show clone-associated expression')
    im=f.imshow(mm.to_numpy(),cmap='RdBu_r',vmin=-30,vmax=30,aspect='auto',interpolation='nearest')
    f.set_yticks(range(len(nds)),['Marrow / blood','Influenza blood','EBV organoids','Tonsil','COVID blood'],fontsize=5.8)
    f.set_xticks(range(3),['PC','Memory','IFN'],fontsize=5.8)
    cb=fig.colorbar(im,ax=f,orientation='horizontal',fraction=.07,pad=.26,aspect=26)
    cb.set_label('Clonal resemblance above matched genes (pp)',fontsize=5.6);cb.ax.tick_params(labelsize=5.4,length=2)
    # G: measured non-GC binding label on an exploratory family map
    g=p(fig,95,228,88,50,'G','EBV organoids: measured GFP status')
    color_map(g,'ebv','gfp','Fraction GFP+ cells per family',note_y=-.32)
    save(fig,'Supplementary_14');AUDIT['outputs'].append('Supplementary_14')


def figure4():
    from external_validation_figures import draw_larry
    import sys
    draw_larry(sys.modules[__name__])


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


def supplementary15():
    """Quantify, dataset by dataset, how much expression organisation receptor-defined families explain."""
    folder=DATA/'clonal_information_summary'
    t=pd.read_csv(folder/'dataset_evidence.csv')
    g=pd.read_csv(folder/'module_evidence.csv')
    fig=new_page('Supplementary Figure 15','Quantifying what receptor-defined families explain across twelve analyses',
                 'Sampling, clonal expression signal, module resemblance and their limits; every value is computed.',height=314)
    # A: sampling and reliable coverage, quantitative
    a=p(fig,29,8,124,76,'A','Captured sampling decides how many families can be interpreted')
    y=np.arange(len(t));cols=[BLUE if x=='GC anchors' else VIOLET if x=='Non-GC anchor' else GREY for x in t.evidence_tier]
    a.barh(y,t.expanded_clones,color='#e3e6ea',height=.72)
    a.barh(y,t.reliable_profiles,color=cols,height=.72)
    for i,r in t.iterrows():
        a.text(r.expanded_clones*1.12,i,f'{r.reliable_profiles/max(r.expanded_clones,1)*100:.0f}%',va='center',fontsize=5.4,color=INK)
    a.set_yticks(y,t.label,fontsize=6);a.invert_yaxis();a.set_xscale('log');a.set_xlim(1,4e4)
    a.set_xlabel('Families (log scale): expanded, light; reliable, solid',fontsize=6)
    a.text(.99,.02,'% = reliable (profile reliability ≥0.5)\nof expanded (≥2 captured cells)',transform=a.transAxes,ha='right',va='bottom',fontsize=5.6,color=INK)
    # B: observed vs shuffled clonal expression signal
    b=p(fig,29,100,124,72,'B','Do related cells resemble each other beyond the library baseline?')
    for i,r in t.iterrows():
        b.plot([100*r.shuffled,100*r.observed],[i,i],color=cols[i],lw=1.2)
        b.plot([100*r.shuffled,100*r.shuffled_q95],[i,i],color=INK,lw=.7)
        b.text(100*r.observed+1.2,i,f'{100*r.observed:.0f}%',va='center',fontsize=5.4,color=cols[i])
    b.scatter(100*t.shuffled,y,marker='o',s=19,facecolors='white',edgecolors=INK,lw=.7,zorder=3,label='Shuffled mean')
    b.scatter(100*t.observed,y,s=23,c=cols,edgecolors='white',lw=.3,zorder=4,label='Observed')
    b.scatter([100*t.observed[t.p_value>0.05].iloc[0]],[t.index[t.p_value>0.05][0]],s=44,facecolors='none',edgecolors=INK,lw=.8,zorder=5,label='Not significant')
    b.set_yticks(y,t.label,fontsize=6);b.invert_yaxis();b.set_xlim(0,72)
    b.set_xlabel('Expression variation associated with clone identity (%)',fontsize=6)
    b.axhline(5.5,color='#dde1e5',lw=.6);b.axhline(6.5,color='#dde1e5',lw=.6)
    b.legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=3,frameon=False,fontsize=5.8)
    note(b,'500 within-library shuffles; black segment: shuffled mean–95th percentile.',-.36,5.8)
    # C: module-level clonal resemblance
    ds=t.dataset.iloc[:7].tolist();sets=['germinal centre','dark zone / cycling','light zone','plasma cell','memory','interferon']
    q=g[g.dataset.isin(ds)].copy()
    q['contrast']=100*(q.median_icc-q.matched_background_median_icc)
    m=q.pivot(index='dataset',columns='gene_set',values='contrast').reindex(index=ds,columns=sets)
    c=p(fig,29,216,88,52,'C','Which expression programmes are more similar within families?')
    im=c.imshow(m.to_numpy(),cmap='RdBu_r',vmin=-30,vmax=30,aspect='auto',interpolation='nearest')
    c.set_yticks(range(7),t.label.iloc[:7],fontsize=5.8)
    c.set_xticks(range(6),['GC','Cycling','LZ','PC','Memory','IFN'],fontsize=5.8)
    cb=fig.colorbar(im,ax=c,orientation='horizontal',fraction=.07,pad=.24,aspect=28)
    cb.set_label('Clonal resemblance above expression-matched genes (percentage points)',fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5,length=2)
    # D: what determines detectable explanatory power
    d=p(fig,131,216,52,52,'D','Detectable signal grows with captured sampling')
    for tier,col,mk in [('GC anchors',BLUE,'o'),('Non-GC anchor',VIOLET,'s'),('Tested coverage',GREY,'^')]:
        z=t[t.evidence_tier.eq(tier)]
        d.scatter(z.cells_in_expanded,100*z.excess,s=14+55*np.sqrt(z.reliable_profiles/t.reliable_profiles.max()),
                  color=col,marker=mk,alpha=.85,edgecolors='white',lw=.4,label=tier.replace('Tested coverage','Further coverage'))
    lab={'mouse_np':('NP-OVA',.55,1.5,'right'),'mouse_rbd':('RBD',.55,-3.5,'right'),
         'gc_np_pc':('NP PC/GC',1.2,-3.5,'left'),'malaria':('PcAS early',.8,2.5,'right'),
         'malaria_late':('PcAS late',1.25,1.5,'left'),'ln_vaccine':('Human GC',.7,3.5,'right'),
         'bone_marrow_pc':('Marrow',.75,-4.5,'right'),'flu':('Flu blood',1.2,2.0,'left'),
         'flu_lung':('Flu lung',1.25,1.5,'left'),'ebv':('EBV',1.25,-4.0,'left'),
         'tonsil':('Tonsil',1.25,1.5,'left'),'stephenson':('COVID',1.25,1.5,'left')}
    for i,r in t.iterrows():
        name,fx,fy,ha=lab[r.dataset]
        d.text(r.cells_in_expanded*fx if ha=='left' else r.cells_in_expanded*fx,100*r.excess+fy,name,fontsize=4.9,va='center',ha=ha,color=INK)
    d.set_xscale('log');d.set_xlim(2e2,3e5);d.set_ylim(-6,66)
    d.set_xlabel('Captured cells in expanded families (log)',fontsize=5.8)
    d.set_ylabel('Excess clonal expression\nvariation (%)',fontsize=5.8)
    d.legend(frameon=False,fontsize=5.4,loc='lower right',handletextpad=.1,borderaxespad=0)
    note(d,'Point size: reliable profiles. Open ring in B marks the one\nunderpowered analysis (NP-OVA PC/GC, p = 0.224).',-.42,5.6)
    AUDIT['sources'][str((folder/'dataset_evidence.csv').relative_to(ROOT))]=len(t)
    save(fig,'Supplementary_15');AUDIT['outputs'].append('Supplementary_15')


def figure5():
    from external_validation_figures import draw_nsclc
    import sys
    draw_nsclc(sys.modules[__name__])


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
    fig=new_page('Supplementary Figure 3','Plasmodium sharing controls and candidate selection',height=463)
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
    f=p(fig,0,258,87,62,'F','Is shared occupancy more than\nexpected from isotype composition?');share_plot(f)
    g=p(fig,104,258,79,62,'G','Co-observed cells of individual GC–PB clones')
    clone_gallery(g,'malaria_late','cell_state',{'GC':BLUE,'PB':RED,'Memory':GREEN},'malaria_gc_pb',require=['GC','PB'])
    # H/I: the early-infection clone map (Figure 3B) gets its cell-level context here.
    h=p(fig,0,368,54,52,'H','Cell states underlying\nthe early clone map')
    cell_map(h,'malaria','cell_state',{'GC':BLUE,'PB':RED,'Memory':GREEN})
    legend(h,{'GC':BLUE,'PB':RED,'Memory':GREEN,'Other':GREY},ncol=2,y=-.20)
    i=p(fig,65,368,53,52,'I','Early families by sampling day')
    cl=clones('malaria').dropna(subset=['x','y']);cl=cl[cl.reliability>=.5]
    day=cl.donor.str.extract(r'D(\d+)_')[0].astype(float)
    for dy,col in [(7,GREY),(10,GOLD),(14,BLUE)]:
        q=cl[day.eq(dy)]
        i.scatter(q.x,q.y,s=1+5*np.sqrt(q.n_cells),color=col,alpha=.75,edgecolors='white',linewidths=.18,rasterized=True)
    map_axis(i); clone_axis(i); note(i,f'{len(cl):,} reliable clones; clone UMAP',-.14)
    legend(i,{'Day 7':GREY,'Day 10':GOLD,'Day 14':BLUE},ncol=3,y=-.32)
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
    from external_validation_figures import draw_benchmark
    import sys
    draw_benchmark(sys.modules[__name__])


def supplementary7():
    from benchmark_figure import supplementary
    supplementary(new_page,p,lambda f,n: save(f,n))
    AUDIT['outputs'].append('Supplementary_7')


def supplementary8():
    from gc_reclustering_panels import cell_gates, myc_panel, markers, gate_composition, selection
    chosen=selection()
    fig=new_page('Supplementary Figure 8','Historical Top2a display and current model-antigen GC state evidence',
                 'A is an archived output; B–E show reconstructed GSE246382 geometry, measured gates and descriptive RNA.',height=310)
    a=p(fig,0,8,96,79,'A','Top2a on the historical clone map')
    archived_panel(a,'legacy_notebook_clone_Top2a.png')
    note(a,'Notebook cell 33 output; cohort and sampling day unresolved.',-.04,6)
    b=p(fig,106,8,77,58,'B','GSE246382: measured cell compartments')
    cell_gates(b)
    note(b,'884 cells; RNA UMAP reconstructed from notebook code.\nFACS gates are measured; 762 cells have called receptors.',-.18,6)
    c=p(fig,0,116,96,55,'C','GSE246382: mean Myc RNA per receptor group')
    myc_panel(c)
    note(c,f'New coordinates, same {chosen["n_points"]} V–D–J groups as Figure 2B.\nMyc is averaged over captured members of each group.',-.25,6)
    d=p(fig,111,113,72,77,'D','GSE246382: marker expression by clone cluster')
    source=markers(d)
    AUDIT.setdefault('gc_reclustering_tables',{})[str(source.relative_to(ROOT))]={
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'values':'Mean RNA over clones grouped by unsupervised Leiden clone_cluster'}
    e=p(fig,0,226,96,42,'E','Measured compartment composition by clone cluster')
    gate_composition(e)
    f=p(fig,111,219,72,59,'F','Scope of the reconstruction'); f.set_axis_off()
    f.text(0,.98,f'{chosen["n_points"]} same-mouse V–D–J groups; ≥1 cell.\n'
           f'UMAP k={chosen["umap_k"]}, min_dist={chosen["min_dist"]}; seed 123.\n'
           'Scanpy graph k=15; Leiden r=0.3, seed 0.\n\n'
           'Gates and markers annotate captured states.\n'
           'They do not assign direction or future fate.\n'
           'V–D–J groups can merge distinct junctions.\n'
           'The 2D branch shape is parameter-dependent.\n\n'
           'Selection/output interpretation is restricted\n'
           'to this model-antigen GC application.\n'
           'A remains a separate historical display.',
           va='top',fontsize=6.4,linespacing=1.5)
    save(fig,'Supplementary_8');AUDIT['outputs'].append('Supplementary_8')


def supplementary16():
    from external_validation_figures import draw_reporter_audit
    import sys
    draw_reporter_audit(sys.modules[__name__])


def main(biological_only=False):
    main_figures=[figure1,figure2,figure3,figure4,figure5]+([] if biological_only else [figure6])
    supplements=[supplementary1,supplementary2,supplementary3,supplementary4,supplementary5,supplementary6]+([] if biological_only else [supplementary7])+[supplementary8,supplementary9,supplementary10,supplementary11,supplementary12,supplementary13,supplementary14,supplementary15,supplementary16]
    for f in main_figures+supplements+[tested_datasets]:f()
    AUDIT['outputs']=[f'Figure_{i}' for i in range(1,6 if biological_only else 7)]+[f'Supplementary_{i}' for i in range(1,7 if biological_only else 8)]+['Supplementary_8','Supplementary_9','Supplementary_10','Supplementary_11','Supplementary_12','Supplementary_13','Supplementary_14','Supplementary_15','Supplementary_16']
    AUDIT['benchmark_included']=not biological_only
    (HERE/'figure_audit.json').write_text(json.dumps(AUDIT,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--biological-only',action='store_true',help='Reproduce Figure1-5/S1-6/S8 while native benchmark jobs are pending.')
    parser.add_argument('--gc-only',action='store_true',help='Refresh Figure2/S8 and their audit entries only.')
    args=parser.parse_args()
    if args.gc_only:
        audit_path=HERE/'figure_audit.json'
        if audit_path.exists(): AUDIT=json.loads(audit_path.read_text())
        figure2(); supplementary8()
        AUDIT['outputs']=list(dict.fromkeys(AUDIT['outputs']))
        audit_path.write_text(json.dumps(AUDIT,indent=2)+'\n')
    else:
        main(args.biological_only)
