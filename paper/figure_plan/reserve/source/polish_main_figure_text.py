#!/usr/bin/env python
"""Main figures only: concise editable typography, legends and A4 layout.

Reuses the existing scientific artists and frozen analyses. Supplementary
figures and input tables are not rewritten. SVG retains editable text.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.transforms import Bbox
import numpy as np
import figstyle as fs
import figure1_a4 as concept
import assemble_manuscript_figures as assembly
import redesign_figure2_and_model_supps as overview

HERE = Path(__file__).resolve().parents[2]
WORK = HERE / 'reserve/main_figure_text_2026-10-16_142405'
PREVIEW = WORK / 'previews'
STEMS = {1:'Figure_1', 2:'Figure_2_GC_overview', 3:'Figure_3_clone_states', 4:'Figure_4_measured_validation'}
STYLE = {'font.family':'sans-serif', 'font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
         'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
         'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'legend.fontsize':7.6,
         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
         'svg.hashsalt':'threadfin-main-typography-20261016'}


def locate(fig, ax, x, top, w, h):
    ax.set_position([x/210, 1-(top+h)/297, w/210, h/297])


def paper(fig):
    old_h = fig.get_figheight()*25.4
    positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    texts = [(t, t.get_position()) for t in fig.texts]
    fig.set_size_inches(210/25.4, 297/25.4)
    fig.page_height = fig.mm_h = 297
    for ax, b in positions:
        locate(fig, ax, b.x0*210, (1-b.y1)*old_h, b.width*210, b.height*old_h)
    for t, (x,y) in texts:
        t.set_position((x, 1-(1-y)*old_h/297))


def visible_texts(fig):
    candidates = list(fig.texts)
    for ax in fig.axes:
        candidates += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            candidates += [ax.xaxis.label, ax.yaxis.label]
            candidates += ax.get_xticklabels() + ax.get_yticklabels()
            candidates += [ax.xaxis.offsetText, ax.yaxis.offsetText]
        legend = ax.get_legend()
        if legend and legend.get_visible():
            candidates += legend.get_texts() + [legend.get_title()]
    seen = set()
    return [t for t in candidates if not (id(t) in seen or seen.add(id(t)))
            and t.get_visible() and t.get_text().strip()]


def enlarge(fig, body=7.6, title=9, axes=8):
    for t in visible_texts(fig):
        t.set_fontsize(max(body, t.get_fontsize()))
    for ax in fig.axes:
        if ax.get_title(): ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontsize(axes)
        ax.yaxis.label.set_fontsize(axes)
        ax.tick_params(labelsize=7.5, pad=2)
    for t in list(fig.texts):
        if t.get_text().startswith('Figure '): t.remove()
        elif len(t.get_text()) == 1: t.set_fontsize(11)


def figtext(fig, x, y, s, size=7.6, **kwargs):
    return fig.text(x/210, 1-y/297, s, fontsize=size, va='top', **kwargs)


def simple_legend(ax, labels, colors, *, loc='upper left', anchor=(0,-.18), ncol=1, size=7.6):
    old = ax.get_legend()
    if old: old.remove()
    handles = [Line2D([],[],marker='o',ls='',color=c,markersize=3.8,label=s) for s,c in zip(labels,colors)]
    return ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol,
                     fontsize=size, frameon=False, borderpad=0, borderaxespad=0,
                     handlelength=.8, handletextpad=.4, columnspacing=.8, labelspacing=.35)


def capture(number):
    caught = []
    def hold(fig, *args, **kwargs): caught.append(fig)
    args = SimpleNamespace(preview=True, dpi=600)
    with contextlib.redirect_stdout(io.StringIO()):
        if number == 2:
            save, prov, audit = overview.save, overview.PROV, overview.AUDIT
            overview.save, overview.PROV, overview.AUDIT = hold, WORK/'source_records', WORK/'source_records'
            overview.PROV.mkdir(parents=True, exist_ok=True)
            try: overview.figure2(args)
            finally: overview.save, overview.PROV, overview.AUDIT = save, prov, audit
        else:
            save = assembly.save
            assembly.save = hold
            try:
                if number == 3: assembly.figure3(args, assembly.prepare())
                else: assembly.figure4(args)
            finally: assembly.save = save
    assert len(caught) == 1
    return caught[0]


def figure1():
    # A4: a, restored fish-thread concept graphic; b, inside Threadfin
    # (figure1_a4.py). Earlier versions are archived in
    # reserve/figure1_restore_2026-10-16/.
    return concept.figure()


def figure2(fig):
    paper(fig);enlarge(fig)
    table=fig.axes[0]
    for t in table.texts:
        s=t.get_text()
        if s in ['Cell population included','Shown in']:
            t.set_text({'Cell population included':'Cell types','Shown in':'Figure'}[s]);t.set_fontsize(8.3)
        elif s in ['Model antigen','Plasmodium infection','Human vaccination']:t.set_fontsize(8.6)
        elif s in ['GC','PB','PC','Mem']:t.set_fontsize(7.3)
        elif ' · ' in s and 'cells' in s:
            t.set_fontsize(7.0)
            t.set_text(s.replace('1 identifiable mouse','1 mouse').replace('QC cells','cells')
                        .replace('36 h H2B-mCherry division reporter','36 h division reporter')
                        .replace('treated / saline / naive','treated / control / naive'))
        elif s.startswith('Paired scRNA'):
            t.set_text('Paired scRNA-seq + BCR data. Coloured boxes mark sampled cell types.');t.set_fontsize(7.3)
        elif 'GSE' in s or 'experiment' in s:t.set_fontsize(8)
        else:t.set_fontsize(7.3)
    titles=['NP–CGG, day 14\nGSE305877 · 508 clones',
            'Plasmodium, days 10–42\nGSE286215 · 1,015 clones',
            'Human vaccination\nGSE195673 · 1,243 clones']
    for ax,title in zip(fig.axes[1:4],titles):
        ax.set_title(title,fontsize=8.8,pad=5)
        for t in list(ax.texts):t.remove()
        leg=ax.get_legend()
        for t in leg.get_texts():t.set_fontsize(7.6)
    rename={'Plasmodium | Profile sampling':'Cells per clone',
            'Clonal coherence vs shuffle':'Clone coherence',
            'Programme bootstrap stability':'Programme stability',
            'Plasmodium | RNA states':'Plasmodium cell types',
            'Programme map-back (3.2% of cells)':'Clone programmes on cells',
            'RNA-state mix | each clone equal':'Cell types per programme'}
    for ax in fig.axes:
        title=ax.get_title()
        if title in rename:ax.set_title(rename[title],fontsize=9,pad=5)
        if ax.get_ylabel()=='Mean bootstrap Jaccard':ax.set_ylabel('Bootstrap Jaccard',fontsize=8)
        if ax.get_ylabel()=='Mean within-clone fraction (%)':ax.set_ylabel('Mean cell fraction per clone (%)',fontsize=8)
        if ax.get_xlabel()=='Captured cells per profile':ax.set_xlabel('Captured cells per clone',fontsize=8)
        if title=='Programme bootstrap stability':
            for t in ax.texts:
                val=float(t.get_text());t.set_position((t.get_position()[0],max(val+.055,.80) if val>.6 else val+.045))
        if title=='Plasmodium | Profile sampling':
            locate(fig,ax,13,176,52,38)
            ax.set_ylabel('Clones',fontsize=8,labelpad=2)
        if title in ['Plasmodium | RNA states','Programme map-back (3.2% of cells)','RNA-state mix | each clone equal']:
            b=ax.get_position();locate(fig,ax,b.x0*210,233,b.width*210,b.height*297)
        if not ax.axison and ax is not table:
            b=ax.get_position();locate(fig,ax,b.x0*210,286,b.width*210,b.height*297)
            for t in ax.get_legend().get_texts():t.set_fontsize(7.2)
    for t in fig.texts:
        s=t.get_text();x,y=t.get_position()
        if s.startswith('Programme labels are local'):
            t.set_text('Colours identify programmes within each dataset. Point area shows captured clone size.');t.set_fontsize(7.3)
        elif s.startswith('Map-back:'):
            t.set_text('2,350 / 72,359 cells mapped (3.25%); grey = other cells. Bar labels = clone counts.')
            t.set_position((x,1-294/297));t.set_fontsize(7.0)
        elif s=='500 shuffles within sampled mouse.':t.set_text('500 shuffles within each mouse.');t.set_fontsize(7.3)
        elif s in ['1,015 profiles; 830 contain two cells.','50 resamples; dashed line = 0.75.']:
            t.set_fontsize(7.3)
        if s=='A':t.set_position((x,1-1.2/297))
        if s=='D':t.set_position((x,1-227/297))
        if (1-y)*297==225:t.set_position((x,1-222/297))


def figure3(fig):
    paper(fig);enlarge(fig,body=7.6,title=9,axes=8)
    layouts={'a':(18,6,62,62),'b':(91,6,53,53),'c':(151,6,53,53),
             'd':(7,85,53,46),'e':(79,85,53,46),'f':(145,85,63,46),
             'g':(24,148,57,57),'h':(128,148,57,57),
             'i':(24,224,57,57),'j':(128,224,57,57)}
    titles={'a':'GSE203132 | Cell types','b':'Clone programmes','c':'Clone programmes on cells',
            'd':'Clone state diversity','e':'Clones containing memory cells','f':'Within-mouse correlations',
            'g':'GSE284622 | Cell programmes','h':'Clone programmes',
            'i':'Original GC/nonGC sort','j':'GC-sort fraction per clone'}
    for k,ax in fig.panel_axes.items():
        locate(fig,ax,*layouts[k]);ax.set_title(titles[k],fontsize=8.8,pad=5)
        if ax.get_xlabel()=='Clone x':ax.set_xlabel('Clone UMAP 1',fontsize=8)
        if ax.get_ylabel()=='Clone y':ax.set_ylabel('Clone UMAP 2',fontsize=8)
    for t in fig.texts:
        if t.get_text() in layouts:
            key=t.get_text();x,top,w,h=layouts[key]
            letter_y=top-1.2 if key in 'abc' else top-2.5
            t.set_position(((x-4)/210,1-letter_y/297));t.set_va('bottom')
    # Remove the old external, counted cell-type key.
    for ax in list(fig.axes):
        if any(t.get_text()=='RNA state' for t in ax.texts):ax.remove()
    pal=assembly.ref.STATE_COLORS['S2']
    names={'Fcrl5+ MemB':'Fcrl5+ memory','CD23+ MemB':'CD23+ memory','other MemB':'Other memory','unassigned':'Unassigned'}
    simple_legend(fig.panel_axes['a'],[names.get(s,s) for s in pal],list(pal.values()),loc='lower left',anchor=(.015,.015),size=7.4)
    fig.panel_axes['e'].set_ylabel('Clones with memory cells (%)',fontsize=8)
    fig.panel_axes['e'].set_xlabel('Clone UMAP 1 quartile',fontsize=8)
    fig.panel_axes['f'].set_ylabel('Mouse',fontsize=8)
    fig.panel_axes['f'].set_xlabel('Spearman ρ (entropy, UMAP 1)',fontsize=7.7)
    for t in list(fig.panel_axes['d'].texts):
        if t.get_text().startswith('Spearman'):
            t.remove();figtext(fig,7,137.6,'ρ = −0.742; n = 322 clones',7.6)
    for ax in fig.axes:
        if ax in fig.panel_axes.values():continue
        ss=[t.get_text() for t in ax.texts]
        if 'Programme' in ss:
            # Two existing shared keys; retain colour identities and key order.
            x=ax.get_position().x0*210
            locate(fig,ax,91 if x>80 else 24,70 if x>80 else 211,113 if x>80 else 168,8.4)
        elif 'Original sort' in ss:
            locate(fig,ax,24,289,64,8.4)
        elif any('958 clones;' in s for s in ss):
            locate(fig,ax,128,291,75,5)
            for t in ax.texts:t.set_text('958 clones; size = captured cells');t.set_fontsize(7.2)
        elif ax.get_title()=='PC cluster (zoom)':
            old_top=(1-ax.get_position().y1)*297
            locate(fig,ax,91,160 if old_top<220 else 236,26,26)
            ax.set_title('PC cluster (zoom)',fontsize=7.8,pad=3)
        elif hasattr(ax,'_colorbar'):
            if ax.get_title()=='entropy':
                locate(fig,ax,62,92,1.9,35);ax.set_title('Entropy',fontsize=7.6,pad=3)
            else:
                locate(fig,ax,190,231,2,43);ax.set_title('GC fraction',fontsize=7.6,pad=6)
            ax.tick_params(labelsize=7.2,pad=1.5)
    # Panels g and i still use exactly the same limits and frame geometry.
    fig.panel_boxes_mm = {k:list(v) for k,v in layouts.items()}


def figure4(fig):
    paper(fig);enlarge(fig,body=7.6,title=8.7,axes=8)
    # Artist order is the fixed 4-panel-per-row manuscript assembly.
    data_axes=[a for a in fig.axes if a.get_label()!='<colorbar>']
    assert len(data_axes)==12
    xs=[8,60,116,163];tops=[9,111,230];heights=[45,45,45]
    title_rows=[['RBD cells\nDivision reporter','Recent-division\nfraction','RBD-binding\nfraction','Dark-zone\nfraction'],
                ['LARRY cells, day 2\nLater neutrophil fraction','Clone profiles, day 2\nLater neutrophil fraction','Day-6 fate prediction\nHeld-out R²','Day-6 fate fractions\nSister wells'],
                ['PC9 cells, days 0–7\nLater persister fraction','Clone profiles, days 0–7\nLater persister fraction','Day-14 gate\nassociation','']]
    # Link existing colourbars before moving their parent axes.
    cbars={}
    old_shapes={ax:ax.get_position().height*297/(ax.get_position().width*210) for ax in data_axes}
    for ax in data_axes:
        for col in ax.collections:
            cb=getattr(col,'colorbar',None)
            if cb is not None:cbars[ax]=cb
    for n,ax in enumerate(data_axes):
        row,col=divmod(n,4);w=35 if col==2 else 39
        h=heights[row];top=tops[row];left=xs[col]
        # Retain each saved map's geometric display ratio instead of stretching it.
        if ax in cbars or n==0:w=h/old_shapes[ax]
        if row==0 and col==2:left=112
        if row==0 and col==3:left=161
        locate(fig,ax,left,top,w,h)
        ax.set_title(title_rows[row][col],fontsize=8.5,pad=6)
        if ax in cbars:
            cb=cbars[ax];locate(fig,cb.ax,left+w+1,top+3,1.8,h-6)
            cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=7.2,pad=1,length=1.5)
        # All long notes are replaced below with short statements outside plots.
        for t in list(ax.texts):
            if t.get_transform()==ax.transAxes and t.get_position()[1]<0:t.remove()
        oldleg=ax.get_legend()
        if oldleg:oldleg.remove()
    for t in fig.texts:
        if t.get_text() in 'ABC' and len(t.get_text())==1:
            row='ABC'.index(t.get_text());t.set_position((.6/210,1-(tops[row]-8)/297));t.set_va('top')
    simple_legend(data_axes[0],['mCherry high','mCherry low'],[fs.ORANGE,fs.BLUE],anchor=(0,-.18),size=7.5)
    notesA=['36,188 GC-sorted cells\n36 h division reporter',
            '1,887 clones\nLow mCherry: ≥6 divisions',
            '1,544 clones\nRecorded RBD-binding gate',
            '343 clones\nRecorded zone gate']
    for x,s in zip([8,60,112,161],notesA):figtext(fig,x,74,s,7.4,linespacing=1.25,color=fs.INK2)
    pred=data_axes[6]
    pred.set_ylabel('')
    pred.set_xticks(pred.get_xticks(),[str(i) for i in [1,2,3,4,5]*2],fontsize=7.4,rotation=0,ha='center')
    pred.set_yticks([0,.1,.2,.3])
    for t in pred.texts:
        if t.get_text() in ['Neutrophil','Monocyte']:
            t.set_fontsize(7.2)
    sw=data_axes[7]
    sw.yaxis.tick_left();sw.yaxis.set_label_position('left')
    sw.set_xlabel('Fraction in well 1',fontsize=7.7)
    sw.set_ylabel('Fraction in well 2',fontsize=7.7)
    sw.set_xticks([0,.5,1]);sw.set_yticks([0,.5,1])
    simple_legend(sw,['Neutrophil','Monocyte'],[fs.RED,fs.VIOLET],anchor=(0,-.25),size=7.5)
    notesB=['402 day-2 cells\nColour: day-6 neutrophil fraction\n15-NN r = 0.48',
            '172 clones with ≥2 day-2 cells\nProfiles use day 2 only',
            '1 Training mean\n2 Clone size\n3 RNA mean\n4 RNA mean + variance\n5 Threadfin kernel',
            '211 shared barcodes\nρ = 0.73 (neutrophils)\nρ = 0.69 (monocytes)\n'+r'P = $2\times10^{-4}$']
    for col,(x,s) in enumerate(zip(xs,notesB)):
        figtext(fig,x,174 if col<3 else 184,s,7.3,linespacing=1.25,color=fs.INK2)
    assoc=data_axes[10]
    assoc.set_title('Day-14 gate association\nR² above shuffled labels',fontsize=8.5,pad=6,y=1.28)
    assoc.set_ylabel('')
    assoc.set_yticks([0,.025,.05,.075],['0','0.025','0.050','0.075'])
    assoc.set_xticks([0,1],['mCherry high\n(persister)','mCherry low\n(cycling)'],fontsize=7.2)
    handles=[Line2D([],[],color=fs.BLUE,lw=4,label='Mean profile'),Line2D([],[],color=fs.RED,lw=4,label='Kernel profile')]
    assoc.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.15),fontsize=7.4,
                 frameon=False,borderaxespad=0,borderpad=0,labelspacing=.3,handlelength=1.2)
    notesC=['8,894 early cells\n15-NN r = 0.21',
            '1,016 clones\nProfiles use days 0–7 only',
            '1,016 clones\n'+r'Both P = $5\times10^{-4}$']
    for x,s in zip(xs,notesC):figtext(fig,x,289,s,7.3,linespacing=1.2,color=fs.INK2)
    readout=data_axes[11]
    for t in list(readout.texts):t.remove()
    readout.text(0,1,'PC9 + osimertinib',transform=readout.transAxes,va='top',fontsize=8.7,fontweight='bold')
    readout.text(0,.82,'Early clone profiles\nassociate with the later\npersister-gate fraction.\n\nDays 0–7: RNA profiles\nDay 14: measured gate',
                 transform=readout.transAxes,va='top',fontsize=8,linespacing=1.35,color=fs.INK2)


def plot_fingerprint(fig):
    result=[]
    for ax in fig.axes:
        if not ax.axison or ax.get_label()=='<colorbar>':continue
        for c in ax.collections:
            if isinstance(c,PathCollection):
                a=np.asarray(c.get_offsets());result.append(hashlib.sha256(a.tobytes()).hexdigest())
        for container in ax.containers:
            if hasattr(container,'datavalues'):result.append(np.asarray(container.datavalues).tolist())
    return result


def audit(fig,number):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=visible_texts(fig)
    pairs=[];outside=[];marks=[]
    boxes=[t.get_window_extent(renderer) for t in texts]
    for i,(t,b) in enumerate(zip(texts,boxes)):
        if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1):outside.append(t.get_text())
        for u,c in zip(texts[i+1:],boxes[i+1:]):
            inter=Bbox.intersection(b,c)
            if inter is not None and inter.width>.3 and inter.height>.3:pairs.append([t.get_text(),u.get_text()])
        # Check scatter symbols against all text, including legends placed in-map.
        for ax in fig.axes:
            if ax.get_label()=='<colorbar>' or (not ax.axison and number!=1):continue
            for coll in ax.collections:
                if not isinstance(coll,PathCollection):continue
                pts=coll.get_offset_transform().transform(coll.get_offsets())
                if not len(pts):continue
                keep=(pts[:,0]>=ax.bbox.x0)&(pts[:,0]<=ax.bbox.x1)&(pts[:,1]>=ax.bbox.y0)&(pts[:,1]<=ax.bbox.y1)
                sizes=np.asarray(coll.get_sizes());r=(np.sqrt(np.max(sizes))*.5*fig.dpi/72) if len(sizes) else 0
                n=int(np.count_nonzero(keep&(pts[:,0]>=b.x0-r)&(pts[:,0]<=b.x1+r)&(pts[:,1]>=b.y0-r)&(pts[:,1]<=b.y1+r)))
                if n:marks.append({'text':t.get_text(),'points':n})
    page_mm=[round(fig.get_figwidth()*25.4,2),round(fig.get_figheight()*25.4,2)]
    report={'figure':number,'page_mm':page_mm, 'text_overlaps':pairs,'text_outside_page':outside,
            'text_scatter_candidates':marks,'font_sizes_pt':sorted({t.get_fontsize() for t in texts}),
            'text_inventory':[{'text':t.get_text(),'font_pt':t.get_fontsize()} for t in texts],
            'svg_text_mode':'editable text; rasterised scatter layers remain embedded images'}
    (WORK/(STEMS[number]+'_text_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(STEMS[number], 'overlap',pairs,'outside',outside,'scatter',marks,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',choices=['1','2','3','4','all'],default='all')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args();WORK.mkdir(exist_ok=True);PREVIEW.mkdir(exist_ok=True)
    for n in range(1,5):
        if args.figure not in [str(n),'all']:continue
        if n == 4:
            # The evidence revision has its own source-table and layout checks.
            # Keep this shared command from restoring the superseded Figure 4.
            import figure4_evidence
            figure4_evidence.render(preview=args.preview, dpi=600)
            continue
        plt.rcParams.update(STYLE)
        fig=figure1() if n==1 else capture(n)
        before=plot_fingerprint(fig)
        if n>1:globals()['figure'+str(n)](fig)
        after=plot_fingerprint(fig)
        assert before==after,'Scientific point coordinates or bar values changed'
        report=audit(fig,n)
        report['scientific_point_coordinates_and_bar_values_unchanged']=before==after
        (WORK/(STEMS[n]+'_text_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        fig.savefig(PREVIEW/(STEMS[n]+'.png'),dpi=170,facecolor='white')
        if not args.preview:
            assert not report['text_overlaps'],report['text_overlaps']
            assert not report['text_outside_page'],report['text_outside_page']
            fig.savefig(HERE/(STEMS[n]+'.svg'),dpi=600,facecolor='white')
            fig.savefig(HERE/(STEMS[n]+'.pdf'),dpi=600,facecolor='white')
            fig.savefig(HERE/(STEMS[n]+'.png'),dpi=600,facecolor='white')
            fig.savefig(HERE/'reserve/current_previews'/(STEMS[n]+'.png'),dpi=170,facecolor='white')
        plt.close(fig)


if __name__=='__main__':main()
