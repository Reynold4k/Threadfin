#!/usr/bin/env python
"""Figure 2 and detailed model-antigen supplements, rebuilt from numeric source tables.

Fixed coordinates and exact labels; random overplotting order only. Original S2 is unchanged.
"""
from pathlib import Path
import argparse,json,hashlib,sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import fig2_gc_overview as old
import schematics as sk
from fig_clone_states_combined import STATE_COLORS
from figstyle import BLUE,RED,VIOLET,ORANGE,GREEN,AQUA,INK,INK2
HERE=Path(__file__).resolve().parents[2]
ROOT=HERE.parents[2]
CASE=ROOT/'case_studies/results'  # replaced below by explicit workspace root
WORK=Path(__file__).resolve().parents[5]
CASE=WORK/'Threadfin/case_studies/results'
AUDIT=HERE/'reserve/story_relayout_2026-10-16'
PROV=HERE/'reserve/provenance'
PREVIEW=HERE/'reserve/current_previews'
GRAY='#e3e3de'
PROG=old.PROG_COL
SEED=20261016
plt.rcParams.update({'font.size':7,'axes.titlesize':7.4,'axes.labelsize':6.9,
 'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.5,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.linewidth':.5})
CACHE={}

def data(ds):
 if ds not in CACHE:
  c=pd.read_csv(CASE/ds/'cells.csv.gz',index_col=0)
  t=pd.read_csv(CASE/ds/'clone_table.csv').set_index('clone_id',drop=False)
  assert c.index.is_unique and t.index.is_unique
  assert np.isfinite(c[['umap_1','umap_2']].to_numpy()).all()
  assert np.isfinite(t[['x','y']].to_numpy()).all()
  CACHE[ds]=(c,t)
 return CACHE[ds]

def page(h):
 f=plt.figure(figsize=(210/25.4,h/25.4));f.mm_h=h;return f

def txt(f,x,y,s,size=7,**kw):
 return f.text(x/210,1-y/f.mm_h,s,fontsize=size,va='top',color=INK,**kw)

def axat(f,x,y,w,h,letter=None,title=None):
 ax=f.add_axes([x/210,1-(y+h)/f.mm_h,w/210,h/f.mm_h])
 if letter:txt(f,x-5,y-6,letter,9.5,fontweight='bold')
 if title:ax.set_title(title,pad=4)
 return ax

def frame(ax,title='',unit='Cell',ylabel=True):
 if title:ax.set_title(title,pad=4)
 ax.set_xticks([]);ax.set_yticks([])
 ax.set_xlabel(f'{unit} UMAP 1',labelpad=2)
 if ylabel:ax.set_ylabel(f'{unit} UMAP 2',labelpad=2)
 for s in ax.spines.values():s.set_color('#898781');s.set_linewidth(.5)

def limits(ax,c,cols):
 xy=c[cols].to_numpy();lo=np.nanmin(xy,axis=0);hi=np.nanmax(xy,axis=0)
 span=np.maximum(hi-lo,1e-6);lo-=.04*span;hi+=.04*span
 ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
 # Equal geometric scale prevents stretching the numerical embedding.
 ax.set_aspect('equal',adjustable='datalim')

def categorical(ax,c,labels,pal,title='',unit='Cell',s=None,alpha=.85):
 cols=['umap_1','umap_2'] if unit=='Cell' else ['x','y']
 labels=pd.Series(np.asarray(labels),index=c.index)
 use=labels.notna()&labels.isin(pal)
 area=np.repeat(.55 if unit=='Cell' else 5,len(c)) if s is None else np.broadcast_to(np.asarray(s),len(c))
 if (~use).any():ax.scatter(c.loc[~use,cols[0]],c.loc[~use,cols[1]],s=area[~use],c=GRAY,lw=0,alpha=.6,rasterized=True)
 order=np.random.default_rng(SEED).permutation(np.flatnonzero(use.to_numpy()))
 if len(order):
  ax.scatter(c.iloc[order][cols[0]],c.iloc[order][cols[1]],s=area[order],
   c=labels.iloc[order].map(pal).tolist(),lw=0,alpha=alpha,rasterized=True)
 frame(ax,title,unit);limits(ax,c,cols)

def key(ax,pal,ncol=2,labels=None,y=-.18,size=6):
 handles=[Line2D([],[],marker='o',ls='',color=col,markersize=3,label=(labels or {}).get(k,k)) for k,col in pal.items()]
 return ax.legend(handles=handles,ncol=ncol,fontsize=size,loc='upper left',bbox_to_anchor=(-.02,y),
  handlelength=.8,handletextpad=.35,columnspacing=.65,borderaxespad=0,frameon=False)

def figkey(f,x,y,pal,ncol=4,w=190,size=6.5,labels=None):
 ax=axat(f,x,y,w,8);ax.set_axis_off()
 handles=[Line2D([],[],marker='o',ls='',color=col,markersize=3,label=(labels or {}).get(k,k)) for k,col in pal.items()]
 ax.legend(handles=handles,ncol=ncol,fontsize=size,loc='upper left',bbox_to_anchor=(0,0,1,1),
  mode='expand',handlelength=.8,handletextpad=.35,borderaxespad=0,frameon=False)
 return ax

def numeric(ax,c,v,title='',unit='Cell',cmap='viridis',vmin=None,vmax=None,s=None,label=None,ticks=None):
 pos=ax.get_position()
 ax.set_position([pos.x0,pos.y0,pos.width-4.5/210,pos.height])
 cols=['umap_1','umap_2'] if unit=='Cell' else ['x','y']
 v=np.asarray(v,dtype=float);ok=np.isfinite(v)
 area=np.repeat(.6 if unit=='Cell' else 6,len(c)) if s is None else np.broadcast_to(np.asarray(s),len(c))
 if (~ok).any():ax.scatter(c.loc[~ok,cols[0]],c.loc[~ok,cols[1]],s=area[~ok],c=GRAY,lw=0,alpha=.6,rasterized=True)
 order=np.flatnonzero(ok)[np.argsort(v[ok],kind='stable')]
 artist=ax.scatter(c.iloc[order][cols[0]],c.iloc[order][cols[1]],s=area[order],c=v[order],
  cmap=cmap,vmin=vmin,vmax=vmax,lw=0,alpha=.9,rasterized=True)
 frame(ax,title,unit);limits(ax,c,cols)
 pos=ax.get_position();f=ax.figure
 ca=f.add_axes([pos.x1+.006,pos.y0+.12*pos.height,.008,.76*pos.height])
 cb=f.colorbar(artist,cax=ca,ticks=ticks);cb.ax.tick_params(labelsize=5.5,pad=1,length=1.5)
 if label:cb.ax.set_title(label,fontsize=5.7,pad=3)
 return artist

def cloneprog(ax,ds,title,legend=True):
 c,t=data(ds);pal={p:PROG[p] for p in sorted(t.clone_programme.dropna().unique())}
 categorical(ax,t,t.clone_programme,pal,title,unit='Clone',s=np.clip(t.n_cells.to_numpy()*1.2,3,28))
 ax.text(.03,.97,f'{len(t):,} profiles',transform=ax.transAxes,va='top',fontsize=6.4,
  bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
 if legend:key(ax,pal,ncol=len(pal),y=-.17,size=6.5)
 return pal

def mapped(c,t):return c.clone_id.map(t.clone_programme)

def new_scores():return pd.read_csv(CASE/'np_cgg_wt_d14/cells_with_scores.csv.gz',index_col=0)

def save(f,name,args):
 # Tight bounding box removes paper margins; labels/legends remain inside exported content.
 f.canvas.draw();PREVIEW.mkdir(parents=True,exist_ok=True);PROV.mkdir(parents=True,exist_ok=True)
 f.savefig(PREVIEW/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.03,facecolor='white')
 if not args.preview:
  f.savefig(HERE/(name+'.pdf'),dpi=400,bbox_inches='tight',pad_inches=.03,facecolor='white')
  f.savefig(HERE/(name+'.png'),dpi=args.dpi,bbox_inches='tight',pad_inches=.03,facecolor='white')
 texts=[]
 for t in f.findobj(match=matplotlib.text.Text):
  if t.get_visible() and t.get_text():texts.append({'text':t.get_text(),'font_pt':t.get_fontsize()})
 (PROV/(name+'_layout.json')).write_text(json.dumps({'date':'2026-10-16','tight_crop':True,
   'random_draw_order_seed':SEED,'texts':texts},indent=2))
 plt.close(f);print('Saved',name,flush=True)

def designs(f):
 ax=axat(f,5,0,204,86);sk.canvas(ax,204,86)
 ax.set_aspect('auto')
 txt(f,.4,.2,'A',10,fontweight='bold')
 y=83
 ax.text(134,y,'Cell population included',fontsize=7,ha='left',va='center',fontweight='bold')
 ax.text(185,y,'Shown in',fontsize=7,ha='left',va='center',fontweight='bold')
 rows=[('Model antigen',BLUE,[
 ('NP–CGG · GSE305877','WT day 14 · 1 identifiable mouse · 13,727 QC cells · NP+ B220+ sort',('germinal centre',),'B; S1 (1)'),
 ('Oral OVA + CT · GSE284622','3 reporter mice · 16,619 cells · GC-fate sort (tdTomato)',('germinal centre','plasma cell'),'Fig. 3'),
 ('NP–OVA · GSE287123','7 mice · 10,109 cells · 36 h H2B-mCherry division reporter',('germinal centre',),'S1 (2)'),
 ('RBD protein / mRNA · GSE287123','10 mice · 36,188 cells · division, bait OR LZ/DZ gates',('germinal centre',),'Fig. 4; S1 (2)'),
 ('NP–OVA · GSE246382','11 mice · 884 cells · day 14 · Smart-seq2 · GC/PC gates; W33L',('germinal centre','plasma cell'),'S1 (1)')]),
 ('Plasmodium infection',RED,[
 ('P. chabaudi AS · experiment 1','20 mice · 40,623 cells · days 0–14 · spleen',('germinal centre','plasmablast','memory'),'—'),
 ('P. chabaudi AS · experiment 2 · GSE286215','42 mice · 72,359 cells · days 10–42 · treated / saline / naive',('germinal centre','plasmablast','memory'),'B–D')]),
 ('Human vaccination',VIOLET,[
 ('BNT162b2 · GSE195673','8 participants · 193,442 cells · days 28–201 · LN FNA + blood',('germinal centre','plasmablast','plasma cell','memory'),'B; S2; S6')])]
 for header,col,entries in rows:
  ax.text(0,y,header,fontsize=7.3,color=col,fontweight='bold',va='center')
  ax.plot([0,204],[y-2.1,y-2.1],color=col,lw=.7,alpha=.45)
  y-=5.0
  for name,detail,kept,where in entries:
   if header=='Human vaccination':sk.person(ax,3.6,y-1.5,s=.45,color=col)
   else:sk.mouse(ax,3.6,y-1.5,s=.45,color=col)
   if header=='Plasmodium infection':sk.parasite(ax,12.3,y-1.5,s=.75,color=col)
   else:sk.syringe(ax,13,y-1.5,s=.40,color=col)
   ax.text(19,y,name,fontsize=6.9,va='center')
   ax.text(19,y-3.0,detail,fontsize=5.9,color=INK2,va='center')
   for i,(short,comp) in enumerate(old.CHIP):
    x=126+i*14.0;on=comp in kept
    ax.add_patch(old.plt_rect(x,y-3.3,11.5,3.7,old.COL[comp] if on else 'none',old.COL[comp] if on else '#aaa9a4',.45))
    ax.text(x+5.75,y-1.3,short,fontsize=6.0,color='white' if on else '#999892',ha='center',va='center',fontweight='bold' if on else 'normal')
   ax.text(185,y-1.3,where,fontsize=6,va='center',color=INK2)
   y-=7.6
  y-=1.5
 # Common footnote is part of panel A, rather than a separate figure title.
 ax.text(0,0,'Paired scRNA + BCR datasets; profiles use recurrent captured clones. Population chips describe study coverage.',fontsize=6.3,color=INK2,va='bottom')

def figure2(args):
 f=page(306);designs(f)
 for i,(ds,title) in enumerate([('np_cgg_wt_d14','NP–CGG, WT day 14\nGSE305877'),('malaria_late','Plasmodium, days 10–42\nGSE286215'),('ln_vaccine','Human vaccination, LN + blood\nGSE195673')]):
  ax=axat(f,8+69*i,99,58,49,'B' if i==0 else None)
  cloneprog(ax,ds,title)
 txt(f,8,163,'Programme labels are local to each dataset; colour is not a shared cell fate. Point area increases with captured clone size (capped).',6.4)
 c,t=data('malaria_late');summary=json.loads((CASE/'malaria_late/summary.json').read_text())
 prog=pd.read_csv(CASE/'malaria_late/programmes.csv').set_index('programme')
 ax=axat(f,10,176,55,38,'C','Plasmodium | Profile sampling')
 counts=np.array([(t.n_cells==2).sum(),(t.n_cells==3).sum(),(t.n_cells==4).sum(),(t.n_cells>=5).sum()])
 ax.bar(np.arange(4),counts,color='#59798a',width=.65)
 for i,n in enumerate(counts):ax.text(i,n+20,str(n),ha='center',fontsize=6.6)
 ax.set_ylim(0,1000);ax.set_xticks(range(4),['2','3','4','≥5']);ax.set_xlabel('Captured cells per profile');ax.set_ylabel('Number of clones')
 txt(f,10,225,'1,015 profiles; 830 contain two cells.',6.3)
 ax=axat(f,80,176,54,38,title='Clonal coherence vs shuffle')
 coh=summary['coherence'];ax.bar([0,1],[coh['icc'],coh['null_mean']],color=[BLUE,'#b8b8b3'],width=.6)
 ax.hlines(coh['null_q95'],.63,1.37,color=INK,lw=.85)
 ax.text(0,coh['icc']+.025,f"{coh['icc']:.3f}",ha='center',fontsize=6.8)
 ax.text(1,.07,f"{coh['null_mean']:.3f}\n95th: {coh['null_q95']:.3f}",ha='center',fontsize=6.2)
 ax.set_xticks([0,1],['Observed','Shuffled']);ax.set_ylim(0,.55);ax.set_ylabel('Clonal ICC')
 ax.text(.97,.97,'P = 0.002',transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
 txt(f,80,225,'500 shuffles within sampled mouse.',6.3)
 ax=axat(f,150,176,54,38,title='Programme bootstrap stability')
 vals=prog.loc[['P1','P2','P3','P4'],'stability']
 ax.bar(range(4),vals,color=[PROG[p] for p in vals.index],width=.65)
 ax.axhline(.75,color=INK2,ls='--',lw=.65)
 for i,v in enumerate(vals):ax.text(i,v+.035,f'{v:.2f}',ha='center',fontsize=6.3)
 ax.set_ylim(0,1.07);ax.set_xticks(range(4),vals.index);ax.set_ylabel('Mean bootstrap Jaccard')
 txt(f,150,225,'50 resamples; dashed line = 0.75.',6.3)
 # D retains the source cellular state view and exact programme map-back.
 pal=STATE_COLORS['S1'];m=mapped(c,t)
 ax=axat(f,8,242,58,45,'D','Plasmodium | RNA states')
 categorical(ax,c,c.cell_state,pal,s=.35,alpha=.7)
 ax=axat(f,77,242,58,45,title='Programme map-back (3.2% of cells)')
 categorical(ax,c,m,{p:PROG[p] for p in sorted(t.clone_programme.unique())},s=np.where(m.notna(),1.3,.22),alpha=.9)
 # Every clone contributes its own state fractions, then clones are averaged equally.
 cc=c[c.clone_id.isin(t.index)]
 ct=pd.crosstab(cc.clone_id,cc.cell_state)
 frac=ct.div(ct.sum(axis=1),axis=0)
 frac['programme']=t.clone_programme
 mean=frac.groupby('programme').mean().reindex(['P1','P2','P3','P4'])
 cell_weighted=pd.crosstab(m,c.cell_state);cell_weighted=cell_weighted.div(cell_weighted.sum(axis=1),axis=0)
 ax=axat(f,150,242,54,45,title='RNA-state mix | each clone equal')
 bottom=np.zeros(4)
 for state,col in pal.items():
  v=mean.get(state,pd.Series(0,index=mean.index)).values
  ax.bar(range(4),v,bottom=bottom,color=col,width=.67,lw=0);bottom+=v
 ax.set_ylim(0,1.08);ax.set_xticks(range(4),mean.index);ax.set_yticks([0,.5,1],['0','50','100'])
 ax.set_ylabel('Mean within-clone fraction (%)');ax.set_xlabel('Programme')
 for i,p in enumerate(mean.index):ax.text(i,1.025,f'{(t.clone_programme==p).sum()}',fontsize=5.9,ha='center')
 figkey(f,8,295,pal,ncol=6,size=6.1)
 txt(f,8,304,'Map-back: 2,350 / 72,359 cells; grey = cells outside the displayed profiles. Numbers above composition bars are clones.',6.1)
 mean.to_csv(AUDIT/'plasmodium_clone_weighted_states.csv')
 cell_weighted.to_csv(AUDIT/'plasmodium_original_cell_weighted_states.csv')
 (PROV/'Figure_2_GC_overview.json').write_text(json.dumps({'date':'2026-10-16','scope':'method demonstration',
  'B_datasets':['np_cgg_wt_d14','malaria_late','ln_vaccine'],'C':summary['coherence'],
  'stability':vals.to_dict(),'D_weighting':'Each clone is normalised to state fractions; clone fractions averaged equally within programme.',
  'D_cells':len(c),'D_profiles':len(t),'D_mapped_cells':int(m.notna().sum()),
  'moved_C':'Supplementary_clone_compartments','fixed_coordinates':True},indent=2))
 save(f,'Figure_2_GC_overview',args)

def suppax(f,row,col,letter,title):
 return axat(f,6+52*col,([11,82,150,216] if getattr(f, 'reporter_page', False) else [11,77,150,216])[row],40,41,letter,title)

def score(ax,c,col,title):
 v=c[col].astype(float);numeric(ax,c,v,title,vmin=float(v.quantile(.02)),vmax=float(v.quantile(.98)),label='RNA score')

def cellprog(ax,ds,title='Programme map-back'):
 c,t=data(ds);p={k:PROG[k] for k in sorted(t.clone_programme.dropna().unique())}
 if ds=='gut_gc_reporter':p={'P1':'#405BD8','P2':'#E9AE14','P3':'#E44336'}
 categorical(ax,c,mapped(c,t),p,title,s=np.where(mapped(c,t).notna(),.9,.4));key(ax,p,ncol=len(p),size=5.8)
 return c,t

def cf(ax,t,col,title,cmap='viridis'):
 numeric(ax,t,pd.to_numeric(t[col],errors='coerce'),title,'Clone',s=np.clip(t.n_cells.to_numpy()*1.2,3,26),
  cmap=cmap,vmin=0,vmax=1,ticks=[0,.5,1],label='Fraction')

def clusterplot(ax,c,column,title):
 labels=c[column].astype(str);cats=sorted(labels.unique(),key=lambda s:int(s) if s.isdigit() else 999)
 pal={s:plt.cm.tab20(i%20) for i,s in enumerate(cats)}
 categorical(ax,c,labels,pal,title)
 for lab in cats:
  xy=c.loc[labels.eq(lab),['umap_1','umap_2']].median()
  ax.text(*xy,str(lab),fontsize=5.5,ha='center',va='center',bbox={'fc':'white','ec':'none','alpha':.85,'pad':.4})

def supplement_np(args):
 f=page(274);c=new_scores();_,t=data('np_cgg_wt_d14')
 txt(f,1,0,'NP–CGG | GSE305877 | WT day 14: 13,727 QC cells, 508 profiles; one identifiable mouse',8,fontweight='bold')
 clusterplot(suppax(f,0,0,'A','scRNA clusters'),c,'rna_cluster','scRNA clusters')
 for j,col in enumerate(['GC score','LZ score','Cycling score'],1):score(suppax(f,0,j,chr(65+j),col),c,col,col)
 pal={'Paired BCR':BLUE,'No eligible pair':GRAY};lab=np.where(c.clone_id.notna(),'Paired BCR','No eligible pair')
 ax=suppax(f,1,0,'E','Paired receptor coverage');categorical(ax,c,lab,pal);key(ax,pal,ncol=1,size=5.8)
 cellprog(suppax(f,1,1,'F','Programme map-back'),'np_cgg_wt_d14')
 v=c.groupby('clone_id')['GC score'].mean().reindex(t.index)
 numeric(suppax(f,1,2,'G','Clone mean GC score'),t,v,unit='Clone',s=np.clip(t.n_cells*1.2,3,26),label='RNA score')
 ax=suppax(f,1,3,'H','Programme stability');pt=pd.read_csv(CASE/'np_cgg_wt_d14/programmes.csv').set_index('programme')
 ax.bar(range(len(pt)),pt.stability,color=[PROG[p] for p in pt.index]);ax.axhline(.75,color=INK2,ls='--',lw=.6)
 ax.set_ylim(0,1);ax.set_xticks(range(len(pt)),pt.index);ax.set_ylabel('Bootstrap Jaccard')
 for i,v in enumerate(pt.stability):ax.text(i,v+.02,f'{v:.2f}',ha='center',fontsize=6)
 txt(f,6,132,'RNA scores are descriptive overlays. 6,959 cells have eligible paired BCRs; 3,182 belong to displayed profiles. P1/P2 stability <0.75.',6.2)
 txt(f,1,139,'NP–OVA plate study | GSE246382 | day 14: 884 cells, 96 profiles; 11 mice',8,fontweight='bold')
 pc,pt=data('gc_np_pc');w=pd.read_csv(WORK/'story_review_2026-10-12/gc_np_pc_W33L_current.csv')
 pal={'dark zone':'#2a78d6','light zone':'#42b5aa','Myc+ light zone':'#efb13e','plasma cell':VIOLET}
 ax=suppax(f,2,0,'I','Recorded sorting gates');categorical(ax,pc,pc.fate,pal,s=1.5);key(ax,pal,ncol=2,labels={'dark zone':'DZ','light zone':'LZ','Myc+ light zone':'Myc+ LZ','plasma cell':'PC'},size=5.7)
 pal={'germline W33':BLUE,'W33L':RED}
 ax=suppax(f,2,1,'J','Sequence-derived W33L');categorical(ax,pc,pc.w33l,pal,s=1.7);key(ax,pal,ncol=1,size=5.8)
 pal={s:plt.cm.tab20(i) for i,s in enumerate(sorted(pc.donor.astype(str).unique()))}
 ax=suppax(f,2,2,'K','Mouse identity');categorical(ax,pc,pc.donor.astype(str),pal,s=1.5)
 key(ax,pal,ncol=4,size=5.3)
 ax=suppax(f,2,3,'L','Profiled clone members')
 lab=np.where(pc.clone_id.isin(pt.index),'In ≥2-cell profile','Other cells')
 pal={'In ≥2-cell profile':BLUE,'Other cells':GRAY};categorical(ax,pc,lab,pal,s=1.5);key(ax,pal,ncol=1,size=5.7)
 cf(suppax(f,3,0,'M','Clone PC-gate fraction'),w,'fate:plasma cell','Clone PC-gate fraction','cividis')
 cf(suppax(f,3,1,'N','Clone callable W33L fraction'),w,'W33L_fraction','Clone callable W33L fraction','magma_r')
 ax=suppax(f,3,2,'O','W33L vs captured PC fraction');z=w.dropna(subset=['W33L_fraction','PC_fraction'])
 ax.scatter(z.W33L_fraction,z.PC_fraction,s=np.clip(z.n_cells*2.0,3,22),c=BLUE,lw=0,alpha=.8,rasterized=True)
 ax.set_xlim(-.05,1.05);ax.set_ylim(-.05,1.05);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_xlabel('Callable W33L fraction');ax.set_ylabel('Captured PC fraction')
 ax.text(.97,.97,f'n = {len(z)}',transform=ax.transAxes,ha='right',va='top',fontsize=6)
 cellprog(suppax(f,3,3,'P','Programme map-back'),'gc_np_pc')
 txt(f,6,270,'Grey: unavailable annotation or no displayed profile. W33L fractions use callable cells only. Fractions describe captured cells, not output rates.',6.1)
 save(f,'Supplementary_model_antigen_NP',args)

def supplement_reporters(args):
 f=page(282);f.reporter_page=True
 txt(f,1,0,'NP–OVA division reporter | GSE287123 | 10,109 GC-sorted cells, 471 profiles',8,fontweight='bold')
 c,t=data('mouse_np');pal={'mCherry-low':BLUE,'mCherry-high':ORANGE};short={'mCherry-low':'Low (≥6 divisions)','mCherry-high':'High'}
 ax=suppax(f,0,0,'A','Measured division gate');categorical(ax,c,c.division_gate,pal);key(ax,pal,ncol=1,labels=short,size=5.7)
 ax=suppax(f,0,1,'B','Sequence-derived W33L');palw={'germline W33':BLUE,'W33L':RED};categorical(ax,c,c.w33l,palw);key(ax,palw,ncol=1,size=5.8)
 cellprog(suppax(f,0,2,'C','Programme map-back'),'mouse_np')
 cf(suppax(f,0,3,'D','Clone recent-division fraction'),t,'division_gate:mCherry-low','Clone recent-division fraction')
 txt(f,1,69,'RBD protein / mRNA | GSE287123 | 36,188 GC-sorted cells, 1,887 profiles',8,fontweight='bold')
 c,t=data('mouse_rbd')
 clusterplot(suppax(f,1,0,'E','scRNA clusters'),c,'leiden_state','scRNA clusters')
 ax=suppax(f,1,1,'F','Measured division gate');categorical(ax,c,c.division_gate,pal);key(ax,pal,ncol=1,labels=short,size=5.7)
 ax=suppax(f,1,2,'G','Protein arm: bait gate');p={'RBD-':'#8fabb6','RBD+':RED};categorical(ax,c,c.rbd_bait,p);key(ax,p,ncol=2,size=5.8)
 ax=suppax(f,1,3,'H','mRNA arm: zone gate');p={'LZ':ORANGE,'DZ':BLUE};categorical(ax,c,c.zone_gate,p);key(ax,p,ncol=2,size=5.8)
 txt(f,6,140,'Bait and zone labels belong to separate experimental arms; grey cells were not measured for that annotation.',6.2)
 cellprog(suppax(f,2,0,'I','Programme map-back'),'mouse_rbd')
 cf(suppax(f,2,1,'J','Clone recent-division fraction'),t,'division_gate:mCherry-low','Clone recent-division fraction')
 cf(suppax(f,2,2,'K','Clone RBD+ fraction'),t,'rbd_bait:RBD+','Clone RBD+ fraction','cividis')
 cf(suppax(f,2,3,'L','Clone dark-zone fraction'),t,'zone_gate:DZ','Clone dark-zone fraction','magma_r')
 txt(f,1,204,'Oral OVA + CT | GSE284622 | 16,619 cells, 958 profiles; core comparison in Figure 3',8,fontweight='bold')
 c,t=data('gut_gc_reporter')
 ax=suppax(f,3,0,'M','Source RNA annotation');p=STATE_COLORS['S3'];categorical(ax,c,c.celltype,p);key(ax,p,ncol=2,labels={'B_cell':'B cell','GC_Bcell':'GC B cell','plasma':'PC','other':'Other'},size=5.8)
 # Exact source metadata join; categorical sort label, never fluorescence intensity.
 meta=pd.read_csv('/data/scratch/projects/punim1236/threadfin_data/gse284622_gut_gc/gex_reporter/metadata.csv.gz',index_col='cell')
 assert meta.index.is_unique and c.index.isin(meta.index).all()
 sort=meta.reindex(c.index)['gc'];assert sort.isin(['GC','nonGC']).all()
 for cc,mc in [('sample','sample_id'),('tissue','tissue'),('celltype','celltype')]:
  assert c[cc].eq(meta.reindex(c.index)[mc]).all()
 assert np.allclose(sort.eq('GC').groupby(c.clone_id).mean().reindex(t.index),t['gc_sort:GC'],rtol=0,atol=1e-12)
 ax=suppax(f,3,1,'N','Original GC/nonGC sort');p={'GC':'#d94845','nonGC':BLUE};categorical(ax,c,sort,p);key(ax,p,ncol=2,size=5.8)
 ax=suppax(f,3,2,'O','Sampling tissue');p={'MLN':'#7d53ac','PP':'#1e9b8d','LP':'#de9739'};categorical(ax,c,c.tissue,p);key(ax,p,ncol=3,size=5.8)
 cellprog(suppax(f,3,3,'P','Programme map-back'),'gut_gc_reporter')
 txt(f,6,274,'Division gates: 36 h dilution window. Sort/gate labels are categorical; RNA clusters are descriptive. Grey = missing label or no displayed profile.',6.1)
 save(f,'Supplementary_model_antigen_reporters',args)

def supplement_compartments(args):
 f=page(83)
 for i,(ds,title) in enumerate([('gut_gc_reporter','Oral OVA · GSE284622'),('malaria_late','Plasmodium · GSE286215'),('ln_vaccine','Human LN + blood · GSE195673')]):
  c,t=data(ds);dom=old.dominant(ds,c).reindex(t.index).fillna('other')
  ax=axat(f,8+69*i,8,58,53,chr(65+i),title)
  categorical(ax,t,dom,old.COL,unit='Clone',s=np.clip(t.n_cells.to_numpy()*1.5,2.5,30))
  ax.text(.03,.97,f'{len(t):,} profiles',transform=ax.transAxes,va='top',fontsize=6.5)
 figkey(f,8,69,old.COL,ncol=5,size=7)
 txt(f,8,77,'Colour = dominant named compartment among captured clone members; ties or no named compartment are grey.',6.2)
 txt(f,8,81,'Point area increases with captured clone size (capped). Compartments are RNA-derived annotations; proximity describes RNA-profile similarity.',6.2)
 save(f,'Supplementary_clone_compartments',args)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['figure2','np','reporters','compartments','all'],default='all')
 ap.add_argument('--preview',action='store_true');ap.add_argument('--dpi',type=int,default=600);args=ap.parse_args()
 AUDIT.mkdir(parents=True,exist_ok=True)
 for name,fun in [('figure2',figure2),('np',supplement_np),('reporters',supplement_reporters),('compartments',supplement_compartments)]:
  if args.only in [name,'all']:fun(args)
