#!/usr/bin/env python
"""Public GSE305877 WT day-14 replicate 1: donor-resolved descriptive example.

Other WT replicates are pooled and are not analysed without per-cell HTO assignments.
No author state labels, sort labels, isotype or repertoire sequences enter RNA features.
"""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Threadfin'))
FIG=ROOT/'Threadfin/paper/figure_plan'
RAW=FIG/'reserve/story_relayout_2026-10-16/data/GSE305877'
OUT=ROOT/'Threadfin/case_studies/results/np_cgg_wt_d14'
OUT.mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd

def call():
 from threadfin.dandelion import read_contigs,define_clones_dandelion
 t=read_contigs(RAW/'GSM9186599_filtered_contig_annotations_Experiment1_D14_WT_BCR.csv.gz')
 t=t[t['productive'].astype(str).str.upper().isin(['TRUE','T','1'])].copy()
 if 'full_length' in t:t=t[t['full_length'].astype(str).str.upper().isin(['TRUE','T','1'])].copy()
 t=t[t.locus.isin(['IGH','IGK','IGL'])]
 chains=pd.crosstab(t.cell_id,t.locus).reindex(columns=['IGH','IGK','IGL'],fill_value=0)
 good=chains.index[(chains.IGH==1)&((chains.IGK+chains.IGL)==1)]
 audit={'productive_bcr_cells':int(len(chains)),'exactly_one_heavy_one_light':int(len(good)),
        'exclusion':'Cells with missing or more than one productive full-length heavy/light contig are not clone-called.'}
 t=t[t.cell_id.isin(good)].copy();t['donor']='WT_d14_rep1'
 t['cell_id']='WT_d14_rep1|'+t['cell_id'].astype(str)
 t['sequence_id']='WT_d14_rep1|'+t['sequence_id'].astype(str)
 ids=define_clones_dandelion(t,donor_key='donor',use_light_chain=True,identity=.85)
 ids.to_csv(OUT/'clone_ids.csv')
 h=t[t.locus.eq('IGH')].set_index('cell_id')[['v_call','j_call','c_call','junction_aa']]
 h['clone_id']=ids;h.to_csv(OUT/'bcr_per_cell.csv')
 (OUT/'clone_call_audit.json').write_text(json.dumps(dict(ids.attrs)|audit,indent=2))


def analyse():
 import scanpy as sc
 import threadfin as tf
 start=time.time()
 def log(s):print(f'[{time.time()-start:.0f}s] {s}',flush=True)
 rawfile=RAW/'GSM9186600_filtered_feature_bc_matrix_Experiment1_D14_WT_GEX.h5'
 a=sc.read_10x_h5(rawfile);a.var_names_make_unique();a.obs_names='WT_d14_rep1|'+a.obs_names
 a.obs['donor']='WT_d14_rep1';a.obs['sample']='WT_d14_rep1';a.obs['day']='d14'
 a.var['mt']=a.var_names.str.upper().str.startswith('MT-')
 sc.pp.calculate_qc_metrics(a,qc_vars=['mt'],percent_top=None,log1p=False,inplace=True)
 before=a.n_obs
 keep=(a.obs.n_genes_by_counts>=200)&(a.obs.total_counts>=500)&(a.obs.pct_counts_mt<20)
 a=a[keep].copy();sc.pp.filter_genes(a,min_cells=3)
 b=pd.read_csv(OUT/'bcr_per_cell.csv',index_col=0)
 a.obs['clone_id']=a.obs_names.map(b.clone_id)
 a.obs['isotype']=a.obs_names.map(b.c_call)
 a.obs['heavy_v_gene']=a.obs_names.map(b.v_call)
 sizes=a.obs.clone_id.value_counts()
 info={'dataset':'GSE305877','subset':'WT day 14 experiment 1, one identifiable mouse',
       'source_gex':'GSM9186600','source_bcr':'GSM9186599','n_cells_raw':before,'n_cells':a.n_obs,
       'n_cells_paired_bcr':int(a.obs.clone_id.notna().sum()),'n_clones_all':int(len(sizes)),
       'n_profiles_ge2':int((sizes>=2).sum()),'n_cells_profiles_ge2':int(sizes[sizes>=2].sum()),
       'qc':{'min_genes':200,'min_counts':500,'max_mito_percent_exclusive':20,'min_cells_per_gene':3},
       'pooled_replicates':'Excluded: per-cell HTO assignments absent in downloaded processed files.',
       'map_input':'RNA only; immunoglobulin/receptor genes excluded; no author state labels or isotype supplied.',
       'replication':'One mouse: descriptive, not a between-animal validation.',
       'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.glob('*') if p.is_file()}}
 log(info)
 tf.pp.prepare_embedding(a,batch_key=None,random_state=0)
 sc.pp.neighbors(a,use_rep='X_threadfin',n_neighbors=15,random_state=0)
 sc.tl.leiden(a,resolution=.6,random_state=0,flavor='igraph',n_iterations=2,directed=False,key_added='rna_cluster')
 sc.tl.umap(a,random_state=0)
 # Compact post hoc, RNA-derived descriptive scores. No annotation-based cell filtering.
 sets={'GC score':['Bcl6','Aicda','Rgs13','S1pr2','Mef2b'],
       'LZ score':['Cd83','Cd86','Nfkbia','H2-DMa','H2-Oa'],
       'Cycling score':['Mki67','Top2a','Cenpf','Ccnb1','Cdk1'],
       'PC score':['Prdm1','Xbp1','Mzb1','Sdc1','Irf4']}
 info['post_hoc_gene_sets']=sets
 for name,genes in sets.items():
  present=[g for g in genes if g in a.var_names]
  v=a.layers['log_norm'][:,a.var_names.get_indexer(present)].toarray()
  z=(v-v.mean(0))/np.maximum(v.std(0),1e-6)
  a.obs[name]=z.mean(1)
 log('RNA UMAP and descriptive scores complete')
 tf.tl.clone_profiles(a,basis='X_threadfin',context_key='donor',donor_key='donor',representation='mean',min_cells=2)
 coh=tf.tl.clonal_coherence(a,strata_key='sample',n_perm=500,verbose=True)
 info['coherence']={k:coh[k] for k in ['icc','null_mean','null_q95','excess','p_value','n_clones','n_cells']}
 tf.tl.clone_profiles(a,basis='X_threadfin',context_key='donor',donor_key='donor',representation='kernel',min_cells=2)
 tf.tl.embed_clones(a,n_neighbors=50,min_dist=.8,random_state=0)
 info['clone_map']={k:v for k,v in a.uns['threadfin']['profiles']['map_params'].items() if k not in ('attempts','tuning')}
 prog=tf.tl.find_programmes(a,n_boot=50,random_state=0,verbose=True)
 prog.to_csv(OUT/'programmes.csv')
 t=a.uns['threadfin']['profiles']['clone_table'].copy()
 if 'clone_id' not in t:t=t.rename_axis('clone_id').reset_index()
 t.to_csv(OUT/'clone_table.csv',index=False)
 if 'split_tests' in a.uns['threadfin']['programmes']:a.uns['threadfin']['programmes']['split_tests'].to_csv(OUT/'programme_split_tests.csv',index=False)
 obs=a.obs.copy();obs['umap_1']=a.obsm['X_umap'][:,0];obs['umap_2']=a.obsm['X_umap'][:,1]
 obs.to_csv(OUT/'cells.csv.gz')
 for g in ['Bcl6','Aicda','Cd83','Mki67','Xbp1','Prdm1','Nr4a1']:
  if g in a.var_names:obs[g]=a.layers['log_norm'][:,a.var_names.get_loc(g)].toarray().ravel()
 obs.to_csv(OUT/'cells_with_scores.csv.gz')
 info['n_programmes']=int(prog.shape[0]);info['programme_counts']=t.clone_programme.value_counts().to_dict()
 info['rna_cluster_counts']=a.obs.rna_cluster.value_counts().to_dict()
 a.obs.groupby('rna_cluster',observed=True)[list(sets)].mean().to_csv(OUT/'rna_cluster_scores.csv')
 sc.tl.rank_genes_groups(a,'rna_cluster',layer='log_norm',method='wilcoxon',n_genes=20)
 sc.get.rank_genes_groups_df(a,group=None).to_csv(OUT/'rna_cluster_markers.csv',index=False)
 # Keep raw counts, log-normalised expression, and fixed embeddings. Profile tables are separate CSVs.
 a.uns.pop('threadfin',None)
 a.write_h5ad(OUT/'rna_embedding.h5ad',compression='gzip')
 def native(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (OUT/'summary.json').write_text(json.dumps(info,indent=2,default=native)+'\n')
 log('Complete: '+str(OUT))

if __name__=='__main__':
 if '--call' in sys.argv:call()
 else:
  if not (OUT/'clone_ids.csv').exists():
   subprocess.run(['/data/scratch/projects/punim1236/python_envs/dandelion/bin/python',str(Path(__file__).resolve()),'--call'],check=True)
  analyse()
