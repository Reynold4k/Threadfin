#!/usr/bin/env python
"""Prepare and execute published scRNA+scBCR models with audited input adapters.

This is deliberately an adapter, not a reimplementation: it exports the
formats documented by Benisse and BiGCN, invokes their checked-out entry
points, and records only their native outputs.  Reporter/sort labels are never
exported to either method.  Prediction is assessed afterwards by fitting a
ridge readout on whole-donor training folds and evaluating held-out donors.
"""
from __future__ import annotations

import argparse, fcntl, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data"))
SOURCES = ROOT.parent / "internal_validation/competitors/sources"
OUT = ROOT / "case_studies/results/native_benchmark"
sys.path.insert(0, str(ROOT / "case_studies"))


def stamp(out: Path, **kw):
    # Independent native model stages share one manifest. Serialize updates and
    # publish atomically so that a second stage cannot erase a completed status.
    with Path(str(out)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old = json.loads(out.read_text()) if out.exists() else {}
        old.update(kw)
        with tempfile.NamedTemporaryFile('w',dir=out.parent,delete=False) as tmp:
            tmp.write(json.dumps(old,indent=2,sort_keys=True)+'\n')
            temporary=tmp.name
        os.replace(temporary,out)


def load(dataset):
    os.environ["THREADFIN_DATA"] = str(DATA)
    import datasets
    adata, bcr = datasets.LOADERS[dataset]()
    # A native BCR encoder needs junction_aa; this is a published AIRR column,
    # not a Threadfin feature.  Its loader intentionally retains junction only.
    raw = []
    names = {"mouse_np": [("GSM8739133", "Princeton_GC_B_mCherry_HI", "HI"),
                           ("GSM8739136", "Princeton_GC_B_mCherry_Lo", "Lo")],
             "mouse_rbd": [("GSM8739145", "Rebutal_4-5_Tube_A", "A"),
                           ("GSM8739148", "Rebutal_4-5_Tube_B", "B"),
                           ("GSM8739151", "Rebutal_4-5_Tube_C", "C"),
                           ("GSM8739154", "Rebutal_4-5_Tube_D", "D")]}[dataset]
    folder = DATA / "gse287123_np/extracted"
    for gsm, library, prefix in names:
        t = pd.read_csv(folder / f"{gsm}_{library}_airr_rearrangement.tsv.gz", sep="\t", low_memory=False)
        t = t[t.productive.astype(str).str.upper().isin(["T", "TRUE", "1"])]
        t = t[t.v_call.astype(str).str.startswith("IGH")].sort_values("consensus_count", ascending=False)
        t = t.drop_duplicates("cell_id").set_index("cell_id")
        t.index = [f"{prefix}|{x}" for x in t.index.astype(str)]
        raw.append(t[["junction_aa", "junction"]].rename(columns={"junction": "cdr3_nt"}))
    bcr = bcr.join(pd.concat(raw), how="left")
    keep = bcr.junction_aa.notna() & bcr.junction_aa.astype(str).str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
    cells = bcr.index[keep & bcr.index.isin(adata.obs_names)]
    return adata[cells].copy(), bcr.loc[cells].copy()


def export_inputs(dataset):
    """Export the exact documented inputs. No reporter columns are written."""
    import scipy.sparse as sp
    adata, bcr = load(dataset)
    out = OUT / dataset; out.mkdir(parents=True, exist_ok=True)
    # Benisse uses cells x expression representations; log-normalized PCA is
    # explicitly permitted in its README and avoids exporting labels.
    # Shared receptor-excluded RNA basis: BCR loci must not transfer sequence/
    # clonotype information through transcription into an ostensibly joint task.
    genes = adata.var_names.astype(str)
    receptor = genes.str.upper().str.startswith(("IGH", "IGK", "IGL"))
    aexpr = adata[:, ~receptor].copy()
    x = aexpr.X.toarray() if sp.issparse(aexpr.X) else np.asarray(aexpr.X)
    lib = np.maximum(x.sum(1, keepdims=True), 1)
    x = np.log1p(x / lib * 1e4)
    ncomp = min(100, x.shape[0] - 1, x.shape[1] - 1)
    pca = PCA(n_components=ncomp, random_state=0).fit_transform(x)
    ids = adata.obs_names.astype(str)
    pd.DataFrame(pca.T, columns=ids).to_csv(out / "expression_pca.csv")
    contigs = pd.DataFrame({"barcode": ids, "is_cell": True,
        "contig_id": [f"{i}_IGH" for i in ids], "high_confidence": True,
        "length": 0, "chain": "IGH", "v_gene": bcr.v_call.astype(str).values,
        "d_gene": "None", "j_gene": bcr.j_call.astype(str).values,
        "c_gene": bcr.c_call.astype(str).fillna("None").values,
        "full_length": True, "productive": True, "cdr3": bcr.junction_aa.values,
        "cdr3_nt": bcr.cdr3_nt.values, "reads": 0, "umis": 0,
        "raw_clonotype_id": [f"clone_{i}" for i in ids], "raw_consensus_id": "None"})
    contigs.to_csv(out / "contigs.csv", index=False)
    pd.DataFrame({"contigs": contigs.contig_id, "cdr3": contigs.cdr3}).to_csv(out / "bcr_cdr3.csv", index=False)
    # BiGCN's documented input is a BCR embedding plus cell PCA and V/J crude
    # graph.  The embedding itself is the native Benisse encoder output.
    meta = pd.DataFrame({"cell_id": ids, "donor": adata.obs.donor.astype(str).values,
                         "v_call": bcr.v_call.astype(str).values, "j_call": bcr.j_call.astype(str).values})
    meta.to_csv(out / "metadata_no_reporters.csv", index=False)
    stamp(out / "run.json", dataset=dataset, n_cells=int(len(ids)), n_pca=int(ncomp),
          input_status="ready", labels_exported=False, expression_preprocessing="library-size 1e4; log1p; PCA100; exclude IGH/IGK/IGL genes",
          n_receptor_genes_excluded=int(receptor.sum()), source_benisse=str(SOURCES / "Benisse"),
          source_bigcn=str(SOURCES / "BiGCN"), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return out


def run_benisse_encoder(dataset, python=sys.executable):
    out = OUT / dataset; repo = SOURCES / "Benisse"
    start = time.monotonic()
    p = subprocess.run([python, str(repo / "AchillesEncoder.py"), "--input_data", str(out / "bcr_cdr3.csv"),
                        "--output_data", str(out / "benisse_encoded.csv"), "--cuda", "False"], capture_output=True, text=True)
    (out / "benisse_encoder.stdout.log").write_text(p.stdout + p.stderr)
    stamp(out / "run.json", benisse_encoder_status="completed" if p.returncode == 0 else "failed",
          benisse_encoder_returncode=p.returncode, benisse_encoder_seconds=round(time.monotonic()-start, 3),
          benisse_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
    if p.returncode: raise RuntimeError(p.stderr[-2000:])


def run_benisse_r(dataset, rscript="Rscript"):
    """Execute the official sparse-graph R stage after the official encoder."""
    out=OUT / dataset; repo=SOURCES / "Benisse"; target=out / "benisse_r"
    target.mkdir(exist_ok=True); start=time.monotonic()
    # R read.csv(check.names=TRUE) changes our library-prefixed cell IDs. Use
    # reversible syntactic identifiers, without changing any sequence or value.
    expression=pd.read_csv(out/'expression_pca.csv',index_col=0)
    aliases=pd.DataFrame({'cell_id':expression.columns,
                          'alias':[f'cell{i:06d}' for i in range(expression.shape[1])]})
    mapping=aliases.set_index('cell_id').alias
    expression.columns=aliases.alias
    expression.to_csv(out/'expression_benisse.csv')
    contigs=pd.read_csv(out/'contigs.csv')
    contigs['barcode']=contigs.barcode.map(mapping)
    assert contigs.barcode.notna().all()
    contigs.to_csv(out/'contigs_benisse.csv',index=False)
    aliases.to_csv(out/'benisse_cell_aliases.csv',index=False)
    cmd=[rscript, str(repo / "Benisse.R"), str(out / "expression_benisse.csv"), str(out / "contigs_benisse.csv"),
         str(out / "benisse_encoded.csv"), str(target), "1610", "1", "100", "1", "1", "10", "1e-10"]
    p=subprocess.run(cmd, capture_output=True, text=True)
    (out / "benisse_r.stdout.log").write_text(p.stdout+p.stderr)
    stamp(out / "run.json", benisse_r_status="completed" if p.returncode==0 else "failed",
          benisse_r_returncode=p.returncode, benisse_r_seconds=round(time.monotonic()-start,3))
    if p.returncode: raise RuntimeError(p.stderr[-2000:])


def run_bigcn(dataset, python=sys.executable):
    """Run the official BiGCN scripts on their native exact V-CDR3AA-J nodes."""
    out = OUT / dataset; repo = SOURCES / "BiGCN"; work = out / "BiGCN_official"
    if work.exists(): shutil.rmtree(work)
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", "dataset.zip"))
    # Official config overwrites an explicit --num_nodes with its toy-data
    # constant (1494).  Preserve the command-line value only; model equations
    # and all graph construction code remain upstream.
    cfg = work / "config.py"
    cfg.write_text(cfg.read_text().replace("elif args.dataset == 'newFile':\n    args.num_nodes = 1494", "elif args.dataset == 'newFile' and args.num_nodes == 8723:\n    args.num_nodes = 1494"))
    gs = work / "graphStructure.py"
    gs.write_text(gs.read_text().replace("dataset='Covid19'", "dataset=args.dataset"))
    adata, bcr = load(dataset)
    encoded = pd.read_csv(out / "benisse_encoded.csv", index_col=0)
    encoded = encoded.drop(columns="index").set_index(pd.read_csv(out / "benisse_encoded.csv", index_col=0)["index"])
    native = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "cdr3": bcr.junction_aa.astype(str).values,
                           "v": bcr.v_call.astype(str).str.split(",").str[0].str.split("*").str[0].values,
                           "j": bcr.j_call.astype(str).str.split(",").str[0].str.split("*").str[0].values})
    native["node"] = native.v + "|" + native.cdr3 + "|" + native.j
    native = native[native.cdr3.isin(encoded.index)].copy()
    # Official BiGCN pools cell PCs per its V_CDR3H_J node; retain that unit.
    pca = pd.read_csv(out / "expression_pca.csv", index_col=0).T
    native = native.join(pca, on="cell_id")
    nodes = native.groupby("node", sort=True)
    ids = list(nodes.groups); emb = encoded.loc[nodes.cdr3.first().values].copy(); emb.index = ids
    pcs = nodes[[str(i) if str(i) in native.columns else i for i in pca.columns]].mean()
    d = work / "dataset/newFile"; d.mkdir(parents=True)
    (work / "logger").mkdir()
    # Files intentionally have no headers: that is how BiGCN's CSV reader is implemented.
    b = pd.DataFrame([[i, ids[i], ids[i], *emb.loc[ids[i]].to_numpy()] for i in range(len(ids))])
    b.to_csv(d / "BCR_embedding.csv", header=False, index=False)
    rows=[]
    for node, g in nodes:
        for _, r in g.iterrows(): rows.append([r.cell_id, node, *r[pca.columns].to_numpy()])
    pd.DataFrame(rows).to_csv(d / "exp_pca.csv", header=False, index=False)
    ej = pd.DataFrame({"v": nodes.v.first(), "j": nodes.j.first()})
    adj=np.zeros((len(ids),len(ids)), dtype=float)
    e=emb.to_numpy(float); e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    for i in range(len(ids)):
        same=((ej.v.values==ej.v.values[i]) | (ej.j.values==ej.j.values[i])); adj[i,same]=e[i]@e[same].T
    np.savetxt(d / "crudegraph.csv", adj, delimiter=",")
    for sub in ("file", "graph/cos", "embedding", "img"): (work / "dataset/output/newFile" / sub).mkdir(parents=True, exist_ok=True)
    start=time.monotonic(); logs=[]
    for cmd in ([python,"data_process.py","--dataset","newFile","--num_nodes",str(len(ids))],
                [python,"graphStructure.py","--dataset","newFile","--top_k","100"],
                [python,"main.py","--dataset","newFile","--num_nodes",str(len(ids)),"--max_epoch","1500"]):
        p=subprocess.run(cmd,cwd=work,capture_output=True,text=True); logs.extend([" ".join(cmd),p.stdout,p.stderr])
        if p.returncode: break
    (out / "bigcn.stdout.log").write_text("\n".join(logs))
    status="completed" if p.returncode == 0 else "failed"
    stamp(out / "run.json", bigcn_status=status, bigcn_returncode=p.returncode,
          bigcn_seconds=round(time.monotonic()-start,3), bigcn_native_nodes=len(ids),
          bigcn_adapter_patch="preserve --num_nodes over upstream newFile toy constant (1494); pass parsed dataset to graphStructure",
          bigcn_commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip())
    if p.returncode: raise RuntimeError(p.stderr[-2000:])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("dataset", choices=["mouse_np","mouse_rbd"])
    ap.add_argument("--stage", choices=["prepare","encode","benisse-r","bigcn"], default="prepare"); ap.add_argument("--python", default=sys.executable)
    a=ap.parse_args()
    # Model stages read immutable, completed common input. They must not race
    # to overwrite expression_pca.csv while another model is reading it.
    if a.stage=='prepare' or not (OUT/a.dataset/'expression_pca.csv').exists():
        export_inputs(a.dataset)
    if a.stage == "encode": run_benisse_encoder(a.dataset, a.python)
    if a.stage == "benisse-r": run_benisse_r(a.dataset)
    if a.stage == "bigcn": run_bigcn(a.dataset, a.python)

if __name__ == "__main__": main()
