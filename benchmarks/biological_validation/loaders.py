"""Dataset-specific loaders for Threadfin biological validation.

Every loader returns ``(adata, bcr)`` where:

* ``adata``: AnnData of raw counts with dataset metadata in ``obs``
  (``donor``, ``timepoint``, ``condition`` where available; ``state`` only
  when the dataset ships curated labels — otherwise the pipeline computes
  Leiden states).
* ``bcr``: per-cell BCR table indexed by barcodes matching
  ``adata.obs_names``, with at least ``clone_id, v_call, d_call, j_call,
  cdr3`` columns.

Loaders perform no QC beyond what is needed to align the two modalities;
preprocessing happens in validation.standard_preprocess.
"""

from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read_mtx_dir(path: Path, prefix: str | None = None):
    """Read a 10x-style mtx triplet directory into an AnnData."""
    import scanpy as sc

    adata = sc.read_mtx(path / ("matrix.mtx.gz" if (path / "matrix.mtx.gz").exists() else "matrix.mtx"))
    adata = adata.T
    import scipy.sparse as sp

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    bc = pd.read_csv(
        path / ("barcodes.tsv.gz" if (path / "barcodes.tsv.gz").exists() else "barcodes.tsv"),
        header=None)[0].astype(str)
    feat_file = "features.tsv.gz" if (path / "features.tsv.gz").exists() else "genes.tsv.gz"
    if not (path / feat_file).exists():
        feat_file = feat_file.removesuffix(".gz")
    feats = pd.read_csv(path / feat_file, header=None, sep="\t")
    genes = feats[1].astype(str) if feats.shape[1] > 1 else feats[0].astype(str)
    adata.obs_names = bc.values
    adata.var_names = genes.values
    adata.var_names_make_unique()
    return adata


def _extract_tar_member(tar: Path, outdir: Path) -> Path:
    """Extract a tar[.gz] once and return the directory containing the matrix."""
    import os

    marker = outdir / (tar.stem.replace(".tar", "") + ".extracted")
    if marker.exists():
        return Path(marker.read_text().strip())
    outdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf_:
        tf_.extractall(outdir, filter="data")
    # locate directory holding the matrix file
    for root, _dirs, files in os.walk(outdir):
        if any(f.startswith("matrix.mtx") for f in files):
            marker.write_text(root)
            return Path(root)
    raise FileNotFoundError(f"no matrix.mtx found inside {tar}")


def _extract_tar(tar: Path, outdir: Path) -> Path:
    """Plain extraction (no content check); returns the output dir."""
    marker = outdir / ".done"
    if not marker.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf_:
            tf_.extractall(outdir, filter="data")
        marker.touch()
    return outdir


# ---------------------------------------------------------------- Stephenson 2021


def load_stephenson_h5mu(cfg):
    import awkward as ak
    import muon as mu

    m = mu.read_h5mu(cfg["data"])
    gex, airr = m["gex"], m["airr"]
    adata = gex.copy()
    adata.X = adata.layers["raw"].copy()  # raw counts
    adata.obs["state"] = adata.obs["initial_clustering"].astype(str)
    adata.obs["donor"] = adata.obs["patient_id"].astype(str)

    a = airr.obsm["airr"]
    prod = a["productive"]
    productive = ak.fill_none(prod, False) if "bool" in str(ak.type(prod)) else (
        (prod == "T") | (prod == "True"))
    igh = a[(a["locus"] == "IGH") & productive]
    first = ak.firsts(igh)
    bcr = pd.DataFrame(
        {"v_call": ak.to_list(first["v_call"]), "d_call": ak.to_list(first["d_call"]),
         "j_call": ak.to_list(first["j_call"]), "cdr3": ak.to_list(first["cdr3_aa"])},
        index=airr.obs_names,
    ).dropna(subset=["v_call"])
    bcr.index.name = "barcode"
    import threadfin as tf

    bcr = tf.build_clone_key(bcr, strategy="cdr3")  # h5mu has no author clone ids
    return adata, bcr


# ---------------------------------------------------------------- GSE317492 EBV organoid


_EBV_CONDITIONS = {
    "d0": ("d0", "uninfected"),
    "d4": ("d4", "uninfected"),
    "d7": ("d7", "uninfected"),
    "d14-gfp-neg-b": ("d14", "GFP-"),
    "d14_gfp-pos-b": ("d14", "GFP+"),
    "d21-gfp-neg-b": ("d21", "GFP-"),
    "d21_gfp-pos-b": ("d21", "GFP+"),
}

_EBV_GEX = {
    "d0": "GSM9473008", "d4": "GSM9473012", "d7": "GSM9473016",
    "d14-gfp-neg-b": "GSM9473020", "d14_gfp-pos-b": "GSM9473023",
    "d21-gfp-neg-b": "GSM9473027", "d21_gfp-pos-b": "GSM9473030",
}
_EBV_BCR = {
    "d0": "GSM9473009", "d4": "GSM9473013", "d7": "GSM9473017",
    "d14-gfp-neg-b": "GSM9473021", "d14_gfp-pos-b": "GSM9473024",
    "d21-gfp-neg-b": "GSM9473028", "d21_gfp-pos-b": "GSM9473031",
}


def load_ebv_gse317492(cfg):
    """10x MTX triplets + cellranger VDJ CSVs; samples concatenated with
    condition/timepoint labels. NOTE: GEO samples pool 4 donors."""
    import scanpy as sc
    import threadfin as tf

    d = Path(cfg["data_dir"])
    adatas, bcrs = [], []
    for tag, (tp, cond) in _EBV_CONDITIONS.items():
        gsm_g, gsm_b = _EBV_GEX[tag], _EBV_BCR[tag]
        sub = d / f"{tag}_mtx"
        sub.mkdir(exist_ok=True)
        for part in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"):
            src = d / f"{gsm_g}_{tag}_gex_{part}"
            dst = sub / part
            if not dst.exists():
                dst.symlink_to(src)
        ad = _read_mtx_dir(sub)
        ad.obs_names = [f"{tag}|{b}" for b in ad.obs_names]
        ad.obs["timepoint"] = tp
        ad.obs["condition"] = cond

        contig = d / f"{gsm_b}_{tag}_vdjB_filtered_contig_annotations.csv.gz"
        tab = tf.read_10x_vdj(str(contig))
        tab.index = [f"{tag}|{b}" for b in tab.index]
        cid = tab["clonotype_id"]
        tab["clonotype_id"] = (tag + "|" + cid.astype(str)).mask(
            cid.isna() | cid.astype(str).isin(("None", "")))
        bcrs.append(tab)
        adatas.append(ad)

    import anndata as ad_

    adata = ad_.concat(adatas, join="outer", fill_value=0)
    bcr = tf.build_clone_key(pd.concat(bcrs), strategy="clonotype_id")
    return adata, bcr


# ---------------------------------------------------------------- GSE175522 flu


def load_flu_gse175522(cfg):
    """Per-sample cellranger tars (GEX) + author AIRR tables (BCR).

    Sample id ``<donor>_<tp>``: donor ids 120648/120667/141393 (young) vs
    141394/141409/141415 (older); tp 0 = pre-vaccination, 7 = day 7.
    """
    import threadfin as tf

    d = Path(cfg["data_dir"])
    young = {"120648", "120667", "141393"}
    adatas, bcrs = [], []
    for gex_tar in sorted(d.glob("*_cellranger.tar.gz")):
        gsm, donor, tp = gex_tar.name.split("_")[:3]
        sample = f"{donor}_{tp}"
        mtx_dir = _extract_tar_member(gex_tar, d / "extracted" / sample)
        ad = _read_mtx_dir(mtx_dir)
        ad.obs_names = [f"{sample}|{b.split('-')[0]}" for b in ad.obs_names]
        ad.obs["donor"] = donor
        ad.obs["timepoint"] = "d0" if tp == "0" else "d7"
        ad.obs["condition"] = "young" if donor in young else "older"
        adatas.append(ad)

        airr = next(d.glob(f"GSM*_{sample}_clone_pass_fil_airr.txt.gz"))
        tab = tf.read_airr(str(airr), cell_col="cell_id")
        tab.index = [f"{sample}|{b.split('-')[0]}" for b in tab.index]
        if "clone_id" in tab.columns:
            pass  # keep author clone ids
        bcrs.append(tab)

    import anndata as ad_

    adata = ad_.concat(adatas, join="outer", fill_value=0)
    bcr = pd.concat(bcrs)
    if "clone_id" not in bcr.columns:
        bcr = tf.build_clone_key(bcr, strategy="vdj")
    return adata, bcr


# ---------------------------------------------------------------- King 2021 tonsil


def load_tonsil_king2021(cfg):
    """Per-donor filtered_feature_bc_matrix tars + scVDJ contig tars +
    author cell-type metadata (CellTypeMetaData.txt)."""
    import threadfin as tf

    d = Path(cfg["data_dir"])
    adatas, bcrs = [], []
    for gex_tar in sorted(d.glob("BCP*_Total_5GEX.tar.gz")):
        donor = gex_tar.name.split("_")[0]
        mtx_dir = _extract_tar_member(gex_tar, d / "extracted" / donor)
        ad = _read_mtx_dir(mtx_dir)
        ad.obs_names = [f"{donor}|{b.split('-')[0]}" for b in ad.obs_names]
        ad.obs["donor"] = donor
        adatas.append(ad)

        vdj_tar = d / f"{donor}_Total_scVDJ.tar.gz"
        vdj_dir = _extract_tar(vdj_tar, d / "extracted" / f"{donor}_vdj")
        csvs = list(Path(vdj_dir).rglob("filtered_contig_annotations.csv*"))
        tab = tf.read_10x_vdj(str(csvs[0]))
        tab.index = [f"{donor}|{b.split('-')[0]}" for b in tab.index]
        # namespace cellranger clonotype ids per donor before concatenation
        cid = tab["clonotype_id"]
        tab["clonotype_id"] = (donor + "|" + cid.astype(str)).mask(
            cid.isna() | cid.astype(str).isin(("None", "")))
        bcrs.append(tab)

    import anndata as ad_

    adata = ad_.concat(adatas, join="outer", fill_value=0)
    bcr = tf.build_clone_key(pd.concat(bcrs), strategy="clonotype_id")
    # merge SHM variants into lineages: exact cellranger clonotypes in the
    # 'Total' libraries are mostly singletons (13 clones >=3 cells);
    # sequence-similarity grouping recovers expanded GC lineages (~266).
    bcr["clone_id"] = tf.define_clones(
        bcr, cdr3_sim_threshold=0.85, same_vj=True,
        method="connected", out_col="seq_clone")

    # author cell-type annotations (non-circular reference states)
    meta_path = d / "CellTypeMetaData.txt"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, sep="\t", index_col=0)
        meta.index = meta.index.astype(str)
        # meta barcodes look like 'BCP2_Total_AAACCTGGTACGAAAT'
        parsed = meta.index.str.extract(r"^BCP(\d+)_[A-Za-z0-9]+_([A-Z]{16})$")
        ok = parsed[0].notna()
        meta = meta[ok.values]
        meta.index = [f"BCP{int(dnr):03d}|{bc}" for dnr, bc in parsed.loc[ok.values].values]
        meta = meta[~meta.index.duplicated(keep="first")]
        adata.obs = adata.obs.join(meta, how="left")
        n_annot = adata.obs["Subset"].notna().sum() if "Subset" in adata.obs.columns else 0
        print(f"[tonsil] author annotations matched: {n_annot}/{adata.n_obs}", flush=True)
        for cand in ("Subset", "CellType", "cell_type"):
            if cand in adata.obs.columns and adata.obs[cand].notna().sum() > 0:
                adata.obs["state"] = adata.obs[cand].astype(str)
                break
    # keep B lineage only (the 'Total' libraries include T cells and others)
    if "Lineage" in adata.obs.columns:
        n0 = adata.n_obs
        adata = adata[adata.obs["Lineage"].astype(str).str.contains("B")].copy()
        print(f"[tonsil] B-lineage subset: {adata.n_obs}/{n0} cells", flush=True)
    return adata, bcr


# ---------------------------------------------------------------- GSE195673 LN vaccine


def load_ln_vaccine_gse195673(cfg):
    """Author-integrated B-cell h5ad + Change-O BCR heavy/light tables.

    The h5ad obs_names are bare 10x barcodes that collide across samples;
    the unique key is obs["cell_id"] ('368-01a_s5@BARCODE-1'), which matches
    the BCR tables' cell_id column.
    """
    import scanpy as sc

    d = Path(cfg["data_dir"])
    adata = sc.read_h5ad(d / "gex_b_cells.h5ad")
    for layer in ("counts", "raw_counts"):
        if layer in adata.layers:
            adata.X = adata.layers[layer].copy()
            break

    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)
    adata.obs_names = adata.obs["cell_id"].values
    if adata.obs_names.duplicated().any():
        raise ValueError("cell_id not unique in LN vaccine h5ad")
    # author cell-state annotation as reference labels
    if "anno_leiden_0.18" in adata.obs.columns:
        adata.obs["state"] = adata.obs["anno_leiden_0.18"].astype(str)
    for col in ("donor", "sample", "tissue"):
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype(str)

    # timepoint/compartment from bcr_meta.tsv via the 'donor_sample' prefix of
    # cell_id (e.g. '368-01a_s5@BARCODE-1' -> donor 368-01a, sample s5)
    meta_path = d / "bcr_meta.tsv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, sep="\t")
        meta["ds"] = meta["donor"].astype(str) + "_" + meta["sample"].astype(str)
        meta = meta.drop_duplicates("ds").set_index("ds")
        ds_key = adata.obs["cell_id"].str.split("@").str[0]
        for src, dst in (("timepoint", "timepoint"), ("sorting", "compartment")):
            if src in meta.columns:
                adata.obs[dst] = ds_key.map(meta[src]).astype("string").values
        n_tp = adata.obs["timepoint"].notna().sum() if "timepoint" in adata.obs.columns else 0
        print(f"[ln] timepoint mapped for {n_tp}/{adata.n_obs} cells", flush=True)

    heavy = pd.read_csv(d / "bcr_heavy.tsv.gz", sep="\t", low_memory=False)
    per_cell = heavy.drop_duplicates("cell_id").set_index("cell_id")
    if "isotype" in per_cell.columns and "c_call" not in per_cell.columns:
        per_cell = per_cell.rename(columns={"isotype": "c_call"})
    keep = [c for c in ("v_call", "d_call", "j_call", "c_call", "cdr3_aa", "clone_id",
                        # discovery layer: author spike-specificity, ELISA-validated
                        # mAb labels, SHM frequency, BCR compartment
                        "s_pos_clone", "elisa", "nuc_RS_freq_19_312", "compartment")
            if c in per_cell.columns]
    bcr = per_cell[keep].rename(columns={"cdr3_aa": "cdr3"})
    bcr.index.name = "barcode"
    if "clone_id" not in bcr.columns:
        import threadfin as tf

        bcr = tf.build_clone_key(bcr, strategy="vdj")
    return adata, bcr


LOADERS = {
    "stephenson_h5mu": load_stephenson_h5mu,
    "ebv_gse317492": load_ebv_gse317492,
    "flu_gse175522": load_flu_gse175522,
    "tonsil_king2021": load_tonsil_king2021,
    "ln_vaccine_gse195673": load_ln_vaccine_gse195673,
}
