"""Loaders for the public paired scRNA-seq + BCR-seq datasets.

Every loader returns ``(adata, bcr)``:

* ``adata`` - raw UMI counts in ``X``; ``obs`` holds ``donor``, ``sample`` and
  the dataset's metadata (time point, tissue, sort gate, author cell type ...);
* ``bcr`` - one row per cell (index = ``adata.obs_names``) with the heavy-chain
  ``v_call``, ``j_call``, ``junction`` or ``cdr3_nt``, ``c_call`` and any
  receptor-level labels the authors measured (antigen binding, mutation
  frequency, affinity markers). Clones are *not* defined here: the pipeline
  calls :func:`threadfin.define_clones` on this table, within donors.

Raw files live under ``$THREADFIN_DATA`` (default below); download commands
and checksums are in ``download.sh``, ``datasets_manifest.tsv`` and ``md5_manifest.txt``.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parent / "data"))


# =========================================================================== helpers


def read_mtx_dir(path: Path, prefix: str):
    """10x matrix triplet (``prefix + matrix.mtx.gz`` ...) -> AnnData (cells x features)."""
    import anndata as ad
    import scipy.io
    import scipy.sparse as sp

    def pick(stem):
        for cand in (f"{prefix}{stem}.gz", f"{prefix}{stem}"):
            if (path / cand).exists():
                return path / cand
        raise FileNotFoundError(f"{path}/{prefix}{stem}[.gz] not found")

    x = sp.csr_matrix(scipy.io.mmread(pick("matrix.mtx")).T)
    barcodes = pd.read_csv(pick("barcodes.tsv"), header=None)[0].astype(str).to_numpy()
    feats = pd.read_csv(pick("features.tsv"), header=None, sep="\t")
    var = pd.DataFrame({
        "gene_ids": feats[0].astype(str).to_numpy(),
        "feature_type": (feats[2] if feats.shape[1] > 2 else pd.Series("Gene Expression", index=feats.index)).to_numpy(),
    }, index=feats[1].astype(str).to_numpy())
    out = ad.AnnData(X=x, obs=pd.DataFrame(index=barcodes), var=var)
    out.var_names_make_unique()
    return out


def demux_hashtags(counts: pd.DataFrame, min_total: int = 5) -> pd.Series:
    """Assign cells to hashtags (CLR + per-hashtag 2-means threshold).

    Each hashtag's CLR-normalised counts are split into background and signal
    by 1-D k-means; a cell positive for exactly one hashtag is a singlet,
    for several a ``doublet``, for none (or with < ``min_total`` counts)
    ``negative``.
    """
    from sklearn.cluster import KMeans

    x = counts.to_numpy(dtype=float)
    logx = np.log1p(x)
    clr = logx - logx.mean(axis=0, keepdims=True)
    positive = np.zeros_like(clr, dtype=bool)
    for j in range(clr.shape[1]):
        km = KMeans(2, n_init=10, random_state=0).fit(clr[:, [j]])
        hi = int(np.argmax(km.cluster_centers_.ravel()))
        positive[:, j] = km.labels_ == hi
    n_pos = positive.sum(axis=1)
    out = np.where(n_pos == 1, np.asarray(counts.columns)[positive.argmax(axis=1)], "negative")
    out = np.where(n_pos > 1, "doublet", out)
    out = np.where(x.sum(axis=1) < min_total, "negative", out)
    return pd.Series(out, index=counts.index, name="hashtag")


def _translate(nt: str) -> str:
    from threadfin.sequence import _CODON_TABLE

    return "".join(_CODON_TABLE.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))


def w33l_status(airr: pd.DataFrame) -> pd.Series:
    """VH186.2 (IGHV1-72) Kabat W33L, the canonical NP high-affinity mutation.

    The query and germline V regions are put in a common frame (they can
    start at different points in the two alignment strings, see
    :func:`threadfin.clones.align_starts`) and translated; Kabat position 33
    is the W of the germline CDR1 motif ``SYWMH``. Returns the residue found
    there - ``"L"`` (high affinity) or ``"W"`` (germline) - and NaN when the
    cell does not use IGHV1-72 or the motif is not found.
    """
    from threadfin.clones import v_region_windows

    out = pd.Series(np.nan, index=airr.index, dtype=object)
    use = airr["v_call"].astype(str).str.split(",").str[0].str.split("*").str[0].eq("IGHV1-72")
    if not use.any():
        return out
    sub = airr[use]
    win = v_region_windows(sub, "sequence_alignment", "germline_alignment")
    for idx in sub.index:
        g, q = sub.at[idx, "germline_alignment"], sub.at[idx, "sequence_alignment"]
        q0, g0, n = win.loc[idx]
        if not (isinstance(g, str) and isinstance(q, str)) or not np.isfinite(n):
            continue
        q0, g0, n = int(q0), int(g0), int(n)
        g_aa, q_aa = _translate(g[g0:g0 + n].upper()), _translate(q[q0:q0 + n].upper())
        pos = g_aa.find("SYWMH")
        if pos >= 0 and pos + 2 < len(q_aa):
            out.at[idx] = q_aa[pos + 2]
    return out


def _airr_heavy(path, keep_extra=()) -> pd.DataFrame:
    """Productive heavy chains of an AIRR file, one per cell (highest UMI count)."""
    import threadfin as tf

    tab = pd.read_csv(path, sep="\t", low_memory=False)
    prod = tab["productive"].astype(str).str.upper().isin(("T", "TRUE", "1"))
    tab = tab[prod & tab["v_call"].astype(str).str.upper().str.startswith("IGH")]
    if {"sequence_alignment", "germline_alignment"} <= set(tab.columns):
        tab["mutation_frequency"] = tf.tl.mutation_frequency(tab)
    cnt = next((c for c in ("umi_count", "duplicate_count", "consensus_count") if c in tab.columns), None)
    if cnt:
        tab = tab.sort_values(cnt, ascending=False, kind="mergesort")
    tab = tab.drop_duplicates("cell_id").set_index("cell_id")
    tab.index = tab.index.astype(str)
    return tab


def read_10x_h5(path):
    """10x ``filtered_feature_bc_matrix.h5`` -> AnnData with a ``feature_type`` column."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, "r") as f:
        g = f["matrix"]
        x = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]).T.tocsr()
        var = pd.DataFrame({k: [v.decode() for v in g["features"][k][:]] for k in ("id", "name", "feature_type")})
        obs = pd.DataFrame(index=[b.decode() for b in g["barcodes"][:]])
    var.index = pd.Index(var["name"])
    a = ad.AnnData(X=x.astype(np.float32), obs=obs, var=var)
    a.var_names_make_unique()
    return a


def _contigs_heavy(path) -> pd.DataFrame:
    """Productive heavy chains of a 10x ``filtered_contig_annotations.csv``, one per cell.

    Handles both the current column names (``v_call``) and the older ones
    (``v_gene``), which differ between Cell Ranger versions.
    """
    t = pd.read_csv(path)
    t = t.rename(columns={"v_gene": "v_call", "j_gene": "j_call", "d_gene": "d_call", "c_gene": "c_call",
                          "cdr3_nt": "junction"})
    t = t[t["productive"].astype(str).str.upper().isin(("TRUE", "T", "1")) & t["chain"].eq("IGH")]
    count = next((c for c in ("umis", "reads") if c in t.columns), None)
    if count:
        t = t.sort_values(count, ascending=False, kind="mergesort")
    return t.drop_duplicates("barcode").set_index("barcode")



# =========================================================================== human datasets


def load_ln_vaccine():
    """Kim et al. 2022 Nature: SARS-CoV-2 mRNA vaccine, LN FNA + blood, 8 donors.

    Labels: spike binding of the clone (authors' ``s_pos_clone``, from flow
    probe sorting and ELISA of recombinant mAbs), ELISA of expressed mAbs,
    author SHM frequency, isotype, tissue, time point.
    """
    import scanpy as sc

    d = DATA / "gse195673_ln_vaccine"
    adata = sc.read_h5ad(d / "gex_b_cells.h5ad")
    adata.X = adata.layers["raw_counts"].copy()
    for layer in list(adata.layers):
        del adata.layers[layer]
    adata.obsm.clear()
    adata.obsp.clear()
    adata.uns.clear()
    adata.obs_names = adata.obs["cell_id"].astype(str).values
    adata.var_names = adata.var["gene_name"].astype(str).values if "gene_name" in adata.var else adata.var_names
    adata.var_names_make_unique()
    obs = adata.obs
    obs["donor"] = obs["donor"].astype(str)
    obs["sample"] = obs["donor"] + "_" + obs["sample"].astype(str)
    obs["state"] = obs["anno_leiden_0.18"].astype(str)
    meta = pd.read_csv(d / "bcr_meta.tsv", sep="\t")
    meta["key"] = meta["donor"].astype(str) + "_" + meta["sample"].astype(str)
    meta = meta.drop_duplicates("key").set_index("key")
    obs["timepoint"] = obs["sample"].map(meta["timepoint"]).astype(str)
    obs["tissue"] = obs["tissue"].astype(str)
    obs["sort"] = obs["sample"].map(meta["sorting"]).astype(str)
    adata.obs = obs[["donor", "sample", "tissue", "timepoint", "sort", "state"]].copy()

    heavy = pd.read_csv(d / "bcr_heavy.tsv.gz", sep="\t", low_memory=False)
    heavy = heavy[heavy["cell_id"].notna()].drop_duplicates("cell_id").set_index("cell_id")
    bcr = pd.DataFrame({
        "v_call": heavy["v_call"], "j_call": heavy["j_call"], "junction": heavy["junction"],
        "c_call": heavy["isotype"],
        "author_clone_id": heavy["clone_id"].astype(str),
        "spike_binding": heavy["s_pos_clone"].map(lambda v: "S+" if str(v).upper() == "TRUE" else "S-"),
        "elisa": heavy["elisa"].map(lambda v: {"TRUE": "ELISA+", "FALSE": "ELISA-"}.get(str(v).upper())),
        "mutation_frequency": pd.to_numeric(heavy["nuc_RS_freq_19_312"], errors="coerce"),
    })
    bcr.index = bcr.index.astype(str)
    return adata, bcr[bcr.index.isin(adata.obs_names)]


def load_flu():
    """Wang et al. 2023: seasonal influenza vaccine, PBMC B cells, 6 donors x (d0, d7)."""
    import anndata as ad

    d = DATA / "gse175522_flu"
    young = {"120648", "120667", "141393"}
    ads, bcrs = [], []
    for tar in sorted(d.glob("*_cellranger.tar.gz")):
        _, donor, tp = tar.name.split("_")[:3]
        sample = f"{donor}_d{tp}"
        mtx = _extracted_matrix_dir(tar, d / "extracted" / f"{donor}_{tp}")
        a = read_mtx_dir(mtx, prefix="")
        a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
        a.obs_names = [f"{sample}|{b}" for b in a.obs_names]
        a.obs["donor"], a.obs["sample"] = donor, sample
        a.obs["timepoint"] = f"d{tp}"
        a.obs["age_group"] = "young" if donor in young else "older"
        ads.append(a)
        airr = next(d.glob(f"GSM*_{donor}_{tp}_clone_pass_fil_airr.txt.gz"))
        h = _airr_heavy(airr)
        h.index = [f"{sample}|{b}" for b in h.index]
        h["author_clone_id"] = donor + "|" + h["clone_id"].astype(str)
        bcrs.append(h)
    adata = ad.concat(ads, join="outer", fill_value=0)
    bcr = pd.concat(bcrs)
    keep = ["v_call", "j_call", "junction", "c_call", "author_clone_id", "mutation_frequency"]
    return adata, bcr.loc[bcr.index.isin(adata.obs_names), keep]


def _extracted_matrix_dir(tar: Path, outdir: Path) -> Path:
    """Directory holding matrix.mtx(.gz) inside an (already) extracted tarball."""
    if not outdir.exists() or not any(outdir.rglob("matrix.mtx*")):
        outdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(outdir, filter="data")
    hit = next(outdir.rglob("matrix.mtx*"))
    return hit.parent


def load_tonsil():
    """King et al. 2021 Sci Immunol: paediatric tonsil, 6 donors, 'Total' libraries."""
    import anndata as ad
    import threadfin as tf

    d = DATA / "king2021_tonsil"
    ads, bcrs = [], []
    for tar in sorted(d.glob("BCP*_Total_5GEX.tar.gz")):
        donor = tar.name.split("_")[0]
        a = read_mtx_dir(_extracted_matrix_dir(tar, d / "extracted" / donor), prefix="")
        a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
        a.obs_names = [f"{donor}|{b.split('-')[0]}" for b in a.obs_names]
        a.obs["donor"], a.obs["sample"] = donor, donor
        ads.append(a)
        csv = next((d / "extracted" / f"{donor}_vdj").rglob("filtered_contig_annotations.csv*"))
        b = tf.read_10x_vdj(str(csv))
        b.index = [f"{donor}|{x.split('-')[0]}" for x in b.index]
        bcrs.append(b)
    adata = ad.concat(ads, join="outer", fill_value=0)
    meta = pd.read_csv(d / "CellTypeMetaData.txt", sep="\t", index_col=0)
    parsed = meta.index.astype(str).str.extract(r"^BCP(\d+)_[A-Za-z0-9]+_([ACGT]{16})$")
    ok = parsed[0].notna().to_numpy()
    meta = meta[ok]
    meta.index = [f"BCP{int(n):03d}|{bc}" for n, bc in parsed[ok].to_numpy()]
    meta = meta[~meta.index.duplicated()]
    adata.obs["lineage"] = meta["Lineage"].reindex(adata.obs_names).astype(str).values
    adata.obs["state"] = meta["Subset"].reindex(adata.obs_names).astype(str).values
    adata = adata[adata.obs["lineage"].str.contains("B")].copy()  # B-lineage cells only
    bcr = pd.concat(bcrs)
    return adata, bcr.loc[bcr.index.isin(adata.obs_names), ["v_call", "j_call", "cdr3_nt", "c_call"]]


_EBV = {  # sample tag -> (GEX GSM, BCR GSM, time point, condition)
    "d0": ("GSM9473008", "GSM9473009", "d0", "uninfected"),
    "d4": ("GSM9473012", "GSM9473013", "d4", "uninfected"),
    "d7": ("GSM9473016", "GSM9473017", "d7", "uninfected"),
    "d14-gfp-neg-b": ("GSM9473020", "GSM9473021", "d14", "GFP-"),
    "d14_gfp-pos-b": ("GSM9473023", "GSM9473024", "d14", "GFP+"),
    "d21-gfp-neg-b": ("GSM9473027", "GSM9473028", "d21", "GFP-"),
    "d21_gfp-pos-b": ("GSM9473030", "GSM9473031", "d21", "GFP+"),
}


def load_ebv():
    """Mitul et al. 2026 PNAS: EBV infection of tonsil organoids, d0-d21, GFP+/- sorted.

    Every sample pools the same four donors, so ``donor`` is the pool and
    clones are defined across samples (infected and uninfected cells of one
    clone are linked).
    """
    import anndata as ad
    import threadfin as tf

    d = DATA / "gse317492_ebv"
    ads, bcrs = [], []
    for tag, (gsm_g, gsm_b, tp, cond) in _EBV.items():
        a = read_mtx_dir(d, prefix=f"{gsm_g}_{tag}_gex_")
        a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
        a.obs_names = [f"{tag}|{b}" for b in a.obs_names]
        a.obs["donor"], a.obs["sample"] = "pool", tag
        a.obs["timepoint"], a.obs["condition"] = tp, cond
        ads.append(a)
        b = tf.read_10x_vdj(str(d / f"{gsm_b}_{tag}_vdjB_filtered_contig_annotations.csv.gz"))
        b.index = [f"{tag}|{x}" for x in b.index]
        bcrs.append(b)
    adata = ad.concat(ads, join="outer", fill_value=0)
    gfp = adata.obs["condition"].map({"GFP+": "GFP+", "GFP-": "GFP-"})
    adata.obs["gfp"] = gfp.astype("object")
    bcr = pd.concat(bcrs)
    return adata, bcr.loc[bcr.index.isin(adata.obs_names), ["v_call", "j_call", "cdr3_nt", "c_call"]]


def load_stephenson():
    """Stephenson et al. 2021 Nat Med: COVID-19 PBMC, 5k BCR+ B-lineage cells (scirpy subset)."""
    import awkward as ak
    import mudata as md

    m = md.read_h5mu(DATA / "stephenson2021_5k.h5mu")
    gex = m["gex"]
    adata = gex[:, gex.var["feature_types"] == "Gene Expression"].copy()  # drop 192 CITE-seq antibody tags
    adata.X = adata.layers["raw"].copy()
    del adata.layers["raw"]
    adata.obsm.clear()
    obs = adata.obs
    obs["donor"] = obs["patient_id"].astype(str)
    obs["sample"] = obs["sample_id"].astype(str)
    obs["state"] = obs["full_clustering"].astype(str)
    obs["severity"] = obs["Status_on_day_collection_summary"].astype(str)
    obs["site"] = obs["Site"].astype(str)
    adata.obs = obs[["donor", "sample", "state", "severity", "site"]].copy()
    a = m["airr"].obsm["airr"]
    prod = ak.fill_none(a["productive"], False)
    first = ak.firsts(a[(a["locus"] == "IGH") & prod])
    bcr = pd.DataFrame({k: ak.to_list(first[k]) for k in ("v_call", "j_call", "junction", "c_call", "mu_freq")},
                       index=m["airr"].obs_names.astype(str)).rename(columns={"mu_freq": "mutation_frequency"})
    bcr["mutation_frequency"] = pd.to_numeric(bcr["mutation_frequency"], errors="coerce")
    bcr = bcr.dropna(subset=["v_call", "junction"])
    return adata, bcr[bcr.index.isin(adata.obs_names)]


# =========================================================================== mouse GC (Merkenschlager 2025)

_GSE287123 = DATA / "gse287123_np" / "extracted"


def _mouse_library(gsm_gex: str, name: str, gsm_vdj: str):
    a = read_mtx_dir(_GSE287123, prefix=f"{gsm_gex}_{name}_GEX_HTO_")
    hto = a[:, a.var["gene_ids"].str.startswith("HTO_")]
    counts = pd.DataFrame(hto.X.toarray(), index=a.obs_names, columns=hto.var_names)
    a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
    a.obs["hashtag"] = demux_hashtags(counts).values
    a.obs["library"] = name
    airr = _airr_heavy(_GSE287123 / f"{gsm_vdj}_{name}_airr_rearrangement.tsv.gz")
    return a, airr


def load_mouse_np():
    """NP-OVA immunised H2B-mCherry mice (7 mice, hashtagged), GC B cells sorted
    mCherry-high (few divisions) or mCherry-low (many divisions).

    Clone-level ground truth: fraction of a clone's cells in the mCherry-low
    (highly divided) gate, VH186.2 W33L affinity mutation, mutation frequency.
    """
    import anndata as ad

    ads, bcrs = [], []
    for gex, name, vdj, gate in (("GSM8739131", "Princeton_GC_B_mCherry_HI", "GSM8739133", "mCherry-high"),
                                 ("GSM8739134", "Princeton_GC_B_mCherry_Lo", "GSM8739136", "mCherry-low")):
        a, airr = _mouse_library(gex, name, vdj)
        a.obs["division_gate"] = gate
        prefix = name.split("_")[-1]
        a.obs_names = [f"{prefix}|{b}" for b in a.obs_names]
        airr.index = [f"{prefix}|{b}" for b in airr.index]
        airr["w33"] = w33l_status(airr)
        ads.append(a)
        bcrs.append(airr)
    adata = ad.concat(ads, join="outer", fill_value=0)
    ht = adata.obs["hashtag"].astype(str)
    keep = ~ht.isin(["negative", "doublet"])
    adata = adata[keep.to_numpy()].copy()
    adata.obs["donor"] = adata.obs["hashtag"].str.split("_").str[0].values  # M1..M7
    adata.obs["sample"] = adata.obs["donor"] + "_" + adata.obs["division_gate"]
    bcr = pd.concat(bcrs)
    bcr = bcr.loc[bcr.index.isin(adata.obs_names), ["v_call", "j_call", "junction", "c_call",
                                                     "mutation_frequency", "w33", "sequence_alignment",
                                                     "germline_alignment", "v_sequence_start",
                                                     "v_sequence_end"]]
    bcr["w33l"] = bcr["w33"].map(lambda r: "W33L" if r == "L" else ("germline W33" if r == "W" else None))
    return adata, bcr


def load_mouse_rbd():
    """RBD protein- or mRNA-vaccinated H2B-mCherry mice, GC B cells (tubes A-D).

    Hashtags 1-5: five protein-vaccinated mice sorted RBD-bait+/- x mCherry
    high/low (full factorial). Hashtags 6-10: five mRNA-vaccinated mice sorted
    light zone / dark zone x mCherry high/low.
    """
    import anndata as ad

    tubes = (("GSM8739143", "Rebutal_4-5_Tube_A", "GSM8739145"), ("GSM8739146", "Rebutal_4-5_Tube_B", "GSM8739148"),
             ("GSM8739149", "Rebutal_4-5_Tube_C", "GSM8739151"), ("GSM8739152", "Rebutal_4-5_Tube_D", "GSM8739154"))
    ads, bcrs = [], []
    for gex, name, vdj in tubes:
        a, airr = _mouse_library(gex, name, vdj)
        tube = name.split("_")[-1]
        a.obs_names = [f"{tube}|{b}" for b in a.obs_names]
        airr.index = [f"{tube}|{b}" for b in airr.index]
        ads.append(a)
        bcrs.append(airr)
    adata = ad.concat(ads, join="outer", fill_value=0)
    ht = adata.obs["hashtag"].astype(str)
    adata = adata[~ht.isin(["negative", "doublet"]).to_numpy()].copy()
    tag = adata.obs["hashtag"].astype(str)  # e.g. M1_RBD_neg_mCherry_low / M6_mRNA_LZ_mCherry_hi
    adata.obs["donor"] = tag.str.split("_").str[0].values
    adata.obs["arm"] = np.where(tag.str.contains("_RBD_"), "RBD protein", "mRNA")
    adata.obs["rbd_bait"] = tag.map(lambda s: "RBD+" if "_RBD_pos" in s else ("RBD-" if "_RBD_neg" in s else None)).values
    adata.obs["zone_gate"] = tag.map(lambda s: "LZ" if "_LZ_" in s else ("DZ" if "_DZ_" in s else None)).values
    adata.obs["division_gate"] = np.where(tag.str.contains("mCherry_low"), "mCherry-low", "mCherry-high")
    adata.obs["sample"] = (adata.obs["donor"] + "_" + adata.obs["rbd_bait"].fillna(adata.obs["zone_gate"]).astype(str)
                           + "_" + adata.obs["division_gate"]).values
    bcr = pd.concat(bcrs)
    keep = ["v_call", "j_call", "junction", "c_call", "mutation_frequency", "sequence_alignment",
            "germline_alignment", "v_sequence_start", "v_sequence_end"]
    return adata, bcr.loc[bcr.index.isin(adata.obs_names), keep]


# =========================================================================== mouse GC + plasma cells (ElTanbouly 2023)

_GSE246382 = DATA / "gse246382_np_pc"


def _imgt_germlines(folder: Path) -> dict:
    """Ungapped IMGT germline sequences, keyed by allele name (e.g. ``IGHV1-72*01``)."""
    out = {}
    for f in sorted(folder.glob("*.fasta")):
        name, seq = None, []
        for line in f.read_text().splitlines() + [">"]:
            if line.startswith(">"):
                if name:
                    out[name] = "".join(seq).replace(".", "").upper()
                name, seq = (line.split("|")[1] if "|" in line else None), []
            else:
                seq.append(line.strip())
    return out


def _parse_trust4_annot(path: Path) -> pd.DataFrame:
    """TRUST4 ``annot.fa`` -> one row per assembled contig with gene calls and alignment coordinates."""
    import gzip
    import re

    rows, header, seq = [], None, []
    seg = re.compile(r"(IG[HKL][VDJ][^(]*)\((\d+)\):\((\d+)-(\d+)\):\((\d+)-(\d+)\):([\d.]+)")
    with gzip.open(path, "rt") as fh:
        for line in list(fh) + [">"]:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header:
                    rec = {"cid": header.split()[0], "contig": "".join(seq)}
                    for m in seg.finditer(header):
                        kind = m.group(1)[3]  # V, D or J
                        if kind + "_gene" in rec:
                            continue  # keep the best (first) call
                        rec.update({kind + "_gene": m.group(1), kind + "_qs": int(m.group(3)), kind + "_qe": int(m.group(4)),
                                    kind + "_gs": int(m.group(5)), kind + "_ge": int(m.group(6)),
                                    kind + "_identity": float(m.group(7))})
                    cdr3 = re.search(r"CDR3\((\d+)-(\d+)\):[\d.]+=([ACGTN]+)", header)
                    if cdr3:
                        rec.update(cdr3_start=int(cdr3.group(1)), cdr3_end=int(cdr3.group(2)), cdr3_nt=cdr3.group(3))
                    rows.append(rec)
                header, seq = line[1:], []
            else:
                seq.append(line)
    return pd.DataFrame(rows)


def _v_alignment(rec, germ: dict):
    """Gapless query/germline V alignment from TRUST4 coordinates (None if lengths differ)."""
    g = germ.get(rec["V_gene"])
    if g is None or pd.isna(rec.get("V_qs")):
        return None
    q = rec["contig"][int(rec["V_qs"]):int(rec["V_qe"]) + 1]
    gg = g[int(rec["V_gs"]):int(rec["V_ge"]) + 1]
    return (q, gg, int(rec["V_gs"])) if len(q) == len(gg) else None


def gc_np_pc_receptors(folder: Path = None) -> pd.DataFrame:
    """Per-cell heavy and light chains of GSE246382 with V-region mutations and germline-anchored sequences.

    Columns: ``v_call``, ``j_call``, ``junction`` (CDR3 nt), ``c_call``, ``mutation_frequency`` (heavy V),
    ``n_mutations``, ``w33l`` (IGHV1-72 Kabat 33: W33L high-affinity / germline W33), light-chain calls and
    ``heavy_v_seq`` / ``heavy_v_germline`` / ``heavy_v_start`` for lineage reconstruction.
    """
    folder = folder or _GSE246382
    germ = _imgt_germlines(DATA / "imgt_mouse")
    rep = pd.read_csv(folder / "GSE246382_trust_report.tsv.gz", sep="\t").rename(columns={"#count": "count"})
    rep["cell"] = rep["cid"].str.replace(r"_\d+_S\d+_R1_001_assemble\d+$", "", regex=True)
    rep["chain"] = rep["V"].astype(str).str[:3]
    rep = rep[~rep["CDR3aa"].astype(str).str.contains(r"out_of_frame|\*|_|\?", regex=True)]
    ann = _parse_trust4_annot(folder / "GSE246382_trust_annot.fa.gz").set_index("cid")
    out = {}
    for chain, cols in (("IGH", "heavy"), ("light", "light")):
        sub = rep[rep["chain"].eq("IGH")] if chain == "IGH" else rep[rep["chain"].isin(["IGK", "IGL"])]
        sub = sub.sort_values("count", ascending=False).drop_duplicates("cell").set_index("cell")
        out[cols] = sub
    h, l = out["heavy"], out["light"]
    rows = []
    for cell, r in h.iterrows():
        rec = ann.loc[r["cid"]] if r["cid"] in ann.index else None
        row = {"cell": cell, "v_call": r["V"], "j_call": r["J"], "junction": r["CDR3nt"], "c_call": r["C"],
               "cdr3_aa": r["CDR3aa"]}
        aln = _v_alignment(rec, germ) if rec is not None else None
        if aln:
            q, g, start = aln
            mism = sum(a != b for a, b in zip(q, g) if a != "N" and b != "N")
            row.update(n_mutations=mism, mutation_frequency=mism / max(len(q), 1), heavy_v_seq=q,
                       heavy_v_germline=g, heavy_v_start=start)
            if r["V"].split("*")[0] == "IGHV1-72":
                off = (3 - start % 3) % 3  # first full codon of the germline V
                g_aa, q_aa = _translate(g[off:]), _translate(q[off:])
                pos = g_aa.find("SYWMH")
                if pos >= 0 and pos + 2 < len(q_aa):
                    row["w33l"] = "W33L" if q_aa[pos + 2] == "L" else ("germline W33" if q_aa[pos + 2] == "W" else "other")
        if cell in l.index:
            lr = l.loc[cell]
            row.update(light_v_call=lr["V"], light_j_call=lr["J"], light_junction=lr["CDR3nt"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell")


def load_gc_np_pc():
    """ElTanbouly et al. 2023 J Exp Med (GSE246382): NP-OVA day 14, popliteal lymph node.

    Smart-seq2 of FACS-sorted germinal-centre B cells (dark zone, light zone, and in c-Myc-GFP mice
    Myc-GFP+ light-zone cells, i.e. recently positively selected) and plasma cells; BCRs assembled with
    TRUST4. Each cell carries its sorted compartment, so a clone's fate split (plasma cell vs Myc+
    light zone vs dark zone) is measured, not inferred. 11 mice (7 c-Myc-GFP, 4 C57BL/6).
    """
    import scipy.io

    folder = _GSE246382
    x = scipy.io.mmread(folder / "GSE246382_umiDedup-Exact.mtx.gz").tocsr().T.tocsr()  # cells x genes
    feats = pd.read_csv(folder / "GSE246382_features.tsv.gz", sep="\t", header=None)
    cells = pd.read_csv(folder / "GSE246382_barcodes.tsv.gz", header=None)[0].astype(str)
    import anndata as ad

    adata = ad.AnnData(X=x.astype(np.float32), obs=pd.DataFrame(index=cells.values),
                       var=pd.DataFrame({"gene_ids": feats[0].values}, index=feats[1].values))
    adata.var_names_make_unique()
    meta = pd.read_csv(folder / "cell_metadata.csv", index_col=0)
    adata.obs = adata.obs.join(meta[["compartment", "mouse", "genotype"]])
    comp = adata.obs["compartment"].astype(str)
    adata.obs["donor"] = adata.obs["mouse"].astype(str)
    adata.obs["sample"] = adata.obs["mouse"].astype(str)  # one Smart-seq2 plate set per mouse
    adata.obs["fate"] = np.select([comp.eq("Plasma"), comp.eq("LZ myc+"), comp.str.startswith("DZ")],
                                  ["plasma cell", "Myc+ light zone", "dark zone"], "light zone")
    adata.obs["plasma_cell"] = np.where(comp.eq("Plasma"), "PC", "GC")
    keep = (np.asarray((adata.X > 0).sum(axis=1)).ravel() >= 500)
    adata = adata[keep].copy()
    bcr = gc_np_pc_receptors(folder)
    return adata, bcr.loc[bcr.index.isin(adata.obs_names)]


# =========================================================================== influenza infection (lung and node)

_GSE317692 = DATA / "gse317692_flu_lung"


def load_flu_lung():
    """GSE317692: influenza A (PR8) infection of mice; lung and mediastinal lymph node B cells, day 20.

    Each of 15 mice contributes both tissues (hashtagged, one 10x pool per
    tissue), so a clone can be followed from the draining lymph node into the
    infected lung. Cells were stained with haemagglutinin tetramers of the
    infecting strain (PR8) and of a heterologous strain (Cal09), giving an
    antigen-specificity label, and seven of the mice lack B-cell alpha-v
    integrin, a regulator of germinal-centre dynamics - a genetic perturbation
    to test clone-level measurements against.
    """
    import anndata as ad

    folder = _GSE317692
    meta = pd.read_csv(folder / "sample_annotation.csv.gz", encoding="utf-8-sig")
    meta["mouseId"] = meta["mouseId"].astype(str)
    groups = meta.drop_duplicates("mouseId").set_index("mouseId")["studyGroup"]
    tissue_of_pool = meta.drop_duplicates("pool").set_index("pool")["tissue"]

    ads, bcrs = [], []
    for pool, h5 in ((1, "pool616-1_1_matrix.h5"), (2, "pool616-1_2_matrix.h5")):
        a = read_10x_h5(folder / h5)
        tag = f"P{pool}"
        a.obs_names = [f"{tag}|{b}" for b in a.obs_names]
        prot = a[:, a.var["feature_type"] != "Gene Expression"].copy()
        counts = pd.DataFrame(np.asarray(prot.X.todense()), index=prot.obs_names, columns=prot.var_names)
        tetramers = [c for c in counts.columns if "HA" in c.upper()]
        hto = counts.drop(columns=tetramers)
        a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
        a.obs["mouse"] = demux_hashtags(hto).values
        a.obs["tissue"] = tissue_of_pool[pool]
        for t in tetramers:                       # CLR within the pool, then a 2-means cut per tetramer
            logc = np.log1p(counts[t].to_numpy())
            a.obs[f"tetramer_{t.replace(' ', '')}"] = logc - logc.mean()
        a.obs["ha_binding"] = _tetramer_call(counts[tetramers])
        ads.append(a)
        airr = next(folder.glob(f"*pool616-1_{pool}_airr_rearrangement.tsv.gz"))
        b = _airr_heavy(airr)
        b.index = [f"{tag}|{i}" for i in b.index]
        bcrs.append(b)

    adata = ad.concat(ads, join="outer", fill_value=0)
    keep = ~adata.obs["mouse"].astype(str).isin(["negative", "doublet"])
    adata = adata[keep.to_numpy()].copy()
    adata.obs["donor"] = adata.obs["mouse"].astype(str)
    adata.obs["genotype"] = adata.obs["donor"].map(groups).astype(str)        # Control or cKO (B-cell alpha-v)
    adata.obs["sample"] = adata.obs["donor"] + "_" + adata.obs["tissue"].astype(str)
    bcr = pd.concat(bcrs)
    keep_cols = ["v_call", "j_call", "junction", "c_call", "mutation_frequency", "sequence_alignment",
                 "germline_alignment", "v_sequence_start", "v_sequence_end"]
    bcr = bcr.loc[bcr.index.isin(adata.obs_names), [c for c in keep_cols if c in bcr.columns]]
    return adata, bcr


def _tetramer_call(counts: pd.DataFrame, min_total: int = 3) -> pd.Series:
    """Antigen-probe call per cell: which haemagglutinin tetramer(s) the cell bound."""
    from sklearn.cluster import KMeans

    logc = np.log1p(counts.to_numpy(dtype=float))
    clr = logc - logc.mean(axis=0, keepdims=True)
    pos = np.zeros_like(clr, dtype=bool)
    for j in range(clr.shape[1]):
        km = KMeans(2, n_init=10, random_state=0).fit(clr[:, [j]])
        pos[:, j] = km.labels_ == int(np.argmax(km.cluster_centers_.ravel()))
    names = np.asarray([c.replace(" ", "") for c in counts.columns])
    n = pos.sum(axis=1)
    out = np.where(n == 0, "non-binding", "")
    out = np.where(n == 1, names[pos.argmax(axis=1)], out)
    out = np.where(n > 1, "both strains", out)
    out = np.where(counts.to_numpy().sum(axis=1) < min_total, "non-binding", out)
    return pd.Series(out, index=counts.index)



# =========================================================================== outside the germinal centre

_GSE253857 = DATA / "gse253857_bmpc"


def load_bone_marrow_pc():
    """GSE253857: human bone-marrow plasma cells and memory B cells, with blood counterparts.

    Antibody-secreting cells that reach the bone marrow are the source of
    long-lived serum antibody, and they get there long after any germinal
    centre has closed. Cells were sorted by compartment (plasma cells, memory
    B cells) and, in some samples, by what their antibody binds: the spike
    protein of SARS-CoV-2, reflecting a recent vaccination, or tetanus toxoid,
    reflecting an immunisation decades earlier. The contrast asks whether
    clonal structure is a germinal-centre phenomenon or outlives it.

    The deposited contig files carry no germline alignment, so mutation load
    is not available for this dataset.
    """
    import anndata as ad

    folder = _GSE253857
    ads, bcrs = [], []
    for h5 in sorted(folder.glob("*.filtered_feature_bc_matrix.h5")):
        name = h5.name.replace(".filtered_feature_bc_matrix.h5", "")
        contig = next((c for c in folder.glob("*_BCR.filtered_contig_annotations.csv.gz")
                       if c.name.replace("_BCR.filtered_contig_annotations.csv.gz", "")
                       == name.replace("-", "_")), None)
        if contig is None:                      # no receptors for this run
            continue
        a = read_10x_h5(h5)
        a = a[:, a.var["feature_type"] == "Gene Expression"].copy()
        a.obs_names = [f"{name}|{b}" for b in a.obs_names]
        tissue = "blood" if name.lower().startswith("blood") or "blood" in name.lower() else "bone marrow"
        sort = _bmpc_sort(name)
        a.obs["sample"] = name
        a.obs["tissue"] = tissue
        a.obs["sorted_as"] = sort
        a.obs["antigen"] = ("SARS-CoV-2 spike" if "spike" in name.lower() else
                            "tetanus toxoid" if "tetanus" in name.lower() else "not sorted by antigen")
        a.obs["donor"] = _bmpc_donor(name)
        ads.append(a)
        b = _contigs_heavy(contig)
        b.index = [f"{name}|{i}" for i in b.index]
        bcrs.append(b)
    adata = ad.concat(ads, join="outer", fill_value=0)
    bcr = pd.concat(bcrs)
    bcr = bcr.loc[bcr.index.isin(adata.obs_names), ["v_call", "j_call", "junction", "c_call"]]
    return adata, bcr


def _bmpc_sort(name: str) -> str:
    low = name.lower()
    if "pcs-bmem" in low or "pc_bmem" in low or "pcs_bmem" in low:
        return "plasma and memory"
    if "pcs" in low or low.endswith("_pc"):
        return "plasma cells"
    if "bmem" in low or "bsm" in low:
        return "memory B cells"
    return "antigen-sorted"


def _bmpc_donor(name: str) -> str:
    """Donor id from the sample name; pooled runs are kept as their own group."""
    import re

    if "pool" in name.lower():
        return "pooled donors"
    m = re.findall(r"(\d{3,4}|\d)(?=$|[_-])", name)
    return f"donor {m[-1]}" if m else "unknown"



# =========================================================================== malaria time course

_GSE286215 = DATA / "gse286215_malaria"


def load_malaria(experiment: str = "experiment1"):
    """GSE286215: splenic B cells through a Plasmodium infection in mice.

    A time course of a live infection, with five hashtagged mice at each
    sampling day and the authors' own cell annotation (naive follicular,
    bystander, activated, germinal centre, plasmablast, memory). Mice are
    sacrificed at each day, so the same clone cannot be followed through time;
    what the series shows is how clonal structure develops as the infection
    runs. Experiment 2 adds later days and an antimalarial-treatment arm.
    """
    import anndata as ad
    import scipy.io
    import scipy.sparse as sp

    folder = _GSE286215
    x = scipy.io.mmread(folder / f"{experiment}_counts.mtx").tocsr().T.tocsr()      # cells x genes
    genes = pd.read_csv(folder / f"{experiment}_genes.csv").iloc[:, 0].astype(str)
    cells = pd.read_csv(folder / f"{experiment}_barcodes.csv").iloc[:, 0].astype(str)
    meta = pd.read_csv(folder / f"{experiment}_metadata.csv", index_col=0, low_memory=False)
    adata = ad.AnnData(X=x.astype(np.float32), obs=pd.DataFrame(index=cells.values),
                       var=pd.DataFrame(index=genes.values))
    adata.var_names_make_unique()
    adata.obs = adata.obs.join(meta)
    adata.obs["timepoint"] = adata.obs["orig.ident"].astype(str)
    adata.obs["mouse"] = adata.obs["hash.ID"].astype(str).str.replace("-TotalC", "", regex=False)
    # mice are sacrificed at each sampling day, so a mouse is a (day, hashtag) pair
    adata.obs["donor"] = adata.obs["timepoint"] + "_" + adata.obs["mouse"]
    adata.obs["sample"] = adata.obs["donor"]
    for col in ("annotation1", "annotation2", "clusters_compare"):
        if col in adata.obs:
            adata.obs["cell_state"] = adata.obs[col].astype(str)
            break
    if "treatment" in meta.columns:
        adata.obs["treatment"] = adata.obs["treatment"].astype(str)

    n = "1" if experiment.endswith("1") else "2"
    bcr = pd.read_csv(folder / f"exp{n}_bcr.tsv.gz", sep="\t", low_memory=False)
    bcr = bcr.loc[:, ~bcr.columns.duplicated()]
    prod = bcr["productive"].astype(str).str.upper().isin(("T", "TRUE", "1"))
    bcr = bcr[prod & bcr["v_call"].astype(str).str.upper().str.startswith("IGH")].copy()
    bcr["cell"] = bcr["cell_id"].astype(str) + "-1"            # the matrix keeps the 10x suffix
    count = next((c for c in ("umi_count", "duplicate_count", "consensus_count") if c in bcr.columns), None)
    if count:
        bcr = bcr.sort_values(count, ascending=False, kind="mergesort")
    bcr = bcr.drop_duplicates("cell").set_index("cell")
    import threadfin as tf

    if {"sequence_alignment", "germline_alignment"} <= set(bcr.columns):
        bcr["mutation_frequency"] = tf.tl.mutation_frequency(bcr)
    keep = [c for c in ("v_call", "j_call", "junction", "c_call", "mutation_frequency", "sequence_alignment",
                        "germline_alignment", "v_sequence_start", "v_sequence_end") if c in bcr.columns]
    return adata, bcr.loc[bcr.index.isin(adata.obs_names), keep]



LOADERS = {
    "ln_vaccine": load_ln_vaccine,
    "flu": load_flu,
    "tonsil": load_tonsil,
    "ebv": load_ebv,
    "stephenson": load_stephenson,
    "mouse_np": load_mouse_np,
    "mouse_rbd": load_mouse_rbd,
    "gc_np_pc": load_gc_np_pc,
    "flu_lung": load_flu_lung,
    "bone_marrow_pc": load_bone_marrow_pc,
    "malaria": load_malaria,
}
