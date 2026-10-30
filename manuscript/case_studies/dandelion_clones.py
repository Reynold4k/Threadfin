#!/usr/bin/env python
"""Re-define every dataset's clones with Dandelion, so the light chain counts.

Threadfin's own clone caller groups cells by the heavy chain; the light chain is at best an optional
post-hoc split and in practice the loaders discard it (`_airr_heavy` keeps only IGH). A B-cell clone is
one rearrangement of *both* chains, so this script hands clone calling to Dandelion — a maintained,
cited implementation of the paired definition — and writes one clone id per cell.

Run it in the Dandelion environment (`python_envs/dandelion`, Python 3.11, sc-dandelion 0.5.7, which pins
matplotlib<3.9 and so cannot share the main environment). The resulting tables are read back by the case
studies, so nothing downstream needs Dandelion installed.

Cells are matched to the cached embeddings by the same barcode rule each loader uses, and the donor is
taken from the embedding rather than re-derived, so a clone can never span two animals or participants.

Usage:
    python dandelion_clones.py [dataset ...] [--heavy-only] [--identity 0.85]
Output: case_studies/results/dandelion_clones/<dataset>.csv  (cell, clone_id)  + _summary.csv
"""
from __future__ import annotations

import os
import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # the folder that holds the repository clone
DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parent / "data"))
CACHE = ROOT / "internal_validation" / "algorithm_revision" / "embedding_cache"
OUT = ROOT / "Threadfin" / "manuscript" / "case_studies" / "results" / "dandelion_clones"

# GSE287123 libraries: (VDJ accession, library name, obs-name prefix) — the prefix rule is the loader's
GSE287123_NP = [("GSM8739133", "Princeton_GC_B_mCherry_HI", "HI"),
                ("GSM8739136", "Princeton_GC_B_mCherry_Lo", "Lo")]


def _gse287123_rbd_libraries():
    """RBD arms: every airr file whose library is not one of the NP ones."""
    out = []
    for path in sorted(glob.glob(str(DATA / "gse287123_np/extracted/*_airr_rearrangement.tsv.gz"))):
        stem = Path(path).name.replace("_airr_rearrangement.tsv.gz", "")
        gsm, name = stem.split("_", 1)
        if name in {n for _, n, _ in GSE287123_NP}:
            continue
        out.append((gsm, name, name.split("_")[-1]))
    return out



TENX_TO_AIRR = {"barcode": "cell_id", "chain": "locus", "v_gene": "v_call", "d_gene": "d_call",
                "j_gene": "j_call", "c_gene": "c_call", "cdr3_nt": "junction", "cdr3": "junction_aa",
                "contig_id": "sequence_id", "umis": "umi_count", "reads": "consensus_count"}


def _tenx(path, cell_fn):
    """A 10x `filtered_contig_annotations.csv` as contig-level AIRR."""
    t = pd.read_csv(path, low_memory=False)
    t = t.rename(columns={k: v for k, v in TENX_TO_AIRR.items() if k in t.columns})
    if "is_cell" in t.columns:
        t = t[t["is_cell"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    if "high_confidence" in t.columns:
        t = t[t["high_confidence"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    t["cell"] = [cell_fn(b) for b in t["cell_id"].astype(str)]
    return t


def _trust4_report(path, cells_path=None):
    """GSE246382's TRUST4 report as a minimal AIRR table.

    TRUST4 reports CDR3 and the V/D/J/C calls, not a full alignment, which is all `find_clones` needs:
    it groups on V gene, J gene, junction length and junction identity.
    """
    t = pd.read_csv(path, sep="\t", low_memory=False)
    t = t.rename(columns={"CDR3nt": "junction", "CDR3aa": "junction_aa", "V": "v_call",
                          "D": "d_call", "J": "j_call", "C": "c_call", "cid": "cell_id",
                          "#count": "consensus_count"})
    t = t[t["junction"].astype(str).str.len() > 9]
    t = t[~t["junction_aa"].astype(str).isin(("out_of_frame", "partial", "?"))]
    t["locus"] = t["v_call"].astype(str).str.upper().str[:3]
    # TRUST4 ids look like M1_F02_1_S222_R1_001_assemble0; the matrix barcodes are the plate-well stems
    # TRUST4 ids are M1_F02_1_S222_R1_001_assemble0; the matrix barcode is the plate well, M1_F02
    t["cell"] = t["cell_id"].astype(str).str.replace(r"_\d+_S\d+_R1_001_assemble\d+$", "", regex=True)
    t["productive"] = "T"
    return t


def contigs_for(dataset: str) -> pd.DataFrame:
    """Contig-level AIRR for a dataset, with `cell` matching the cached embedding's obs names."""
    if dataset in ("mouse_np", "mouse_rbd"):
        libs = GSE287123_NP if dataset == "mouse_np" else _gse287123_rbd_libraries()
        frames = []
        for gsm, name, prefix in libs:
            f = DATA / "gse287123_np/extracted" / f"{gsm}_{name}_airr_rearrangement.tsv.gz"
            if not f.exists():
                continue
            t = pd.read_csv(f, sep="\t", low_memory=False)
            t["cell"] = prefix + "|" + t["cell_id"].astype(str)
            t["library"] = name
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    if dataset in ("malaria", "malaria_late"):
        n = "1" if dataset == "malaria" else "2"
        t = pd.read_csv(DATA / "gse286215_malaria" / f"exp{n}_bcr.tsv.gz", sep="\t", low_memory=False)
        t = t.loc[:, ~t.columns.duplicated()]
        t["cell"] = t["cell_id"].astype(str) + "-1"      # the count matrix keeps the 10x suffix
        return t

    if dataset == "ln_vaccine":
        d = DATA / "gse195673_ln_vaccine"
        frames = []
        for f in ("bcr_heavy.tsv.gz", "bcr_light.tsv.gz"):
            t = pd.read_csv(d / f, sep="\t", low_memory=False)
            t = t[t["cell_id"].notna()]
            t["cell"] = t["cell_id"].astype(str)
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    if dataset == "flu_lung":
        frames = []
        for f in sorted(glob.glob(str(DATA / "gse317692_flu_lung/*airr_rearrangement.tsv.gz"))):
            # GSM9476721_pool616-1_1_airr... -> pool 1 -> the loader's prefix is "P1"
            tag = "P" + Path(f).name.split("_")[2]
            t = pd.read_csv(f, sep="\t", low_memory=False)
            t["cell"] = tag + "|" + t["cell_id"].astype(str)
            frames.append(t)
        return pd.concat(frames, ignore_index=True)


    if dataset == "gc_np_pc":
        d = DATA / "gse246382_np_pc"
        return _trust4_report(d / "GSE246382_trust_report.tsv.gz", d / "GSE246382_barcodes.tsv.gz")

    if dataset == "gut_gc_reporter":
        frames = []
        for m in ("mouse2", "mouse3", "mouse4"):
            f = DATA / "gse284622_gut_gc" / f"GSE284621_{m}_airr_rearrangement.tsv.gz"
            t = pd.read_csv(f, sep="\t", low_memory=False)
            t["barcode_plain"] = (t["cell_id"].astype(str)
                                  .str.replace(r"-\d+$", "", regex=True).str.replace(r"-1$", "", regex=True))
            t["cell"] = t["sample_id"].astype(str) + "_" + t["barcode_plain"]
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    if dataset == "covid_bcell":
        frames = []
        for f in sorted(glob.glob(str(DATA / "gse171703_covid_bcell/*_filtered_contig_annotations.csv.gz"))):
            tag = Path(f).name.split("_")[1]
            frames.append(_tenx(f, lambda b, tag=tag: f"{tag}|{b}"))
        return pd.concat(frames, ignore_index=True)

    if dataset == "spont_gc":
        t = pd.read_csv(DATA / "gse203132_spontaneous_gc/GSE203132_10X_clone-pass_germ-pass.tsv.gz",
                        sep="\t", low_memory=False)
        t = t.loc[:, ~t.columns.duplicated()]
        t["cell"] = t["mid"].astype(str) + "|" + t["barcode_10x"].astype(str)
        return t

    if dataset == "ebv":
        frames = []
        for f in sorted(glob.glob(str(DATA / "gse317492_ebv/*_vdjB_filtered_contig_annotations.csv.gz"))):
            # GFP-positive sample names contain an underscore (d14_gfp-pos-b).
            # Preserve the complete sample tag used by datasets._load_ebv.
            tag = Path(f).name.removesuffix("_vdjB_filtered_contig_annotations.csv.gz").split("_", 1)[1]
            frames.append(_tenx(f, lambda b, tag=tag: f"{tag}|{b}"))
        return pd.concat(frames, ignore_index=True)

    if dataset == "bone_marrow_pc":
        frames = []
        for f in sorted(glob.glob(str(DATA / "gse253857_bmpc/*_BCR.filtered_contig_annotations.csv.gz"))):
            name = Path(f).name.replace("_BCR.filtered_contig_annotations.csv.gz", "")
            frames.append(_tenx(f, lambda b, name=name: f"{name}|{b}"))
        return pd.concat(frames, ignore_index=True)

    if dataset == "tonsil":
        frames = []
        for f in sorted(glob.glob(str(DATA / "king2021_tonsil/extracted/*_vdj/**/filtered_contig_annotations.csv*"),
                                  recursive=True)):
            donor = Path(f).parts[[i for i, p in enumerate(Path(f).parts) if p.endswith("_vdj")][0]].replace("_vdj", "")
            frames.append(_tenx(f, lambda b, donor=donor: f"{donor}|{b.split('-')[0]}"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if dataset == "flu":
        frames = []
        for f in sorted(glob.glob(str(DATA / "gse175522_flu/*_clone_pass_fil_airr.txt.gz"))):
            parts = Path(f).name.split("_")
            donor, tp = parts[1], parts[2]            # e.g. GSM5340846_120648_0_... -> 120648, 0
            t = pd.read_csv(f, sep="\t", low_memory=False)
            bc = t["cell_id"].astype(str)
            bc = bc.where(bc.str.contains("-"), bc + "-1")   # obs names keep the 10x suffix
            t["cell"] = f"{donor}_d{tp}|" + bc
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    raise ValueError(f"No contig rule for {dataset!r}; add one before re-running it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["malaria", "malaria_late", "mouse_np", "mouse_rbd",
                                                    "ln_vaccine", "flu_lung", "gc_np_pc",
                                                    "ebv", "bone_marrow_pc", "tonsil", "flu"])
    ap.add_argument("--identity", type=float, default=0.85)
    ap.add_argument("--heavy-only", action="store_true",
                    help="for deposits with no light chain (e.g. GSE175522)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "Threadfin"))
    import anndata as ad

    from threadfin.dandelion import define_clones_dandelion

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        print(f"\n=== {ds}", flush=True)
        obs = ad.read_h5ad(CACHE / f"{ds}_embedding.h5ad").obs
        obs.index = obs.index.astype(str)
        t = contigs_for(ds)
        before = t["cell"].nunique()
        t = t[t["cell"].isin(obs.index)]
        t["donor"] = t["cell"].map(obs["donor"].astype(str))
        print(f"    contigs {len(t):,} over {t['cell'].nunique():,} cells "
              f"(of {before:,} in the deposit); {t['donor'].nunique()} donors", flush=True)
        if "locus" not in t.columns:
            t["locus"] = t["v_call"].astype(str).str.upper().str[:3]
        print(f"    loci: {t['locus'].value_counts().to_dict()}", flush=True)

        s = define_clones_dandelion(t, cell_key="cell", donor_key="donor",
                                    use_light_chain=not a.heavy_only,
                                    identity=a.identity, verbose=True)
        out = s.rename_axis("cell").reset_index().rename(columns={s.name: "clone_id"})
        out.to_csv(OUT / f"{ds}.csv", index=False)
        (OUT / f"{ds}.json").write_text(json.dumps(s.attrs, indent=2) + "\n")
        rows.append({"dataset": ds, **s.attrs})
        print(f"    wrote {OUT / f'{ds}.csv'}", flush=True)

    if rows:
        summary = pd.DataFrame(rows)
        path = OUT / "_summary.csv"
        if path.exists():
            summary = pd.concat([pd.read_csv(path), summary]).drop_duplicates("dataset", keep="last")
        summary.to_csv(path, index=False)
        print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
