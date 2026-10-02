"""Input/output helpers for paired scRNA-seq + scBCR-seq data.

Supports the two most common BCR annotation formats:

* 10x Genomics Cell Ranger ``filtered_contig_annotations.csv``
* AIRR rearrangement TSV (e.g. IgBlast ``fmt19`` / Change-O ``db-pass``)

The main entry points are :func:`read_10x_vdj`, :func:`read_airr` and
:func:`attach_bcr`.
"""

from __future__ import annotations

import pandas as pd

_HEAVY = "IGH"
_LIGHT = ("IGK", "IGL")

_CLONE_INFO_KEY = "threadfin_clones"


def read_10x_vdj(
    path: str,
    productive_only: bool = True,
    high_confidence_only: bool = True,
) -> pd.DataFrame:
    """Read a 10x Cell Ranger ``filtered_contig_annotations.csv`` file.

    One cell can carry multiple contigs (heavy + light chain). The contig with
    the highest UMI count is kept per chain class.

    Parameters
    ----------
    path
        Path to the CSV (optionally gzipped).
    productive_only
        Keep only productive contigs.
    high_confidence_only
        Keep only high-confidence contigs.

    Returns
    -------
    DataFrame indexed by cell barcode with columns
    ``v_call, d_call, j_call, c_call, cdr3, cdr3_nt, umis, clonotype_id,
    v_call_light, cdr3_light, n_contigs``.
    """
    contigs = pd.read_csv(path)
    required = {"barcode", "chain", "v_gene", "j_gene", "cdr3"}
    missing = required - set(contigs.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if productive_only and "productive" in contigs.columns:
        contigs = contigs[contigs["productive"] == True]  # noqa: E712
    if high_confidence_only and "high_confidence" in contigs.columns:
        contigs = contigs[contigs["high_confidence"] == True]  # noqa: E712
    contigs = contigs[contigs["chain"].isin([_HEAVY, *_LIGHT])]
    if len(contigs) == 0:
        raise ValueError("No productive BCR contigs left after filtering.")

    umis = contigs["umis"] if "umis" in contigs.columns else pd.Series(1, index=contigs.index)
    contigs = contigs.assign(_umis=umis).sort_values(
        "_umis", ascending=False, kind="mergesort")

    n_contigs = contigs.groupby("barcode").size().rename("n_contigs")

    heavy = contigs[contigs["chain"] == _HEAVY].drop_duplicates("barcode").set_index("barcode")
    light = contigs[contigs["chain"].isin(_LIGHT)].drop_duplicates("barcode").set_index("barcode")

    per_cell = pd.DataFrame(
        {
            "v_call": heavy["v_gene"],
            "d_call": heavy["d_gene"] if "d_gene" in heavy.columns else pd.NA,
            "j_call": heavy["j_gene"],
            "c_call": heavy["c_gene"] if "c_gene" in heavy.columns else pd.NA,
            "cdr3": heavy["cdr3"],
            "cdr3_nt": heavy["cdr3_nt"] if "cdr3_nt" in heavy.columns else pd.NA,
            "umis": heavy["_umis"],
            "clonotype_id": (
                heavy["raw_clonotype_id"] if "raw_clonotype_id" in heavy.columns else pd.NA
            ),
            "v_call_light": light["v_gene"],
            "cdr3_light": light["cdr3"],
            "n_contigs": n_contigs,
        }
    )
    per_cell.index.name = "barcode"
    return per_cell


def read_airr(path: str, cell_col: str | None = None) -> pd.DataFrame:
    """Read an AIRR rearrangement table (IgBlast fmt19 / Change-O TSV).

    Parameters
    ----------
    path
        Path to the (optionally gzipped) TSV file.
    cell_col
        Column holding the cell barcode. When ``None``, ``cell_id`` is used if
        present, otherwise the trailing ``_contig_<n>`` suffix of
        ``sequence_id`` is stripped to recover the barcode.

    Returns
    -------
    DataFrame indexed by cell barcode with columns
    ``v_call, d_call, j_call, cdr3`` (whichever are present).
    """
    bcr = pd.read_csv(path, sep="\t")
    if cell_col is None:
        if "cell_id" in bcr.columns:
            cell_col = "cell_id"
        elif "sequence_id" in bcr.columns:
            bcr = bcr.assign(cell_id=bcr["sequence_id"].str.replace(r"_contig_\d+$", "", regex=True))
            cell_col = "cell_id"
        else:
            raise ValueError("Cannot find a cell barcode column; pass `cell_col` explicitly.")

    keep = [c for c in ("v_call", "d_call", "j_call", "c_call", "cdr3", "cdr3_aa",
                        "junction_aa", "clone_id", "locus") if c in bcr.columns]
    if "cdr3" not in keep and "cdr3_aa" in keep:
        bcr = bcr.rename(columns={"cdr3_aa": "cdr3"})
    elif "cdr3" not in keep and "junction_aa" in keep:
        bcr = bcr.rename(columns={"junction_aa": "cdr3"})
    # some AIRR tables ship an empty junction_aa but a populated nucleotide
    # junction; translate it wherever the amino-acid CDR3 is missing
    if "junction" in bcr.columns:
        from .sequence import translate_nt

        if "cdr3" not in bcr.columns:
            bcr["cdr3"] = bcr["junction"].map(translate_nt)
        elif bcr["cdr3"].isna().any():
            na = bcr["cdr3"].isna()
            bcr["cdr3"] = bcr["cdr3"].astype(object)
            bcr.loc[na, "cdr3"] = bcr.loc[na, "junction"].map(translate_nt)
    # prefer the heavy chain for per-cell V/D/J calls when a locus column exists
    if "locus" in bcr.columns and (bcr["locus"] == "IGH").any():
        bcr = bcr[bcr["locus"] == "IGH"]
    per_cell = bcr.drop_duplicates(cell_col).set_index(cell_col)
    keep = [c for c in ("v_call", "d_call", "j_call", "c_call", "cdr3", "clone_id")
            if c in per_cell.columns]
    per_cell = per_cell[keep]
    per_cell.index.name = "barcode"
    return per_cell


def build_clone_key(
    bcr_table: pd.DataFrame,
    strategy: str = "vdj",
    out_col: str = "clone_id",
) -> pd.DataFrame:
    """Add a clonotype key to a per-cell BCR table.

    Parameters
    ----------
    bcr_table
        Per-cell BCR table as returned by :func:`read_10x_vdj` or
        :func:`read_airr`.
    strategy
        ``"vdj"``: ``v_call_d_call_j_call`` of the heavy chain (germline-gene
        level clonotype, as in the original Threadfin manuscript).
        ``"clonotype_id"``: use the Cell Ranger clonotype id (somatic-level).
        ``"cdr3"``: use the heavy-chain CDR3 amino acid sequence.
    out_col
        Name of the output column.

    Returns
    -------
    The input table with an added ``out_col`` column.
    """
    bcr_table = bcr_table.copy()
    if strategy == "vdj":
        for col in ("v_call", "d_call", "j_call"):
            if col not in bcr_table.columns:
                raise ValueError(f"strategy='vdj' requires a '{col}' column.")
        key = (
            bcr_table["v_call"].astype(str)
            + "_"
            + bcr_table["d_call"].astype(str)
            + "_"
            + bcr_table["j_call"].astype(str)
        )
        missing = bcr_table[["v_call", "d_call", "j_call"]].isna().any(axis=1)
        bcr_table[out_col] = key.mask(missing)
    elif strategy in ("clonotype_id", "cdr3"):
        if strategy not in bcr_table.columns:
            raise ValueError(f"strategy='{strategy}' requires a '{strategy}' column.")
        vals = bcr_table[strategy]
        is_na = vals.isna() | vals.astype(str).isin(("None", "", "nan"))
        bcr_table[out_col] = vals.astype(str).mask(is_na)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    return bcr_table


def attach_bcr(
    adata,
    bcr_table: pd.DataFrame,
    clone_col: str = "clone_id",
    prefix: str = "bcr_",
    summarize: bool = True,
    verbose: bool = True,
) -> "object":
    """Attach per-cell BCR annotations to ``adata.obs`` (vectorized).

    Cells without a BCR get ``NaN``. Every column of ``bcr_table`` is added
    to ``obs`` with ``prefix`` (the clone column keeps its own name).
    Columns from a previous call with the same prefix are replaced.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with ``obs_names`` matching the BCR barcodes.
    bcr_table
        Per-cell BCR table indexed by barcode, with a ``clone_col`` column
        (see :func:`threadfin.define_clones` or :func:`build_clone_key`).
    clone_col
        Column of ``bcr_table`` holding the clone id.
    prefix
        Prefix for the added ``obs`` columns.
    summarize
        Also store a clone-level summary (modal V/D/J calls and CDR3 per
        clone) in ``adata.uns['threadfin_clones']``; only needed by the legacy
        sequence-blending functions (``cdr3_weight``, ``joint_embedding``).

    Returns
    -------
    The modified ``adata`` (in place and returned).
    """
    if clone_col not in bcr_table.columns:
        raise ValueError(f"bcr_table needs a '{clone_col}' column; see define_clones().")

    tab = bcr_table.copy()
    tab.index = tab.index.astype(str)
    if tab.index.has_duplicates:
        raise ValueError("bcr_table has duplicated barcodes; keep one row per cell.")
    adata.obs_names = adata.obs_names.astype(str)

    n_matched = int(adata.obs_names.isin(tab.index).sum())
    if n_matched == 0:
        raise ValueError(
            "No overlap between adata.obs_names and BCR barcodes — check barcode formatting."
        )

    renamed = {c: prefix + c for c in tab.columns if c != clone_col}
    stale = [c for c in list(renamed.values()) + [clone_col] if c in adata.obs.columns]
    obs = adata.obs.drop(columns=stale)
    adata.obs = obs.join(tab.rename(columns=renamed), how="left")

    n_clones = int(tab.loc[tab.index.isin(adata.obs_names), clone_col].nunique())
    if summarize:
        clone_info = (
            tab.dropna(subset=[clone_col])
            .groupby(clone_col)
            .agg({c: (lambda s: (lambda m: m.iloc[0] if len(m) else None)(s.mode()))
                  for c in tab.columns if c != clone_col})
        )
        clone_info["n_cells_bcr_table"] = tab[clone_col].value_counts()
        adata.uns[_CLONE_INFO_KEY] = clone_info

    adata.uns.setdefault("threadfin", {})["attach_bcr"] = {
        "n_cells_total": int(adata.n_obs),
        "n_cells_with_bcr": n_matched,
        "n_clones": n_clones,
    }
    if verbose:
        print(f"[threadfin] attached BCR: {n_matched}/{adata.n_obs} cells, {n_clones} clones.", flush=True)
    return adata
