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
    contigs = contigs.assign(_umis=umis).sort_values("_umis", ascending=False)

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

    keep = [c for c in ("v_call", "d_call", "j_call", "c_call", "cdr3", "cdr3_aa") if c in bcr.columns]
    if "cdr3" not in keep and "cdr3_aa" in keep:
        bcr = bcr.rename(columns={"cdr3_aa": "cdr3"})
        keep = ["cdr3" if c == "cdr3_aa" else c for c in keep]
    per_cell = bcr.drop_duplicates(cell_col).set_index(cell_col)[keep]
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
        bcr_table[out_col] = (
            bcr_table["v_call"].astype(str)
            + "_"
            + bcr_table["d_call"].astype(str)
            + "_"
            + bcr_table["j_call"].astype(str)
        )
    elif strategy in ("clonotype_id", "cdr3"):
        if strategy not in bcr_table.columns:
            raise ValueError(f"strategy='{strategy}' requires a '{strategy}' column.")
        bcr_table[out_col] = bcr_table[strategy].astype(str)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    return bcr_table


def attach_bcr(
    adata,
    bcr_table: pd.DataFrame,
    clone_col: str = "clone_id",
    prefix: str = "bcr_",
) -> "object":
    """Attach per-cell BCR annotations to ``adata.obs`` (vectorized).

    Cells without a BCR get ``NaN``. A clone-level summary table
    (one row per clone: v/d/j calls, consensus CDR3, number of cells) is
    stored in ``adata.uns['threadfin_clones']`` for downstream use by
    :func:`threadfin.clonotype_recluster`.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with ``obs_names`` matching the BCR barcodes.
    bcr_table
        Per-cell BCR table indexed by barcode. Should already contain a
        ``clone_col`` column (see :func:`build_clone_key`).
    clone_col
        Column of ``bcr_table`` holding the clonotype id.
    prefix
        Prefix for the added ``obs`` columns.

    Returns
    -------
    The modified ``adata`` (in place and returned).
    """
    if clone_col not in bcr_table.columns:
        raise ValueError(f"bcr_table needs a '{clone_col}' column; see build_clone_key().")

    tab = bcr_table.copy()
    tab.index = tab.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)

    n_matched = adata.obs_names.isin(tab.index).sum()
    if n_matched == 0:
        raise ValueError(
            "No overlap between adata.obs_names and BCR barcodes — check barcode formatting."
        )

    cols = {c: prefix + c for c in tab.columns}
    adata.obs = adata.obs.join(tab.rename(columns=cols), how="left")
    adata.obs[clone_col] = adata.obs[prefix + clone_col]
    adata.obs = adata.obs.drop(columns=[prefix + clone_col])

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
        "n_cells_with_bcr": int(n_matched),
        "n_clones": int(clone_info.shape[0]),
    }
    print(
        f"[threadfin] attached BCR: {n_matched}/{adata.n_obs} cells, "
        f"{clone_info.shape[0]} clones."
    )
    return adata
