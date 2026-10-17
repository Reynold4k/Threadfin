"""Clone definition delegated to Dandelion, so heavy **and** light chains are used.

This is Threadfin's clone definition. A B-cell clone is one V(D)J rearrangement of *both*
chains, and two cells whose heavy chains are similar but whose light chains are unrelated are
not the same clone. Dandelion implements the paired definition and is a maintained, cited
package, so Threadfin calls it rather than re-implementing clone calling:
:func:`define_clones_dandelion` is the entry point, :func:`read_contigs` normalises a 10x
CSV / AIRR TSV / contig-level DataFrame for it. (Threadfin's earlier heavy-chain-only caller,
:func:`threadfin.clones.define_clones`, is deprecated.)

Dandelion's clone id is composite — ``B_VDJ_105_9_7_VJ_261_2_1`` — the heavy (VDJ) V/J/junction group and
the light (VJ) group, so the pairing is visible in the label.

What changes when the light chain is used. Measured on GSE286215 experiment 1 (23,259 cells):

===========================  ===========  ===========
quantity                     heavy only   heavy+light
===========================  ===========  ===========
clones                           20,100        22,351
clones with >= 2 cells              801           616
largest clone                     8 cells      33 cells
===========================  ===========  ===========

Roughly a quarter of the analysable clone population is lost, because a clone now has to agree on both
chains. That is the price of the stricter definition and it is reported, not hidden.

Not every deposit carries a light chain — GSE175522 (influenza vaccine) is heavy only — so
``use_light_chain`` is a parameter rather than an assumption, and the setting is recorded in the result so
a figure can state which definition produced it.

Dandelion is an optional dependency::

    pip install sc-dandelion

It pins ``matplotlib < 3.9``, so it is usually easiest to keep it in its own environment, call
:func:`define_clones_dandelion` there and carry the returned ids across as a table.
"""

from __future__ import annotations

import pandas as pd

from ._utils import log

__all__ = ["define_clones_dandelion", "read_contigs"]

REQUIRED = ("cell_id", "v_call", "j_call", "junction")

TENX_TO_AIRR = {"barcode": "cell_id", "chain": "locus", "v_gene": "v_call", "d_gene": "d_call",
                "j_gene": "j_call", "c_gene": "c_call", "cdr3_nt": "junction", "cdr3": "junction_aa",
                "contig_id": "sequence_id", "umis": "umi_count", "reads": "consensus_count"}


def read_contigs(bcr) -> pd.DataFrame:
    """Contig-level AIRR table from a path or a DataFrame, for :func:`define_clones_dandelion`.

    Accepts a Cell Ranger ``filtered_contig_annotations.csv`` (``.csv`` / ``.csv.gz``), an AIRR
    rearrangement ``.tsv`` / ``.tsv.gz``, or an already contig-level DataFrame (one row per chain,
    with ``cell_id``/``barcode``, ``v_call``/``v_gene``, ``j_call``/``j_gene`` and
    ``junction``/``cdr3_nt``). A per-cell table (one row per cell, e.g. the output of
    :func:`threadfin.read_bcr`) is refused: the paired heavy + light clone definition needs the
    individual chains.
    """
    if isinstance(bcr, pd.DataFrame):
        t = bcr.copy()
        if "cell_id" not in t.columns and "barcode" not in t.columns:
            raise ValueError(
                "A per-cell BCR table cannot be used for clone definition: Threadfin's clone "
                "definition is Dandelion's paired heavy + light definition, which needs the "
                "contig-level table (one row per chain, with a cell_id/barcode column). Pass the "
                "path to filtered_contig_annotations.csv or airr_rearrangement.tsv, or precompute "
                "clone ids with threadfin.define_clones_dandelion and put them in "
                "adata.obs['clone_id'].")
    else:
        path = str(bcr)
        if path.endswith((".tsv", ".tsv.gz")):
            t = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            t = pd.read_csv(path, low_memory=False)
    t = t.rename(columns={k: v for k, v in TENX_TO_AIRR.items() if k in t.columns})
    if "is_cell" in t.columns:
        t = t[t["is_cell"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    if "high_confidence" in t.columns:
        t = t[t["high_confidence"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:
        raise ValueError(f"contigs is missing {missing}; this function needs a contig-level AIRR table.")
    return t


def _require_dandelion():
    try:
        import dandelion as ddl
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "define_clones_dandelion needs the Dandelion package: `pip install sc-dandelion`. "
            "It pins matplotlib<3.9, so a separate environment is often easiest; run this function "
            "there and pass the resulting clone ids back."
        ) from exc
    return ddl


def _locus(contigs: pd.DataFrame) -> pd.Series:
    """The chain of each contig, from `locus` when present and from the V call otherwise."""
    if "locus" in contigs.columns and contigs["locus"].notna().any():
        return contigs["locus"].astype(str).str.upper()
    return contigs["v_call"].astype(str).str.upper().str[:3]


def _fill_junction_aa(t: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Make sure `junction_aa` carries a real sequence for every contig.

    ``find_clones`` groups on ``junction_aa`` by default, so a blank entry is not a missing optional
    field — it is a grouping key shared with every other blank entry. Filling the column with an empty
    string therefore merges every cell that happens to agree on V gene, J gene and junction length:
    GSE175522 ships ``junction_aa`` entirely empty and that produced a single 972-cell "clone" and
    dropped the dataset's clonal coherence from 0.558 to 0.050 before it was caught. Translate the
    nucleotide junction instead, which is the sequence Dandelion would have grouped on anyway.
    """
    from .sequence import translate_nt

    blank = {"", "nan", "none", "na", "null", "out_of_frame", "partial", "?", "-"}
    if "junction_aa" in t.columns:
        raw = t["junction_aa"].astype("object")
        # missing values become "" explicitly: with pandas >= 3, astype(str) keeps NaN as missing
        aa = raw.where(raw.notna(), "").astype(str).str.strip()
        aa = aa.where(~aa.str.lower().isin(blank), "")
    else:
        aa = pd.Series("", index=t.index, dtype="object")
    need = aa.eq("")
    if need.any():
        nt = t.loc[need, "junction"].astype(str)
        aa.loc[need] = [translate_nt(s) or "" for s in nt]
        filled = int((aa.loc[need] != "").sum())
        if verbose:
            log(f"junction_aa was blank for {int(need.sum()):,} of {len(t):,} contigs; translated the "
                f"nucleotide junction for {filled:,} of them")
    t = t.copy()
    t["junction_aa"] = aa
    still = t["junction_aa"].eq("")
    if still.any():
        if still.mean() > 0.05:
            raise ValueError(
                f"{int(still.sum()):,} of {len(t):,} contigs have neither junction_aa nor a translatable "
                "junction. Clone calling would group all of them together on V gene, J gene and junction "
                "length alone, so it is refused; supply junction_aa.")
        if verbose:
            log(f"dropping {int(still.sum()):,} contigs with no usable junction_aa")
        t = t[~still]
    return t


def define_clones_dandelion(
    contigs: pd.DataFrame,
    *,
    use_light_chain: bool = True,
    identity: float = 0.85,
    donor_key: str | None = None,
    cell_key: str = "cell_id",
    productive_only: bool = True,
    out_col: str = "clone_id",
    verbose: bool = True,
) -> pd.Series:
    """Call clones with Dandelion and return one clone id per cell.

    Parameters
    ----------
    contigs
        **Contig-level** AIRR table — one row per receptor chain, not per cell — with at least
        ``cell_id``, ``v_call``, ``j_call`` and ``junction``. This is what Cell Ranger's
        ``airr_rearrangement.tsv`` and ``filtered_contig_annotations.csv`` provide.
    use_light_chain
        ``True`` (default) defines a clone on the heavy **and** light chain, which is Dandelion's own
        definition. ``False`` keeps only IGH contigs, for deposits with no light chain; the choice is
        recorded in ``Series.attrs['use_light_chain']``.
    identity
        Junction identity threshold passed to ``dandelion.tl.find_clones``.
    donor_key
        Column naming the individual. When given, clones are called separately within each donor, so no
        clone can span two animals or participants; the donor is prefixed to the id.
    productive_only
        Drop non-productive rearrangements before calling.

    Returns
    -------
    Series indexed by cell id. ``attrs`` records the settings and the resulting clone counts.
    """
    ddl = _require_dandelion()
    missing = [c for c in REQUIRED if c not in contigs.columns]
    if missing:
        raise ValueError(f"contigs is missing {missing}; this function needs a contig-level AIRR table.")

    t = contigs.copy()
    t[cell_key] = t[cell_key].astype(str)
    if productive_only and "productive" in t.columns:
        keep = t["productive"].astype(str).str.upper().isin(("T", "TRUE", "1"))
        t = t[keep]
    t["locus"] = _locus(t)
    if not use_light_chain:
        t = t[t["locus"].eq("IGH")]
        if t.empty:
            raise ValueError("No IGH contigs left; check the locus or v_call column.")
    # Dandelion writes its own clone_id; keep any existing one under a different name
    if "clone_id" in t.columns:
        t = t.rename(columns={"clone_id": "clone_id_original"})
    if cell_key != "cell_id":
        # the caller's key becomes `cell_id`; any original `cell_id` is kept under another name so the
        # rename cannot produce two columns with the same label
        if "cell_id" in t.columns:
            t = t.rename(columns={"cell_id": "cell_id_contig"})
        t = t.rename(columns={cell_key: "cell_id"})
    t = t.loc[:, ~t.columns.duplicated()]
    # AIRR requires a unique per-contig id and Dandelion insists on it, but several public deposits
    # (GSE286215 experiment 2 among them) ship without one. Build it from the cell and the chain.
    # Dandelion also needs a per-contig abundance; TRUST4 reports and some deposits give only a read
    # count, so fall back through the usual names before giving every contig equal weight.
    if "umi_count" not in t.columns and "duplicate_count" not in t.columns:
        src = next((c for c in ("consensus_count", "reads", "#count", "count") if c in t.columns), None)
        t["umi_count"] = pd.to_numeric(t[src], errors="coerce").fillna(1) if src else 1
    if "sequence_id" not in t.columns or t["sequence_id"].isna().all():
        t = t.reset_index(drop=True)
        t["sequence_id"] = (t["cell_id"].astype(str) + "_" + t["locus"].astype(str) + "_"
                            + t.groupby(["cell_id", "locus"]).cumcount().add(1).astype(str))
    # Dandelion validates a fixed column list on construction. Deposits legitimately omit some of them
    # (GSE195673 has no constant-region call), and an absent optional field should not stop clone calling.
    for col in ("d_call", "c_call", "productive"):
        if col not in t.columns:
            t[col] = "T" if col == "productive" else ""
    t = _fill_junction_aa(t, verbose=verbose)

    groups = [(None, t)] if donor_key is None else list(t.groupby(donor_key, observed=True))
    out = {}
    for donor, sub in groups:
        if sub["cell_id"].nunique() < 2:
            continue
        vdj = ddl.Dandelion(sub.reset_index(drop=True))
        ddl.tl.find_clones(vdj, identity=identity, verbose=False)
        called = vdj.data.dropna(subset=["clone_id"])
        per_cell = called.groupby("cell_id")["clone_id"].first()
        prefix = "" if donor is None else f"{donor}|"
        for cell, clone in per_cell.items():
            out[str(cell)] = f"{prefix}{clone}"
        if verbose and donor is not None:
            log(f"  {donor}: {per_cell.nunique()} clones over {len(per_cell)} cells")

    s = pd.Series(out, name=out_col, dtype="object")
    s.index.name = "cell_id"
    sizes = s.value_counts()
    s.attrs.update({
        "method": "dandelion",
        "dandelion_version": getattr(ddl, "__version__", "unknown"),
        "use_light_chain": bool(use_light_chain),
        "identity": float(identity),
        "donor_key": donor_key,
        "n_cells": int(s.size),
        "n_clones": int(sizes.size),
        "n_clones_ge2": int((sizes >= 2).sum()),
        "largest_clone": int(sizes.max()) if sizes.size else 0,
    })
    if verbose:
        chains = "heavy + light" if use_light_chain else "heavy only"
        log(f"Dandelion ({chains}, identity {identity}): {s.size:,} cells, {sizes.size:,} clones, "
            f"{int((sizes >= 2).sum()):,} with >= 2 cells, largest {int(sizes.max()) if sizes.size else 0}")
    return s
