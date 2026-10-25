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

    pip install "threadfin[dandelion]"        # or: pip install "sc-dandelion<1" "matplotlib<3.9"

Dandelion 0.5.7 imports ``matplotlib.cm.get_cmap``, which matplotlib 3.9 removed, but its package metadata
does not pin matplotlib, so the extra pins ``matplotlib < 3.9`` for it.

Dandelion's clone assignment depends on the order in which Python iterates over sets of sequences, which is
randomised in every new Python session unless the environment variable ``PYTHONHASHSEED`` is fixed before
Python starts. Repeated runs therefore differ by a few clones (GSE287123: 8,729 or 8,731 clones from the same
contigs). Set ``PYTHONHASHSEED=0`` (for a Jupyter kernel: ``python -m ipykernel install ... --env
PYTHONHASHSEED 0``) when clone ids must be reproducible; the value in use is recorded in the result.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from pathlib import Path

import pandas as pd

from ._utils import log

__all__ = ["define_clones_dandelion", "read_contigs"]

REQUIRED = ("cell_id", "v_call", "j_call", "junction")

TENX_TO_AIRR = {"barcode": "cell_id", "chain": "locus", "v_gene": "v_call", "d_gene": "d_call",
                "j_gene": "j_call", "c_gene": "c_call", "cdr3_nt": "junction", "cdr3": "junction_aa",
                "contig_id": "sequence_id", "umis": "umi_count", "reads": "consensus_count"}


def _contigs_from_scirpy(obj, airr_mod: str = "airr", airr_key: str = "airr") -> pd.DataFrame:
    """One row per receptor chain from a scirpy AnnData or MuData (``obsm['airr']``)."""
    try:
        import scirpy as ir
    except ImportError as exc:  # pragma: no cover - only without the optional dependency
        raise ImportError("Reading receptor data from a scirpy AnnData/MuData needs scirpy: pip install scirpy") from exc
    try:
        cells = ir.io.to_airr_cells(obj, airr_mod=airr_mod, airr_key=airr_key)
    except (KeyError, ValueError) as exc:
        raise ValueError("This AnnData/MuData holds no receptor data where scirpy keeps it (obsm['airr'] of the "
                         "'airr' modality). Pass the object returned by scirpy.io.read_10x_vdj / read_airr, or the "
                         "contig table itself.") from exc
    rows = [{**chain, "cell_id": cell.cell_id} for cell in cells for chain in cell.chains]
    if not rows:
        raise ValueError("The scirpy object holds no receptor chains.")
    return pd.DataFrame(rows)


def read_contigs(bcr, obs_names=None) -> pd.DataFrame:
    """Contig-level AIRR table from a path or a DataFrame, for :func:`define_clones_dandelion`.

    Accepts a Cell Ranger ``filtered_contig_annotations.csv`` (``.csv`` / ``.csv.gz``), an AIRR
    rearrangement ``.tsv`` / ``.tsv.gz``, a scirpy AnnData or MuData (receptor data in ``obsm['airr']``),
    or an already contig-level DataFrame (one row per chain,
    with ``cell_id``/``barcode``, ``v_call``/``v_gene``, ``j_call``/``j_gene`` and
    ``junction``/``cdr3_nt``). A per-cell table (one row per cell, e.g. the output of
    :func:`threadfin.read_bcr`) is refused: the paired heavy + light clone definition needs the
    individual chains. When a table has both a 10x ``barcode`` and an AIRR ``cell_id`` column, the one that
    matches more of ``obs_names`` (by default ``cell_id``) names the cells.
    """
    if not isinstance(bcr, (pd.DataFrame, str, Path)) and (hasattr(bcr, "mod") or hasattr(bcr, "obsm")):
        bcr = _contigs_from_scirpy(bcr)               # scirpy AnnData / MuData
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
    if "barcode" in t.columns and "cell_id" in t.columns:
        # both a 10x barcode and an AIRR cell id: keep the one that names the cells of adata (by default the
        # AIRR cell_id), so that renaming cannot leave two columns called cell_id
        keep = "cell_id"
        if obs_names is not None:
            names = pd.Index(obs_names).astype(str)
            hits = {c: int(t[c].astype(str).isin(names).sum()) for c in ("barcode", "cell_id")}
            keep = "barcode" if hits["barcode"] > hits["cell_id"] else "cell_id"
        t = t.drop(columns="cell_id" if keep == "barcode" else "barcode")
    # 10x column names become AIRR names, unless the AIRR column is already there
    t = t.rename(columns={k: v for k, v in TENX_TO_AIRR.items() if k in t.columns and v not in t.columns})
    t = t.loc[:, ~t.columns.duplicated()]
    if "is_cell" in t.columns:
        t = t[t["is_cell"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    if "high_confidence" in t.columns:
        t = t[t["high_confidence"].astype(str).str.upper().isin(("TRUE", "T", "1"))]
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:
        raise ValueError(f"contigs is missing {missing}; this function needs a contig-level AIRR table.")
    return t


def _quiet():
    """Dandelion prints a notice ("The AIRR data needs to undergo sanitization ...") for every table whose
    boolean columns are not spelled T/F, twice per donor even with ``verbose=False``; hide it."""
    return contextlib.redirect_stdout(io.StringIO())


def _dandelion_version(ddl) -> str:
    """Installed version of sc-dandelion. Its own ``__version__`` asks setuptools-scm first, which reports the
    version of whatever git repository the current directory is in."""
    try:
        from importlib.metadata import version

        return version("sc-dandelion")
    except Exception:
        return str(getattr(ddl, "__version__", "unknown"))


def _require_dandelion():
    try:
        with warnings.catch_warnings():
            # Dandelion imports nxviz, which prints a long notice about its own API change on import
            warnings.filterwarnings("ignore", message=r"\s*nxviz has a new API")
            import dandelion as ddl
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "Clone calling needs the Dandelion package: pip install \"threadfin[dandelion]\" (or: pip install "
            "\"sc-dandelion<1\" \"matplotlib<3.9\"; "
            "Dandelion 0.5.7 needs matplotlib < 3.9 but does not say so). Alternatively call clones elsewhere and "
            "put the ids in adata.obs['clone_id']."
        ) from exc
    major = _dandelion_version(ddl).split(".")[0]
    if major.isdigit() and int(major) >= 1:
        from . import __version__

        warnings.warn(
            f"sc-dandelion {_dandelion_version(ddl)} is installed; Threadfin {__version__} is validated with "
            "sc-dandelion 0.5.x. Dandelion 1.0 calls clones differently (on the GSE287123 tutorial data: 7,208 "
            "instead of 8,736 clones, largest clone 494 instead of 252 cells), so results will not match the "
            "documentation or the manuscript. Install the validated version with: pip install \"sc-dandelion<1\"",
            UserWarning, stacklevel=3)
    return ddl


def _locus(contigs: pd.DataFrame) -> pd.Series:
    """The chain of each contig, from `locus` when present and from the V call otherwise."""
    if "locus" in contigs.columns and contigs["locus"].notna().any():
        return contigs["locus"].astype(str).str.upper()
    return contigs["v_call"].astype(str).str.upper().str[:3]


#: chains that carry a D segment: the BCR heavy chain and the TCR beta and delta chains
VDJ_LOCI = ("IGH", "TRB", "TRD")


def _receptor_name(locus: pd.Series) -> str | None:
    """'BCR', 'TCR' or 'BCR or TCR': the receptor of the contigs. A few contigs of the other kind (a stray B cell
    among T cells) do not count; a kind needs at least 5 % of the IG and TR contigs."""
    kind = locus.astype(str).str.upper().str[:2].value_counts()
    kind = kind[kind.index.isin(["IG", "TR"])]
    share = kind / max(int(kind.sum()), 1)
    return " or ".join(name for prefix, name in (("IG", "BCR"), ("TR", "TCR")) if share.get(prefix, 0) >= 0.05) or None
#: Dandelion's own junction-identity defaults: B-cell clones tolerate somatic hypermutation, T-cell clones do not
DEFAULT_IDENTITY = {"ig": 0.85, "tr-ab": 1.0, "tr-gd": 1.0}


def _cells_with_heavy_chain(t: pd.DataFrame, cell_key: str = "cell_id") -> pd.Index:
    """Cells that carry at least one VDJ chain (BCR heavy chain, TCR beta or delta) in a contig-level table."""
    locus = t["locus"] if "locus" in t.columns else _locus(t)
    return pd.Index(pd.unique(t.loc[locus.astype(str).str.upper().isin(VDJ_LOCI).to_numpy(), cell_key].astype(str)))


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


def _unique_sequence_ids(t: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Make contig ids unique. Cell Ranger names contigs after the barcode (``AAAC...-1_contig_1``), so two
    libraries of one donor can carry the same id for different cells; Dandelion indexes contigs by this id and
    would mix those cells' chains. Duplicated ids are prefixed with the cell id."""
    sid = t["sequence_id"].astype(str)
    if not sid.duplicated().any():
        return t
    t = t.copy()
    dup = sid.duplicated(keep=False)
    t["sequence_id"] = sid.where(~dup, t["cell_id"].astype(str) + "|" + sid)
    if t["sequence_id"].duplicated().any():           # the same cell listed a contig twice
        t["sequence_id"] = t["sequence_id"] + "_" + t.groupby("sequence_id").cumcount().astype(str)
    if verbose:
        log(f"{int(dup.sum()):,} contigs shared their id with a contig of another cell (barcode reused across "
            "libraries); they were renamed after their cell")
    return t


def define_clones_dandelion(
    contigs: pd.DataFrame,
    *,
    use_light_chain: bool = True,
    identity: float | dict | None = None,
    donor_key: str | None = None,
    cell_key: str = "cell_id",
    productive_only: bool = True,
    require_heavy_chain: bool = True,
    check_contigs: bool = False,
    out_col: str = "clone_id",
    verbose: bool = True,
) -> pd.Series:
    """Call clones with Dandelion and return one clone id per cell.

    Parameters
    ----------
    contigs
        **Contig-level** AIRR table — one row per receptor chain, not per cell — with at least
        ``cell_id``, ``v_call``, ``j_call`` and ``junction``. This is what Cell Ranger's
        ``airr_rearrangement.tsv`` and ``filtered_contig_annotations.csv`` provide. BCR and TCR contigs are
        both accepted.
    use_light_chain
        ``True`` (default) defines a clone on both chains (BCR heavy + light; TCR beta + alpha), which is
        Dandelion's own definition. ``False`` keeps only the VDJ chain (IGH, TRB, TRD), for deposits with no
        light (alpha) chain; the choice is recorded in ``Series.attrs['use_light_chain']``.
    identity
        Junction amino-acid identity required within a clone, passed to ``dandelion.tl.find_clones``.
        ``None`` (default) uses Dandelion's defaults: 0.85 for BCR, which allows for somatic hypermutation,
        and 1.0 (identical junctions) for TCR, which does not mutate. A number applies to every chain type;
        a dict sets them separately (keys ``"ig"``, ``"tr-ab"``, ``"tr-gd"``).
    donor_key
        Column naming the individual. When given, clones are called separately within each donor, so no
        clone can span two animals or participants; the donor is prefixed to the id.
    productive_only
        Drop non-productive rearrangements before calling.
    require_heavy_chain
        ``True`` (default) returns clone ids only for cells with a (productive) VDJ chain — the BCR heavy
        chain, or the TCR beta or delta chain. Dandelion
        also groups cells whose heavy chain was not recovered, by their light chain alone (ids such as
        ``B_VJ_...``). Light-chain rearrangements are far less diverse than heavy-chain ones, so these
        groups join unrelated cells and are not clones; in GSE287123 they were 5,276 of 21,939 cells and
        740 of 2,626 groups of two or more cells. ``False`` keeps them.
    check_contigs
        ``True`` runs Dandelion's own contig check (``dandelion.pp.check_contigs``) before the clones are
        called, as Dandelion's tutorial does: identical contigs of a cell are merged, a cell with several
        heavy chains keeps the one that clearly dominates in UMIs (IgM with IgD is allowed; two light chains
        are allowed too, as in allelic inclusion), and cells without a dominant heavy chain — often doublets —
        are set aside instead of joining clones. It needs UMI
        counts (``umi_count`` or ``duplicate_count``); without them every cell with two chains of one kind is
        set aside. Read counts (``consensus_count``) break ties and are replaced by the UMI count when absent.
        ``False`` (default, as in the manuscript) calls clones on all productive contigs.

    Returns
    -------
    Series indexed by cell id. ``attrs`` records the settings, the resulting clone counts (with
    ``n_cells_with_combined_id``, the cells with two productive chains of one kind, and
    ``n_cells_with_combined_id_two_heavy``, those among them with two heavy chains), and the
    ``PYTHONHASHSEED`` in use (``None`` when unset; clone calls are then not bit-reproducible between Python
    sessions, see the module notes).
    """
    required = [cell_key if c == "cell_id" else c for c in REQUIRED]
    if any(c not in contigs.columns for c in required) and cell_key == "cell_id":
        # Cell Ranger's filtered_contig_annotations.csv names the columns barcode, v_gene, j_gene, cdr3_nt;
        # read_contigs renames them to AIRR names (and keeps is_cell / high_confidence contigs only)
        contigs = read_contigs(contigs)
    missing = [c for c in required if c not in contigs.columns]
    if missing:
        raise ValueError(f"contigs is missing {missing}; this function needs a contig-level AIRR table "
                         f"(cell_key={cell_key!r} names the column with the cell barcode).")
    ddl = _require_dandelion()

    t = contigs.copy()
    t[cell_key] = t[cell_key].astype(str)
    if productive_only and "productive" in t.columns:
        keep = t["productive"].astype(str).str.strip().str.upper().isin(("T", "TRUE", "1", "1.0"))
        t = t[keep]
    t["locus"] = _locus(t)
    if not use_light_chain:
        t = t[t["locus"].isin(VDJ_LOCI)]
        if t.empty:
            raise ValueError("No heavy-chain (IGH) or TCR beta/delta contigs left; check the locus or v_call column.")
    if identity is None:
        identity = dict(DEFAULT_IDENTITY)
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
    t = _unique_sequence_ids(t, verbose=verbose)
    # Dandelion validates a fixed column list on construction. Deposits legitimately omit some of them
    # (GSE195673 has no constant-region call), and an absent optional field should not stop clone calling.
    for col in ("d_call", "c_call", "productive"):
        if col not in t.columns:
            t[col] = "T" if col == "productive" else ""
    t = _fill_junction_aa(t, verbose=verbose)
    with_heavy = set()

    groups = [(None, t)] if donor_key is None else list(t.groupby(donor_key, observed=True))
    out, n_ambiguous, combined, two_heavy = {}, 0, set(), set()
    if check_contigs and "umi_count" in t.columns and pd.to_numeric(t["umi_count"], errors="coerce").nunique() <= 1:
        warnings.warn("check_contigs=True but the contigs have no UMI counts, so no chain can dominate: every cell "
                      "with two chains of one kind will be set aside.", UserWarning, stacklevel=2)
    if check_contigs and "consensus_count" not in t.columns:
        # the check breaks ties in UMIs on read counts; without them the UMI count stands in
        umis = t["umi_count"] if "umi_count" in t.columns else t["duplicate_count"]
        t["consensus_count"] = pd.to_numeric(umis, errors="coerce").fillna(1).clip(lower=1)
    for donor, sub in groups:
        if sub["cell_id"].nunique() < 2:
            continue
        with _quiet():
            vdj = ddl.Dandelion(sub.reset_index(drop=True))
        if check_contigs:
            with _quiet(), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # returns the Dandelion object alone, or (Dandelion, AnnData) when an AnnData is passed
                checked = ddl.pp.check_contigs(vdj, productive_only=productive_only, verbose=False)
            vdj = checked[0] if isinstance(checked, tuple) else checked
            n_ambiguous += int(vdj.data["ambiguous"].astype(str).eq("T").sum()) if "ambiguous" in vdj.data else 0
        # find_clones ignores contigs flagged ambiguous, so a cell whose heavy chains were all set aside has
        # no heavy chain left to call a clone on
        kept = vdj.data if "ambiguous" not in vdj.data else vdj.data[vdj.data["ambiguous"].astype(str) != "T"]
        with_heavy.update(_cells_with_heavy_chain(kept))
        heavy = kept[_locus(kept).isin(VDJ_LOCI).to_numpy()].drop_duplicates(["cell_id", "v_call", "j_call", "junction_aa"])
        n_heavy = heavy.groupby("cell_id").size()
        two_heavy.update(n_heavy.index[n_heavy >= 2].astype(str))
        with _quiet():
            ddl.tl.find_clones(vdj, identity=identity, verbose=False)
        # a cell Dandelion could not place (e.g. all its heavy chains set aside by the check) gets an empty id
        clone_ids = vdj.data["clone_id"]
        called = vdj.data[clone_ids.notna() & clone_ids.astype(str).str.strip().ne("")]
        per_cell = called.groupby("cell_id")["clone_id"].first()
        # Dandelion joins the clones of a cell with two productive chains of one kind ("a|b")
        combined.update(per_cell.index[per_cell.astype(str).str.contains("|", regex=False)].astype(str))
        prefix = "" if donor is None else f"{donor}|"
        for cell, clone in per_cell.items():
            out[str(cell)] = f"{prefix}{clone}"
        if verbose and donor is not None:
            log(f"  {donor}: {per_cell.nunique()} clones over {len(per_cell)} cells")

    s = pd.Series(out, name=out_col, dtype="object")
    s.index.name = "cell_id"
    n_without_heavy = int((~s.index.isin(with_heavy)).sum())
    if require_heavy_chain and n_without_heavy:
        s = s[s.index.isin(with_heavy)]
        if verbose:
            log(f"{n_without_heavy:,} cells without a heavy (or TCR beta/delta) chain were grouped by their light "
                "(alpha) chain alone; they are left without a clone (require_heavy_chain=False keeps them)")
    sizes = s.value_counts()
    n_combined = int(s.index.isin(combined).sum())
    n_combined_heavy = int((s.index.isin(combined) & s.index.isin(two_heavy)).sum())
    s.attrs.update({
        "method": "dandelion",
        "receptor": _receptor_name(t["locus"]),
        "dandelion_version": _dandelion_version(ddl),
        "use_light_chain": bool(use_light_chain),
        "require_heavy_chain": bool(require_heavy_chain),
        "check_contigs": bool(check_contigs),
        "n_contigs_ambiguous": n_ambiguous,
        "n_cells_without_heavy_chain": n_without_heavy,
        "n_cells_with_combined_id": n_combined,
        "n_cells_with_combined_id_two_heavy": n_combined_heavy,
        "identity": identity if isinstance(identity, dict) else float(identity),
        "donor_key": donor_key,
        "n_cells": int(s.size),
        "n_clones": int(sizes.size),
        "n_clones_ge2": int((sizes >= 2).sum()),
        "largest_clone": int(sizes.max()) if sizes.size else 0,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    })
    if verbose:
        chains = "both chains" if use_light_chain else "VDJ chain only"
        ident = (f"identity {identity['ig']:g} (BCR), {identity['tr-ab']:g} (TCR)"
                 if isinstance(identity, dict) and {"ig", "tr-ab"} <= set(identity) else f"identity {identity}")
        log(f"Dandelion ({chains}, {ident}): {s.size:,} cells, {sizes.size:,} clones, "
            f"{int((sizes >= 2).sum()):,} with >= 2 cells, largest {int(sizes.max()) if sizes.size else 0}")
        if n_combined >= max(10, 0.02 * s.size):
            share = (f"{n_combined:,} cells ({100 * n_combined / s.size:.0f}%) carry two productive chains of one "
                     f"kind ({n_combined_heavy:,} two heavy (beta) chains, the others two light (alpha) chains)")
            joined = "Dandelion gives each a clone id that joins both, so most of them form a clone of their own"
            if check_contigs:     # the check allows two light (alpha) chains per cell, as in allelic inclusion
                log(f"{share} after the contig check; {joined}.")
            else:
                log(f"{share}: doublets, ambient RNA or allelic inclusion; {joined}. check_contigs=True keeps the "
                    "dominant heavy chain of a cell (Dandelion allows two light chains).")
        if os.environ.get("PYTHONHASHSEED") is None:
            log("PYTHONHASHSEED is not set, so Dandelion's clone calls can differ by a few clones between "
                "Python sessions; set it (e.g. to 0) before starting Python for reproducible clone ids")
    return s
