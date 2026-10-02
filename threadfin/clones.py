"""B-cell clones: definition from BCR sequences, and clone-level summaries.

Clone definition (:func:`define_clones`) follows the established rules for
B-cell receptors (Gupta et al. 2017; Nouri & Kleinstein 2018):

* cells are compared **only within a donor** - two people never share a clone,
  so identical sequences from different donors are convergent, not clonal;
* candidate relatives must use the same IGHV gene, the same IGHJ gene and a
  junction of the same length (somatic hypermutation rarely changes length);
* sequences closer than a normalised Hamming-distance threshold are joined by
  single linkage. With ``threshold="auto"`` the threshold is the valley of the
  bimodal distance-to-nearest distribution (the "density" method of SHazaM);
* optionally, heavy-chain clones are split by light chain.

Unlike exact clonotypes, this merges the hypermutated variants of one
germinal-centre lineage into a single clone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._utils import log

_ISOTYPES = ("IGHM", "IGHD", "IGHG", "IGHA", "IGHE")


# =========================================================================== clone definition


def _gene_call(x) -> str | None:
    """First call of a (possibly multi-call) V/J assignment, without allele."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    s = str(x).split(",")[0].split("*")[0].strip()
    return s or None


def _encode(seqs) -> np.ndarray:
    """Equal-length strings -> uint8 matrix."""
    return np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(len(seqs), -1)


def _hamming_rows(codes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Normalised Hamming distances between ``rows`` and all sequences."""
    return (codes[rows][:, None, :] != codes[None, :, :]).mean(axis=2)


def _chunk_rows(codes: np.ndarray, budget: float = 2e8) -> int:
    """Rows per chunk so that one (chunk x n x length) comparison stays ~200 MB."""
    n, length = codes.shape
    return int(max(8, min(4096, budget // max(n * length, 1))))


def _distance_to_nearest(codes: np.ndarray) -> np.ndarray:
    """Distance of every (unique) sequence to its nearest other sequence."""
    n = codes.shape[0]
    out = np.full(n, np.nan)
    if n < 2:
        return out
    chunk = _chunk_rows(codes)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = _hamming_rows(codes, rows)
        d[np.arange(rows.size), rows] = np.inf
        out[rows] = d.min(axis=1)
    return out


def _linkage_edges(codes: np.ndarray, threshold: float):
    """Pairs (i < j) with normalised Hamming distance <= threshold."""
    n = codes.shape[0]
    chunk = _chunk_rows(codes)
    src, dst = [], []
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = _hamming_rows(codes, rows)
        i, j = np.nonzero(d <= threshold + 1e-12)
        i = rows[i]
        keep = i < j
        src.append(i[keep])
        dst.append(j[keep])
    return np.concatenate(src), np.concatenate(dst)


def find_threshold(distances, *, default: float = 0.15, lo: float = 0.02, hi: float = 0.25,
                   max_first_mode: float = 0.12) -> dict:
    """Threshold at the density valley of a distance-to-nearest distribution.

    Clonally related sequences sit close to their nearest relative, unrelated
    sequences far away, so the distance-to-nearest distribution is bimodal;
    the valley between the modes separates them. The first ("related") mode
    must lie below ``max_first_mode`` and the valley inside ``[lo, hi]``;
    otherwise (for example in repertoires dominated by unmutated naive B
    cells, where no clean related mode exists) ``default`` is returned. The
    upper bound follows common practice for nucleotide junction distances
    (thresholds of roughly 0.05-0.20; Gupta et al. 2017).
    """
    from scipy.stats import gaussian_kde

    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    info = {"threshold": default, "method": "default", "n": int(d.size)}
    if d.size < 50 or np.ptp(d) == 0:
        return info
    grid = np.linspace(0.0, max(0.6, float(np.quantile(d, 0.99))), 601)
    dens = gaussian_kde(d)(grid)
    peaks = np.flatnonzero((dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])) + 1
    first = peaks[grid[peaks] <= max_first_mode]
    if first.size:
        p1 = first[np.argmax(dens[first])]
        later = peaks[peaks > p1]
        if later.size:
            p2 = later[np.argmax(dens[later])]
            valley = p1 + int(np.argmin(dens[p1:p2 + 1]))
            if lo <= grid[valley] <= hi:
                info.update(threshold=float(grid[valley]), method="density")
    return info


def define_clones(
    bcr: pd.DataFrame,
    *,
    donor_key: str | None = "donor",
    v_col: str = "v_call",
    j_col: str = "j_call",
    junction_col: str | None = None,
    threshold: float | str = "auto",
    light_chain: bool = False,
    light_v_col: str = "v_call_light",
    light_j_col: str = "j_call_light",
    light_junction_col: str | None = None,
    out_col: str = "clone_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """Group cells into B-cell clones (lineages) from their receptor sequences.

    Parameters
    ----------
    bcr
        Per-cell table (one row per cell, e.g. from :func:`threadfin.read_bcr`)
        with V gene, J gene and junction (or CDR3) sequence of the heavy chain.
    donor_key
        Column identifying the individual. Clones are never formed across
        donors. ``None`` treats all cells as one donor.
    junction_col
        Heavy-chain junction/CDR3 column. Default: the first available of
        ``junction``, ``cdr3_nt`` (nucleotides, preferred) and
        ``junction_aa``, ``cdr3`` (amino acids).
    threshold
        Maximum normalised Hamming distance for single linkage, or ``"auto"``
        (density valley, :func:`find_threshold`; defaults 0.15 for
        nucleotides and 0.20 for amino acids if no valley is found).
    light_chain
        Split heavy-chain clones whose cells carry different light chains
        (light V gene, light J gene and junction length). Cells without a
        light chain join the clone's most common light-chain group.
    out_col
        Name of the new clone-id column. Ids look like ``"<donor>|C00001"``,
        numbered by decreasing size within each donor.

    Returns
    -------
    Copy of ``bcr`` with ``out_col`` (NaN where V, J or junction is
    missing). ``.attrs['clone_definition']`` records the threshold, the
    sequence type and the number of clones.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    out = bcr.copy()
    if junction_col is None:
        junction_col = next((c for c in ("junction", "cdr3_nt", "junction_aa", "cdr3") if c in out.columns), None)
    for col in (v_col, j_col, junction_col):
        if col is None or col not in out.columns:
            raise KeyError(f"define_clones needs a '{col}' column (V gene, J gene, junction).")
    seq_type = "nt" if junction_col in ("junction", "cdr3_nt") else "aa"
    default = 0.15 if seq_type == "nt" else 0.20

    donor = out[donor_key].astype(str) if donor_key is not None else pd.Series("all", index=out.index)
    junction = out[junction_col].astype("object").where(out[junction_col].notna())
    junction = junction.map(lambda s: str(s).strip().upper() if isinstance(s, str) and s.strip() else None)
    work = pd.DataFrame({
        "donor": donor, "v": out[v_col].map(_gene_call), "j": out[j_col].map(_gene_call), "seq": junction,
    }, index=out.index).dropna()
    work["len"] = work["seq"].str.len()

    # unique sequences per partition (donor, V, J, junction length)
    uniq = work.drop_duplicates(["donor", "v", "j", "len", "seq"]).reset_index(drop=True)
    parts = uniq.groupby(["donor", "v", "j", "len"], sort=False).indices

    # 1) threshold
    if threshold == "auto":
        dtn = np.full(len(uniq), np.nan)
        for idx in parts.values():
            if idx.size >= 2:
                dtn[idx] = _distance_to_nearest(_encode(uniq.loc[idx, "seq"].tolist()))
        info = find_threshold(dtn, default=default)
        if info["method"] == "default":
            warnings.warn(
                f"No bimodal distance-to-nearest distribution found; using the default "
                f"threshold {default} ({seq_type}).", stacklevel=2,
            )
        thr = info["threshold"]
    else:
        thr, info = float(threshold), {"threshold": float(threshold), "method": "user", "n": None}

    # 2) single linkage within each partition
    group = np.empty(len(uniq), dtype=np.int64)
    next_id = 0
    for idx in parts.values():
        if idx.size == 1:
            group[idx] = next_id
            next_id += 1
            continue
        src, dst = _linkage_edges(_encode(uniq.loc[idx, "seq"].tolist()), thr)
        adj = coo_matrix((np.ones(src.size), (src, dst)), shape=(idx.size, idx.size))
        n_comp, lab = connected_components(adj, directed=False)
        group[idx] = next_id + lab
        next_id += n_comp
    uniq["group"] = group
    key = ["donor", "v", "j", "len", "seq"]
    work = work.merge(uniq[key + ["group"]], on=key, how="left").set_index(work.index)

    # 3) optional light-chain split
    if light_chain:
        work["group"] = _split_by_light(out, work, light_v_col, light_j_col, light_junction_col)

    # 4) readable ids, numbered by size within donor
    sizes = work.groupby(["donor", "group"]).size().rename("n").reset_index()
    sizes = sizes.sort_values(["donor", "n", "group"], ascending=[True, False, True])
    sizes["rank"] = sizes.groupby("donor").cumcount() + 1
    sizes["clone"] = sizes["donor"] + "|C" + sizes["rank"].astype(str).str.zfill(5)
    lut = sizes.set_index(["donor", "group"])["clone"]
    labels = pd.Series(lut.reindex(pd.MultiIndex.from_arrays([work["donor"], work["group"]])).to_numpy(),
                       index=work.index)
    out[out_col] = labels.reindex(out.index)

    n_clones = int(labels.nunique())
    expanded = int((labels.value_counts() >= 2).sum())
    out.attrs["clone_definition"] = {
        "threshold": thr, "threshold_method": info["method"], "sequence": seq_type,
        "junction_col": junction_col, "donor_key": donor_key, "light_chain": light_chain,
        "n_cells": int(labels.notna().sum()), "n_clones": n_clones, "n_expanded": expanded,
    }
    log(f"define_clones: {labels.notna().sum()} cells -> {n_clones} clones ({expanded} with >= 2 "
        f"cells); {seq_type} Hamming threshold {thr:.3f} ({info['method']}).", verbose)
    return out


def _split_by_light(bcr, work, v_col, j_col, junction_col):
    """Sub-split heavy-chain groups by light chain (V, J, junction length)."""
    if junction_col is None:
        junction_col = next((c for c in ("junction_light", "cdr3_nt_light", "cdr3_light") if c in bcr.columns), None)
    if v_col not in bcr.columns or junction_col is None:
        raise KeyError("light_chain=True needs light-chain V and junction columns.")
    light = pd.Series(
        bcr.loc[work.index, v_col].map(_gene_call).astype(str) + "|"
        + (bcr.loc[work.index, j_col].map(_gene_call).astype(str) if j_col in bcr.columns else "")
        + "|" + bcr.loc[work.index, junction_col].astype("object").map(
            lambda s: str(len(s)) if isinstance(s, str) else None).astype(str),
        index=work.index,
    )
    has_light = bcr.loc[work.index, v_col].notna() & bcr.loc[work.index, junction_col].notna()
    new = work["group"].astype(str).copy()
    for grp, cells in work.groupby("group").groups.items():
        lc = light.loc[cells][has_light.loc[cells]]
        if lc.nunique() <= 1:
            continue
        top = lc.value_counts().index[0]
        sub = light.loc[cells].where(has_light.loc[cells], top)
        codes_ = {lab: k for k, lab in enumerate(sub.value_counts().index)}
        new.loc[cells] = [f"{grp}_L{codes_[s]}" for s in sub]
    return pd.factorize(new)[0]


SKIP_BASES = np.frombuffer(b"-.N", dtype=np.uint8)


def _encode_seq(s: str) -> np.ndarray:
    return np.frombuffer(s.upper().encode("ascii", "replace"), dtype=np.uint8)


def _mismatch_rate(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    ok = ~(np.isin(a, SKIP_BASES) | np.isin(b, SKIP_BASES))
    return int((a[ok] != b[ok]).sum()), int(ok.sum())


def _best_start(needle: str, haystack: str, max_offset: int, probe: int) -> tuple[int, float]:
    """Offset in ``haystack`` where ``needle`` best matches, and that mismatch rate."""
    a = _encode_seq(needle[:probe])
    if a.size == 0 or len(haystack) < a.size:
        return 0, 1.0
    b = _encode_seq(haystack[: a.size + max_offset])
    if b.size < a.size:
        return 0, 1.0
    from numpy.lib.stride_tricks import sliding_window_view

    ok = ~np.isin(a, SKIP_BASES)
    if not ok.any():
        return 0, 1.0
    counts = (sliding_window_view(b, a.size)[:, ok] != a[ok]).sum(axis=1)
    i = int(np.argmin(counts))
    return i, float(counts[i] / ok.sum())


def align_starts(sequence: str, germline: str, *, max_offset: int = 400, probe: int = 200) -> tuple[int, int]:
    """Where the V region starts in the query and in the germline string.

    AIRR files should give both in one coordinate system, but pipelines
    differ in how much upstream sequence (5' UTR, leader) each string
    carries, and the AIRR coordinate columns refer to the raw read rather
    than to these alignment strings. The start of one string is therefore
    located inside the other, in whichever direction matches better; the
    result is ``(0, 0)`` for files that already agree.
    """
    g_in_s, rate_a = _best_start(germline, sequence, max_offset, probe)   # germline starts later in the query
    s_in_g, rate_b = _best_start(sequence, germline, max_offset, probe)   # query starts later in the germline
    return (g_in_s, 0) if rate_a <= rate_b else (0, s_in_g)


def v_region_windows(airr: pd.DataFrame, sequence_col: str, germline_col: str, *, align: str = "auto",
                     max_offset: int = 400) -> pd.DataFrame:
    """Where the V segment sits in the query and germline strings, for every row.

    Starts come from :func:`align_starts` (``align="auto"``) or are both 0
    (``align="none"``, the AIRR standard). The length is the V length from
    ``v_sequence_start`` / ``v_sequence_end`` when available, so the junction
    - whose N additions have no germline to compare with - is excluded.

    Returns 0-based ``query_start``, ``germline_start`` and ``length``
    (NaN where the V region cannot be located).
    """
    def col(name):
        return airr[name] if name in airr.columns else None

    qs, qe = col("v_sequence_start"), col("v_sequence_end")
    seq, germ = airr[sequence_col].to_numpy(), airr[germline_col].to_numpy()
    out = np.full((len(airr), 3), np.nan)
    for k in range(len(airr)):
        s_, g_ = seq[k], germ[k]
        if not (isinstance(s_, str) and isinstance(g_, str)) or not s_ or not g_:
            continue
        q0, g0 = align_starts(s_, g_, max_offset=max_offset) if align == "auto" else (0, 0)
        if qs is not None and qe is not None and pd.notna(qs.iloc[k]) and pd.notna(qe.iloc[k]):
            n = int(qe.iloc[k]) - int(qs.iloc[k]) + 1          # V length, not an offset
        else:
            n = min(len(s_) - q0, len(g_) - g0)
        n = int(min(n, len(s_) - q0, len(g_) - g0))
        if n > 0:
            out[k] = (q0, g0, n)
    return pd.DataFrame(out, index=airr.index, columns=["query_start", "germline_start", "length"])


def mutation_frequency(
    airr: pd.DataFrame,
    *,
    sequence_col: str = "sequence_alignment",
    germline_col: str = "germline_alignment",
    align: str = "auto",
) -> pd.Series:
    """V-region somatic mutation frequency from AIRR alignments.

    Counts substitutions between the query and its germline over the V
    segment (:func:`v_region_windows`), ignoring gaps (``-``, ``.``) and
    ambiguous ``N``; returns mismatches / compared positions per row (NaN
    when the V region cannot be located).

    With ``align="auto"`` (the default) the two strings are first put in a
    common coordinate system, because pipelines differ in how much upstream
    sequence each carries; without this, cells of such a dataset all look
    hypermutated. ``align="none"`` compares them from their starts, as the
    AIRR standard specifies. Insertions and deletions are not modelled, so a
    clone with an indel in its V region can look more mutated than it is.
    The windows used are reported in ``.attrs['windows']``.
    """
    if sequence_col not in airr.columns or germline_col not in airr.columns:
        raise KeyError(f"mutation_frequency needs '{sequence_col}' and '{germline_col}'.")
    if align not in ("auto", "none"):
        raise ValueError("align must be 'auto' or 'none'.")
    seq, germ = airr[sequence_col].to_numpy(), airr[germline_col].to_numpy()
    win = v_region_windows(airr, sequence_col, germline_col, align=align, max_offset=400)
    out = np.full(len(airr), np.nan)
    for k in range(len(airr)):
        s_, g_ = seq[k], germ[k]
        q0, g0, n = win.iloc[k]
        if not (isinstance(s_, str) and isinstance(g_, str)) or not np.isfinite(n):
            continue
        q0, g0, n = int(q0), int(g0), int(n)
        mism, compared = _mismatch_rate(_encode_seq(s_[q0:q0 + n]), _encode_seq(g_[g0:g0 + n]))
        if compared > 0:
            out[k] = mism / compared
    res = pd.Series(out, index=airr.index, name="mutation_frequency")
    res.attrs["windows"] = win
    return res


# =========================================================================== summaries


def _isotype_class(c_call):
    """First call before ',' collapsed to the heavy-chain class (IGHG1 -> IGHG)."""
    if c_call is None or (isinstance(c_call, float) and np.isnan(c_call)) or c_call is pd.NA:
        return None
    first = str(c_call).split(",")[0].strip()
    for cls in _ISOTYPES:
        if first.upper().startswith(cls):
            return cls
    return first or None


def isotype_class(values) -> pd.Series:
    """Collapse constant-region calls (``IGHG1``, ``Ighg2c``) to classes (``IGHG``)."""
    return pd.Series(values).map(_isotype_class)


def clone_isotype_summary(adata, *, cluster_key: str | None = "clone_cluster") -> pd.DataFrame:
    """Per-group heavy-chain isotype composition.

    Fractions are over cells with a non-missing ``bcr_c_call`` in the group.
    With ``cluster_key=None`` the summary is per ``clone_id`` instead.

    Returns
    -------
    Tidy DataFrame with columns ``[group, isotype, fraction, n_cells]``.
    """
    if "bcr_c_call" not in adata.obs.columns:
        raise KeyError("'bcr_c_call' not found in adata.obs. Attach BCR annotations first.")
    group_col = cluster_key if cluster_key is not None else "clone_id"
    if group_col not in adata.obs.columns:
        raise KeyError(f"'{group_col}' not found in adata.obs.")
    df = pd.DataFrame({
        "group": adata.obs[group_col].astype("string"),
        "isotype": adata.obs["bcr_c_call"].map(_isotype_class),
    }).dropna()
    counts = df.groupby(["group", "isotype"]).size().rename("n_cells")
    totals = df.groupby("group").size()
    out = counts.reset_index()
    out["fraction"] = out["n_cells"] / out["group"].map(totals)
    out = out.rename(columns={"group": group_col})[[group_col, "isotype", "fraction", "n_cells"]]
    return out.sort_values([group_col, "isotype"]).reset_index(drop=True)


def _cell_shm(adata, mut_col: str | None) -> pd.Series:
    """Per-cell SHM values: ``obs[mut_col]``, else ``100 - obs['bcr_v_identity']``."""
    if mut_col is not None and mut_col in adata.obs.columns:
        return pd.to_numeric(adata.obs[mut_col], errors="coerce")
    if "bcr_v_identity" in adata.obs.columns:
        return 100.0 - pd.to_numeric(adata.obs["bcr_v_identity"], errors="coerce")
    raise ValueError(
        f"SHM summary needs '{mut_col}' or 'bcr_v_identity' in adata.obs "
        "(an AIRR mutation count or V-gene identity column)."
    )


def clone_shm_summary(adata, *, mut_col: str | None = "bcr_mu_count", cluster_key: str | None = None) -> pd.DataFrame:
    """Mean/median somatic hypermutation per clone (and per cluster).

    Returns
    -------
    Tidy DataFrame ``[level, group, shm_mean, shm_median, n_cells]``.
    """
    if "clone_id" not in adata.obs.columns:
        raise KeyError("'clone_id' not found in adata.obs.")
    df = pd.DataFrame({"clone_id": adata.obs["clone_id"], "shm": _cell_shm(adata, mut_col)})
    df = df.dropna(subset=["clone_id", "shm"])

    def _summarize(frame, level, name):
        g = frame.groupby(name)["shm"].agg(["mean", "median", "count"])
        g.columns = ["shm_mean", "shm_median", "n_cells"]
        g["n_cells"] = g["n_cells"].astype(int)
        g.insert(0, "group", g.index.astype(str))
        g.insert(0, "level", level)
        return g.reset_index(drop=True)

    parts = [_summarize(df, "clone", "clone_id")]
    if cluster_key is not None:
        if cluster_key not in adata.obs.columns:
            raise KeyError(f"'{cluster_key}' not found in adata.obs.")
        dfc = df.assign(cluster=adata.obs[cluster_key]).dropna(subset=["cluster"])
        parts.append(_summarize(dfc, "cluster", "cluster"))
    return pd.concat(parts, ignore_index=True)


def clone_fate_table(adata, *, clone_key: str = "clone_id", time_key: str = "timepoint",
                     state_key: str = "state") -> pd.DataFrame:
    """Cell counts per (clone, timepoint, state) for clones with >= 2 cells."""
    for col in (clone_key, time_key, state_key):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")
    df = adata.obs[[clone_key, time_key, state_key]].dropna()
    sizes = df.groupby(clone_key).size()
    df = df[df[clone_key].isin(sizes[sizes >= 2].index)]
    out = df.groupby([clone_key, time_key, state_key], observed=True).size().rename("n_cells").reset_index()
    return out.sort_values([clone_key, time_key, state_key]).reset_index(drop=True)


def community_transition(adata, *, time_key: str, cluster_key: str = "clone_cluster",
                         clone_key: str = "clone_id") -> pd.DataFrame:
    """Cluster-membership transition matrix for multi-timepoint clones.

    Each clone's modal ``cluster_key`` is taken per timepoint and transitions
    are counted between consecutive sorted timepoints.

    .. warning::
        If ``cluster_key`` is a *clone-level* label (one value per clone, as
        produced by :func:`threadfin.clonotype_recluster` or
        :func:`threadfin.tl.find_programmes`), every snapshot of a clone has
        the same label and this matrix is diagonal by construction. Use
        :func:`threadfin.tl.clonal_memory`, which profiles each time point
        separately, to measure whether clones keep their state.
    """
    for col in (clone_key, time_key, cluster_key):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")
    df = adata.obs[[clone_key, time_key, cluster_key]].dropna()
    if len(df) and df.groupby(clone_key, observed=True)[cluster_key].nunique().max() <= 1:
        warnings.warn(
            f"'{cluster_key}' is constant within every clone, so transitions are diagonal by "
            "construction; use threadfin.tl.clonal_memory() instead.", stacklevel=2,
        )
    mode = df.groupby([clone_key, time_key], observed=True)[cluster_key].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    tab = mode.unstack(time_key)
    cols = list(tab.columns)
    as_num = pd.to_numeric(pd.Index(cols), errors="coerce")
    if len(cols) and pd.notna(np.asarray(as_num)).all():
        ordered = [cols[i] for i in np.argsort(np.asarray(as_num, dtype=float), kind="stable")]
    else:
        import re

        m = [re.match(r"^(.*?)(\d+)$", str(c)) for c in cols]
        if all(m):
            ordered = [c for _, c in sorted(zip(m, cols), key=lambda t: (t[0].group(1), int(t[0].group(2))))]
        else:
            ordered = sorted(cols, key=str)
    tab = tab[ordered]
    tab = tab[tab.notna().sum(axis=1) >= 2]
    counts: dict = {}
    clusters: set = set()
    for a, b in zip(ordered[:-1], ordered[1:]):
        pair = tab[[a, b]].dropna()
        for f_, t_ in zip(pair[a], pair[b]):
            counts[(f_, t_)] = counts.get((f_, t_), 0) + 1
            clusters.update((f_, t_))
    labels = sorted(clusters, key=str)
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for (f_, t_), n in counts.items():
        mat.loc[f_, t_] = n
    mat.index.name = "from_cluster"
    mat.columns.name = "to_cluster"
    return mat


def shm_gradient_test(adata, *, cluster_key: str = "clone_cluster", order: list | None = None,
                      mut_col: str | None = "bcr_mu_count") -> dict:
    """Kruskal-Wallis test of somatic hypermutation across communities (cell level).

    Kept for backward compatibility. Cells of one clone are not independent;
    for inference use :func:`threadfin.tl.association_test` on a clone-level
    SHM column, which counts clones and permutes within donors.
    """
    from scipy.stats import kruskal, spearmanr

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"'{cluster_key}' not found in adata.obs.")
    df = pd.DataFrame({"community": adata.obs[cluster_key], "shm": _cell_shm(adata, mut_col)}).dropna()
    groups = [g["shm"].to_numpy() for _, g in df.groupby("community", observed=True)]
    if len(groups) < 2:
        raise ValueError(f"Need at least two communities with SHM values in '{cluster_key}'.")
    stat, pval = kruskal(*groups)
    medians = df.groupby("community", observed=True)["shm"].median()
    rho = sp = None
    if order is not None:
        missing = [c for c in order if c not in medians.index]
        if missing:
            raise ValueError(f"order lists communities absent from '{cluster_key}': {missing}")
        r = spearmanr(medians.loc[list(order)].to_numpy(), np.arange(len(order)))
        rho, sp = float(r.statistic), float(r.pvalue)
    return {"kruskal_H": float(stat), "kruskal_p": float(pval), "spearman_rho": rho,
            "spearman_p": sp, "community_medians": medians}


def public_clone_summary(adata, *, clone_key: str = "clone_id", donor_key: str,
                         cluster_key: str | None = "clone_cluster") -> pd.DataFrame:
    """Clone ids observed in >= 2 donors.

    Note: true B-cell clones cannot be shared between people. With donor-aware
    clone ids (:func:`define_clones`) this table should be empty; entries
    indicate either *exact-sequence* clonotype keys (convergent "public"
    receptors) or clone ids that were not namespaced by donor.
    """
    cols = [clone_key, donor_key]
    for col in (clone_key, donor_key):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")
    if cluster_key is not None:
        if cluster_key not in adata.obs.columns:
            raise KeyError(f"'{cluster_key}' not found in adata.obs.")
        cols.append(cluster_key)
    df = adata.obs[cols].dropna()
    rows = []
    for clone, grp in df.groupby(clone_key, observed=True):
        donors = sorted(grp[donor_key].unique(), key=str)
        if len(donors) < 2:
            continue
        row = {clone_key: clone, "n_donors": int(len(donors)), "n_cells": int(len(grp)), "donor_list": list(donors)}
        if cluster_key is not None:
            freq = grp[cluster_key].value_counts(normalize=True)
            row["dominant_cluster"] = freq.index[0]
            row["state_purity"] = float(freq.iloc[0])
        rows.append(row)
    out_cols = [clone_key, "n_donors", "n_cells", "donor_list"]
    if cluster_key is not None:
        out_cols += ["dominant_cluster", "state_purity"]
    out = pd.DataFrame(rows, columns=out_cols)
    return out.sort_values(["n_donors", "n_cells"], ascending=[False, False], kind="stable").reset_index(drop=True)
