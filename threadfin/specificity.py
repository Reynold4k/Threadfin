"""Antigen-specificity discovery: label mapping and reference-database matching.

Two routes to per-clone specificity annotation:

* :func:`annotate_specificity` — map author/experimental labels onto clones,
  or match clones against an antibody database (e.g. CoV-AbDab) by heavy V
  gene + CDRH3 similarity.
* :func:`specificity_enrichment` — Fisher-exact enrichment of specificities
  in clone communities with BH FDR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sequence import _gene, cdr3_similarity

# CoV-AbDab (https://opig.stats.ox.ac.uk/webapps/covabdab/) column names.
# Users with differently-named tables pass ``ref_cols`` overrides; any key
# not given falls back to these defaults.
_REF_COLS = {
    "v": "Heavy V Gene",
    "j": "Heavy J Gene",
    "cdr3": "CDRH3",
    "specificity": "Binds to",
    "name": "Name",
}

# sequence.cdr3_similarity(mode="auto") returns 0.0 beyond this length gap;
# used to pre-filter reference entries before any alignment is attempted.
_MAX_LEN_DIFF = 4


def _norm_gene(call, prefix: str) -> str:
    """Normalize a gene call to allele-free form.

    Handles multi-calls ('IGHV3-23*01,IGHV3-23*02' -> 'IGHV3-23'), the
    bare '3-23' style seen in some CoV-AbDab rows (bare '3-23' with
    ``prefix='IGHV'`` -> 'IGHV3-23'), and CoV-AbDab's species suffix
    ('IGHV4-31 (Human)' -> 'IGHV4-31').
    """
    g = _gene(str(call).split("(")[0].strip()).upper()
    if g and not g.startswith("IG") and g[0].isdigit():
        g = prefix + g
    return g


def _clean_seq(seq) -> str | None:
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper().replace("-", "").replace(".", "")
    return s or None


def _clean_specificity(val) -> str | None:
    """First antigen of a (possibly ';'-separated) 'Binds to' field."""
    if not isinstance(val, str):
        return None
    first = val.split(";")[0].strip()
    return first or None


def _modal(series: pd.Series):
    """Deterministic modal value: lexicographically smallest mode."""
    s = series.dropna()
    if len(s) == 0:
        return None
    m = s.astype(str).mode()
    return m.sort_values(kind="stable").iloc[0] if len(m) else None


def _check_obs(adata, cols, context: str) -> None:
    for col in cols:
        if col not in adata.obs.columns:
            raise KeyError(
                f"'{col}' not found in adata.obs (needed for {context}). "
                "Attach BCR annotations first (threadfin.attach_bcr) or pass "
                "the matching column name explicitly."
            )


def _clone_table(adata, clone_key, v_col, j_col, cdr3_col) -> pd.DataFrame:
    """One row per clone: modal (v, j, cdr3) and cell count."""
    df = adata.obs[[clone_key, v_col, j_col, cdr3_col]].dropna(subset=[clone_key])
    rows = []
    for clone, grp in df.groupby(clone_key, observed=True, sort=True):
        rows.append(
            {
                "clone": clone,
                "n_cells": int(len(grp)),
                "v": _norm_gene(_modal(grp[v_col]), "IGHV"),
                "j": _norm_gene(_modal(grp[j_col]), "IGHJ"),
                "cdr3": _clean_seq(_modal(grp[cdr3_col])),
            }
        )
    return pd.DataFrame(rows)


def annotate_specificity(
    adata,
    *,
    clone_key: str = "clone_id",
    labels: pd.DataFrame | None = None,
    label_col: str | None = None,
    reference=None,
    ref_cols: dict | None = None,
    cdr3_col: str = "bcr_cdr3",
    v_col: str = "bcr_v_call",
    j_col: str = "bcr_j_call",
    identity: float = 0.85,
    key_added: str = "specificity",
):
    """Annotate clones with antigen specificity.

    Exactly one of ``labels`` / ``reference`` must be given.

    **Label mode** (``labels`` DataFrame with columns ``clone_key`` and
    ``label_col``): per-clone specificity labels (e.g. LIBRA-seq
    ``s_pos_clone`` calls or author-validated mAb panels) are mapped onto
    ``obs[key_added]`` by clone identity. Cells whose clone has no label get
    NA; on duplicate label rows for one clone the first is kept.

    **Reference-matching mode** (``reference``: path or DataFrame of an
    antibody database such as CoV-AbDab): per clone, the modal heavy-chain
    (v_call, j_call, cdr3_aa) is taken over its cells; reference entries are
    bucketed by normalized V gene (no all-pairs comparison) and, within the
    clone's bucket, pre-filtered to CDR3 length difference <= 4 before
    :func:`threadfin.sequence.cdr3_similarity` is computed. The best hit with
    similarity >= ``identity`` assigns its specificity; ties are broken by
    shared J gene, then reference order. A per-clone match table is stored in
    ``adata.uns[key_added + "_clones"]``.

    Null model / limitations: a reference hit is *evidence*, not proof, of
    specificity — matching uses the heavy chain only (no light chain, no SHM
    profile) and convergent public CDR3s can collide across antigens. The
    ``identity`` threshold is a heuristic, not a calibrated error rate;
    validate key hits experimentally (e.g. with the clone's recombinant mAb).
    Label mode carries no model at all — labels are trusted as given.

    Parameters
    ----------
    adata
        AnnData with ``obs[clone_key]`` (and ``obs[v_col]``, ``obs[j_col]``,
        ``obs[cdr3_col]`` in reference mode).
    clone_key
        Column of ``obs`` holding clone identities.
    labels
        DataFrame with columns ``clone_key`` and ``label_col``.
    label_col
        Column of ``labels`` holding the specificity label.
    reference
        Path to a CSV or a DataFrame of an antibody database.
    ref_cols
        Explicit mapping of logical names (``"v"``, ``"j"``, ``"cdr3"``,
        ``"specificity"``, ``"name"``) to reference columns. Unspecified
        keys default to CoV-AbDab column names.
    cdr3_col, v_col, j_col
        Columns of ``obs`` with heavy-chain CDR3 amino acids and V/J calls.
    identity
        Minimum CDR3 similarity (0-1) for a reference hit.
    key_added
        Name of the added ``obs`` column; the reference-mode clone match
        table goes to ``uns[key_added + "_clones"]``.

    Returns
    -------
    The modified ``adata`` (in place and returned), with
    ``obs[key_added]`` (categorical; NA where unannotated). In reference
    mode, ``adata.uns[key_added + "_clones"]`` holds a DataFrame with
    columns ``[clone_key, n_cells, best_score, hit_name, specificity]``,
    one row per matched clone.
    """
    if (labels is None) == (reference is None):
        raise ValueError("Pass exactly one of `labels` or `reference`.")
    _check_obs(adata, [clone_key], "specificity annotation")

    if labels is not None:
        if label_col is None:
            raise ValueError("Label mode requires `label_col`.")
        for col in (clone_key, label_col):
            if col not in labels.columns:
                raise KeyError(
                    f"'{col}' not found in the labels table. Label mode needs "
                    f"a DataFrame with columns {clone_key!r} and {label_col!r}."
                )
        lab = labels[[clone_key, label_col]].dropna(subset=[clone_key])
        lab = lab.drop_duplicates(subset=[clone_key], keep="first")
        mapping = dict(zip(lab[clone_key], lab[label_col]))
        adata.obs[key_added] = adata.obs[clone_key].map(mapping).astype("category")
        return adata

    # ---- reference-matching mode ----
    if not (0.0 < identity <= 1.0):
        raise ValueError("`identity` must be in (0, 1].")
    _check_obs(adata, [v_col, j_col, cdr3_col], "reference matching")

    ref = pd.read_csv(reference) if isinstance(reference, (str, Path)) else reference
    if not isinstance(ref, pd.DataFrame):
        raise TypeError("`reference` must be a path to a CSV or a pandas DataFrame.")
    cols = {**_REF_COLS, **(ref_cols or {})}
    for logical in ("v", "cdr3", "specificity"):
        if cols[logical] not in ref.columns:
            raise KeyError(
                f"Reference table has no column {cols[logical]!r} (logical name "
                f"{logical!r}). Available columns: {list(ref.columns)}. Pass "
                "`ref_cols` to map logical names onto your table's columns."
            )
    name_col = cols.get("name")
    if name_col is None or name_col not in ref.columns:
        ref = ref.assign(**{"_ref_name": [f"ref_{i}" for i in range(len(ref))]})
        name_col = "_ref_name"

    # bucket reference entries by normalized heavy V gene; within a bucket,
    # group by CDR3 length so only near-equal lengths are ever aligned
    v_arr = ref[cols["v"]].tolist()
    cdr3_arr = ref[cols["cdr3"]].tolist()
    spec_arr = ref[cols["specificity"]].tolist()
    name_arr = ref[name_col].tolist()
    j_col_ref = cols.get("j")
    j_arr = ref[j_col_ref].tolist() if j_col_ref in ref.columns else [None] * len(ref)
    buckets: dict[str, dict[int, list[tuple]]] = {}
    for order in range(len(ref)):
        vg = _norm_gene(v_arr[order], "IGHV")
        cdr3 = _clean_seq(cdr3_arr[order])
        spec = _clean_specificity(spec_arr[order])
        if not vg or cdr3 is None or spec is None:
            continue
        entry = (cdr3, str(name_arr[order]), spec, _norm_gene(j_arr[order], "IGHJ"), order)
        buckets.setdefault(vg, {}).setdefault(len(cdr3), []).append(entry)

    clones = _clone_table(adata, clone_key, v_col, j_col, cdr3_col)
    matches: dict = {}
    rows = []
    for rec in clones.itertuples(index=False):
        if not rec.v or rec.cdr3 is None:
            continue
        by_len = buckets.get(rec.v)
        if not by_len:
            continue
        best = None  # (score, same_j, -order, entry)
        for length, entries in by_len.items():
            if abs(length - len(rec.cdr3)) > _MAX_LEN_DIFF:
                continue
            for cdr3_r, name, spec, j_r, order in entries:
                score = cdr3_similarity(rec.cdr3, cdr3_r)
                if score < identity:
                    continue
                same_j = bool(rec.j) and bool(j_r) and rec.j == j_r
                cand = (score, same_j, -order, (name, spec))
                if best is None or cand[:3] > best[:3]:
                    best = cand
        if best is not None:
            name, spec = best[3]
            matches[rec.clone] = spec
            rows.append(
                {
                    clone_key: rec.clone,
                    "n_cells": rec.n_cells,
                    "best_score": float(best[0]),
                    "hit_name": name,
                    "specificity": spec,
                }
            )

    adata.obs[key_added] = adata.obs[clone_key].map(matches).astype("category")
    adata.uns[key_added + "_clones"] = pd.DataFrame(
        rows, columns=[clone_key, "n_cells", "best_score", "hit_name", "specificity"]
    )
    return adata


def specificity_enrichment(
    adata,
    *,
    cluster_key: str = "clone_cluster",
    specificity_key: str = "specificity",
) -> pd.DataFrame:
    """Fisher-exact enrichment of specificities in clone communities.

    For every (community, specificity) pair, tests whether cells annotated
    with that specificity are over-represented in the community
    (one-sided Fisher exact, ``alternative="greater"``), with
    Benjamini-Hochberg FDR over all tested pairs.

    Null model / limitations: the null is independence between community and
    specificity *at cell level*. Clonal expansion violates cell independence
    (cells of one clone share both), so p-values are anti-conservative for
    expanded clones — treat them as a ranking device and confirm key
    enrichments with a clone-level sensitivity analysis (e.g. counting
    clones instead of cells).

    Parameters
    ----------
    adata
        AnnData with ``obs[cluster_key]`` and ``obs[specificity_key]``
        (the latter typically from :func:`annotate_specificity`).
    cluster_key
        Column of ``obs`` holding clone-community labels.
    specificity_key
        Column of ``obs`` holding specificity annotations.

    Returns
    -------
    Tidy DataFrame, one row per (community, specificity) pair, sorted by
    p-value, with columns ``[cluster_key, specificity, n_cells, fraction,
    odds_ratio, pvalue, fdr]``. ``n_cells`` is the number of cells with
    that specificity in the community; ``fraction`` is that count over all
    specificity-annotated cells of the community.
    """
    from scipy.stats import fisher_exact

    _check_obs(adata, [cluster_key, specificity_key], "specificity enrichment")

    obs = adata.obs[[cluster_key, specificity_key]].dropna()
    if len(obs) == 0:
        raise ValueError(
            f"No cells with both '{cluster_key}' and '{specificity_key}' present."
        )
    obs = obs.astype(str)
    clusters = pd.Categorical(obs[cluster_key]).categories
    specs = pd.Categorical(obs[specificity_key]).categories

    rows = []
    n = len(obs)
    for cl in clusters:
        in_cl = (obs[cluster_key] == cl).to_numpy()
        n_cl = int(in_cl.sum())
        for sp in specs:
            in_sp = (obs[specificity_key] == sp).to_numpy()
            a = int((in_cl & in_sp).sum())
            b = n_cl - a
            c = int(in_sp.sum()) - a
            d = n - a - b - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    cluster_key: cl,
                    "specificity": sp,
                    "n_cells": a,
                    "fraction": float(a / n_cl) if n_cl else 0.0,
                    "odds_ratio": float(odds),
                    "pvalue": float(p),
                }
            )

    out = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    # Benjamini-Hochberg FDR (monotone in the sorted p-values)
    m = len(out)
    ranks = np.arange(1, m + 1)
    fdr = out["pvalue"] * m / ranks
    out["fdr"] = np.minimum.accumulate(fdr.to_numpy()[::-1])[::-1].clip(max=1.0)
    return out
