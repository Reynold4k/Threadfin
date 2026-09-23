"""STARTRAC-style clonal dynamics indices: migration, transition, expansion.

Clean clone-level re-implementation of the STARTRAC indices (Zhang et al.,
*Nature* 2018, doi:10.1038/s41586-018-0694-x) for paired scRNA-seq +
scBCR-seq data: pairwise group migration (STARTRAC-migr), state transition
(STARTRAC-tran) and per-group clonal expansion (STARTRAC-expa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(adata, cols) -> None:
    for col in cols:
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")


def clone_distribution(
    adata,
    *,
    clone_key: str = "clone_id",
    group_key: str,
    min_clone_size: int = 2,
) -> pd.DataFrame:
    """Clones x groups matrix of within-clone cell fractions.

    Each row is a clone; entry ``p[c, g]`` is the fraction of clone ``c``'s
    cells found in group ``g``, so every row sums to 1. Cells with a missing
    ``clone_key`` or ``group_key`` value are excluded. Clones with fewer than
    ``min_clone_size`` cells are dropped: fractions estimated from one or two
    cells are dominated by sampling noise and would otherwise inflate the
    downstream migration/transition indices.

    Returns
    -------
    DataFrame with index = ``clone_key`` values, columns = ``group_key``
    values, values = within-clone fractions (rows sum to 1).
    """
    _require_columns(adata, (clone_key, group_key))

    df = adata.obs[[clone_key, group_key]].dropna()
    sizes = df.groupby(clone_key, observed=True).size()
    keep = sizes[sizes >= min_clone_size].index
    df = df[df[clone_key].isin(keep)]

    counts = pd.crosstab(df[clone_key], df[group_key])
    dist = counts.div(counts.sum(axis=1), axis=0).astype(float)
    dist.index.name = clone_key
    dist.columns.name = group_key
    return dist


def _pairwise_index(adata, clone_key: str, split_key: str, min_clone_size: int) -> pd.DataFrame:
    dist = clone_distribution(
        adata, clone_key=clone_key, group_key=split_key, min_clone_size=min_clone_size
    )
    return dist.T.dot(dist)


def migration_index(
    adata,
    *,
    clone_key: str = "clone_id",
    group_key: str,
    min_clone_size: int = 2,
) -> pd.DataFrame:
    """Pairwise group migration index (STARTRAC-migr analogue).

    .. math::

        \\mathrm{migr}(g_1, g_2) = \\sum_c p_{c,g_1} \\cdot p_{c,g_2}

    where ``p[c, g]`` is the fraction of clone ``c``'s cells in group ``g``
    (see :func:`clone_distribution`). Re-implementation of the STARTRAC-migr
    index (Zhang et al., *Nature* 2018) at clone level.

    Interpretation caveats: the index conflates *clone sharing* (clones
    present in both groups) with *clone size* — a single large clone spread
    across two groups raises migr(g1, g2) more than many small shared clones.
    It is also sensitive to ``min_clone_size`` and to uneven cell sampling
    between groups; compare values only within one dataset, not across
    cohorts.

    Returns
    -------
    Symmetric square DataFrame (rows and columns = ``group_key`` values);
    the diagonal is ``sum_c p[c, g]**2``, the within-group clone
    concentration.
    """
    return _pairwise_index(adata, clone_key, group_key, min_clone_size)


def transition_index(
    adata,
    *,
    clone_key: str = "clone_id",
    state_key: str,
    min_clone_size: int = 2,
) -> pd.DataFrame:
    """Pairwise state transition index (STARTRAC-tran analogue).

    Same formula as :func:`migration_index` applied to cell states instead of
    groups:

    .. math::

        \\mathrm{tran}(s_1, s_2) = \\sum_c p_{c,s_1} \\cdot p_{c,s_2}

    where ``p[c, s]`` is the fraction of clone ``c``'s cells in state ``s``.
    High off-diagonal values indicate clones spanning both states (STARTRAC,
    Zhang et al., *Nature* 2018). The same caveats apply: the index mixes
    clone sharing with clone size and is sensitive to ``min_clone_size``.

    Returns
    -------
    Symmetric square DataFrame (rows and columns = ``state_key`` values).
    """
    return _pairwise_index(adata, clone_key, state_key, min_clone_size)


def expansion_index(
    adata,
    *,
    clone_key: str = "clone_id",
    group_key: str,
) -> pd.Series:
    """Per-group clonal expansion index (STARTRAC-expa analogue).

    .. math::

        \\mathrm{expa}(g) = 1 - \\frac{H_g}{\\log n_g}

    where ``H_g`` is the Shannon entropy (natural log) of the clone-size
    distribution within group ``g`` and ``n_g`` is the number of distinct
    clones in ``g`` (STARTRAC, Zhang et al., *Nature* 2018). Ranges from 0
    (all clones equal size, no expansion) towards 1 (one clone dominates).
    Groups with a single clone are defined as 0 (the normalized entropy is
    undefined for ``n_g == 1``). Cells missing ``clone_key`` or ``group_key``
    are excluded; no minimum clone size is applied, matching the original
    definition.

    Returns
    -------
    Series indexed by ``group_key`` values, name ``"expansion_index"``.
    """
    _require_columns(adata, (clone_key, group_key))

    df = adata.obs[[clone_key, group_key]].dropna()
    out: dict = {}
    for group, sub in df.groupby(group_key, observed=True):
        sizes = sub.groupby(clone_key, observed=True).size().to_numpy(dtype=float)
        n_clones = len(sizes)
        if n_clones <= 1:
            out[group] = 0.0
            continue
        p = sizes / sizes.sum()
        h = float(-(p * np.log(p)).sum())
        out[group] = 1.0 - h / np.log(n_clones)

    s = pd.Series(out, name="expansion_index", dtype=float)
    s.index.name = group_key
    return s
