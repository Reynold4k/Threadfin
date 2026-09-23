"""Clone-level annotation summaries: isotype, SHM, fate tracking, transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

_ISOTYPES = ("IGHM", "IGHD", "IGHG", "IGHA", "IGHE")


def _isotype_class(c_call):
    """First call before ',' collapsed to the heavy-chain class (IGHG1->IGHG)."""
    if c_call is None or (isinstance(c_call, float) and np.isnan(c_call)) or c_call is pd.NA:
        return None
    first = str(c_call).split(",")[0].strip()
    for cls in _ISOTYPES:
        if first.startswith(cls):
            return cls
    return first or None


def clone_isotype_summary(adata, *, cluster_key: str | None = "clone_cluster") -> pd.DataFrame:
    """Per-group heavy-chain isotype composition.

    Fractions are over cells with a non-missing ``bcr_c_call`` in the group.
    With ``cluster_key=None`` the summary is per ``clone_id`` instead.

    Returns
    -------
    Tidy DataFrame with columns ``[group, isotype, fraction, n_cells]``.
    """
    if "bcr_c_call" not in adata.obs.columns:
        raise KeyError(
            "'bcr_c_call' not found in adata.obs. Attach BCR annotations "
            "first (threadfin.attach_bcr)."
        )
    group_col = cluster_key if cluster_key is not None else "clone_id"
    if group_col not in adata.obs.columns:
        raise KeyError(f"'{group_col}' not found in adata.obs.")

    df = pd.DataFrame(
        {
            "group": adata.obs[group_col].astype("string"),
            "isotype": adata.obs["bcr_c_call"].map(_isotype_class),
        }
    ).dropna()
    counts = df.groupby(["group", "isotype"]).size().rename("n_cells")
    totals = df.groupby("group").size()
    out = counts.reset_index()
    out["fraction"] = out["n_cells"] / out["group"].map(totals)
    out = out.rename(columns={"group": group_col})
    out = out[[group_col, "isotype", "fraction", "n_cells"]]
    return out.sort_values([group_col, "isotype"]).reset_index(drop=True)


def clone_shm_summary(
    adata, *, mut_col: str | None = "bcr_mu_count", cluster_key: str | None = None
) -> pd.DataFrame:
    """Mean/median somatic hypermutation per clone (and per cluster).

    Uses ``obs[mut_col]``; if absent, falls back to ``obs['bcr_v_identity']``
    with ``100 - identity`` as a mutation proxy.

    Returns
    -------
    Tidy DataFrame with columns ``[level, group, shm_mean, shm_median,
    n_cells]``; ``level`` is ``"clone"`` (and ``"cluster"`` if
    ``cluster_key`` is given).
    """
    if "clone_id" not in adata.obs.columns:
        raise KeyError("'clone_id' not found in adata.obs.")
    if mut_col is not None and mut_col in adata.obs.columns:
        shm = pd.to_numeric(adata.obs[mut_col], errors="coerce")
    elif "bcr_v_identity" in adata.obs.columns:
        shm = 100.0 - pd.to_numeric(adata.obs["bcr_v_identity"], errors="coerce")
    else:
        raise ValueError(
            f"SHM summary needs '{mut_col}' or 'bcr_v_identity' in adata.obs "
            "(an AIRR mutation count or V-gene identity column)."
        )

    df = pd.DataFrame({"clone_id": adata.obs["clone_id"], "shm": shm})
    df = df.dropna(subset=["clone_id", "shm"])

    def _summarize(frame: pd.DataFrame, level: str, name: str) -> pd.DataFrame:
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


def clone_fate_table(
    adata,
    *,
    clone_key: str = "clone_id",
    time_key: str = "timepoint",
    state_key: str = "state",
) -> pd.DataFrame:
    """Cell counts per (clone, timepoint, state) for clones with >= 2 cells.

    Returns
    -------
    Long-format DataFrame with columns ``[clone_key, time_key, state_key,
    n_cells]``.
    """
    for col in (clone_key, time_key, state_key):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")

    df = adata.obs[[clone_key, time_key, state_key]].dropna()
    sizes = df.groupby(clone_key).size()
    keep = sizes[sizes >= 2].index
    df = df[df[clone_key].isin(keep)]
    out = (
        df.groupby([clone_key, time_key, state_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    return out.sort_values([clone_key, time_key, state_key]).reset_index(drop=True)


def community_transition(
    adata,
    *,
    time_key: str,
    cluster_key: str = "clone_cluster",
    clone_key: str = "clone_id",
) -> pd.DataFrame:
    """Cluster-membership transition matrix for multi-timepoint clones.

    Each clone's modal ``cluster_key`` is taken per timepoint; transitions
    are counted between consecutive sorted timepoints for clones observed at
    >= 2 distinct timepoints.

    Returns
    -------
    Square DataFrame (rows = from-cluster, columns = to-cluster, values =
    clone counts).
    """
    for col in (clone_key, time_key, cluster_key):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")

    df = adata.obs[[clone_key, time_key, cluster_key]].dropna()
    mode = (
        df.groupby([clone_key, time_key], observed=True)[cluster_key]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    )
    tab = mode.unstack(time_key)

    cols = list(tab.columns)
    as_num = pd.to_numeric(pd.Index(cols), errors="coerce")
    if len(cols) and pd.notna(np.asarray(as_num)).all():
        ordered = [cols[i] for i in np.argsort(np.asarray(as_num, dtype=float), kind="stable")]
    else:
        import re

        m = [re.match(r"^(.*?)(\d+)$", str(c)) for c in cols]
        if all(m):
            ordered = [c for _, c in sorted(zip(m, cols),
                                            key=lambda t: (t[0].group(1), int(t[0].group(2))))]
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
