"""Internal helpers shared across Threadfin modules (not public API)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

UNS_KEY = "threadfin"


def log(msg: str, verbose: bool = True) -> None:
    """Print a progress message with the package prefix."""
    if verbose:
        print(f"[threadfin] {msg}", flush=True)


def get_uns(adata) -> dict:
    """The Threadfin results dict in ``adata.uns`` (created if absent)."""
    return adata.uns.setdefault(UNS_KEY, {})


def require_obs(adata, *cols: str, context: str | None = None) -> None:
    """Raise a helpful ``KeyError`` if any ``obs`` column is missing."""
    for col in cols:
        if col is not None and col not in adata.obs.columns:
            hint = f" (needed for {context})" if context else ""
            raise KeyError(f"'{col}' not found in adata.obs{hint}.")


def require_complete_groups(adata, *cols: str, context: str | None = None) -> None:
    """Require grouping columns to have a non-empty label for every cell."""
    require_obs(adata, *cols, context=context)
    for col in cols:
        if col is None:
            continue
        values = adata.obs[col]
        blank = values.astype("string").str.strip().eq("").fillna(False)
        bad = values.isna() | blank
        if bad.any():
            hint = f" for {context}" if context else ""
            raise ValueError(
                f"Grouping column '{col}' has {int(bad.sum())} missing or blank value(s){hint}. "
                "Every cell needs a donor/sample/batch label; fill or remove those cells before analysis."
            )


def require_positive_int(value, name: str) -> None:
    """Require a positive integer, rejecting bools and truncating floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def get_basis(adata, basis: str, dtype=np.float64) -> np.ndarray:
    """Dense cell x feature matrix from ``obsm[basis]`` (or ``X`` / a layer).

    ``basis`` may be an ``obsm`` key, ``"X"``, or ``"layer:<name>"``.
    """
    if basis == "X":
        x = adata.X
    elif basis.startswith("layer:"):
        name = basis.split(":", 1)[1]
        if name not in adata.layers:
            raise KeyError(f"layer '{name}' not found in adata.layers.")
        x = adata.layers[name]
    else:
        if basis not in adata.obsm:
            raise KeyError(
                f"'{basis}' not found in adata.obsm. Compute an embedding first "
                "(e.g. threadfin.pp.prepare_embedding or sc.tl.pca)."
            )
        x = adata.obsm[basis]
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 2:
        raise ValueError(
            f"'{basis}' must be a two-dimensional cell-by-feature matrix; got an array with {x.ndim} dimensions."
        )
    if x.shape[0] != adata.n_obs:
        raise ValueError(
            f"'{basis}' has {x.shape[0]} rows but adata has {adata.n_obs} cells. "
            "Provide one embedding row per cell."
        )
    if x.shape[1] == 0:
        raise ValueError(f"'{basis}' has no features. Compute or provide a non-empty embedding.")
    if not np.isfinite(x).all():
        raise ValueError(f"'{basis}' contains NaN or infinite values. Recompute or clean the embedding first.")
    return x


def codes(values: pd.Series | np.ndarray) -> tuple[np.ndarray, pd.Index]:
    """Integer codes (-1 for missing) and the matching category index."""
    cat = pd.Categorical(pd.Series(values).astype("object").where(pd.notna(values)))
    return cat.codes.astype(np.int64), cat.categories


def group_means(x: np.ndarray, group: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group means and sizes of rows of ``x`` (rows with group < 0 ignored)."""
    ok = group >= 0
    sums = np.zeros((n_groups, x.shape[1]))
    np.add.at(sums, group[ok], x[ok])
    n = np.bincount(group[ok], minlength=n_groups).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / n[:, None]
    return means, n


def indicator(group: np.ndarray, n_groups: int, weights: np.ndarray | None = None) -> sp.csr_matrix:
    """Sparse ``(n_groups, n_rows)`` membership matrix (rows with group < 0 dropped)."""
    ok = np.flatnonzero(group >= 0)
    w = np.ones(ok.size) if weights is None else np.asarray(weights, dtype=float)[ok]
    return sp.csr_matrix((w, (group[ok], ok)), shape=(n_groups, group.size))


def permute_within(labels: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Permute ``labels`` among entries sharing the same stratum.

    Entries with a negative label (missing) keep their position, so the
    permutation preserves exactly which cells/clones are labelled within
    each stratum.
    """
    out = labels.copy()
    valid = labels >= 0 if np.issubdtype(labels.dtype, np.integer) else pd.notna(labels)
    if strata is None:
        idx = np.flatnonzero(valid)
        out[idx] = labels[rng.permutation(idx)]
        return out
    strata = np.asarray(strata)
    order = np.flatnonzero(valid)
    s = strata[order]
    for key in np.unique(s):
        idx = order[s == key]
        if idx.size > 1:
            out[idx] = labels[rng.permutation(idx)]
    return out


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def perm_pvalue(observed: float, null: np.ndarray, alternative: str = "greater") -> float:
    """Permutation p-value with the +1 correction (Phipson & Smyth 2010)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        k = int((null >= observed).sum())
    elif alternative == "less":
        k = int((null <= observed).sum())
    elif alternative == "two-sided":
        center = float(np.median(null))
        k = int((np.abs(null - center) >= abs(observed - center)).sum())
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + k) / (null.size + 1)


def modal(series: pd.Series):
    """Deterministic mode (ties broken by sort order); ``None`` if empty."""
    s = series.dropna()
    if s.empty:
        return None
    vc = s.astype(str).value_counts()
    top = vc[vc == vc.iloc[0]].index
    return sorted(top)[0]


def stable_categorical(values, prefix: str) -> pd.Categorical:
    """Relabel integer cluster ids as ``<prefix>1..K`` ordered by size."""
    s = pd.Series(values)
    sizes = s.value_counts()
    mapping = {lab: f"{prefix}{i + 1}" for i, lab in enumerate(sizes.index)}
    cats = [mapping[lab] for lab in sizes.index]
    return pd.Categorical(s.map(mapping), categories=cats)
