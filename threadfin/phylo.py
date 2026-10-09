"""Lineage heritability: is a cell's state shared with its close relatives inside a B-cell clone?

Threadfin's clone profiles describe how clones differ from each other. This module asks the
complementary question inside clones: after removing everything a clone's cells share (the clone's
mean state in each sampling block), are cells that sit close together on the clone's somatic-
hypermutation lineage tree still more alike than distant relatives?

Model
-----
For one state feature ``y`` (a module score, a principal component, or a recorded cell label coded
0/1), cell ``i`` of clone ``c`` sampled in block ``b`` (mouse, sort gate, library...) is modelled as

    y_i = mu_{c,b} + x_i' beta + g_i + e_i,
    g ~ N(0, sigma_g^2 K),   e ~ N(0, sigma_e^2 I),

where ``mu_{c,b}`` is a free intercept for every clone x block *unit* (so clone-wide and block-wide
state, however large, cannot contribute), ``x_i`` are cell covariates (sequencing depth, the cell's
mutation depth on the tree, sample indicators...), and ``K`` is a kinship kernel built from the tree:

* ``"exponential"`` (default): ``K_ij = exp(-d_ij / l)``, ``d_ij`` the number of mutations on the
  tree path between the two cells' genotypes;
* ``"identical"``: ``K_ij = 1`` when the two cells carry the same receptor genotype;
* ``"identical_mutated"``: as ``"identical"``, but only for genotypes that carry at least one
  mutation (cells still identical to the germline need not be recent relatives);
* ``"brownian"``: ``K_ij`` = mutations shared from the root to the two cells' common ancestor,
  scaled by the mean root-to-tip depth (Brownian motion along the tree).

``h2 = sigma_g^2 / (sigma_g^2 + sigma_e^2)`` is the share of within-unit cell variation that is
structured by the lineage; for the exponential and identity kernels it is the correlation of two
cells with the same genotype after the unit mean is removed.

Why not a pairwise correlation of tree and state distances
----------------------------------------------------------
The distance between two cells on a tree is roughly the sum of their depths, so a pairwise
statistic mixes two different effects: a state that depends on how mutated a cell is (a fixed effect
of depth) and genuine resemblance of relatives (the random effect ``g``). Here depth is a covariate
and only ``g`` is tested.

Estimation and inference
------------------------
Every unit is projected onto the contrasts orthogonal to its intercept (Helmert basis ``P``) and
rotated into the eigenbasis of ``P' K P``. In that basis the covariance is diagonal,
``sigma_e^2 (1 + lambda d)``, so the restricted (REML) likelihood of ``lambda = h2 / (1 - h2)``,
pooled over thousands of small trees, is a sum over rows and is maximised by a 1-D search.
``h2`` comes with a profile-likelihood interval and an asymptotic boundary likelihood-ratio test
(``0.5 chi2_0 + 0.5 chi2_1``). **Both are likelihood-based and both have been measured to fail on real
repertoire data**: with per-unit heteroscedastic residuals the boundary LRT rejects 32-38% of the time at a
nominal 5%, and with heavy-tailed residuals 16-23%; the interval's coverage is 0.65-0.87 instead of 0.95.
Use ``p_lrt`` and ``h2_lo``/``h2_hi`` as rough diagnostics only, and quote the permutation p-value and a
bootstrap over clones (and over donors) for any reported interval. The primary test is the REML score
statistic
``Q = sum_i d_i r_i^2 / sum_i r_i^2`` of the null residuals ``r``, with a Freedman-Lane permutation
null: residuals of the null model are permuted among the cells of each unit, which keeps every
unit's tree, its covariates and its set of residual states, and destroys only which cell sits where.

:func:`lineage_power` plants a lineage component of known ``h2`` on the real trees, on top of a real
feature whose own lineage structure has first been destroyed by permutation, and reports bias, type I
error and power, so that a null result can be read as "``h2`` above x would have been detected".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import codes, log, require_positive_int

__all__ = [
    "KERNELS",
    "LineageForest",
    "LineageHeritability",
    "lineage_forest",
    "lineage_heritability",
    "lineage_power",
    "simulate_lineage_trait",
    # pairwise diagnostic, superseded by lineage_heritability (kept for reproducibility)
    "LineageSignal",
    "lineage_signal",
    "planted_calibration",
    "pair_table",
]

KERNELS = ("exponential", "identical", "identical_mutated", "brownian")


# =========================================================================== forest


@dataclass
class LineageForest:
    """Cells placed on per-clone lineage trees, grouped into clone x block units.

    Attributes
    ----------
    n_cells
        Number of rows of the data the forest indexes into.
    units
        One integer array of row indices per unit (all cells of one clone in one block that are
        placed on the clone's tree); every unit has at least two cells.
    dist
        Per unit, the matrix of tree-path distances (mutations) between its cells.
    shared
        Per unit, the matrix of mutations shared from the root to each pair's common ancestor
        (diagonal = the cell's depth).
    depth
        Per row, the root-to-tip depth of the cell's genotype (NaN for rows not in the forest).
    unit_clone, unit_block
        Labels of each unit.
    """

    n_cells: int
    units: list
    dist: list
    shared: list
    depth: np.ndarray
    unit_clone: list
    unit_block: list
    notes: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def rows(self) -> np.ndarray:
        """All row indices that belong to a unit."""
        return np.concatenate(self.units) if self.units else np.empty(0, dtype=np.int64)

    def summary(self) -> dict:
        sizes = np.array([u.size for u in self.units])
        return {"units": int(self.n_units), "cells": int(sizes.sum()) if sizes.size else 0,
                "clones": int(len(set(self.unit_clone))),
                "median_unit_size": float(np.median(sizes)) if sizes.size else float("nan"),
                "max_unit_size": int(sizes.max()) if sizes.size else 0,
                "identical_pairs": int(sum(((d == 0).sum() - d.shape[0]) // 2 for d in self.dist)),
                "pairs": int(sum(d.shape[0] * (d.shape[0] - 1) // 2 for d in self.dist)),
                "mean_depth": float(np.nanmean(self.depth[self.rows])) if self.units else float("nan"),
                **self.notes}

    def subset(self, keep) -> "LineageForest":
        """The forest restricted to rows where ``keep`` is True (units falling below 2 cells drop)."""
        keep = np.asarray(keep, dtype=bool)
        units, dist, shared, uc, ub = [], [], [], [], []
        for u, d, s, c, b in zip(self.units, self.dist, self.shared, self.unit_clone, self.unit_block):
            m = keep[u]
            if m.sum() >= 2:
                units.append(u[m])
                dist.append(d[np.ix_(m, m)])
                shared.append(s[np.ix_(m, m)])
                uc.append(c)
                ub.append(b)
        return LineageForest(self.n_cells, units, dist, shared, self.depth, uc, ub, dict(self.notes))

    def kernel(self, kind: str = "exponential", length_scale: float = 2.0) -> list:
        """Per-unit kinship matrices for ``kind`` (see module docstring)."""
        if kind not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kind!r}.")
        out = []
        if kind == "brownian":
            scale = float(np.nanmean(self.depth[self.rows])) if self.units else 1.0
            scale = scale if scale > 0 else 1.0
        for u, d, s in zip(self.units, self.dist, self.shared):
            if kind == "exponential":
                if not length_scale or length_scale <= 0:
                    raise ValueError("length_scale must be positive for the exponential kernel.")
                k = np.exp(-d / float(length_scale))
            elif kind == "identical":
                k = (d == 0).astype(float)
            elif kind == "identical_mutated":
                mutated = self.depth[u] > 0
                k = ((d == 0) & mutated[:, None] & mutated[None, :]).astype(float)
                np.fill_diagonal(k, 1.0)
            else:
                k = s / scale
            out.append(0.5 * (k + k.T))
        return out


def lineage_forest(clones, pairs, *, blocks=None, depth=None, min_cells: int = 2,
                   max_depth: float | None = None) -> LineageForest:
    """Group cells into clone x block units and collect their lineage-tree distances.

    Parameters
    ----------
    clones
        Clone label per data row (length ``n_cells``).
    pairs
        Long table of within-clone cell pairs with integer row indices ``i``, ``j``, the tree-path
        distance ``d`` (mutations between the two cells' genotypes) and, optionally, ``shared``
        (mutations from the root to their common ancestor). Pairs across clones are ignored.
    blocks
        Sampling block per row. Units are clone x block. Default: one block, i.e. units are whole clones.

        **Use the finest separately processed unit — the library or sort gate, not the animal.** A clone x
        gate interaction (one clone behaving differently in two sorted gates of the same animal, which
        physical sorting all but guarantees) is *not* removed by fitting the gate as a fixed effect. On real
        germinal-centre trees that omitted interaction, with no lineage component present at all, produced
        h2 = 0.09-0.16 and rejected 49-88% of the time. Blocking by animal when gates are nested inside
        animals is therefore not a valid test of lineage heritability.
    depth
        Root-to-tip depth per row. If omitted it is taken from ``pairs`` columns ``depth_i`` and
        ``depth_j``.
    min_cells
        Minimum cells for a unit to be kept (at least 2).
    max_depth
        Drop cells whose depth exceeds this (alignment or assignment artefacts).
    """
    clones = pd.Series(np.asarray(clones, dtype=object)).astype(str).to_numpy()
    n = clones.size
    p = pd.DataFrame({"i": np.asarray(pairs["i"], dtype=np.int64), "j": np.asarray(pairs["j"], dtype=np.int64),
                      "d": np.asarray(pairs["d"], dtype=float)})
    if "shared" in pairs:
        p["shared"] = np.asarray(pairs["shared"], dtype=float)
    if (p[["i", "j"]].to_numpy() < 0).any() or (p[["i", "j"]].to_numpy() >= n).any():
        raise ValueError("pair indices must be row indices into the data (0 .. n_cells-1).")
    if depth is None:
        if not {"depth_i", "depth_j"} <= set(pairs.keys() if hasattr(pairs, "keys") else []):
            raise ValueError("give depth, or pairs with depth_i and depth_j columns.")
        depth = np.full(n, np.nan)
        depth[np.asarray(pairs["i"], dtype=np.int64)] = np.asarray(pairs["depth_i"], dtype=float)
        depth[np.asarray(pairs["j"], dtype=np.int64)] = np.asarray(pairs["depth_j"], dtype=float)
    depth = np.asarray(depth, dtype=float).copy()
    if depth.size != n:
        raise ValueError("depth must have one value per data row.")
    block = np.zeros(n, dtype=object) if blocks is None else np.asarray(blocks, dtype=object).astype(str)
    notes = {"pairs_in": int(len(p))}
    p = p[(clones[p["i"].to_numpy()] == clones[p["j"].to_numpy()]) & np.isfinite(p["d"].to_numpy())]
    p = p[block[p["i"].to_numpy()] == block[p["j"].to_numpy()]]
    if max_depth is not None:
        bad = np.isfinite(depth) & (depth > max_depth)
        notes["cells_dropped_max_depth"] = int(bad.sum())
        p = p[~bad[p["i"].to_numpy()] & ~bad[p["j"].to_numpy()]]
    unit_key = pd.Series(clones).astype(str) + "\x1f" + pd.Series(block).astype(str)
    ucode, ulabels = codes(unit_key)
    p = p.assign(u=ucode[p["i"].to_numpy()])
    units, dist, shared, uc, ub = [], [], [], [], []
    incomplete = 0
    for u, grp in p.groupby("u", sort=True):
        cells = np.unique(np.r_[grp["i"].to_numpy(), grp["j"].to_numpy()])
        m = cells.size
        if m < max(2, min_cells):
            continue
        pos = {c: k for k, c in enumerate(cells)}
        dm = np.full((m, m), np.nan)
        sm = np.full((m, m), np.nan)
        a = np.array([pos[c] for c in grp["i"].to_numpy()])
        b = np.array([pos[c] for c in grp["j"].to_numpy()])
        dm[a, b] = dm[b, a] = grp["d"].to_numpy()
        np.fill_diagonal(dm, 0.0)
        if "shared" in grp:
            sm[a, b] = sm[b, a] = grp["shared"].to_numpy()
        # keep a complete sub-matrix: drop the cell with most missing distances until complete
        keep = np.ones(m, dtype=bool)
        while keep.sum() >= 2 and np.isnan(dm[np.ix_(keep, keep)]).any():
            miss = np.isnan(dm[np.ix_(keep, keep)]).sum(axis=1)
            idx = np.flatnonzero(keep)[np.argmax(miss)]
            keep[idx] = False
            incomplete += 1
        if keep.sum() < max(2, min_cells):
            continue
        cells, dm, sm = cells[keep], dm[np.ix_(keep, keep)], sm[np.ix_(keep, keep)]
        dc = depth[cells]
        if not np.isfinite(dc).all():
            continue
        # shared root path: from pairs when given, else (depth_i + depth_j - d_ij) / 2
        derived = 0.5 * (dc[:, None] + dc[None, :] - dm)
        sm = np.where(np.isfinite(sm), sm, derived)
        np.fill_diagonal(sm, dc)
        units.append(cells.astype(np.int64))
        dist.append(dm)
        shared.append(np.clip(sm, 0, None))
        clone_lab, block_lab = ulabels[u].split("\x1f", 1)
        uc.append(clone_lab)
        ub.append(block_lab)
    notes["cells_dropped_incomplete"] = int(incomplete)
    return LineageForest(n, units, dist, shared, depth, uc, ub, notes)


# =========================================================================== REML engine


def _helmert(n: int) -> np.ndarray:
    """Orthonormal ``(n, n-1)`` basis of the contrasts orthogonal to the constant vector."""
    h = np.zeros((n, n - 1))
    for k in range(1, n):
        h[:k, k - 1] = 1.0
        h[k, k - 1] = -k
        h[:, k - 1] /= np.sqrt(k * (k + 1))
    return h


class _Engine:
    """Per-unit projections that diagonalise the kinship kernel (shared by fit, test, simulation)."""

    def __init__(self, forest: LineageForest, kernels: list):
        self.forest = forest
        sizes = np.array([u.size for u in forest.units])
        self.groups = []  # (unit indices of this size, stacked T (m, n-1, n), stacked d (m, n-1))
        order_d = []
        for n in np.unique(sizes):
            idx = np.flatnonzero(sizes == n)
            P = _helmert(int(n))
            T = np.empty((idx.size, n - 1, n))
            dd = np.empty((idx.size, n - 1))
            for k, u in enumerate(idx):
                M = P.T @ kernels[u] @ P
                w, V = np.linalg.eigh(0.5 * (M + M.T))
                dd[k] = np.clip(w, 0, None)
                T[k] = V.T @ P.T
            rows = np.stack([forest.units[u] for u in idx])  # (m, n)
            self.groups.append((idx, rows, T, dd))
            order_d.append(dd.ravel())
        self.d = np.concatenate(order_d) if order_d else np.empty(0)
        self.N = self.d.size

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Stacked projected rows for a ``(n_cells, k)`` matrix (row order of :attr:`d`)."""
        out = []
        for _, rows, T, _ in self.groups:
            out.append(np.einsum("mij,mjk->mik", T, Y[rows]).reshape(-1, Y.shape[1]))
        return np.concatenate(out, axis=0) if out else np.empty((0, Y.shape[1]))

    def transform_permuted(self, E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """As :meth:`transform`, after permuting the rows of ``E`` within every unit."""
        out = []
        for _, rows, T, _ in self.groups:
            perm = np.argsort(rng.random(rows.shape), axis=1)
            prow = np.take_along_axis(rows, perm, axis=1)
            out.append(np.einsum("mij,mjk->mik", T, E[prow]).reshape(-1, E.shape[1]))
        return np.concatenate(out, axis=0)

    def unit_centre(self, Y: np.ndarray) -> np.ndarray:
        """``Y`` with each unit's mean removed (rows outside the forest are left at 0)."""
        out = np.zeros_like(Y, dtype=float)
        for _, rows, _, _ in self.groups:
            blk = Y[rows]
            out[rows] = blk - blk.mean(axis=1, keepdims=True)
        return out


def _clean_design(X: np.ndarray | None, tol: float = 1e-8) -> np.ndarray | None:
    """Drop numerically redundant columns of the projected design."""
    if X is None or X.size == 0:
        return None
    norms = np.linalg.norm(X, axis=0)
    X = X[:, norms > tol * max(1.0, norms.max(initial=0.0))]
    if X.shape[1] == 0:
        return None
    q, r, piv = _qr_pivot(X)
    diag = np.abs(np.diag(r))
    rank = int((diag > tol * diag.max()).sum()) if diag.size else 0
    return X[:, np.sort(piv[:rank])] if rank else None


def _qr_pivot(X):
    from scipy.linalg import qr

    return qr(X, mode="economic", pivoting=True)


def _reml_profile(z: np.ndarray, X: np.ndarray | None, d: np.ndarray, lam: float) -> float:
    """Profiled REML log-likelihood (up to a constant) at ``lambda = sigma_g^2 / sigma_e^2``."""
    w = 1.0 / (1.0 + lam * d)
    if X is None:
        q = float(np.dot(w * z, z))
        n_eff, logdet = z.size, 0.0
    else:
        XtW = X.T * w
        A = XtW @ X
        beta = np.linalg.solve(A, XtW @ z)
        r = z - X @ beta
        q = float(np.dot(w * r, r))
        sign, logdet = np.linalg.slogdet(A)
        n_eff = z.size - X.shape[1]
    if q <= 0:
        return -np.inf
    return -0.5 * (np.log1p(lam * d).sum() + logdet + n_eff * np.log(q / n_eff))


_H_GRID = np.r_[0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, np.arange(0.1, 0.951, 0.05)]


def _fit_h2(z, X, d):
    """REML h2 (golden-section refined), 95% profile interval and boundary LRT."""
    lam = _H_GRID / (1 - _H_GRID)
    ll = np.array([_reml_profile(z, X, d, l) for l in lam])
    k = int(np.argmax(ll))
    lo_h, hi_h = _H_GRID[max(k - 1, 0)], _H_GRID[min(k + 1, _H_GRID.size - 1)]
    f = lambda h: -_reml_profile(z, X, d, h / (1 - h))  # noqa: E731
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo_h, hi_h
    c, e = b - gr * (b - a), a + gr * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(40):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + gr * (b - a)
            fe = f(e)
    h_hat = 0.5 * (a + b)
    l_hat = -f(h_hat)
    if ll[k] > l_hat:
        h_hat, l_hat = float(_H_GRID[k]), float(ll[k])
    l0 = float(ll[0])
    lrt = max(0.0, 2 * (l_hat - l0))
    p_lrt = 0.5 * stats.chi2.sf(lrt, 1) if lrt > 0 else 1.0
    # Profile interval: the two points where 2 (l_hat - l(h)) crosses the chi2_1 0.95 cut-off, found by
    # linear interpolation *between* grid points. Snapping to the grid instead (as an earlier version did)
    # makes the interval a staircase and destroys its coverage even when the model is correct.
    fine = np.unique(np.r_[_H_GRID, h_hat])
    llf = np.array([_reml_profile(z, X, d, h / (1 - h)) for h in fine])
    drop = 2 * (l_hat - llf)
    cut = stats.chi2.ppf(0.95, 1)
    inside = drop <= cut

    def cross(i, j):
        """Linear interpolation of the cut-off crossing between grid points i (inside) and j (outside)."""
        x0, x1, y0, y1 = fine[i], fine[j], drop[i], drop[j]
        return float(x0 + (x1 - x0) * (cut - y0) / (y1 - y0)) if y1 != y0 else float(x0)

    if not inside.any():
        lo = hi = float("nan")
    else:
        idx = np.flatnonzero(inside)
        first, last = int(idx[0]), int(idx[-1])
        lo = float(fine[first]) if first == 0 else cross(first, first - 1)
        hi = float(fine[last]) if last == fine.size - 1 else cross(last, last + 1)
    return float(h_hat), lo, hi, float(lrt), float(p_lrt)


@dataclass
class LineageHeritability:
    """Result of :func:`lineage_heritability` (one row per feature in :attr:`table`)."""

    table: pd.DataFrame
    combined_p: float
    settings: dict
    forest: dict

    def __repr__(self) -> str:
        cols = ["feature", "h2", "h2_lo", "h2_hi", "p_perm"]
        return ("LineageHeritability(" f"{self.forest.get('units')} units, {self.forest.get('cells')} cells, "
                f"kernel={self.settings.get('kernel')}, combined p={self.combined_p:.3g})\n"
                + self.table[cols].round(4).to_string(index=False))


def lineage_heritability(
    y,
    forest: LineageForest,
    *,
    covariates=None,
    depth_effect: bool = True,
    kernel: str = "exponential",
    length_scale: float = 2.0,
    n_perm: int = 999,
    feature_names=None,
    random_state: int = 0,
    verbose: bool = True,
) -> LineageHeritability:
    """Estimate and test lineage heritability of cell state inside clones.

    Parameters
    ----------
    y
        ``(n_cells,)`` or ``(n_cells, k)`` state features, row-aligned with the data the forest
        indexes (module scores, principal components, or 0/1 recorded labels). Rows outside the
        forest are ignored; rows inside must be finite (use :meth:`LineageForest.subset`).
    forest
        Output of :func:`lineage_forest`.
    covariates
        Optional ``(n_cells, p)`` cell-level fixed effects (log sequencing depth, quality metrics,
        one-hot sample indicators...). Clone x block intercepts are always included.
    depth_effect
        Add the cell's root-to-tip mutation depth as a fixed effect (recommended: it separates "more
        mutated cells differ" from "relatives resemble each other").
    kernel, length_scale
        Kinship kernel (``"exponential"``, ``"identical"``, ``"identical_mutated"``, ``"brownian"``)
        and, for the exponential kernel, its length scale in mutations.
    n_perm
        Freedman-Lane permutations for the score test (0 to skip).

    Returns
    -------
    :class:`LineageHeritability` with, per feature, the REML ``h2`` and its 95% profile interval,
    the boundary likelihood-ratio test, the score statistic and its permutation p-value, and a
    combined permutation p-value over all features (sum of standardised scores).
    """
    if forest.n_units == 0:
        raise ValueError("the forest has no unit with two or more cells.")
    Y = np.asarray(y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if Y.shape[0] != forest.n_cells:
        raise ValueError(f"y has {Y.shape[0]} rows; the forest indexes {forest.n_cells}.")
    rows = forest.rows
    if not np.isfinite(Y[rows]).all():
        raise ValueError("y has missing values on cells of the forest; subset the forest first.")
    names = list(feature_names) if feature_names is not None else [f"y{k}" for k in range(Y.shape[1])]
    eng = _Engine(forest, forest.kernel(kernel, length_scale))
    parts = []
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        C = C[:, None] if C.ndim == 1 else C
        if C.shape[0] != forest.n_cells or not np.isfinite(C[rows]).all():
            raise ValueError("covariates must be finite with one row per data row.")
        parts.append(C)
    if depth_effect:
        parts.append(np.nan_to_num(forest.depth, nan=0.0)[:, None])
    Xraw = np.column_stack(parts) if parts else None
    Xt = _clean_design(eng.transform(Xraw)) if Xraw is not None else None
    Z = eng.transform(Y)
    d = eng.d

    # null-model residuals (OLS in the projected space == within-unit OLS)
    if Xt is not None:
        Q, _ = np.linalg.qr(Xt)
        R0 = Z - Q @ (Q.T @ Z)
        p_fix = Xt.shape[1]
    else:
        Q, R0, p_fix = None, Z.copy(), 0
    # cell-level null residuals for Freedman-Lane: unit-centred y minus the fitted covariate part
    Yc = eng.unit_centre(Y)
    if Xraw is not None and Xt is not None:
        Xc = eng.unit_centre(Xraw)
        beta0, *_ = np.linalg.lstsq(eng.transform(Xc), Z, rcond=None)
        E0 = Yc - Xc @ beta0
    else:
        E0 = Yc

    def score(R):
        return (d[:, None] * R * R).sum(axis=0) / np.maximum((R * R).sum(axis=0), 1e-300)

    obs = score(R0)
    rows_out = []
    for k, name in enumerate(names):
        h, lo, hi, lrt, plrt = _fit_h2(Z[:, k], Xt, d)
        rows_out.append({"feature": name, "h2": h, "h2_lo": lo, "h2_hi": hi, "lrt": lrt, "p_lrt": plrt,
                         "score": float(obs[k])})
    tab = pd.DataFrame(rows_out)
    combined = float("nan")
    if n_perm:
        require_positive_int(n_perm, "n_perm")
        rng = np.random.default_rng(random_state)
        null = np.empty((n_perm, Y.shape[1]))
        for b in range(n_perm):
            Rb = eng.transform_permuted(E0, rng)
            if Q is not None:
                Rb = Rb - Q @ (Q.T @ Rb)
            null[b] = score(Rb)
        mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
        tab["score_z"] = (obs - mu) / np.where(sd > 0, sd, np.nan)
        tab["p_perm"] = (1 + (null >= obs[None, :] - 1e-12).sum(axis=0)) / (n_perm + 1)
        zsum_null = ((null - mu) / np.where(sd > 0, sd, np.nan)).sum(axis=1)
        zsum = np.nansum(tab["score_z"].to_numpy())
        combined = float((1 + (zsum_null >= zsum - 1e-12).sum()) / (n_perm + 1))
    fs = forest.summary()
    tab["n_cells"] = fs["cells"]
    tab["n_units"] = fs["units"]
    tab["n_clones"] = fs["clones"]
    settings = {"kernel": kernel, "length_scale": float(length_scale) if kernel == "exponential" else None,
                "depth_effect": bool(depth_effect), "n_fixed_effects": int(p_fix), "n_perm": int(n_perm),
                "random_state": int(random_state), "N_projected_rows": int(eng.N)}
    res = LineageHeritability(tab, combined, settings, fs)
    if verbose:
        log(f"lineage heritability ({kernel}): {fs['units']} units, {fs['cells']} cells; "
            + "; ".join(f"{r.feature} h2={r.h2:.3f}" for r in tab.itertuples()), verbose)
    return res


# =========================================================================== simulation / calibration


def simulate_lineage_trait(forest: LineageForest, h2: float, *, kernel: str = "exponential",
                           length_scale: float = 2.0, base=None, random_state: int = 0) -> np.ndarray:
    """A feature with lineage heritability ``h2`` on the forest's real trees.

    ``base`` (one value per row) supplies the non-heritable part: it is permuted within every unit
    (destroying any lineage structure it had) and standardised within units; without ``base`` the
    non-heritable part is Gaussian. Rows outside the forest are NaN.
    """
    if not 0 <= h2 < 1:
        raise ValueError("h2 must be in [0, 1).")
    rng = np.random.default_rng(random_state)
    out = np.full(forest.n_cells, np.nan)
    ks = forest.kernel(kernel, length_scale)
    base = None if base is None else np.asarray(base, dtype=float)
    for u, K in zip(forest.units, ks):
        w, V = np.linalg.eigh(K)
        g = V @ (np.sqrt(np.clip(w, 0, None)) * rng.normal(size=u.size))
        if base is None:
            e = rng.normal(size=u.size)
        else:
            e = base[u][rng.permutation(u.size)]
            e = (e - e.mean()) / (e.std() if e.std() > 0 else 1.0)
        out[u] = np.sqrt(h2) * g + np.sqrt(1 - h2) * e
    return out


def lineage_power(forest: LineageForest, *, base=None, h2_grid=(0.0, 0.02, 0.05, 0.1, 0.2),
                  n_sim: int = 100, kernel: str = "exponential", length_scale: float = 2.0,
                  depth_effect: bool = True, n_perm: int = 199, alpha: float = 0.05,
                  random_state: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Bias, type I error (``h2 = 0``) and power of :func:`lineage_heritability` on these trees."""
    rng = np.random.default_rng(random_state)
    rows = []
    for h2 in h2_grid:
        est, p_perm, p_lrt = [], [], []
        for s in range(n_sim):
            seed = int(rng.integers(2**31))
            y = simulate_lineage_trait(forest, h2, kernel=kernel, length_scale=length_scale, base=base,
                                       random_state=seed)
            r = lineage_heritability(np.nan_to_num(y), forest, kernel=kernel, length_scale=length_scale,
                                     depth_effect=depth_effect, n_perm=n_perm, random_state=seed + 1,
                                     verbose=False)
            est.append(r.table["h2"].iloc[0])
            p_perm.append(r.table["p_perm"].iloc[0] if n_perm else np.nan)
            p_lrt.append(r.table["p_lrt"].iloc[0])
        est, p_perm, p_lrt = map(np.asarray, (est, p_perm, p_lrt))
        rows.append({"h2_true": h2, "n_sim": n_sim, "h2_mean": est.mean(), "h2_sd": est.std(ddof=1),
                     "rejection_perm": float((p_perm < alpha).mean()) if n_perm else np.nan,
                     "rejection_lrt": float((p_lrt < alpha).mean()), "kernel": kernel,
                     "length_scale": length_scale if kernel == "exponential" else np.nan})
        log(f"  h2={h2:.3f}: mean estimate {est.mean():.3f}, rejection perm "
            f"{rows[-1]['rejection_perm']:.2f}, LRT {rows[-1]['rejection_lrt']:.2f}", verbose)
    return pd.DataFrame(rows)


# =========================================================================== pairwise diagnostic
# The functions below implement the first, pairwise statistic (correlation of within-unit tree
# distances with state distances). It is superseded by lineage_heritability because a pair's tree
# distance is roughly the sum of the two cells' depths, which confounds a depth effect with lineage
# resemblance; it is kept so that earlier outputs remain reproducible.


def pair_table(clones, blocks=None, distances=None) -> pd.DataFrame:
    """Within-clone, within-block cell pairs with their receptor distance."""
    if distances is None:
        raise ValueError("distances are required: give within-clone cell-pair receptor distances.")
    d = pd.DataFrame({"i": np.asarray(distances["i"], dtype=np.int64),
                      "j": np.asarray(distances["j"], dtype=np.int64),
                      "d": np.asarray(distances["d"], dtype=float)})
    clone_codes, _ = codes(pd.Series(clones).astype(str))
    if blocks is None:
        block_codes = clone_codes
    else:
        block_codes, _ = codes(pd.Series(clones).astype(str) + "\x1f" + pd.Series(blocks).astype(str))
    same = (clone_codes[d["i"]] == clone_codes[d["j"]]) & (block_codes[d["i"]] == block_codes[d["j"]])
    out = d[same & np.isfinite(d["d"])].copy()
    out["block"] = block_codes[out["i"].to_numpy()]
    out.attrs["dropped_cross_block"] = int((~same).sum())
    out.attrs["block_codes"] = block_codes
    return out.reset_index(drop=True)


def _block_center(v: np.ndarray, block: np.ndarray, n_blocks: int) -> np.ndarray:
    cnt = np.bincount(block, minlength=n_blocks)
    tot = np.bincount(block, weights=v, minlength=n_blocks)
    return v - (tot / np.maximum(cnt, 1))[block]


def _residualise(v: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    if cov is None or cov.size == 0:
        return v
    coef, *_ = np.linalg.lstsq(cov, v, rcond=None)
    return v - cov @ coef


def _state_distance(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    diff = x[i] - x[j]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _pooled_r(s_c, e, block, n_blocks, cov) -> float:
    e_c = _block_center(e, block, n_blocks)
    if cov is not None:
        e_c = _residualise(e_c, cov)
    num = float(s_c @ e_c)
    den = np.sqrt(float(s_c @ s_c) * float(e_c @ e_c))
    return num / den if den > 0 else np.nan


def _permute_within_blocks(block_of_cell: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(block_of_cell, kind="stable")
    sizes = np.bincount(block_of_cell)
    out = np.empty(order.size, dtype=np.int64)
    start = 0
    for s in sizes:
        if s:
            seg = order[start:start + s]
            out[seg] = seg[rng.permutation(s)]
            start += s
    return out


@dataclass
class LineageSignal:
    """Result of :func:`lineage_signal` (pairwise diagnostic)."""

    r: float
    p_value: float
    null_mean: float
    null_sd: float
    n_pairs: int
    n_blocks: int
    n_clones: int
    n_cells: int
    settings: dict = field(default_factory=dict)
    identical_pairs: int = 0
    r_identical: float = float("nan")
    null: np.ndarray | None = field(default=None, repr=False)
    h2_equivalent: float | None = None
    h2_ci: tuple | None = None

    @property
    def z(self) -> float:
        return (self.r - self.null_mean) / self.null_sd if self.null_sd else float("nan")


def lineage_signal(x, clones, distances, *, blocks=None, covariates=None, include_identical: bool = True,
                   n_perm: int = 2000, random_state: int = 0, verbose: bool = True) -> LineageSignal:
    """Pooled within-unit correlation of receptor distance and state distance (superseded)."""
    require_positive_int(n_perm, "n_perm")
    x = np.asarray(x, dtype=float)
    pairs = pair_table(clones, blocks, distances)
    if not include_identical:
        pairs = pairs[pairs["d"] > 0].reset_index(drop=True)
    if len(pairs) < 10:
        raise ValueError(f"only {len(pairs)} usable within-clone pairs; need at least 10.")
    block_of_cell = pairs.attrs["block_codes"]
    used, block = np.unique(pairs["block"].to_numpy(), return_inverse=True)
    n_blocks = used.size
    i = pairs["i"].to_numpy()
    j = pairs["j"].to_numpy()
    s_c = _block_center(pairs["d"].to_numpy(), block, n_blocks)
    cov = None
    if covariates is not None:
        c = np.asarray(covariates, dtype=float)
        c = c[:, None] if c.ndim == 1 else c
        parts = []
        for k in range(c.shape[1]):
            parts.append(_block_center(c[i, k] + c[j, k], block, n_blocks))
            parts.append(_block_center(np.abs(c[i, k] - c[j, k]), block, n_blocks))
        cov = np.column_stack(parts)
        s_c = _residualise(s_c, cov)
    obs = _pooled_r(s_c, _state_distance(x, i, j), block, n_blocks, cov)
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = _permute_within_blocks(block_of_cell, rng)
        null[b] = _pooled_r(s_c, _state_distance(x, perm[i], perm[j]), block, n_blocks, cov)
    finite = null[np.isfinite(null)]
    centre = finite.mean()
    p = (1 + np.sum(np.abs(finite - centre) >= abs(obs - centre) - 1e-12)) / (finite.size + 1)
    ident = pairs["d"].to_numpy() == 0
    r_ident = float("nan")
    if ident.sum() >= 10 and (~ident).sum() >= 10:
        e = _state_distance(x, i, j)
        e_c = _block_center(e, block, n_blocks)
        r_ident = float(e_c[ident].mean() - e_c[~ident].mean())
    res = LineageSignal(
        r=float(obs), p_value=float(p), null_mean=float(centre),
        null_sd=float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        n_pairs=int(len(pairs)), n_blocks=int(n_blocks),
        n_clones=int(pd.unique(np.asarray(clones, dtype=object)[np.unique(np.r_[i, j])]).size),
        n_cells=int(np.unique(np.r_[i, j]).size), null=null, identical_pairs=int(ident.sum()),
        r_identical=r_ident,
        settings={"include_identical": bool(include_identical), "n_perm": int(n_perm),
                  "random_state": int(random_state), "blocked": blocks is not None,
                  "n_covariates": 0 if cov is None else cov.shape[1], "n_features": int(x.shape[1]),
                  "dropped_cross_block": int(pairs.attrs.get("dropped_cross_block", 0))},
    )
    log(f"lineage signal: r = {obs:+.4f}, p = {p:.3g}; {res.n_clones} clones, {res.n_pairs} pairs.", verbose)
    return res


def planted_calibration(x, clones, distances, *, blocks=None, h2_grid=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
                        n_sim: int = 50, n_perm: int = 299, length_scale: float | str = "median",
                        alpha: float = 0.05, random_state: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Planted-signal calibration of the pairwise diagnostic (plants on the real, un-nullified data)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    pairs = pair_table(clones, blocks, distances)
    block_of_cell = pairs.attrs["block_codes"]
    used, block = np.unique(pairs["block"].to_numpy(), return_inverse=True)
    n_blocks = used.size
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    d = pairs["d"].to_numpy()
    s_c = _block_center(d, block, n_blocks)
    pos = d[d > 0]
    ls = float(np.median(pos)) if length_scale == "median" and pos.size else (
        1.0 if length_scale == "median" else float(length_scale))
    cells_in_block = {b: np.flatnonzero(block_of_cell == b) for b in np.unique(block_of_cell)}
    dist_lookup = {}
    for b in used:
        idx = cells_in_block[b]
        pos_of = {c: k for k, c in enumerate(idx)}
        m = np.zeros((idx.size, idx.size))
        sel = pairs["block"].to_numpy() == b
        for a_, b_, dd in zip(i[sel], j[sel], d[sel]):
            m[pos_of[a_], pos_of[b_]] = m[pos_of[b_], pos_of[a_]] = dd
        c = np.exp(-m / ls)
        np.fill_diagonal(c, 1.0)
        w, v = np.linalg.eigh(c)
        dist_lookup[b] = (idx, v * np.sqrt(np.clip(w, 0, None)))
    xb = x - pd.DataFrame(x).groupby(block_of_cell).transform("mean").to_numpy()
    base_var = float(np.einsum("ij,ij->", xb, xb) / xb.shape[0])
    rows = []
    for h2 in h2_grid:
        hits, rs = 0, []
        for _ in range(n_sim):
            xs = x.copy()
            if h2 > 0:
                u = rng.normal(size=x.shape[1])
                u /= np.linalg.norm(u)
                amp = np.sqrt(h2 / (1 - h2) * base_var)
                for b, (idx, L) in dist_lookup.items():
                    xs[idx] += amp * (L @ rng.normal(size=idx.size))[:, None] * u[None, :]
            obs = _pooled_r(s_c, _state_distance(xs, i, j), block, n_blocks, None)
            null = np.empty(n_perm)
            for k in range(n_perm):
                perm = _permute_within_blocks(block_of_cell, rng)
                null[k] = _pooled_r(s_c, _state_distance(xs, perm[i], perm[j]), block, n_blocks, None)
            centre = null.mean()
            pv = (1 + np.sum(np.abs(null - centre) >= abs(obs - centre) - 1e-12)) / (null.size + 1)
            hits += pv < alpha
            rs.append(obs)
        rows.append({"h2_planted": h2, "n_sim": int(n_sim), "power": hits / n_sim, "mean_r": float(np.mean(rs)),
                     "sd_r": float(np.std(rs, ddof=1)) if n_sim > 1 else np.nan, "length_scale": ls})
    return pd.DataFrame(rows)
