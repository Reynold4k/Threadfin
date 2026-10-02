"""Clone state profiles: variance components and shrunken clone effects.

The model
---------
For cell ``i`` of clone ``c`` sampled in context ``s`` (a sample, donor or
batch), with expression embedding ``x_i`` (one row of e.g. a PCA), Threadfin
fits the one-way random-effects model, independently for every feature ``j``::

    x_ij = mu_sj + u_cj + e_ij,   u_cj ~ N(0, tau2_j),   e_ij ~ N(0, sigma2_j)

* ``mu_s`` is the context mean, estimated from *all* cells of the context, so
  a clone is described relative to the cells it was sampled with;
* ``tau2_j`` is the between-clone (clonally inherited) variance and
  ``sigma2_j`` the within-clone variance;
* the **clonal intraclass correlation**
  ``ICC = sum(tau2) / sum(tau2 + sigma2)`` is the fraction of transcriptional
  variance explained by clone identity;
* each clone is represented by its BLUP (best linear unbiased predictor)
  ``u_c = lambda_c * mean_i(x_i - mu_s)``, with
  ``lambda_cj = n_c tau2_j / (n_c tau2_j + sigma2_j)``. Small clones are
  shrunk towards their context, expanded clones keep their own profile, and
  expression directions without clonal signal (``tau2_j = 0``) drop out;
* a clone's **reliability** is the tau2-weighted mean of ``lambda_cj``
  (Spearman-Brown): the expected squared correlation between the estimated
  and the true clone profile.

With ``representation="kernel"`` the same model is applied to random Fourier
features of the embedding (Rahimi & Recht 2007). Each clone is then summarised
by a shrunken *kernel mean embedding* of its cell distribution instead of its
centroid, so a clone split between two states is distinguished from a clone
sitting between them.

Variance components use the moment (ANOVA) estimators of the unbalanced
one-way design (Searle, Casella & McCulloch 1992, ch. 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import codes, get_basis, get_uns, log, modal, require_obs

PROFILE_KEY = "profiles"


# --------------------------------------------------------------------------- variance components


@dataclass
class VarianceComponents:
    """Moment estimates of the one-way random-effects model, per feature."""

    sigma2: np.ndarray
    tau2: np.ndarray
    n0: float
    n_clones: int
    n_cells: int
    grand_mean: np.ndarray = field(repr=False)

    @property
    def icc_per_feature(self) -> np.ndarray:
        """ICC of each feature (e.g. each principal component)."""
        tot = self.tau2 + self.sigma2
        return np.divide(self.tau2, tot, out=np.zeros_like(tot), where=tot > 0)

    @property
    def icc(self) -> float:
        """Pooled clonal ICC: share of total variance explained by clone."""
        tot = float(np.sum(self.tau2 + self.sigma2))
        return float(np.sum(self.tau2) / tot) if tot > 0 else 0.0

    def shrinkage(self, n) -> np.ndarray:
        """Shrinkage factors ``lambda`` for clone sizes ``n``: ``(len(n), n_features)``."""
        n = np.asarray(n, dtype=float)[:, None]
        num = n * self.tau2[None, :]
        den = num + self.sigma2[None, :]
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def reliability(self, n) -> np.ndarray:
        """tau2-weighted mean shrinkage (Spearman-Brown reliability) per clone."""
        lam = self.shrinkage(n)
        total = self.tau2.sum()
        w = self.tau2 / total if total > 0 else np.zeros_like(self.tau2)
        return lam @ w

    def min_cells_for(self, target: float = 0.5) -> int:
        """Smallest clone size whose reliability reaches ``target``."""
        rho = self.icc
        if rho <= 0:
            return np.iinfo(np.int32).max
        if rho >= 1:
            return 1
        # Spearman-Brown: R(n) = n*rho / (1 + (n - 1)*rho), solved for n.
        n = target * (1 - rho) / (rho * (1 - target))
        return int(max(1, np.ceil(n - 1e-9)))

    def as_dict(self) -> dict:
        """Plain dict (numpy arrays) suitable for ``adata.uns``."""
        return {
            "icc": self.icc,
            "sigma2": np.asarray(self.sigma2),
            "tau2": np.asarray(self.tau2),
            "icc_per_feature": self.icc_per_feature,
            "n0": float(self.n0),
            "n_clones": int(self.n_clones),
            "n_cells": int(self.n_cells),
        }


def variance_components(residuals: np.ndarray, clone_codes: np.ndarray) -> VarianceComponents:
    """ANOVA estimators of ``sigma2`` and ``tau2`` from clones with >= 2 cells.

    Parameters
    ----------
    residuals
        ``(n_cells, n_features)`` context-centred values.
    clone_codes
        Integer clone code per row; negative values are ignored. Only clones
        with at least two rows contribute, because a singleton carries no
        information about within-clone variance.
    """
    cc = np.asarray(clone_codes)
    ok = cc >= 0
    sizes = np.bincount(cc[ok]) if ok.any() else np.array([], dtype=int)
    multi = np.flatnonzero(sizes >= 2)
    if multi.size < 2:
        raise ValueError("Need at least two clones with >= 2 cells to estimate variance components.")
    # re-index the multi-cell clones as 0..C-1; every other row is dropped
    remap = -np.ones(sizes.size, dtype=np.int64)
    remap[multi] = np.arange(multi.size)
    g = np.where(ok, remap[np.where(ok, cc, 0)], -1)
    keep = g >= 0
    r, g = residuals[keep], g[keep]
    n_clones = multi.size
    n_c = np.bincount(g, minlength=n_clones).astype(float)
    n_total = float(n_c.sum())

    # per-clone sums via a sparse indicator matrix (much faster than np.add.at)
    member = sp.csr_matrix((np.ones(g.size), (g, np.arange(g.size))), shape=(n_clones, g.size))
    sums = np.asarray(member @ r)
    means = sums / n_c[:, None]
    grand = r.mean(axis=0)
    # sums of squares via sum(x^2) - sum(n_c * mean_c^2): one pass, no loops
    ss_raw = np.einsum("ij,ij->j", r, r)
    ss_between_raw = np.einsum("c,cj->j", n_c, means**2)
    ss_within = ss_raw - ss_between_raw
    ss_between = ss_between_raw - n_total * grand**2
    ms_within = np.maximum(ss_within, 0.0) / (n_total - n_clones)
    ms_between = np.maximum(ss_between, 0.0) / (n_clones - 1)
    # effective clone size of an unbalanced design
    n0 = (n_total - float(np.sum(n_c**2)) / n_total) / (n_clones - 1)
    tau2 = np.maximum((ms_between - ms_within) / n0, 0.0)
    return VarianceComponents(
        sigma2=ms_within, tau2=tau2, n0=float(n0), n_clones=int(n_clones),
        n_cells=int(n_total), grand_mean=grand,
    )


# --------------------------------------------------------------------------- feature maps


def _center_by(x: np.ndarray, ctx: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Subtract each context's mean (cells with no context: the global mean)."""
    n_ctx = int(ctx.max()) + 1 if (ctx >= 0).any() else 0
    valid = (ctx >= 0) & finite
    member = sp.csr_matrix((np.ones(valid.sum()), (ctx[valid], np.flatnonzero(valid))), shape=(n_ctx, x.shape[0]))
    cnt = np.asarray(member.sum(axis=1)).ravel()
    mu = np.asarray(member @ x) / np.maximum(cnt, 1)[:, None]
    fallback = x[finite].mean(axis=0)
    return x - np.where((ctx >= 0)[:, None], mu[np.maximum(ctx, 0)], fallback)


def neighbour_average(x: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """Average every cell with its ``k`` nearest cells (itself included).

    Borrows strength from cells in the same local state, which denoises the
    profiles of small clones. Used for programme profiles only: the
    coherence and memory statistics are computed on unsmoothed cells.
    """
    n = x.shape[0]
    k = int(min(k, n))
    if n > 50_000:
        from pynndescent import NNDescent

        idx = NNDescent(x, n_neighbors=k, random_state=random_state).neighbor_graph[0]
    else:
        from sklearn.neighbors import NearestNeighbors

        idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x, return_distance=False)
    out = np.zeros_like(x)
    for j in range(k):  # column-wise accumulation keeps memory at O(n * features)
        out += x[idx[:, j]]
    return out / k


def median_bandwidth(x: np.ndarray, n_sample: int = 2000, rng=None) -> float:
    """Median distance between random pairs of cells (Gretton et al. 2012)."""
    rng = rng or np.random.default_rng(0)
    n = x.shape[0]
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    d = np.linalg.norm(x[idx] - x[rng.permutation(idx)], axis=1)
    return float(np.median(d[d > 0])) if np.any(d > 0) else 1.0


def local_bandwidth(x: np.ndarray, k: int = 30, n_sample: int = 2000, rng=None) -> float:
    """Kernel bandwidth from the typical distance to a cell's k-th neighbour.

    A local scale (about the width of one cell state) resolves finer
    differences between clones' state distributions than the median
    heuristic, but every cell's features are noisier, so only larger clones
    reach a given reliability. Useful for datasets with many large clones.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = rng or np.random.default_rng(0)
    n = x.shape[0]
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    k = int(min(k, n - 1))
    dist = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x[idx])[0][:, -1]
    bw = float(np.median(dist))
    return bw if bw > 0 else 1.0


def _random_fourier_features(
    x: np.ndarray, n_features: int, bandwidth: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Gaussian-kernel random Fourier features with the given bandwidth."""
    p = x.shape[1]
    w = rng.normal(0.0, 1.0 / bandwidth, size=(p, n_features))
    phase = rng.uniform(0.0, 2 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(x @ w + phase), bandwidth


# --------------------------------------------------------------------------- model object


class ProfileModel:
    """Context-centred cell features plus the fitted variance components.

    It holds cell-level matrices, so it is not stored in ``adata``; it is
    rebuilt on demand from the parameters saved by :func:`clone_profiles`
    (see :func:`model_from_adata`).
    """

    def __init__(self, residuals, clone_codes, clone_index, context_codes, vc, params):
        self.residuals = residuals
        self.clone_codes = clone_codes
        self.clone_index = clone_index
        self.context_codes = context_codes
        self.vc = vc
        self.params = params

    def group_blups(self, group_codes: np.ndarray, n_groups: int, weights: np.ndarray | None = None):
        """BLUPs for any grouping of cells (clones, time snapshots, bootstraps).

        Parameters
        ----------
        group_codes
            Group code per cell (negative = not in any group).
        n_groups
            Number of groups.
        weights
            Optional per-cell weights (e.g. Poisson bootstrap counts).

        Returns
        -------
        ``(blup, n_eff, reliability)`` where ``n_eff`` is the weighted number
        of cells per group.
        """
        g = np.asarray(group_codes)
        ok = np.flatnonzero(g >= 0)
        w = np.ones(g.size) if weights is None else np.asarray(weights, dtype=float)
        member = sp.csr_matrix((w[ok], (g[ok], ok)), shape=(n_groups, g.size))
        sums = np.asarray(member @ self.residuals)
        n_eff = np.asarray(member.sum(axis=1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / n_eff[:, None]
        means = np.nan_to_num(means - self.vc.grand_mean[None, :])
        blup = self.vc.shrinkage(n_eff) * means
        return blup, n_eff, self.vc.reliability(n_eff)


def build_model(
    adata,
    *,
    clone_key: str = "clone_id",
    basis: str = "X_pca",
    context_key: str | None = None,
    representation: str = "mean",
    n_features: int = 256,
    bandwidth: float | str = "median",
    smooth: int = 0,
    random_state: int = 0,
) -> ProfileModel:
    """Build the profile model: features, context centring, variance components."""
    require_obs(adata, clone_key, context_key, context="clone profiles")
    x = get_basis(adata, basis)
    finite = np.isfinite(x).all(axis=1)
    if not finite.all():
        x = np.where(finite[:, None], x, 0.0)
    params = {
        "clone_key": clone_key,
        "basis": basis,
        "context_key": context_key,
        "representation": representation,
        "n_features": int(n_features),
        "smooth": int(smooth),
        "random_state": int(random_state),
    }

    if representation not in ("mean", "kernel"):
        raise ValueError("representation must be 'mean' or 'kernel'.")
    if context_key is not None:
        ctx, _ = codes(adata.obs[context_key])
    else:
        ctx = np.zeros(adata.n_obs, dtype=np.int64)

    # 1) remove each context's mean, so a clone is compared with the cells it
    #    was sampled with (this also removes technical shifts between samples)
    resid = _center_by(x, ctx, finite)
    rng = np.random.default_rng(random_state)
    if representation == "kernel":
        # bandwidth from unsmoothed cells, so smoothed and unsmoothed models share one kernel
        if bandwidth is None or bandwidth == "median":
            bandwidth = median_bandwidth(resid[finite], rng=rng)
        elif bandwidth == "local":
            bandwidth = local_bandwidth(resid[finite], rng=rng)
    if smooth and smooth > 1:
        resid = np.where(finite[:, None], neighbour_average(np.where(finite[:, None], resid, 0.0), smooth,
                                                            random_state), 0.0)

    # 2) "kernel": describe each cell by random Fourier features of its
    #    context-centred position, then centre again so that a clone's profile
    #    is the difference between its own cell distribution and its context's
    if representation == "kernel":
        feats, bw = _random_fourier_features(resid, n_features, float(bandwidth), rng)
        params["bandwidth"] = bw
        resid = _center_by(feats, ctx, finite)

    # 3) variance components from clones with >= 2 cells
    clone_codes, clone_index = codes(adata.obs[clone_key])
    clone_codes = np.where(finite, clone_codes, -1)
    vc = variance_components(resid, clone_codes)
    return ProfileModel(resid, clone_codes, clone_index, ctx, vc, params)


# --------------------------------------------------------------------------- public API


def clone_profiles(
    adata,
    *,
    clone_key: str = "clone_id",
    basis: str = "X_pca",
    context_key: str | None = None,
    donor_key: str | None = None,
    representation: str = "mean",
    n_features: int = 256,
    n_components: int = 30,
    bandwidth: float | str = "median",
    smooth: int | None = None,
    min_cells: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimate a shrunken, context-adjusted state profile for every clone.

    Parameters
    ----------
    adata
        AnnData with ``obs[clone_key]`` and an expression embedding.
    clone_key
        ``obs`` column with clone ids (missing = no BCR). Clone ids must be
        unique across donors (see :func:`threadfin.define_clones`).
    basis
        ``obsm`` key of the embedding. Use a batch-integrated embedding built
        without immunoglobulin genes (:func:`threadfin.pp.prepare_embedding`).
    context_key
        ``obs`` column of sampling contexts whose mean is removed before
        profiling (e.g. sample, or donor x timepoint). Choose *nuisance*
        strata only: biology you want to compare between clones (tissue,
        sort gate, infection status) must not be a context.
    donor_key
        Optional donor column, recorded per clone for stratified tests.
    representation
        ``"mean"`` (shrunken centroid) or ``"kernel"`` (shrunken kernel mean
        embedding from ``n_features`` random Fourier features, reduced to
        ``n_components`` principal components).
    bandwidth
        Kernel bandwidth: ``"median"`` (median distance between cells, the
        standard heuristic), ``"local"`` (distance to the 30th nearest
        neighbour; sharper, but needs larger clones) or a number.
    smooth
        Average each cell with its ``smooth`` nearest cells before profiling
        (denoises small clones by borrowing strength from their local state).
        Default: 15 for ``"kernel"``, 0 for ``"mean"``. The coherence test and
        clonal memory always use unsmoothed cells.
    min_cells
        Clones with fewer cells are not profiled.

    Returns
    -------
    Clone table (one row per profiled clone) with ``n_cells``,
    ``reliability`` and, when available, ``donor`` and ``n_contexts``.
    Profiles and model parameters go to ``adata.uns['threadfin']['profiles']``.
    """
    if smooth is None:
        smooth = 15 if representation == "kernel" else 0
    model = build_model(
        adata, clone_key=clone_key, basis=basis, context_key=context_key,
        representation=representation, n_features=n_features, bandwidth=bandwidth,
        smooth=smooth, random_state=random_state,
    )
    blup, n_eff, rel = model.group_blups(model.clone_codes, len(model.clone_index))
    keep = n_eff >= min_cells
    if keep.sum() < 3:
        raise ValueError(f"Only {int(keep.sum())} clones have >= {min_cells} cells.")
    ids = pd.Index(model.clone_index[keep].astype(str), name=clone_key)
    feats = blup[keep]
    names = [f"f{j}" for j in range(feats.shape[1])]
    reduction = None
    if representation == "kernel":
        # PCA of the clone x feature matrix keeps the downstream graph compact
        raw_mean = feats.mean(axis=0)
        u, s, vt = np.linalg.svd(feats - raw_mean, full_matrices=False)
        k = int(min(n_components, int((s > 1e-12).sum())))
        feats = u[:, :k] * s[:k]
        names = [f"kpc{j}" for j in range(k)]
        reduction = {"components": vt[:k], "raw_mean": raw_mean}

    table = pd.DataFrame({"n_cells": n_eff[keep].astype(int), "reliability": rel[keep]}, index=ids)
    obs = adata.obs
    if donor_key is not None:
        require_obs(adata, donor_key)
        sub = obs[[clone_key, donor_key]].dropna()
        donor = sub.groupby(sub[clone_key].astype(str), observed=True)[donor_key].agg(modal)
        table["donor"] = donor.reindex(table.index).astype(str).values
    if context_key is not None:
        sub = obs[[clone_key, context_key]].dropna()
        nctx = sub.groupby(sub[clone_key].astype(str), observed=True)[context_key].nunique()
        table["n_contexts"] = nctx.reindex(table.index).fillna(0).astype(int).values

    vc = model.vc
    get_uns(adata)[PROFILE_KEY] = {
        "features": pd.DataFrame(feats, index=table.index, columns=names),
        "clone_table": table,
        "variance_components": vc.as_dict(),
        "params": {**model.params, "donor_key": donor_key, "min_cells": int(min_cells),
                   "n_components": int(n_components)},
        "reduction": reduction,
    }
    log(
        f"clone profiles: {table.shape[0]} clones (>= {min_cells} cells); clonal ICC = "
        f"{vc.icc:.3f} ({vc.n_clones} clones, {vc.n_cells} cells); reliability >= 0.5 "
        f"needs >= {vc.min_cells_for(0.5)} cells per clone.",
        verbose,
    )
    return table


def get_profiles(adata) -> dict:
    """Stored profile results (raises if :func:`clone_profiles` was not run)."""
    prof = get_uns(adata).get(PROFILE_KEY)
    if prof is None:
        raise ValueError("Run threadfin.tl.clone_profiles() (or threadfin.run()) first.")
    return prof


def model_from_adata(adata, *, unsmoothed: bool = False, representation: str | None = None) -> ProfileModel:
    """Rebuild the :class:`ProfileModel` used by :func:`clone_profiles`.

    ``unsmoothed=True`` skips neighbour averaging and ``representation``
    overrides the stored representation; both are used by statistics that
    must treat cells as independent draws (coherence, memory).
    """
    p = get_profiles(adata)["params"]
    return build_model(
        adata, clone_key=p["clone_key"], basis=p["basis"], context_key=p["context_key"],
        representation=representation or p["representation"], n_features=p["n_features"],
        bandwidth=p.get("bandwidth", "median"), smooth=0 if unsmoothed else p.get("smooth", 0),
        random_state=p["random_state"],
    )


def project_features(adata, blup: np.ndarray) -> np.ndarray:
    """Map raw BLUPs (bootstrap replicates, time snapshots) into profile space."""
    red = get_profiles(adata).get("reduction")
    if red is None:
        return blup
    return (blup - np.asarray(red["raw_mean"])) @ np.asarray(red["components"]).T
