"""Sampling-aware clone state proportions on a frozen reference.

Integer counts in fixed Voronoi regions have a multinomial likelihood with a
Dirichlet prior. Credible intervals condition on the reference and partition;
they do not include reference uncertainty, donor effects or capture bias.
A fixed Gaussian mixture provides an optional smooth display, not a fitted
cell-level mixture model. This module uses no graph diffusion, RFF or PCA.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp
from scipy.stats import beta
from sklearn.cluster import KMeans
from ._utils import get_basis, get_uns, require_obs, require_positive_int


def _matrix(x, n_features=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Provide a non-empty, finite cell-by-feature matrix.")
    if n_features is not None and x.shape[1] != n_features:
        raise ValueError("Query and reference must use the same frozen feature coordinates.")
    return x


def _labels(values, n, name):
    s = pd.Series(values)
    if len(s) != n or s.isna().any() or s.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} needs one non-missing, non-blank label per cell.")
    unique = pd.Index(pd.unique(s))
    if not unique.astype(str).is_unique:
        raise ValueError(f"{name} must remain unique when converted to strings.")
    return s.astype(str).to_numpy()


@dataclass
class DensityResult:
    """Posterior region weights and conditional marginal intervals, one row per clone."""
    proportions: pd.DataFrame
    raw_proportions: pd.DataFrame
    counts: pd.DataFrame
    background: pd.DataFrame
    posterior_alpha: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    sampling_lower: pd.DataFrame
    sampling_upper: pd.DataFrame
    clone_table: pd.DataFrame
    settings: dict

    def to_uns(self):
        """Tables and metadata suitable for AnnData serialization."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def contrast(self, left, right, *, n_draws=4000, random_state=0):
        """Posterior differences (left minus right), conditional on the reference.

        Intervals are marginal, without multiplicity correction. The posterior
        probability of a positive difference is not a p-value. Dependence due
        to a shared, uncertain reference is not modelled.
        """
        require_positive_int(n_draws, "n_draws")
        left, right = str(left), str(right)
        a, b = self.posterior_alpha.loc[left].to_numpy(), self.posterior_alpha.loc[right].to_numpy()
        rng = np.random.default_rng(random_state)
        delta = (rng.dirichlet(a, n_draws) - rng.dirichlet(b, n_draws)
                 if left != right else np.zeros((n_draws, len(a))))
        tail = (1 - self.settings["credible_level"]) / 2
        return pd.DataFrame({
            "difference": self.proportions.loc[left] - self.proportions.loc[right],
            "lower": np.quantile(delta, tail, axis=0),
            "upper": np.quantile(delta, 1 - tail, axis=0),
            "probability_positive": (delta > 0).mean(axis=0),
        }, index=self.proportions.columns)


class StateDensityModel:
    """Reusable regions and a clone-balanced reference background.

    Freeze RNA preprocessing as well as this model for inductive validation.
    Reusing an embedding fitted on query cells is transductive. When a query
    clone occurs in the reference, its entire contribution is excluded from
    the background, including all contexts. Region centres remain fixed; this
    does not constitute a refit without that clone. Use disjoint reference
    clones for a fully independent evaluation.
    """

    @classmethod
    def fit(cls, x, clone_ids, *, contexts=None, n_states=16,
            prior_strength=2.0, background_floor=0.05,
            scales=(0.5, 1.0, 2.0), random_state=0, batch_size=4096):
        """Fit once, optionally choosing prior strength on reference split cells.

        The default prior strength 2.0 is fixed. 'auto' chooses from a fixed
        grid using clone-balanced held-out-cell predictive log scores and
        leave-own-clone-out backgrounds. It never uses outcome labels.
        A small uniform background component keeps every region positive.
        """
        x = _matrix(x)
        require_positive_int(n_states, "n_states")
        require_positive_int(batch_size, "batch_size")
        if not 2 <= n_states <= len(x):
            raise ValueError("n_states must be between 2 and the number of reference cells.")
        clones = _labels(clone_ids, len(x), "clone_ids")
        ctx = np.repeat("all", len(x)) if contexts is None else _labels(contexts, len(x), "contexts")
        if np.unique(clones).size < 2:
            raise ValueError("At least two reference clones are required for leave-clone-out backgrounds.")
        if not np.isfinite(background_floor) or not 0 < background_floor <= 1:
            raise ValueError("background_floor must be in (0, 1].")
        scales = np.asarray(scales, dtype=float)
        if scales.ndim != 1 or not scales.size or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("scales must be a non-empty vector of positive finite numbers.")
        if prior_strength != "auto":
            if not np.isfinite(float(prior_strength)) or float(prior_strength) <= 0:
                raise ValueError("prior_strength must be positive or 'auto'.")
        obj = cls()
        obj.n_states, obj.n_features, obj.batch_size = int(n_states), x.shape[1], int(batch_size)
        obj.background_floor, obj.scales = float(background_floor), scales.copy()
        obj.random_state = int(random_state)
        _, inverse, sizes = np.unique(clones, return_inverse=True, return_counts=True)
        km = KMeans(n_clusters=n_states, n_init=10, random_state=random_state)
        km.fit(x, sample_weight=1.0 / sizes[inverse])
        obj.centres = np.asarray(km.cluster_centers_)
        if np.unique(obj.centres, axis=0).shape[0] < n_states:
            raise ValueError("Reference has fewer distinct regions than n_states; reduce n_states.")
        states, distances = obj._assign(x, return_distances=True)
        positive = distances[distances > 0]
        obj.bandwidth = float(np.median(positive)) if positive.size else float(np.median(pdist(obj.centres)))
        if obj.bandwidth <= 0:
            raise ValueError("Reference has zero spatial spread.")
        obj.reference = obj._aggregate(states, clones, ctx)
        obj._index_reference()
        obj.calibration = pd.DataFrame(columns=["prior_strength", "mean_log_score", "n_profiles"])
        obj.prior_method = "reference_split_cell" if prior_strength == "auto" else "fixed"
        obj.prior_strength = 2.0 if prior_strength == "auto" else float(prior_strength)
        if prior_strength == "auto":
            obj._calibrate(states, clones, ctx)
        return obj

    @property
    def state_names(self):
        return [f"state_{i}" for i in range(self.n_states)]

    def _assign(self, x, *, return_distances=False):
        states = np.empty(len(x), dtype=np.int64)
        distances = np.empty(len(x)) if return_distances else None
        for start in range(0, len(x), self.batch_size):
            stop = min(start + self.batch_size, len(x))
            d2 = cdist(x[start:stop], self.centres, metric="sqeuclidean")
            states[start:stop] = d2.argmin(axis=1)
            if return_distances:
                distances[start:stop] = np.sqrt(d2[np.arange(stop-start), states[start:stop]])
        return (states, distances) if return_distances else states

    def _aggregate(self, states, clones, contexts):
        keys = pd.MultiIndex.from_arrays([clones, contexts], names=["clone", "context"])
        code, levels = pd.factorize(keys, sort=True)
        counts = np.bincount(code * self.n_states + states,
                             minlength=len(levels) * self.n_states).reshape(-1, self.n_states)
        out = levels.to_frame(index=False)
        out.columns = ["clone", "context"]
        out[self.state_names] = counts
        return out

    def _index_reference(self):
        cols = self.state_names
        self._global = self.reference.groupby("clone", sort=True)[cols].sum()
        self._global = self._global.div(self._global.sum(axis=1), axis=0)
        ref = self.reference.copy()
        ref[cols] = ref[cols].div(ref[cols].sum(axis=1), axis=0)
        self._context = {name: group.set_index("clone")[cols]
                         for name, group in ref.groupby("context", sort=True)}
        self._global_sum = self._global.sum(axis=0).to_numpy()
        self._context_sum = {key: value.sum(axis=0).to_numpy() for key, value in self._context.items()}

    def _background(self, clone, context):
        tab = self._context.get(context)
        fallback = tab is None or len(tab) - int(clone in tab.index) < 1
        if fallback:
            tab, total = self._global, self._global_sum.copy()
        else:
            total = self._context_sum[context].copy()
        count, excluded = len(tab), clone in tab.index
        if excluded:
            total -= tab.loc[clone].to_numpy()
            count -= 1
        if count < 1:
            raise ValueError("No independent reference clone remains for this query.")
        q = np.maximum(total / count, 0)
        q = (1 - self.background_floor) * q + self.background_floor / self.n_states
        return q / q.sum(), fallback, excluded, count

    def _calibrate(self, states, clones, contexts):
        rng = np.random.default_rng(self.random_state + 104729)
        groups = pd.DataFrame({"clone": clones, "context": contexts}).groupby(
            ["clone", "context"], sort=True).indices
        grid = np.array([0.5, 1., 2., 5., 10., 20., 50., 100.])
        scores = []
        for (clone, context), idx in groups.items():
            if len(idx) < 4:
                continue
            idx = rng.permutation(idx)
            cut = len(idx) // 2
            train = np.bincount(states[idx[:cut]], minlength=self.n_states)
            test = np.bincount(states[idx[cut:]], minlength=self.n_states)
            q, _, _, _ = self._background(clone, context)
            pred = (train[None, :] + grid[:, None] * q) / (cut + grid[:, None])
            scores.append((clone, (np.log(pred) * test).sum(axis=1) / test.sum()))
        if len({c for c, _ in scores}) < 6:
            raise ValueError("Automatic prior calibration needs at least six reference clones with >=4 cells.")
        balanced = pd.DataFrame([row for _, row in scores], index=[c for c, _ in scores]).groupby(level=0).mean()
        mean_score = balanced.mean(axis=0).to_numpy()
        self.prior_strength = float(grid[np.argmax(mean_score)])
        self.calibration = pd.DataFrame({"prior_strength": grid, "mean_log_score": mean_score,
                                         "n_profiles": len(balanced)})

    def metadata(self):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.centres).tobytes())
        digest.update(self.reference.to_csv(index=False).encode())
        params = {"n_states": self.n_states, "n_features": self.n_features,
                  "prior_strength": self.prior_strength, "prior_method": self.prior_method,
                  "background_floor": self.background_floor,
                  "bandwidth": self.bandwidth, "scales": self.scales.tolist(), "random_state": self.random_state}
        digest.update(json.dumps(params, sort_keys=True).encode())
        return {**params, "reference_sha256": digest.hexdigest(),
                "interval_scope": "conditional marginal Dirichlet; fixed partition and reference",
                "smoothing_scope": "fixed Gaussian-mixture display; no graph diffusion",
                "reference_weighting": "one vote per clone; own clone excluded from background"}

    def transform(self, x, clone_ids, *, contexts=None, credible_level=0.95):
        """Estimate proportions without refitting reference or preprocessing."""
        return self.transform_batches([(x, clone_ids, contexts)], credible_level=credible_level)

    def transform_batches(self, batches, *, credible_level=0.95):
        """Stream (embedding, clone_ids, contexts_or_None) batches.

        Clones may span batches and contexts. Counts are accumulated before
        posterior estimation. Auxiliary memory is O(batch_size * n_states +
        n_clone_contexts * n_states), excluding input and retained reference.
        """
        if not np.isfinite(credible_level) or not 0 < credible_level < 1:
            raise ValueError("credible_level must be strictly between zero and one.")
        totals = {}
        for x, clone_ids, contexts in batches:
            x = _matrix(x, self.n_features)
            clones = _labels(clone_ids, len(x), "clone_ids")
            ctx = np.repeat("all", len(x)) if contexts is None else _labels(contexts, len(x), "contexts")
            aggregate = self._aggregate(self._assign(x), clones, ctx)
            for row in aggregate.itertuples(index=False, name=None):
                key, count = (row[0], row[1]), np.asarray(row[2:], dtype=np.int64)
                totals[key] = totals.get(key, np.zeros(self.n_states, dtype=np.int64)) + count
        if not totals:
            raise ValueError("No query cells were supplied.")
        grouped = {}
        for (clone, context), counts in sorted(totals.items()):
            grouped.setdefault(clone, []).append((context, counts))
        ids, counts, backgrounds, rows = [], [], [], []
        for clone, parts in grouped.items():
            n = sum(int(c.sum()) for _, c in parts)
            raw_count = sum((c for _, c in parts), np.zeros(self.n_states, dtype=np.int64))
            bg = np.zeros(self.n_states)
            fallback_n, excluded_n, reference_n = 0, 0, []
            for ctx, c in parts:
                q, fallback, excluded, n_ref = self._background(clone, ctx)
                bg += c.sum() * q / n
                fallback_n += int(c.sum()) * int(fallback)
                excluded_n += int(c.sum()) * int(excluded)
                reference_n.append(n_ref)
            ids.append(clone)
            counts.append(raw_count)
            backgrounds.append(bg)
            rows.append({"n_cells": n, "n_contexts": len(parts),
                         "data_weight": n / (n + self.prior_strength),
                         "global_background_fraction": fallback_n / n,
                         "own_clone_excluded_fraction": excluded_n / n,
                         "min_reference_clones": min(reference_n)})
        index = pd.Index(ids, name="clone_id")
        counts, backgrounds = np.asarray(counts), np.asarray(backgrounds)
        n = counts.sum(axis=1, keepdims=True)
        alpha = counts + self.prior_strength * backgrounds
        total = alpha.sum(axis=1, keepdims=True)
        tail = (1 - credible_level) / 2
        # Exact binomial intervals with Bonferroni correction over this clone's
        # fixed regions. Unlike posterior intervals these retain coverage at
        # pure/rare-state boundaries under independent multinomial sampling.
        sampling_tail = tail / self.n_states
        sampling_lower = np.zeros_like(alpha)
        sampling_upper = np.ones_like(alpha)
        positive, incomplete = counts > 0, counts < n
        sampling_lower[positive] = beta.ppf(sampling_tail, counts, n-counts+1)[positive]
        sampling_upper[incomplete] = beta.ppf(1-sampling_tail, counts+1, n-counts)[incomplete]
        make = lambda values: pd.DataFrame(values, index=index, columns=self.state_names)
        return DensityResult(
            make(alpha / total), make(counts / n), make(counts), make(backgrounds), make(alpha),
            make(beta.ppf(tail, alpha, total - alpha)),
            make(beta.ppf(1 - tail, alpha, total - alpha)),
            make(sampling_lower), make(sampling_upper),
            pd.DataFrame(rows, index=index),
            {**self.metadata(), "credible_level": float(credible_level),
             "sampling_interval_scope": "exact Bonferroni simultaneous regions within each clone; iid counts"},

        )

    def annotate_regions(self, x, values, clone_ids):
        """Describe fixed regions with numeric gene/module/annotation values.

        Rows of values must correspond to rows of x. Cells are first averaged
        within each (clone, region), then clones are weighted equally within
        the region. This is a descriptive annotation, with no p-value or
        independent validation claim; it cannot change the fitted partition.
        """
        x = _matrix(x, self.n_features)
        clones = _labels(clone_ids, len(x), "clone_ids")
        values = pd.DataFrame(values).reset_index(drop=True)
        if len(values) != len(x) or not values.columns.is_unique:
            raise ValueError("values must have one aligned row per cell and unique columns.")
        numeric = values.to_numpy(dtype=float)
        if numeric.ndim != 2 or not numeric.shape[1] or not np.isfinite(numeric).all():
            raise ValueError("Region annotations must contain finite numeric values.")
        states = self._assign(x)
        # Grouping keys are separate from feature columns, avoiding name collisions.
        keys = pd.MultiIndex.from_arrays([clones, states], names=["clone", "region"])
        frame = pd.DataFrame(numeric, index=keys, columns=values.columns)
        per_clone = frame.groupby(level=["clone", "region"], sort=True).mean()
        means = per_clone.groupby(level="region").mean().reindex(range(self.n_states))
        coverage = pd.DataFrame({
            "n_cells": np.bincount(states, minlength=self.n_states),
            "n_clones": per_clone.groupby(level="region").size().reindex(range(self.n_states), fill_value=0),
        })
        means.index = coverage.index = pd.Index(self.state_names, name="region")
        return {"values": means, "coverage": coverage}

    def log_density(self, points, proportions):
        """Log density of one clone's normalised Gaussian-mixture display.

        Scales have equal weights. Values depend on embedding coordinates;
        they cannot be compared across independently fitted embeddings.
        """
        x = _matrix(points, self.n_features)
        p = np.asarray(proportions, dtype=float)
        if p.shape != (self.n_states,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError("proportions must be a nonnegative probability vector in state order.")
        logp = np.full(self.n_states, -np.inf)
        np.log(p, out=logp, where=p > 0)
        out = np.empty(len(x))
        for start in range(0, len(x), self.batch_size):
            d2 = cdist(x[start:start+self.batch_size], self.centres, metric="sqeuclidean")
            by_scale = []
            for scale in self.scales:
                variance = (self.bandwidth * scale) ** 2
                logkernel = -0.5 * (d2 / variance + self.n_features * np.log(2 * np.pi * variance))
                by_scale.append(logsumexp(logkernel + logp, axis=1))
            out[start:start+len(d2)] = logsumexp(by_scale, axis=0) - np.log(len(self.scales))
        return out

    def save(self, path):
        """Portable NPZ model without pickle or executable estimator state."""
        params = {**self.metadata(), "batch_size": self.batch_size}
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, centres=self.centres, parameters=json.dumps(params),
                                reference_clones=self.reference["clone"].to_numpy(dtype=str),
                                reference_contexts=self.reference["context"].to_numpy(dtype=str),
                                reference_counts=self.reference[self.state_names].to_numpy(),
                                calibration=self.calibration.to_numpy(dtype=float))

    @classmethod
    def load(cls, path):
        """Read a saved model with pickle disabled and verify its checksum."""
        with np.load(path, allow_pickle=False) as saved:
            params = json.loads(str(saved["parameters"]))
            obj = cls()
            for key in ("n_states", "n_features", "prior_strength", "prior_method", "background_floor",
                        "bandwidth", "random_state", "batch_size"):
                setattr(obj, key, params[key])
            obj.scales, obj.centres = np.asarray(params["scales"]), saved["centres"].copy()
            obj.reference = pd.DataFrame({"clone": saved["reference_clones"],
                                          "context": saved["reference_contexts"]})
            obj.reference[obj.state_names] = saved["reference_counts"]
            obj.calibration = pd.DataFrame(saved["calibration"],
                                           columns=["prior_strength", "mean_log_score", "n_profiles"])
        obj._index_reference()
        if obj.metadata()["reference_sha256"] != params["reference_sha256"]:
            raise ValueError("Saved reference checksum does not match its data.")
        return obj


def clone_densities(adata, *, reference_model, clone_key="clone_id", basis="X_pca",
                    context_key=None, credible_level=0.95, key_added="densities"):
    """AnnData wrapper requiring an explicitly fitted StateDensityModel.

    Cells without a clone are omitted. Reference and query must share feature
    coordinates. Results are separate from existing mean/kernel profiles.
    """
    require_obs(adata, clone_key, context_key, context="clone densities")
    valid = adata.obs[clone_key].notna().to_numpy()
    x = get_basis(adata, basis)[valid]
    contexts = adata.obs.loc[valid, context_key] if context_key is not None else None
    result = reference_model.transform(x, adata.obs.loc[valid, clone_key],
                                       contexts=contexts, credible_level=credible_level)
    get_uns(adata)[key_added] = result.to_uns()
    return result
