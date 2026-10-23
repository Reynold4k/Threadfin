"""Automatic choice of the clone-map layout.

The clone map (:func:`threadfin.tl.embed_clones`) is a UMAP of the clone profiles. Two settings decide
what it looks like: ``n_neighbors``, the number of profile neighbours each clone is tied to, and
``min_dist``, how closely clones may pack on the map. Too few neighbours or too little spacing breaks a
continuum of clones into separate islands; too many smooths away local structure. The best values differ
between datasets, because the number of clones and the shape of their distribution differ.
:func:`tune_clone_map` chooses both settings from the data, so that the map is as faithful to the clone
profiles as possible while holding together as one picture.

What is scored
--------------
Each layout is scored on three properties that need no annotation (:func:`map_quality`):

* **coherence**: the fraction of clones in the largest connected piece of the map. Two clones are
  connected when they lie within ``1.5 D / sqrt(N)`` of each other (``D``, the extent of the map; ``N``,
  the number of clones). At Scanpy's default point size (120,000 / N), this is roughly the distance at
  which two drawn points touch, so a piece is what a reader sees as one island.
* **local fidelity**: the fraction of each clone's 30 nearest neighbours in profile space that are also
  among its 30 nearest neighbours on the map (kNN preservation; Kobak & Berens 2019).
* **global fidelity**: the Spearman correlation between profile-space and map distances over random
  pairs of clones (Kobak & Berens 2019).

The score is the mean of the two fidelities, minus a penalty when coherence falls below a target::

    score = (local + global) / 2 - penalty * max(0, target - coherence)

By default the target is set by the clone profiles themselves: the fraction of clones in the largest
connected component of the mutual 15-nearest-neighbour graph of profiles (the graph scale used for
programme discovery), capped at 0.99. A map is therefore not allowed to be more fragmented than the
profiles it shows, while a small group of clones that is genuinely separate in profile space may stay
apart.

How the settings are searched
-----------------------------
A UMAP layout is random. The same settings can give a connected map with one seed and a broken one with
another, and the settings that are most faithful on one layout are often the ones closest to breaking
apart. The search is therefore built around the worst case rather than the typical layout.

1. **Ladder.** The clones are laid out at every combination of a ladder of neighbour counts (10, 18, 32,
   56, 100 and 180, about 1.8-fold apart and at most half the number of clones) and of ``min_dist``
   (0.3, 0.55, 0.8 and 0.95), each with two random seeds shared by all settings. A setting is credited
   with its mean fidelity and with the *lowest* coherence among its layouts.
2. **Smoothing.** Each setting's score is averaged with those of its neighbours on the ladder (Gaussian
   weights), so that a setting surrounded by good settings is preferred to an isolated peak at the edge
   where the map starts to break apart.
3. **Confirmation.** The best settings are laid out with four further seeds and re-scored on all six
   layouts.
4. **Choice.** Among the confirmed settings that held together in every layout, the one with the best
   smoothed score wins, recomputed with all layouts so far, so that a setting whose neighbours on the
   ladder break apart is not chosen even if it held together itself. Settings within ``tolerance`` of
   the best are treated as equally good, and the most coherent of these is chosen.
5. **Rescue.** If the winner still falls short of the target in one of its layouts, the most coherent
   settings seen so far are confirmed as well. If none of them holds together either, the most coherent
   settings are laid out again with weaker repulsion between clones (UMAP ``repulsion_strength`` 0.5, then
   0.25). Weaker repulsion moves UMAP along the attraction-repulsion spectrum towards more continuous
   layouts (Böhm, Berens & Kobak 2022), which keeps weakly connected groups of clones together.

Layouts during the search use 200 optimisation epochs, which scores like the full layout at a fraction of
the cost; the final map is laid out at UMAP's defaults. Above ``max_clones`` clones, the search runs on a
random subset, ``n_neighbors`` is scaled up to the full data in proportion to the number of clones (which
keeps the neighbourhood the same size in profile space), and the best settings are confirmed again on all
clones before one is chosen.

``method="adam"`` replaces steps 1 and 2 with a stochastic-gradient search: Adam (Kingma & Ba 2015) on
``log n_neighbors`` and ``logit min_dist``, with gradients estimated by simultaneous perturbation (SPSA;
Spall 1992) from pairs of layouts that share a seed. It then goes through the same confirmation, choice
and rescue. The ladder is the default because it explores the whole range of settings and its choice
varies less between random seeds.

References
----------
Böhm, J. N., Berens, P. & Kobak, D. Attraction-repulsion spectrum in neighbor embeddings. J. Mach. Learn.
Res. 23, 1-32 (2022).
Kingma, D. P. & Ba, J. Adam: a method for stochastic optimization. ICLR (2015).
Kobak, D. & Berens, P. The art of using t-SNE for single-cell transcriptomics. Nat. Commun. 10, 5416 (2019).
Spall, J. C. Multivariate stochastic approximation using a simultaneous perturbation gradient
approximation. IEEE Trans. Autom. Control 37, 332-341 (1992).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd

from ._utils import log

#: ladder of settings searched by the default method
N_NEIGHBORS_LADDER = (10, 18, 32, 56, 100, 180)
MIN_DIST_LADDER = (0.3, 0.55, 0.8, 0.95)
#: weaker repulsion tried, in this order, when no standard setting holds together
REPULSION_RESCUE = (0.5, 0.25)
#: settings never leave this box
N_NEIGHBORS_RANGE = (5, 250)
MIN_DIST_RANGE = (0.0, 0.99)


# --------------------------------------------------------------------------- scoring a layout


def _extent(xy: np.ndarray) -> float:
    """Side of the map: the larger of the two axis ranges, ignoring the outer 0.5% of clones."""
    lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
    return float(np.max(hi - lo))


def _coherence(xy: np.ndarray, touch: float = 1.5) -> float:
    """Fraction of clones in the largest piece of the map (single linkage at ``touch * D / sqrt(N)``)."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph

    n = len(xy)
    radius = touch * _extent(xy) / np.sqrt(n)
    graph = radius_neighbors_graph(xy, radius=max(radius, 1e-12), mode="connectivity")
    _, labels = connected_components(graph, directed=False)
    return float(np.bincount(labels).max() / n)


def _without_self(idx: np.ndarray, k: int) -> np.ndarray:
    """Each row's first ``k`` neighbours, with the row's own index removed wherever it appears."""
    keep = idx != np.arange(len(idx))[:, None]
    order = np.argsort(~keep, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :k]


class _Reference:
    """Everything about the clone profiles that every layout is compared with, computed once."""

    def __init__(self, features: np.ndarray, *, k_max: int, k_local: int = 30, n_pairs: int = 20000,
                 random_state: int = 0):
        from sklearn.neighbors import NearestNeighbors

        self.features = np.ascontiguousarray(features, dtype=np.float32)
        n = len(self.features)
        self.k_local = int(max(5, min(k_local, n // 4)))
        # exact neighbours, self included as the first column, as UMAP expects them
        k_all = int(min(max(k_max, self.k_local + 1, 16), n - 1))
        dist, idx = NearestNeighbors(n_neighbors=k_all).fit(self.features).kneighbors(self.features)
        self.knn_dist, self.knn_idx = dist.astype(np.float32), idx.astype(np.int64)
        self.local_idx = _without_self(self.knn_idx, self.k_local)
        rng = np.random.default_rng(random_state)
        i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
        keep = i != j
        self.pairs = (i[keep], j[keep])
        self.profile_dist = np.linalg.norm(self.features[i[keep]] - self.features[j[keep]], axis=1)

    def intrinsic_coherence(self, k: int = 15) -> float:
        """Fraction of clones in the largest component of the mutual k-nearest-neighbour graph of profiles."""
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        n = len(self.features)
        nb = _without_self(self.knn_idx, int(min(k, self.knn_idx.shape[1] - 1)))
        rows = np.repeat(np.arange(n), nb.shape[1])
        g = sp.csr_matrix((np.ones(rows.size), (rows, nb.ravel())), shape=(n, n))
        _, labels = connected_components(g.multiply(g.T), directed=False)
        return float(np.bincount(labels).max() / n)

    def layout(self, n_neighbors: int, min_dist: float, spread: float, seed: int, repulsion: float = 1.0,
               n_epochs: int | None = None) -> np.ndarray:
        import umap

        k = int(max(2, min(n_neighbors, self.knn_idx.shape[1])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = umap.UMAP(n_neighbors=k, min_dist=float(min_dist), spread=spread, random_state=int(seed),
                              repulsion_strength=float(repulsion), n_epochs=n_epochs,
                              precomputed_knn=(self.knn_idx[:, :k].copy(), self.knn_dist[:, :k].copy()))
            return model.fit_transform(self.features)

    def quality(self, xy: np.ndarray, target: float, penalty: float) -> dict:
        from scipy.stats import spearmanr
        from sklearn.neighbors import NearestNeighbors

        k = self.k_local
        near = _without_self(NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False), k)
        local = float((near[:, :, None] == self.local_idx[:, None, :]).any(axis=2).mean())
        i, j = self.pairs
        glob = float(spearmanr(self.profile_dist, np.linalg.norm(xy[i] - xy[j], axis=1)).statistic)
        coh = _coherence(xy)
        score = 0.5 * (local + glob) - penalty * max(0.0, target - coh)
        return {"score": score, "coherence": coh, "local_fidelity": local, "global_fidelity": glob}


def _target(ref: _Reference, coherence_target) -> float:
    if isinstance(coherence_target, str):
        if coherence_target != "auto":
            raise ValueError("coherence_target must be 'auto' or a number between 0 and 1.")
        return min(0.99, ref.intrinsic_coherence())
    if not isinstance(coherence_target, Real) or not 0 < coherence_target <= 1:
        raise ValueError("coherence_target must be 'auto' or a number between 0 and 1.")
    return float(coherence_target)


def map_quality(features, xy, *, coherence_target: float | str = "auto", penalty: float = 5.0,
                random_state: int = 0) -> dict:
    """Score a clone map: coherence, local and global fidelity, and the tuner's combined score.

    Parameters
    ----------
    features
        Clone profiles, one row per clone (``adata.uns['threadfin']['profiles']['features']``).
    xy
        The map, one row per clone in the same order.
    coherence_target, penalty
        As in :func:`tune_clone_map`.

    Returns
    -------
    ``dict`` with ``coherence`` (fraction of clones in the largest piece), ``local_fidelity`` (30-NN
    preservation), ``global_fidelity`` (Spearman correlation of distances), ``score`` and the
    ``coherence_target`` used.
    """
    feats = np.asarray(features, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if feats.shape[0] != xy.shape[0]:
        raise ValueError("features and xy must have one row per clone, in the same order.")
    if feats.shape[0] < 15:
        raise ValueError(f"Need at least 15 clones to score a clone map, got {feats.shape[0]}.")
    ref = _Reference(feats, k_max=31, random_state=random_state)
    target = _target(ref, coherence_target)
    return ref.quality(xy, target, penalty) | {"coherence_target": target}


# --------------------------------------------------------------------------- the search


@dataclass
class MapTuning:
    """Result of :func:`tune_clone_map`."""

    n_neighbors: int
    min_dist: float
    repulsion_strength: float
    score: float
    coherence: float
    local_fidelity: float
    global_fidelity: float
    coherence_target: float
    method: str
    n_layouts: int
    seconds: float
    history: pd.DataFrame = field(repr=False)
    candidates: pd.DataFrame = field(repr=False)

    def to_uns(self) -> dict:
        """Record for ``adata.uns`` (tables stay DataFrames, so the AnnData can be written to h5ad)."""
        rec = {k: v for k, v in self.as_dict().items() if k not in ("history", "candidates")}
        return rec | {"history": self.history.copy(), "candidates": self.candidates.copy()}

    def as_dict(self) -> dict:
        """JSON-friendly record of the search."""
        return {"n_neighbors": int(self.n_neighbors), "min_dist": float(self.min_dist),
                "repulsion_strength": float(self.repulsion_strength), "score": float(self.score),
                "coherence": float(self.coherence), "local_fidelity": float(self.local_fidelity),
                "global_fidelity": float(self.global_fidelity), "coherence_target": float(self.coherence_target),
                "method": self.method, "n_layouts": int(self.n_layouts), "seconds": float(self.seconds),
                "history": self.history.to_dict(orient="list"),
                "candidates": self.candidates.to_dict(orient="list")}


def _key(setting) -> tuple:
    k, md, rep = setting
    return int(k), round(float(md), 3), round(float(rep), 3)


def _layout_quality(ref: _Reference, key: tuple, spread: float, n_epochs, target: float, penalty: float) -> dict:
    """One layout at ``key = (n_neighbors, min_dist, repulsion, seed)`` and its score (runs in a worker)."""
    k, md, rep, seed = key
    xy = ref.layout(k, md, spread, seed, repulsion=rep, n_epochs=n_epochs)
    return ref.quality(xy, target, penalty)


def resolve_n_jobs(n_jobs) -> int:
    """``n_jobs`` as a positive number of processes (``-1``: every available core; ``None``: one)."""
    import os

    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0:
        raise ValueError(f"n_jobs must be a positive integer, -1 (all cores) or None, got {n_jobs!r}.")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                                  # macOS / Windows
        available = os.cpu_count() or 1
    return int(available + 1 + n_jobs) if n_jobs < 0 else int(min(n_jobs, available))


class _Search:
    """Lays clones out at requested settings, scores each layout once and keeps every result."""

    def __init__(self, ref: _Reference, target: float, penalty: float, spread: float, n_epochs, n_jobs: int = 1):
        self.ref, self.target, self.penalty, self.spread, self.n_epochs = ref, target, penalty, spread, n_epochs
        self.n_jobs = int(n_jobs)
        self.cache: dict = {}
        self.rows: list = []

    def evaluate(self, setting, seed: int, stage: str) -> dict:
        self.evaluate_many([(setting, seed, stage)])
        return self.cache[_key(setting) + (int(seed),)]

    def evaluate_many(self, requests) -> None:
        """Lay out and score every requested ``(setting, seed, stage)`` not seen before.

        With ``n_jobs > 1`` the layouts are computed in parallel processes. A layout depends only on its
        setting and seed, and results are stored in the order requested, so the outcome is identical for
        any ``n_jobs``.
        """
        todo, planned = [], set()
        for setting, seed, stage in requests:
            key = _key(setting) + (int(seed),)
            if key not in self.cache and key not in planned:
                planned.add(key)
                todo.append((key, stage))
        if not todo:
            return
        args = (self.spread, self.n_epochs, self.target, self.penalty)
        if self.n_jobs > 1 and len(todo) > 1:
            from joblib import Parallel, delayed

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = Parallel(n_jobs=min(self.n_jobs, len(todo)))(
                    delayed(_layout_quality)(self.ref, key, *args) for key, _ in todo)
        else:
            results = [_layout_quality(self.ref, key, *args) for key, _ in todo]
        for (key, stage), q in zip(todo, results):
            self.cache[key] = q
            self.rows.append(dict(stage=stage, n_neighbors=key[0], min_dist=key[1], repulsion_strength=key[2],
                                  seed=key[3], **q))

    def count(self, setting) -> int:
        key = _key(setting)
        return sum(1 for kk in self.cache if kk[:3] == key)

    def robust(self, setting) -> dict:
        """Mean fidelity and worst coherence over every layout of a setting, and the resulting score."""
        key = _key(setting)
        qs = pd.DataFrame([q for kk, q in self.cache.items() if kk[:3] == key])
        local, glob = float(qs["local_fidelity"].mean()), float(qs["global_fidelity"].mean())
        coh = float(qs["coherence"].min())
        score = 0.5 * (local + glob) - self.penalty * max(0.0, self.target - coh)
        return dict(n_neighbors=key[0], min_dist=key[1], repulsion_strength=key[2], n_layouts=len(qs),
                    score=score, coherence=coh, coherence_mean=float(qs["coherence"].mean()),
                    local_fidelity=local, global_fidelity=glob)

    def seen(self) -> pd.DataFrame:
        keys = sorted({kk[:3] for kk in self.cache})
        return pd.DataFrame([self.robust(k) for k in keys])


def _smoothed(search: _Search, ks, mds, sigma: float) -> pd.DataFrame:
    """Robust score of every ladder setting (all its layouts so far), averaged with its ladder neighbours."""
    from scipy.ndimage import gaussian_filter

    raw = np.array([[search.robust((k, md, 1.0))["score"] for md in mds] for k in ks])
    # normalised smoothing: each setting is averaged with the ladder neighbours that exist (no padding)
    smooth = gaussian_filter(raw, sigma, mode="constant") / gaussian_filter(np.ones_like(raw), sigma, mode="constant")
    return pd.DataFrame([(int(k), round(float(md), 3), raw[i, j], smooth[i, j]) for i, k in enumerate(ks)
                         for j, md in enumerate(mds)], columns=["n_neighbors", "min_dist", "score", "smoothed"])


def _ladder_candidates(search: _Search, ks, mds, rng, n_seeds: int, top: int, sigma: float, verbose: bool):
    """Steps 1-2: every ladder setting with shared seeds, worst-case coherence, Gaussian smoothing."""
    seeds = [int(rng.integers(2**31 - 1)) for _ in range(n_seeds)]
    search.evaluate_many([((k, md, 1.0), s, "ladder") for k in ks for md in mds for s in seeds])
    tab = _smoothed(search, ks, mds, sigma)
    for _, r in tab.nlargest(3, "smoothed").iterrows():
        log(f"  ladder: n_neighbors {int(r['n_neighbors'])}, min_dist {r['min_dist']:.2f} -> score "
            f"{r['score']:.3f} (smoothed {r['smoothed']:.3f})", verbose)
    picks = pd.concat([tab.nlargest(top, "smoothed"), tab.nlargest(2, "score")])
    return list(dict.fromkeys((int(k), round(float(md), 3), 1.0)
                              for k, md in zip(picks["n_neighbors"], picks["min_dist"])))


def _adam_candidates(search: _Search, theta0, active, lo, hi, from_theta, rng, top: int, opts: dict,
                     verbose: bool):
    """Steps 1-2 for method='adam': a warm-up, then SPSA gradient estimates fed to Adam."""
    n_iter, patience = opts.get("n_iter", 30), opts.get("patience", 8)
    lr, c0 = opts.get("learning_rate", 0.25), opts.get("perturbation", 0.35)
    b1, b2 = opts.get("betas", (0.8, 0.99))
    target, theta = search.target, theta0.copy()
    for _ in range(opts.get("max_warmup", 6)):
        k_w, md_w = from_theta(theta)
        for _ in range(2):
            search.evaluate((k_w, md_w, 1.0), int(rng.integers(2**31 - 1)), "warm-up")
        coh = search.robust((k_w, md_w, 1.0))["coherence"]
        if coh >= target or (theta[active] >= hi[active] - 1e-9).all():
            break
        log(f"  warm-up: n_neighbors {k_w}, min_dist {md_w:.2f} keeps {100 * coh:.1f}% of clones in one "
            "piece; widening.", verbose)
        theta = np.clip(theta + np.array([np.log(2.0), 1.0]) * active, lo, hi)
    m, v = np.zeros(2), np.zeros(2)
    best, stall = -np.inf, 0
    for t in range(1, n_iter + 1):
        c_t = c0 / t ** 0.101
        delta = np.where(rng.random(2) < 0.5, -1.0, 1.0) * active
        seed = int(rng.integers(2**31 - 1))
        (k_p, md_p), (k_m, md_m) = from_theta(np.clip(theta + c_t * delta, lo, hi)), \
            from_theta(np.clip(theta - c_t * delta, lo, hi))
        s_p = search.evaluate((k_p, md_p, 1.0), seed, "adam")["score"]
        s_m = search.evaluate((k_m, md_m, 1.0), seed, "adam")["score"]
        # SPSA: one seed for both sides, so the difference is about the settings, not the seed
        grad = np.zeros(2)
        grad[active] = (s_p - s_m) / (2.0 * c_t * delta[active])
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        theta = np.clip(theta + lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8) * active, lo, hi)
        log(f"  Adam step {t:2d}: n_neighbors {k_p} / {k_m}, min_dist {md_p:.2f} / {md_m:.2f} -> score "
            f"{s_p:.3f} / {s_m:.3f}", verbose)
        if max(s_p, s_m) > best + 1e-3:
            best, stall = max(s_p, s_m), 0
        else:
            stall += 1
            if stall >= patience:
                break
    seen = search.seen().nlargest(top, "score")
    return list(zip(seen["n_neighbors"].astype(int), seen["min_dist"].round(3), seen["repulsion_strength"]))


def _confirm(search: _Search, candidates, rng, n_layouts: int, tolerance: float, ladder=None):
    """Steps 3-4: each candidate to ``n_layouts`` layouts, then the choice.

    Settings that held together in every layout are preferred. With a ladder (``(ks, mds, sigma)``), the
    choice uses the smoothed score, recomputed with every layout so far, so that a setting surrounded by
    settings that break apart loses to one in a robust neighbourhood. Ties go to the more coherent and
    then to the standard (stronger) repulsion.
    """
    requests, planned = [], {}
    for setting in candidates:
        key = _key(setting)
        need = max(0, n_layouts - search.count(setting) - planned.get(key, 0))
        planned[key] = planned.get(key, 0) + need
        requests += [(setting, int(rng.integers(2**31 - 1)), "confirm") for _ in range(need)]
    search.evaluate_many(requests)
    tab = pd.DataFrame([search.robust(s) for s in candidates])
    tab["smoothed"] = tab["score"]
    if ladder is not None:
        sm = _smoothed(search, *ladder).set_index(["n_neighbors", "min_dist"])["smoothed"]
        std = tab["repulsion_strength"] == 1.0
        idx = list(zip(tab.loc[std, "n_neighbors"], tab.loc[std, "min_dist"]))
        tab.loc[std, "smoothed"] = sm.reindex(idx).to_numpy()
        tab["smoothed"] = tab["smoothed"].fillna(tab["score"])
    tab = tab.sort_values("smoothed", ascending=False).reset_index(drop=True)
    pool = tab[tab["coherence"] >= search.target]
    pool = pool if len(pool) else tab
    tied = pool[pool["smoothed"] >= pool["smoothed"].max() - tolerance]
    win = tied.sort_values(["coherence", "repulsion_strength", "smoothed"], ascending=False).iloc[0]
    return tab, win


def _most_coherent(search: _Search, exclude, n: int, repulsion: float = 1.0):
    """The ``n`` settings with the highest worst-case coherence so far, other than those in ``exclude``."""
    seen = search.seen()
    seen = seen[seen["repulsion_strength"] == repulsion]
    done = {_key(s) for s in exclude}
    keep = [_key((k, md, rep)) not in done
            for k, md, rep in zip(seen["n_neighbors"], seen["min_dist"], seen["repulsion_strength"])]
    seen = seen[keep].sort_values(["coherence", "score"], ascending=False).head(n)
    return [_key((k, md, rep)) for k, md, rep in zip(seen["n_neighbors"], seen["min_dist"], seen["repulsion_strength"])]


def tune_clone_map(
    features,
    *,
    n_neighbors: int | str = "auto",
    min_dist: float | str = "auto",
    spread: float = 1.0,
    method: str = "ladder",
    coherence_target: float | str = "auto",
    penalty: float = 5.0,
    tolerance: float = 0.005,
    ladder_seeds: int = 2,
    top: int = 4,
    n_layouts: int = 6,
    rescue: int = 3,
    allow_weaker_repulsion: bool = True,
    smoothing: float = 0.7,
    search_epochs: int | None = 200,
    max_clones: int | None = 3000,
    full_layouts: int = 2,
    adam_options: dict | None = None,
    random_state: int = 0,
    n_jobs: int | None = 1,
    verbose: bool = True,
) -> MapTuning:
    """Choose UMAP ``n_neighbors`` and ``min_dist`` for a clone map.

    Looks for the most faithful layout that keeps at least ``coherence_target`` of the clones in one
    connected piece in every layout tried, which makes the choice robust to the randomness of UMAP (see
    the module documentation for the score and the search). A setting passed as a number is held fixed
    and only the other is tuned.

    Parameters
    ----------
    features
        Clone profiles, one row per clone.
    n_neighbors, min_dist
        ``"auto"`` to tune the setting, or a number to hold it fixed.
    spread
        UMAP ``spread``, held fixed.
    method
        ``"ladder"`` (default; a fixed ladder of settings, smoothed and confirmed) or ``"adam"``
        (stochastic-gradient search; options in ``adam_options``).
    coherence_target
        Fraction of clones that must lie in one connected piece of the map. ``"auto"`` (default) takes
        it from the profiles: the fraction of clones in the largest component of their mutual
        15-nearest-neighbour graph, capped at 0.99, so the map is not more fragmented than the profiles.
        Pass a number (e.g. 0.99 or 1.0) to demand a single piece regardless.
    penalty
        Score lost per unit of coherence below the target. The default (5) makes the target binding:
        one percentage point of coherence outweighs any realistic gain in fidelity.
    tolerance
        Confirmed settings whose score is within this distance of the best are treated as equally
        faithful, and the most coherent of them is chosen (seed-to-seed noise is about 0.005).
    ladder_seeds, top, n_layouts
        Layouts per ladder setting; how many of the best settings are confirmed; how many layouts each
        confirmed setting is scored on in total.
    rescue, allow_weaker_repulsion
        When the winner falls short of the target, how many of the most coherent settings seen are also
        confirmed, and whether weaker repulsion may then be tried.
    smoothing
        Width (in ladder steps) of the Gaussian that averages each ladder setting with its neighbours.
    search_epochs
        UMAP optimisation epochs for layouts during the search (``None``: UMAP's default).
    max_clones, full_layouts
        Above ``max_clones`` clones, the search runs on a random subset of this size, ``n_neighbors`` is
        scaled up to the full data in proportion to the number of clones, and the best settings are then
        confirmed on every clone with ``full_layouts`` layouts each. ``None`` always searches on every
        clone.
    adam_options
        For ``method="adam"``: ``n_iter`` (30), ``patience`` (8), ``learning_rate`` (0.25),
        ``perturbation`` (0.35), ``betas`` ((0.8, 0.99)), ``max_warmup`` (6) and ``start`` (default
        ``sqrt(N)`` neighbours between 15 and 100, and ``min_dist`` 0.3).
    random_state
        Seed for layout seeds and the clone subset; the result is reproducible for a given seed.
    n_jobs
        Processes used to compute layouts in parallel (``-1``: all available cores). Every layout depends
        only on its setting and seed, so the chosen settings are identical for any ``n_jobs``; only the
        time changes.

    Returns
    -------
    :class:`MapTuning` with the chosen settings (including ``repulsion_strength``, 1.0 unless the rescue
    needed weaker repulsion), their confirmed scores, every layout scored (``history``) and the confirmed
    candidates (``candidates``).
    """
    t0 = time.time()
    feats = np.asarray(features, dtype=float)
    n_all = feats.shape[0]
    if n_all < 15:
        raise ValueError(f"Need at least 15 clones to tune a clone map, got {n_all}.")
    if method not in {"ladder", "adam"}:
        raise ValueError("method must be 'ladder' or 'adam'.")
    for name, value in (("n_neighbors", n_neighbors), ("min_dist", min_dist)):
        if isinstance(value, str) and value != "auto":
            raise ValueError(f"{name} must be 'auto' or a number, got {value!r}.")
    tune_k, tune_md = n_neighbors == "auto", min_dist == "auto"
    if not (tune_k or tune_md):
        raise ValueError("Nothing to tune: pass n_neighbors='auto' and/or min_dist='auto'.")
    rng = np.random.default_rng(random_state)

    # search on a random subset when there are many clones; neighbour counts are scaled back up
    feats_all, scale = feats, 1.0
    if max_clones is not None and n_all > max_clones:
        feats = feats[np.sort(rng.choice(n_all, size=int(max_clones), replace=False))]
        scale = n_all / feats.shape[0]
    n = feats.shape[0]
    k_lo = N_NEIGHBORS_RANGE[0]
    k_cap = int(max(k_lo, min(N_NEIGHBORS_RANGE[1], n - 1, n // 2 if n >= 30 else n - 1)))

    ref = _Reference(feats, k_max=k_cap, random_state=int(rng.integers(2**31 - 1)))
    target = _target(ref, coherence_target)
    n_jobs = resolve_n_jobs(n_jobs)
    search = _Search(ref, target, penalty, spread, search_epochs, n_jobs)
    k_fixed = int(np.clip(round(n_neighbors / scale), k_lo, k_cap)) if not tune_k else None
    log(f"choosing clone-map settings ({method}) on {n:,} clones"
        f"{f' (random subset of {n_all:,})' if scale > 1 else ''}; target: at least {100 * target:.1f}% of "
        "clones in one piece.", verbose)

    if method == "ladder":
        ks = [k for k in N_NEIGHBORS_LADDER if k <= k_cap] or [k_cap]
        if len(ks) < 2:
            ks = sorted({k_lo, k_cap})
        ks = ks if tune_k else [k_fixed]
        mds = list(MIN_DIST_LADDER) if tune_md else [float(min_dist)]
        candidates = _ladder_candidates(search, ks, mds, rng, ladder_seeds, top, smoothing, verbose)
        ladder = (ks, mds, smoothing)
    else:
        ladder = None
        opts = dict(adam_options or {})
        md_hi = MIN_DIST_RANGE[1]
        lo = np.array([np.log(k_lo), -4.0])
        hi = np.array([np.log(k_cap), 4.0])
        active = np.array([tune_k, tune_md])
        start = opts.get("start") or (int(np.clip(round(np.sqrt(n_all)), 15, 100)), 0.3)
        k0 = start[0] / scale if tune_k else k_fixed
        md0 = start[1] if tune_md else float(min_dist)

        def from_theta(th):
            k = float(np.exp(th[0])) if tune_k else k_fixed
            md = md_hi / (1.0 + np.exp(-th[1])) if tune_md else float(min_dist)
            return int(np.clip(round(k), k_lo, k_cap)), round(float(md), 3)

        p = np.clip(md0, 0.005, md_hi - 0.005) / md_hi
        theta0 = np.clip(np.array([np.log(np.clip(k0, k_lo, k_cap)), np.log(p / (1 - p))]), lo, hi)
        candidates = _adam_candidates(search, theta0, active, lo, hi, from_theta, rng, top, opts, verbose)

    table, win = _confirm(search, candidates, rng, n_layouts, tolerance, ladder)
    if win["coherence"] < target and rescue > 0:
        extra = _most_coherent(search, candidates, rescue)
        if extra:
            log(f"  the best setting broke apart in one of its layouts; confirming {len(extra)} more "
                "coherent settings.", verbose)
            candidates = candidates + extra
            table, win = _confirm(search, candidates, rng, n_layouts, tolerance, ladder)
    if win["coherence"] < target and allow_weaker_repulsion:
        base = _most_coherent(search, [], max(rescue, 1)) + [_key(s) for s in candidates[:2]]
        for rep in REPULSION_RESCUE:
            weaker = list(dict.fromkeys((k, md, rep) for k, md, _ in base))
            log(f"  no setting held together in every layout; trying weaker repulsion ({rep}).", verbose)
            candidates = candidates + weaker
            table, win = _confirm(search, candidates, rng, n_layouts, tolerance, ladder)
            if win["coherence"] >= target:
                break
    if win["coherence"] < target:
        log(f"  warning: no setting kept {100 * target:.1f}% of clones in one piece in every layout; the "
            f"most coherent one reached {100 * win['coherence']:.1f}%.", verbose)

    k_full = lambda k: int(np.clip(round(k * scale), k_lo, min(N_NEIGHBORS_RANGE[1], n_all - 1)))  # noqa: E731
    rows = list(search.rows)
    if scale > 1:
        # a setting that holds together on the subset need not do so on every clone: the best settings are
        # confirmed again on the full data, with neighbour counts scaled up, before one is chosen
        full_ref = _Reference(feats_all, k_max=min(N_NEIGHBORS_RANGE[1], n_all - 1),
                              random_state=int(rng.integers(2**31 - 1)))
        full_target = _target(full_ref, coherence_target)
        full = _Search(full_ref, full_target, penalty, spread, search_epochs, n_jobs)
        ranked = table.sort_values(["coherence", "smoothed"], ascending=False)
        picks = [(int(n_neighbors) if not tune_k else k_full(k), float(md), float(rep))
                 for k, md, rep in zip(ranked["n_neighbors"], ranked["min_dist"], ranked["repulsion_strength"])]
        picks = list(dict.fromkeys(picks))[:max(top, 2)]
        log(f"  confirming the {len(picks)} best settings on all {n_all:,} clones.", verbose)
        table, win = _confirm(full, picks, rng, full_layouts, tolerance)
        if win["coherence"] < full_target and tune_md:
            # rescue on all clones: the widest spacings, where weakly connected groups stay attached
            ladder_k = [k for k in N_NEIGHBORS_LADDER if k <= k_cap] or [k_cap]
            ks_full = sorted({int(n_neighbors) if not tune_k else k_full(k) for k in ladder_k})
            corner = [(k, md, 1.0) for k in ks_full for md in MIN_DIST_LADDER[-2:]]
            corner = [c for c in corner if _key(c) not in {_key(p) for p in picks}]
            log(f"  no setting held together on all clones; scanning {len(corner)} wide-spacing settings.",
                verbose)
            seed = int(rng.integers(2**31 - 1))
            full.evaluate_many([(c, seed, "full data") for c in corner])
            corner.sort(key=lambda c: (full.robust(c)["coherence"], full.robust(c)["score"]), reverse=True)
            picks = picks + corner[:3]
            table, win = _confirm(full, picks, rng, full_layouts, tolerance)
        if win["coherence"] < full_target and allow_weaker_repulsion:
            for rep in REPULSION_RESCUE:
                coherent_first = table.sort_values(["coherence", "score"], ascending=False)
                base = list(zip(coherent_first["n_neighbors"].astype(int), coherent_first["min_dist"]))[:2]
                weaker = list(dict.fromkeys((int(k), float(md), rep) for k, md in base))
                log(f"  no setting held together on all clones; trying weaker repulsion ({rep}).", verbose)
                picks = picks + weaker
                table, win = _confirm(full, picks, rng, full_layouts, tolerance)
                if win["coherence"] >= full_target:
                    break
        rows += [r | {"stage": "full data"} for r in full.rows]
        target, scale = full_target, 1.0
        if win["coherence"] < target:
            log(f"  warning: on all clones, the most coherent setting kept {100 * win['coherence']:.1f}% of "
                f"clones in one piece (target {100 * target:.1f}%).", verbose)
    k_best = int(n_neighbors) if not tune_k else k_full(win["n_neighbors"])
    table["n_neighbors_full_data"] = [int(n_neighbors) if not tune_k else k_full(k) for k in table["n_neighbors"]]
    result = MapTuning(
        n_neighbors=k_best, min_dist=float(win["min_dist"]), repulsion_strength=float(win["repulsion_strength"]),
        score=float(win["score"]), coherence=float(win["coherence"]), local_fidelity=float(win["local_fidelity"]),
        global_fidelity=float(win["global_fidelity"]), coherence_target=target, method=method,
        n_layouts=len(rows), seconds=time.time() - t0, history=pd.DataFrame(rows), candidates=table)
    rep_txt = "" if result.repulsion_strength == 1.0 else f", repulsion_strength {result.repulsion_strength:g}"
    log(f"clone-map settings: n_neighbors {result.n_neighbors}, min_dist {result.min_dist:.2f}{rep_txt}; "
        f"in every layout at least {100 * result.coherence:.1f}% of clones form one piece (local fidelity "
        f"{result.local_fidelity:.3f}, global fidelity {result.global_fidelity:.3f}; {result.n_layouts} "
        f"layouts, {result.seconds:.0f} s).", verbose)
    return result


# --------------------------------------------------------------------------- the map that is returned


def layout_clone_map(features, *, n_neighbors: int, min_dist: float, repulsion_strength: float = 1.0,
                     spread: float = 1.0, coherence_target: float | str = "auto", tuning: MapTuning | None = None,
                     max_retries: int = 4, random_state: int = 0):
    """Lay the clones out at given settings, at UMAP's default number of epochs, and check the result.

    When the settings came from :func:`tune_clone_map` (``tuning``) and this layout falls short of the
    coherence target, up to ``max_retries`` further layouts are tried: two more seeds, then the next
    confirmed settings that held together in every layout of the search. The most coherent layout is
    returned. With settings given by hand, the layout is returned as it is.

    Returns
    -------
    ``(xy, quality, setting, attempts)``: the coordinates, :func:`map_quality` of the returned layout,
    the ``(n_neighbors, min_dist, repulsion_strength, seed)`` that produced it, and every layout tried.
    """
    import umap

    feats = np.asarray(features, dtype=float)
    n = feats.shape[0]
    target = tuning.coherence_target if tuning is not None else coherence_target

    def run(k, md, rep, seed):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value.*")
            xy = umap.UMAP(n_neighbors=int(min(k, n - 1)), min_dist=float(md), spread=spread,
                           repulsion_strength=float(rep), random_state=int(seed)).fit_transform(feats)
        return xy, map_quality(feats, xy, coherence_target=target)

    setting = (int(n_neighbors), float(min_dist), float(repulsion_strength), int(random_state))
    xy, quality = run(*setting)
    names = ("n_neighbors", "min_dist", "repulsion_strength", "seed")
    attempts = [dict(zip(names, setting), coherence=quality["coherence"], score=quality["score"])]
    if tuning is not None and quality["coherence"] < quality["coherence_target"] - 0.005:
        tries = [setting[:3] + (int(random_state) + i,) for i in (1, 2)]
        for _, r in tuning.candidates.iterrows():
            alt = (int(r["n_neighbors_full_data"]), float(r["min_dist"]), float(r["repulsion_strength"]))
            if alt != setting[:3] and r["coherence"] >= tuning.coherence_target:
                tries.append(alt + (int(random_state),))
        best = (quality["coherence"], quality["score"], xy, quality, setting)
        for trial in tries[:max_retries]:
            xy_t, q_t = run(*trial)
            attempts.append(dict(zip(names, trial), coherence=q_t["coherence"], score=q_t["score"]))
            if (q_t["coherence"], q_t["score"]) > best[:2]:
                best = (q_t["coherence"], q_t["score"], xy_t, q_t, trial)
            if q_t["coherence"] >= q_t["coherence_target"] - 0.005:
                break
        _, _, xy, quality, setting = best
    return xy, quality, setting, attempts
