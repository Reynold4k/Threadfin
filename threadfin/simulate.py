"""Simulate paired expression + clone data with known clonal programmes.

The generative model mirrors the structure Threadfin is meant to recover:

1. ``n_states`` cell states are points (centres) in a ``dim``-dimensional
   expression embedding.
2. Each clonal programme ``g`` has a state composition ``theta_g`` (for
   example "mostly plasmablast" or "half germinal centre, half memory").
3. Each clone belongs to one donor and one programme; its own composition is
   ``theta_c ~ Dirichlet(concentration * theta_g)``, so clones of a programme
   are similar but not identical, and it carries a small heritable offset
   ``u_c`` (clonal memory within a state).
4. Clone sizes follow a truncated Zipf law, so most clones are singletons,
   as in real repertoires.
5. Cells are sampled in contexts (libraries / samples) that add a technical
   mean shift. By default every clone is captured in a single context of its
   donor (``clone_nesting="nested"``), so context effects masquerade as clone
   effects unless they are removed.

Each cell: ``x = centre[state] + u_clone + shift[context] + noise``.

Scenarios
---------
``"default"``
    As above.
``"null"``
    One programme and no clonal offsets: clone identity carries no state
    information beyond sampling context (for type-I error checks).
``"bifurcation"``
    Programme A splits its cells between states 1 and 2, programme B sits in
    a state placed exactly at their midpoint: the two programmes have the
    same centroid but different distributions.

Time courses
------------
With ``n_timepoints > 1`` every cell is sampled at a random time point and
each sample (donor x time point) is its own context. Between consecutive time
points a clone keeps its programme with probability ``memory`` and otherwise
switches to a different one, so ``memory`` is the ground truth for
:func:`threadfin.tl.clonal_memory`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _distinct_compositions(rng, n_programmes, n_states, alpha, min_dist=0.35, tries=200):
    """Programme state compositions that are pairwise distinct (Hellinger)."""
    best = None
    for _ in range(tries):
        theta = rng.dirichlet(np.full(n_states, alpha), size=n_programmes)
        h = np.sqrt(0.5 * ((np.sqrt(theta)[:, None, :] - np.sqrt(theta)[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(h, np.inf)
        if best is None or h.min() > best[0]:
            best = (h.min(), theta)
        if h.min() >= min_dist:
            return theta
    return best[1]


def simulate_repertoire(
    *,
    scenario: str = "default",
    n_donors: int = 4,
    contexts_per_donor: int = 3,
    n_programmes: int = 4,
    n_states: int = 6,
    n_clones: int = 3000,
    clone_size_exponent: float = 2.2,
    max_clone_size: int = 300,
    dim: int = 20,
    state_separation: float = 1.5,
    programme_sparsity: float = 0.4,
    clone_concentration: float | None = 15.0,
    clonal_offset: float = 0.4,
    noise: float = 1.0,
    context_shift: float = 1.0,
    clone_nesting: str = "nested",
    n_timepoints: int = 1,
    memory: float = 1.0,
    random_state: int = 0,
):
    """Simulate an AnnData with ground-truth clones, programmes and states.

    Parameters
    ----------
    scenario
        ``"default"``, ``"null"`` or ``"bifurcation"`` (see module docstring).
    clone_size_exponent
        Zipf exponent (larger = more singletons).
    state_separation
        Standard deviation of state centres (larger = more distinct states).
    programme_sparsity
        Dirichlet concentration of programme compositions (smaller = each
        programme dominated by fewer states).
    clone_concentration
        How closely clones follow their programme (``None`` = exactly).
    clonal_offset
        SD of the clone-specific continuous offset.
    noise, context_shift
        SD of cell noise and of context mean shifts.
    clone_nesting
        ``"nested"`` (each clone sampled in one context) or ``"spread"``.
    n_timepoints, memory
        Number of sampling time points and the probability that a clone keeps
        its programme from one time point to the next (see module docstring).

    Returns
    -------
    AnnData with ``X`` = ``obsm['X_pca']`` = the embedding and ``obs`` columns
    ``clone_id``, ``donor``, ``context``, ``true_programme``, ``true_state``
    and ``clone_size``; parameters in ``uns['simulation']``.
    """
    from anndata import AnnData

    rng = np.random.default_rng(random_state)
    if scenario not in ("default", "null", "bifurcation"):
        raise ValueError("scenario must be 'default', 'null' or 'bifurcation'.")
    if scenario == "null":
        n_programmes, clonal_offset, clone_concentration = 1, 0.0, None

    # 1) states and programme compositions
    centres = rng.normal(0.0, state_separation, size=(n_states, dim))
    theta = _distinct_compositions(rng, n_programmes, n_states, programme_sparsity)
    if scenario == "bifurcation":
        if n_states < 3 or n_programmes < 2:
            raise ValueError("bifurcation needs >= 3 states and >= 2 programmes.")
        centres[2] = 0.5 * (centres[0] + centres[1])  # state 2 = midpoint of states 0 and 1
        theta[0] = np.eye(n_states)[0] * 0.5 + np.eye(n_states)[1] * 0.5
        theta[1] = np.eye(n_states)[2]

    # 2) clones: donor, programme (per time point), composition, offset, size
    donor = rng.integers(0, n_donors, size=n_clones)
    n_t = max(1, int(n_timepoints))
    programme_t = np.empty((n_clones, n_t), dtype=np.int64)
    programme_t[:, 0] = rng.integers(0, n_programmes, size=n_clones)
    for t in range(1, n_t):
        keep = rng.random(n_clones) < memory
        other = (programme_t[:, t - 1] + rng.integers(1, max(n_programmes, 2), size=n_clones)) % max(n_programmes, 1)
        programme_t[:, t] = np.where(keep | (n_programmes == 1), programme_t[:, t - 1], other)
    theta_ct = np.empty((n_clones, n_t, n_states))
    for t in range(n_t):
        for c in range(n_clones):
            g = programme_t[c, t]
            if t > 0 and g == programme_t[c, t - 1]:
                theta_ct[c, t] = theta_ct[c, t - 1]  # same programme -> same composition
            elif clone_concentration is None:
                theta_ct[c, t] = theta[g]
            else:
                theta_ct[c, t] = rng.dirichlet(clone_concentration * theta[g] + 1e-3)
    offset = rng.normal(0.0, clonal_offset, size=(n_clones, dim)) if clonal_offset > 0 else np.zeros((n_clones, dim))
    sizes = np.minimum(rng.zipf(clone_size_exponent, size=n_clones), max_clone_size)

    # 3) cells: time point, state, context
    clone_of_cell = np.repeat(np.arange(n_clones), sizes)
    n_cells = clone_of_cell.size
    tp = rng.integers(0, n_t, size=n_cells) if n_t > 1 else np.zeros(n_cells, dtype=np.int64)
    u = rng.random(n_cells)
    cdf = np.cumsum(theta_ct[clone_of_cell, tp], axis=1)
    state = (u[:, None] > cdf).sum(axis=1).clip(max=n_states - 1)
    d = donor[clone_of_cell]
    if n_t > 1:
        ctx = tp  # one sample per donor x time point
        shifts = rng.normal(0.0, context_shift, size=(n_donors, n_t, dim))
    else:
        shifts = rng.normal(0.0, context_shift, size=(n_donors, contexts_per_donor, dim))
        if clone_nesting == "nested":
            ctx = rng.integers(0, contexts_per_donor, size=n_clones)[clone_of_cell]
        elif clone_nesting == "spread":
            ctx = rng.integers(0, contexts_per_donor, size=n_cells)
        else:
            raise ValueError("clone_nesting must be 'nested' or 'spread'.")
    x = (centres[state] + offset[clone_of_cell] + shifts[d, ctx]
         + rng.normal(0.0, noise, size=(n_cells, dim)))
    programme = programme_t[:, 0]

    obs = pd.DataFrame({
        "clone_id": [f"D{di}|C{ci:05d}" for di, ci in zip(d, clone_of_cell)],
        "donor": [f"D{di}" for di in d],
        "context": [f"D{di}_S{ci}" for di, ci in zip(d, ctx)],
        "true_programme": [f"G{g}" for g in programme_t[clone_of_cell, tp]],
        "true_state": [f"S{s}" for s in state],
        "clone_size": sizes[clone_of_cell],
    }, index=[f"cell{i}" for i in range(n_cells)])
    if n_t > 1:
        obs["timepoint"] = [f"t{t}" for t in tp]
        obs["context"] = [f"D{di}_t{t}" for di, t in zip(d, tp)]
        obs["programme_at_start"] = [f"G{g}" for g in programme[clone_of_cell]]
    adata = AnnData(X=x.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = x.astype(np.float32)
    adata.uns["simulation"] = {
        "scenario": scenario, "n_donors": n_donors, "contexts_per_donor": contexts_per_donor,
        "n_programmes": n_programmes, "n_states": n_states, "n_clones": n_clones,
        "clone_size_exponent": clone_size_exponent, "dim": dim,
        "state_separation": state_separation, "programme_sparsity": programme_sparsity,
        "clone_concentration": -1.0 if clone_concentration is None else float(clone_concentration),
        "clonal_offset": clonal_offset, "noise": noise, "context_shift": context_shift,
        "clone_nesting": clone_nesting, "random_state": random_state,
        "n_timepoints": n_t, "memory": float(memory),
        "programme_composition": theta,
    }
    return adata
