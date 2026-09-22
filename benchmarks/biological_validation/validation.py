"""Core validation routines for Threadfin's multi-dataset biological validation.

Everything here works on a standardized AnnData produced by loaders.py:
- obs: clone_id (str/NaN), state (reference labels), donor, plus any
  dataset-specific external metadata (timepoint, condition, ...).
- obsm: X_pca and X_umap.
- BCR clone info already attached via threadfin.attach_bcr.

Null-model and held-out strategies are documented per function.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import threadfin as tf


# ---------------------------------------------------------------- QC


def compute_qc(adata, clone_key: str = "clone_id", state_key: str = "state") -> dict:
    """Dataset-level QC summary for the validation report."""
    clones = adata.obs[clone_key].dropna()
    sizes = clones.value_counts()
    qc = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_bcr_positive_cells": int(len(clones)),
        "frac_bcr_positive": round(len(clones) / adata.n_obs, 4),
        "n_clonotypes": int(sizes.size),
        "clone_size_median": float(sizes.median()) if len(sizes) else None,
        "clone_size_max": int(sizes.max()) if len(sizes) else None,
        "expanded_clones_min2": int((sizes >= 2).sum()),
        "expanded_clones_min3": int((sizes >= 3).sum()),
        "expanded_clones_min5": int((sizes >= 5).sum()),
    }
    for key in ("donor", "timepoint", "condition", state_key):
        if key in adata.obs.columns:
            qc[f"levels_{key}"] = {str(k): int(v) for k, v in
                                   adata.obs[key].value_counts(dropna=False).items()}
    return qc


# ---------------------------------------------------------------- standard preprocessing


def standard_preprocess(adata, batch_key: str | None = None, seed: int = 0):
    """Standard scanpy pipeline; adds X_pca/X_umap and leiden 'state' labels
    when the dataset has no curated state annotation."""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    sc.pp.pca(adata, n_comps=40, random_state=seed)
    sc.pp.neighbors(adata, n_pcs=40, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    if "state" not in adata.obs.columns:
        sc.tl.leiden(adata, resolution=0.8, random_state=seed, key_added="state")
    return adata


# ---------------------------------------------------------------- parameter grid + robustness


def run_parameter_grid(
    adata,
    bases=("X_pca", "X_umap"),
    resolutions=(0.3, 0.8),
    cdr3_weights=(0.0, 0.3),
    min_clone_size: int = 3,
    n_neighbors: int = 20,
    seed: int = 0,
):
    """Run Threadfin over a parameter grid; returns (adata, results list).

    Each run's labels are stored in obs under a unique key; the clone map of
    each run is kept in the returned results (not in uns, to avoid clashes).
    """
    results = []
    for basis in bases:
        if basis not in adata.obsm:
            continue
        for res in resolutions:
            for cw in cdr3_weights:
                if cw > 0 and "threadfin_clones" not in adata.uns:
                    continue
                key = f"tf__{basis.lstrip('X_')}__r{res}__cw{cw}"
                ad = tf.clonotype_recluster(
                    adata.copy(), basis=basis, min_clone_size=min_clone_size,
                    n_neighbors=n_neighbors, resolution=res, cdr3_weight=cw,
                    random_state=seed, key_added=key,
                )
                adata.obs[key] = ad.obs[key]
                res_entry = {
                    "key": key, "basis": basis, "resolution": res, "cdr3_weight": cw,
                    "min_clone_size": min_clone_size, "n_neighbors": n_neighbors,
                    "seed": seed,
                    "n_clone_clusters": int(adata.obs[key].nunique()),
                    "n_cells_labeled": int(adata.obs[key].notna().sum()),
                }
                if "state" in adata.obs.columns:
                    res_entry["concordance"] = tf.metrics.state_concordance(
                        adata, key, "state")
                    enr = tf.metrics.state_enrichment(adata, key, "state")
                    sig = enr[enr["fdr"] < 0.05]
                    res_entry["n_significant_enrichments"] = int(len(sig))
                    res_entry["top_enrichment"] = (
                        sig.iloc[0].to_dict() if len(sig) else None)
                results.append(res_entry)
    return adata, results


def basis_robustness(adata, key_a: str, key_b: str, clone_key: str = "clone_id") -> dict:
    """Clone-level agreement between two Threadfin runs (e.g. PCA vs UMAP basis)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    obs = adata.obs[[clone_key, key_a, key_b]].dropna()
    clone_lab = obs.groupby(clone_key).agg({key_a: "first", key_b: "first"})
    return {
        "key_a": key_a, "key_b": key_b,
        "n_clones_compared": int(len(clone_lab)),
        "clone_level_ari": float(adjusted_rand_score(clone_lab[key_a], clone_lab[key_b])),
        "clone_level_nmi": float(normalized_mutual_info_score(clone_lab[key_a], clone_lab[key_b])),
    }


# ---------------------------------------------------------------- null models


def null_permutation_purity(
    adata,
    clone_key: str = "clone_id",
    state_key: str = "state",
    donor_key: str | None = None,
    min_clone_size: int = 3,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Null 1 — clone-label permutation.

    Clone ids are permuted among cells **within each donor** (or globally if
    no donor key), which preserves the clone-size distribution and the
    state composition but destroys any clone→state relationship. The observed
    mean clone state-purity is compared against this null distribution.
    """
    rng = np.random.default_rng(seed)

    def mean_purity(clone_series):
        obs = pd.DataFrame({"clone": clone_series, "state": adata.obs[state_key]}).dropna()
        sizes = obs["clone"].value_counts()
        keep = sizes[sizes >= min_clone_size].index
        obs = obs[obs["clone"].isin(keep)]
        if len(obs) == 0:
            return np.nan
        pur = obs.groupby("clone")["state"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        return float(pur.mean())

    real = mean_purity(adata.obs[clone_key])

    nulls = []
    obs_df = adata.obs
    nonnull_mask = obs_df[clone_key].notna()
    for _ in range(n_permutations):
        perm_clones = obs_df[clone_key].copy()
        if donor_key and donor_key in obs_df.columns:
            for _, sub in obs_df.groupby(donor_key, observed=True):
                mask = sub[clone_key].notna()
                vals = sub.loc[mask, clone_key].to_numpy().copy()
                rng.shuffle(vals)
                perm_clones.loc[sub.index[mask]] = vals
        else:
            vals = obs_df.loc[nonnull_mask, clone_key].to_numpy().copy()
            rng.shuffle(vals)
            perm_clones.loc[nonnull_mask] = vals
        nulls.append(mean_purity(perm_clones))

    nulls = np.array([x for x in nulls if not np.isnan(x)])
    p = (1 + int((nulls >= real).sum())) / (len(nulls) + 1)
    return {"real_mean_purity": real, "null_mean": float(nulls.mean()),
            "null_sd": float(nulls.std()), "nulls": nulls.tolist(),
            "p_value": float(p), "n_permutations": int(len(nulls)),
            "preserves": "clone-size distribution, donor composition, state composition; "
                         "destroys clone→state link (within donor)"}


def null_random_communities(
    adata,
    cluster_key: str,
    state_key: str = "state",
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Null 3 — size-matched random clone communities.

    Clone-cluster labels are permuted across *clones*, preserving community
    sizes and clone state composition. The number of significant (FDR<0.05)
    state enrichments under this null estimates the false-positive level of
    the enrichment readout.
    """
    rng = np.random.default_rng(seed)
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"obs['{cluster_key}'] missing; run the grid first")

    real_enr = tf.metrics.state_enrichment(adata, cluster_key, state_key)
    real_n_sig = int((real_enr["fdr"] < 0.05).sum())

    # clone → label table (clones that were clustered)
    clone_lab = adata.obs[[clone_key, cluster_key]].dropna().drop_duplicates(clone_key)
    labels = clone_lab[cluster_key].to_numpy()

    null_sig = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        mapping = dict(zip(clone_lab[clone_key], perm_labels))
        adata.obs["_null_cc"] = adata.obs[clone_key].map(mapping).astype("category")
        enr = tf.metrics.state_enrichment(adata, "_null_cc", state_key)
        null_sig.append(int((enr["fdr"] < 0.05).sum()))
    adata.obs.drop(columns=["_null_cc"], inplace=True)

    null_sig = np.array(null_sig)
    p = (1 + int((null_sig >= real_n_sig).sum())) / (len(null_sig) + 1)
    return {"real_n_significant": real_n_sig, "null_mean": float(null_sig.mean()),
            "null_sd": float(null_sig.std()), "p_value": float(p),
            "preserves": "community sizes, clone→state composition; "
                         "destroys community→state link"}


# ---------------------------------------------------------------- held-out validation


def heldout_split_validation(
    adata,
    clone_key: str = "clone_id",
    basis: str = "X_pca",
    min_clone_cells: int = 8,
    n_repeats: int = 5,
    seed: int = 0,
    resolution: float = 0.5,
) -> dict:
    """Held-out split-clone validation.

    For every clone with >= ``min_clone_cells`` cells, the cells are randomly
    split into two halves, treated as two independent pseudo-clones
    (``__A``/``__B``), and Threadfin is re-run. If clone communities capture
    real transcriptional structure (rather than overfitting the specific
    cells), the two held-out halves of a clone should be assigned to the same
    clone community far more often than chance.

    Chance level is computed as sum_k p_k^2 over the community-size
    distribution of each repeat.
    """
    rng = np.random.default_rng(seed)
    obs = adata.obs
    big = obs[clone_key].value_counts()
    big = big[big >= min_clone_cells].index.tolist()
    if len(big) < 5:
        return {"error": "fewer than 5 clones meet min_clone_cells; skipped",
                "n_big_clones": len(big)}

    rates, chances = [], []
    for _ in range(n_repeats):
        pseudo = obs[clone_key].astype("object").copy()
        for c in big:
            cells = obs.index[obs[clone_key] == c].to_numpy()
            rng.shuffle(cells)
            half = len(cells) // 2
            pseudo.loc[cells[:half]] = f"{c}__A"
            pseudo.loc[cells[half:]] = f"{c}__B"
        ad = adata.copy()
        ad.obs["pseudo_clone"] = pseudo
        tf.clonotype_recluster(ad, clone_key="pseudo_clone", basis=basis,
                               min_clone_size=3, resolution=resolution,
                               random_state=int(rng.integers(1e6)),
                               key_added="pseudo_cluster")
        lab = (ad.obs[["pseudo_clone", "pseudo_cluster"]].dropna()
               .groupby("pseudo_clone")["pseudo_cluster"]
               .agg(lambda s: s.mode().iloc[0]))
        matches = np.mean([lab.get(f"{c}__A") == lab.get(f"{c}__B")
                           for c in big if f"{c}__A" in lab.index and f"{c}__B" in lab.index])
        freqs = lab.value_counts(normalize=True)
        chances.append(float((freqs**2).sum()))
        rates.append(float(matches))

    return {
        "n_big_clones": len(big), "n_repeats": n_repeats, "basis": basis,
        "min_clone_cells": min_clone_cells,
        "cocluster_rate_mean": float(np.mean(rates)),
        "cocluster_rate_sd": float(np.std(rates)),
        "chance_rate_mean": float(np.mean(chances)),
        "rates": rates, "chances": chances,
        "interpretation": "cocluster_rate >> chance_rate => clone communities are "
                          "reproducible on held-out cells of the same clone",
    }


# ---------------------------------------------------------------- IO helpers


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Categorical, pd.Series)):
            return list(o)
        return super().default(o)
