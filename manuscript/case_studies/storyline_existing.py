"""Read frozen Dandelion assignments and coordinates for downstream story analyses.

Raw expression is loaded only to reproduce the original marker scores. Neither
the Threadfin embedding nor Leiden, UMAP, or clone definitions are recomputed.
"""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT.parent / "internal_validation/algorithm_revision/embedding_cache"
ORIGINAL = HERE / "results/storyline_verification_2026-10-14/original"

def load_existing(name, stamp=print):
    from datasets import LOADERS
    stamp(f"Reading frozen coordinates and Dandelion IDs for {name}")
    frozen = ad.read_h5ad(CACHE / f"{name}_embedding.h5ad")
    cells = pd.read_csv(HERE / f"results/{name}/cells.csv.gz", index_col=0)
    assert frozen.obs_names.equals(cells.index), "Frozen cell order differs from saved UMAP"
    ids = pd.read_csv(HERE / f"results/dandelion_clones/{name}.csv").set_index("cell")["clone_id"]
    # light-chain-only Dandelion groups ("B_VJ_...") are not clones; run_case_study.py drops them too
    ids = ids[~ids.astype(str).str.split("|").str[-1].str.startswith("B_VJ_")]
    clone_ids = ids.reindex(cells.index)
    assert clone_ids.fillna("").equals(cells["clone_id"].fillna("")), "Saved Dandelion IDs differ"
    stamp("Loading raw expression for marker scores only")
    data, _ = LOADERS[name]()
    assert data.obs_names.equals(frozen.obs_names), "Raw expression cell order differs"
    data.obs = frozen.obs.copy()
    data.obs["clone_id"] = clone_ids.values
    data.obs["isotype"] = cells["isotype"].values
    data.obs["leiden_state"] = cells["leiden_state"].astype(str).values
    if name == "spont_gc":
        prior = pd.read_csv(ORIGINAL / name / "spont_gc_fate_cells.csv.gz", index_col=0)
        data.obs["mutation_frequency"] = prior["mutation_frequency"].reindex(data.obs_names).values
    data.obsm["X_threadfin"] = np.asarray(frozen.obsm["X_threadfin"])
    data.obsm["X_umap"] = cells[["umap_1", "umap_2"]].to_numpy()
    data.layers["log_norm"] = data.X.copy()
    sc.pp.normalize_total(data, target_sum=1e4, layer="log_norm")
    sc.pp.log1p(data, layer="log_norm")
    stamp(f"Frozen cell/clone IDs and UMAP validated: {data.n_obs} cells")
    return data

def neighbor_indices(xy, k=15):
    """Actual k nearest neighbors, excluding self even in tied coordinates."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) <= k:
        raise ValueError(f"{len(xy)} observations cannot supply {k} non-self neighbors")
    candidates = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)
    return np.vstack([row[row != i][:k] for i, row in enumerate(candidates)])

def agreement_and_null(xy, labels, k=15, n_perm=100, seed=0):
    idx = neighbor_indices(xy, k)
    labels = np.asarray(labels)
    observed = float((labels[idx] == labels[:, None]).mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        lab = rng.permutation(labels)
        null.append(float((lab[idx] == lab[:, None]).mean()))
    null = np.asarray(null)
    return {"agreement": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "null_q95": float(np.quantile(null, 0.95)),
            "p_perm": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
            "n": int(len(labels)), "k": k, "n_perm": n_perm}
