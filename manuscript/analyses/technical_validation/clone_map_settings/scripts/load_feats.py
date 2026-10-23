"""Rebuild clone-profile features for a cached dataset exactly as clone_map_sweep.features_for does."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表")
sys.path.insert(0, str(ROOT / "Threadfin"))
CACHE = ROOT / "internal_validation/algorithm_revision/embedding_cache"
CLONES = ROOT / "Threadfin/case_studies/results/dandelion_clones"

def features(dataset, min_cells=2, cache=None):
    import anndata as ad, threadfin as tf
    out = Path(cache or (Path(__file__).parent / f"feats_{dataset}_min{min_cells}.npz"))
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return z["feats"], z["sizes"], z["rel"], z["ids"]
    a = ad.read_h5ad(CACHE / f"{dataset}_embedding.h5ad")
    a.obs_names = a.obs_names.astype(str)
    ids = pd.read_csv(CLONES / f"{dataset}.csv").set_index("cell")["clone_id"]
    a.obs["clone_id"] = ids.reindex(a.obs_names).values
    a = a[a.obs["clone_id"].notna()].copy()
    ctx = "donor" if "donor" in a.obs else None
    tf.tl.clone_profiles(a, basis="X_threadfin", context_key=ctx, donor_key="donor",
                         representation="kernel", min_cells=min_cells, verbose=False)
    prof = a.uns["threadfin"]["profiles"]
    t, f = prof["clone_table"], prof["features"]
    np.savez(out, feats=f.loc[t.index].to_numpy(), sizes=t["n_cells"].to_numpy(),
             rel=t["reliability"].to_numpy(), ids=np.asarray(t.index.astype(str)))
    return features(dataset, min_cells, cache)
