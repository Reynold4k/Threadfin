"""Threadfin's calls of scanpy use the argument names of the installed scanpy.

scanpy 1.13 renamed several arguments (inplace=False to out=True, batch_key to batch, layer to use, score_name to
out, use_rep to use) and warns about the old names, which scanpy 1.10-1.12 need; it also reworded its Leiden
warning, which the earlier v3 function silences.
"""

import os
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

import threadfin as tf
from threadfin import programs


def test_scanpy_is_called_without_deprecated_arguments():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"batch": np.repeat(["a", "b"], 150), "clone_cluster": np.tile(["A", "B"], 150)},
                       index=[f"c{i}" for i in range(300)])
    counts = rng.poisson(rng.gamma(1.0, 1.0, size=(3, 120))[rng.integers(0, 3, 300)]).astype(np.float32)
    adata = AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(120)]))
    sim = tf.sim.simulate_repertoire(n_clones=200, random_state=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf.pp.prepare_embedding(adata, batch_key="batch", n_top_genes=60, n_comps=5, verbose=False)
        tf.pp.FrozenExpressionModel.fit(AnnData(counts.copy(), var=adata.var.copy()), n_top_genes=50, n_comps=5)
        programs.community_markers(adata, n_genes=5, layer="log_norm")
        programs.community_score(adata, {"prog": ["G1", "G2", "G3"]})
        tf.clonotype_recluster(sim, basis="X_pca", embedding_mode="distance_profiles", cluster_on="embedding",
                               n_neighbors=10, umap_n_neighbors=15, random_state=0, cluster_random_state=0,
                               copy=True)
    package = os.path.dirname(tf.__file__)
    ours = [f"{os.path.basename(w.filename)}:{w.lineno} {w.message}" for w in caught
            if issubclass(w.category, (FutureWarning, DeprecationWarning)) and w.filename.startswith(package)]
    assert not ours, ours
    assert not [w for w in caught if "implementation of leiden" in str(w.message)
                or "implementations of leiden" in str(w.message)]
