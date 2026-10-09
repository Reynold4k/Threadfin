"""A train-only expression embedding for prospective/held-out validation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from ._utils import require_positive_int


class FrozenExpressionModel:
    """HVG selection, scaling and PCA fitted on reference cells only.

    Per-cell normalisation is to 10,000 counts, followed by log1p.
    Seurat HVGs exclude immunoglobulin and TCR genes. Mean, standard
    deviation, clipping and PCA are frozen for all query cells. There is no
    Harmony step: a query donor cannot change the fitted coordinates.
    This controls train/query leakage, not biological batch confounding.
    """

    @staticmethod
    def _counts(adata, layer):
        x = adata.X if layer is None else adata.layers[layer]
        values = x.data if sp.issparse(x) else np.asarray(x)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Frozen expression embedding requires finite non-negative counts.")
        if not adata.var_names.is_unique:
            raise ValueError("Gene names must be unique.")
        return x

    @staticmethod
    def _lognorm(x):
        total = np.asarray(x.sum(axis=1)).ravel()
        if np.any(total <= 0):
            raise ValueError("Every input cell needs a positive library size.")
        factor = 1e4 / total
        if sp.issparse(x):
            out = x.tocsr().multiply(factor[:, None]).tocsr()
            out.data = np.log1p(out.data)
            return out
        return np.log1p(np.asarray(x, dtype=float) * factor[:, None])

    @classmethod
    def fit(cls, reference, *, counts_layer=None, n_top_genes=2000,
            n_comps=30, clip=10.0, random_state=0, batch_size=2048):
        """Fit all data-dependent choices on reference cells, without mutating them."""
        import anndata as ad
        import scanpy as sc
        from .pp import ig_gene_mask
        for name, value in [("n_top_genes", n_top_genes), ("n_comps", n_comps), ("batch_size", batch_size)]:
            require_positive_int(value, name)
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be positive and finite.")
        if reference.n_obs < 3:
            raise ValueError("At least three reference cells are required.")
        x = cls._lognorm(cls._counts(reference, counts_layer))
        genes = pd.Index(reference.var_names).astype(str)
        keep = ~ig_gene_mask(genes, constant=True, tr_genes=True)
        if keep.sum() < 2:
            raise ValueError("At least two non-receptor genes are required.")
        tmp = ad.AnnData(x[:, keep], var=pd.DataFrame(index=genes[keep]))
        hv = sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top_genes, tmp.n_vars),
                                        flavor="seurat", inplace=False)
        selected = np.flatnonzero(keep)[hv.highly_variable.to_numpy()]
        sub = x[:, selected]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        obj = cls()
        obj.gene_universe = genes.to_numpy(dtype=str)
        obj.genes = genes[selected].to_numpy(dtype=str)
        obj.mean = sub.mean(axis=0)
        obj.scale = sub.std(axis=0, ddof=1)
        obj.scale[obj.scale <= 0] = 1.0
        obj.clip, obj.batch_size = float(clip), int(batch_size)
        z = np.clip((sub-obj.mean) / obj.scale, -clip, clip)
        n_comps = min(n_comps, reference.n_obs-1, len(selected))
        pca = PCA(n_components=n_comps, svd_solver="randomized", random_state=random_state)
        pca.fit(z)
        obj.components = pca.components_
        obj.pca_mean = pca.mean_
        obj.explained_variance_ratio = pca.explained_variance_ratio_
        obj.random_state = int(random_state)
        obj.n_reference_cells = int(reference.n_obs)
        return obj

    def transform(self, query, *, counts_layer=None):
        """Project query cells in batches; no fit, centering update or integration."""
        counts = self._counts(query, counts_layer)
        genes = pd.Index(query.var_names).astype(str)
        if len(genes) != len(self.gene_universe) or set(genes) != set(self.gene_universe):
            raise ValueError("Query must contain the same gene universe as the reference (order may differ).")
        selected = genes.get_indexer(self.genes)
        out = np.empty((query.n_obs, len(self.components)))
        for start in range(0, query.n_obs, self.batch_size):
            stop = min(start+self.batch_size, query.n_obs)
            sub = self._lognorm(counts[start:stop])[:, selected]
            sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
            z = np.clip((sub-self.mean) / self.scale, -self.clip, self.clip)
            out[start:stop] = (z-self.pca_mean) @ self.components.T
        return out

    def save(self, path):
        """Save frozen numeric parameters without pickle."""
        parameters = {k: getattr(self, k) for k in ("clip", "batch_size", "random_state", "n_reference_cells")}
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, parameters=json.dumps(parameters),
                                **{k: getattr(self, k) for k in
                                   ("gene_universe", "genes", "mean", "scale", "components",
                                    "pca_mean", "explained_variance_ratio")})

    @classmethod
    def load(cls, path):
        """Read frozen numeric parameters with pickle disabled."""
        obj = cls()
        with np.load(path, allow_pickle=False) as saved:
            for key, value in json.loads(str(saved["parameters"])).items():
                setattr(obj, key, value)
            for key in ("gene_universe", "genes", "mean", "scale", "components", "pca_mean",
                        "explained_variance_ratio"):
                setattr(obj, key, saved[key].copy())
        return obj
