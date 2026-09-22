"""Publication-quality plotting for Threadfin results.

All functions accept an optional matplotlib ``ax``, never call
``plt.show()``, and return the figure/axes for further customization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_clone_map(adata) -> pd.DataFrame:
    clone_map = adata.uns.get("threadfin", {}).get("clone_map")
    if clone_map is None:
        raise ValueError("Run threadfin.clonotype_recluster() first.")
    return clone_map


def clone_map(
    adata,
    color: str = "clone_cluster",
    size_by: str = "n_cells",
    size_range: tuple = (20, 400),
    ax=None,
    palette: dict | None = None,
    title: str | None = None,
    save: str | None = None,
):
    """Scatter plot of the clonotype map (one point per clonotype).

    Point size encodes clone size (number of cells). ``color`` accepts any
    column of the clone map (e.g. ``clone_cluster``, ``clonal_pseudotime``,
    ``n_cells``, or a V gene name).
    """
    cm = _get_clone_map(adata)
    if "x" not in cm.columns or "y" not in cm.columns:
        raise ValueError("Clone map has no embedding; run clonotype_recluster(embed_clones=True).")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    sizes = cm[size_by].astype(float)
    lo, hi = size_range
    if sizes.max() > sizes.min():
        s = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * (hi - lo) + lo
    else:
        s = np.full(len(sizes), (lo + hi) / 2)

    values = cm[color]
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        cats = pd.Categorical(values)
        if palette is None:
            cmap = plt.get_cmap("tab20")
            palette = {c: cmap(i % 20) for i, c in enumerate(cats.categories)}
        for cat in cats.categories:
            m = np.asarray(cats == cat)
            ax.scatter(cm["x"][m], cm["y"][m], s=s[m], color=palette.get(cat, "grey"),
                       label=str(cat), alpha=0.85, edgecolors="white", linewidths=0.3)
        ax.legend(title=color, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    else:
        sc = ax.scatter(cm["x"], cm["y"], s=s, c=values.astype(float), cmap="viridis",
                        alpha=0.85, edgecolors="white", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label=color)

    ax.set_xlabel("Clone UMAP 1")
    ax.set_ylabel("Clone UMAP 2")
    ax.set_title(title or f"Clonotype map (n={len(cm)} clones, colored by {color})")
    fig = ax.get_figure()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax


def cells(
    adata,
    color: str = "clone_cluster",
    basis: str = "X_umap",
    ax=None,
    palette: dict | None = None,
    s: float = 10,
    title: str | None = None,
    save: str | None = None,
):
    """UMAP of single cells colored by clone cluster (the 'map-back' view)."""
    if basis not in adata.obsm:
        raise KeyError(f"{basis!r} not in adata.obsm.")
    if color not in adata.obs.columns:
        raise KeyError(f"{color!r} not in adata.obs.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    xy = np.asarray(adata.obsm[basis])
    values = adata.obs[color]

    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        cats = pd.Categorical(values)
        if palette is None:
            cmap = plt.get_cmap("tab20")
            palette = {c: cmap(i % 20) for i, c in enumerate(cats.categories)}
        # unassigned cells first, in light grey
        na = values.isna().to_numpy()
        if na.any():
            ax.scatter(xy[na, 0], xy[na, 1], s=s, color="lightgrey", alpha=0.3,
                       label="no BCR", rasterized=True)
        for cat in cats.categories:
            m = np.asarray(cats == cat) & ~na
            ax.scatter(xy[m, 0], xy[m, 1], s=s, color=palette.get(cat, "grey"),
                       label=str(cat), alpha=0.8, rasterized=True)
        ax.legend(title=color, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  frameon=False, markerscale=2)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=pd.to_numeric(values, errors="coerce"),
                        cmap="coolwarm", alpha=0.8, rasterized=True)
        plt.colorbar(sc, ax=ax, label=color)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"Cells colored by {color}")
    fig = ax.get_figure()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax


def signature_heatmap(
    adata,
    signatures: dict,
    groupby: str = "clone_cluster",
    layer: str | None = None,
    z_score: bool = True,
    ax=None,
    cmap: str = "viridis",
    save: str | None = None,
):
    """Mean gene-signature expression per clone cluster (z-scored).

    Parameters
    ----------
    adata
        AnnData (expects log-normalized expression in ``X`` or ``layer``).
    signatures
        Mapping of signature name -> list of genes, e.g.
        ``{"Plasma": ["PRDM1", "XBP1"], "Memory": ["CCR6"]}``.
    groupby
        ``obs`` column to group by (default: clone clusters).
    layer
        Expression layer; ``None`` uses ``X``.
    z_score
        Z-score each signature across groups.
    """
    obs_groups = adata.obs[groupby]
    groups = [g for g in pd.Categorical(obs_groups).categories if g is not np.nan]

    x = adata.layers[layer] if layer else adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    expr = pd.DataFrame(np.asarray(x), index=adata.obs_names, columns=adata.var_names)

    mat = pd.DataFrame(index=groups, columns=list(signatures), dtype=float)
    for g in groups:
        cells_idx = (obs_groups == g).to_numpy()
        if cells_idx.sum() == 0:
            continue
        sub = expr.iloc[cells_idx]
        for sig, genes in signatures.items():
            present = [ge for ge in genes if ge in expr.columns]
            if present:
                mat.loc[g, sig] = float(sub[present].to_numpy().mean())
    mat = mat.dropna(how="all")

    if z_score:
        mat = (mat - mat.mean()) / mat.std(ddof=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 0.9 * mat.shape[1]), max(3, 0.6 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > mat.to_numpy().std() else "black")
    plt.colorbar(im, ax=ax, label="z-scored mean expression" if z_score else "mean expression")
    ax.set_xlabel("Gene signature")
    ax.set_ylabel(groupby)
    fig = ax.get_figure()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax, mat
