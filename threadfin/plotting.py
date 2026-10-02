"""Figures for Threadfin results (matplotlib; publication style).

Every function accepts an optional ``ax`` (or ``fig``), never calls
``plt.show()`` and returns the figure for further editing. ``save=`` writes
the figure (PDF/SVG keep text editable).

Colour rules used throughout (validated categorical palette, fixed order):

* programmes / categories take slots 1-8 in a fixed order and never cycle;
  a 9th category folds into grey "other";
* scatter plots with more than three categories are drawn as small
  multiples (each category highlighted on grey), because more than three
  hues cannot all be told apart when any two points may touch;
* text is always ink-coloured; colour lives on the marks only.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# validated categorical order (adjacent-pair CVD and normal-vision checks pass)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a7e", "#c03a3a"]


# --------------------------------------------------------------------------- style


def set_style(font_size: float = 7.0) -> None:
    """Nature-style defaults: small sans text, hairline axes, editable PDF text."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def category_colors(categories) -> dict:
    """Fixed-order colours for categories; beyond eight, grey 'other'."""
    cats = list(categories)
    return {c: (PALETTE[i] if i < len(PALETTE) else OTHER) for i, c in enumerate(cats)}


def _finish(fig, save):
    if save:
        fig.savefig(save)
    return fig


def _ax(ax, figsize=(3.2, 2.6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.get_figure(), ax


def _clone_table(adata):
    prof = adata.uns.get("threadfin", {}).get("profiles")
    return None if prof is None else prof["clone_table"]


# --------------------------------------------------------------------------- clone map


def clone_map(
    adata,
    color: str | None = None,
    *,
    size_range: tuple = (2, 60),
    facet: bool | None = None,
    ncols: int = 4,
    ax=None,
    fig=None,
    title: str | None = None,
    save: str | None = None,
    **legacy_kwargs,
):
    """Scatter of clones (one point per clone; size = number of cells).

    With :func:`threadfin.tl.find_programmes` results, points are placed by
    the UMAP of clone profiles and ``color`` defaults to ``clone_programme``.
    More than three categories are drawn as small multiples (one panel per
    category, highlighted on grey) unless ``facet=False``. Numeric columns use
    a single-hue blue scale.

    Falls back to the v3 layout (``uns['threadfin']['clone_map']`` from
    :func:`threadfin.clonotype_recluster`) when no v4 clone table exists.
    """
    table = _clone_table(adata)
    if table is None or "x" not in table.columns:
        return _legacy_clone_map(adata, color=color or "clone_cluster", ax=ax, title=title, save=save,
                                 **legacy_kwargs)
    color = color or "clone_programme"
    tab = table.dropna(subset=["x", "y"])
    if color not in tab.columns:
        raise KeyError(f"'{color}' is not a column of the clone table.")
    n = tab["n_cells"].astype(float)
    lo, hi = size_range
    s = lo + (hi - lo) * (np.log1p(n) - np.log1p(n.min())) / max(np.log1p(n.max()) - np.log1p(n.min()), 1e-9)
    order = np.argsort(n.to_numpy())[::-1]  # draw big clones first, small on top
    values = tab[color]
    categorical = not pd.api.types.is_numeric_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype)

    if categorical:
        cats = [c for c in pd.Categorical(values.dropna()).categories if (values == c).any()]
        if facet is None:
            facet = len(cats) > 3
        if facet:
            nrow = int(np.ceil(len(cats) / ncols))
            if fig is None:
                fig, axes = plt.subplots(nrow, ncols, figsize=(1.45 * ncols, 1.45 * nrow), squeeze=False)
            else:
                axes = np.array(fig.subplots(nrow, ncols, squeeze=False))
            for k, ax_k in enumerate(axes.ravel()):
                ax_k.set_xticks([]), ax_k.set_yticks([])
                for sp_ in ax_k.spines.values():
                    sp_.set_visible(False)
                if k >= len(cats):
                    continue
                cat = cats[k]
                hit = (values == cat).to_numpy()
                ax_k.scatter(tab["x"].to_numpy()[order], tab["y"].to_numpy()[order], s=s.to_numpy()[order] * 0.5,
                             color=OTHER, alpha=0.35, linewidths=0, rasterized=True)
                idx = order[hit[order]]
                ax_k.scatter(tab["x"].to_numpy()[idx], tab["y"].to_numpy()[idx], s=s.to_numpy()[idx],
                             color=PALETTE[0], edgecolors="white", linewidths=0.3, rasterized=True)
                ax_k.set_title(f"{cat}  (n={int(hit.sum())})", fontsize=mpl.rcParams["font.size"], color=INK, pad=2)
            if title:
                fig.suptitle(title, color=INK)
            return _finish(fig, save)
        fig, ax = _ax(ax)
        colors = category_colors(cats)
        for cat in cats:
            hit = (values == cat).to_numpy()
            idx = order[hit[order]]
            ax.scatter(tab["x"].to_numpy()[idx], tab["y"].to_numpy()[idx], s=s.to_numpy()[idx],
                       color=colors[cat], edgecolors="white", linewidths=0.3, label=str(cat), rasterized=True)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=1.2, handletextpad=0.3)
    else:
        fig, ax = _ax(ax)
        cmap = mpl.colors.LinearSegmentedColormap.from_list("tf_blue", BLUE_RAMP)
        sc_ = ax.scatter(tab["x"].to_numpy()[order], tab["y"].to_numpy()[order], s=s.to_numpy()[order],
                         c=values.astype(float).to_numpy()[order], cmap=cmap, edgecolors="white",
                         linewidths=0.3, rasterized=True)
        fig.colorbar(sc_, ax=ax, shrink=0.7, label=color)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("clone profile UMAP 1")
    ax.set_ylabel("clone profile UMAP 2")
    ax.set_title(title or f"{len(tab):,} clones", color=INK)
    return _finish(fig, save)


def _legacy_clone_map(adata, color="clone_cluster", size_by="n_cells", size_range=(20, 400), ax=None,
                      palette=None, title=None, save=None):
    """v3 clone map (one point per clonotype from clonotype_recluster)."""
    cm = adata.uns.get("threadfin", {}).get("clone_map")
    if cm is None:
        raise ValueError("Run threadfin.run() / threadfin.tl.find_programmes() (or the v3 "
                         "clonotype_recluster) first.")
    if "x" not in cm.columns:
        raise ValueError("Clone map has no embedding; run clonotype_recluster(embed_clones=True).")
    fig, ax = _ax(ax, figsize=(7, 6))
    sizes = cm[size_by].astype(float)
    lo, hi = size_range
    s = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * (hi - lo) + lo
         if sizes.max() > sizes.min() else np.full(len(sizes), (lo + hi) / 2))
    values = cm[color]
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        cats = pd.Categorical(values)
        palette = palette or category_colors(cats.categories)
        for cat in cats.categories:
            m = np.asarray(cats == cat)
            ax.scatter(cm["x"][m], cm["y"][m], s=s[m], color=palette.get(cat, OTHER), label=str(cat),
                       alpha=0.85, edgecolors="white", linewidths=0.3)
        ax.legend(title=color, loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        sc_ = ax.scatter(cm["x"], cm["y"], s=s, c=values.astype(float), cmap="viridis", alpha=0.85,
                         edgecolors="white", linewidths=0.3)
        plt.colorbar(sc_, ax=ax, label=color)
    ax.set_xlabel("Clone UMAP 1")
    ax.set_ylabel("Clone UMAP 2")
    ax.set_title(title or f"Clonotype map (n={len(cm)} clones, colored by {color})")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax


# --------------------------------------------------------------------------- statistics panels


def coherence(adata, *, ax=None, save=None):
    """Observed clonal ICC next to the within-stratum permutation null.

    Two bars: the share of variance explained by clone identity, and the same
    statistic after shuffling clone labels within strata (whisker = 95% of
    permutations). The excess of the first over the second is the clonal
    signal beyond sampling.
    """
    res = adata.uns.get("threadfin", {}).get("coherence")
    if res is None:
        raise ValueError("Run threadfin.tl.clonal_coherence() first.")
    fig, ax = _ax(ax, figsize=(2.2, 2.2))
    null = np.asarray(res["null"]) * 100
    obs = res["icc"] * 100
    lo, hi = np.quantile(null, [0.025, 0.975])
    ax.bar([0, 1], [obs, null.mean()], width=0.6, color=[PALETTE[0], OTHER])
    ax.errorbar([1], [null.mean()], yerr=[[null.mean() - lo], [hi - null.mean()]], fmt="none",
                ecolor=INK_2, elinewidth=0.8, capsize=2)
    top = max(obs, hi)
    for x, v in ((0, obs), (1, null.mean())):
        ax.text(x, (hi if x == 1 else v) + 0.03 * top, f"{v:.1f}%", ha="center", va="bottom", color=INK)
    ax.set_xticks([0, 1], ["observed", "shuffled\nwithin samples"])
    ax.set_ylim(0, top * 1.25)
    ax.set_ylabel("variance explained by clone (%)")
    ax.set_title(f"clonal coherence (p = {res['p_value']:.2g})", color=INK)
    return _finish(fig, save)


def programme_composition(adata, state_key: str, *, programme_key: str = "clone_programme",
                          ax=None, save=None, legend: bool = True):
    """Stacked bars: fraction of each programme's cells in each cell state."""
    df = adata.obs[[programme_key, state_key]].dropna()
    tab = pd.crosstab(df[programme_key], df[state_key])
    frac = tab.div(tab.sum(axis=1), axis=0)
    states = list(frac.sum(axis=0).sort_values(ascending=False).index)
    if len(states) > 8:  # fold the tail into "other"
        frac["other"] = frac[states[7:]].sum(axis=1)
        states = states[:7] + ["other"]
    frac = frac[states]
    colors = category_colors(states)
    if "other" in colors:
        colors["other"] = OTHER
    fig, ax = _ax(ax, figsize=(3.0, 0.25 * len(frac) + 0.8))
    left = np.zeros(len(frac))
    ypos = np.arange(len(frac))[::-1]
    for st in states:
        vals = frac[st].to_numpy()
        ax.barh(ypos, vals, left=left, height=0.7, color=colors[st], edgecolor="white", linewidth=0.8, label=str(st))
        left += vals
    ax.set_yticks(ypos, [str(i) for i in frac.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of programme cells")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=0.8, handletextpad=0.4)
    return _finish(fig, save)


def stability(adata, *, ax=None, save=None):
    """Bootstrap stability (Jaccard) of each programme, with the 0.75 / 0.6 guides."""
    prog = adata.uns.get("threadfin", {}).get("programmes")
    if prog is None:
        raise ValueError("Run threadfin.tl.find_programmes() first.")
    summ = prog["summary"]
    fig, ax = _ax(ax, figsize=(2.6, 2.0))
    x = np.arange(len(summ))
    ax.bar(x, summ["stability"], width=0.6, color=PALETTE[0])
    for thr, lab in ((prog["params"]["stability_threshold"], "stable"), (0.6, "do not interpret below")):
        ax.axhline(thr, color=MUTED, linewidth=0.6)
        ax.text(1.02, thr, lab, transform=ax.get_yaxis_transform(), color=INK_2, va="center", ha="left",
                fontsize=mpl.rcParams["font.size"] - 1, clip_on=False)
    ax.set_xticks(x, [str(i) for i in summ.index])
    ax.set_xlim(-0.6, len(summ) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("bootstrap Jaccard")
    ax.set_title("programme stability", color=INK)
    return _finish(fig, save)


def association(table: pd.DataFrame, *, level=None, ax=None, save=None, title: str | None = None):
    """Forest plot of clone-level programme effects from :func:`threadfin.tl.association_test`.

    Categorical labels: Mantel-Haenszel odds ratios with 95% CIs (log scale)
    for one ``level``; numeric labels: rank effects. Filled markers mark
    FDR < 0.05.
    """
    tab = table.copy()
    fig, ax = _ax(ax, figsize=(2.8, 0.22 * tab["programme"].nunique() + 0.9))
    if "level" in tab.columns:
        level = level if level is not None else tab.groupby("level")["n_clones_level"].first().idxmin()
        tab = tab[tab["level"].astype(str) == str(level)]
        y = np.arange(len(tab))[::-1]
        est = tab["odds_ratio"].to_numpy(dtype=float)
        lo = tab["ci_low"].to_numpy(dtype=float)
        hi = tab["ci_high"].to_numpy(dtype=float)
        ok = np.isfinite(est) & (est > 0)
        ax.hlines(y[ok], np.where(np.isfinite(lo[ok]), lo[ok], est[ok]),
                  np.where(np.isfinite(hi[ok]), hi[ok], est[ok]), color=INK_2, linewidth=0.8)
        sig = (tab["fdr"] < 0.05).to_numpy()
        ax.scatter(est[ok & sig], y[ok & sig], color=PALETTE[0], s=18, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(est[ok & ~sig], y[ok & ~sig], facecolor="white", edgecolor=PALETTE[0], s=18, zorder=3,
                   linewidths=0.8)
        ax.set_xscale("log")
        ax.axvline(1.0, color=MUTED, linewidth=0.6)
        ax.set_xlabel(f"odds ratio (clones), '{level}'")
    else:
        y = np.arange(len(tab))[::-1]
        eff = tab["rank_effect"].to_numpy(dtype=float)
        sig = (tab["fdr"] < 0.05).to_numpy()
        ax.scatter(eff[sig], y[sig], color=PALETTE[0], s=18, zorder=3)
        ax.scatter(eff[~sig], y[~sig], facecolor="white", edgecolor=PALETTE[0], s=18, zorder=3, linewidths=0.8)
        ax.axvline(0.0, color=MUTED, linewidth=0.6)
        ax.set_xlabel("rank effect (programme vs other clones)")
    ax.set_yticks(y, tab["programme"].astype(str))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title or (str(tab["label"].iloc[0]) if len(tab) else ""), color=INK)
    return _finish(fig, save)


def memory(res: dict, *, ax=None, save=None):
    """Clonal memory: how much a clone changes between snapshots, vs random clones.

    Bars show the mean noise-corrected squared change of the same clone and of
    random same-donor clones; the title gives the memory index with its 95% CI.
    """
    fig, ax = _ax(ax, figsize=(2.2, 2.0))
    vals = [res["mean_change"], res["mean_change_random"]]
    ax.bar([0, 1], vals, width=0.6, color=[PALETTE[0], OTHER])
    ax.set_xticks([0, 1], ["same\nclone", "random\nclone"])
    ax.set_ylabel("state change between snapshots")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    lo, hi = res["memory_index_ci"]
    ax.set_title(f"memory index {res['memory_index']:.2f} [{lo:.2f}, {hi:.2f}]", color=INK)
    return _finish(fig, save)


def heritability(table: pd.DataFrame, *, highlight: dict | None = None, n_label: int = 12,
                 ax=None, save=None):
    """Genes ranked by clonal ICC; optional gene sets highlighted (<= 3 sets)."""
    fig, ax = _ax(ax, figsize=(3.0, 2.2))
    tab = table.sort_values("icc", ascending=False)
    rank = np.arange(1, len(tab) + 1)
    ax.scatter(rank, tab["icc"], s=3, color=OTHER, linewidths=0, rasterized=True)
    if highlight:
        for (name, genes), col in zip(highlight.items(), PALETTE[:3]):
            m = tab.index.isin(genes)
            ax.scatter(rank[m], tab["icc"][m], s=12, color=col, edgecolors="white", linewidths=0.3, label=name)
        ax.legend(loc="upper right", handletextpad=0.3)
    for i, (gene, row) in enumerate(tab.head(n_label).iterrows()):
        ax.text(rank[i] * 1.15, row["icc"], gene, fontsize=mpl.rcParams["font.size"] - 1.5, color=INK_2,
                va="center", fontstyle="italic")
    ax.set_xscale("log")
    ax.set_xlabel("gene rank")
    ax.set_ylabel("clonal ICC")
    return _finish(fig, save)


def overview(adata, *, state_key: str | None = None, save=None):
    """Four-panel summary of a :func:`threadfin.run` result."""
    set_style()
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.6)
    sub = fig.add_subfigure(gs[:, 0])
    clone_map(adata, fig=sub, ncols=2, facet=True)
    sub.suptitle("a  clonal programmes on the clone map", x=0.02, ha="left", color=INK, fontweight="bold")
    ax_b = fig.add_subplot(gs[0, 1])
    coherence(adata, ax=ax_b)
    ax_c = fig.add_subplot(gs[0, 2])
    stability(adata, ax=ax_c)
    ax_d = fig.add_subplot(gs[1, 1:])
    if state_key is not None:
        programme_composition(adata, state_key, ax=ax_d)
        ax_d.set_title("programme composition by cell state", color=INK)
    else:
        summ = adata.uns["threadfin"]["programmes"]["summary"]
        ax_d.bar(range(len(summ)), summ["n_clones"], width=0.6, color=PALETTE[0])
        ax_d.set_xticks(range(len(summ)), [str(i) for i in summ.index])
        ax_d.set_ylabel("clones")
        ax_d.set_title("programme sizes", color=INK)
    for ax, lab, dx in ((ax_b, "b", -0.28), (ax_c, "c", -0.28), (ax_d, "d", -0.12)):
        ax.text(dx, 1.12, lab, transform=ax.transAxes, fontweight="bold", color=INK,
                fontsize=mpl.rcParams["font.size"] + 2)
    return _finish(fig, save)


# --------------------------------------------------------------------------- cell-level panels (v3)


def cells(adata, color: str = "clone_cluster", basis: str = "X_umap", ax=None, palette: dict | None = None,
          s: float = 10, title: str | None = None, save: str | None = None):
    """Cells on an embedding coloured by an ``obs`` column (cells without a value in grey)."""
    if basis not in adata.obsm:
        raise KeyError(f"{basis!r} not in adata.obsm.")
    if color not in adata.obs.columns:
        raise KeyError(f"{color!r} not in adata.obs.")
    fig, ax = _ax(ax, figsize=(7, 6))
    xy = np.asarray(adata.obsm[basis])
    values = adata.obs[color]
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        cats = pd.Categorical(values)
        palette = palette or category_colors(cats.categories)
        na = values.isna().to_numpy()
        if na.any():
            ax.scatter(xy[na, 0], xy[na, 1], s=s, color="lightgrey", alpha=0.3, label="no BCR", rasterized=True)
        for cat in cats.categories:
            m = np.asarray(cats == cat) & ~na
            ax.scatter(xy[m, 0], xy[m, 1], s=s, color=palette.get(cat, OTHER), label=str(cat), alpha=0.8,
                       rasterized=True)
        ax.legend(title=color, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2)
    else:
        cmap = mpl.colors.LinearSegmentedColormap.from_list("tf_blue", BLUE_RAMP)
        sc_ = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=pd.to_numeric(values, errors="coerce"), cmap=cmap,
                         alpha=0.8, rasterized=True)
        plt.colorbar(sc_, ax=ax, label=color)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"Cells colored by {color}")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax


def signature_heatmap(adata, signatures: dict, groupby: str = "clone_cluster", layer: str | None = None,
                      z_score: bool = True, ax=None, cmap: str = "viridis", save: str | None = None):
    """Mean expression of gene signatures per group (z-scored across groups).

    Only the signature genes are densified, so this is safe on large datasets.
    """
    obs_groups = adata.obs[groupby]
    groups = list(pd.Categorical(obs_groups.dropna()).categories)
    genes = sorted({g for gs_ in signatures.values() for g in gs_ if g in adata.var_names})
    x = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    expr = pd.DataFrame(np.asarray(x), index=adata.obs_names, columns=genes)
    mat = pd.DataFrame(index=groups, columns=list(signatures), dtype=float)
    for g in groups:
        idx = (obs_groups == g).to_numpy()
        if idx.sum() == 0:
            continue
        sub = expr.iloc[idx]
        for sig, gs_ in signatures.items():
            present = [ge for ge in gs_ if ge in expr.columns]
            if present:
                mat.loc[g, sig] = float(sub[present].to_numpy().mean())
    mat = mat.dropna(how="all")
    if z_score:
        mat = (mat - mat.mean()) / mat.std(ddof=0)
    fig, ax = _ax(ax, figsize=(max(4, 0.9 * mat.shape[1]), max(3, 0.6 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    vals = mat.to_numpy(dtype=float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(vals[i, j]) > np.nanstd(vals) else "black")
    plt.colorbar(im, ax=ax, label="z-scored mean expression" if z_score else "mean expression")
    ax.set_xlabel("Gene signature")
    ax.set_ylabel(groupby)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax, mat
