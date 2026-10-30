#!/usr/bin/env python
"""Draw every Threadfin figure of the tutorial-path runs, one PNG per figure, with an index per dataset and a
gallery page.

run_dataset.py keeps, for every dataset, the AnnData after ``threadfin.run`` (<results>/<dataset>.h5ad) and the
record of the run with its settings (<results>/<dataset>.json). This script draws from those two files only, so
the figures can be redrawn without running the analysis again. For each dataset it writes
<gallery>/<dataset>/: the overview that ``result.plot()`` saves, every other ``threadfin.pl`` figure that applies
to the dataset (clone maps, coherence, stability, programme composition, label effects, programme associations,
clonal memory, gene heritability, cells, marker signatures), a README.md that says what each image shows, and
figures.json (the same as data). <gallery>/README.md lists every dataset with a thumbnail and links.

Three figures need inputs that ``threadfin.run`` does not compute: the cell embedding of ``pl.cells`` (a UMAP of
the run's own expression embedding), the gene table of ``pl.heritability`` (``tl.gene_heritability``, default
settings) and the expression of the case-study marker sets for ``pl.signature_heatmap``. They are computed once
from the kept h5ad and cached in <results>/<dataset>_gallery_cache/.

PNGs are drawn at 300 dpi; a figure that would exceed 1 MB is drawn again at a lower resolution. Every PNG is
re-encoded losslessly (no PNG row filter + zlib level 9, which suits flat-colour figures, or Pillow's optimiser,
whichever is smaller; ``--zopfli`` adds a slow zopflipng pass when that program is installed).

Usage:
    python make_gallery.py <results_dir> <gallery_dir> [--datasets ds ...] [--page-only] [--no-page] [--zopfli]
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MANUSCRIPT / "case_studies"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

import threadfin as tf  # noqa: E402
from summarise_results import NAMES  # noqa: E402

DPI = 300                         # publication resolution
DPI_FALLBACK = (250, 200, 150)    # used only when a PNG would exceed MAX_BYTES
MAX_BYTES = 1_000_000
THUMB_WIDTH = 480
INK = getattr(tf.pl, "INK", "#0b0b0b")

# Programme-composition panels, one PNG per label for each entry: (name in the file name, keyword arguments of
# tf.pl.programme_composition, what one bar shows). An entry whose keywords the installed threadfin does not accept
# is left out and noted in the index, so the clone-weighted panel appears from threadfin 4.6.0 on (weight="clones").
COMPOSITION_WEIGHTS = [
    ("cells", {}, "the fraction of the programme's cells in each level of `{label}`"),
    ("clones", {"weight": "clones"}, "the mean over the programme's clones of each clone's fraction of cells in each "
                                     "level of `{label}`, so that every clone counts once"),
]


# --------------------------------------------------------------------------- reading a kept result


def _read_elem():
    try:
        from anndata.io import read_elem        # anndata >= 0.11
    except ImportError:                         # pragma: no cover
        from anndata.experimental import read_elem
    return read_elem


def read_result(h5: Path):
    """The AnnData of a run without its expression matrices (obs, var, obsm and uns only)."""
    import anndata as ad
    import h5py

    read_elem = _read_elem()
    with h5py.File(h5, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        obsm = {k: read_elem(f["obsm"][k]) for k in f["obsm"]} if "obsm" in f else {}
        uns = read_elem(f["uns"]) if "uns" in f else {}
    return ad.AnnData(obs=obs, var=var, obsm=obsm, uns=uns)


def read_layer(h5: Path, name: str):
    import h5py

    with h5py.File(h5, "r") as f:
        if "layers" in f and name in f["layers"]:
            return _read_elem()(f["layers"][name])
    return None


def run_settings(ds: str, rec: dict) -> dict:
    """Settings of the run: recorded in the json since this script exists, else those of run_dataset.py."""
    if rec.get("settings"):
        return dict(rec["settings"])
    from run_dataset import SETTINGS

    s = SETTINGS[ds]
    return {"clone_key": "clone_id", "donor_key": "donor", "sample_key": s["sample_key"],
            "state_key": s.get("state_key"), "time_key": s.get("time_key"),
            "basis": "X_published" if ds.startswith("larry") else "X_threadfin"}


def threadfin_result(adata, rec: dict, settings: dict):
    """A ``ThreadfinResult`` rebuilt from the kept AnnData, so that ``plot()`` draws what the run drew."""
    uns = adata.uns.get("threadfin", {})
    if "run" in uns and hasattr(tf.ThreadfinResult, "from_adata"):   # threadfin >= 4.6.0 records the run itself
        return tf.ThreadfinResult.from_adata(adata)
    summary = uns.get("programmes", {}).get("summary")
    programmes = summary if summary is not None else pd.DataFrame(
        columns=["n_clones", "n_cells", "mean_reliability", "mean_confidence", "stability"])
    state_key, time_key = settings.get("state_key"), settings.get("time_key")
    composition = None
    if state_key and len(programmes):
        weight = settings.get("composition_weight")
        kw = {"weight": weight} if weight and "weight" in inspect.signature(tf.tl.programme_composition).parameters \
            else {}
        composition = tf.tl.programme_composition(adata, state_key, **kw)
    values = dict(adata=adata, coherence=uns["coherence"], programmes=programmes,
                  clones=uns["profiles"]["clone_table"], composition=composition,
                  tests=dict(uns.get("associations", {})), profile_tests=dict(uns.get("profile_associations", {})),
                  memory=(uns.get("memory") or {}).get(time_key) if time_key else None,
                  settings=settings, skipped=rec.get("skipped") or {})
    names = {f.name for f in dataclasses.fields(tf.ThreadfinResult)}
    return tf.ThreadfinResult(**{k: v for k, v in values.items() if k in names})


# --------------------------------------------------------------------------- inputs that run() does not compute


def _stamp(h5: Path) -> str:
    st = h5.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def cell_embedding(adata, basis: str, cache: Path, stamp: str):
    """2-D coordinates of every cell for pl.cells, and the name of their axes.

    A 2-D basis (LARRY's published embedding) is used as it is; otherwise a UMAP of the run's embedding with
    the settings of the case studies (15 neighbours, seed 0).
    """
    x = np.asarray(adata.obsm[basis], dtype=np.float32)
    if x.shape[1] == 2:
        return x, basis[2:] if basis.startswith("X_") else basis
    f = cache / "cell_umap.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if str(z["stamp"]) == stamp and z["xy"].shape[0] == adata.n_obs:
            return z["xy"], "UMAP"
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names.copy()), obsm={"X": x})
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=0)
    sc.tl.umap(tmp, random_state=0)
    xy = np.asarray(tmp.obsm["X_umap"], dtype=np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, xy=xy, stamp=stamp, basis=basis)
    return xy, "UMAP"


def case_study_gene_sets(ds: str) -> dict:
    """The marker sets of the case studies (run_case_study.GENE_SETS_HUMAN, in mouse case for mouse data)."""
    with warnings.catch_warnings():             # run_case_study silences warnings on import; keep them here
        import run_case_study
    return run_case_study.gene_sets_for(ds)


def expression_inputs(adata, h5: Path, cache: Path, stamp: str, gene_sets: dict):
    """Gene heritability table and the expression of the marker-set genes, both cached.

    Returns ``(heritability, signature_adata)``; either is None when the data have no genes.
    """
    import anndata as ad
    import scipy.sparse as sp

    meta_f, herit_f = cache / "expression_inputs.json", cache / "gene_heritability.csv"
    sig_f, sig_genes_f = cache / "signature_expression.npz", cache / "signature_genes.json"
    fresh = meta_f.exists() and json.loads(meta_f.read_text()).get("stamp") == stamp
    if not fresh:
        for old in (meta_f, herit_f, sig_f, sig_genes_f):   # from an earlier h5ad of the same name
            old.unlink(missing_ok=True)
        layer = read_layer(h5, "log_norm") if adata.n_vars else None
        meta = {"stamp": stamp, "threadfin": tf.__version__, "layer": "log_norm", "heritability": None,
                "signature_genes": []}
        if layer is not None:
            adata.layers["log_norm"] = layer
            try:
                herit = tf.tl.gene_heritability(adata, verbose=False)
                cache.mkdir(parents=True, exist_ok=True)
                herit.to_csv(herit_f)
                meta["heritability"] = "tl.gene_heritability(adata) with its defaults"
                meta["receptor_gene_icc"] = adata.uns["threadfin"].get("gene_heritability_receptor_icc")
            except ValueError as e:
                meta["heritability"] = f"not computed: {e}"
            first = {}                                     # gene names need not be unique; take the first
            for i, g in enumerate(adata.var_names.astype(str)):
                first.setdefault(g, i)
            genes = sorted({g for gs in gene_sets.values() for g in gs if g in first})
            if genes:
                idx = np.array([first[g] for g in genes])
                sub = sp.csr_matrix(layer[:, idx]) if sp.issparse(layer) else sp.csr_matrix(np.asarray(layer)[:, idx])
                cache.mkdir(parents=True, exist_ok=True)
                sp.save_npz(sig_f, sub)
                sig_genes_f.write_text(json.dumps(genes))
                meta["signature_genes"] = genes
            del adata.layers["log_norm"]
        cache.mkdir(parents=True, exist_ok=True)
        meta_f.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    herit = pd.read_csv(herit_f, index_col=0) if herit_f.exists() else None
    sig = None
    if sig_f.exists() and sig_genes_f.exists():
        genes = json.loads(sig_genes_f.read_text())
        sig = ad.AnnData(X=sp.load_npz(sig_f), obs=adata.obs.copy(), var=pd.DataFrame(index=genes))
    return herit, sig


# --------------------------------------------------------------------------- PNG writing


def _png_filter0(a: np.ndarray) -> bytes:
    """PNG with no row filter and zlib level 9: much smaller than adaptive filtering for flat-colour figures."""
    h, w, c = a.shape
    raw = np.empty((h, 1 + w * c), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a.reshape(h, -1)
    comp = zlib.compressobj(9, zlib.DEFLATED, 15, 9)
    data = comp.compress(raw.tobytes()) + comp.flush()

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    colour_type = {3: 2, 4: 6}[c]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0))
            + chunk(b"IDAT", data) + chunk(b"IEND", b""))


def optimise_png(data: bytes) -> bytes:
    """The smallest lossless encoding of a PNG among: no row filter + zlib 9, Pillow's optimiser, the input."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode == "RGBA" and img.getextrema()[3] == (255, 255):
        img = img.convert("RGB")                 # an opaque alpha channel carries nothing
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    pixels = np.asarray(img)
    candidates = [data, _png_filter0(pixels)]
    buf = io.BytesIO()
    img.save(buf, format="png", optimize=True)
    candidates.append(buf.getvalue())
    for cand in sorted(candidates, key=len):
        back = np.asarray(Image.open(io.BytesIO(cand)).convert(img.mode))
        if back.shape == pixels.shape and np.array_equal(back, pixels):
            return cand
    return data                                  # pragma: no cover - the input itself always decodes


def zopflipng_path() -> str | None:
    return os.environ.get("ZOPFLIPNG") or shutil.which("zopflipng") or next(
        (p for p in ("/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/bin/zopflipng",)
         if os.path.exists(p)), None)


def _zopfli_one(args) -> tuple[str, int, int]:
    exe, path = args
    path = Path(path)
    before = path.stat().st_size
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "z.png"
        try:
            subprocess.run([exe, "-y", "--filters=0", "--iterations=1", str(path), str(out)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        except (subprocess.SubprocessError, OSError):
            return str(path), before, before
        if out.exists() and 0 < out.stat().st_size < before:
            shutil.move(str(out), path)
    return str(path), before, path.stat().st_size


def save_png(fig, path: Path) -> dict:
    """Save at 300 dpi (lower only if the PNG would exceed 1 MB), re-encoded losslessly."""
    from PIL import Image

    for dpi in (DPI,) + DPI_FALLBACK:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        data = optimise_png(buf.getvalue())
        if len(data) <= MAX_BYTES:
            break
    path.write_bytes(data)
    w, h = Image.open(io.BytesIO(data)).size
    return {"dpi": dpi, "bytes": len(data), "width": w, "height": h}


def thumbnail(src: Path, dst: Path, width: int = THUMB_WIDTH) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGB")
    img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(optimise_png(buf.getvalue()))


# --------------------------------------------------------------------------- the figures of one dataset


@dataclasses.dataclass
class GalleryFigure:
    name: str                     # file stem
    title: str                    # heading in the index
    shows: str                    # what the image shows, one or two sentences
    draw: Callable                # returns a matplotlib figure (or a tuple whose first item is one)


def _fig(out):
    return out[0] if isinstance(out, tuple) else out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "label"


def _strip_text_panels(fig) -> None:
    """Remove panels that hold only a sentence (the overview's message when there are no programmes)."""
    for ax in list(fig.get_axes()):
        if not ax.axison and ax.texts and not (ax.lines or ax.patches or ax.collections or ax.images):
            ax.remove()


def _is_categorical(values: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
            or isinstance(values.dtype, pd.CategoricalDtype))


def _natural_key(value) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def _ordered(values: pd.Series, order: str = "frequency", like: pd.Series | None = None) -> pd.Series:
    """Categorical of ``values`` (as text) with its levels in display order.

    ``order``: "natural" (P1, P2, ..., P10; M1, M2, ..., M10), "frequency" (most frequent first) or
    "categories" (the order of the categorical ``like``, for ordered annotations such as time points).
    """
    text = values.astype(object).where(values.notna(), None).map(lambda v: None if v is None else str(v))
    present = list(text.dropna().value_counts().index)            # most frequent first
    if order == "natural":
        present = sorted(present, key=_natural_key)
    elif order == "categories" and like is not None and isinstance(like.dtype, pd.CategoricalDtype):
        known = [str(c) for c in like.cat.categories]
        present = [c for c in known if c in set(present)] + [c for c in present if c not in set(known)]
    return pd.Series(pd.Categorical(text, categories=present), index=values.index)


def _order_for(column: str, obs_values: pd.Series | None) -> str:
    if column in ("clone_programme", "clone_programme_assigned", "donor"):
        return "natural"
    if obs_values is not None and isinstance(obs_values.dtype, pd.CategoricalDtype) and obs_values.cat.ordered:
        return "categories"
    return "frequency"


def _facet_figure(n: int, panel: float = 1.75, max_cols: int = 4):
    ncols = min(max_cols, n)
    nrow = math.ceil(n / ncols)
    return plt.figure(figsize=(panel * ncols, panel * nrow + 0.35)), ncols


def _legend_handles(leg) -> list:
    return list(leg.legend_handles) if hasattr(leg, "legend_handles") else list(leg.legendHandles)


def draw_clone_map(adata, column: str, *, title: str, label: str | None = None):
    """pl.clone_map with a figure sized for its panels; ``label`` renames the colour bar.

    Categories are always drawn as small multiples (one panel per level, every clone in grey behind), so that
    clones without a level stay visible and every panel uses the same marker sizes.
    """
    values = adata.uns["threadfin"]["profiles"]["clone_table"][column]
    if _is_categorical(values):
        n = int(values.dropna().astype(str).nunique())
        fig, ncols = _facet_figure(n)
        fig = _fig(tf.pl.clone_map(adata, column, facet=True, ncols=ncols, fig=fig, title=title))
        fig.subplots_adjust(hspace=0.35, wspace=0.08, top=1 - 0.5 / fig.get_figheight())
        _centre_suptitle(fig)
        return fig
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    tf.pl.clone_map(adata, column, ax=ax, title=title)
    if label:
        fig.axes[-1].set_ylabel(label)
    return fig


def drawable_levels(table: pd.DataFrame) -> tuple[list, list]:
    """Levels of a categorical association table with at least one finite, positive odds ratio, and the rest.

    An odds ratio of 0 or infinity (every clone of the level inside, or outside, one programme) has no place on
    pl.association's log axis, so a level made only of such programmes would be an empty panel.
    """
    drawn, empty = [], []
    for level in dict.fromkeys(table["level"].astype(str)):
        est = table.loc[table["level"].astype(str) == level, "odds_ratio"].to_numpy(dtype=float)
        (drawn if (np.isfinite(est) & (est > 0)).any() else empty).append(level)
    return drawn, empty


def draw_association(table: pd.DataFrame, label: str):
    """pl.association for every drawable level of a categorical label (one panel each), or once for a number."""
    n_prog = table["programme"].nunique()
    height = 0.22 * n_prog + 0.9
    if "level" not in table.columns:
        fig, ax = plt.subplots(figsize=(3.0, height))
        tf.pl.association(table, ax=ax, title=label)
        return fig
    levels, _ = drawable_levels(table)
    ncols = min(3, len(levels))
    nrow = math.ceil(len(levels) / ncols)
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.9 * ncols, (height + 0.45) * nrow), squeeze=False)
    for ax, level in zip(axes.ravel(), levels):
        tf.pl.association(table, level=level, ax=ax, title=f"{label}: {level}")
        _readable_log_axis(ax)
    for ax in axes.ravel()[len(levels):]:
        ax.remove()
    fig.tight_layout(w_pad=2.0, h_pad=1.2)
    return fig


def _readable_log_axis(ax) -> None:
    """A log odds-ratio axis spanning less than two decades shows one labelled tick, or crowded minor labels;
    widen it to include 0.1, 1 and 10 (the data are unchanged) and leave minor ticks unlabelled."""
    if ax.get_xscale() != "log":
        return
    lo, hi = ax.get_xlim()
    if np.log10(hi) - np.log10(lo) < 2:
        ax.set_xlim(min(lo, 10 ** -1.05), max(hi, 10 ** 1.05))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def draw_label_effects(results: dict):
    """pl.label_effects with its legend moved beside the axes, where no bar can run into it."""
    fig = _fig(tf.pl.label_effects(results))
    ax = fig.axes[0]
    leg = ax.get_legend()
    if leg is not None:
        handles, labels = _legend_handles(leg), [t.get_text() for t in leg.get_texts()]
        leg.remove()
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                  fontsize=matplotlib.rcParams["font.size"] - 1)
    return fig


def draw_heritability(table: pd.DataFrame):
    """pl.heritability with the gene names in a column beside the axes, each joined to its point by a hairline.

    pl.heritability writes each name next to its point, so genes of nearly equal ICC overprint one another. The
    column keeps their order (highest ICC on top), and the y axis keeps its range (an ICC cannot exceed 1).
    """
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    tf.pl.heritability(table, ax=ax)
    ax.set_title(f"{len(table):,} genes", color=INK)
    ranked = table.sort_values("icc", ascending=False)
    rank = {}
    for i, (gene, icc) in enumerate(zip(ranked.index.astype(str), ranked["icc"].to_numpy(dtype=float)), 1):
        rank.setdefault(gene, (i, icc))
    texts = sorted((t for t in ax.texts if t.get_text() in rank), key=lambda t: rank[t.get_text()][0])
    if not texts:
        return fig
    points = [rank[t.get_text()] for t in texts]  # each name's own point: (rank, ICC)
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72 / fig.dpi
    step = min(texts[0].get_fontsize() * 1.4 / height_pt * (hi - lo), (hi - lo) / len(texts))
    beside = ax.get_yaxis_transform()            # x in axes fractions, y in data units
    for i, (t, (px, py)) in enumerate(zip(texts, points)):
        y = hi - (i + 0.5) * step
        t.set_transform(beside)
        t.set_position((1.05, y))
        t.set_ha("left")
        t.set_va("center")
        ax.annotate("", xy=(px, py), xycoords="data", xytext=(1.04, y), textcoords=beside,
                    arrowprops=dict(arrowstyle="-", color="#898781", linewidth=0.35, shrinkA=0, shrinkB=2))
    return fig


def _centre_suptitle(fig) -> None:
    """Centre a figure title over its panels rather than over the whole canvas."""
    title = getattr(fig, "_suptitle", None)
    axes = fig.get_axes()
    if title is not None and axes:
        left = min(a.get_position().x0 for a in axes)
        right = max(a.get_position().x1 for a in axes)
        title.set_x((left + right) / 2)


def _contrast_texts(ax) -> None:
    """Numbers on a heatmap in white on dark cells and in ink on light cells."""
    if not ax.images:
        return
    im = ax.images[0]
    arr = np.asarray(im.get_array(), dtype=float)
    for t in ax.texts:
        x, y = t.get_position()
        i, j = int(round(y)), int(round(x))
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and np.isfinite(arr[i, j]):
            r, g, b, _ = im.cmap(im.norm(arr[i, j]))
            t.set_color("white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK)


def draw_cells(adata, color: str, basis: str, axis_name: str, title: str, *, na_label: str = "no value"):
    """pl.cells; a categorical colour as small multiples (each level on the grey of all cells), as pl.clone_map
    draws categories, so that no level is hidden under the levels drawn after it."""
    n = adata.n_obs
    s = float(np.clip(15000 / max(n, 1), 0.2, 8.0))
    values = adata.obs[color]
    levels = (list(_ordered(values, _order_for(color, values), values).cat.categories)
              if _is_categorical(values) else [])
    if len(levels) >= 2:
        fig, ncols = _facet_figure(len(levels), panel=1.9)
        axes = np.array(fig.subplots(math.ceil(len(levels) / ncols), ncols, squeeze=False))
        key = "_gallery_level"
        text = values.astype(object).where(values.notna(), None).map(lambda v: None if v is None else str(v))
        for ax, level in zip(axes.ravel(), levels):
            hit = text.eq(level).to_numpy()
            adata.obs[key] = pd.Categorical(np.where(hit, level, None), categories=[level])
            tf.pl.cells(adata, color=key, basis=basis, ax=ax, s=s, palette={level: tf.pl.PALETTE[0]})
            ax.get_legend().remove()
            ax.set_title(f"{level}  (n={int(hit.sum()):,})", fontsize=matplotlib.rcParams["font.size"], color=INK, pad=2)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        del adata.obs[key]
        for ax in axes.ravel()[len(levels):]:
            ax.remove()
        fig.suptitle(title, color=INK)
        fig.subplots_adjust(hspace=0.3, wspace=0.08, top=1 - 0.5 / fig.get_figheight())
        _centre_suptitle(fig)
        return fig
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    tf.pl.cells(adata, color=color, basis=basis, ax=ax, s=s, title=title)
    ax.set_xlabel(f"{axis_name} 1")
    ax.set_ylabel(f"{axis_name} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    leg = ax.get_legend()
    if leg is not None:
        if color == "clone_programme":
            leg.set_title("programme")
        for text in leg.get_texts():
            if text.get_text() == "no BCR":     # pl.cells names every missing value so; here it may be any label
                text.set_text(na_label)
        for handle in _legend_handles(leg):
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([16.0])
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(1.0)
    fig.tight_layout()
    return fig


def plan(ds: str, adata, result, settings: dict, rec: dict, h5: Path, cache: Path):
    """Every figure that applies to this dataset, in the order of the index, and notes on those left out."""
    uns = adata.uns["threadfin"]
    table = uns["profiles"]["clone_table"]
    n_prog = len(result.programmes)
    state_key, time_key = settings.get("state_key"), settings.get("time_key")
    tests = [t for t in rec.get("test") or [] if t in adata.obs.columns]
    labels = list(dict.fromkeys([k for k in [state_key, *tests] if k and k in adata.obs.columns]))
    cell_labels = [k for k in labels if _is_categorical(adata.obs[k]) and adata.obs[k].dropna().nunique() >= 2]
    number_labels = [k for k in labels if not _is_categorical(adata.obs[k]) and adata.obs[k].notna().any()]
    figs: list[GalleryFigure] = []
    notes: list[dict] = []
    panels = "one panel per {}, every other clone in grey"

    # 1. the overview of result.plot()
    def overview():
        fig = _fig(result.plot())
        if n_prog == 0:
            _strip_text_panels(fig)
        return fig

    panel_d = (f"programme composition by `{state_key}`" if state_key and n_prog else "programme sizes")
    figs.append(GalleryFigure("overview", "Overview (`result.plot()`)",
                              "The figure that `result.plot()` saves: (a) clonal programmes on the clone map, (b) "
                              "clonal coherence, (c) programme stability, (d) " + panel_d + ".", overview))

    # 2. clonal coherence
    coh = uns["coherence"]
    figs.append(GalleryFigure("coherence", "Clonal coherence",
                              "Share of transcriptional variance explained by clone identity (clonal ICC), observed "
                              f"and with clone labels shuffled within each `{coh.get('strata_key')}` (bar: mean of "
                              f"{int(coh.get('n_perm', 0))} permutations; whisker: their central 95 %).",
                              lambda: tf.pl.coherence(adata)))

    # 3. clone maps
    if n_prog >= 2:
        table["_gallery_programme"] = _ordered(table["clone_programme"], "natural").values
        figs.append(GalleryFigure("clone_map_programmes", "Clone map: programmes",
                                  "The clone map (UMAP of clone profiles; one point per clone, area by its number of "
                                  "cells) coloured by clonal programme, " + panels.format("programme") + ".",
                                  lambda: draw_clone_map(adata, "_gallery_programme", title="clones by programme")))
        core = table["core"].astype(bool) if "core" in table.columns else pd.Series(True, index=table.index)
        if "clone_programme_assigned" in table.columns and not core.all():
            table["_gallery_programme_assigned"] = _ordered(table["clone_programme_assigned"], "natural").values
            post = uns["programmes"]["params"].get("min_posterior")
            figs.append(GalleryFigure("clone_map_programmes_assigned", "Clone map: programme of every clone",
                                      f"Every profiled clone ({len(table):,}, of which {int(core.sum()):,} define the "
                                      "programmes) coloured by the programme it most likely belongs to (posterior >= "
                                      f"{post}), " + panels.format("programme") + ".",
                                      lambda: draw_clone_map(adata, "_gallery_programme_assigned",
                                                             title="clones by most likely programme")))
    figs.append(GalleryFigure("clone_map_clone_size", "Clone map: clone size",
                              "The clone map coloured by the number of cells of each clone.",
                              lambda: draw_clone_map(adata, "n_cells", title=f"{len(table):,} clones",
                                                     label="cells in the clone")))
    figs.append(GalleryFigure("clone_map_reliability", "Clone map: profile reliability",
                              "The clone map coloured by the reliability of each clone's profile (0 to 1; it grows "
                              "with the number of cells sampled from the clone).",
                              lambda: draw_clone_map(adata, "reliability", title=f"{len(table):,} clones",
                                                     label="profile reliability")))
    if "donor" in table.columns and table["donor"].nunique() >= 2:
        table["_gallery_donor"] = _ordered(table["donor"], "natural").values
        figs.append(GalleryFigure("clone_map_donor", "Clone map: donor",
                                  "The clone map coloured by donor, " + panels.format("donor") + ".",
                                  lambda: draw_clone_map(adata, "_gallery_donor", title="clones by donor")))
    used = set(table.columns)
    for label in labels:
        try:
            values = tf.tl.clone_labels(adata, label).reindex(table.index.astype(str))
        except Exception as e:                    # noqa: BLE001 - one label must not stop the other figures
            notes.append({"name": f"clone_map_{_safe(label)}", "error": f"{type(e).__name__}: {e}"})
            continue
        categorical = _is_categorical(values)
        n_lev = int(values.dropna().astype(str).nunique())
        if n_lev < 2:
            notes.append({"name": f"clone_map_{_safe(label)}",
                          "error": f"every profiled clone has the same `{label}` (or none)"})
            continue
        column = label if label not in used else f"{label} of the clone"
        used.add(column)
        # .values keeps the categorical dtype, so the panels follow the order of the levels
        table[column] = (_ordered(values, _order_for(label, adata.obs[label]), adata.obs[label]) if categorical
                         else pd.to_numeric(values, errors="coerce")).values
        shows = ((f"The clone map coloured by `{label}`, the level held by at least half of each clone's cells, "
                  + panels.format("level") + " (clones without such a majority appear only in grey).")
                 if categorical else f"The clone map coloured by the mean `{label}` of each clone's cells.")
        figs.append(GalleryFigure(f"clone_map_{_safe(label)}", f"Clone map: {label}", shows,
                                  lambda c=column, lab=label, cat=categorical:
                                  draw_clone_map(adata, c, title=f"clones by {lab}", label=None if cat else lab)))

    # 4. programmes: stability and composition
    if n_prog >= 2:
        prog = uns["programmes"]
        figs.append(GalleryFigure("stability", "Programme stability",
                                  "Bootstrap stability of each programme (mean Jaccard overlap with the best-matching "
                                  f"group over {prog['params'].get('n_boot')} bootstrap replicates), with the "
                                  f"{prog['params'].get('stability_threshold')} (stable) and 0.6 guides.",
                                  lambda: tf.pl.stability(adata)))
        accepted = set(inspect.signature(tf.pl.programme_composition).parameters)
        weights = [w for w in COMPOSITION_WEIGHTS if set(w[1]) <= accepted]
        for unit, kwargs, what in COMPOSITION_WEIGHTS:
            if cell_labels and (unit, kwargs, what) not in weights:
                notes.append({"name": f"composition_*_{unit}", "error": "this threadfin's pl.programme_composition "
                              "has no " + ", ".join(f"{k}=" for k in kwargs) + " option"})
        for label in cell_labels:
            for unit, kwargs, what in weights:
                def composition(lab=label, kw=kwargs, u=unit):
                    fig = _fig(tf.pl.programme_composition(adata, lab, **kw))
                    fig.axes[0].set_title(f"{lab} ({u})", color=INK)
                    return fig
                figs.append(GalleryFigure(f"composition_{_safe(label)}_{unit}",
                                          f"Programme composition: {label} ({unit})",
                                          f"For each programme (one bar), {what.format(label=label)} (levels beyond "
                                          "the seventh pooled as 'other').", composition))

    # 5. clone-level tests
    if result.profile_tests:
        sized = any(_given(r.get("size_bins")) for r in result.profile_tests.values())   # threadfin >= 4.6.0
        figs.append(GalleryFigure("label_effects", "Label effects",
                                  "For each tested label, the share of the variance between clone profiles that it "
                                  "explains (blue) and the same with the label shuffled among clones of the same donor "
                                  + ("and of similar size " if sized else "") + "(grey), with the permutation P value.",
                                  lambda: draw_label_effects(result.profile_tests)))
    for label, tab in result.tests.items():
        name = f"association_{_safe(label)}"
        if "level" in tab.columns:
            drawn, empty = drawable_levels(tab)
            if not drawn:
                notes.append({"name": name, "error": f"every odds ratio of `{label}` is 0 or infinite, so none can be "
                                                     "placed on a log axis (" + ", ".join(empty) + ")"})
                continue
            kind = ("Mantel-Haenszel odds ratio within donor x clone-size bins" if _given(tab.attrs.get("size_bins"))
                    else "Mantel-Haenszel odds ratio across donors") + " with its 95 % CI, one panel per level"
            if empty:
                kind += ("; levels whose odds ratios are all 0 or infinite cannot be placed on a log axis and are left "
                         "out (" + ", ".join(empty) + ")")
        else:
            kind = "rank effect (programme against the other clones)"
        figs.append(GalleryFigure(name, f"Programme association: {label}",
                                  f"Clone-level association of each programme with `{label}`: {kind}. Filled markers: "
                                  "FDR < 0.05.", lambda t=tab, lab=label: draw_association(t, lab)))
    if result.memory is not None:
        figs.append(GalleryFigure(f"memory_{_safe(time_key)}", f"Clonal memory over {time_key}",
                                  f"How much a clone's profile changes between levels of `{time_key}`, against "
                                  + ("other clones of the same donor seen at the same two levels"
                                     if result.memory.get("reference", "all") == "persistent"
                                     else "random clones of the same donor")
                                  + "; the title gives the memory index with its 95 % CI.",
                                  lambda: tf.pl.memory(result.memory)))

    # 6. genes: heritability and marker signatures (computed once from the kept h5ad, then cached)
    stamp = _stamp(h5)
    gene_sets = {}
    if adata.n_vars:
        try:
            gene_sets = case_study_gene_sets(ds)
        except Exception as e:                    # noqa: BLE001 - the signatures are optional
            print(f"[gallery] {ds}: no marker sets ({type(e).__name__}: {e})", flush=True)
    herit, sig = None, None
    if adata.n_vars:
        try:
            herit, sig = expression_inputs(adata, h5, cache, stamp, gene_sets)
        except Exception as e:                    # noqa: BLE001 - e.g. memory; the other figures still apply
            notes.append({"name": "heritability, signatures", "error": f"{type(e).__name__}: {e}"})
    if herit is not None and len(herit):
        figs.append(GalleryFigure("heritability", "Gene heritability",
                                  "Clonal ICC of every gene expressed in at least 5 % of the cells of clones with two "
                                  "or more cells (receptor genes excluded), by rank; the 12 highest are named "
                                  "(`tl.gene_heritability` with its defaults).", lambda: draw_heritability(herit)))

    # 7. cells on the cell embedding
    basis = settings.get("basis") or "X_threadfin"
    xy = None
    if basis in adata.obsm and (n_prog >= 2 or cell_labels or number_labels):
        try:
            xy, axis_name = cell_embedding(adata, basis, cache, stamp)
        except Exception as e:                    # noqa: BLE001 - the other figures still apply
            notes.append({"name": "cells", "error": f"{type(e).__name__}: {e}"})
    if xy is not None:
        adata.obsm["X_gallery"] = xy
        where = (f"the run's 2-D embedding (`{basis}`)" if axis_name != "UMAP"
                 else f"a UMAP of the embedding the run used (`{basis}`)")
        if n_prog >= 2:
            figs.append(GalleryFigure("cells_programmes", "Cells: programmes",
                                      f"Cells on {where}: the cells of each programme's clones, one panel per "
                                      "programme, on the grey of all cells.",
                                      lambda: draw_cells(adata, "clone_programme", "X_gallery", axis_name,
                                                         "cells by programme", na_label="no programme")))
        for label in cell_labels:
            figs.append(GalleryFigure(f"cells_{_safe(label)}", f"Cells: {label}",
                                      f"Cells on {where}: the cells of each level of `{label}`, one panel per level, "
                                      "on the grey of all cells.",
                                      lambda lab=label: draw_cells(adata, lab, "X_gallery", axis_name,
                                                                   f"cells by {lab}")))
        for label in number_labels:
            figs.append(GalleryFigure(f"cells_{_safe(label)}", f"Cells: {label}",
                                      f"Cells on {where}, coloured by `{label}`.",
                                      lambda lab=label: draw_cells(adata, lab, "X_gallery", axis_name, lab)))

    # 8. marker signatures per programme
    if n_prog >= 2 and sig is not None:
        present = {k: [g for g in v if g in set(sig.var_names)] for k, v in gene_sets.items()}
        present = {k: v for k, v in present.items() if len(v) >= 3}
        if len(present) >= 2:
            z = n_prog >= 3                      # across two programmes a z-score is +1 or -1 by construction

            def signatures():
                fig = _fig(tf.pl.signature_heatmap(sig, present, groupby="clone_programme", layer=None, z_score=z))
                ax = fig.axes[0]
                ax.set_title("case-study marker sets", color=INK)
                ax.set_ylabel("programme")
                _contrast_texts(ax)
                return fig
            figs.append(GalleryFigure("signatures", "Marker signatures by programme",
                                      "Mean log-normalised expression of the case-study marker sets ("
                                      + ", ".join(present) + ") in the cells of each programme"
                                      + (", z-scored across programmes." if z else
                                         " (not z-scored: across two programmes a z-score is only +1 or -1)."),
                                      signatures))
    return figs, notes


# --------------------------------------------------------------------------- one dataset


def _given(bins) -> bool:
    """Whether a recorded list of clone-size bins is set (a list, or an array after an h5ad round trip)."""
    return bins is not None and np.size(bins) > 0


def _fmt_call(settings: dict, rec: dict, batch_key: str | None) -> str:
    parts = ["adata"] + (["bcr=contigs"] if settings.get("receptor") or not str(settings.get("basis")).endswith(
        "published") else [])
    strata = settings.get("strata_key")                                    # threadfin >= 4.6.0
    for key, value in (("donor_key", settings.get("donor_key")), ("sample_key", settings.get("sample_key")),
                       ("strata_key", strata if strata != settings.get("sample_key") else None),
                       ("batch_key", batch_key), ("state_key", settings.get("state_key")),
                       ("time_key", settings.get("time_key"))):
        if value:
            parts.append(f'{key}="{value}"')
    if settings.get("basis") and settings["basis"] != "X_threadfin":
        parts.append(f'basis="{settings["basis"]}"')
    if settings.get("composition_weight") not in (None, "cells"):        # threadfin >= 4.6.0
        parts.append(f'composition_weight="{settings["composition_weight"]}"')
    if rec.get("test"):
        parts.append("test=[" + ", ".join(f'"{t}"' for t in rec["test"]) + "]")
    return "tf.run(" + ", ".join(parts) + ", n_jobs=8)"


def write_index(ds: str, out: Path, rec: dict, settings: dict, done: list[dict], failed: list[dict],
                batch_key: str | None = None) -> None:
    name = NAMES.get(ds, ds)
    lines = [f"# {name} (`{ds}`)", ""]
    if rec.get("cells") is not None:
        icc = rec.get("icc")
        lines.append(f"threadfin {rec.get('threadfin')} · {rec['cells']:,} cells · {rec.get('clones_ge2') or 0:,} clones "
                     f"with two or more cells · clonal ICC {icc:.3f} (shuffled {rec.get('icc_shuffled', float('nan')):.3f}, "
                     f"P = {rec.get('p_value', float('nan')):.2g}) · {rec.get('programmes')} "
                     f"programme{'s' if rec.get('programmes') != 1 else ''}")
        lines.append("")
    lines += [f"The run: `{_fmt_call(settings, rec, batch_key)}` ([`run_dataset.py`](../../run_dataset.py)); the figures are "
              "drawn from the kept result by [`make_gallery.py`](../../make_gallery.py).", ""]
    lines += ["| figure | what it shows |", "|---|---|"]
    for f in done:
        lines.append(f"| [{f['title']}](#{_anchor(f['title'])}) | {f['shows']} |")
    lines.append("")
    if rec.get("skipped"):
        lines += ["Not run by `tf.run` on these data:", ""]
        lines += [f"* {what}: {why}" for what, why in rec["skipped"].items()]
        lines.append("")
    if failed:
        lines += ["Not drawn:", ""]
        lines += [f"* `{f['name']}`: {f['error']}" for f in failed]
        lines.append("")
    for f in done:
        lines += [f"## {f['title']}", "", f"![{f['title']}]({f['file']})", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _anchor(title: str) -> str:
    """GitHub's heading anchor."""
    a = title.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def render_dataset(ds: str, results: Path, gallery: Path, zopfli: bool = False) -> dict:
    tf.pl.set_style()
    rec = json.loads((results / f"{ds}.json").read_text())
    out = gallery / ds
    out.mkdir(parents=True, exist_ok=True)
    h5 = results / f"{ds}.h5ad"
    settings = run_settings(ds, rec)
    done, failed = [], []
    batch_key = None
    t0 = time.time()
    if not h5.exists():                          # an older run without its AnnData: the saved overview only
        src = results / f"{ds}_overview.png"
        if src.exists():
            data = optimise_png(src.read_bytes())
            (out / "overview.png").write_bytes(data)
            done.append({"name": "overview", "file": "overview.png", "title": "Overview (`result.plot()`)",
                         "shows": "The figure that `result.plot()` saved during the run.", "bytes": len(data)})
        failed.append({"name": "all other figures", "error": f"{h5.name} was not kept by this run"})
    else:
        adata = read_result(h5)
        batch_key = (adata.uns.get("threadfin", {}).get("prepare_embedding") or {}).get("batch_key")
        result = threadfin_result(adata, rec, settings)
        cache = results / f"{ds}_gallery_cache"
        specs, notes = plan(ds, adata, result, settings, rec, h5, cache)
        failed.extend(notes)
        for fig_spec in specs:
            t = time.time()
            try:
                fig = _fig(fig_spec.draw())
                info = save_png(fig, out / f"{fig_spec.name}.png")
                plt.close("all")
                done.append({"name": fig_spec.name, "file": f"{fig_spec.name}.png", "title": fig_spec.title,
                             "shows": fig_spec.shows, **info, "seconds": round(time.time() - t, 1)})
                print(f"[gallery] {ds}: {fig_spec.name}.png {info['bytes'] / 1e3:.0f} kB at {info['dpi']} dpi "
                      f"({time.time() - t:.1f} s)", flush=True)
            except Exception as e:                # noqa: BLE001 - one failed figure must not stop the gallery
                plt.close("all")
                failed.append({"name": fig_spec.name, "error": f"{type(e).__name__}: {e}"})
                print(f"[gallery] {ds}: {fig_spec.name} FAILED: {type(e).__name__}: {e}", flush=True)
    for stale in out.glob("*.png"):             # images of figures that no longer apply
        if stale.name not in {d["file"] for d in done}:
            stale.unlink()
    if zopfli and (exe := zopflipng_path()):
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max(1, cpus)) as pool:
            for path, _, after in pool.map(_zopfli_one, [(exe, str(out / d["file"])) for d in done]):
                next(d for d in done if d["file"] == Path(path).name)["bytes"] = after
    if (out / "overview.png").exists():
        thumbnail(out / "overview.png", gallery / "thumbnails" / f"{ds}.png")
    manifest = {"dataset": ds, "name": NAMES.get(ds, ds), "threadfin_run": rec.get("threadfin"),
                "threadfin_drawn": tf.__version__, "record": {k: rec.get(k) for k in (
                    "cells", "clones_ge2", "icc", "icc_shuffled", "p_value", "programmes", "test", "skipped")},
                "settings": settings, "figures": done, "not_drawn": failed, "seconds": round(time.time() - t0, 1)}
    (out / "figures.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    write_index(ds, out, rec, settings, done, failed, batch_key)
    print(f"[gallery] {ds}: {len(done)} figures, {sum(d['bytes'] for d in done) / 1e6:.1f} MB, "
          f"{len(failed)} not drawn, {time.time() - t0:.0f} s", flush=True)
    return manifest


# --------------------------------------------------------------------------- the gallery page


def write_page(gallery: Path) -> None:
    rows, n_img, n_bytes, versions = [], 0, 0, set()
    for f in sorted(gallery.glob("*/figures.json")):
        m = json.loads(f.read_text())
        ds, r = m["dataset"], m["record"]
        n_img += len(m["figures"])
        n_bytes += sum(d["bytes"] for d in m["figures"])
        versions.add(str(m.get("threadfin_run")))
        thumb = f"thumbnails/{ds}.png"
        cell = f'<a href="{ds}/README.md"><img src="{thumb}" width="240" alt="{ds} overview"></a>' \
            if (gallery / thumb).exists() else ""
        icc = (f"{r['icc']:.3f} ({r['icc_shuffled']:.3f})" if r.get("icc") is not None else "—")
        rows.append(f"| [{m['name']}]({ds}/README.md) | {cell} | {len(m['figures'])} | "
                    f"{r.get('cells') or 0:,} | {r.get('clones_ge2') or 0:,} | {icc} | {r.get('programmes', '—')} |")
    lines = ["# Figures of the tutorial-path runs", "",
             "Every figure that Threadfin draws for each public dataset of the [tutorial path](../README.md): the "
             "overview that `result.plot()` saves and every other `threadfin.pl` figure that applies, drawn by "
             "[`make_gallery.py`](../make_gallery.py) from the result that [`run_dataset.py`](../run_dataset.py) "
             f"kept (threadfin {', '.join(sorted(versions))}). Each dataset's page says what every image shows.", "",
             "| dataset | overview | figures | cells | clones ≥ 2 cells | clonal ICC (shuffled) | programmes |",
             "|---|---|---|---|---|---|---|", *rows, "",
             f"{n_img} images, {n_bytes / 1e6:.1f} MB.", ""]
    (gallery / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[gallery] page: {len(rows)} datasets, {n_img} images, {n_bytes / 1e6:.1f} MB", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", type=Path, help="run_dataset.py output folder (<dataset>.json and <dataset>.h5ad)")
    p.add_argument("gallery", type=Path, help="folder for the figures (one subfolder per dataset) and README.md")
    p.add_argument("--datasets", nargs="+", help="datasets to draw (default: every <dataset>.json in results)")
    p.add_argument("--page-only", action="store_true", help="only rewrite the gallery page")
    p.add_argument("--no-page", action="store_true", help="do not rewrite the gallery page")
    p.add_argument("--zopfli", action="store_true", help="also run zopflipng on every PNG (slow; ~10 %% smaller)")
    a = p.parse_args(argv)
    a.gallery.mkdir(parents=True, exist_ok=True)
    if not a.page_only:
        names = a.datasets or sorted(f.stem for f in a.results.glob("*.json") if f.stem in NAMES)
        for ds in names:
            render_dataset(ds, a.results, a.gallery, zopfli=a.zopfli)
    if not a.no_page:
        write_page(a.gallery)


if __name__ == "__main__":
    main()
