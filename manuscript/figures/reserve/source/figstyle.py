"""Shared figure style and millimetre page layout for the paper figures.

Extracted from the old ``legacy_panels`` module (now archived under
``_archive_pre_dandelion/``) so the current, Dandelion-based figure scripts keep working.
Only the style constants and the page helpers live here; panel-drawing code belongs in the
individual figure scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parents[2]
EXPORT_DPI = 1200

# ------------------------------------------------------------------ style
MM = 1 / 25.4
A4_W, A4_H = 210 * MM, 297 * MM
AREA_W = 183.0                         # mm, Nature double column
LEFT = (210 - AREA_W) / 2              # mm
TOP = 18.0                             # mm, space for the figure title
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY, LIGHT, INK, INK2, MUTED = "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e", "#898781"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.0, "axes.titlesize": 6.5, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "lines.linewidth": 1.0, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": EXPORT_DPI,
})


def new_page(number: str, title: str, subtitle: str = "", height: float = 297.0):
    """A figure page 183 mm wide and ``height`` mm tall, laid out in millimetres from its top-left."""
    fig = plt.figure(figsize=(A4_W, height * MM))
    fig.page_height = height
    fig.text(LEFT * MM / A4_W, 1 - 9 * MM / (height * MM), f"{number} | {title}", fontsize=8.5,
             fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(LEFT * MM / A4_W, 1 - 14 * MM / (height * MM), subtitle, fontsize=6.5, color=INK2, va="top")
    return fig


def panel(fig, x, y, w, h, letter=None, letter_dx=-4.5):
    """Axes at (x, y) mm from the top-left of the 183-mm figure area, size (w, h) mm."""
    page = getattr(fig, "page_height", 297.0)
    left = (LEFT + x) * MM / A4_W
    bottom = 1 - (TOP + y + h) * MM / (page * MM)
    ax = fig.add_axes([left, bottom, w * MM / A4_W, h * MM / (page * MM)])
    if letter:
        fig.text((LEFT + x + letter_dx) * MM / A4_W, 1 - (TOP + y - 2.5) * MM / (page * MM), letter,
                 fontsize=8.5, fontweight="bold", color=INK, va="bottom", ha="left")
    return ax


def placeholder(ax, text, fontsize=5.8):
    """Dashed box saying what a hand-drawn panel will show (text wrapped to the box width)."""
    import textwrap

    ax.set_axis_off()
    box = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.02",
                         transform=ax.transAxes, facecolor="#f6f6f3", edgecolor=MUTED, linewidth=0.6,
                         linestyle=(0, (4, 3)))
    ax.add_patch(box)
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72 * 0.9
    chars = max(int(width_pt / (fontsize * 0.5)), 12)
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, chars) or [""])
    ax.text(0.5, 0.5, "PLACEHOLDER\n" + "\n".join(lines), transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=INK2, linespacing=1.4)


def save(fig, name, folder=None, *, dpi=EXPORT_DPI):
    out = Path(folder or HERE)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=dpi)
    fig.savefig(out / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print("wrote", out / name)


def spread(values, gap):
    """Shift label positions apart so that neighbours are at least ``gap`` apart (keeps order)."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    out = v[order].copy()
    for i in range(1, out.size):
        out[i] = max(out[i], out[i - 1] + gap)
    shift = (out - v[order]).mean()          # re-centre the block on the data
    out -= shift
    back = np.empty_like(out)
    back[order] = out
    return back
