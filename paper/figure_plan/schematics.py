#!/usr/bin/env python
"""Line-art schematics for the manuscript figures.

Everything here is drawn with matplotlib primitives in a local coordinate
system, so the panels scale with the page and stay vector in the PDF. The
style is deliberately plain: thin strokes, one accent colour per panel, no
shading and no clip art.

Each ``draw_*`` function takes an axes and fills it; the ``icon`` helpers draw
one object at a position and are meant to be composed.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY, LIGHT, INK, INK2, MUTED = "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e", "#898781"
PAPER = "#f6f6f3"
LW = 0.6


# --------------------------------------------------------------------------- canvas


def canvas(ax, w=100.0, h=30.0):
    """Blank drawing area with ``w`` x ``h`` local units and no decoration."""
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    ax.set_aspect("auto")
    return ax


def label(ax, x, y, text, size=5.2, color=INK, ha="center", va="top", weight=None, style=None, wrap=None):
    """Text at (x, y) in local units; ``wrap`` is the width in local units to wrap long text to."""
    if wrap is not None:
        import textwrap

        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
        chars = max(int(wrap / span * width_pt / (size * 0.5)), 16)
        text = "\n".join(textwrap.wrap(text, chars))
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, style=style,
            linespacing=1.35, zorder=5)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=LW, head=2.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_width={head / 2},head_length={head}",
                                 mutation_scale=1, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=2))


def card(ax, x, y, w, h, color=MUTED, face=PAPER, lw=LW, dashed=False, radius=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=color, lw=lw, zorder=0,
                                linestyle=(0, (3, 2)) if dashed else "-"))


# --------------------------------------------------------------------------- icons


def mouse(ax, x, y, s=1.0, color=INK2):
    """Side view of a mouse, facing right; (x, y) is the centre of the body."""
    ax.add_patch(Ellipse((x, y), 7.0 * s, 4.2 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
    ax.add_patch(Circle((x + 4.0 * s, y + 0.8 * s), 1.7 * s, facecolor="white", edgecolor=color, lw=LW, zorder=4))
    ax.add_patch(Circle((x + 3.4 * s, y + 2.3 * s), 0.9 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
    ax.plot([x + 5.6 * s], [y + 0.6 * s], marker=".", ms=1.2 * s, color=color, zorder=5)
    t = np.linspace(0, np.pi, 30)
    ax.plot(x - 3.4 * s - 2.6 * s * np.sin(t / 2), y - 0.6 * s + 1.6 * s * np.sin(t), color=color, lw=LW, zorder=2)
    for dx in (-1.6, 1.4):                                  # legs
        ax.plot([x + dx * s, x + dx * s], [y - 2.0 * s, y - 3.2 * s], color=color, lw=LW, zorder=2)


def person(ax, x, y, s=1.0, color=INK2):
    """Head and shoulders, for the human datasets."""
    ax.add_patch(Circle((x, y + 2.2 * s), 1.5 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
    t = np.linspace(np.pi, 2 * np.pi, 40)
    ax.add_patch(Polygon(np.column_stack([x + 3.0 * s * np.cos(t), y - 1.4 * s - 3.0 * s * np.sin(t)]),
                         closed=True, facecolor="white", edgecolor=color, lw=LW, zorder=2))


def syringe(ax, x, y, s=1.0, color=INK2):
    """Syringe pointing right; (x, y) is the needle tip."""
    ax.add_patch(Rectangle((x - 6.5 * s, y - 0.9 * s), 4.2 * s, 1.8 * s, facecolor="white",
                           edgecolor=color, lw=LW, zorder=3))
    ax.plot([x - 2.3 * s, x], [y, y], color=color, lw=LW, zorder=3)
    ax.plot([x - 7.4 * s, x - 6.5 * s], [y, y], color=color, lw=LW, zorder=3)
    ax.plot([x - 7.4 * s] * 2, [y - 1.3 * s, y + 1.3 * s], color=color, lw=LW, zorder=3)


def organ(ax, x, y, s=1.0, kind="node", color=INK2, fill="white"):
    """A small organ glyph: lymph node, spleen, lung, bone or tonsil."""
    if kind == "node":
        ax.add_patch(Ellipse((x, y), 5.4 * s, 3.8 * s, angle=-20, facecolor=fill, edgecolor=color,
                             lw=LW, zorder=3))
        for a in (-0.9, 0.0, 0.9):                       # follicles
            ax.add_patch(Circle((x + a * 1.3 * s, y + a * 0.4 * s), 0.55 * s, facecolor="none",
                                edgecolor=color, lw=LW * 0.8, zorder=4))
        ax.plot([x - 3.6 * s, x - 2.2 * s], [y - 1.6 * s, y - 0.9 * s], color=color, lw=LW, zorder=2)
    elif kind == "spleen":
        t = np.linspace(0, 2 * np.pi, 80)
        r = 2.1 * s * (1 + 0.28 * np.cos(t) + 0.12 * np.cos(2 * t))
        ax.add_patch(Polygon(np.column_stack([x + 1.25 * r * np.cos(t + 0.6), y + r * np.sin(t + 0.6)]),
                             closed=True, facecolor=fill, edgecolor=color, lw=LW, zorder=3))
    elif kind == "lung":
        for sgn in (-1, 1):
            t = np.linspace(0, 2 * np.pi, 60)
            ax.add_patch(Polygon(np.column_stack([
                x + sgn * (1.4 * s + 1.5 * s * (1 + 0.25 * np.cos(t)) * np.abs(np.cos(t / 2)) * 0.9),
                y + 2.6 * s * np.sin(t)]), closed=True, facecolor=fill, edgecolor=color, lw=LW, zorder=3))
        ax.plot([x, x], [y + 2.4 * s, y + 4.0 * s], color=color, lw=LW, zorder=2)
    elif kind == "bone":
        ax.add_patch(Rectangle((x - 3.2 * s, y - 0.9 * s), 6.4 * s, 1.8 * s, facecolor=fill,
                               edgecolor=color, lw=LW, zorder=3))
        for sgn in (-1, 1):
            for dy in (-1.0, 1.0):
                ax.add_patch(Circle((x + sgn * 3.3 * s, y + dy * 1.0 * s), 1.1 * s, facecolor=fill,
                                    edgecolor=color, lw=LW, zorder=3))
    elif kind == "blood":
        ax.add_patch(Polygon([[x, y + 2.8 * s], [x - 2.0 * s, y - 0.6 * s], [x + 2.0 * s, y - 0.6 * s]],
                             closed=True, facecolor=fill, edgecolor=color, lw=LW, zorder=3))
        ax.add_patch(Circle((x, y - 0.6 * s), 2.0 * s, facecolor=fill, edgecolor=color, lw=LW, zorder=3))


def parasite(ax, x, y, s=1.0, color=INK2):
    """Infected red cell, for the Plasmodium time course."""
    ax.add_patch(Circle((x, y), 2.0 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
    t = np.linspace(0.4, 2 * np.pi + 0.4, 40)
    ax.plot(x - 0.3 * s + 1.0 * s * np.cos(t), y + 0.2 * s + 0.75 * s * np.sin(t), color=color, lw=LW, zorder=4)


def virus(ax, x, y, s=1.0, color=INK2):
    ax.add_patch(Circle((x, y), 1.6 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([x + 1.6 * s * np.cos(a), x + 2.5 * s * np.cos(a)],
                [y + 1.6 * s * np.sin(a), y + 2.5 * s * np.sin(a)], color=color, lw=LW * 0.8, zorder=3)


def gate(ax, x, y, s=1.0, color=INK2, accent=BLUE, seed=0, frac=0.3):
    """A sort gate: a small cloud of cells with a box around the fraction taken."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1, (70, 2))
    ax.add_patch(Rectangle((x - 3.0 * s, y - 3.0 * s), 6.0 * s, 6.0 * s, facecolor="white",
                           edgecolor=color, lw=LW, zorder=2))
    inside = p[:, 0] > np.quantile(p[:, 0], 1 - frac)
    ax.scatter(x + 0.9 * s * p[:, 0], y + 0.9 * s * p[:, 1], s=0.6, color=GREY, linewidths=0, zorder=3)
    ax.scatter(x + 0.9 * s * p[inside, 0], y + 0.9 * s * p[inside, 1], s=0.9, color=accent,
               linewidths=0, zorder=4)
    ax.add_patch(Rectangle((x + 0.5 * s, y - 2.2 * s), 2.3 * s, 4.4 * s, facecolor="none",
                           edgecolor=accent, lw=LW, zorder=5))


def droplets(ax, x, y, s=1.0, color=INK2):
    """Single-cell capture: cells in droplets."""
    for i, dx in enumerate((-2.6, 0.0, 2.6)):
        ax.add_patch(Circle((x + dx * s, y), 1.25 * s, facecolor="white", edgecolor=color, lw=LW, zorder=3))
        ax.add_patch(Circle((x + dx * s, y), 0.45 * s, facecolor=[BLUE, ORANGE, AQUA][i], edgecolor="none",
                            zorder=4))


def reads(ax, x, y, s=1.0, color=INK2, accent=BLUE):
    """Stacked sequencing reads, the top one highlighted as the receptor."""
    for i in range(4):
        w = [5.4, 4.2, 4.8, 3.6][i]
        ax.plot([x - w / 2 * s, x + w / 2 * s], [y + (1.5 - i) * 1.25 * s] * 2,
                color=accent if i == 0 else color, lw=1.1 if i == 0 else LW, solid_capstyle="round", zorder=3)


def cells(ax, x, y, n=12, s=1.0, color=BLUE, spread=2.4, seed=0, size=2.0, edge=False):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, spread * s, (n, 2))
    ax.scatter(x + p[:, 0], y + p[:, 1], s=size, color=color, linewidths=0.25 if edge else 0,
               edgecolors="white" if edge else "none", zorder=4)


def brace(ax, x, y0, y1, color=MUTED, width=1.0, lw=LW):
    """A thin curly brace opening to the right, spanning y0..y1 at x."""
    ym = (y0 + y1) / 2
    for a, b in ((y0, ym), (y1, ym)):
        t = np.linspace(0, 1, 30)
        ax.plot(x + width * np.sin(t * np.pi / 2), a + (b - a) * t, color=color, lw=lw, zorder=2)
    ax.plot([x + width, x + width * 1.6], [ym, ym], color=color, lw=lw, zorder=2)


# --------------------------------------------------------------------------- composed panels


def design_row(ax, steps, y=None, w=100.0, h=30.0, gap=None, title_size=5.0):
    """A left-to-right design strip: each step is ``(draw, caption)``.

    ``draw`` is called as ``draw(ax, x, y)`` at the centre of its slot.
    """
    canvas(ax, w, h)
    y = h * 0.62 if y is None else y
    n = len(steps)
    slot = w / n
    xs = [slot * (i + 0.5) for i in range(n)]
    arm = min(slot * 0.16, 5.0) if gap is None else gap
    for i, (draw, text) in enumerate(steps):
        draw(ax, xs[i], y)
        label(ax, xs[i], y - h * 0.28, text, size=title_size, color=INK2)
        if i < n - 1:
            mid = (xs[i] + xs[i + 1]) / 2
            arrow(ax, mid - arm, y, mid + arm, y)
    return xs, y


def draw_workflow(ax):
    """Figure 1: what the method does to the data, from reads to a clone that can be tested."""
    canvas(ax, 183, 34)
    boxes = [
        (1, 32, "Paired single-cell\nRNA and receptor"),
        (38, 32, "Clones called\nwithin each donor"),
        (75, 32, "Expression embedding\nwith no receptor genes"),
        (112, 32, "Each clone described\nagainst its own context"),
        (149, 33, "Reliability from the\ncells the clone has"),
    ]
    by, bh = 6.5, 9.0                                        # box band
    gy = 23.0                                                # glyph band centre
    for i, (x, w, text) in enumerate(boxes):
        card(ax, x, by, w, bh, color=BLUE if i == 0 else GREY, face="#eef4fc" if i == 0 else "white",
             lw=0.9 if i == 0 else LW)
        label(ax, x + w / 2, by + bh / 2, text, size=5.0, va="center")
        if i:
            arrow(ax, x - 4.4, by + bh / 2, x - 0.9, by + bh / 2)

    reads(ax, 17, gy, s=1.0)
    for j, c in enumerate((BLUE, ORANGE, AQUA)):
        cells(ax, 46 + 9 * j, gy, n=7, spread=1.3, color=c, seed=j, size=1.8)
    cells(ax, 91, gy, n=55, spread=3.0, color=GREY, seed=7, size=1.6)
    cells(ax, 128, gy, n=16, spread=2.6, color=BLUE, seed=3, size=2.0)
    cells(ax, 128, gy, n=55, spread=4.4, color=LIGHT, seed=11, size=1.2)
    t = np.linspace(0, 1, 60)
    ax.plot(152 + 26 * t, gy - 4.0 + 7.6 * (4 * t) / (1 + 4 * t), color=BLUE, lw=1.0, zorder=3)
    ax.plot([152, 178], [gy - 4.0] * 2, color=GREY, lw=LW, zorder=2)
    label(ax, 165, gy + 5.2, "reliability", size=4.3, color=MUTED, va="bottom")
    label(ax, 165, gy - 5.8, "cells in the clone", size=4.3, color=MUTED)

    label(ax, 91.5, 2.6, "A clone is described by where all of its cells sit, relative to the cells it was "
                         "sampled with, and how much of that description is signal rather than sampling is "
                         "known before any test is run.", size=5.0, color=INK2, wrap=170)


def draw_memory(ax):
    """Figure 1: the clonal memory index, as a picture."""
    canvas(ax, 100, 40)
    rng = np.random.default_rng(1)
    label(ax, 50, 39.5, "Does a clone keep its state when the context changes?", size=5.2, va="top")

    for k, x in enumerate((17, 52)):
        box = Rectangle((x - 11, 17), 22, 14, facecolor="white", edgecolor=GREY, lw=LW, zorder=1)
        ax.add_patch(box)
        label(ax, x, 32.0, ("before", "after")[k], size=5.0, color=INK2, va="bottom")
        for j, c in enumerate((BLUE, ORANGE)):
            base = rng.normal(0, 1, (16, 2))
            off = np.array([[-6.0, 0.0], [6.0, 0.0]])[j] + (np.array([[0.4, 0.3], [-10.6, -0.4]])[j] if k else 0)
            pts = ax.scatter(x + off[0] + 1.6 * base[:, 0], 24 + off[1] + 2.0 * base[:, 1], s=1.8, color=c,
                             linewidths=0, zorder=3)
            pts.set_clip_path(box)
            if k:                                            # faint ghost of where the clone used to be
                g = ax.scatter(x + np.array([-6.0, 6.0])[j] + 1.6 * base[:, 0], 24 + 2.0 * base[:, 1],
                               s=1.6, color=LIGHT, linewidths=0, zorder=2)
                g.set_clip_path(box)
    arrow(ax, 29, 24, 40, 24)
    label(ax, 34.5, 25.0, "another tissue,\nor months later", size=4.4, color=MUTED, va="bottom")

    ax.plot([72, 72], [19, 29], color=GREY, lw=LW)
    for yv, lab, col in ((28.0, "same clone,\nstill alike", BLUE), (20.5, "same clone,\nnow unlike", ORANGE)):
        ax.plot([70.6, 73.4], [yv, yv], color=col, lw=1.3)
        label(ax, 75, yv, lab, size=4.4, color=INK2, ha="left", va="center")
    label(ax, 72, 32.0, "clonal memory", size=5.0, color=INK2, va="bottom")
    label(ax, 50, 12.5, "The index compares a clone with itself across the change, against a random clone of "
                        "the same donor, after subtracting the noise expected from sampling few cells. One "
                        "means the clone is unchanged; zero means knowing it before says nothing about it "
                        "after.", size=4.6, color=INK2, va="top", wrap=96)


def draw_groups_or_continuum(ax):
    """Figure 1: when clones form groups and when they form a continuum."""
    canvas(ax, 100, 40)
    rng = np.random.default_rng(4)
    label(ax, 50, 39.5, "Are there groups of clones, or one continuous spread?", size=5.2, va="top")

    a = np.vstack([rng.normal([-2.3, 0], 0.75, (40, 2)), rng.normal([2.3, 0.1], 0.75, (40, 2))])
    b = rng.normal(0, 1.0, (80, 2)) * [2.0, 0.8]
    for x0, pts, name, note in ((23, a, "groups", "a split that beats one spread"),
                                (70, b, "continuum", "no split beats one spread")):
        box = Rectangle((x0 - 15, 18), 30, 14, facecolor="white", edgecolor=GREY, lw=LW, zorder=1)
        ax.add_patch(box)
        sc = ax.scatter(x0 + 3.4 * pts[:, 0], 25 + 3.4 * pts[:, 1], s=2.2, color=BLUE, linewidths=0, zorder=3)
        sc.set_clip_path(box)
        label(ax, x0, 33.0, name, size=5.2, color=INK, va="bottom", weight="bold")
        label(ax, x0, 16.5, note, size=4.5, color=INK2, wrap=32)
        if name == "groups":
            ax.plot([x0, x0], [19, 31], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=4)
    label(ax, 50, 10.0, "Each point is a clone. The split is kept only when two groups explain the spread "
                        "better than a single cloud of the same shape, so a gradient is never cut into "
                        "categories.", size=4.6, color=INK2, va="top", wrap=96)


def draw_gc_design(ax):
    """Figure 2: model-antigen germinal-centre experiments."""
    canvas(ax, 183, 30)
    y = 19.0
    label(ax, 0, 29.5, "Model antigens: what the cell had been doing is measured, not inferred", size=5.4,
          ha="left", va="top", weight="bold")

    syringe(ax, 12, y, 1.1)
    label(ax, 7, y - 5.0, "NP-OVA or RBD", size=4.6, color=INK2)
    arrow(ax, 14, y, 19, y)
    mouse(ax, 27, y, 1.1)
    arrow(ax, 35, y, 40, y)
    organ(ax, 46, y, 1.2, "node")
    label(ax, 46, y - 5.0, "draining node,\nday 7-21", size=4.6, color=INK2)
    arrow(ax, 53, y, 59, y)

    for i, (x, col, name, frac) in enumerate(((68, BLUE, "divisions\n(reporter)", 0.3),
                                              (83, ORANGE, "dark or\nlight zone", 0.45),
                                              (98, AQUA, "antigen\nbait", 0.25))):
        gate(ax, x, y, 1.1, accent=col, seed=i, frac=frac)
        label(ax, x, y - 4.6, name, size=4.5, color=INK2)
    ax.plot([64, 64, 102, 102], [y - 9.6, y - 10.6, y - 10.6, y - 9.6], color=MUTED, lw=LW)
    label(ax, 83, y - 11.2, "sorted by what the cell had been doing", size=4.8, color=INK2)

    arrow(ax, 105, y, 111, y)
    droplets(ax, 120, y, 1.2)
    label(ax, 120, y - 5.0, "single-cell capture", size=4.6, color=INK2)
    arrow(ax, 129, y, 135, y)
    reads(ax, 144, y, 1.2)
    label(ax, 144, y - 5.0, "transcriptome\n+ receptor", size=4.6, color=INK2)

    card(ax, 155, y - 14.0, 28, 22.5, color=GREY, face="white")
    label(ax, 169, y + 7.0, "three experiments", size=4.8, color=INK, va="top", weight="bold")
    for k, text in enumerate(("NP-OVA, divisions\n373 clones", "RBD vaccine, divisions,\nzones and bait, 1,414",
                              "NP-OVA, sorted zones\nand plasma cells, 113")):
        label(ax, 157, y + 2.6 - 4.9 * k, text, size=4.4, color=INK2, ha="left", va="top")


def draw_infection_design(ax):
    """Figure 3: influenza and Plasmodium infections."""
    canvas(ax, 183, 32)
    label(ax, 0, 31, "Live infection: antigen for weeks, many germinal centres at once", size=5.4,
          ha="left", va="top", weight="bold")

    # influenza, top row
    y1 = 21.0
    mouse(ax, 6, y1, 0.8)
    virus(ax, 16, y1 + 0.5, 0.8, color=ORANGE)
    label(ax, 11, y1 - 4.2, "influenza A,\nintranasal", size=4.5, color=INK2)
    arrow(ax, 20, y1, 26, y1)
    organ(ax, 31, y1, 0.85, "lung")
    organ(ax, 41, y1, 0.9, "node")
    label(ax, 36, y1 - 5.0, "lung and draining node, day 20", size=4.5, color=INK2)
    arrow(ax, 47, y1, 53, y1)
    gate(ax, 59, y1, 0.8, accent=ORANGE, seed=5, frac=0.25)
    label(ax, 59, y1 - 4.4, "haemagglutinin\ntetramers", size=4.5, color=INK2)
    card(ax, 68, y1 - 4.0, 22, 8.0, color=VIOLET, face="white")
    label(ax, 79, y1 + 0.2, "alpha-v integrin\nknockout arm", size=4.5, color=VIOLET, va="center")

    ax.plot([95, 95], [4, 27], color=LIGHT, lw=0.8)

    # Plasmodium, right
    y2 = 21.0
    mouse(ax, 103, y2, 0.8)
    parasite(ax, 112, y2 + 0.5, 0.9, color=AQUA)
    label(ax, 108, y2 - 4.2, "Plasmodium", size=4.5, color=INK2)
    arrow(ax, 116, y2, 121, y2)
    organ(ax, 126, y2, 0.85, "spleen")
    label(ax, 126, y2 - 4.4, "spleen", size=4.5, color=INK2)
    ax.annotate("", xy=(180, y2 - 1), xytext=(134, y2 - 1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=LW, shrinkA=0, shrinkB=0))
    for d in (0, 4, 7, 10, 14, 21, 28, 35, 42):                # the nine real sampling days
        x = 136 + 41 * d / 42
        ax.plot([x, x], [y2 - 2.4, y2 + 0.4], color=MUTED, lw=LW)
        label(ax, x, y2 + 1.2, str(d), size=4.2, color=INK2, va="bottom")
    label(ax, 157, y2 - 4.0, "day after infection\nnine sampling days, five mice each, antimalarial arm",
          size=4.4, color=INK2)


def draw_nongc_design(ax):
    """Figure 4: clones outside a germinal centre."""
    canvas(ax, 183, 30)
    label(ax, 0, 29, "Outside the germinal centre: what a clone is when nothing is selecting it", size=5.4,
          ha="left", va="top", weight="bold")
    y = 19.0

    person(ax, 7, y - 1, 1.0)
    label(ax, 7, y - 6.0, "human donors", size=4.5, color=INK2)
    arrow(ax, 13, y, 19, y)
    organ(ax, 25, y, 0.9, "bone")
    organ(ax, 36, y, 0.9, "blood")
    label(ax, 30, y - 6.0, "bone marrow and blood", size=4.5, color=INK2)
    arrow(ax, 43, y, 49, y)
    for i, (dx, col, name) in enumerate(((0, BLUE, "plasma cells"), (12, AQUA, "memory B cells"),
                                         (24, YELLOW, "antigen-sorted"))):
        gate(ax, 57 + dx, y, 0.8, accent=col, seed=10 + i, frac=0.3)
        label(ax, 57 + dx, y - 4.4, name, size=4.3, color=INK2)
    label(ax, 69, y - 8.0, "sorted by compartment, and by what the antibody binds", size=4.5, color=INK2)

    ax.plot([95, 95], [4, 26], color=LIGHT, lw=0.8)

    mouse(ax, 104, y, 0.8)
    parasite(ax, 113, y + 0.5, 0.9, color=AQUA)
    arrow(ax, 117, y, 123, y)
    organ(ax, 128, y, 0.85, "spleen")
    label(ax, 116, y - 6.0, "Plasmodium, first two weeks", size=4.5, color=INK2)
    card(ax, 138, y - 9.0, 44, 17.0, color=GREY, face="white")
    for i in range(26):
        cx = 142 + 3.0 * (i % 13)
        cy = y + 4.8 - 3.4 * (i // 13)
        n = 1 if i % 4 else 2
        for k in range(n):
            ax.add_patch(Circle((cx + 1.1 * k, cy), 0.6, facecolor=BLUE if n > 1 else GREY,
                                edgecolor="none", zorder=4))
    label(ax, 160, y - 1.2, "most clones hold one or two cells: the regime where a per-clone reliability "
                            "decides what can be said at all", size=4.4, color=INK2, va="top", wrap=42)


def draw_benchmark_logic(ax):
    """Supplementary: why the comparison is possible at all."""
    canvas(ax, 183, 30)
    label(ax, 0, 29, "Every tool is asked the same question", size=5.4, ha="left", va="top", weight="bold")
    y = 18.0
    card(ax, 2, y - 6, 32, 12, color=BLUE, face="#eef4fc")
    label(ax, 18, y, "paired RNA\nand receptor", size=5.0, va="center")
    for i, name in enumerate(("Threadfin", "clone centroid", "cell-cluster mix", "clone2vec",
                              "scirpy modularity", "Dandelion V(D)J")):
        yy = y + 9.0 - 3.6 * i
        arrow(ax, 35, y, 56, yy)
        card(ax, 57, yy - 1.5, 30, 3.0, color=GREY, face="white", radius=0.8)
        label(ax, 72, yy, name, size=4.4, va="center")
        arrow(ax, 88, yy, 104, y)
    card(ax, 105, y - 6, 34, 12, color=GREY, face="white")
    label(ax, 122, y, "one description\nper clone", size=5.0, va="center")
    arrow(ax, 140, y, 146, y)
    card(ax, 147, y - 7, 35, 14, color=ORANGE, face="#fdf0ea")
    label(ax, 164.5, y, "how much of a measured\nclone property it explains\n(same clones, same test)",
          size=4.6, va="center")
