#!/usr/bin/env python
"""Figure 1 (A4): Threadfin concept (a) and how it works (b).

a  The restored concept graphic (``figure1_programmes.draw``) without the
   section titles or the size key: a graphical abstract of the package.
b  Reached by an arrow from the fish. Three colour-coded sections of
   numbered image cards with minimal labels; one set of synthetic cells and
   three clones (A, B, C, each spanning two transcriptional states) runs
   through them.
   Algorithm pipeline   1  the same cells on the scRNA UMAP (coloured by
                           state) and linked by scBCR clone: clones cannot
                           be separated on the cell UMAP;
                        2  overlapping clones, each summarised by one
                           Frechet (kernel) mean;
                        3  clone embedding: each clone sits where its
                           Frechet mean falls; clones with the same state
                           footprint (A, A') are neighbours; cells of the two
                           right arms interleave evenly on the scRNA UMAP.
   Clonal coherence     4  observed clonal ICC against 500 clone-label
   test                    shuffles within each sample.
   Biological           5  GC B and plasma cells form one mass on the scRNA
   inference               UMAP; Threadfin resolves their clones into a
                           continuous structure, a GC hub with two parallel
                           branches (no state names on the branches);
                        6  position along the clone embedding tracks antigen
                           binding and fate mapping (summary of the readouts
                           in Figures 2-4).
   Every element is schematic; Clonotrace's density-surface and
   optimal-transport imagery is not used.

Usage:  python figure1_a4.py [--preview]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

import figure1_overview as base
import figure1_programmes as concept
from figure1_overview import ACCENT, BLUE, GOLD, INK, MUTED, RED, TEAL

HERE = concept.HERE
WORK = HERE / 'reserve/figure1_a4_2026-10-16'
PROVENANCE = HERE / 'reserve/provenance/Figure_1_a4.json'
STEM = 'Figure_1'
PAGE_W, PAGE_H = 210.0, 297.0

# Panel a: the concept graphic in its own design units.
A_X, A_Y = (-2.0, 185.0), (8.5, 178.5)
A_K = 0.89                                             # mm per design unit
A_W, A_H = (A_X[1] - A_X[0]) * A_K, (A_Y[1] - A_Y[0]) * A_K
A_LEFT, A_TOP = (PAGE_W - A_W) / 2, 4.0
A_FONTS = {'head': 9, 'sub': 8.3, 'body': 6.9, 'small': 6.1, 'label': 7.8}
FISH_BELLY = (36.0, 160.0)                             # design units, below the anal fin

# Panel b layout (page millimetres).
B_TOP = A_TOP + A_H + 3.0
B_TITLE = ('Threadfin integrates scRNA-seq and BCR data into clone embeddings '
           'to infer relationships between clones')
S_PIPE = (6.0, B_TOP + 9.5, 198.0, 62.0)               # x, y, w, h of each section box
S_FID = (6.0, S_PIPE[1] + S_PIPE[3] + 4.0, 66.0, 57.0)
S_BIO = (76.0, S_FID[1], 128.0, 57.0)
SECTIONS = [  # box, title, face, title colour
    (S_PIPE, 'Algorithm pipeline', '#EAF1F7', '#2B5D82'),
    (S_FID, 'Clonal coherence test', '#FBF3E1', '#86600F'),
    (S_BIO, 'Biological inference', '#E8F4F0', '#256E60'),
]
CARD_TOP, CARD_PAD = 8.5, 3.0                          # card offset from the section top; bottom pad
LABEL, SECTION_TITLE, PANEL_TITLE = 6.2, 7.8, 9.0

CELL = '#C4CED6'
EMBED_GREY = '#AEBAC4'
LIGHT_CELL = '#D9E0E6'
STATES = [BLUE, GOLD, RED]                             # transcriptional states, as the scRNA cloud in a
CLONES = dict(zip('ABC', base.FAMILY_COLORS))          # receptor-defined clones, as the trees in a
A_PRIME = '#A99BD0'                                    # a second clone with A's footprint
PROGRAMMES = [BLUE, TEAL, RED]                         # clonal programmes 1-3, as the clone map in a
FOOTPRINT = {'A': (0, 1), 'B': (1, 2), 'C': (0, 2)}    # states spanned by each clone

D = base.rna_data()                                    # synthetic cells: x, y, state
STATE = D[:, 2].astype(int)
CARDS = []                                             # (card box, its texts) for the audit


# ----------------------------------------------------------------------------- shared geometry
def place(box, pad=.4, pts=None):
    """Map cell coordinates (default: D) into box (x0, y0, w, h), keeping the aspect ratio."""
    x0, y0, w, h = box
    pts = D if pts is None else pts
    (xa, xb), (ya, yb) = (pts[:, 0].min(), pts[:, 0].max()), (pts[:, 1].min(), pts[:, 1].max())
    s = min((w - 2 * pad) / (xb - xa), (h - 2 * pad) / (yb - ya))
    cx, cy = x0 + w / 2, y0 + h / 2
    return lambda x, y: (cx + (np.asarray(x) - (xa + xb) / 2) * s, cy + (np.asarray(y) - (ya + yb) / 2) * s)


def clone_members():
    """Eight cells per clone, four in each of its two states, away from state borders."""
    rng = np.random.default_rng(31)
    centre = np.array([D[STATE == s, :2].mean(axis=0) for s in range(3)])
    used, members = set(), {}
    for name, states in FOOTPRINT.items():
        idx = []
        for s in states:
            pool = np.flatnonzero(STATE == s)
            pool = pool[np.linalg.norm(D[pool, :2] - centre[s], axis=1) < 7.5]
            pool = np.array([i for i in pool if i not in used])
            pick = rng.choice(pool, 4, replace=False)
            idx.extend(pick.tolist())
            used.update(pick.tolist())
        members[name] = np.array(idx)
    return members


MEMBERS = clone_members()


HUB = np.array([.47, .50])
ARMS = [(180.0, .40), (-38.0, .47), (38.0, .47)]      # programme 1 left, 2 upper right, 3 lower right


def arm_axis(p):
    theta, length = ARMS[p]
    d = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    return d, length


def embedding_layout():
    """Clone-embedding coordinates (unit square, y down): three arms from a hub, as the map in a."""
    rng = np.random.default_rng(12)
    xy, programme, size = [], [], []
    for p in range(3):
        d, length = arm_axis(p)
        normal = np.array([-d[1], d[0]])
        t = np.sqrt(rng.uniform(.02, 1, 38))           # denser towards the arm tip
        spread = (.018 + .062 * t) * rng.normal(size=38)
        xy.append(HUB + np.outer(t * length, d) + np.outer(spread, normal))
        programme += [p] * 38
        size += np.clip(rng.integers(2, 15, 38) - (t < .3) * 4, 2, None).tolist()
    special = {'A': (.17, .465), "A'": (.235, .55), 'C': (.82, .22), 'B': (.79, .79)}
    return np.clip(np.vstack(xy), .03, .97), np.array(programme), np.array(size), special


def arm_tip(p, extra=1.04):
    d, length = arm_axis(p)
    return HUB + d * length * extra


def arm_halo(ax, to, side, p, **style):
    d, length = arm_axis(p)
    centre = to(*(HUB + d * length * .66))
    ax.add_patch(Ellipse(centre, .62 * length * side, .25 * side, angle=ARMS[p][0], **style))


EMBED = embedding_layout()


def gc_pc_data():
    """Synthetic GC B and plasma cells for card 5: one mass on the scRNA UMAP."""
    rng = np.random.default_rng(58)
    theta = rng.uniform(0, 2 * np.pi, 560)
    radius = np.sqrt(rng.uniform(0, 1, len(theta)))
    u, v = radius * np.cos(theta), radius * np.sin(theta)
    x = 17 * u + 2.2 * (v - .1) ** 2
    y = 15.5 * v - 2.4 * np.sin(np.pi * u)
    plasma = .8 * u + .6 * v + rng.normal(0, .09, len(u)) > .42
    return np.column_stack([x, y, plasma])


GC_PC = gc_pc_data()
FORK_SPLIT, FORK_GAP = .34, .25                        # where the branches leave the hub; half their spacing


def fork_centre(x, sign):
    """Centre line of one branch: leaves the GC hub, then runs parallel to the other."""
    s = np.clip((np.asarray(x) - FORK_SPLIT) / .26, 0, 1)
    return .5 + sign * FORK_GAP * s * s * (3 - 2 * s)


FORK_HUB = (.16, .5, .13, .15)                         # GC hub: centre and radii (unit box)
FORK_BAND = '#EDF1F5'                                  # pale underlay that makes the structure read as one


def fork_path(sign, n=400):
    """One branch's centre line from the hub to its tip, with unit normals and arc-length fraction."""
    x = np.linspace(FORK_HUB[0], .98, n)
    path = np.column_stack([x, fork_centre(x, sign)])
    step = np.gradient(path, axis=0)
    tangent = step / np.linalg.norm(step, axis=1, keepdims=True)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return path, np.column_stack([-tangent[:, 1], tangent[:, 0]]), arc / arc[-1]


def fork_layout(n_hub=105, n_branch=150):
    """Idealised clone embedding (unit box, y down): a GC hub and two parallel branches.

    Clones are spread evenly (jittered sunflower in the hub, stratified along each branch);
    progress runs from 0 in the hub to 1 at the branch tips and colours them from GC-like
    (blue) to plasma-like (red)."""
    rng = np.random.default_rng(9)
    cx, cy, rx, ry = FORK_HUB
    i = np.arange(n_hub) + .5
    r = np.clip(np.sqrt(i / n_hub) + rng.normal(0, .03, n_hub), 0, 1)
    a = i * np.pi * (3 - np.sqrt(5)) + rng.normal(0, .25, n_hub)
    xy = [np.column_stack([cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)])]
    progress = [.1 * (xy[0][:, 0] - cx + rx) / (2 * rx)]
    for sign in (-1, 1):
        path, normal, frac = fork_path(sign)
        s = .12 + .88 * (np.arange(n_branch) + rng.uniform(0, 1, n_branch)) / n_branch   # past the hub's core
        k = np.searchsorted(frac, s).clip(0, len(frac) - 1)
        xy.append(path[k] + normal[k] * rng.normal(0, .03, n_branch)[:, None])
        progress.append(.1 + .9 * (s - .12) / .88)
    xy, progress = np.vstack(xy), np.concatenate(progress)
    order = rng.permutation(len(xy))
    return xy[order], np.clip(progress[order], 0, 1), rng.integers(1, 8, len(xy))[order]


FORK = fork_layout()


def territory(pts, bandwidth=5.2, grid=180):
    """Smooth outline of a clone's cells: one Gaussian-density contour enclosing them all."""
    import contourpy
    lo, hi = pts.min(axis=0) - 3 * bandwidth, pts.max(axis=0) + 3 * bandwidth
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
    dens = lambda x, y: np.exp(-.5 * ((x[..., None] - pts[:, 0]) ** 2 + (y[..., None] - pts[:, 1]) ** 2)
                               / bandwidth ** 2).sum(axis=-1)
    level = .6 * dens(pts[:, 0][:, None], pts[:, 1][:, None]).min()
    lines = contourpy.contour_generator(gx, gy, dens(gx, gy)).lines(level)
    return max(lines, key=lambda line: np.ptp(line[:, 0]) * np.ptp(line[:, 1]))


# ----------------------------------------------------------------------------- drawing helpers
def T(ax, x, y, s, size=LABEL, color=MUTED, **kwargs):
    kwargs.setdefault('va', 'center')
    return ax.text(x, y, s, fontsize=size, color=color, **kwargs)


def arrow(ax, start, end, color=ACCENT, width=1.0, scale=8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, color=color,
                                 linewidth=width, shrinkA=0, shrinkB=0, zorder=6))


def cells(ax, to, colours=CELL, size=1.3, zorder=3, alpha=1.0):
    X, Y = to(D[:, 0], D[:, 1])
    ax.scatter(X, Y, s=size, c=colours, linewidths=0, zorder=zorder, alpha=alpha)
    return X, Y


def clone_links(ax, X, Y, width=.6, dot=6):
    for name, idx in MEMBERS.items():
        pts = np.column_stack([X[idx], Y[idx]])
        tree = minimum_spanning_tree(cdist(pts, pts)).tocoo()
        for i, j in zip(tree.row, tree.col):
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], color=CLONES[name], linewidth=width, zorder=4,
                    solid_capstyle='round')
        ax.scatter(*pts.T, s=dot, c=CLONES[name], edgecolors='white', linewidths=.3, zorder=5)


def footprint(ax, centre, width, states, color):
    """Mini cell map: the cells of the given states coloured, the rest pale."""
    (xa, xb), (ya, yb) = (D[:, 0].min(), D[:, 0].max()), (D[:, 1].min(), D[:, 1].max())
    s = width / (xb - xa)
    keep = np.arange(0, len(D), 2)
    X = centre[0] + (D[keep, 0] - (xa + xb) / 2) * s
    Y = centre[1] + (D[keep, 1] - (ya + yb) / 2) * s
    lit = np.isin(STATE[keep], states)
    ax.scatter(X[~lit], Y[~lit], s=.75, c=LIGHT_CELL, linewidths=0, zorder=4)
    ax.scatter(X[lit], Y[lit], s=.95, c=color, linewidths=0, zorder=5)
    return width, (yb - ya) * s


def star(ax, x, y, color, size=76):
    ax.scatter([x], [y], s=size, marker='*', c=color, edgecolors='white', linewidths=.45, zorder=8)


def embed_map(box_x, box_y, side):
    return lambda u, v: (box_x + np.asarray(u) * side, box_y + np.asarray(v) * side)


def embedding_dots(ax, to, colours=EMBED_GREY, alpha=1.0, zorder=4):
    xy, programme, size, _ = EMBED
    X, Y = to(xy[:, 0], xy[:, 1])
    ax.scatter(X, Y, s=2.0 + 1.35 * size, c=colours, edgecolors='white', linewidths=.28, alpha=alpha,
               zorder=zorder)
    return X, Y


# ----------------------------------------------------------------------------- cards
def card_cells_and_clones(ax, box):
    """1: the same cells, coloured by transcriptional state and linked by scBCR clone."""
    x0, y0, w, h = box
    T(ax, x0 + 6.8, y0 + 3.5, 'scRNA-seq', LABEL, BLUE, weight='bold')
    cells(ax, place((x0 + 1.5, y0 + 6.0, w - 3.0, 18.8)), [STATES[s] for s in STATE], 1.3, alpha=.9)
    T(ax, x0 + 2.4, y0 + 27.4, 'scBCR-seq', LABEL, ACCENT, weight='bold')
    X, Y = cells(ax, place((x0 + 1.5, y0 + 29.8, w - 3.0, 18.8)), CELL, 1.3)
    clone_links(ax, X, Y)


def card_frechet(ax, box):
    """2: overlapping clones on the cell UMAP, each pooled into one Frechet mean."""
    x0, y0, w, h = box
    T(ax, x0 + w - 2.4, y0 + 3.5, 'Fréchet mean', LABEL, INK, ha='right')
    star_y = y0 + 9.6
    X, Y = cells(ax, place((x0 + 1.0, y0 + 15.0, w - 2.0, h - 16.0), pad=3.2), CELL, 1.3)
    xs = {'A': x0 + .22 * w, 'C': x0 + .50 * w, 'B': x0 + .78 * w}
    for name in 'ABC':
        idx = MEMBERS[name]
        pts = np.column_stack([X[idx], Y[idx]])
        ax.add_patch(Polygon(territory(pts), closed=True, facecolor=to_rgba(CLONES[name], .11),
                             edgecolor=to_rgba(CLONES[name], .7), linewidth=.6, zorder=3.5))
        for p in pts:
            ax.plot([p[0], xs[name]], [p[1], star_y], color=CLONES[name], linewidth=.28, alpha=.4, zorder=4)
        ax.scatter(*pts.T, s=6.5, c=CLONES[name], edgecolors='white', linewidths=.3, zorder=5)
        star(ax, xs[name], star_y, CLONES[name])


def mixed_map(ax, centre, width, colours, frac=.5, seed=8):
    """Mini scRNA UMAP: cells of several arms mapped back, interleaved evenly."""
    (xa, xb), (ya, yb) = (D[:, 0].min(), D[:, 0].max()), (D[:, 1].min(), D[:, 1].max())
    s = width / (xb - xa)
    X = centre[0] + (D[:, 0] - (xa + xb) / 2) * s
    Y = centre[1] + (D[:, 1] - (ya + yb) / 2) * s
    rng = np.random.default_rng(seed)
    lit = rng.random(len(D)) < frac
    which = rng.integers(0, len(colours), len(D))
    ax.scatter(X[~lit], Y[~lit], s=.8, c=LIGHT_CELL, linewidths=0, zorder=4)
    order = rng.permutation(np.flatnonzero(lit))
    ax.scatter(X[order], Y[order], s=1.5, c=[colours[k] for k in which[order]], linewidths=0, zorder=5)
    return width, (yb - ya) * s


def card_embedding(ax, box):
    """3: clones sit where their Frechet means fall; the two right arms are invisible on the cell UMAP."""
    x0, y0, w, h = box
    side = h - 6.5
    ex0, ey0 = x0 + (w - side) / 2, y0 + 5.0
    to = embed_map(ex0, ey0, side)
    T(ax, ex0 + side / 2, y0 + 3.5, 'clone embedding', LABEL, INK, ha='center')
    for p, colour in enumerate(PROGRAMMES):
        arm_halo(ax, to, side, p, facecolor=to_rgba(colour, .09), edgecolor='none', zorder=3)
    embedding_dots(ax, to)
    special = EMBED[3]
    a, a2 = to(*special['A']), to(*special["A'"])
    ax.plot([a[0], a2[0]], [a[1], a2[1]], color=CLONES['A'], linewidth=1.2, zorder=7, solid_capstyle='round')
    ax.scatter(*a2, s=30, c=A_PRIME, edgecolors='white', linewidths=.4, zorder=8)
    for name in 'ABC':
        star(ax, *to(*special[name]), CLONES[name])
    # Left: A and A' share one state footprint, so they are neighbours.
    glyph_w = 15.0
    for name, color, cy in [('A', CLONES['A'], y0 + 15.5), ("A'", A_PRIME, y0 + 34.5)]:
        centre = (x0 + 2.0 + glyph_w / 2, cy)
        gw, gh = footprint(ax, centre, glyph_w, FOOTPRINT['A'], color)
        target = to(*special[name])
        ax.plot([centre[0] + gw / 2 + .8, target[0]], [cy, target[1]], color=color, linewidth=.5,
                linestyle=(0, (1.4, 1.1)), zorder=6)
    # Right: cells of the two right arms, mapped back, interleave evenly on the scRNA UMAP.
    mw = 19.0
    mc = (x0 + w - 2.0 - mw / 2, y0 + 28.0)
    gw, gh = mixed_map(ax, mc, mw, [TEAL, RED])
    T(ax, mc[0], mc[1] - gh / 2 - 2.6, 'scRNA UMAP', LABEL, MUTED, ha='center')
    for p, dy in [(1, -gh * .28), (2, gh * .28)]:
        tip = to(*arm_tip(p))
        ax.plot([tip[0], mc[0] - mw / 2 - .6], [tip[1], mc[1] + dy], color=PROGRAMMES[p], linewidth=.6,
                linestyle=(0, (1.4, 1.1)), zorder=6)
        ax.scatter(*tip, s=5, c=PROGRAMMES[p], zorder=7)


def card_fidelity(ax, box):
    """4: observed clonal ICC against 500 clone-label shuffles within each sample."""
    x0, y0, w, h = box
    left, right, base_y = x0 + 6.5, x0 + w - 5.5, y0 + 31.0
    scale = (right - left) / .5                        # clonal ICC axis, 0 to 0.5
    ax.plot([left, right], [base_y, base_y], color=INK, linewidth=.6, zorder=4)
    for v in (0, .1, .2, .3, .4, .5):
        x = left + v * scale
        ax.plot([x, x], [base_y, base_y + .9], color=INK, linewidth=.5, zorder=4)
        if v in (0, .2, .4):
            T(ax, x, base_y + 2.7, f'{v:g}', LABEL - .4, MUTED, ha='center')
    width = .009
    centres = np.arange(.004, .085, width)
    for c in centres:
        height = np.exp(-.5 * ((c - .035) / .015) ** 2) * 16.5
        ax.add_patch(Rectangle((left + (c - width / 2) * scale, base_y - height), width * scale * .82, height,
                               facecolor='#AEB9C2', edgecolor='none', zorder=4))
    null_edge = left + .08 * scale
    obs_x, top = left + .45 * scale, y0 + 11.5
    ax.plot([obs_x, obs_x], [base_y, top], color=ACCENT, linewidth=1.6, zorder=5, solid_capstyle='butt')
    ax.scatter([obs_x], [top], s=26, c=ACCENT, zorder=6)
    gap_y = base_y - 7.0
    ax.add_patch(FancyArrowPatch((null_edge + .8, gap_y), (obs_x - .9, gap_y), arrowstyle='<|-|>',
                                 mutation_scale=5, color=MUTED, linewidth=.5, zorder=5))
    T(ax, (null_edge + obs_x) / 2, gap_y - 2.2, 'P < 0.01', LABEL - .2, INK, ha='center')
    T(ax, x0 + 3.0, y0 + 8.6, 'shuffled', LABEL, MUTED, weight='bold')
    T(ax, x0 + 3.0, y0 + 11.2, '500 permutations within each sample', LABEL - .6, MUTED)
    T(ax, obs_x, top - 3.2, 'observed', LABEL, ACCENT, weight='bold', ha='center')
    T(ax, x0 + w / 2, base_y + 6.8, 'clonal ICC = τ² / (τ² + σ²)', LABEL, MUTED, ha='center')


def card_structure(ax, box):
    """5: GC B and plasma cells are one mass on the scRNA UMAP; Threadfin resolves their clones
    into a continuous structure, a GC hub with two parallel branches."""
    x0, y0, w, h = box
    mid = y0 + 27.0
    # Left: the cells, coloured by state, form one mass.
    T(ax, x0 + 14.0, y0 + 9.6, 'scRNA UMAP', LABEL, MUTED, ha='center')
    X, Y = place((x0 + 2.5, mid - 15.0, 23.0, 30.0), pts=GC_PC)(GC_PC[:, 0], GC_PC[:, 1])
    order = np.random.default_rng(3).permutation(len(GC_PC))
    ax.scatter(X[order], Y[order], s=1.5, c=[(BLUE, RED)[int(k)] for k in GC_PC[order, 2]], linewidths=0,
               alpha=.9, zorder=4)
    arrow(ax, (x0 + 26.6, mid), (x0 + 34.4, mid), ACCENT, .9, 7)
    T(ax, x0 + 30.5, mid - 2.5, 'Threadfin', LABEL - .6, ACCENT, weight='bold', ha='center')
    # Right: their clones form a GC hub and two parallel branches, continuous from blue to red.
    ex0, ey0, ew, eh = x0 + 36.5, mid - 15.0, w - 39.0, 30.0
    T(ax, ex0 + ew / 2, y0 + 9.6, 'clone embedding', LABEL, INK, weight='bold', ha='center')
    cx, cy, rx, ry = FORK_HUB
    hub_half, band_half = ry * eh + 1.2, 1.85           # mm; band_half matches the 10.5-pt branch stroke
    ax.add_patch(Ellipse((ex0 + cx * ew, ey0 + cy * eh), 2 * rx * ew + 2.4, 2 * hub_half,
                         facecolor=FORK_BAND, edgecolor='none', zorder=3))
    neck = np.linspace(cx, FORK_SPLIT + .06, 40)        # smooth taper from the hub into the stem
    u = (neck - cx) / (neck[-1] - cx)
    half = band_half + (hub_half - band_half) * (1 - u * u * (3 - 2 * u))
    nx = ex0 + neck * ew
    ax.add_patch(Polygon(np.column_stack([np.r_[nx, nx[::-1]], ey0 + cy * eh + np.r_[-half, half[::-1]]]),
                         closed=True, facecolor=FORK_BAND, edgecolor='none', zorder=3))
    for sign in (-1, 1):
        path = fork_path(sign)[0]
        ax.plot(ex0 + path[:, 0] * ew, ey0 + path[:, 1] * eh, color=FORK_BAND, linewidth=10.5,
                solid_capstyle='round', zorder=3)
    xy, progress, size = FORK
    ax.scatter(ex0 + xy[:, 0] * ew, ey0 + xy[:, 1] * eh, s=1.4 + .95 * size,
               c=LinearSegmentedColormap.from_list('fork', concept.GRADIENT)(progress), edgecolors='white',
               linewidths=.2, zorder=4)


def card_continuity(ax, box):
    """6: position along the clone embedding tracks antigen specificity and fate mapping."""
    x0, y0, w, h = box
    rng = np.random.default_rng(15)
    left, right = x0 + 7.5, x0 + w - 3.0
    t = np.sort(rng.uniform(0, 1, 34))
    xs = left + t * (right - left)
    # One branch of card 5, unrolled: a continuous band of clones from blue to red.
    strip = LinearSegmentedColormap.from_list('continuum', concept.GRADIENT)
    band = (np.arange(110) + np.random.default_rng(16).uniform(0, 1, 110)) / 110
    ax.plot([left + 1.2, right - 1.2], [y0 + 8.4] * 2, color=FORK_BAND, linewidth=10.5, solid_capstyle='round',
            zorder=4)
    ax.scatter(left + band * (right - left), y0 + 8.4 + np.random.default_rng(17).normal(0, .55, len(band)),
               s=1.4 + .95 * np.random.default_rng(18).integers(1, 8, len(band)), c=strip(band),
               edgecolors='white', linewidths=.2, zorder=5)
    arrow(ax, (left, y0 + 12.6), (right, y0 + 12.6), MUTED, .6, 5)
    for top, bottom, centre, steep, colour, label in [(y0 + 15.8, y0 + 26.0, .45, .14, '#B07A16', 'antigen binding'),
                                                      (y0 + 28.8, y0 + 39.0, .62, .08, RED, 'fate mapping')]:
        ax.plot([left - 1.2, left - 1.2], [top, bottom], color=MUTED, linewidth=.5, zorder=4)
        ax.plot([left - 1.2, right], [bottom, bottom], color=MUTED, linewidth=.5, zorder=4)
        curve = 1 / (1 + np.exp(-(np.linspace(0, 1, 80) - centre) / steep))
        ax.plot(np.linspace(left, right, 80), bottom - (.08 + .84 * curve) * (bottom - top), color=colour,
                linewidth=1.0, zorder=6)
        value = np.clip(1 / (1 + np.exp(-(t - centre) / steep)) + rng.normal(0, .09, len(t)), 0, 1)
        ax.scatter(xs, bottom - (.08 + .84 * value) * (bottom - top), s=4.5, c=colour, alpha=.55,
                   edgecolors='none', zorder=5)
        T(ax, left + .4, top + .9, label, LABEL - .2, colour, weight='bold')
    T(ax, (left + right) / 2, y0 + h - 3.4, 'position along clone embedding', LABEL - .2, MUTED, ha='center')


# ----------------------------------------------------------------------------- panel b
def card(ax, number, box, draw):
    x0, y0, w, h = box
    first = len(ax.texts)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=1.5',
                                facecolor='white', edgecolor='#DCE4EA', linewidth=.5, zorder=1))
    ax.add_patch(Circle((x0 + 3.4, y0 + 3.4), 2.1, facecolor=ACCENT, edgecolor='none', zorder=9))
    T(ax, x0 + 3.4, y0 + 3.45, str(number), 7.0, 'white', weight='bold', ha='center', zorder=10)
    draw(ax, box)
    CARDS.append((box, list(ax.texts)[first:]))


def section(ax, box, title, face, title_color):
    x, y, w, h = box
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.2',
                                facecolor=face, edgecolor='none', zorder=0))
    T(ax, x + 4.0, y + 4.5, title, SECTION_TITLE, title_color, weight='bold')


def panel_b(ax):
    T(ax, 10.0, B_TOP + 4.8, B_TITLE, PANEL_TITLE, INK, weight='bold')
    for box, title, face, colour in SECTIONS:
        section(ax, box, title, face, colour)
    # Algorithm pipeline: three cards joined by arrows.
    x, y, w, h = S_PIPE
    top, height = y + CARD_TOP, h - CARD_TOP - CARD_PAD
    widths, gap, left = [38.0, 52.0, 86.0], 7.0, x + 4.0
    for number, (width, draw) in enumerate(zip(widths, [card_cells_and_clones, card_frechet, card_embedding]), 1):
        card(ax, number, (left, top, width, height), draw)
        if number > 1:
            arrow(ax, (left - gap + 1.3, top + height / 2), (left - 1.3, top + height / 2))
        left += width + gap
    # Fidelity test: one card.
    x, y, w, h = S_FID
    card(ax, 4, (x + 4.0, y + CARD_TOP, w - 8.0, h - CARD_TOP - CARD_PAD), card_fidelity)
    fid = (x + 4.0, y + CARD_TOP, w - 8.0, h - CARD_TOP - CARD_PAD)
    # Biological inference: the continuous clonal structure, then the continuity summary.
    x, y, w, h = S_BIO
    top, height = y + CARD_TOP, h - CARD_TOP - CARD_PAD
    widths, gap, left = [70.0, 44.0], 6.0, x + 4.0
    arrow(ax, (fid[0] + fid[2] + .8, top + height / 2), (left - .8, top + height / 2))
    for number, (width, draw) in enumerate(zip(widths, [card_structure, card_continuity]), 5):
        card(ax, number, (left, top, width, height), draw)
        if number > 5:
            arrow(ax, (left - gap + 1.0, top + height / 2), (left - 1.0, top + height / 2))
        left += width + gap


# ----------------------------------------------------------------------------- page
def figure():
    CARDS.clear()
    plt.rcParams.update(concept.STYLE)
    plt.rcParams['svg.hashsalt'] = 'threadfin-figure1-a4-20261016'
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    page = fig.add_axes([0, 0, 1, 1])
    page.set_xlim(0, PAGE_W)
    page.set_ylim(PAGE_H, 0)
    page.set_axis_off()

    ax_a = fig.add_axes([A_LEFT / PAGE_W, 1 - (A_TOP + A_H) / PAGE_H, A_W / PAGE_W, A_H / PAGE_H])
    ax_a.set_xlim(*A_X)
    ax_a.set_ylim(A_Y[1], A_Y[0])
    ax_a.set_aspect('equal')
    ax_a.set_axis_off()
    ax_a.patch.set_visible(False)
    concept.draw(ax_a, k=A_K, fonts=A_FONTS, header=False, key=False)

    for letter, y in [('a', A_TOP + .4), ('b', B_TOP + .4)]:
        page.text(3.2, y, letter, fontsize=11, weight='bold', color=INK, va='top', ha='left')

    belly = (A_LEFT + (FISH_BELLY[0] - A_X[0]) * A_K, A_TOP + (FISH_BELLY[1] - A_Y[0]) * A_K)
    arrow(page, belly, (belly[0], B_TOP + 1.4), ACCENT, 1.3, 10)
    panel_b(page)
    return fig


def audit(fig):
    report = concept.audit(fig)
    report['figure'] = 'Figure 1 (A4)'
    report['page_mm'] = [PAGE_W, PAGE_H]
    renderer = fig.canvas.get_renderer()
    page = fig.axes[0]
    outside_card = []
    for (x0, y0, w, h), texts in CARDS:
        (cx0, cy0), (cx1, cy1) = page.transData.transform([(x0 + .3, y0 + .3), (x0 + w - .3, y0 + h - .3)])
        lo_x, hi_x, lo_y, hi_y = min(cx0, cx1), max(cx0, cx1), min(cy0, cy1), max(cy0, cy1)
        for t in texts:
            b = t.get_window_extent(renderer)
            if b.x0 < lo_x or b.x1 > hi_x or b.y0 < lo_y or b.y1 > hi_y:
                outside_card.append(t.get_text())
    report['text_outside_card'] = outside_card
    return report


def manifest():
    return {
        'figure': 'Figure 1 (A4): a, Threadfin concept; b, algorithm pipeline, clonal coherence test, biological inference',
        'page_mm': [PAGE_W, PAGE_H],
        'panel_a': {'source': 'figure1_programmes.draw(header=False, key=False)', 'mm_per_design_unit': A_K},
        'panel_b_title': B_TITLE,
        'sections': {
            'Algorithm pipeline': ['1 scRNA-seq states / scBCR clone links on the same cells',
                                   '2 overlapping clones pooled into Frechet means',
                                   "3 clone embedding: points at Frechet means; A and A' share a footprint and sit "
                                   'together; cells of the two right arms interleave evenly on the scRNA UMAP'],
            'Clonal coherence test': ['4 observed clonal ICC vs 500 clone-label shuffles within samples'],
            'Biological inference': ['5 GC B and plasma cells: one mass on the scRNA UMAP; their clones form a '
                                     'continuous GC hub with two parallel, unlabelled branches in the clone '
                                     'embedding (coloured GC-like to plasma-like)',
                                     '6 position along the clone embedding tracks antigen binding and fate mapping']},
        'clones': {k: {'states': FOOTPRINT[k], 'n_cells': int(len(v))} for k, v in MEMBERS.items()},
        'embedding_clones': int(len(EMBED[0])),
        'card5': {'cells': int(len(GC_PC)), 'plasma_cells': int(GC_PC[:, 2].sum()),
                  'clones': int(len(FORK[0]))},
        'scope': 'Schematic; synthetic cells, clones, embeddings, statistics and readouts',
        'not_used': 'Clonotrace density surfaces, kNN-graph diffusion and optimal-transport distance matrix',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--preview', action='store_true', help='write only the 170-dpi preview')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    fig = figure()
    report = audit(fig)
    (WORK / f'{STEM}_text_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('overlaps', report['text_overlaps'], 'outside', report['text_outside_page'],
          'touching', report['text_touching_dots'], 'outside card', report['text_outside_card'], flush=True)
    fig.savefig(WORK / f'{STEM}_preview.png', dpi=170, facecolor='white')
    if not args.preview:
        assert not report['text_overlaps'], report['text_overlaps']
        assert not report['text_outside_page'], report['text_outside_page']
        assert not report['text_touching_dots'], report['text_touching_dots']
        assert not report['text_outside_card'], report['text_outside_card']
        for suffix in ['svg', 'pdf', 'png']:
            fig.savefig(HERE / f'{STEM}.{suffix}', dpi=600, facecolor='white')
        fig.savefig(HERE / 'reserve/current_previews' / f'{STEM}.png', dpi=170, facecolor='white')
        PROVENANCE.write_text(json.dumps(manifest(), ensure_ascii=False, indent=2) + '\n')
        print('wrote', HERE / STEM, '{svg,pdf,png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
