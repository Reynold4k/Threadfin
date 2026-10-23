#!/usr/bin/env python
"""Figure 1: Threadfin concept and workflow (restored fish-thread layout).

Restores the 2026-10-13 composition drawn by ``figure1_overview.py``: paired
scRNA-seq / scBCR-seq inputs, the threadfin fish whose filaments carry both
inputs into Threadfin, and the two clone-level outputs (clone map with clonal
programmes; continuous clone-state gradient). Revised 2026-10-16 for the
method-paper framing:

* the clone map shows three neutral clonal programmes (former "GC cycling",
  "Memory-like" and "Output-like" regions become programmes 1, 2 and 3);
* the selection-node outline, its label and pointer, and the hypothetical
  return-to-GC arrow are removed; former node clones keep their coordinates
  and sizes and take the programme of the arm they point towards, so each
  programme stays one connected region meeting at the junction;
* the lower panel is titled "Clone embedding" without end labels; its
  continuous colour blend runs between the programme 1 and 3 colours;
* no in-figure title (the legend carries it); Arial-metric type >= 6.6 pt.

Every coordinate, clone size and mixture is synthetic and uses the original
seeds; nothing here is fitted to data or pooled from Figures 2-4.

Usage:  python figure1_programmes.py [--preview]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.transforms import Bbox

import figure1_overview as base
from figure1_overview import ACCENT, BLUE, GOLD, INK, MUTED, RED, RULE, TEAL, arrow, receptor

HERE = Path(__file__).resolve().parents[2]
WORK = HERE / 'reserve/figure1_restore_2026-10-16'
PROVENANCE = HERE / 'reserve/provenance/Figure_1_restored_programmes.json'
STEM = 'Figure_1'

MARGIN_X = 4.0                       # design units beside the original 183-unit drawing
Y_SPAN = (-1.0, 179.5)               # design units shown vertically
PAGE_W = 210.0                       # mm, as Figures 2-4
K = PAGE_W / (183 + 2 * MARGIN_X)    # mm per design unit
PAGE_H = (Y_SPAN[1] - Y_SPAN[0]) * K
DOT_AREA = base.DOT_AREA * K ** 2    # keeps the original dot-to-drawing proportion

PROGRAMMES = [BLUE, TEAL, RED]       # clonal programmes 1, 2, 3 (original region colours)
PROGRAMME_TEXT = ['#2D6A96', '#2F7D71', '#A84B64']
PROGRAMME_NAMES = ['Clonal programme 1', 'Clonal programme 2', 'Clonal programme 3']
GRADIENT = [BLUE, '#5D76BF', '#9368B8', '#B55E9A', RED]   # LCh blend between the P1 and P3 colours
NODE_CENTRE = (124.0, 65.0)
ARM_DIRECTIONS = [116.6, 323.1, 69.9]   # degrees (y down) from the node to the P1, P2, P3 bridges
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
         'svg.hashsalt': 'threadfin-figure1-programmes-20261016'}
FONTS = {'head': 10, 'sub': 9, 'body': 7.5, 'small': 6.6, 'label': 8.5}


def txt(ax, x, y, text, size=7.5, color=INK, **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, va='center', linespacing=1.2, **kwargs)


# ----------------------------------------------------------------------------- inputs
def bcr_input(ax, f):
    """Membrane receptor -> per-cell H/L contigs -> clones within each donor."""
    txt(ax, 0, 75, 'scBCR-seq', f['sub'], ACCENT, weight='bold')
    txt(ax, 0, 80.5, 'Heavy + light chains', f['body'], MUTED)
    ax.add_patch(Circle((7.5, 93), 4.8, facecolor='#EFF4F7', edgecolor='#7899AF', linewidth=.85))
    ax.add_patch(Ellipse((6.7, 94), 5.5, 5.2, angle=-25, facecolor='#C6D6E1', edgecolor='none'))
    for degrees in [-100, -25, 50, 135]:
        theta = np.deg2rad(degrees)
        receptor(ax, 7.5 + 4.7 * np.sin(theta), 93 - 4.7 * np.cos(theta), degrees, .70)
    arrow(ax, (14.6, 91.8), (18.4, 91.8), color='#8DADBF', width=.7)
    for y, label, widths, letters in [(87.5, 'H', [8.5, 2.8, 4.2], ['V', 'D', 'J']),
                                      (96.1, 'L', [8.5, 4.2], ['V', 'J'])]:
        txt(ax, 19.6, y, label, f['small'], ACCENT, ha='center', weight='bold')
        left = 22.2
        for i, (width, letter) in enumerate(zip(widths, letters)):
            color = ['#385B73', '#91B0C2', '#61889F'][i]
            ax.add_patch(Rectangle((left, y - 1.6), width, 3.2, facecolor=color,
                                   edgecolor='white', linewidth=.5))
            txt(ax, left + width / 2, y, letter, f['small'], 'white', ha='center')
            left += width + .6
    ax.plot([28.7, 28.7, 38.9, 38.9], [89.8, 90.3, 90.3, 89.8], color=MUTED, linewidth=.5)
    txt(ax, 33.8, 92.2, 'junction', f['small'], MUTED, ha='center')
    txt(ax, 7.5, 102.3, 'B cell', f['body'], MUTED, ha='center')
    txt(ax, 29.6, 102.3, 'Paired by cell barcode', f['body'], MUTED, ha='center')
    base.receptor_trees(ax)
    txt(ax, 0, 119.2, 'Clones within each donor', f['body'], MUTED)


def inputs(ax, f, k=K, header=True):
    if header:
        txt(ax, 0, 4, 'Paired single cells', f['head'], weight='bold')
    txt(ax, 0, 13, 'scRNA-seq', f['sub'], BLUE, weight='bold')
    txt(ax, 0, 18.5, 'Expression state', f['body'], MUTED)
    data = base.rna_data()
    for state, color in enumerate([BLUE, GOLD, RED]):
        q = data[data[:, 2] == state]
        ax.scatter(q[:, 0], q[:, 1], s=3.4 * k ** 2, color=color, marker='o', alpha=.86,
                   edgecolors='white', linewidths=.14, zorder=2)
    txt(ax, 0, 61.5, 'One dot = one cell', f['body'], MUTED)
    ax.plot([0, 42], [67, 67], color=RULE, linewidth=.7)
    bcr_input(ax, f)
    base.threadfin_fish(ax)          # approved PNG plus the four tapered filament ribbons


# ----------------------------------------------------------------------------- clone map
def programme_data():
    """Original synthetic clones (x, y, programme, captured cells).

    Former selection-node clones keep position and size; each joins the
    programme whose bridge direction is angularly closest around the node.
    """
    data = base.landscape_data()
    state = data[:, 2].astype(int)                 # 0 GC cycling, 1 node, 2 output, 3 memory
    programme = np.select([state == 0, state == 3, state == 2], [0, 1, 2], -1)
    node = state == 1
    offset = data[node, :2] - NODE_CENTRE
    angle = np.degrees(np.arctan2(offset[:, 1], offset[:, 0])) % 360
    gap = np.abs((angle[:, None] - np.asarray(ARM_DIRECTIONS)[None, :] + 180) % 360 - 180)
    programme[node] = np.argmin(gap, axis=1)
    return np.column_stack([data[:, :2], programme, data[:, 3]])


def size_key(ax, f, dot_area, x0=150.5, y_label=4.3, y_dots=10.6):
    txt(ax, x0 - 1.8, y_label, 'Captured cells', f['body'], MUTED)
    for x, n in [(x0, 5), (x0 + 10, 20), (x0 + 21.5, 80)]:
        ax.scatter([x], [y_dots], s=dot_area * n, c=MUTED, marker='o', edgecolors='white', linewidths=.3)
        txt(ax, x + 1.2 + .23 * np.sqrt(n), y_dots, str(n), f['body'], MUTED)


def clone_map(ax, f, dot_area=DOT_AREA, header=True, key=True):
    if header:
        txt(ax, 60, 4, 'Discover clonal programmes', f['head'], weight='bold')
    txt(ax, 60, 11 if header else 13, 'One dot = one clone', f['body'], MUTED)
    if key:
        size_key(ax, f, dot_area)
    for centre, wh, angle, color in [((83, 64), (22, 47), -17, BLUE),
                                     ((165, 39), (23, 16), -25, TEAL),
                                     ((165, 106), (35, 18), 12, RED)]:
        ax.add_patch(Ellipse(centre, *wh, angle=angle, facecolor=to_rgba(color, .06),
                             edgecolor='none', zorder=0))
    data = programme_data()
    drawn = data[np.argsort(-data[:, 3], kind='stable')]      # small clones above large ones
    ax.scatter(drawn[:, 0], drawn[:, 1], s=dot_area * drawn[:, 3],
               c=np.asarray(PROGRAMMES)[drawn[:, 2].astype(int)], marker='o',
               alpha=.88, edgecolors='white', linewidths=.28, zorder=3)
    txt(ax, 79, 34.5, PROGRAMME_NAMES[0], f['label'], PROGRAMME_TEXT[0], weight='bold', ha='center')
    txt(ax, 151.5, 34.5, PROGRAMME_NAMES[1], f['label'], PROGRAMME_TEXT[1], weight='bold', ha='right')
    txt(ax, 163, 122, PROGRAMME_NAMES[2], f['label'], PROGRAMME_TEXT[2], weight='bold', ha='center')
    return data


# ----------------------------------------------------------------------------- gradient
def gradient(ax, f, dot_area=DOT_AREA):
    ax.plot([87, 183], [132, 132], color=RULE, linewidth=.7)
    txt(ax, 89, 139, 'Clone embedding', f['head'], weight='bold')
    d = base.continuum_data()
    display_x = 90 + (d[:, 0] - 65) * 89 / 114      # same horizontal refit as the original
    cmap = LinearSegmentedColormap.from_list('clone_embedding', GRADIENT)
    ax.scatter(display_x, d[:, 1], s=dot_area * d[:, 3], c=cmap(d[:, 2]), marker='o',
               edgecolors='white', linewidths=.28, alpha=.95, zorder=3)


def draw(ax, *, k=K, fonts=None, header=True, key=True):
    """Draw the concept graphic on ``ax`` in design units, at ``k`` mm per unit.

    ``header`` adds the two section titles and ``key`` the captured-cell size
    key; the A4 Figure 1 (``figure1_a4.py``) uses neither.
    """
    f = {**FONTS, **(fonts or {})}
    dot_area = base.DOT_AREA * k ** 2
    inputs(ax, f, k, header)
    clone_map(ax, f, dot_area, header, key)
    gradient(ax, f, dot_area)


def figure():
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    fig.page_height = PAGE_H
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-MARGIN_X, 183 + MARGIN_X)
    ax.set_ylim(Y_SPAN[1], Y_SPAN[0])
    ax.set_aspect('equal')
    ax.set_axis_off()
    draw(ax)
    return fig


# ----------------------------------------------------------------------------- checks + export
def audit(fig):
    """Text overlaps, text beyond the page and text touching any dot."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text().strip()]
    boxes = [t.get_window_extent(renderer) for t in texts]
    overlaps, outside, touching = [], [], []
    for i, (t, b) in enumerate(zip(texts, boxes)):
        if not (fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1)):
            outside.append(t.get_text())
        for u, c in zip(texts[i + 1:], boxes[i + 1:]):
            inter = Bbox.intersection(b, c)
            if inter is not None and inter.width > .3 and inter.height > .3:
                overlaps.append([t.get_text(), u.get_text()])
        for coll in (c for ax in fig.axes for c in ax.collections if isinstance(c, PathCollection)):
            pts = coll.get_offset_transform().transform(coll.get_offsets())
            radius = np.sqrt(np.broadcast_to(coll.get_sizes(), (len(pts),))) * .5 * fig.dpi / 72
            near = ((pts[:, 0] + radius >= b.x0) & (pts[:, 0] - radius <= b.x1)
                    & (pts[:, 1] + radius >= b.y0) & (pts[:, 1] - radius <= b.y1))
            if near.any():
                touching.append({'text': t.get_text(), 'points': int(near.sum())})
    return {'figure': 1, 'page_mm': [round(PAGE_W, 2), round(PAGE_H, 2)],
            'text_overlaps': overlaps, 'text_outside_page': outside, 'text_touching_dots': touching,
            'font_sizes_pt': sorted({t.get_fontsize() for t in texts}),
            'text_inventory': [{'text': t.get_text(), 'font_pt': t.get_fontsize()} for t in texts],
            'svg_text_mode': 'editable text; the approved fish PNG is the only embedded image'}


def manifest():
    original = base.landscape_data()
    data = programme_data()
    node = original[:, 2] == 1
    return {
        'figure': 'Figure 1 | Threadfin concept and workflow',
        'restored_from': 'figure1_overview.py layout rendered 2026-10-13 (reserve/main_figure_text_2026-10-16_142405/Figure_1.png)',
        'replaces': 'A4 portrait reflow of 2026-10-16 14:44 (archived in reserve/figure1_restore_2026-10-16/before/)',
        'page_mm': [round(PAGE_W, 2), round(PAGE_H, 2)],
        'mm_per_design_unit': round(K, 4),
        'removed': ['figure title and subtitle', 'selection-node outline, fill, label and pointer',
                    'dashed return-to-GC arrow with its two labels',
                    'gradient end labels (Cycling / Output-like)'],
        'renamed': {'GC cycling': PROGRAMME_NAMES[0], 'Memory-like': PROGRAMME_NAMES[1],
                    'Output-like states': PROGRAMME_NAMES[2],
                    'Identify GC selection-associated nodes': 'Discover clonal programmes',
                    'Resolve cycling-to-output clone gradients': 'Clone embedding'},
        'former_node_clones': {'n': int(node.sum()),
                               'rule': 'angularly closest bridge direction around the node centre',
                               'node_centre': NODE_CENTRE, 'arm_directions_deg': ARM_DIRECTIONS,
                               'assigned': {name: int(np.sum(data[node, 2] == p))
                                            for p, name in enumerate(PROGRAMME_NAMES)}},
        'clone_coordinates_and_sizes_unchanged': bool(np.array_equal(original[:, [0, 1, 3]], data[:, [0, 1, 3]])),
        'clone_geometry_sha256': hashlib.sha256(np.ascontiguousarray(original[:, [0, 1, 3]]).tobytes()).hexdigest(),
        'clones_on_map': int(len(data)),
        'gradient_clones': int(len(base.continuum_data())),
        'programme_colours': dict(zip(PROGRAMME_NAMES, PROGRAMMES)),
        'gradient_palette': GRADIENT,
        'seeds': {'map': base.ART_SEED, 'gradient': 328, 'rna_cells': 217},
        'fish_asset_sha256': hashlib.sha256(base.FISH_ASSET.read_bytes()).hexdigest(),
        'scope': 'Synthetic schematic: no fitted embedding, inferred lineage direction or pooled data',
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
          'touching', report['text_touching_dots'], flush=True)
    fig.savefig(WORK / f'{STEM}_preview.png', dpi=170, facecolor='white')
    if not args.preview:
        assert not report['text_overlaps'], report['text_overlaps']
        assert not report['text_outside_page'], report['text_outside_page']
        assert not report['text_touching_dots'], report['text_touching_dots']
        for suffix in ['svg', 'pdf', 'png']:
            fig.savefig(HERE / f'{STEM}.{suffix}', dpi=600, facecolor='white')
        fig.savefig(HERE / 'reserve/current_previews' / f'{STEM}.png', dpi=170, facecolor='white')
        PROVENANCE.write_text(json.dumps(manifest(), ensure_ascii=False, indent=2) + '\n')
        print('wrote', HERE / STEM, '{svg,pdf,png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
