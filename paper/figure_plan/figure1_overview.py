"""Figure 1: vector scientific diagrams with the approved raster fish mascot.

Every coordinate, family, size and programme mixture is synthetic.
The pseudo-UMAP is authored conceptual geometry, not a fitted embedding or a
pooled analysis of Figures 2 and 3. The RNA cloud depicts three expression
clusters; separate coloured receptor trees illustrate three within-donor
families, not an inferred cross-family ancestry. The curved return arrow is
a biological hypothesis, not an estimated software output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path as FilePath

import numpy as np
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, PathPatch, Polygon, Rectangle,
)
from matplotlib.path import Path

INK = '#213544'
MUTED = '#65737D'
RULE = '#DAE3E8'
BLUE = '#397DAD'
GOLD = '#C79A31'
RED = '#BD6179'
TEAL = '#44988B'
ACCENT = '#294D65'
STATE = [BLUE, GOLD, RED, TEAL]
NAMES = ['GC cycling', 'Selection-associated', 'Output-like', 'Memory-like']
FAMILY_COLORS = ['#7865A8', '#C4805B', '#448F85']
GRADIENT_COLORS = ['#225CC5', '#13A9B8', '#F2A12C', '#DC405C']
ART_SEED = 8126
DOT_AREA = .78  # points squared per illustrative captured cell
FISH_ASSET = FilePath(__file__).resolve().parent / 'assets/threadfin_fish_cartoon.png'
FISH_PLACEMENT = (0, 132, 83)  # left, top, width in figure-coordinate millimetres
VISUAL_REFERENCES = [
    {'paper': 'Dandelion, Nature Biotechnology (2024), Fig. 1',
     'url': 'https://www.nature.com/articles/s41587-023-01734-7',
     'motif': 'Paired measurements, V(D)J segments and clone-network abstraction'},
    {'paper': 'scRepertoire 2, PLOS Computational Biology (2025), Fig. 1',
     'url': 'https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012760',
     'motif': 'Membrane BCR on a B cell, contigs and receptor-based grouping'},
]


def txt(ax, x, y, text, size=7, color=INK, **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, va='center',
                   linespacing=1.25, **kwargs)


def path(ax, vertices, color=RULE, width=.8, dash=None, arrow=False, z=2, head=8):
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(vertices) - 1)
    curve = Path(vertices, codes)
    if arrow:
        artist = FancyArrowPatch(path=curve, arrowstyle='-|>', mutation_scale=head,
                                 color=color, linewidth=width, zorder=z,
                                 linestyle=dash or '-')
    else:
        artist = PathPatch(curve, fill=False, color=color, linewidth=width,
                           linestyle=dash or '-', zorder=z, capstyle='round')
    ax.add_patch(artist)
    return artist


def arrow(ax, start, end, color=ACCENT, width=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8,
                                color=color, linewidth=width, shrinkA=0, shrinkB=0))


def fin_extension(ax, controls, width=.65, name='fin'):
    """A tapered, softly shaded fin ribbon following tangent-matched cubics.

    Width is in figure millimetres, rather than a uniform line stroke. The
    first few points overlap the PNG's existing fin tip, matching its tangent
    and pale gold / blue-green colours without modifying the approved asset.
    """
    controls = np.asarray(controls, dtype=float)
    samples = []
    tangents = []
    for i in range(0, len(controls)-1, 3):
        c = controls[i:i+4]
        t = np.linspace(0, 1, 100)[int(i > 0):, None]
        samples.append((1-t)**3*c[0] + 3*(1-t)**2*t*c[1]
                       + 3*(1-t)*t**2*c[2] + t**3*c[3])
        tangents.append(3*(1-t)**2*(c[1]-c[0])
                        + 6*(1-t)*t*(c[2]-c[1]) + 3*t**2*(c[3]-c[2]))
    xy = np.vstack(samples)
    tangent = np.vstack(tangents)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-9)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = distance / distance[-1]
    ribbon_width = width*(1-.7*s)*np.minimum(1, (1-s)/.10)**.7 + .012
    offset = normal*ribbon_width[:, None]

    def strip(lo, hi, color, alpha=1, outline=False, z=5):
        polygon = np.vstack([xy+lo*offset, (xy+hi*offset)[::-1]])
        artist = Polygon(polygon, closed=True, facecolor=color, alpha=alpha,
                         edgecolor='#315C6D' if outline else 'none',
                         linewidth=.22 if outline else 0, joinstyle='round', zorder=z)
        artist.set_gid(f'{name}-outline' if outline else f'{name}-shading')
        ax.add_patch(artist)

    strip(-.5, .5, '#E6ECD7', outline=True)
    strip(-.43, -.10, '#9FBFC0', .72, z=5.1)
    strip(.02, .42, '#FCF2CA', .88, z=5.1)
    strip(-.10, .06, '#FFFFFF', .62, z=5.2)


def receptor(ax, x, y, theta, scale=1):
    """Original membrane-Ig glyph: paired heavy chains and shorter light chains."""
    rad = np.deg2rad(theta)
    rotation = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    for coords, color, width in [
        ([(-.23, 0), (-.23, -1.8), (-1.65, -3.45)], ACCENT, 1.0),
        ([(.23, 0), (.23, -1.8), (1.65, -3.45)], ACCENT, 1.0),
        ([(-1.17, -1.5), (-2.32, -2.93)], '#8DADBF', .9),
        ([(1.17, -1.5), (2.32, -2.93)], '#8DADBF', .9),
    ]:
        p = np.asarray(coords)*scale @ rotation.T + [x, y]
        ax.plot(*p.T, color=color, linewidth=width, solid_capstyle='round', zorder=4)


def rna_data():
    """A single compact synthetic cell manifold, partitioned into three states."""
    rng=np.random.default_rng(217)
    theta=rng.uniform(0,2*np.pi,620)
    radius=np.sqrt(rng.uniform(0,1,len(theta)))
    u,v=radius*np.cos(theta),radius*np.sin(theta)
    x=22+17*u-2.6*(v+.2)**2
    y=41+14*v+2.8*np.sin(np.pi*u)
    centres=np.array([[-.48,.04],[.28,-.50],[.34,.43]])
    d=((np.column_stack([u,v])[:,None,:]-centres)**2).sum(axis=2)
    group=np.argmin(d+rng.normal(0,.075,d.shape),axis=1)
    return np.column_stack([x,y,group])


def receptor_trees(ax):
    """Three separately rooted, schematic within-family sequence trees."""
    topology=np.array([[0,0],[0,-2.4],[-2.6,-4.3],[2.7,-5.1],
                       [-5.2,-8.0],[-.7,-7.2],[1.0,-8.7],[5.2,-8.0]])
    edges=[(0,1),(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]
    for x,color in zip([7,23,39],FAMILY_COLORS):
        xy=topology+[x,115]
        for i,j in edges:
            ax.plot(xy[[i,j],0],xy[[i,j],1],color=color,linewidth=.9,zorder=3)
        ax.scatter(xy[:,0],xy[:,1],s=[6,6,6,6,10,10,10,10],color=color,
                   edgecolors='white',linewidths=.25,marker='o',zorder=4)


def threadfin_fish(ax):
    """Embed the approved cartoon, extending its fins with matching ribbons."""
    artwork = imread(FISH_ASSET)
    left, top, width = FISH_PLACEMENT
    height = width * artwork.shape[0] / artwork.shape[1]

    def anchor(px, py):
        return (left + width * px / artwork.shape[1],
                top + height * py / artwork.shape[0])

    ax.imshow(artwork, extent=(left, left + width, top + height, top),
              origin='upper', interpolation='lanczos', zorder=4)
    # Start slightly inside each original tip and follow its local direction;
    # shared tangents between successive cubics avoid elbows and tight kinks.
    fin_extension(ax, [anchor(104,80), anchor(91,65), anchor(84,52), anchor(82,40),
                       (2.7,127), (12,123), (28,121),
                       (43,119.125), (49,115), (49,103),
                       (49,91), (50,45), (43,41)], .72, 'rna-fin')
    # This branch starts on the preceding cubic (t=.8), with its tangent.
    fin_extension(ax, [(48.256,109.3), (49.3,105.55), (48,102), (44.5,102)],
                  .40, 'bcr-fin')
    fin_extension(ax, [anchor(1728,180), anchor(1720,166), anchor(1705,152), anchor(1690,150),
                       (69.5,137.75), (63.5,124), (65.5,109),
                       (67,97.75), (70,92), (73,87)], .54, 'gc-fin')
    fin_extension(ax, [anchor(1733,835), (81.63,171.97), (82,175.5), (84.5,175.5),
                       (88,175.5), (86.5,160.5), (90,156)], .44, 'gradient-fin')


def bcr_input(ax):
    """Membrane receptor → per-cell H/L contigs → within-donor families."""
    txt(ax, 0, 75, 'scBCR-seq', 8, ACCENT, weight='bold')
    txt(ax, 0, 80.5, 'Heavy + light-chain sequences', 6.5, MUTED)
    ax.add_patch(Circle((7.5, 93), 4.8, facecolor='#EFF4F7', edgecolor='#7899AF', linewidth=.85))
    ax.add_patch(Ellipse((6.7, 94), 5.5, 5.2, angle=-25,
                        facecolor='#C6D6E1', edgecolor='none'))
    for degrees in [-100, -25, 50, 135]:
        theta=np.deg2rad(degrees)
        receptor(ax, 7.5+4.7*np.sin(theta), 93-4.7*np.cos(theta), degrees, .70)
    arrow(ax, (15, 91), (19, 91), color='#8DADBF', width=.65)
    for y, label, widths, letters in [
        (87.5, 'H', [8.5, 2.8, 4.2], ['V', 'D', 'J']),
        (95, 'L', [8.5, 4.2], ['V', 'J']),
    ]:
        txt(ax, 19, y, label, 5.9, ACCENT, ha='center', weight='bold')
        left=22.
        for i,(width, letter) in enumerate(zip(widths, letters)):
            color=['#385B73', '#91B0C2', '#61889F'][i]
            ax.add_patch(Rectangle((left, y-1.55), width, 3.1,
                                    facecolor=color, edgecolor='white', linewidth=.5))
            txt(ax,left+width/2,y,letter,5.3,'white',ha='center')
            left += width+.6
    ax.plot([28.5, 28.5, 38.7, 38.7], [89.7, 90.2, 90.2, 89.7], color=MUTED, linewidth=.5)
    txt(ax, 33.6, 92, 'junction', 4.7, MUTED, ha='center')
    txt(ax, 7.5, 102, 'B cell', 5.9, MUTED, ha='center')
    txt(ax, 29.5, 102, 'Paired by cell barcode', 5.7, MUTED, ha='center')
    receptor_trees(ax)
    txt(ax, 0, 118, 'Sequence families within donors', 6.0, MUTED)


def inputs(ax):
    txt(ax, 0, 4, 'Paired single cells', 8.6, weight='bold')
    txt(ax, 0, 13, 'scRNA-seq', 8, BLUE, weight='bold')
    txt(ax, 0, 18, 'Expression state', 6.5, MUTED)
    data=rna_data()
    for state,color in enumerate(STATE[:3]):
        q=data[data[:,2]==state]
        ax.scatter(q[:,0],q[:,1],s=3.4,color=color,marker='o',alpha=.84,
                   edgecolors='white',linewidths=.12,zorder=2)
    txt(ax, 0, 61, 'One dot = one cell', 6.2, MUTED)
    ax.plot([0, 42], [67, 67], color=RULE, linewidth=.7)
    bcr_input(ax)
    threadfin_fish(ax)


def cloud(rng, centre, scale, n, angle=0):
    xy = np.clip(rng.normal(size=(n, 2)), -2.15, 2.15)*scale
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return xy @ rotation.T + centre


def bridge(rng, controls, n, width):
    t = np.linspace(0, 1, n)
    c = np.asarray(controls)
    xy = ((1-t)**3)[:, None]*c[0] + (3*(1-t)**2*t)[:, None]*c[1]
    xy += (3*(1-t)*t*t)[:, None]*c[2] + (t**3)[:, None]*c[3]
    return xy + rng.normal(size=(n, 2))*width


def landscape_data():
    """Explicit conceptual data; no biological input or UMAP fitting."""
    rng = np.random.default_rng(ART_SEED)
    regions = [
        (cloud(rng, [82, 64], [4.5, 10.7], 115, -17), 0),
        (cloud(rng, [124, 65], [4.1, 5.1], 78, -25), 1),
        (cloud(rng, [166, 107], [6.8, 3.6], 102, 12), 2),
        (cloud(rng, [165, 39], [4.9, 3.2], 48, -25), 3),
        (bridge(rng, [[83, 79], [87, 102], [111, 91], [123, 68]], 64, 1.55), 0),
        (bridge(rng, [[127, 71], [135, 95], [154, 98], [165, 107]], 80, 1.50), 2),
        (bridge(rng, [[129, 59], [144, 50], [148, 44], [166, 39]], 48, 1.35), 3),
    ]
    chunks = []
    for xy, state in regions:
        count = rng.integers(2, 19, len(xy))
        # A visual scenario, NOT a biological rule that centrality implies size.
        central = np.exp(-.5*np.sum(((xy-[123, 75])/[18, 20])**2, axis=1))
        expanded = rng.random(len(xy)) < .24*central
        count += np.round(expanded*central*rng.integers(35, 110, len(xy))).astype(int)
        chunks.append(np.column_stack([xy, np.full(len(xy), state), count]))
    return np.concatenate(chunks)


def landscape(ax):
    txt(ax, 60, 4, 'Identify GC selection-associated nodes', 9.5, weight='bold')
    txt(ax, 60, 11, 'One point = one receptor family', 6.5, MUTED)
    for centre, wh, angle, color in [
        ((83, 64), (22, 47), -17, BLUE), ((124, 65), (26, 30), -25, GOLD),
        ((165, 106), (35, 18), 12, RED), ((165, 39), (23, 16), -25, TEAL),
    ]:
        is_selection=color==GOLD
        ax.add_patch(Ellipse(centre, *wh, angle=angle,
                            facecolor=to_rgba(color, .12 if is_selection else .045),
                            edgecolor=to_rgba(color,.85) if is_selection else 'none',
                            linewidth=1.05 if is_selection else 0,zorder=0))
    data = landscape_data()
    for state, color in enumerate(STATE):
        q = data[data[:, 2] == state]
        ax.scatter(q[:, 0], q[:, 1], s=DOT_AREA*q[:, 3], c=color, marker='o',
                   alpha=.86, edgecolors='white', linewidths=.25, zorder=3)
    # A hypothesis-level arrow between regions, never an ancestry edge
    # connecting different family dots.
    path(ax, [(121, 52), (115, 24), (70, 21), (76, 46)],
         BLUE, 1.25, (0, (4, 3)), arrow=True, z=4)
    txt(ax, 96, 19, 'Return to GC cycling?', 7.8, BLUE, weight='bold', ha='center')
    txt(ax, 96, 24.5, 'Candidate direction', 6.2, MUTED, ha='center')
    txt(ax, 158, 65, 'Selection-associated\nGC node', 8.3, '#A77A19', weight='bold', ha='center')
    arrow(ax,(140,65),(136,65),color='#A77A19',width=.95)
    txt(ax, 164, 24, 'Memory-like', 7.1, TEAL, weight='bold', ha='center')
    txt(ax, 80, 99, 'GC cycling', 7.7, BLUE, weight='bold', ha='center')
    txt(ax, 162, 121, 'Output-like states', 7.7, RED, weight='bold', ha='center')
    return data


def continuum_data():
    """Independent idealised linear map, inspired by the Fig. 3 question."""
    rng = np.random.default_rng(328)
    t = np.linspace(0, 1, 245)
    x = 65+114*t+rng.normal(0, 1.0, len(t))
    y = 159+5.0*np.sin(2*np.pi*(t-.1))+rng.normal(0, 1.65, len(t))
    central=np.exp(-.5*((t-.5)/.17)**2)
    count=rng.integers(2, 15, len(t))
    count += np.round((rng.random(len(t))<.25*central)*central*rng.integers(25,85,len(t))).astype(int)
    # Blend programme occupancy; no selection-node label is transferred here.
    return np.column_stack([x,y,t,count])


def continuum(ax):
    ax.plot([87, 183], [132, 132], color=RULE, linewidth=.7)
    txt(ax, 89, 139, 'Resolve cycling-to-output\nclone gradients', 7.8, weight='bold')
    txt(ax, 155, 138, 'Captured cells', 6.0, MUTED, ha='center')
    for x, n in [(144, 5), (155, 20), (169, 80)]:
        ax.scatter([x], [143], s=DOT_AREA*n, c=MUTED, marker='o',
                   edgecolors='white', linewidths=.3)
        txt(ax, x+2.5, 143, str(n), 5.6, MUTED)
    d=continuum_data()
    # Refit only the conceptual drawing's horizontal display extent, allowing
    # the approved fish to retain its natural aspect ratio and legible labels.
    display_x = 90 + (d[:,0]-65) * 89/114
    cmap=LinearSegmentedColormap.from_list('cycling_output',GRADIENT_COLORS)
    color=cmap(d[:,2])
    ax.scatter(display_x,d[:,1],s=DOT_AREA*d[:,3],c=color,marker='o',
               edgecolors='white',linewidths=.25,alpha=.98,zorder=3)
    txt(ax, 90, 174, 'Cycling', 6.5, GRADIENT_COLORS[0],weight='bold')
    txt(ax, 180, 174, 'Output-like', 6.5, GRADIENT_COLORS[-1],ha='right',weight='bold')


def draw(ax):
    ax.set_xlim(0, 183)
    ax.set_ylim(183, 0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    inputs(ax)
    data=landscape(ax)
    continuum(ax)
    return data


def manifest():
    data = landscape_data()
    return {
        'artwork': 'Vector scientific diagrams with an approved AI-generated raster fish; synthetic pseudo-UMAP, not an experimental result',
        'layout': 'Paired inputs and transparent Threadfin fish at left; functional GC-node and cycling/output-gradient maps at right',
        'primary_landscape_width_fraction': 123/183,
        'seed': ART_SEED,
        'background_families': len(data),
        'continuum_families': len(continuum_data()),
        'illustrative_rna_cells': len(rna_data()),
        'illustrative_rna_clusters': 3,
        'receptor_sequence_trees': 3,
        'receptor_tree_colours': FAMILY_COLORS,
        'state_order': NAMES,
        'size': 'Area proportional to synthetic captured-cell count for every clone point',
        'colour': 'Upper map: dominant captured RNA state. Lower map: vivid blue/teal/amber/rose continuous programme balance. Receptor-tree colours identify separate families, not RNA states.',
        'continuum_palette': GRADIENT_COLORS,
        'shape': 'All clone dots are circles; no timepoint marker shapes',
        'central_expansion': 'Some central profiles are larger by illustrative design only; empirical size uses captured cells, not map location',
        'dot_area_points_squared_per_cell': DOT_AREA,
        'bcr_input': 'Original membrane-Ig and H/L V(D)J-contig glyphs; grouping uses IGH with optional light-chain refinement',
        'receptor_tree_scope': 'Illustrative within-family sequence relationships; no common root across families, measured phylogeny or inferred output trajectory',
        'workflow_topology': {'input_sources':2,'shared_input_filaments':1,'output_filaments':2,
                              'workflow_arrowheads':0,'integration_glyph':'approved threadfin fish cartoon PNG'},
        'fin_extensions': 'Four tapered vector ribbons with pale gold/blue-green shading; overlapping pixel anchors and tangent-matched cubic joins extend the approved illustration organically',
        'fish_asset': {'path':'paper/figure_plan/assets/threadfin_fish_cartoon.png',
                       'sha256':hashlib.sha256(FISH_ASSET.read_bytes()).hexdigest(),
                       'placement_left_top_width_mm':FISH_PLACEMENT,
                       'provenance':'Built-in image generation from the user-supplied fish reference; user approved',
                       'rendering':'Original RGBA pixels embedded at natural aspect ratio; vector fin extensions overlap the existing tips; source PNG unchanged'},
        'fish_reference': 'User-supplied 9df853968acdacd99d25b52a2e3fd48f.png',
        'algorithm_glyph': 'The unshrunk mean of RNA kernel feature vectors is a squared-distance Frechet mean in feature space; final profiles also use context adjustment, shrinkage and projection. No optimal-transport solver is claimed.',
        'selection_emphasis': 'Gold outline and aligned annotation identify a conceptual node; the outline is not a confidence region',
        'visual_references': VISUAL_REFERENCES,
        'asset_reuse': 'Approved generated fish PNG embedded; no article artwork embedded',
        'continuum_interpretation': 'Idealised GC/cycling to output-associated profile variation; no imposed selection node or inferred temporal trajectory',
        'default_profile': 'Context-centred and shrunken kernel mean embedding; mean option remains available',
        'dashed_arrow': 'Candidate selection-to-cycling return; requires directional lineage/time evidence',
        'biological_scope': 'GC-specific hypothesis; not memory-to-GC re-entry, demonstrated fate or between-family ancestry',
        'study_scope': 'Fig. 2 model-antigen selection context and Fig. 3 infection population / output-gradient questions; studies not pooled',
        'implementation_anchors': ['threadfin/api.py:run', 'threadfin/profiles.py:build_model',
                                   'threadfin/profiles.py:clone_profiles', 'threadfin/programmes.py:find_programmes'],
    }
