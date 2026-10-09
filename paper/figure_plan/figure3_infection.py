"""Figure 3: a central late-infection map with donor-aware interpretation.

Consumes saved Threadfin outputs; does not infer a temporal trajectory or
alter the selected clone coordinates. Source tables are refreshed with the
normal figure-generation workflow.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from matplotlib.patches import Circle, FancyBboxPatch

from experimental_designs import filled_mouse, spleen, capture


def compact_design(ax, b):
    """Compact vector schematic preserving the two user-specified timelines."""
    ax.set(xlim=(0, 183), ylim=(0, 38))
    ax.set_axis_off()
    muted = '#64717c'

    def text(x, y, s, size=6, **kw):
        return ax.text(x, y, s, fontsize=size, color=kw.pop('color', b.INK),
                       va='center', **kw)

    # Shared capture workflow: icons have room to breathe, with one baseline.
    text(0, 35, 'PcAS infection', 6.8, weight='bold')
    filled_mouse(ax, 10, 24, 1.03)
    spleen(ax, 32, 24, 1.05)
    capture(ax, 54, 24, .83)
    for start, end in [(18, 25), (39, 46)]:
        ax.annotate('', xy=(end, 24), xytext=(start, 24),
                    arrowprops=dict(arrowstyle='->', color=muted, lw=.8,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
    text(10, 15.5, 'Infected mouse', 5.8, ha='center')
    text(32, 15.5, 'B cells', 5.8, ha='center')
    text(54, 15.5, 'RNA · VDJ · HTO', 5.5, ha='center')
    text(0, 7.2, 'Terminal spleen sampling', 6, weight='bold')
    text(0, 3.5, 'Different mice at each date', 5.7, color=muted)
    ax.plot([65, 65], [1.3, 37], color='#e0e6eb', lw=.65)

    # Matched, lightly tinted cohort cards: colour is an accent, not a frame.
    def card(y0, height, color, tint):
        ax.add_patch(FancyBboxPatch(
            (70, y0), 112.5, height,
            boxstyle='round,pad=0,rounding_size=1.25',
            facecolor=tint, edgecolor='#dde5ea', lw=.45, zorder=0))
        ax.plot([70.2, 70.2], [y0+1.3, y0+height-1.3],
                color=color, lw=1.8, solid_capstyle='round', zorder=1)

    def timeline(y, day_start, day_end, days, color, counts=(), naive=()):
        x0, x1 = 95, 178.5
        px = lambda day: x0 + (x1-x0)*(day-day_start)/(day_end-day_start)
        ax.plot([x0, x1], [y, y], color='#a8b5bf', lw=.65, zorder=1)
        for i, day in enumerate(days):
            control = day in naive
            dot_color = '#a9b3be' if control else color
            ax.add_patch(Circle((px(day), y), .77, facecolor=dot_color,
                                edgecolor='white', lw=.45, zorder=3))
            text(px(day), y-2.3, f'd{day}', 5.3, ha='center',
                 color=muted if control else b.INK)
            if counts:
                text(px(day), y+2.5, counts[i], 5.6, ha='center',
                     weight='bold', color=dot_color)
        return px

    card(20.5, 17.2, b.BLUE, '#f4f8fc')
    text(74, 35.3, 'Experiment 1 · early infection', 6.3,
         weight='bold', color=b.BLUE)
    text(179, 35.3, '40,623 B cells', 5.5, ha='right', color=muted)
    text(74, 31.7, 'Mice', 5.2, color=muted)
    timeline(29.2, 0, 14, [0, 4, 7, 10, 14], b.BLUE,
             ['1', '4', '5', '5', '5'], naive=(0,))
    text(74, 23.8, 'Broad B-cell sort · TCRβ− CD19+ B220int–hi', 5.3)
    text(74, 21.5, 'TotalSeq-C pools; d0: one naive mouse in the d4 pool', 5.2,
         color=muted)

    card(.5, 18.2, b.GREEN, '#f2f8f6')
    text(74, 16.3, 'Experiment 2 · late infection + treatment', 6.3,
         weight='bold', color=b.GREEN)
    text(179, 16.3, '72,359 B cells', 5.5, ha='right', color=muted)
    text(74, 13.1, 'Anti-malarials from d7: artesunate + pyrimethamine', 5.4,
         color=b.RED)
    px = timeline(8.1, 7, 42, [10, 14, 21, 28, 35, 42], b.GREEN)
    ax.plot([px(7), px(42)], [10.9, 10.9], color=b.RED, lw=1.9,
            alpha=.6, solid_capstyle='round')
    ax.plot([px(7), px(7)], [8.1, 10.9], color=b.RED, lw=.6,
            linestyle=(0, (1.2, 1.5)), alpha=.75)
    text(74, 2.4, '3 saline + 3 drug + 1 naive/day · IgD-low sort + IgD-high spike-in',
         5.3)


def source_tables(b):
    out = b.DATA / 'figure23_review'
    out.mkdir(exist_ok=True)
    early = b.clones('malaria').join(pd.read_csv(b.DATA / 'malaria/clone_gene_scores.csv', index_col=0))
    early = early[early.reliability.ge(.5)].dropna(subset=['x', 'y'])
    gc = early[early['cell_state:GC'].ge(.5)]
    pure = early[early['cell_state:PB'].eq(1)].copy()
    pure['distance_to_gc_reference'] = np.hypot(pure.x-gc.x.mean(), pure.y-gc.y.mean())
    pure['day'] = pure.donor.astype(str).str.extract(r'D(\d+)_').astype(int)
    correlations = []
    for donor, q in pure.groupby('donor', sort=True):
        r = spearmanr(q.distance_to_gc_reference, q['dark zone / cycling']).statistic if len(q) >= 5 else np.nan
        correlations.append({'donor': donor, 'day': int(q.day.iloc[0]), 'n_families': len(q),
                             'rho_distance_cycling': r, 'eligible_for_direction_summary': len(q) >= 5})
    corr = pd.DataFrame(correlations)
    pure.to_csv(out / 'figure3E_pure_pb_families.csv')
    corr.to_csv(out / 'figure3E_within_mouse_correlations.csv', index=False)

    late = b.clones('malaria_late')
    late = late[late.reliability.ge(.5)].copy()
    late['day'] = late.donor.astype(str).str.extract(r'D(\d+)').astype(int)
    late['gc_enriched'] = late['cell_state:GC'].ge(.5)
    infected = late[late.treatment.isin(['Saline', 'Artesunate'])]
    donor = infected.groupby(['donor', 'day', 'treatment'], sort=True).agg(
        n_reliable_families=('n_cells', 'size'), n_gc_enriched=('gc_enriched', 'sum'),
        fraction_gc_enriched=('gc_enriched', 'mean')).reset_index()
    donor.to_csv(out / 'figure3F_mouse_gc_fractions.csv', index=False)
    eligible = corr[corr.eligible_for_direction_summary]
    audit = {'pure_pb_families': len(pure), 'pure_pb_mice': int(pure.donor.nunique()),
             'within_mouse_min_families': 5, 'within_mouse_eligible_mice': len(eligible),
             'within_mouse_negative_correlations': int(eligible.rho_distance_cycling.lt(0).sum()),
             'median_within_mouse_rho': float(eligible.rho_distance_cycling.median()),
             'gc_reference_families': len(gc), 'gc_reference_rule': 'reliable early families with GC fraction >=0.5; mean saved 2D coordinates',
             'panel_E_limit': 'Descriptive RNA-derived association in a 2D display; not independent validation of maturation or GC origin',
             'panel_F_mice': len(donor), 'panel_F_reliable_families': len(infected),
             'panel_F_denominator': 'all reliable families within each mouse, separately by treatment',
             'panel_F_gc_rule': 'captured GC fraction >=0.5; no UMAP-distance classification',
             'panel_F_limit': 'Different terminal mice at each day; changing captured repertoires, not tracked families or causal treatment effects'}
    (out / 'figure3_interpretation.json').write_text(json.dumps(audit, indent=2)+'\n')
    return pure, corr, donor, audit


def draw(fig, b):
    pure, corr, donor, audit = source_tables(b)
    a = b.p(fig, 0, 5, 183, 38, 'A', 'Two infection cohorts with different sampling designs')
    compact_design(a, b)

    early = b.p(fig, 0, 53, 85, 61, 'B', 'Early infection: output-biased family profiles')
    b.color_map(early, 'malaria', 'cell_state:PB', 'Fraction PB cells', note_y=-.28)
    cell = b.p(fig, 111, 53, 72, 61, 'D', 'Captured cell states in later infection')
    b.cell_map(cell, 'malaria_late', 'cell_state', {'GC': b.BLUE, 'PB': b.RED, 'Memory': b.GREEN})
    b.legend(cell, {'GC': b.BLUE, 'PB': b.RED, 'Memory-like': b.GREEN, 'Other': b.GREY}, ncol=2, y=-.15)

    selected = json.loads((b.DATA / 'malaria_late_embedding_audit/selected.json').read_text())
    map_path = b.DATA / 'malaria_late_embedding_audit/maps' / (selected['name']+'.csv')
    coords = pd.read_csv(map_path, index_col=0)
    # C is centred on the page and has the largest plotting area.
    hero = b.p(fig, 21.5, 139, 140, 88, 'C', 'Later infection: GC, PB and memory-associated family states')
    q = b.color_map(hero, 'malaria_late', 'cell_state:GC', 'Fraction GC cells', coords=coords, note_y=-.17)
    assert coords.index.is_unique and set(q.index) == set(coords.index)
    assert coords.reindex(q.index)[['x', 'y']].notna().all().all()
    c = b.cells('malaria_late')
    memory = pd.crosstab(c.clone_id, c.cell_state, normalize='index')['Memory']
    labelled = q.assign(Memory=memory.reindex(q.index), xm=q.x, ym=q.y)
    b.arm_labels(hero, labelled, 'xm', 'ym', 'Memory',
                 pos={'GC arm': (-1.2, 9.8), 'PB arm': (16.3, 4.1), 'Memory arm': (-1.2, .1)})
    for label in hero.texts:
        if label.get_text() in {'GC arm', 'PB arm', 'Memory arm'}:
            label.set_text(label.get_text().replace(' arm', '-enriched'))
    b.note(hero, 'Neighbourhoods reflect captured profile similarity; region labels describe state enrichment.', -.225, 5.5)

    e = b.p(fig, 9, 259, 79, 47, 'E', 'Pure-PB families retain\ncycling-state variation')
    for day, color in [(7, b.GREY), (10, b.GOLD), (14, b.BLUE)]:
        z = pure[pure.day.eq(day)]
        e.scatter(z.distance_to_gc_reference, z['dark zone / cycling'], s=7.5,
                  color=color, alpha=.75, linewidths=0, label=f'Day {day} (n={len(z)})')
    e.set_xlabel('Distance to GC-enriched reference (2D)', fontsize=5.8)
    e.set_ylabel('Cycling module score', fontsize=6)
    e.legend(frameon=False, fontsize=5.4, loc='upper right')
    b.note(e, f"{len(pure)} pure-PB families; RNA-derived description.\n"
           f"Within-mouse direction: {audit['within_mouse_negative_correlations']}/{audit['within_mouse_eligible_mice']} negative "
           f"(≥5 families/mouse; median ρ={audit['median_within_mouse_rho']:.2f}).", -.33, 5.5)

    f = b.p(fig, 112, 259, 71, 47, 'F', 'GC-enriched families are captured\nat late infection timepoints')
    days = [10, 14, 21, 28, 35, 42]
    for treatment, label, color, dx in [('Saline', 'Saline', b.BLUE, -.10),
                                       ('Artesunate', 'Anti-malarial', b.GOLD, .10)]:
        z = donor[donor.treatment.eq(treatment)]
        medians = []
        for i, day in enumerate(days):
            v = z[z.day.eq(day)].sort_values('donor').fraction_gc_enriched
            f.scatter(i+dx+np.linspace(-.035, .035, len(v)), v, s=10,
                      color=color, alpha=.85, linewidths=.2, edgecolors='white', zorder=3)
            medians.append(v.median())
        f.plot(np.arange(len(days))+dx, medians, color=color, lw=.85, label=label)
    f.set(xticks=range(len(days)), xticklabels=days, ylim=(-.035, 1.04))
    f.set_xlabel('Day after infection', fontsize=6)
    f.set_ylabel('GC-enriched / reliable families per mouse', fontsize=5.8)
    f.legend(frameon=False, fontsize=5.5, loc='upper left')
    b.note(f, 'Each dot is one mouse; lines join group medians.\nGC-enriched: ≥50% GC cells. Three mice/arm/day;\nindependent terminal samples, not tracked clones.', -.33, 5.5)

    b.AUDIT['sources'][str(map_path.relative_to(b.ROOT))] = len(coords)
    b.AUDIT['figure3_review'] = {**audit, 'selected_map': str(map_path.relative_to(b.ROOT)),
        'coordinates_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
        'map_parameters': selected['parameters'],
        'layout_mm': {'A': [0,5,183,38], 'B': [0,53,85,61], 'C': [21.5,139,140,88],
                      'D': [111,53,72,61], 'E': [9,259,79,47], 'F': [112,259,71,47]},
        'source_tables': 'case_studies/results/figure23_review/'}
