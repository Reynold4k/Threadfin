#!/usr/bin/env python3
"""Figure 4 in the typography and graphical style of current Figures 1–3.

Uses frozen coordinates and verified statistics. No embedding/model refit.
--preview writes only the versioned preview, source copies and layout audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[2]
CASE = ROOT / 'Threadfin/manuscript/case_studies/results'
SR = ROOT / 'story_review_2026-10-12'
# non-B-cell comparison results moved out of the package repository on 2026-10-18
LOCAL = ROOT / 'internal_validation/repo_validation_2026-10-18/case_studies/results'
WORK = HERE / 'reserve/figure4_visual_upgrade_2026-10-17'
STEM = 'Figure_4_measured_validation'
INK = '#0b0b0b'
MUTED = '#52514e'
AXIS = '#898781'
MISSING = '#e1e0d9'
BLUE, RED, VIOLET, STEEL = '#2a78d6', '#e34948', '#4a3aa7', '#56778b'
TARGETS = [('Neutrophil', RED), ('Monocyte', VIOLET)]
INPUTS = [CASE/'mouse_rbd/clone_table.csv', CASE/'mouse_rbd/label_effects.csv',
          CASE/'mouse_rbd/summary.json', LOCAL/'clonotrace_larry/larry_day2_clonemap_fate.csv.gz',
          LOCAL/'clonotrace_larry/future_audit.json', SR/'LARRY_OOF_barcode_bootstrap.csv',
          SR/'LARRY_paired_method_differences.csv', SR/'LARRY_sister_well_points.csv',
          SR/'LARRY_sister_well_validation.csv', LOCAL/'watermelon/early_clone_table.csv',
          LOCAL/'watermelon/summary.json']


def style():
    # Same font, black text, axis sizes and categorical palette as Figures 2–3.
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.6,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': AXIS,
        'axes.linewidth': .55, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 2,
        'ytick.major.size': 2, 'xtick.major.width': .5, 'ytick.major.width': .5,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'threadfin-figure4-visual-20261017',
        'savefig.facecolor': 'white'})


def text(fig, x, y, s, size=8, **kwargs):
    return fig.text(x/210, 1-y/297, s, fontsize=size, va='top', linespacing=1.15, **kwargs)


def panel(fig, x, y, w, h, letter, title, *, label_x=None, title_y=None,
          subtitle=None, title_size=9):
    ax = fig.add_axes([x/210, 1-(y+h)/297, w/210, h/297])
    ty = y-14 if title_y is None else title_y
    text(fig, x-5 if label_x is None else label_x, ty-.2, letter, 11, weight='bold')
    text(fig, x+w/2, ty, title, title_size, ha='center')
    if subtitle:
        text(fig, x+w/2, ty+6, subtitle, 7.6, ha='center', color=MUTED)
    return ax


def horizontal_fraction_bar(fig, artist, x, y, w, label, *, label_y=None):
    cax = fig.add_axes([x/210, 1-(y+2.2)/297, w/210, 2.2/297])
    cb = fig.colorbar(artist, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.outline.set_linewidth(.45)
    cb.ax.tick_params(labelsize=7.5, pad=1.5, length=1.5)
    text(fig, x+w/2, y-5 if label_y is None else label_y, label, 7.8, ha='center')
    return cb


def clone_map(ax, table, column, *, size=18, size_by_count=False):
    valid = pd.to_numeric(table[column], errors='coerce').notna()
    ax.scatter(table.loc[~valid, 'x'], table.loc[~valid, 'y'], s=5,
               c=MISSING, linewidths=0, rasterized=True)
    z = table.loc[valid].sample(frac=1, random_state=12)
    areas = np.clip(z.n_cells*1.5, 4, 35) if size_by_count else size
    sc = ax.scatter(z.x, z.y, c=z[column], s=areas, cmap='viridis', vmin=0, vmax=1,
                    linewidths=0, alpha=.95, rasterized=True)
    ax.margins(.045)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Clone UMAP 1', labelpad=2)
    ax.set_ylabel('Clone UMAP 2', labelpad=2)
    return sc, int(valid.sum())


def watermelon_design(ax):
    """Measured sampling times and exact eligibility intersection, not a fitted trajectory."""
    ax.set_axis_off(); ax.set_xlim(-.05, 1.05); ax.set_ylim(0, 1)
    xs = np.asarray([0, 3, 7, 14])/14
    ax.annotate('', xy=(1.04, .84), xytext=(-.02, .84),
                arrowprops={'arrowstyle': '->', 'color': AXIS, 'lw': .8})
    for day, x in zip([0, 3, 7, 14], xs):
        ax.scatter([x], [.84], s=22, c=BLUE if day<14 else STEEL, zorder=3,
                   edgecolors='white', linewidths=.6)
        ax.text(x, .93, f'Day {day}' if day in [0, 14] else str(day),
                ha='center', va='center', fontsize=7.6)
    # Early time points form one RNA input; day 14 contributes measured gates.
    ax.plot([0, 0, .5, .5], [.78, .73, .73, .78], color=BLUE, lw=.8)
    ax.plot([.25, .25], [.73, .67], color=BLUE, lw=.8)
    ax.plot([1, 1, .86], [.78, .73, .67], color=STEEL, lw=.8)
    ax.text(.22, .59, 'Early RNA', ha='center', fontsize=8.2, color=BLUE)
    ax.text(.86, .59, 'Later gates', ha='center', fontsize=8.2, color=STEEL)
    ax.text(.22, .49, '2,089 profiles', ha='center', fontsize=7.6)
    ax.text(.86, .49, '1,466 barcodes', ha='center', fontsize=7.6)
    for x in [.22, .86]:
        ax.add_patch(FancyArrowPatch((x, .41), (.54, .25), arrowstyle='-',
                                    color=AXIS, mutation_scale=6, linewidth=.8))
    ax.text(.54, .19, '1,016 shared', ha='center', va='center', fontsize=9.2, weight='bold')
    ax.text(.54, .07, '8,894 early cells', ha='center', va='center', fontsize=7.8)


def build_figure():
    style()
    fig = plt.figure(figsize=(210/25.4, 297/25.4))
    rbd = pd.read_csv(INPUTS[0]); effects = pd.read_csv(INPUTS[1]).set_index('label')
    lmap = pd.read_csv(INPUTS[3]); oof = pd.read_csv(INPUTS[5]); delta = pd.read_csv(INPUTS[6])
    wells = pd.read_csv(INPUTS[7]); ws = pd.read_csv(INPUTS[8])
    wm = pd.read_csv(INPUTS[9]); wms = json.loads(INPUTS[10].read_text())
    metric_rows = []; map_counts = {}

    # a–d: three parallel measured readouts, one shared fraction scale.
    specs = [
        ('a', 10, 'RBD | Recent division', 'division_gate:mCherry-low', '1,887 clones · 10 mice', '36 h reporter'),
        ('b', 61, 'RBD-binding fraction', 'rbd_bait:RBD+', '1,544 clones · 5 mice', 'Protein cohort'),
        ('c', 112, 'Dark-zone fraction', 'zone_gate:DZ', '343 clones · 5 mice', 'mRNA cohort')]
    for letter, x, title, col, sub, note in specs:
        ax = panel(fig, x, 22, 42, 50, letter, title, subtitle=sub)
        sc, n = clone_map(ax, rbd, col, size_by_count=True)
        map_counts[letter] = n
        text(fig, x+21, 79, note, 7.6, ha='center', color=MUTED)
    horizontal_fraction_bar(fig, sc, 35, 90, 86, 'Fraction of captured clone members', label_y=85)
    fig.legend(handles=[Line2D([], [], marker='o', color=MISSING, ls='', markersize=4,
                              label='Not measured')], loc='upper left', bbox_to_anchor=(132/210, 1-89/297),
               fontsize=7.6, handlelength=.7, handletextpad=.4, borderaxespad=0)
    ax = panel(fig, 172, 22, 30, 50, 'd', 'Readout association', label_x=157,
               subtitle='Above shuffled labels')
    keys = [s[3] for s in specs]; vals = effects.loc[keys, 'excess_r2'].to_numpy()
    ax.bar(range(3), vals, width=.65, color=STEEL)
    ax.set_xticks(range(3), ['Div.', 'RBD+', 'DZ'])
    ax.set_ylim(0, .105); ax.set_yticks([0, .05, .1])
    ax.set_ylabel('Excess association R²', labelpad=2)
    for i, (key, val) in enumerate(zip(keys, vals)):
        ax.text(i, val+.003, f'{val:.3f}', ha='center', fontsize=7.8)
        metric_rows.append({'panel': 'd', 'readout': key, 'value': float(val),
                            'p_value': float(effects.loc[key, 'p_value'])})
    text(fig, 187, 79, 'All P = 0.00050', 7.6, ha='center')

    # e–h: early profiles, complete OOF comparison, paired increments, sister wells.
    ax = panel(fig, 10, 116, 50, 63, 'e', 'LARRY | Early profiles',
               subtitle='Day 2 · 172 barcodes')
    sc, n = clone_map(ax, lmap, 'fate', size=21)
    map_counts['e'] = n
    horizontal_fraction_bar(fig, sc, 14, 191, 42, 'Day-6 neutrophil fraction', label_y=186)

    ax = panel(fig, 93, 116, 58, 63, 'f', 'Day-6 prediction', label_x=66,
               subtitle='3 × 5 barcode-held-out folds')
    methods = [('Training_mean', 'Training mean'), ('Size_only', 'Clone size'),
               ('RNA_mean', 'RNA mean'), ('RNA_mean_variance', 'RNA mean + variance'),
               ('Threadfin_mean', 'Threadfin mean'), ('Threadfin_kernel', 'Threadfin kernel')]
    for j, (target, color) in enumerate(TARGETS):
        sub = oof[oof.target.eq(target)].set_index('method')
        for i, (method, label) in enumerate(methods):
            z = sub.loc[method]; y = 5-i+(.13 if j==0 else -.13)
            ax.errorbar(z.repeated_OOF_R2, y,
                        xerr=[[z.repeated_OOF_R2-z.boot_low], [z.boot_high-z.repeated_OOF_R2]],
                        fmt='D' if method=='Threadfin_kernel' else 'o', ms=4.1,
                        color=color, elinewidth=.9, capsize=2, markeredgewidth=.9)
            metric_rows.append({'panel': 'f', 'target': target, 'method': method,
                                'value': float(z.repeated_OOF_R2), 'low': float(z.boot_low), 'high': float(z.boot_high)})
    ax.axvline(0, color=AXIS, lw=.65, ls=':')
    ax.set_yticks(range(5, -1, -1), [m[1] for m in methods], fontsize=7.7)
    ax.set_xlim(-.075, .34); ax.set_xticks([0, .1, .2, .3]); ax.set_ylim(-.5, 5.5)
    ax.set_xlabel('Out-of-fold R²', labelpad=2)
    handles = [Line2D([], [], color=c, marker='o', linestyle='', markersize=4, label=t) for t,c in TARGETS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(77/210, 1-191/297),
               ncol=2, handlelength=.7, handletextpad=.4, columnspacing=.9, borderaxespad=0, fontsize=7.8)
    text(fig, 122, 198, 'Points and 95% CIs', 7.6, ha='center', color=MUTED)

    ax = panel(fig, 169, 116, 33, 23, 'g', 'Kernel − RNA mean', label_x=157,
               subtitle='Paired difference')
    for j, (target, color) in enumerate(TARGETS):
        z = delta[(delta.target==target) & (delta.baseline=='RNA_mean')].iloc[0]
        ax.errorbar(z.delta_R2_kernel_minus_baseline, 1-j,
                    xerr=[[z.delta_R2_kernel_minus_baseline-z.boot_low],
                          [z.boot_high-z.delta_R2_kernel_minus_baseline]],
                    fmt='o', ms=4.3, color=color, elinewidth=1, capsize=2)
        metric_rows.append({'panel': 'g', 'target': target, 'value': float(z.delta_R2_kernel_minus_baseline),
                            'low': float(z.boot_low), 'high': float(z.boot_high)})
    ax.axvline(0, color=AXIS, lw=.8, ls='--')
    ax.set_yticks([1, 0], ['Neu.', 'Mono.'])
    ax.set_xlim(-.03, .1); ax.set_xticks([0, .05, .1]); ax.set_ylim(-.55, 1.55)
    ax.set_xlabel('ΔR²', labelpad=1)

    ax = panel(fig, 169, 165, 33, 29, 'h', 'Across sister wells', label_x=157,
               title_y=152, subtitle='211 shared barcodes')
    for target, color in TARGETS:
        z = wells[wells.target.eq(target)]
        ax.scatter(z.well1, z.well2, c=color, s=6.5, alpha=.55, linewidths=0, rasterized=True)
    ax.plot([0,1], [0,1], color=AXIS, lw=.65, ls=':')
    ax.set_xlim(-.04, 1.04); ax.set_ylim(-.04, 1.04)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel('Fraction in well 1', labelpad=1)
    ax.set_ylabel('Well 2', labelpad=1)
    rho = ws.set_index('target').rho
    text(fig, 185.5, 205, f'ρ = {rho["Neutrophil"]:.2f} / {rho["Monocyte"]:.2f}', 7.6, ha='center')

    # i–k: visual experimental design replaces the previous prose block.
    ax = panel(fig, 10, 230, 49, 55, 'i', 'WATERMELON | PC9',
               title_y=216, subtitle='Osimertinib treatment')
    watermelon_design(ax)
    ax = panel(fig, 75, 231, 54, 51, 'j', 'Early clone profiles',
               title_y=216, subtitle='Days 0–7 · 1,016 clones')
    wm_used = wm.dropna(subset=['gate14:mCherry-high'])
    sc, n = clone_map(ax, wm_used, 'gate14:mCherry-high', size=9)
    map_counts['j'] = n
    cax = fig.add_axes([132/210, 1-282/297, 2.1/210, 44/297])
    cb = fig.colorbar(sc, cax=cax, ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=7.5, pad=1, length=1.5); cb.outline.set_linewidth(.45)
    text(fig, 102, 291.5, 'Day-14 non-cycling fraction', 7.6, ha='center')

    ax = panel(fig, 156, 231, 46, 51, 'k', 'Day-14 gate association', label_x=145, title_y=216)
    for offset, label, key, color in [(-.18, 'Shrunken mean', 'label_effects_mean_profiles', BLUE),
                                      (.18, 'Kernel', 'label_effects', RED)]:
        vals = [wms['early_to_late'][key]['gate14:'+g]['excess_r2'] for g in ['mCherry-high', 'mCherry-low']]
        ax.bar(np.arange(2)+offset, vals, width=.32, color=color, label=label)
        for x, val, gate in zip(np.arange(2)+offset, vals, ['mCherry-high', 'mCherry-low']):
            ax.text(x, val+.002, f'{val:.3f}', ha='center', fontsize=7.8)
            metric_rows.append({'panel': 'k', 'method': label, 'readout': gate, 'value': float(val)})
    ax.set_xticks([0,1], ['Non-cycling\n(high)', 'Cycling\n(low)'], fontsize=7.8)
    ax.set_ylabel('Excess association R²', labelpad=2)
    ax.set_ylim(0, .09); ax.set_yticks([0, .04, .08])
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=2,
              fontsize=7.5, handlelength=.8, handletextpad=.4, columnspacing=.8,
              borderpad=0, borderaxespad=0)
    # All metrics remain in a machine-readable visual-data table.
    fig.threadfin_visual_metrics = metric_rows
    fig.threadfin_map_counts = map_counts
    assert map_counts == {'a':1887, 'b':1544, 'c':343, 'e':172, 'j':1016}, map_counts
    return fig


def audit(fig):
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); w,h=fig.canvas.get_width_height()
    records=[]; outside=[]
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip(): continue
        box=artist.get_window_extent(ren)
        if not np.isfinite(box.extents).all(): continue
        item={'text':artist.get_text(), 'font_pt':artist.get_fontsize(),
              'bbox_px':[float(v) for v in box.extents]}
        records.append(item)
        if box.x0<0 or box.y0<0 or box.x1>w or box.y1>h: outside.append(item)
    overlaps=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            aa=a['bbox_px']; bb=b['bbox_px']
            dx=min(aa[2],bb[2])-max(aa[0],bb[0]); dy=min(aa[3],bb[3])-max(aa[1],bb[1])
            if dx>2 and dy>2: overlaps.append([a['text'],b['text'],round(dx*dy,2)])
    return {'page_mm':[210,297], 'text_outside_page':outside, 'potential_text_overlaps':overlaps,
            'minimum_font_pt':min(x['font_pt'] for x in records),
            'word_count':sum(len(re.findall(r'\S+', x['text'])) for x in records), 'text_records':records}


def render(preview=False, dpi=600):
    WORK.mkdir(parents=True, exist_ok=True); (WORK/'source_data').mkdir(exist_ok=True)
    fig=build_figure(); report=audit(fig)
    (WORK/'layout_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    fig.savefig(WORK/'preview.png', dpi=180)
    pd.DataFrame(fig.threadfin_visual_metrics).to_csv(WORK/'source_data/plotted_statistics.csv', index=False)
    (WORK/'source_data/displayed_map_counts.json').write_text(json.dumps(fig.threadfin_map_counts, indent=2)+'\n')
    if not preview:
        assert not report['text_outside_page'], report['text_outside_page']
        assert not report['potential_text_overlaps'], report['potential_text_overlaps']
        for ext in ['pdf','svg','png']: fig.savefig(HERE/f'{STEM}.{ext}', dpi=dpi)
        fig.savefig(HERE/'reserve/current_previews'/f'{STEM}.png', dpi=170)
    for p in INPUTS:
        (WORK/'source_data'/f'{p.parent.name}__{p.name}').write_bytes(p.read_bytes())
    prov={'generator':str(Path(__file__).relative_to(HERE)), 'page_mm':[210,297],
          'revision':'2026-10-17 visual alignment with Figures 1–3', 'no_upstream_refit':True,
          'point_coordinates':'Unchanged saved x/y values; panel scaling and deterministic draw order only. No jitter.',
          'statistics':'Frozen verified statistics; all six LARRY methods and paired confidence intervals retained.',
          'ci':'Historical conditional barcode bootstrap of fixed OOF predictions; no model-refit uncertainty.',
          'inputs':[{'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in INPUTS],
          'changes':['Same 9 pt titles, 8 pt axes and black text as Figures 2–3',
                     'Removed full-width banners and explanatory prose from the plate; details retained in legend',
                     'One viridis 0–1 scale for all measured fractions; missing remains grey',
                     'Larger plot regions and increased minimum font size',
                     'New i: actual WATERMELON sampling timeline and eligibility intersection; old i/j become j/k',
                     'All outcome comparisons and conditional paired confidence intervals retained'],
          'scientific_limits_retained_in_legend':['RBD binding and zone are separate mouse cohorts',
                     'LARRY early preprocessing is shared; paired gain intervals include zero',
                     'Sister-well set overlaps prediction set in 14 barcodes from one culture experiment',
                     'WATERMELON is retrospective association among captured survivors, not survival prediction',
                     'WATERMELON mean has larger association statistics; no universal kernel advantage'],
          'map_counts':fig.threadfin_map_counts, 'layout_audit':str((WORK/'layout_audit.json').relative_to(HERE))}
    (WORK/'provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2)+'\n')
    if not preview:
        (HERE/'reserve/provenance'/f'{STEM}.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2)+'\n')
    plt.close(fig)
    print(json.dumps({'preview':str(WORK/'preview.png'), 'exported':not preview,
                      'minimum_font_pt':report['minimum_font_pt'], 'word_count':report['word_count'],
                      'outside':len(report['text_outside_page']), 'overlaps':report['potential_text_overlaps']}, ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true'); parser.add_argument('--dpi', type=int, default=600)
    args=parser.parse_args(); render(args.preview, args.dpi)

if __name__=='__main__': main()
