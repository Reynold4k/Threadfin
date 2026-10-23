#!/usr/bin/env python
"""Assemble the manuscript's four-figure sequence from existing frozen analyses."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import textwrap
from pathlib import Path
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import fig_clone_states_combined_v6 as v6
import fig2_gc_overview as overview
import fig3_measured_biology as measured

HERE = Path(__file__).resolve().parents[2]
fs, ref, base = v6.fs, v6.ref, v6.base
ARCHIVE = HERE / 'reserve'
SUPP = ARCHIVE / 'supplementary'
PROV = ARCHIVE / 'provenance'
PREVIEWS = ARCHIVE / 'current_previews'


def previous(name):
    path = PROV / name
    if not path.exists():
        path = HERE / name
    return json.loads(path.read_text())


def prepare():
    datasets, w, _, paths, _ = ref.load_data()
    prior = previous('figure_clone_states_stats_v4.json')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes == prior['input_sha256']
    check = v6.validate_ghi(datasets['S2'], w, prior['panels'])
    display, sort_metadata = v6.load_sort(datasets['S3'], v6.METADATA)
    base.ax_at, base.left_key, base.shared_key = v6.ax_at, v6.left_key, v6.shared_key
    return datasets, w, prior, display, sort_metadata, hashes, check


def save(fig, stem, args, folder=HERE):
    folder.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    preview = PREVIEWS / f'{stem}.png'
    fig.savefig(preview, dpi=200)
    print('Preview:', preview, flush=True)
    if not args.preview:
        fs.save(fig, stem, folder=folder, dpi=args.dpi)
    else:
        plt.close(fig)


def remove_axes(fig, keep):
    for ax in list(fig.axes):
        if ax not in keep:
            ax.remove()
    ref.LEGEND_TEXTS[:] = [t for t in ref.LEGEND_TEXTS if t.axes in fig.axes]


def shift_up(fig, mm):
    for ax in fig.axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + mm / fig.page_height, box.width, box.height])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, y + mm / fig.page_height))
    for boxes in [fig.panel_boxes_mm, fig.legend_boxes_mm]:
        for box in boxes.values():
            box[1] -= mm


def memory_upper(fig, datasets, w, prior, report):
    base.main_upper(fig, datasets, w, prior, report)
    v6.clarify_ghi(fig, prior)
    keep = set(fig.axes)
    for key in 'abc':
        keep.remove(fig.panel_axes[key])
    # Remove the S1-only left key and programme key, alongside its three maps.
    for ax in fig.axes:
        if ax.get_title() == '' and any(t.get_text() in ['Cell state'] for t in ax.texts):
            keep.discard(ax)
        box = ax.get_position()
        top = (1 - box.y1) * fig.page_height
        if abs(top - 68) < .01 and ax not in fig.panel_axes.values():
            keep.discard(ax)
    remove_axes(fig, keep)
    for text in list(fig.texts):
        if text.get_text() in 'abc' and len(text.get_text()) == 1:
            text.remove()
    for key in 'abc':
        fig.panel_axes.pop(key)
        fig.panel_boxes_mm.pop(key)
    fig.legend_boxes_mm.pop('a', None)
    fig.legend_boxes_mm.pop('b_c', None)
    shift_up(fig, 54)


def programme_pc_zoom(fig, ds, sort_metadata, report):
    """Mirror panel i's spatial inset in panel g, retaining programme colours."""
    cells, mapped = ds['cells'], ds['mapped']
    palette = ref.PROG_COLORS['S3']
    limits = sort_metadata['pc_zoom']['bounds']
    assert np.array_equal(limits, report['panels']['i']['pc_zoom']['bounds'])
    ax = fig.panel_axes['g']
    rectangle = v6.Rectangle(
        (limits[0][0], limits[1][0]),
        limits[0][1] - limits[0][0], limits[1][1] - limits[1][0],
        linewidth=.6, edgecolor='#60656a', facecolor='none', linestyle=(0, (3, 2)))
    ax.add_patch(rectangle)

    # Copy the existing inset's geometry and translate it to the programme row.
    box = list(fig.auxiliary_boxes_mm['l_pc_zoom'])
    box[1] += fig.panel_boxes_mm['g'][1] - fig.panel_boxes_mm['i'][1]
    zoom = ref.at(fig, *box)
    fig.auxiliary_boxes_mm['g_pc_zoom'] = box
    zoom.set_facecolor('white')
    missing = mapped.isna()
    zoom.scatter(cells.loc[missing, 'umap_1'], cells.loc[missing, 'umap_2'],
                 s=.40, c=ref.GRAY, alpha=.8, linewidths=0, rasterized=True)
    indices = cells.index[~missing].to_numpy().copy()
    np.random.default_rng(v6.RANDOM_SEED).shuffle(indices)
    zoom.scatter(cells.loc[indices, 'umap_1'], cells.loc[indices, 'umap_2'],
                 s=.40, c=mapped.loc[indices].map(palette), alpha=.8,
                 linewidths=0, rasterized=True)
    ref.finish_map(zoom, limits)
    zoom.set_xticks([])
    zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color('#9d9d9d')
        spine.set_linewidth(.45)
    zoom.set_title('PC cluster (zoom)', fontsize=6.8, pad=2)
    fig.add_artist(v6.ConnectionPatch(
        xyA=(limits[0][1], limits[1][1]), coordsA='data', axesA=ax,
        xyB=(0, 0), coordsB='axes fraction', axesB=zoom,
        color='#a0a0a0', linewidth=.4, clip_on=False))

    inside = cells['umap_1'].between(*limits[0]) & cells['umap_2'].between(*limits[1])
    assert int(inside.sum()) == sort_metadata['pc_zoom']['n_all_cells_in_window']
    report['panels']['g']['pc_zoom'] = {
        'bounds': copy.deepcopy(limits),
        'matches_panel': 'i',
        'n_all_cells_in_window': int(inside.sum()),
        'programme_counts_in_window': mapped.loc[inside].fillna('no programme').value_counts().to_dict(),
        'draw_order': 'same randomized programme order as panel g, over gray unassigned cells',
        'marker_area_pt2': .40,
        'alpha': .8,
        'coordinates_and_labels_modified': False,
    }


def figure3(args, prepared):
    datasets, w, prior, display, sort_metadata, hashes, check = prepared
    fig = v6.page(346)
    report = {'panels': {}, 'input_sha256': hashes, 'g_h_i_validation': check,
              'source': 'v6 S2/S3 panels; S1 Plasmodium moved to Figure 2'}
    memory_upper(fig, datasets, w, prior['panels'], report)
    v6.draw_s3(fig, datasets['S3'], display, sort_metadata, prior['panels'], report, shift=54)
    mapping = dict(zip('defghijklm', 'abcdefghij'))
    for text in fig.texts:
        if text.get_text() in mapping:
            text.set_text(mapping[text.get_text()])
    fig.panel_axes = {mapping[k]: ax for k, ax in fig.panel_axes.items()}
    fig.panel_boxes_mm = {mapping[k]: box for k, box in fig.panel_boxes_mm.items()}
    report['panels'] = {mapping[k]: report['panels'][k] for k in mapping}
    report['panel_mapping_from_v6'] = mapping
    programme_pc_zoom(fig, datasets['S3'], sort_metadata, report)
    fig.text(13.5 / 210, 1 - 7 / 346,
             'Figure 3 | Clone profiles resolve memory-associated branches and two plasma arms',
             fontsize=8.4, fontweight='bold', va='top')
    report['layout'] = v6.validate(fig, 'Figure 3', 346, args.dpi)
    report['interpretation'] = {
        'memory': 'RNA-annotated state composition; P3/P4 are post-hoc gene-score subgroups',
        'plasma': 'Whole-clone profile arms; original GC/nonGC library sort is a post-hoc external overlay',
        'not_inferred': ['time direction', 'two proven memory fates', 'GC negativity proves no GC history']}
    if not args.preview:
        PROV.mkdir(parents=True, exist_ok=True)
        (PROV / 'Figure_3_clone_states.json').write_text(json.dumps(ref.native(report), ensure_ascii=False, indent=2) + '\n')
    save(fig, 'Figure_3_clone_states', args)


def figure2(args, prepared):
    datasets, _, _, _, _, hashes, _ = prepared
    ds = datasets['S1']
    for _, _, rows in overview.DESIGNS:
        for i, row in enumerate(rows):
            row = list(row)
            if 'oral OVA / NP-OVA' in row[0]:
                row[0] = row[0].replace('oral OVA / NP-OVA', 'oral OVA')
                row[-1] = 'Figs. 2, 3'
            elif row[-1] == 'Figure 3':
                row[-1] = 'Figure 4'
            elif 'experiment 2' in row[0]:
                row[-1] = 'panels B–D'
            rows[i] = tuple(row)
    fig = fs.new_page('Figure 2', 'Clone profiles organise B-cell states across experimental systems', height=330)
    fig.panel_boxes_mm, fig.legend_boxes_mm, fig.panel_axes = {}, {}, {}
    overview.designs(fs.panel(fig, 0, 4, 183, 88, 'A'))
    for row, top in [('B', 98), ('C', 166)]:
        for i, (name, title) in enumerate([
            ('gut_gc_reporter', 'Gut oral OVA · GSE284622'),
            ('malaria_late', 'Plasmodium d10–42 · GSE286215'),
            ('ln_vaccine', 'Human lymph node · GSE195673')]):
            ax = fs.panel(fig, 2 + i * 61, top, 58, 52, row if i == 0 else None)
            if row == 'B':
                overview.programme_map(ax, name, title)
            else:
                overview.clone_embedding(ax, name, title, with_note=False)
    cells, clones, mapped = (ds[k] for k in ['cells', 'clones', 'mapped'])
    palette = {p: overview.PROG_COL[p] for p in sorted(clones['clone_programme'].dropna().unique())}
    top = 247
    ax = v6.ax_at(fig, 31, top, 53, 53, 'D', 'Plasmodium | Cell states',
                   coordinates=('Cell UMAP 1', 'Cell UMAP 2'))
    ref.draw_categories(ax, cells, cells['cell_state'], ref.STATE_COLORS['S1'], ['umap_1', 'umap_2'], .35, alpha=.7)
    ref.finish_map(ax, ds['cell_limits'])
    v6.left_key(fig, 'D_states', top, cells['cell_state'], ref.STATE_COLORS['S1'], 'Cell state')
    ax = v6.ax_at(fig, 91, top, 53, 53, 'D2', 'Clone programmes on cells',
                   coordinates=('Cell UMAP 1', 'Cell UMAP 2'))
    # Only D needs a printed row letter.
    for text in list(fig.texts):
        if text.get_text() == 'D2': text.remove()
    ref.draw_categories(ax, cells, mapped, palette, ['umap_1', 'umap_2'], .48, alpha=.88)
    ref.finish_map(ax, ds['cell_limits'])
    ax = v6.legend_box(fig, 'D_programmes', 91, 308, 53, 10)
    v6.legend_text(ax, 0, .1, 'Programme', fontsize=6.8, fontweight='bold', va='top')
    for i, (label, color) in enumerate(palette.items()):
        ax.scatter([i * 13 + .7], [5], s=11, c=color, linewidths=0)
        v6.legend_text(ax, i * 13 + 2.5, 5, label, fontsize=6.8, va='center')
    composition = pd.crosstab(mapped, cells['cell_state']).reindex(index=list(palette), fill_value=0)
    fraction = composition.div(composition.sum(axis=1), axis=0)
    ax = v6.ax_at(fig, 151, top, 53, 53, 'D3', 'Captured states by programme')
    for text in list(fig.texts):
        if text.get_text() == 'D3': text.remove()
    bottom = np.zeros(len(fraction))
    for state, color in ref.STATE_COLORS['S1'].items():
        values = fraction[state].to_numpy() if state in fraction else np.zeros(len(fraction))
        ax.bar(np.arange(4), values, bottom=bottom, color=color, width=.68, linewidth=0)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(4), fraction.index, fontsize=6.8)
    ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'], fontsize=6.8)
    ax.set_ylabel('Captured cell fraction', fontsize=7, labelpad=3)
    ax.set_xlabel('Clone programme', fontsize=7, labelpad=3)
    fig.text(31 / 210, 1 - 324 / 330,
             f'Plasmodium: {len(clones):,} clone profiles; {mapped.notna().sum():,} mapped cells of {len(cells):,}; grey = other cells.',
             fontsize=6.8, va='top', color=fs.INK2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = [t.get_text() for t in fig.findobj(match=matplotlib.text.Text)
               if t.get_visible() and t.get_text() and
               (t.get_window_extent(renderer).x0 < -1 or t.get_window_extent(renderer).y0 < -1 or
                t.get_window_extent(renderer).x1 > fig.bbox.x1 + 1 or t.get_window_extent(renderer).y1 > fig.bbox.y1 + 1)]
    assert not outside, outside
    if not args.preview:
        PROV.mkdir(parents=True, exist_ok=True)
        (PROV / 'Figure_2_GC_overview.json').write_text(json.dumps(ref.native({
            'input_sha256': hashes, 'new_D_dataset': 'GSE286215', 'new_D_n_clones': len(clones),
            'new_D_n_mapped_cells': int(mapped.notna().sum()), 'new_D_cell_state_counts': composition.to_dict(),
            'previous_gut_D': 'Moved to Figure 3 and its supplements', 'text_outside_page': outside,
            'cross_references': {'oral_OVA': 'Figures 2 and 3', 'RBD': 'Figure 4'}}), indent=2) + '\n')
    save(fig, 'Figure_2_GC_overview', args)


def figure4(args):
    fig = fs.new_page('Figure 4', 'Clone profiles relate to experimental annotations and barcode-linked outcomes', height=232)
    measured.row_rbd(fig, 6)
    measured.row_larry(fig, 74)
    measured.row_watermelon(fig, 142)
    for ax in fig.axes:
        title = ax.get_title()
        if title.startswith('Day-2 profile predicts') or title.startswith('Early clone profile vs later'):
            box = ax.get_position()
            ax.set_position([box.x0 + 10/210, box.y0, box.width - 10/210, box.height])
            ax.yaxis.labelpad = 1
        if title.startswith('The same clones'):
            ax.set_ylabel('')
        if title == 'Sister wells reproduce the bias':
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.yaxis.labelpad = 2
        if ax.get_label() == '<colorbar>':
            ax.set_yticks([0,.5,1])
            ax.set_yticklabels(['0','0.5','1'])
            ax.tick_params(labelsize=5.5, pad=1)
        elif title:
            ax.set_title(title, fontsize=6.8)
        for text in ax.texts:
            x,y=text.get_position()
            if text.get_transform() == ax.transAxes and y < 0:
                width_pt=ax.get_position().width*fig.get_figwidth()*72
                nchars=max(22,int(width_pt/(5.8*.46)))
                text.set_text('\n'.join(textwrap.fill(line,width=nchars,break_long_words=False)
                                         for line in text.get_text().splitlines()))
                text.set_position((x,-.235));text.set_fontsize(5.8)
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_text() and text.get_fontsize()<5.5:
            text.set_fontsize(5.5)
        value = text.get_text()
        if "A clone's day 0–7 state predicts" in value:
            text.set_text(value.replace("A clone's day 0–7 state predicts\nits persister propensity a week later\n",
                                        "Early clone profiles associate with\nlater persister-gate composition\n"))
        elif value == 'clone mean (scRNA)':
            text.set_text('Threadfin mean profile')
        if '10⁻⁴' in text.get_text():
            text.set_text(text.get_text().replace('2×10⁻⁴',r'$2\times10^{-4}$')
                          .replace('5×10⁻⁴',r'$5\times10^{-4}$'))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox=text.get_window_extent(renderer)
            if bbox.x0 < -.5 or bbox.y0 < -.5 or bbox.x1 > fig.bbox.width+.5 or bbox.y1 > fig.bbox.height+.5:
                outside.append(text.get_text())
    if not args.preview:
        (PROV/'Figure_4_measured_validation.json').write_text(json.dumps({'source':'Existing experimental overlays and outcome statistics',
            'height_mm':232,'text_outside_page':outside,'changes':['Renumbered 3 to 4','WATERMELON association terminology',
            'Mean-profile comparator name','Label spacing and visible bottom notes','Superscript minus rendered with mathtext']},indent=2)+'\n')
    assert not outside, outside
    save(fig, 'Figure_4_measured_validation', args)


def supplements(args, prepared):
    datasets, w, prior, _, _, _, _ = prepared
    # Current supporting view of former g/h/i, with no obsolete three-programme display.
    fig = v6.page(96)
    report = {'panels': {}}
    base.main_upper(fig, datasets, w, prior['panels'], report)
    v6.clarify_ghi(fig, prior['panels'])
    keep = {fig.panel_axes[k] for k in 'ghi'}
    keep.update(ax for ax in fig.axes if ax.get_title() == 'entropy')
    remove_axes(fig, keep)
    for text in list(fig.texts):
        if text.get_text() not in 'ghi' or len(text.get_text()) != 1:
            text.remove()
    fig.panel_boxes_mm = {k: fig.panel_boxes_mm[k] for k in 'ghi'}
    fig.panel_axes = {k: fig.panel_axes[k] for k in 'ghi'}
    fig.legend_boxes_mm = {}
    shift_up(fig, 131)
    for text in fig.texts:
        text.set_text({'g': 'a', 'h': 'b', 'i': 'c'}[text.get_text()])
    fig.text(7 / 210, 1 - 4 / 96, 'Supplement to Figure 3 | Memory-associated state breadth and donor consistency',
             fontsize=8, fontweight='bold', va='top')
    v6.validate(fig, 'memory support', 96, args.dpi)
    save(fig, 'spont_gc_story_panel', args)

    fig = v6.page(85)
    supp_report = {'panels': {}}
    base.supplement(fig, datasets['S3'], previous('figure_clone_states_stats_v3.json')['panels'], supp_report)
    fig.panel_axes['b'].set_ylim(-.06, 1.60)
    v6.validate(fig, 'GC-sort support', 85, args.dpi)
    save(fig, 'Supplementary_Figure_3_GC_sort', args, SUPP)

    ds = datasets['S2']
    cells, clones = ds['cells'], ds['clones']
    has_m = cells['fate'].isin(ref.M_LABELS).groupby(cells['clone_id']).any().reindex(clones.index)
    fig = v6.page(95)
    ax = v6.ax_at(fig, 24, 18, 64, 64, 'a', 'Clones containing a MemB-labelled cell', coordinates=('Clone x', 'Clone y'))
    ax.scatter(clones['x'], clones['y'], s=clones['n_cells']*.45, c=ref.GRAY, linewidths=0, rasterized=True)
    ax.scatter(clones.loc[has_m,'x'], clones.loc[has_m,'y'], s=clones.loc[has_m,'n_cells']*.45,
               c='#008300', linewidths=0, rasterized=True)
    ref.finish_map(ax, ds['clone_limits'])
    ax = v6.ax_at(fig, 128, 18, 64, 64, 'b', 'Memory-associated clone groups')
    summaries=[]
    for i, (prog, color) in enumerate(ref.PROG_COLORS['S2'].items()):
        mask=clones['clone_programme'].eq(prog); n=int(mask.sum()); positive=int(has_m[mask].sum()); frac=positive/n
        ax.bar(i,frac,width=.64,color=color)
        ax.text(i,frac+.018,f'{frac:.1%}\n{positive}/{n}',fontsize=6.8,ha='center')
        summaries.append({'programme':prog,'n':n,'memory_positive':positive,'fraction':frac})
    ax.set_xticks(range(4),[s['programme'] for s in summaries],fontsize=6.8)
    ax.set_ylim(0,.58);ax.set_ylabel('Clones with MemB (%)',fontsize=7.2)
    ax.set_yticks([0,.2,.4],['0%','20%','40%'],fontsize=6.8)
    fig.text(13.5/210,1-4/95,'Supplement to Figure 3 | RNA annotations supporting memory-associated branches',
             fontsize=8,fontweight='bold',va='top')
    if not args.preview:
        (PROV/'Figure_3_memory_annotation.json').write_text(json.dumps({'groups':summaries,
            'note':'P3/P4 are the prior P3a/P3b gene-score subgroups; RNA-derived and post-hoc, not independent fate validation.'},indent=2)+'\n')
    v6.validate(fig,'memory branch support',95,args.dpi)
    save(fig,'Supplementary_Figure_3_memory_annotation',args,SUPP)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',choices=['2','3','4','supplement','all'],default='all')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    SUPP.mkdir(parents=True,exist_ok=True);PROV.mkdir(parents=True,exist_ok=True)
    prepared=prepare() if args.figure!='4' else None
    for label,fn in [('2',figure2),('3',figure3),('4',figure4),('supplement',supplements)]:
        if args.figure in [label,'all']:
            print('Stage:',label,flush=True)
            fn(args) if label=='4' else fn(args,prepared)


if __name__=='__main__':
    main()
