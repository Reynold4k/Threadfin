"""Plot audited real-data clone coordinates; never fit or select a geometry."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'case_studies/results/gc_np_pc_clone_embedding'
AUDIT_DATA = HERE.parents[1] / 'case_studies/results/gc_legacy_parameter_audit'
COLORS = {'light zone': '#d9a643', 'Myc+ light zone': '#4b9a8b',
          'dark zone': '#3478ad', 'plasma cell': '#c65359', 'mixed': '#9da4ab'}
LABELS = {'light zone': 'LZ', 'Myc+ light zone': 'Myc+ LZ',
          'dark zone': 'DZ', 'plasma cell': 'PC', 'mixed': 'Mixed'}


def selection():
    return json.loads((AUDIT_DATA / 'selected.json').read_text())


def table(preset=None, seed=123):
    if preset is None:
        chosen = selection()
        path = AUDIT_DATA / 'maps' / (chosen['name'] + '.csv')
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != chosen['coordinates_sha256']:
            raise RuntimeError('Selected clone coordinates changed after audit.')
        return pd.read_csv(path, index_col=0)
    summary = json.loads((DATA / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise RuntimeError('Real GSE246382 reclustering is incomplete.')
    return pd.read_csv(DATA / f'{preset}_seed{seed}_clone_map.csv', index_col=0)


def axes_style(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.margins(.055)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def scatter(ax, tab, color='gate', size_factor=1):
    # Marker AREA is directly proportional to actual captured cells.
    sizes = size_factor * 5 * tab.n_cells
    if color in {'gate', 'cluster'}:
        categories = COLORS if color == 'gate' else cluster_colors(tab)
        values = tab.dominant_gate if color == 'gate' else tab.clone_cluster.astype(str)
        for state, value in categories.items():
            use = values.eq(state)
            if use.any():
                ax.scatter(tab.loc[use, 'x'], tab.loc[use, 'y'], s=sizes[use],
                           color=value, edgecolors='white', linewidths=.35, alpha=.95)
        result = None
    else:
        result = ax.scatter(tab.x, tab.y, s=sizes, c=tab[color], cmap='viridis',
                            edgecolors='white', linewidths=.35, alpha=.95)
    axes_style(ax)
    return result


def cluster_colors(tab):
    labels = sorted(tab.clone_cluster.astype(str).unique(), key=int)
    return {label: plt.get_cmap('tab10')(int(label) % 10) for label in labels}


def clone_legends(ax, tab, fontsize=6, size_factor=1, separate=False):
    handles = [Line2D([], [], color=color, marker='o', ls='', markersize=4,
                      label=label) for label, color in cluster_colors(tab).items()]
    x = 0 if separate else 1.02
    first = ax.legend(handles=handles, title='clone_cluster', frameon=False,
                      loc='upper left', bbox_to_anchor=(x, 1), fontsize=fontsize,
                      title_fontsize=fontsize, borderaxespad=0, handletextpad=.5)
    ax.add_artist(first)
    counts = sorted(set([int(tab.n_cells.min()), 3, 10, int(tab.n_cells.max())]))
    counts = [n for n in counts if tab.n_cells.min() <= n <= tab.n_cells.max()]
    handles = [ax.scatter([], [], s=size_factor * 5 * n, color='#444444',
                          edgecolor='white', linewidth=.35, label=str(n)) for n in counts]
    ax.legend(handles=handles, title='Captured cells', frameon=False,
              loc='upper left', bbox_to_anchor=(x, .40), fontsize=fontsize,
              title_fontsize=fontsize, borderaxespad=0, labelspacing=.9)


def gate_legend(ax, tab, y=-.05, size=6):
    available = [s for s in COLORS if s in set(tab.dominant_gate)]
    handles = [Line2D([], [], color=COLORS[s], marker='o', ls='', markersize=4,
                      label=LABELS[s]) for s in available]
    ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0, y),
              ncol=3, fontsize=size, handletextpad=.3, columnspacing=.8, borderaxespad=0)


def main_node_panel(ax, legend_ax):
    tab = table()
    scatter(ax, tab, 'cluster', size_factor=1.2)
    ax.set_xlabel('Clone UMAP 1', fontsize=6, labelpad=2)
    ax.set_ylabel('Clone UMAP 2', fontsize=6, labelpad=2)
    legend_ax.set_axis_off()
    clone_legends(legend_ax, tab, size_factor=1.2, separate=True, fontsize=6.5)
    return tab


def myc_panel(ax):
    ax.set_axis_off()
    inner = ax.inset_axes([0, .13, 1, .85])
    tab = table(); im = scatter(inner, tab, 'marker:Myc')
    inner.set_xlabel('Clone UMAP 1', fontsize=5.5, labelpad=1.5)
    inner.set_ylabel('Clone UMAP 2', fontsize=5.5, labelpad=1.5)
    cb = ax.figure.colorbar(im, ax=inner, orientation='horizontal', fraction=.04, pad=.08, aspect=28)
    cb.set_label('Mean log-normalised Myc RNA', fontsize=6)
    return tab


def cell_gates(ax):
    ax.set_axis_off()
    inner = ax.inset_axes([0, .02, 1, .96])
    cells = pd.read_csv(AUDIT_DATA / 'cells.csv.gz', index_col=0)
    basis = selection()['basis']
    xy = ['legacy_umap_1', 'legacy_umap_2'] if basis == 'legacy' else ['umap_1', 'umap_2']
    for state, col in COLORS.items():
        q = cells[cells.fate.eq(state)]
        if len(q): inner.scatter(q[xy[0]], q[xy[1]], s=1, color=col, alpha=.65, linewidths=0, rasterized=True)
    axes_style(inner)
    handles = [Line2D([], [], color=COLORS[s], marker='o', ls='', markersize=4, label=LABELS[s]) for s in COLORS if s != 'mixed']
    ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0, -.03), ncol=4, fontsize=6,
              handletextpad=.3, columnspacing=.6, borderaxespad=0)


def markers(ax):
    tab = table()
    wanted = ['Myc', 'Bcl6', 'Aicda', 'Top2a', 'Mki67', 'Prdm1', 'Xbp1', 'Jchain']
    genes = [g for g in wanted if 'marker:' + g in tab]
    tab['clone_cluster'] = tab.clone_cluster.astype(str)
    means = tab.groupby('clone_cluster')[['marker:' + g for g in genes]].mean()
    order = sorted(means.index, key=int)
    means = means.reindex(order).T
    source = AUDIT_DATA / 'display_marker_means.csv'; means.to_csv(source)
    im = ax.imshow(means.to_numpy(), aspect='auto', cmap='viridis', vmin=0)
    ax.set_xticks(range(len(order)), order, fontsize=6)
    ax.set_xlabel('clone_cluster', fontsize=6)
    ax.set_yticks(range(len(genes)), genes, fontsize=6.5)
    for (i, j), value in np.ndenumerate(means.to_numpy()):
        ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=5.5,
                color='#26313b' if value > means.to_numpy().max() * .7 else 'white')
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=.035, pad=.17, aspect=28)
    cb.set_label('Mean clone-averaged log-normalised RNA', fontsize=5.8)
    return source


def gate_composition(ax):
    tab = table()
    cols = ['gate:' + s for s in COLORS if s != 'mixed']
    means = tab.groupby('clone_cluster')[cols].mean()
    bottom = np.zeros(len(means))
    for name in COLORS:
        if name == 'mixed': continue
        values = means['gate:' + name].to_numpy()
        ax.bar(means.index.astype(str), values, bottom=bottom, color=COLORS[name], width=.75,
               linewidth=.5, edgecolor='white', label=LABELS[name])
        bottom += values
    ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    ax.set_ylabel('Mean family-level fraction', fontsize=6.5)
    ax.set_xlabel('clone_cluster', fontsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=4, fontsize=6,
              frameon=False, columnspacing=.8, handlelength=1)


def review():
    out = HERE / 'review'; out.mkdir(exist_ok=True)
    tab = table()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    scatter(ax, tab, 'cluster', size_factor=3)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title('GSE246382 · clone embedding', fontsize=12)
    clone_legends(ax, tab, fontsize=9, size_factor=3)
    chosen = selection()
    unit = 'same-mouse V–D–J groups' if chosen['definition'] == 'donor_vdj' else 'same-mouse sequence families'
    fig.text(.08, .025, f'{len(tab)} {unit} · {int(tab.n_cells.sum())} cells · ≥1 cell/group · seed 123', fontsize=8.5)
    fig.subplots_adjust(left=.1, right=.74, bottom=.12, top=.9)
    fig.savefig(out / 'GSE246382_selected_clone_embedding.png', dpi=250)
    fig.savefig(out / 'GSE246382_selected_clone_embedding.pdf'); plt.close(fig)
    selected = table()
    stability = pd.read_csv(AUDIT_DATA / 'seed_stability.csv')
    row = stability[(stability.basis == chosen['basis']) & (stability.definition == chosen['definition'])
                    & (stability.min_clone_size == chosen['min_clone_size'])
                    & (stability.umap_k == chosen['umap_k']) & (stability.min_dist == chosen['min_dist'])
                    & (stability.resolution == chosen['resolution'])].iloc[0]
    maps = [chosen['name'].replace('seed123', 'seed'+str(seed)) for seed in (123,7,0,1,42,99,2024)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax, name in zip(axes.flat, maps):
        tab = pd.read_csv(AUDIT_DATA / 'maps' / (name + '.csv'), index_col=0)
        # Colours transfer the primary partition by family ID solely to compare layouts.
        tab['clone_cluster'] = selected.clone_cluster.reindex(tab.index)
        scatter(ax, tab, 'cluster', size_factor=1.4)
        ax.set_title('UMAP seed ' + name.rsplit('seed', 1)[1], fontsize=10)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(.05,.8,'Same input groups and parameters\nSeven UMAP seeds; graph/Leiden seed 0\n\nColours follow seed-123 membership\nfor visual comparison.\n\nIndependent Leiden agreement:\n'
                       f'median ARI {row.ari_median:.3f}; minimum {row.ari_min:.3f}.\nExact branch shape is not invariant.',
                       transform=axes.flat[-1].transAxes,fontsize=9,va='top',linespacing=1.7)
    fig.suptitle('Selected parameters: layouts across all seven seeds', fontsize=12)
    fig.text(.5, .015, 'Coordinates and independent partitions for every run: case_studies/results/gc_legacy_parameter_audit/maps/', ha='center', fontsize=8)
    fig.subplots_adjust(left=.02, right=.98, bottom=.07, top=.9, wspace=.10,hspace=.20)
    fig.savefig(out / 'GSE246382_selected_seed_comparison.png', dpi=200)
    fig.savefig(out / 'GSE246382_selected_seed_comparison.pdf'); plt.close(fig)
    # Preserve the earlier 49-family preset comparison as a labelled control.
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.2))
    for j, preset in enumerate(('cohesive', 'continuous', 'discrete')):
        tab = table(preset)
        params = json.loads((DATA / f'{preset}_seed123_parameters.json').read_text())
        for i, col in enumerate(('cluster', 'marker:Myc', 'marker:Plasma cell')):
            ax = axes[i, j]; im = scatter(ax, tab, col)
            if i == 0:
                ax.set_title(f'{preset}: UMAP k={params["umap_n_neighbors"]}, min_dist={params["min_dist"]}\nScanpy k={params["n_neighbors"]}; Leiden r={params["resolution"]}; {tab.clone_cluster.nunique()} clusters', fontsize=8)
                handles = [Line2D([], [], color=c, marker='o', ls='', markersize=4, label=k)
                           for k, c in cluster_colors(tab).items()]
                ax.legend(handles=handles, title='clone_cluster', frameon=False, loc='lower left', fontsize=6, title_fontsize=6)
            else:
                cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.04, pad=.02)
                cb.ax.tick_params(labelsize=6)
    for i, label in enumerate(('clone_cluster', 'Myc RNA', 'Plasma-cell module')):
        axes[i, 0].set_ylabel(label, fontsize=9)
    fig.suptitle('GSE246382: identical 49 same-mouse families, three fixed reclustering presets', fontsize=11)
    fig.text(.5, .018, 'Cell-UMAP centroids → distance-row UMAP → Scanpy graph/Leiden; seed 123; ≥3 cells/family. Gates/markers excluded.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, bottom=.07, top=.9, wspace=.28, hspace=.35)
    fig.savefig(out / 'GSE246382_reclustering_presets.png', dpi=200)
    fig.savefig(out / 'GSE246382_reclustering_presets.pdf'); plt.close(fig)


if __name__ == '__main__': review()
