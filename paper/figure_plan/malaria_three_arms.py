"""Matched views of all three state-enriched regions in Figure 3C."""
import hashlib
import json
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def draw(fig, b):
    selection = json.loads((b.DATA/'malaria_late_embedding_audit/selected.json').read_text())
    source = b.DATA/'malaria_late_embedding_audit/maps'/(selection['name']+'.csv')
    xy = pd.read_csv(source, index_col=0)
    table = b.clones('malaria_late')
    table = table[table.reliability.ge(.5)].copy()
    assert set(table.index) == set(xy.index) and xy.index.is_unique
    table[['x','y']] = xy.reindex(table.index)[['x','y']]
    cells = b.cells('malaria_late')
    fractions = pd.crosstab(cells.clone_id, cells.cell_state, normalize='index')
    for state in ['GC', 'PB', 'Memory']:
        values = fractions[state].reindex(table.index)
        assert values.notna().all()
        if f'cell_state:{state}' in table:
            assert np.allclose(values, table[f'cell_state:{state}'])
        table[state] = values
    table['day'] = table.donor.astype(str).str.extract(r'D(\d+)')[0].astype(int)
    infected = table[table.treatment.isin(['Saline','Artesunate'])]
    by_mouse = infected.groupby(['donor','day','treatment'], observed=True).agg(
        GC=('GC','mean'), PB=('PB','mean'), Memory=('Memory','mean'),
        n_families=('n_cells','size')).reset_index()
    out = b.DATA/'figure23_review'
    table.to_csv(out/'supp13_family_state_fractions.csv')
    by_mouse.to_csv(out/'supp13_mouse_state_fractions.csv', index=False)

    for i, (state, colour, label) in enumerate([
            ('GC', b.BLUE, 'GC-enriched region'), ('PB', b.RED, 'PB-enriched region'),
            ('Memory', b.GREEN, 'Memory-enriched region')]):
        ax = b.p(fig, i*63, 9, 57, 76, 'ABC'[i], label)
        cm = LinearSegmentedColormap.from_list(state, ['#e0e4ea',colour])
        im = ax.scatter(table.x, table.y, c=table[state], s=1+5*np.sqrt(table.n_cells),
                        cmap=cm, vmin=0, vmax=1, edgecolors='white', linewidths=.15,
                        rasterized=True)
        b.map_axis(ax); b.clone_axis(ax)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.04, pad=.065, aspect=25)
        cb.set_ticks([0,.5,1]); cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(f'Captured {state} fraction in each family', fontsize=5.3, labelpad=2)
        b.note(ax, f'{len(table):,} families; same Figure 3C coordinates', -.31, 5.2)

        ax2 = b.p(fig, i*63, 131, 53, 51, 'DEF'[i], f'{state} occupancy\nacross sampling dates')
        days = [10,14,21,28,35,42]
        for treatment, offset, style, filled in [
                ('Saline',-.35,'-',True), ('Artesunate',.35,'--',False)]:
            z = by_mouse[by_mouse.treatment.eq(treatment)]
            medians = []
            for day in days:
                values = z[z.day.eq(day)].sort_values('donor')[state]
                ax2.scatter(day+offset+np.linspace(-.18,.18,len(values)), values,
                            s=12, facecolors=colour if filled else 'white',
                            edgecolors=colour, linewidths=.65, zorder=3)
                medians.append(values.median())
            ax2.plot(np.array(days)+offset, medians, color=colour, ls=style, lw=.9)
        ax2.set(xticks=days, ylim=(-.025,1.025), xlim=(8.5,43.5))
        ax2.set_xlabel('Day after infection', fontsize=6)
        ax2.set_ylabel('Mean captured fraction across families', fontsize=5.3, labelpad=2)
        handles = [Line2D([],[],marker='o',ms=3.2,color=colour,lw=.8,label='Saline'),
                   Line2D([],[],marker='o',ms=3.2,color=colour,mfc='white',ls='--',lw=.8,label='Anti-malarial')]
        ax2.legend(handles=handles, frameon=False, fontsize=5.3,
                   loc='upper left', bbox_to_anchor=(0,-.24), ncol=2,
                   handlelength=1.25, columnspacing=.8)

    foot = b.p(fig,0,216,183,12,None,''); foot.set_axis_off()
    foot.text(0,1,'A–C: every family is shown in every panel; colour changes only. Point size reflects captured-cell count.\n'
                  'D–F: each dot is a mouse; lines join group medians. Three mice per arm/date; controls excluded.\n'
                  'State enrichment and changing capture composition do not establish lineage direction or a future fate.',
              fontsize=5.8, va='top', color=b.INK, linespacing=1.4)
    b.AUDIT['malaria_three_arms'] = {
        'source':str(source.relative_to(b.ROOT)),
        'coordinates_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'n_families_per_map':len(table), 'n_infected_mice':len(by_mouse),
        'n_infected_families':len(infected), 'all_maps_share_coordinates':True,
        'fraction_definition':'Fraction of captured cells per family; equal-family average within mouse',
        'source_tables':'case_studies/results/figure23_review/supp13_*.csv'}
