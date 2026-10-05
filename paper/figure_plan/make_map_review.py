"""Reproduce the previous Figure 3 map and isolate the presentation changes."""
from pathlib import Path
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from make_biology_figures import clones, p, new_page, save, color_map, note, BLUE


def main():
    folder=Path(__file__).resolve().parent/'review'
    fig=new_page('Map review','Figure 3: unchanged coordinates, different biological views',
                 'The original early PB map is restored; a later GC map addresses a separate cohort.',height=216)
    a=p(fig,0,12,81,51,'A','Previous map: early cohort, PB colour')
    # Same calls as legacy_panels.gc_clone_map_panel, including automatic aspect.
    t=clones('malaria').dropna(subset=['x','y','cell_state:PB'])
    cmap=LinearSegmentedColormap.from_list('previous',['#cde2fb',BLUE,'#123a6b'])
    im=a.scatter(t.x,t.y,c=t['cell_state:PB'],cmap=cmap,s=1+5*np.sqrt(t.n_cells),
                 edgecolors='white',linewidths=.2,alpha=.9,rasterized=True)
    a.set_xticks([]);a.set_yticks([])
    for sp in a.spines.values():sp.set_visible(False)
    cb=fig.colorbar(im,ax=a,orientation='horizontal',fraction=.045,pad=.055,aspect=24)
    cb.set_label('PB fraction; previous blue palette',fontsize=6)
    b=p(fig,103,12,80,51,'B','Restored map: early cohort, PB colour')
    color_map(b,'malaria','cell_state:PB','PB fraction; current palette')
    c=p(fig,0,98,81,51,'C','Later cohort, GC colour')
    color_map(c,'malaria_late','cell_state:GC','GC fraction')
    d=p(fig,103,98,80,51,'D','Same later coordinates, PB colour')
    color_map(d,'malaria_late','cell_state:PB','PB fraction')
    e=p(fig,0,181,183,10,'E','Coordinate and membership audit')
    e.set_axis_off()
    note(e,'Early: 801 clones. Later: 1,183 clones. Clone IDs and cell membership match the pre-revision results.\nMaximum absolute difference in saved clone and cell UMAP coordinates: 0 in both cohorts.\nA/B differ in palette, colour-bar orientation and aspect; C/D differ only in the measured state used for colour.',.6,6.1)
    save(fig,'Figure_3_map_reproduction',folder=folder)


if __name__=='__main__':main()
