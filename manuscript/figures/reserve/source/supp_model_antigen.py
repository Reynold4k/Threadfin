#!/usr/bin/env python
"""Consolidated model-antigen supplement using current Dandelion paired-chain clones.

Only figure rendering; no clone calling, package benchmark, or per-mouse tests.
The oral OVA reporter experiment is indexed here and developed in main Figure 3.
FACS readouts and sequence-derived W33L labels are post hoc overlays.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from figstyle import BLUE, INK, INK2, MUTED, ORANGE, VIOLET, new_page, panel, save

ROOT = Path(__file__).resolve().parents[5]
CASE = ROOT / "Threadfin/manuscript/case_studies/results"
SR = ROOT / "story_review_2026-10-12"
HERE = Path(__file__).resolve().parents[2]

def frame(ax, title, unit="Clone"):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.8, color=INK, pad=5)
    ax.set_xlabel(f"{unit} UMAP 1", fontsize=6.2, labelpad=2)
    ax.set_ylabel(f"{unit} UMAP 2", fontsize=6.2, labelpad=2)
    for spine in ax.spines.values():
        spine.set_color(MUTED); spine.set_linewidth(.5)

def note(fig, x, y, text):
    ax=panel(fig,x,y,54,17); ax.set_axis_off()
    ax.text(0,1,text,va="top",fontsize=5.9,color=INK2,linespacing=1.4)

def clone_map(fig, t, col, x, y, letter, title, cmap="viridis", scale=1.4):
    ax=panel(fig,x,y,49,40,letter)
    v=pd.to_numeric(t[col],errors="coerce"); ok=v.notna()
    sizes=np.clip(t.n_cells*scale,3,26)
    ax.scatter(t.x,t.y,s=sizes,c="#e8e8e3",linewidths=0,rasterized=True)
    pts=ax.scatter(t.loc[ok,"x"],t.loc[ok,"y"],s=sizes[ok],c=v[ok],
                   vmin=0,vmax=1,cmap=cmap,linewidths=0,rasterized=True)
    frame(ax,title)
    cb=fig.colorbar(pts,ax=ax,fraction=.046,pad=.018,ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=5)
    return int(ok.sum())

def main(folder):
    c=pd.read_csv(CASE/"gc_np_pc/cells.csv.gz")
    w=pd.read_csv(SR/"gc_np_pc_W33L_current.csv")
    nc=pd.read_csv(CASE/"mouse_np/cells.csv.gz")
    nt=pd.read_csv(CASE/"mouse_np/clone_table.csv")
    rt=pd.read_csv(CASE/"mouse_rbd/clone_table.csv")
    fig=new_page("Supplementary","Model-antigen datasets: clone states and measured readouts",height=286)
    ax=panel(fig,2,2,179,29,"A"); ax.set_axis_off()
    rows=[
        ["Oral OVA + CT","GSE284622; MLN / PP / LP","16,619 cells / 958 profiles","OVA-selected; GC-fate reporter (Fig. 3)"],
        ["NP-OVA; day 14","GSE246382; draining LN","884 cells / 96 profiles","GC/PC gates; sequence-derived W33L"],
        ["NP-OVA; GC only","GSE287123; 36 h reporter","10,109 cells / 471 profiles","Recent division; sequence-derived W33L"],
        ["RBD; GC only","GSE287123; 36 h reporter","36,188 cells / 1,887 profiles","Recent division; bait OR zone gates"]
    ]
    tb=ax.table(cellText=rows,colLabels=["Defined-antigen system","Dataset / sampling","Current analysis coverage","Experimental annotation"],
                colWidths=[.19,.225,.26,.325],bbox=[0,.12,1,.88],cellLoc="left",colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(5.7)
    for (r,k),cell in tb.get_celld().items():
        cell.set_linewidth(.3); cell.set_edgecolor("#dadbd5"); cell.PAD=.035
        cell.set_facecolor("#eaf0f5" if r==0 else ("#f7f7f4" if r%2 else "white"))
        if r==0: cell.set_text_props(weight="bold")
    ax.text(0,0,"Profiles: current clones with ≥2 captured cells. Clone identity: Dandelion paired heavy + light chains.",
            fontsize=5.8,va="bottom",color=INK2)

    ax=panel(fig,2,43,49,40,"B")
    pc=c.fate.eq("plasma cell")
    for label,mask,col in [("GC",~pc,BLUE),("PC",pc,VIOLET)]:
        ax.scatter(c.loc[mask,"umap_1"],c.loc[mask,"umap_2"],s=2.8,c=col,alpha=.75,
                   linewidths=0,rasterized=True,label=label)
    frame(ax,"NP plate study: measured GC/PC","Cell")
    ax.legend(loc="upper left",fontsize=5.8,markerscale=3,handletextpad=.1)
    note(fig,2,93,"884 plate-sorted cells at day 14.\nGC/PC identity comes from the sorting gate.")

    n=clone_map(fig,w,"fate:plasma cell",67,43,"C","Same study: clone PC fraction","cividis",3)
    note(fig,67,93,f"{n} clone profiles; ≥2 cells each.\nColour = captured PC-gate fraction.")

    n=clone_map(fig,w,"W33L_fraction",132,43,"D","Same map: callable W33L fraction","magma_r",3)
    note(fig,132,93,f"{n} callable clones; {(w.W33L_fraction>0).sum()} W33L-positive.\nGrey = W33 status unavailable.")

    ax=panel(fig,2,113,49,40,"E")
    z=w.dropna(subset=["W33L_fraction","PC_fraction"])
    ax.scatter(z.W33L_fraction,z.PC_fraction,s=np.clip(z.n_cells*3,5,32),
               c=BLUE,alpha=.75,linewidths=.4,edgecolors="white",rasterized=True)
    ax.set_title("W33L vs captured PC fraction",fontsize=6.8,pad=5)
    ax.set_xlabel("Callable-cell W33L fraction",fontsize=6.1)
    ax.set_ylabel("Captured PC-gate fraction",fontsize=6.1)
    ax.set_xlim(-.05,1.05); ax.set_ylim(-.05,1.05)
    ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
    note(fig,2,163,f"{len(z)} callable clone profiles; size = cells.\nA sequence marker and a sampled state;\nthese fractions do not measure output rate.")

    ax=panel(fig,67,113,49,40,"F")
    for gate,col,label in [("mCherry-high",ORANGE,"High"),("mCherry-low",BLUE,"Low")]:
        m=nc.division_gate.eq(gate)
        ax.scatter(nc.loc[m,"umap_1"],nc.loc[m,"umap_2"],s=.6,c=col,alpha=.55,
                   linewidths=0,rasterized=True,label=label)
    frame(ax,"NP reporter: measured division gate","Cell")
    ax.legend(loc="upper left",fontsize=5.5,markerscale=4,handletextpad=.1)
    note(fig,67,163,f"{len(nc):,} GC-sorted cells; 36 h dilution window.\nLow mCherry marks ≥6 recent divisions;\nthis experiment does not sample PCs.")

    n=clone_map(fig,nt,"division_gate:mCherry-low",132,113,"G","NP: clone recent-division fraction")
    note(fig,132,163,f"{n} reporter-labelled clone profiles.\n87 profiles have callable W33 status.\nDivision history and PC capture differ.")

    n=clone_map(fig,rt,"division_gate:mCherry-low",2,186,"H","RBD: clone recent-division fraction")
    note(fig,2,236,f"{n:,} reporter-labelled clone profiles.\n36,188 cells; all cells were GC-sorted.\nThe same readouts anchor main Fig. 4A.")
    n=clone_map(fig,rt,"rbd_bait:RBD+",67,186,"I","RBD: measured bait-binding fraction","cividis")
    note(fig,67,236,f"{n:,} bait-labelled clone profiles.\nColour = captured RBD-positive fraction;\na binding gate, not a quantitative affinity.")
    n=clone_map(fig,rt,"zone_gate:DZ",132,186,"J","RBD: measured dark-zone fraction","magma_r")
    note(fig,132,236,f"{n:,} zone-labelled clone profiles.\nBait and zone use separate experimental arms;\ngrey profiles were not measured in that arm.")

    ax=panel(fig,2,261,179,6); ax.set_axis_off()
    ax.text(0,1,"All clone maps use RNA profiles without the displayed gates or W33L labels. Fractions describe captured cells, not total clone size or lineage flux.\n"
            "W33L fractions use callable W33/W33L cells only. Map proximity indicates expression-profile similarity, not shared receptor ancestry.",
            fontsize=5.7,color=INK2,va="top",linespacing=1.4)
    save(fig,"Supplementary_model_antigen_NP",folder=folder,dpi=1200)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default=str(HERE))
    main(p.parse_args().output_dir)
