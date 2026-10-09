"""Original vector redraws of the measured designs, not synthetic FACS data.

Sources: Merkenschlager et al., Nature 2025, Extended Data 1–3,5;
Skinner/Asad et al., Nature Immunology 2026, Fig.1/4 and deposited metadata.
All gate outlines are schematic. No source bitmap or simulated data cloud is used.
"""
from __future__ import annotations
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
import schematics as sk

BLUE='#3478ad'; RED='#c65359'; GREEN='#4b9a8b'; GOLD='#d9a643'
VIOLET='#8b6da5'; INK='#26313b'; GREY='#b9bec5'


def text(ax,x,y,s,size=6,**kw):
    ax.text(x,y,s,fontsize=size,color=kw.pop('color',INK),va=kw.pop('va','center'),
            linespacing=1.25,**kw)


def bcell(ax,x,y,r=2,color=BLUE,light=False):
    ax.add_patch(Circle((x,y),r,fc='#f0f5f8' if light else color,ec=color,lw=.6))
    ax.add_patch(Circle((x-.3*r,y+.1*r),.43*r,fc='white',ec=color,lw=.4,alpha=.7))
    for ang in [-.5,.4,1.3]:
        u=np.array([np.cos(ang),np.sin(ang)])
        start=np.array([x,y])+r*u; tip=np.array([x,y])+1.42*r*u
        ax.plot([start[0],tip[0]],[start[1],tip[1]],color=color,lw=.6)
        perp=np.array([-u[1],u[0]])*.28*r
        for sg in [-1,1]:
            end=tip+.3*r*u+sg*perp
            ax.plot([tip[0],end[0]],[tip[1],end[1]],color=color,lw=.6)


def filled_mouse(ax,x,y,s=1):
    sk.mouse(ax,x,y,s,color=INK)
    # Fill the existing anatomically recognisable outline, preserving its eye.
    ax.add_patch(Ellipse((x,y),7*s,4.2*s,fc='#ece8df',ec=INK,lw=.6,zorder=3))


def spleen(ax,x,y,s=1):
    sk.organ(ax,x,y,s,kind='spleen',color=RED,fill='#eed1d3')
    for dx,dy in [(-.8,.3),(.6,.7),(1.1,-.7)]:
        ax.add_patch(Circle((x+dx*s,y+dy*s),.4*s,fc='#f8f4ea',ec=RED,lw=.35,zorder=4))


def timeline(ax,x0,x1,y,days,color=BLUE,top=False):
    ax.plot([x0,x1],[y,y],color=color,lw=.9)
    for d in days:
        x=x0+(x1-x0)*(d-min(days))/(max(days)-min(days))
        ax.plot([x,x],[y-1,y+1],color=color,lw=.7)
        text(ax,x,y+(3 if top else -3),str(d),5.8,ha='center')


def gate(ax,x,y,w=15,h=8,xlabel='mCherry',kind='division',color=BLUE):
    """Illustrate gate topology; intentionally contains no artificial event cloud."""
    ax.plot([x,x,x+w],[y+h,y,y],color=INK,lw=.5)
    if kind=='division':
        for xx,lab,c in [(x+1,'low',BLUE),(x+w-5,'high',GOLD)]:
            ax.add_patch(Rectangle((xx,y+1),4,h-2,fc=c+'20',ec=c,lw=.7))
            text(ax,xx+2,y+h+1,lab,5.2,ha='center',color=c)
    elif kind=='binding':
        ax.add_patch(Polygon([(x+4,y+2),(x+11,y+7),(x+14,y+6),(x+13,y+3),(x+6,y+1)],
                             fc=GREEN+'20',ec=GREEN,lw=.7))
        text(ax,x+10,y+h+1,'RBD+',5.2,ha='center',color=GREEN)
    elif kind=='zone':
        for xx,lab,c in [(x+1,'LZ',GOLD),(x+w-6,'DZ',BLUE)]:
            ax.add_patch(Rectangle((xx,y+1),5,h-2,fc=c+'20',ec=c,lw=.7))
            text(ax,xx+2.5,y+h+1,lab,5.2,ha='center',color=c)
    text(ax,x+w/2,y-2.4,xlabel,5.2,ha='center')


def capture(ax,x,y,s=1):
    # A barcoded droplet links RNA and receptor records from one cell.
    ax.add_patch(Circle((x,y),4*s,fc='#e7f0f6',ec=BLUE,lw=.6))
    bcell(ax,x-1.2*s,y+.4*s,1.6*s,color=GREEN,light=True)
    ax.add_patch(Circle((x+1.7*s,y-.8*s),1.15*s,fc=GOLD,ec=GOLD,lw=.5))
    for i,col in enumerate([RED,VIOLET,GREEN]):
        ax.plot([x+1.2*s,x+2.2*s],[y-1.4*s+i*.6*s]*2,color=col,lw=.7)


def model_antigen(ax):
    sk.canvas(ax,183,68)
    # Mechanism at top: shutdown then dilution, not a fate arrow.
    text(ax,0,65,'H2B-mCherry reporter: stop synthesis, read 36 h of division history',6.7,weight='bold')
    bcell(ax,7,54,2.3,RED)
    text(ax,7,48,'DOX on',5.7,ha='center',color=RED)
    for j,ys in enumerate([[54],[51,57],[49,52.5,56,59.5]]):
        x=27+j*14
        color=[RED,'#d88b90','#ecd2d4'][j]
        for y in ys:bcell(ax,x,y,1.6,color)
    sk.arrow(ax,11,54,23,54,color=INK)
    for y in [51,57]:sk.arrow(ax,29,54,38,y,color=GREY,head=1)
    for y in [49,52.5,56,59.5]:sk.arrow(ax,43,54,52,y,color=GREY,head=1)
    text(ax,69,58,'High: ≤1 division',5.9,color=GOLD)
    text(ax,69,52,'Low: ≥6 divisions',5.9,color=BLUE)
    text(ax,115,59,'Library gates are recorded',6.1,weight='bold')
    capture(ax,120,49,.8)
    sk.arrow(ax,126,49,136,49,color=BLUE)
    text(ax,139,53,'10x GEX + VDJ',6.2,color=BLUE)
    text(ax,139,47,'HTO retains mouse / sort',5.7)
    ax.plot([0,183],[42,42],color='#d9dfe5',lw=.6)
    for y,name,color,n,extra in [(32,'NP-OVA',BLUE,7,'GL7+ Fas+ GC'),
                                (18,'RBD protein',GREEN,5,'RBD bait ± × reporter'),
                                (4,'RBD mRNA',VIOLET,5,'LZ / DZ × reporter')]:
        filled_mouse(ax,5,y+1,.7);sk.syringe(ax,14,y+3,.6,color=color)
        text(ax,18,y+3,name,6.4,color=color,weight='bold')
        text(ax,18,y-2,f'{n} mice',5.6)
        # Correct d12.5 timing shared by the actual reporter experiments.
        ax.plot([51,79],[y,y],color=color,lw=.8)
        for x,lab in [(51,'d0'),(70,'d12.5'),(79,'d14')]:
            ax.plot([x,x],[y-1,y+1],color=color,lw=.6);text(ax,x,y-3,lab,5.4,ha='center')
        text(ax,51,y+4,'Immunise',5.2,ha='center')
        text(ax,70,y+4,'DOX',5.2,ha='center',color=RED)
        sk.organ(ax,90,y+1,1,kind='node',color=color,fill='#edf2f4')
        text(ax,90,y-4,'LN',5.3,ha='center')
        sk.arrow(ax,95,y+1,102,y+1,color=color)
        text(ax,106,y+3,extra,5.8,color=color)
        if name=='NP-OVA':gate(ax,151,y-3,25,8,kind='division')
        elif name=='RBD protein':gate(ax,149,y-3,12,8,'RBD dual bait',kind='binding');gate(ax,168,y-3,12,8,kind='division')
        else:gate(ax,149,y-3,12,8,'CXCR4 / CD86',kind='zone');gate(ax,168,y-3,12,8,kind='division')
    text(ax,0,-5,'Independent NP, protein and mRNA cohorts; gate outlines are schematic, not measured FACS events.',5.8)


def plasmodium(ax):
    sk.canvas(ax,183,54)
    text(ax,0,51,'PcAS blood-stage infection → terminal spleen sampling → paired cell and receptor records',6.7,weight='bold')
    # Biological sample preparation, explicitly matching the original design.
    ax.add_patch(Circle((9,40),5,fc='#f7e7e8',ec=RED,lw=.7))
    for x,y in [(6.5,41),(11.5,41.8),(9,37.8)]:
        ax.add_patch(Ellipse((x,y),3.4,2.3,fc='#e1a3a7',ec=RED,lw=.4))
    ax.add_patch(Circle((6.5,41),.8,fc='none',ec=VIOLET,lw=1))
    text(ax,9,32.5,'Infected RBCs',5.7,ha='center',color=RED)
    sk.syringe(ax,26,40,.9,color=RED);filled_mouse(ax,38,40,1.4)
    text(ax,38,32.5,'C57BL/6',5.7,ha='center')
    sk.arrow(ax,49,40,59,40,color=INK)
    spleen(ax,68,40,1.5);text(ax,68,32.5,'Spleen',5.7,ha='center')
    sk.arrow(ax,75,40,84,40,color=INK)
    # CD19/B220 parent gate and enrichment, represented without fake events.
    ax.plot([89,89,104],[45,37,37],color=INK,lw=.5)
    ax.add_patch(Rectangle((94,39.5),9,6,fc=BLUE+'20',ec=BLUE,lw=.7))
    text(ax,98,46.5,'CD19+ B220int–hi',5.6,ha='center',color=BLUE)
    text(ax,98,32.5,'Live TCRβ− B cells',5.5,ha='center')
    sk.arrow(ax,108,40,117,40,color=INK)
    for dx,dy,col in [(0,0,BLUE),(6,1.5,GREEN),(6,-2.5,GOLD)]:
        bcell(ax,123+dx,40+dy,1.7,col,light=True)
        ax.add_patch(Rectangle((122+dx,37+dy),2.5,1,fc=col,ec='none'))
    text(ax,126,32.5,'Mouse-specific HTO',5.5,ha='center')
    sk.arrow(ax,134,40,143,40,color=INK);capture(ax,150,40,.85)
    text(ax,160,43.5,'10x GEX',6,color=BLUE)
    text(ax,160,37.5,'+ paired VDJ',6,color=GREEN)
    ax.plot([0,183],[29,29],color='#d9dfe5',lw=.6)
    # Each time point is a separate terminal group, not a tracked mouse.
    text(ax,0,25,'Exp1',6.3,color=BLUE,weight='bold')
    text(ax,0,21,'B-cell landscape',5.6)
    timeline(ax,45,100,23,[0,4,7,10,14],BLUE,top=True)
    text(ax,43,16.5,'1 naive',5.4,ha='center');text(ax,60,16.5,'4 infected',5.4,ha='center')
    text(ax,90,16.5,'5 infected / day (d7, 10, 14)',5.4,ha='center')
    text(ax,126,24,'CD19+ B220int–hi',5.8,color=BLUE)
    text(ax,126,20.5,'No IgD-low enrichment',5.6)
    ax.plot([0,183],[14,14],color='#d9dfe5',lw=.6)
    text(ax,0,10.5,'Exp2',6.3,color=GREEN,weight='bold')
    text(ax,0,6.5,'Persistence + treatment',5.5)
    timeline(ax,51,116,5.5,[10,14,21,28,35,42],GREEN,top=False)
    text(ax,44,10.5,'d7',5.6,ha='center',color=RED)
    sk.arrow(ax,44,9,44,6,color=RED,head=1)
    ax.plot([44,116],[5.5,5.5],color=RED,lw=.6,ls='--',zorder=0)
    text(ax,83,11,'3 saline + 3 drug + 1 naive / day',5.6,ha='center')
    text(ax,126,10,'IgD-low B-cell enrichment',5.8,color=GREEN)
    text(ax,126,6.5,'+ IgD-high naive spike-in',5.6)
    text(ax,0,-2,'Drug begins d7: artesunate + pyrimethamine. Different mice at each day; no within-mouse longitudinal tracking.',5.8)
