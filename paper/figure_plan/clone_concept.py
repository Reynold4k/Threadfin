"""A traceable example: the same 18 cells become three family distributions.

Artwork is conceptual. RNA state is colour; A/B/C identifies sequence-defined
families. Null panels preserve RNA positions/colours and change family labels.
"""
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch
import schematics as sk
from concept_figure import bcell, family_profile, BLUE, RED, GREEN, GOLD, VIOLET
from experimental_designs import INK, GREY

COLORS=[BLUE,GOLD,RED,GREEN]
LABEL_COLORS=['#505095','#a88a28','#80569a','#56834e']
MEMBERS={'A':[0,0,0,1,1,2], 'B':[1,2,2,2,2,3], 'C':[1,1,3,3,3,3]}
FRACTIONS={key:np.bincount(states,minlength=4)/6 for key,states in MEMBERS.items()}

def label(ax,x,y,s,size=6,color=INK,**kwargs):
    ax.text(x,y,s,fontsize=size,color=color,va='center',**kwargs)

def cell(ax,x,y,state,family=None,r=2.1):
    color=COLORS[state]
    if state==2:
        # Plasma-cell morphology: eccentric nucleus and secretory membranes.
        ax.add_patch(Ellipse((x,y),2.45*r,1.85*r,angle=-22,fc='#dbc9e4',ec=RED,lw=.6))
        ax.add_patch(Circle((x-.40*r,y),.53*r,fc=RED,ec=VIOLET,lw=.4))
        for j in range(3):
            a=np.linspace(-1,1,25)
            ax.plot(x+.2*r+(j*.21*r)+.08*r*np.cos(a*2),y+a*r*.48,color=RED,lw=.55)
    else:bcell(ax,x,y,r,color)
    if family:
        label(ax,x-.35*r if state==2 else x,y,family,5.1,color='white',ha='center',fontweight='bold')

def barcode(ax,x,y,family,w=12,h=2):
    # Stylised receptor family signatures, not literal sequences or a classifier.
    pattern={'A':[1,0,1,1,0,1,0,1],'B':[1,1,0,1,0,0,1,1],'C':[0,1,1,0,1,1,0,1],
             'F':[1,0,0,1,1,0,0,1]}[family]
    for i,value in enumerate(pattern):
        ax.plot([x+i*w/7]*2,[y-h/2,y+h/2],color=VIOLET if value else '#d9d4e7',lw=1 if value else .45)

def integration_story(ax):
    sk.canvas(ax,183,112)
    # Four aligned steps give the reader a left-to-right route through the figure.
    for x,title in [(0,'1  Paired cells'),(60,'2  BCR families'),(103,'3  RNA profiles'),(144,'4  Clone map')]:
        label(ax,x,108,title,6.8,color=VIOLET,fontweight='bold')
    label(ax,0,99,'scRNA embedding',6.5,color=BLUE)
    # These labels and counts exactly correspond to the three member collections.
    clouds=[(13,83,22,20,0,'DZ'),(37,84,23,25,1,'LZ'),
            (35,60,27,23,2,'PC'),(12,60,23,24,3,'Memory')]
    placements={0:[(-6,0),(0,3),(6,-1)],
                1:[(-6,3),(0,7),(6,3),(-3,-4),(4,-5)],
                2:[(-8,4),(-1,6),(7,3),(-5,-4),(3,-4)],
                3:[(-6,4),(1,5),(7,0),(-4,-4),(3,-4)]}
    families={0:['A','A','A'],1:['A','A','B','C','C'],
              2:['A','B','B','B','B'],3:['B','C','C','C','C']}
    for x,y,w,h,state,name in clouds:
        ax.add_patch(Ellipse((x,y),w,h,fc=COLORS[state]+'10',ec='none'))
        label(ax,x,y+h/2+1,name,6.1,color=LABEL_COLORS[state],ha='center')
        for (dx,dy),family in zip(placements[state],families[state]):cell(ax,x+dx,y+dy,state,family,r=1.85)
    label(ax,0,41,'scBCR sequences',6.5,color=VIOLET)
    # Matched sequence records below the same cells; family calling stays separate.
    for i,family in enumerate('ABC'):
        y=31-i*11
        label(ax,2,y,family,6.7,color=VIOLET,fontweight='bold')
        barcode(ax,10,y,family,w=19,h=3)
        barcode(ax,34,y,family,w=12,h=3)
    label(ax,0,0,'Colour = state    Letter = family',5.7)
    # Sequence defines families; RNA supplies their captured-state profiles.
    ax.add_patch(FancyArrowPatch((47,95),(109,99),connectionstyle='arc3,rad=-.12',
                               arrowstyle='-|>',mutation_scale=6,color=BLUE,lw=.8))
    ax.add_patch(FancyArrowPatch((50,21),(59,42),arrowstyle='-|>',mutation_scale=6,color=VIOLET,lw=.8))
    for i,family in enumerate('ABC'):
        y=85-i*35
        ax.add_patch(FancyBboxPatch((62,y-13),30,25,boxstyle='round,pad=.4,rounding_size=4',
                                  fc='#faf9fc',ec='#d6cfe4',lw=.65))
        label(ax,77,y+19,'Clone '+family,6.2,color=VIOLET,ha='center',fontweight='bold')
        for j,state in enumerate(MEMBERS[family]):
            cell(ax,68+(j%3)*8,y+4-(j//3)*9,state,family,r=2.0)
        sk.arrow(ax,94,y,104,y,color=VIOLET,head=1.6)
        family_profile(ax,117,y,FRACTIONS[family],r=6.1,label=family)
        label(ax,117,y-12,['GC-rich','PC-rich','Memory-rich'][i],6.2,ha='center',color=LABEL_COLORS[[0,2,3][i]])
    # Keep the mapped examples recognisable while other families form the background.
    rng=np.random.default_rng(7)
    centres=[(151,84),(172,50),(150,17)]
    for x,y in centres:
        points=rng.normal(size=(8,2))*[4,5]+[x,y]
        ax.scatter(points[:,0],points[:,1],s=6,c='#bbb3cf',alpha=.6,linewidths=0)
    for family,(x,y),sy in zip('ABC',centres,[85,50,15]):
        sk.arrow(ax,125,sy,x-6,y,color=VIOLET,head=1.6)
        family_profile(ax,x,y,FRACTIONS[family],r=4.4,label=family)
    label(ax,145,0,'One point = one family',5.7)
    label(ax,65,-7,'RNA distributions → context adjustment → reliability → clone embedding',5.5)

def capabilities_story(ax):
    sk.canvas(ax,183,62)
    # Each output has an interpretable picture and one experimental anchor.
    cards=[(0,BLUE,'GC state bias','NP-OVA / RBD · Fig. 2'),
           (47,RED,'GC–PB sharing','PcAS · Fig. 3'),
           (94,GREEN,'Repeated capture','Human GC · Fig. 4'),
           (141,VIOLET,'Clone coherence','12 analyses · Fig. 5')]
    for x,col,title,source in cards:
        ax.add_patch(FancyBboxPatch((x,2),42,56,boxstyle='round,pad=.25,rounding_size=2.8',fc=col+'08',ec=col+'60',lw=.6))
        label(ax,x+21,53,title,6.8,color=LABEL_COLORS[0] if x==0 else col,ha='center',fontweight='bold')
        label(ax,x+21,7,source,5.8,ha='center')
    # State bias: the same measured reporter assigns the two families differently.
    # Different illustrative IDs prevent readers from linking mouse/human
    # examples to the A/B/C families in the general concept or across studies.
    for x,family,color in [(11,'D',BLUE),(31,'E',GOLD)]:
        ax.add_patch(Ellipse((x,35),16,15,fc=color+'16',ec='none'))
        for dx,dy in [(-3,0),(2,3),(3,-3)]:cell(ax,x+dx,35+dy,0,family,r=1.65)
        label(ax,x,23,'mCherry-low' if family=='D' else 'mCherry-high',5.3,color=INK,ha='center')
        label(ax,x,19,'≥6 divisions' if family=='D' else '<6 divisions',5.2,color=INK,ha='center')
    label(ax,21,13,'Measured reporter gates',5.2,ha='center')
    # The same family in two states, without a directed differentiation edge.
    cell(ax,57,35,0,'F',r=3.1);cell(ax,79,35,2,'F',r=3.1)
    barcode(ax,61,35,'F',w=13,h=2)
    label(ax,57,25,'GC',5.8,color=BLUE,ha='center');label(ax,79,25,'PB',5.8,color=RED,ha='center')
    label(ax,68,15,'Same-mouse family in GC + PB',5.2,ha='center')
    # A snapshot timeline rather than a lineage arrow between cells.
    ax.plot([100,130],[33,33],color=GREY,lw=.6)
    for x,day in [(101,60),(115,110),(129,201)]:
        cell(ax,x,35,0,'G',r=2.1)
        label(ax,x,25,'d'+str(day),5.2,ha='center')
    label(ax,115,15,'Same-donor family, repeat captures',5.2,ha='center')
    # A proper identity-shuffle illustration: expression space stays identical.
    coords=[(-3,0),(0,3),(3,-1),(8,1),(11,4),(13,0)]
    for origin,assign in [(149,'HHHJJJ'),(171,'HJJHJH')]:
        for i,((dx,dy),family) in enumerate(zip(coords,assign)):
            state=0 if i<3 else 3
            cell(ax,origin+dx*.65,34+dy*1.15,state,family,r=1.15)
        # Circle the same three H-labelled cells, whose state purity changes.
        for (dx,dy),family in zip(coords,assign):
            if family=='H':ax.add_patch(Circle((origin+dx*.65,34+dy*1.15),1.65,fc='none',ec=GOLD,lw=.65))
    label(ax,151,22,'Observed',5.4,ha='center');label(ax,174,22,'Shuffled',5.4,ha='center')
    label(ax,162,15,'Same cells, shuffled families',5.2,ha='center')
    # A ribbon from family profiling to the four independently anchored readouts.
    ax.plot([21,162],[62,62],color=VIOLET,lw=.7)
    for x in [21,68,115,162]:sk.arrow(ax,x,62,x,59,color=VIOLET,head=1.4)
