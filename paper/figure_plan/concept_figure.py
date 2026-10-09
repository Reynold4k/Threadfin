"""Biological concept figure with explicit membership versus phenotype semantics."""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, Polygon, Wedge
import schematics as sk
from experimental_designs import text, INK, GREY

# Match the supplied GC illustration, while retaining editable vector artwork.
BLUE='#6666b2'; RED='#a67cb9'; GREEN='#93bd8b'; GOLD='#e4c666'; VIOLET='#8580bd'


def bcell(ax,x,y,r=2,color=BLUE,light=False):
    base=np.asarray(to_rgb(color)); angles=np.linspace(0,2*np.pi,181)
    radius=r*(1+.04*np.sin(angles*18))
    ax.add_patch(Polygon(np.c_[x+radius*np.cos(angles),y+radius*np.sin(angles)],
                         fc=base*.65+.35,ec=base*.75,lw=.55))
    # Layered shading keeps the soft cell style without editing a raster source.
    for i in range(18):
        q=i/18; col=base*(.64-.25*q)+(1-(.64-.25*q))
        ax.add_patch(Circle((x-.10*r*q,y+.10*r*q),r*(.93-.48*q),fc=col,ec='none'))
    ang=np.linspace(0,2*np.pi,81); rr=.47*r*(1+.07*np.sin(3*ang))
    ax.add_patch(Polygon(np.c_[x-.09*r+rr*np.cos(ang),y-.06*r+rr*np.sin(ang)],
                         fc=base*.9+.1,ec=base*.7,lw=.45))
    ax.add_patch(Ellipse((x-.24*r,y+.29*r),.29*r,.13*r,angle=28,fc='white',ec='none',alpha=.6))
    for ang in [.4,1.2,2.4,4.7]:
        u=np.array([np.cos(ang),np.sin(ang)]);v=np.array([-u[1],u[0]])
        st=np.array([x,y])+r*u; tip=st+.30*r*u
        ax.plot([st[0],tip[0]],[st[1],tip[1]],color=color,lw=.5)
        for sign in [-1,1]:
            en=tip+.21*r*u+sign*.12*r*v
            ax.plot([tip[0],en[0]],[tip[1],en[1]],color=color,lw=.5)


def capture(ax,x,y,s=1):
    ax.add_patch(Circle((x,y),4.8*s,fc='#e9e8f4',ec=BLUE,lw=.7))
    bcell(ax,x-1*s,y+.4*s,2.2*s,color=BLUE)
    ax.add_patch(Circle((x+2.2*s,y-1.7*s),1.25*s,fc=GOLD,ec='#b8a256',lw=.5))
    for i in range(3):ax.plot([x+1.5*s,x+2.8*s],[y-2.1*s+i*.5*s]*2,color=VIOLET,lw=.6)


def gc_reaction(ax):
    sk.canvas(ax,183,72)
    # A follicular compartment, not four rectangular pathway nodes.
    ax.add_patch(Ellipse((48,34),78,58,fc='#f8fafb',ec=INK,lw=.8))
    ax.add_patch(Ellipse((31,31),33,43,fc='#dfebf5',ec='none'))
    ax.add_patch(Ellipse((63,36),32,46,fc='#e5f0e9',ec='none'))
    text(ax,27,62,'Dark zone',7,color=BLUE,weight='bold',ha='center')
    text(ax,67,65,'Light zone',7,color=GREEN,weight='bold',ha='center')
    # Several proliferating GC B cells, pairs suggest division.
    for x,y in [(21,23),(25,29),(32,21),(37,28),(30,37),(21,39)]:bcell(ax,x,y,2.2,BLUE,light=True)
    for x,y in [(36,40),(37.5,44)]:bcell(ax,x,y,1.7,BLUE,light=True)
    text(ax,29,13,'Division + SHM',6.1,ha='center',color=BLUE)
    text(ax,29,8,'CXCR4 / MKI67 / AICDA',5.3,ha='center')
    # Branched FDC carrying antigen and a Tfh contact.
    for dx,dy in [(8,0),(5,6),(0,9),(-5,6),(-8,0)]:
        ax.plot([63,63+dx],[42,42+dy],color=GREEN,lw=.9)
        ax.add_patch(Circle((63+dx,42+dy),.75,fc=GOLD,ec=GOLD,lw=.4))
    text(ax,63,55,'Antigen on FDC',5.5,ha='center',color=GREEN)
    bcell(ax,59,32,2.7,GREEN,light=True)
    ax.add_patch(Circle((70,30),3,fc='#ece3f2',ec=VIOLET,lw=.7))
    ax.add_patch(Circle((70,30),1.2,fc='white',ec=VIOLET,lw=.4))
    ax.plot([62,67],[31,30],color=VIOLET,lw=1.2)
    text(ax,74,27,'Tfh',5.5,color=VIOLET)
    text(ax,64,18,'Antigen capture + T-cell help',5.9,ha='center',color=GREEN)
    text(ax,64,12,'Selection-associated states',5.9,ha='center')
    sk.arrow(ax,50,42,40,42,color=BLUE,head=2)
    sk.arrow(ax,39,21,49,21,color=GREEN,head=2)
    text(ax,45,47,'Selected cells',5.1,ha='center')
    text(ax,44,17,'Cycle',5.2,ha='center')
    # Outputs have cells and biological readouts, with no inferred origin arrow.
    text(ax,135,63,'Captured output-like states',6.7,weight='bold',ha='center')
    ax.add_patch(Ellipse((117,45),10,7,fc='#f2d7d9',ec=RED,lw=.8))
    ax.add_patch(Circle((115,45),2,fc='white',ec=RED,lw=.6))
    for i in range(3):ax.plot([117+i*.7,120+i*.7],[42+i*.6,43+i*.6],color=RED,lw=.6)
    text(ax,127,49,'Plasmablast / plasma cell',6.3,color=RED)
    text(ax,127,41,'PRDM1 / XBP1 / MZB1',5.7)
    bcell(ax,117,24,3,GREEN,light=True)
    text(ax,127,28,'Memory-like cell',6.3,color=GREEN)
    text(ax,127,20,'CCR6 / GPR183 / KLF2',5.7)
    # Non-directional relation: ancestry tested with paired BCR.
    ax.plot([89,99,99],[34,34,47],color=VIOLET,lw=.8,ls=':')
    ax.plot([99,110],[47,47],color=VIOLET,lw=.8,ls=':')
    ax.plot([99,99,110],[34,24,24],color=VIOLET,lw=.8,ls=':')
    text(ax,128,6,'Same-family co-occupancy is testable.\nThe sampled cells do not establish output direction.',5.8,ha='center')


def paired_records(ax):
    sk.canvas(ax,86,57)
    capture(ax,9,42,1.1)
    text(ax,9,32,'One cell',6,ha='center')
    sk.arrow(ax,15,43,23,48,color=BLUE)
    sk.arrow(ax,15,41,23,33,color=GREEN)
    text(ax,26,49,'RNA: current state',6.5,color=BLUE,weight='bold')
    text(ax,26,42,'IG genes excluded from embedding',5.4)
    text(ax,26,33,'BCR: family membership',6.5,color=GREEN,weight='bold')
    text(ax,26,26,'Sequence calls within the same donor',5.4)
    # Four tags denote sample/mouse, not additional predictions.
    for i,(label,col) in enumerate([('Mouse',GOLD),('Date',BLUE),('Tissue',RED),('Gate',VIOLET)]):
        x=2+i*21
        ax.add_patch(Polygon([(x,10),(x+16,10),(x+19,14),(x+16,18),(x,18)],fc=col+'20',ec=col,lw=.6))
        text(ax,x+9,14,label,5.8,ha='center')
    text(ax,0,3,'Sampling records define which comparisons are possible.',5.8)


def families_and_states(ax):
    sk.canvas(ax,83,57)
    # State colour, family symbol: membership is not a UMAP distance.
    for x,y,w,h,col,label in [(14,34,23,24,BLUE,'GC'),(43,38,22,18,RED,'PB'),(67,28,23,23,GREEN,'Memory')]:
        ax.add_patch(Ellipse((x,y),w,h,fc=col+'15',ec='none'))
        text(ax,x,y+h/2+4,label,6.3,color=col,ha='center')
    coords=[(8,29,'o',BLUE),(13,36,'o',BLUE),(18,32,'s',BLUE),(11,41,'^',BLUE),
            (39,35,'o',RED),(45,39,'s',RED),(48,35,'s',RED),
            (61,24,'^',GREEN),(68,30,'^',GREEN),(71,23,'o',GREEN)]
    for x,y,m,c in coords:
        bcell(ax,x,y,2.1,c)
        letter={'o':'A','s':'B','^':'C'}[m]
        text(ax,x,y,letter,4.7,ha='center',color='white',weight='bold')
    ax.plot([13,39,71],[36,35,23],color=INK,lw=.6,ls=':',zorder=0)
    text(ax,0,13,'Letter = BCR family; colour = expression state.',5.8)
    text(ax,0,5,'One family may span states. Nearby cells may be unrelated.',5.8)


def profile_map(ax):
    sk.canvas(ax,183,45)
    text(ax,0,41,'Captured distributions',6.6,weight='bold')
    profiles=[(.75,.15,.10),(.15,.75,.10),(.35,.15,.50)]
    for i,fr in enumerate(profiles):
        x=12+i*25;angle=90
        for val,col in zip(fr,[BLUE,RED,GREEN]):
            ax.add_patch(Wedge((x,23),7,angle,angle+360*val,width=2.3,fc=col,ec='white',lw=.5));angle+=360*val
        text(ax,x,23,chr(65+i),6.2,ha='center')
        text(ax,x,11,['GC-biased','PB-biased','Memory-biased'][i],5.7,ha='center')
    sk.arrow(ax,79,23,91,23,color=INK)
    text(ax,86,8,'Context-centred\nclone profiles',5.6,ha='center')
    # Conceptual profile map. The points are families, not cells in a tree.
    for i,(x,y,col) in enumerate([(101,27,BLUE),(107,30,BLUE),(111,24,BLUE),(113,31,BLUE),
                                (125,18,GREEN),(130,22,GREEN),(133,18,GREEN),
                                (145,31,RED),(148,35,RED),(153,28,RED),(157,33,RED)]):
        bcell(ax,x,y,1.7+(i%3)*.18,col)
    text(ax,125,41,'Compare different families',6.6,weight='bold',ha='center')
    text(ax,102,10,'Similar profiles → similar captured state bias',5.7)
    text(ax,102,3,'Distances do not imply common ancestry or future fate.',5.7)
    text(ax,0,0,'Illustrative distributions and positions; not experimental results.',5.5,color='#697580')


def biological_questions(ax):
    sk.canvas(ax,183,28)
    entries=[('GC selection / division','Reporter + FACS labels','Fig. 2',BLUE),
             ('GC/output co-occupancy','Same-mouse BCR + controls','Fig. 3',RED),
             ('GC persistence','Repeated same-donor sampling','Fig. 4',GREEN),
             ('Clonal information gain','Observed vs shuffled families','Fig. 5',VIOLET)]
    for i,(title,evidence,fig,col) in enumerate(entries):
        x=i*47
        bcell(ax,x+3,21,2.7,col)
        text(ax,x+3,21,str(i+1),6,ha='center',color='white',weight='bold')
        text(ax,x+9,21,fig,6.3,color=col,weight='bold')
        text(ax,x,12,title,6.1,color=col)
        text(ax,x,5,evidence,5.5)


def family_profile(ax,x,y,fractions,r=3.4,label=None):
    """An illustrative state distribution, with the supplied cell-art palette."""
    bcell(ax,x,y,r*.76,VIOLET)
    angle=90
    for value,color in zip(fractions,[BLUE,GOLD,RED,GREEN]):
        ax.add_patch(Wedge((x,y),r,angle,angle+360*value,width=r*.23,
                          fc=color,ec='white',lw=.25,zorder=6))
        angle+=360*value
    if label:text(ax,x,y,label,5.1,ha='center',color='white',weight='bold')


def integration_concept(ax):
    """Paired cell-state and sequence spaces yield family distributions.

    No learned BCR encoder is implied: the sequence-space sketch establishes
    family membership. Positions, state fractions and examples are illustrative.
    """
    sk.canvas(ax,183,86)
    text(ax,3,82,'scRNA embedding',8,weight='bold',color=BLUE)
    text(ax,3,39,'scBCR sequence space',8,weight='bold',color=VIOLET)
    text(ax,130,82,'Clone embedding',8,weight='bold',color=VIOLET)
    rng=np.random.default_rng(172)
    # RNA: cells group by captured expression state, regardless of family.
    clouds=[(16,62,12,10,BLUE,'DZ'),(35,70,13,10,GOLD,'LZ'),
            (53,58,12,10,RED,'PC'),(41,47,13,8,GREEN,'Memory')]
    focus=[('A',0,0,0),('B',0,3,-2),('C',1,0,0),
           ('A',1,-3,-1),('A',2,0,0),('B',2,3,2),('C',3,0,0)]
    for x,y,w,h,col,lab in clouds:
        ax.add_patch(Ellipse((x,y),w+5,h+4,fc=col+'11',ec='none'))
        pts=rng.normal(size=(12,2))*[w/4,h/4]+[x,y]
        ax.scatter(pts[:,0],pts[:,1],s=5,color=col,alpha=.40,linewidths=0)
        text(ax,x,y+h/2+3,lab,5.7,ha='center',color=col)
    for lab,i,dx,dy in focus:
        x,y,_,_,col,_=clouds[i];bcell(ax,x+dx,y+dy,1.6,col)
        text(ax,x+dx,y+dy,lab,4.3,ha='center',color='white',weight='bold')
    # BCR: undirected sequence neighbourhoods within three distinct families.
    seq=[(17,19,BLUE,'A'),(37,25,RED,'B'),(54,13,GREEN,'C')]
    for x,y,col,lab in seq:
        ax.add_patch(Ellipse((x,y),17,16,fc=col+'0b',ec='none'))
        points=np.array([[x-3,y-1],[x,y+3],[x+4,y],[x+1,y-4]])
        for i,j in [(0,1),(1,2),(0,3),(2,3)]:
            ax.plot(points[[i,j],0],points[[i,j],1],color=col,lw=.6,alpha=.65)
        for px,py in points:
            bcell(ax,px,py,1.55,VIOLET)
            text(ax,px,py,lab,4.2,ha='center',color='white',weight='bold')
    # Two inputs converge on a barcode-matched family, not concatenated UMAPs.
    sk.arrow(ax,69,62,87,46,color=BLUE,head=2)
    sk.arrow(ax,69,22,87,39,color=VIOLET,head=2)
    family_profile(ax,97,43,[.48,.25,.17,.10],r=5.4,label='A')
    text(ax,97,58,'Threadfin',7.8,ha='center',weight='bold',color=VIOLET)
    # Short barcode visual encodes the matching of modalities for the same cell.
    for i,width in enumerate([.4,.8,.4,1,.5,.6,.4,.8]):
        ax.plot([89+i*2,89+i*2],[32,35],color=VIOLET,lw=width)
    text(ax,97,25,'Paired cells',5.8,ha='center',color=INK)
    sk.arrow(ax,105,43,120,43,color=VIOLET,head=2)
    # One point per family; neighbours have similar captured distributions.
    for x,y,w,h,col in [(139,57,25,24,BLUE),(164,52,24,25,RED),(149,24,27,23,GREEN)]:
        ax.add_patch(Ellipse((x,y),w,h,fc=col+'0e',ec='none'))
    dots=[(133,55,[.65,.2,.1,.05],'A'),(142,63,[.70,.12,.10,.08],None),
          (143,50,[.6,.25,.10,.05],None),(129,64,[.55,.3,.10,.05],None),
          (161,57,[.05,.1,.8,.05],'B'),(171,51,[.10,.05,.75,.10],None),
          (160,46,[.1,.10,.70,.10],None),(145,29,[.1,.15,.1,.65],'C'),
          (155,21,[.15,.1,.05,.70],None),(141,18,[.1,.1,.1,.70],None)]
    for x,y,fr,lab in dots:family_profile(ax,x,y,fr,r=3.3,label=lab)
    # Compact visual key; detailed computation and assumptions belong in legend.
    for x,col,lab in [(4,BLUE,'DZ'),(18,GOLD,'LZ'),(31,RED,'PC'),(44,GREEN,'Memory')]:
        ax.add_patch(Circle((x,1.5),.65,fc=col,ec='none'))
        text(ax,x+1.7,1.5,lab,5.3)
    text(ax,130,5,'One point = one family',6, color=INK)


def capabilities_concept(ax):
    """Minimal visual outputs, each connected to an experimental anchor."""
    sk.canvas(ax,183,63)
    # A hub surrounded by independent questions, without arrows suggesting fate.
    for x,y,fr in [(88,34,[.7,.15,.1,.05]),(96,33,[.1,.1,.7,.1]),(91,25,[.15,.1,.1,.65])]:
        family_profile(ax,x,y,fr,r=3.8)
    text(ax,92,18,'Clone profiles',6.5,ha='center',color=VIOLET)
    for xx,yy in [(63,47),(119,47),(63,14),(119,14)]:
        sk.arrow(ax,82 if xx<90 else 103,36 if yy>30 else 27,xx,yy,color=VIOLET,head=1.7)
    # Division-associated state: reporter dilution depicted, not a future arrow.
    text(ax,2,58,'GC state bias',7.3,weight='bold',color=BLUE)
    for x,y,r,c in [(12,46,2.7,RED),(23,47,2.1,'#c5acd0'),(29,44,1.8,'#dfd2e7'),(32,49,1.7,'#dfd2e7')]:bcell(ax,x,y,r,c)
    text(ax,2,35,'NP-OVA / RBD · Fig. 2',6.1,color=INK)
    # Shared states: the same A receptor identity in GC and PC, undirected.
    text(ax,127,58,'Shared clone states',7.3,weight='bold',color=RED)
    bcell(ax,137,46,2.8,BLUE);bcell(ax,160,46,2.8,RED)
    for x in [137,160]:text(ax,x,46,'A',5.1,ha='center',color='white',weight='bold')
    ax.plot([142,155],[46,46],color=VIOLET,lw=.8,ls=':')
    text(ax,137,40,'GC',5.5,ha='center',color=BLUE);text(ax,160,40,'PB',5.5,ha='center',color=RED)
    text(ax,127,35,'PcAS · Fig. 3',6.1,color=INK)
    # Persistence: separate observations of A, not a continuous cell lineage.
    text(ax,2,24,'Persistence',7.3,weight='bold',color=GREEN)
    ax.plot([11,50],[11,11],color=GREY,lw=.6)
    for x in [12,29,47]:
        bcell(ax,x,11,2.2,GREEN);text(ax,x,11,'A',4.8,ha='center',color='white',weight='bold')
    text(ax,2,1,'Human GC · Fig. 4',6.1,color=INK)
    # Clonal signal versus shuffled identity: two small comparable landscapes.
    text(ax,127,24,'Clonal signal',7.3,weight='bold',color=VIOLET)
    for origin,mixed in [(133,False),(163,True)]:
        for i,(dx,dy) in enumerate([(0,0),(3,2),(1,4),(9,0),(12,2),(10,4)]):
            col=[BLUE,GREEN,RED][i%3] if mixed else BLUE if i<3 else GREEN
            ax.add_patch(Circle((origin+dx,10+dy),.8,fc=col,ec='none'))
    text(ax,138,6,'Observed',5.2,ha='center');text(ax,168,6,'Shuffled',5.2,ha='center')
    text(ax,127,1,'12 datasets · Fig. 5',6.1,color=INK)
