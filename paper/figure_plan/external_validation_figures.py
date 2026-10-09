"""Main Figures 4–6: lineage validation, longitudinal TCRs, measured benchmarks."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import r2_score

METHODS=["Training_mean","RNA_mean","Threadfin_mean","Threadfin_kernel"]
LABELS={"Training_mean":"Training mean","RNA_mean":"RNA mean","RNA_mean_variance":"RNA mean + variance","Size_only":"Capture count",
        "RNA_centroid":"RNA mean","RNA_context_centroid":"Context RNA mean",
        "Threadfin_mean":"Threadfin mean","Threadfin_kernel":"Threadfin kernel",
        "BiGCN":"BiGCN","clone2vec":"clone2vec","Benisse":"Benisse"}
STATE_COLORS={"Undifferentiated":"#b9bec5","Monocyte":"#3478ad","Neutrophil":"#c65359",
              "Baso":"#b394c0","Mast":"#d9a643","Meg":"#987558","Lymphoid":"#4b9a8b",
              "Erythroid":"#e58472","Eos":"#8bb9ce","Ccr7_DC":"#617456","pDC":"#776b9e"}
def read(folder,name):
    return pd.read_csv(folder/name,index_col=0)
def require(folder):
    summary=json.loads((folder/"summary.json").read_text())
    if summary["status"]!="complete":raise ValueError(f"Incomplete analysis: {folder}")
    return summary
def finish(fig,b,name,sources):
    b.save(fig,name);b.AUDIT["outputs"].append(name)
    b.AUDIT[name]={"source_files":{str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  "inference":"Quantitative tests use feature coordinates or held-out predictions, not UMAP geometry."}
def barlegend(ax,palette,y=-.04,ncol=3):
    ax.legend(handles=[Line2D([],[],marker="o",ls="",ms=3,color=c,label=k) for k,c in palette.items()],
              frameon=False,ncol=ncol,fontsize=5.5,loc="upper left",bbox_to_anchor=(0,y),
              handletextpad=.3,columnspacing=.8)
def scatter_map(ax,b,t,color,cmap=None,vmin=None,vmax=None):
    im=ax.scatter(t.x,t.y,c=color,cmap=cmap,vmin=vmin,vmax=vmax,
                  s=1+1.8*np.sqrt(t.n_cells),linewidths=0,alpha=.8,rasterized=True)
    b.map_axis(ax);b.clone_axis(ax);return im
def colorbar(fig,ax,im,label):
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.045,pad=.08,aspect=28)
    cb.set_label(label,fontsize=5.6);cb.ax.tick_params(labelsize=5.3,length=2)
def timeline(ax,b,labels,subtitle):
    ax.set_axis_off()
    for i,(title,text,col) in enumerate(labels):
        x=.03+i*.33
        ax.add_patch(FancyBboxPatch((x,.23),.27,.61,boxstyle="round,pad=.015,rounding_size=.025",
                                   transform=ax.transAxes,facecolor=col+"15",edgecolor=col,lw=.7))
        ax.text(x+.135,.64,title,transform=ax.transAxes,ha="center",fontsize=7,color=col,fontweight="bold")
        ax.text(x+.135,.40,text,transform=ax.transAxes,ha="center",va="center",fontsize=5.8)
        if i<2:ax.annotate("",xy=(x+.32,.53),xytext=(x+.285,.53),xycoords="axes fraction",
                          arrowprops=dict(arrowstyle="-|>",lw=.8,color=b.GREY))
    ax.text(.02,-.01,subtitle,transform=ax.transAxes,fontsize=5.7,color=b.INK)

def draw_larry(b):
    folder=b.DATA/"clonotrace_larry";s=require(folder)
    cells=read(folder,"cell_metadata.csv.gz")
    t=read(folder,"clone_umap_kernel.csv.gz")
    fig=b.new_page("Figure 4","Lineage barcodes connect early clone states to later observations",
                   "LARRY mouse haematopoiesis | GSE140802 | future observations are excluded from the prediction features.",height=284)
    a=b.p(fig,0,5,183,27,"A","One lineage barcode, separately observed at each sampling day")
    timeline(a,b,[("Day 2","Early RNA profiles",b.GREY),("Day 4","Intermediate sampling",b.GOLD),
                  ("Day 6","Observed state composition",b.RED)],
             f"{s['n_input_cells']:,} input cells; {s['n_unique_barcode_cells']:,} barcode-labelled cells; 172 prediction-eligible barcodes.")
    ax=b.p(fig,0,47,79,75,"B","LARRY authors’ cell SPRING map")
    show=cells.sample(min(40000,len(cells)),random_state=31)
    ax.scatter(show.spring_x,show.spring_y,s=1,c=show.state.map(STATE_COLORS),lw=0,alpha=.65,rasterized=True)
    b.map_axis(ax)
    barlegend(ax,STATE_COLORS,-.05,3)
    b.note(ax,"Author SPRING coordinates; 40,000 cells displayed.\nAll eleven author annotations are retained.",-.29,5.6)
    ax=b.p(fig,97,47,86,75,"C","Threadfin kernel: clone-by-day profiles")
    fractions=t.filter(regex="^state:")
    dominant=fractions.idxmax(axis=1).str.replace("state:","",regex=False)
    scatter_map(ax,b,t,dominant.map(STATE_COLORS))
    # Selection depends only on capture coverage, never on attractive geometry.
    eligible=t.groupby("barcode").filter(lambda z:set(z.day)=={2,4,6})
    top=eligible.groupby("barcode").n_cells.min().sort_values(ascending=False).head(3).index
    for k in top:
        q=t[t.barcode.eq(k)].sort_values("day")
        ax.plot(q.x,q.y,color=b.INK,lw=.7,alpha=.9,zorder=3)
        for _,r in q.iterrows():ax.text(r.x,r.y,str(int(r.day)),fontsize=5.3,color=b.INK,zorder=4)
    b.note(ax,f"{len(t):,} clone-day profiles; colour = dominant captured state.\nLines join the same barcode; labels denote day, not inferred pseudotime.",-.12,5.4)
    ax=b.p(fig,8,160,73,66,"D","Do early profiles anticipate later neutrophil capture?")
    pred=pd.read_csv(folder/"future_predictions.csv.gz")
    q=pred[pred["repeat"].eq(0)&pred.method.eq("Threadfin_kernel")&pred.target.eq("Neutrophil")]
    ax.scatter(q.observed,q.predicted,s=10,color=b.RED,alpha=.65,linewidths=0)
    ax.plot([0,1],[0,1],ls="--",color=b.GREY,lw=.8)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel="Observed day-6 neutrophil fraction",ylabel="Out-of-fold predicted fraction")
    ax.text(.05,.94,f"172 held-out barcodes\nPooled R² = {r2_score(q.observed,q.predicted):.2f}",
            transform=ax.transAxes,va="top",fontsize=6)
    ax=b.p(fig,110,160,73,66,"E","Early-state prediction with RNA baselines")
    scores=pd.read_csv(folder/"future_scores.csv")
    order=["Training_mean","Size_only","RNA_mean","RNA_mean_variance","Threadfin_mean","Threadfin_kernel"]
    for j,(target,col) in enumerate([("Monocyte",b.BLUE),("Neutrophil",b.RED)]):
        for i,method in enumerate(order):
            v=scores[scores.target.eq(target)&scores.method.eq(method)].r2
            ax.errorbar(v.median(),i+(j-.5)*.22,xerr=[[v.median()-v.quantile(.25)],[v.quantile(.75)-v.median()]],
                        fmt="o",ms=3,color=col,lw=.9,capsize=1.5)
    ax.set_yticks(range(len(order)),[LABELS[m] for m in order],fontsize=5.8);ax.invert_yaxis()
    ax.axvline(0,ls="--",color=b.GREY,lw=.8);ax.set_xlabel("Out-of-fold R² (median and IQR)",fontsize=6)
    barlegend(ax,{"Monocyte":b.BLUE,"Neutrophil":b.RED},-.22,2)
    foot=b.p(fig,0,253,183,10,None,"");foot.set_axis_off()
    foot.text(0,1,"D–E: ≥2 day-2 and ≥4 day-6 cells per barcode. Three repeated five-fold splits; repeats are not biological replicates.\n"
              "Later RNA-defined state fractions validate temporal association; they do not measure a clone’s complete developmental potential.",
              va="top",fontsize=5.7,linespacing=1.5)
    finish(fig,b,"Figure_4",[folder/"summary.json",folder/"clone_umap_kernel.csv.gz",folder/"future_scores.csv",folder/"future_predictions.csv.gz"])

def draw_nsclc(b):
    folder=b.DATA/"clonotrace_nsclc";s=require(folder)
    t=read(folder,"clone_umap_kernel.csv.gz")
    fig=b.new_page("Figure 5","Exact paired TCRs track changes in captured clone states",
                   "NSCLC blood CD8 T cells | GSE266219 | ten public patients; no response-group assignments inferred.",height=305)
    ax=b.p(fig,0,5,183,25,"A","Retain longitudinal identity without treating distribution shifts as differentiation")
    timeline(ax,b,[("Paired TCR","Patient-specific TRA + TRB",b.GREEN),("Clone × cycle","Separate RNA profiles",b.BLUE),
                   ("Repeated capture","Same receptor, later sample",b.RED)],
             f"{s['n_cells_exact_paired_tcr']:,} cells with one TRA and one TRB; {s['profiles']['kernel']['n_profiles']:,} expanded clone-cycle profiles.")
    ax=b.p(fig,0,48,64,70,"B","Cell-level expression context")
    cell=read(folder,"cell_umap.csv.gz");v=cell["module:cytotoxic"]-cell["module:memory"]
    im=ax.scatter(cell.x,cell.y,s=1,c=v,cmap="coolwarm",vmin=-1.5,vmax=1.5,lw=0,alpha=.75,rasterized=True)
    b.map_axis(ax);colorbar(fig,ax,im,"Cytotoxic − memory RNA score")
    b.note(ax,f"{len(cell):,} cells displayed\nSame RNA modules used in C.",-.31,5.6)
    ax=b.p(fig,80,48,103,85,"C","Clone-state continuum")
    im=scatter_map(ax,b,t,t.cytotoxic-t.memory,"coolwarm",-1.5,1.5)
    colorbar(fig,ax,im,"Mean captured cytotoxic − memory RNA score")
    b.note(ax,f"{len(t):,} reliable clone-cycle profiles; size = captured cells.",-.25,5.6)
    ax=b.p(fig,7,169,73,65,"D","Does exact clone identity retain state information?")
    d=pd.read_csv(folder/"longitudinal_distance_comparison.csv")
    for i,(method,col) in enumerate([("RNA_mean",b.GREY),("Threadfin_mean",b.BLUE),("Threadfin_kernel",b.RED)]):
        z=d[d.method.eq(method)].groupby("patient").distance_ratio.median().sort_index()
        ax.scatter(i+np.linspace(-.12,.12,len(z)),z,s=13,color=col,alpha=.85,edgecolors="white",lw=.25)
        ax.plot([i-.2,i+.2],[z.median()]*2,color=b.INK,lw=1.3)
    ax.axhline(1,color=b.GREY,lw=.8,ls="--")
    ax.set_xticks([0,1,2],["RNA\nmean","Threadfin\nmean","Threadfin\nkernel"],fontsize=5.8)
    ax.set_ylabel("Same-clone / size-matched null distance",fontsize=5.8)
    b.note(ax,"One dot per patient; median across observed intervals.\nBelow 1: closer than a different matched clone.",-.26,5.5)
    ax=b.p(fig,108,169,75,65,"E","Observed state changes vary across patients")
    changes=pd.read_csv(folder/"longitudinal_module_changes.csv.gz")
    changes["interval"]="C"+changes.cycle_from.astype(str)+"→C"+changes.cycle_to.astype(str)
    changes["delta_balance"]=changes.cytotoxic_delta-changes.memory_delta
    by=changes.groupby(["patient","cycle_from","cycle_to","interval"]).delta_balance.median().reset_index()
    intervals=by.sort_values(["cycle_from","cycle_to"]).interval.unique()
    pc={p:c for p,c in zip(sorted(by.patient.unique()),["#3478ad","#c65359","#4b9a8b","#d9a643","#8b6da5",
                                                     "#94684f","#52738f","#cf7c92","#738c4e","#595f72"])}
    for i,interval in enumerate(intervals):
        z=by[by.interval.eq(interval)].sort_values("patient")
        ax.scatter(i+np.linspace(-.18,.18,len(z)),z.delta_balance,s=14,c=b.BLUE,lw=0)
    ax.axhline(0,color=b.GREY,lw=.8,ls="--");ax.set_xticks(range(len(intervals)),intervals,rotation=30,ha="right",fontsize=5.5)
    ax.set_ylabel("Median within-clone change\nin cytotoxic − memory RNA score",fontsize=5.7)
    b.note(ax,"One point per patient/interval; ≥4 cells per clone/date.\nRNA scores describe expression; they are not efficacy readouts.",-.31,5.4)
    foot=b.p(fig,0,269,183,9,None,"");foot.set_axis_off()
    foot.text(0,1,"D: 200 target permutations within patient, interval and capture-count bin; the same clone pairs are used for all three methods.\n"
                   "Expansion, contraction and migration can all change blood clone composition. Clinical-response claims require an audited outcome mapping.",
              va="top",fontsize=5.6,linespacing=1.5)
    finish(fig,b,"Figure_5",[folder/"summary.json",folder/"clone_umap_kernel.csv.gz",
                            folder/"longitudinal_distance_comparison.csv",folder/"longitudinal_module_changes.csv.gz"])

def draw_benchmark(b):
    folder=b.DATA/"native_benchmark";larry=b.DATA/"clonotrace_larry";require(larry)
    scores=pd.read_csv(folder/"heldout_scores.csv")
    scores=scores[scores.min_cells.eq(2)&scores.status.eq("completed")]
    method_order=["RNA_centroid","RNA_context_centroid","Threadfin_mean","Threadfin_kernel",
                  "clone2vec","BiGCN","Benisse","Training_mean"]
    for ds in ["mouse_np","mouse_rbd"]:
        assert set(scores[scores.dataset.eq(ds)].method)==set(method_order)
    fig=b.new_page("Figure 6","Measure predictive value and sampling robustness separately",
                   "Independent reporter readouts test biological signal; disjoint cell samples test the shrinkage estimator.",height=309)
    for i,(ds,title) in enumerate([("mouse_np","NP–OVA: held-out mice"),("mouse_rbd","RBD: held-out mice")]):
        ax=b.p(fig,30+i*91,9,62,71,"AB"[i],title)
        for j,m in enumerate(method_order):
            v=100*scores[scores.dataset.eq(ds)&scores.method.eq(m)&scores.target.eq("division_gate:mCherry-low")].mae
            col=b.RED if m=="Threadfin_kernel" else b.BLUE if m=="Threadfin_mean" else b.GREY
            ax.scatter(v,j+np.linspace(-.16,.16,len(v)),s=9,c=col,lw=0,alpha=.8)
            ax.plot([v.median()],[j],marker="|",ms=8,mew=1.2,color=b.INK)
        ax.set_yticks(range(len(method_order)),[LABELS[m] for m in method_order],fontsize=5.7)
        ax.invert_yaxis();ax.set_xlabel("Division-fraction MAE (percentage points)",fontsize=5.6)
        b.note(ax,"One point per held-out mouse; tick = median.\nLower is better. Identical families and readout model.",-.17,5.3)
    ax=b.p(fig,30,116,83,68,"C","All measured readouts")
    rs=pd.read_csv(folder/"readout_summary.csv");rs=rs[rs.min_cells.eq(2)]
    rs["readout"]=rs.dataset+"|"+rs.target
    matrix=rs.pivot(index="method",columns="readout",values="median_r2").reindex(method_order)
    im=ax.imshow(matrix,cmap="RdBu_r",vmin=-.3,vmax=.8,aspect="auto")
    ax.set_yticks(range(len(method_order)),[LABELS[m] for m in method_order],fontsize=5.5)
    labels=[c.replace("mouse_np|","NP ").replace("mouse_rbd|","RBD ").replace("division_gate:mCherry-low","division")
             .replace("zone_gate:","").replace("rbd_bait:","").replace("mutation_frequency","SHM") for c in matrix.columns]
    labels=["NP\ndivision","RBD\ndivision","RBD\nbinding","RBD\nDZ"]
    assert len(matrix.columns)==4
    ax.set_xticks(range(len(labels)),labels,fontsize=5.4)
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.055,pad=.20,aspect=28)
    cb.set_label("Median held-out R² (0: test-mean reference)",fontsize=5.6)
    cb.ax.tick_params(labelsize=5.3,length=2)
    ax=b.p(fig,137,116,46,68,"D","Reliability is a\nsampling-dependent quantity")
    table=read(larry,"profiles_all_kernel.csv.gz")
    ax.scatter(table.n_cells,table.reliability,s=4,color=b.BLUE,alpha=.25,lw=0,rasterized=True)
    ax.set_xscale("log");ax.set(xlabel="Cells per clone-day",ylabel="Model reliability",ylim=(-.03,1.04))
    ax.axhline(.5,ls="--",color=b.GREY,lw=.7)
    ax=b.p(fig,8,224,73,39,"E","Disjoint-cell validation of shrinkage")
    errors=pd.read_csv(larry/"split_sample_errors.csv.gz")
    stats=errors.groupby("n_sample")[["raw_mse","threadfin_mse"]].mean()
    ax.plot(stats.index,stats.raw_mse*1000,"o-",color=b.GREY,ms=3,lw=1,label="Unshrunk kernel mean")
    ax.plot(stats.index,stats.threadfin_mse*1000,"o-",color=b.RED,ms=3,lw=1,label="Threadfin shrinkage")
    ax.set(xticks=[2,4,8],xlabel="Cells sampled per profile",ylabel="Mean error × 1,000")
    ax.legend(fontsize=5.2,frameon=False,loc="upper right")
    ax=b.p(fig,113,224,70,39,"F","Benefit is largest at low capture")
    ratio=stats.threadfin_mse/stats.raw_mse
    ax.bar(range(3),100*(1-ratio),width=.55,color=[b.RED,b.BLUE,b.GREEN])
    for i,v in enumerate(100*(1-ratio)):ax.text(i,v+.7,f"{v:.1f}%",ha="center",fontsize=6)
    ax.set(xticks=range(3),xticklabels=["2 cells","4 cells","8 cells"],ylabel="Mean error reduction (%)",ylim=(0,36))
    b.note(ax,"250 evaluation clone-day-well profiles; 20 splits each.\nCalibration uses disjoint biological barcodes; smoothing off.",-.32,5.3)
    finish(fig,b,"Figure_6",[folder/"heldout_scores.csv",folder/"readout_summary.csv",
                            larry/"split_sample_errors.csv.gz",larry/"split_sample_audit.json"])

def draw_reporter_audit(b):
    folder=b.DATA/"figure2_biology"
    require(folder)
    t=read(folder,"family_measurements.csv")
    fig=b.new_page("Supplementary Figure 16","Recent division and accumulated SHM are different measurements",
        "RBD reporter: quantitative checks of capture, within-mouse associations and matched family/gate contrasts.",height=244)
    ax=b.p(fig,13,10,70,64,"A","Reliability filtering changes capture coverage")
    vals=[t.n_cells,t.loc[t.plotted,"n_cells"]]
    ax.boxplot([np.log10(v) for v in vals],tick_labels=["All expanded","Displayed"],showfliers=False,widths=.45)
    ax.set_ylabel("log10 captured cells per family",fontsize=6)
    for i,v in enumerate(vals):
        ax.text(i+1,2.7,f"n={len(v):,}\nmedian={v.median():.0f}",ha="center",va="top",fontsize=6)
    ax.set_ylim(0,3.1)
    ax=b.p(fig,113,10,70,64,"B","Association within individual mice")
    z=pd.read_csv(folder/"within_mouse_associations.csv")
    z=z[z.scope.eq("all_expanded")]
    labels=["dark zone / cycling","light zone","mutation_frequency"]
    for j,(arm,col) in enumerate([("RBD protein",b.GOLD),("mRNA",b.BLUE)]):
        for i,label in enumerate(labels):
            q=z[z.arm.eq(arm)&z.variable.eq(label)].sort_values("donor")
            ax.scatter(i+(j-.5)*.25+np.linspace(-.065,.065,len(q)),q.partial_rank_rho,s=15,c=col,lw=0)
    ax.axhline(0,color=b.GREY,ls="--",lw=.8)
    ax.set_xticks(range(3),["Cycling\nRNA","Light-zone\nRNA","V-region\nSHM"],fontsize=5.7)
    ax.set_ylabel("Partial rank correlation with\ncaptured mCherry-low fraction",fontsize=5.7)
    barlegend(ax,{"RBD protein":b.GOLD,"mRNA":b.BLUE},-.20,2)
    b.note(ax,"Controls: capture count and additional measured gate.\nRNA module associations are descriptive.",-.35,5.4)
    ax=b.p(fig,13,119,70,65,"C","SHM contrast in matched family/gate pairs")
    mice=pd.read_csv(folder/"paired_shm_by_mouse.csv")
    summary=json.loads((folder/"summary.json").read_text())["paired_same_family_other_gate"]["arm_summary"]
    for i,(arm,col) in enumerate([("RBD protein",b.GOLD),("mRNA",b.BLUE)]):
        v=100*mice[mice.arm.eq(arm)]["mean"].to_numpy()
        s=summary[arm];mean=100*s["equal_mouse_mean_delta"]
        lo,hi=100*np.asarray(s["mouse_bootstrap_95_interval"])
        ax.scatter(i+np.linspace(-.10,.10,len(v)),v,c=col,s=16,lw=0,zorder=3)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="_",ms=15,c=b.INK,lw=1,capsize=3)
    ax.axhline(0,color=b.GREY,ls="--",lw=.8)
    ax.set_xticks([0,1],["RBD protein","mRNA"],fontsize=6)
    ax.set_ylabel("SHM low − high mCherry\n(percentage points; mouse means)",fontsize=5.7)
    b.note(ax,"218 families; ≥2 cells/gate within the same extra gate.\nFive mice per arm; descriptive mouse-bootstrap intervals.",-.23,5.4)
    ax=b.p(fig,113,119,70,65,"D","Both division gates are more often captured\nin better-sampled families")
    cov=pd.read_csv(folder/"coverage_by_mouse.csv")
    for i,scope in enumerate(["all_expanded","plotted_reliable"]):
        q=cov[cov.scope.eq(scope)].sort_values("donor")
        ax.scatter(i+np.linspace(-.1,.1,len(q)),100*q.fraction_both_gates,
                   c=q.arm.map({"RBD protein":b.GOLD,"mRNA":b.BLUE}),s=16,lw=0)
        ax.plot([i-.2,i+.2],[100*q.fraction_both_gates.median()]*2,color=b.INK,lw=1)
    ax.set_xticks([0,1],["All expanded","Displayed"],fontsize=6)
    ax.set_ylabel("Families spanning both captured gates (%)",fontsize=5.6)
    ax.set_ylim(-3,103)
    b.note(ax,"Dots: individual mice; line: mouse median.\nExtreme gates are sampled, not the full GC population.",-.23,5.4)
    foot=b.p(fig,0,212,183,7,None,"");foot.set_axis_off()
    foot.text(0,1,"mCherry: recent 36-hour division history. V-region SHM: accumulated divergence, not mutations per division.\n"
                  "Matching controls captured family and additional gate; physical division-gate/library confounding remains.",fontsize=5.6,va="top")
    finish(fig,b,"Supplementary_16",[folder/"summary.json",folder/"within_mouse_associations.csv",
        folder/"paired_shm_by_mouse.csv",folder/"coverage_by_mouse.csv"])
