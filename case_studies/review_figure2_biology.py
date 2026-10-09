#!/usr/bin/env python3
"""Audit the biological scope of Figure 2 C-E without selecting UMAP parameters.

Uses committed cell labels/family calls. No causal claim, mutation-rate estimate,
or method ranking follows from these conditional descriptive statistics.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "case_studies/results/mouse_rbd"
OUT = ROOT / "case_studies/results/figure2_biology"
SEED = 20261008

def partial_rank(x, y, controls):
    """Correlation of rank residuals; descriptive, not a permutation p-value."""
    rx=pd.Series(x).rank().to_numpy()
    ry=pd.Series(y).rank().to_numpy()
    c=np.column_stack([np.ones(len(x))]+[pd.Series(v).rank().to_numpy() for v in controls])
    ex=rx-c@np.linalg.lstsq(c,rx,rcond=None)[0]
    ey=ry-c@np.linalg.lstsq(c,ry,rcond=None)[0]
    if np.std(ex)<1e-10 or np.std(ey)<1e-10:return np.nan
    return float(np.corrcoef(ex,ey)[0,1])

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cells=pd.read_csv(DATA/"cells.csv.gz",index_col=0)
    tab=pd.read_csv(DATA/"clone_table.csv",index_col=0)
    scores=pd.read_csv(DATA/"clone_gene_scores.csv",index_col=0)
    cells=cells[cells.clone_id.notna()].copy()
    cells["other_gate"]=cells.zone_gate.fillna(cells.rbd_bait)
    cells["low"]=cells.division_gate.eq("mCherry-low").astype(float)
    cells["arm"]=np.where(cells.zone_gate.notna(),"mRNA","RBD protein")
    sizes=cells.groupby("clone_id").size()
    empirical=cells.groupby("clone_id").agg(
        n_cells=("low","size"),division_fraction=("low","mean"),
        mutation_frequency=("mutation_frequency","mean"),donor=("donor","first"),
        arm=("arm","first"))
    empirical=empirical.loc[empirical.n_cells.ge(2)]
    assert len(empirical)==1414
    np.testing.assert_allclose(empirical.division_fraction,
                              tab.reindex(empirical.index)["division_gate:mCherry-low"])
    empirical["reliability"]=tab.reindex(empirical.index).reliability
    empirical["plotted"]=empirical.reliability.ge(.5)
    empirical["gate_breadth"]=np.select(
        [empirical.division_fraction.eq(0),empirical.division_fraction.eq(1)],
        ["high only","low only"],default="both gates")
    assert empirical.plotted.sum()==381
    empirical=empirical.join(scores)
    empirical.to_csv(OUT/"family_measurements.csv")
    coverage=[]
    for scope,m in [("all_expanded",np.ones(len(empirical),bool)),("plotted_reliable",empirical.plotted)]:
        for (arm,donor),g in empirical[m].groupby(["arm","donor"]):
            coverage.append(dict(scope=scope,arm=arm,donor=donor,n_families=len(g),
                n_cells=int(g.n_cells.sum()),median_n_cells=float(g.n_cells.median()),
                fraction_both_gates=float(g.gate_breadth.eq("both gates").mean()),
                median_mutation_frequency=float(g.mutation_frequency.median())))
    pd.DataFrame(coverage).to_csv(OUT/"coverage_by_mouse.csv",index=False)
    associations=[]
    ykey="division_gate:mCherry-low"
    joined=tab.join(scores)
    for scope,subset in [("all_expanded",joined),("plotted_reliable",joined[joined.reliability.ge(.5)])]:
        for donor,g in subset.groupby("donor"):
            arm="RBD protein" if int(donor[1:])<=5 else "mRNA"
            other="rbd_bait:RBD+" if arm=="RBD protein" else "zone_gate:DZ"
            for label in ["mutation_frequency","dark zone / cycling","light zone"]:
                q=g[[ykey,label,"n_cells",other]].dropna()
                if len(q)<8 or q[ykey].nunique()<2 or q[label].nunique()<2:continue
                raw=float(spearmanr(q[ykey],q[label]).statistic)
                adjusted=partial_rank(q[ykey],q[label],[np.log1p(q.n_cells),q[other]])
                associations.append(dict(scope=scope,arm=arm,donor=donor,variable=label,
                    n_families=len(q),spearman_rho=raw,partial_rank_rho=adjusted))
    assoc=pd.DataFrame(associations)
    assoc.to_csv(OUT/"within_mouse_associations.csv",index=False)
    # Keep the same biological family and the same additional FACS gate.
    # Paired groups come from different physical libraries: confounding remains.
    grouped=cells.dropna(subset=["mutation_frequency"]).groupby(
        ["donor","arm","clone_id","other_gate","division_gate"]).mutation_frequency.agg(["mean","size"])
    paired=grouped.unstack("division_gate")
    good=(paired["size"]["mCherry-high"]>=2)&(paired["size"]["mCherry-low"]>=2)
    paired=paired.loc[good]
    out=paired.index.to_frame(index=False)
    out["shm_high"]=paired["mean"]["mCherry-high"].to_numpy()
    out["shm_low"]=paired["mean"]["mCherry-low"].to_numpy()
    out["n_high"]=paired["size"]["mCherry-high"].to_numpy()
    out["n_low"]=paired["size"]["mCherry-low"].to_numpy()
    out["delta_low_minus_high"]=out.shm_low-out.shm_high
    out.to_csv(OUT/"paired_family_gate_shm.csv",index=False)
    # Equal family then equal mouse weighting; paired gate strata are not
    # independent mice. Avoid cell-level pseudo-replication.
    fam=out.groupby(["arm","donor","clone_id"]).delta_low_minus_high.mean().reset_index()
    mice=fam.groupby(["arm","donor"]).delta_low_minus_high.agg(["mean","median","size"]).reset_index()
    mice.to_csv(OUT/"paired_shm_by_mouse.csv",index=False)
    rng=np.random.default_rng(SEED);arm_summary={}
    for arm,g in mice.groupby("arm"):
        values=g["mean"].to_numpy()
        boot=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1)
        p=float(wilcoxon(values,method="exact").pvalue) if len(values)>1 and np.any(values) else None
        arm_summary[arm]={"n_mice":len(values),"n_families":int(g["size"].sum()),
                         "equal_mouse_mean_delta":float(values.mean()),
                         "mouse_bootstrap_95_interval":np.quantile(boot,[.025,.975]).tolist(),
                         "two_sided_wilcoxon_mouse_p":p}
    summary={"status":"complete","sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [DATA/"cells.csv.gz",DATA/"clone_table.csv",DATA/"clone_gene_scores.csv"]},
        "n_receptor_called_cells":len(cells),"n_expanded_families":len(empirical),
        "n_plotted_families":int(empirical.plotted.sum()),
        "capture_summary":{scope:{"n_families":len(g),"median_cells":float(g.n_cells.median()),
                        "n_both_gates":int(g.gate_breadth.eq("both gates").sum()),
                        "n_mRNA":int(g.arm.eq("mRNA").sum())}
                for scope,g in [("all_expanded",empirical),("plotted_reliable",empirical[empirical.plotted])]},
        "paired_same_family_other_gate":{"n_strata":len(out),"n_families":len(fam),"arm_summary":arm_summary},
        "limitations":["mCherry records recent divisions; mean V mutation frequency is accumulated divergence, not mutations per division.",
                       "Only extreme mCherry gates are sampled; fractions do not estimate in-vivo GC occupancy without sampling weights.",
                       "Additional FACS gate and donor are matched, but division gate and physical library are confounded.",
                       "Gene modules reuse the RNA used to construct profiles.",
                       "The ten mice belong to two experimental arms; bootstrap intervals with five mice/arm are descriptive.",
                       "Reliability-based display filtering is not independent biological validation; all expanded families remain in the audit."]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
