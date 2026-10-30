#!/usr/bin/env python
"""What distinguishes the arms of the PcAS clone embedding, from the receptor rather than the RNA.

The clone embedding separates germinal-centre, plasmablast and memory arms, and the arms are defined by
transcriptome. Asking what they *mean* with the same transcriptome would be circular, so this script asks
the receptor instead, using two readouts that the embedding never saw.

**Convergent clones.** PcAS was sampled from 20 (experiment 1) and 42 (experiment 2) mice. Different
animals that respond to the same parasite with the same V gene, the same J gene and a near-identical CDR3
have converged on one solution, and convergence is the strongest antigen-specificity evidence available
without a tetramer. This design — many animals, one antigen — is what makes the test possible at all, and
the question is whether convergent clones sit preferentially in one arm.

**BASELINe selection pressure.** `shazam`'s BASELINe compares replacement and silent mutations against a
targeting model, separately for the CDR and the framework. Positive CDR selection with negative framework
selection is the signature of antigen-driven affinity maturation. Unlike a lineage tree it pools across
sequences, so it works on this dataset's small clones. This script writes the AIRR table; the estimation
itself runs in R (`baseline_selection.R`, immcantation environment).

**Lineage-tree topology is not attempted here, and the reason is recorded.** PcAS clones are wide but
shallow: experiment 1 reaches a largest clone of 8 cells with 11 clones at 5 or more, experiment 2 reaches
9 with 30, and neither has a single clone of 15. A GCtree topology from 2 to 9 tips carries no usable
signal, so that analysis runs on the deeply sampled repertoires instead (`lineage_topology.py`).

Usage:  python pcas_arm_analyses.py [--datasets malaria malaria_late] [--n-perm 2000]
Output: case_studies/results/pcas_arm_analyses/
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path(os.environ.get("THREADFIN_DATA", Path(__file__).resolve().parent / "data"))
CACHE = ROOT / "internal_validation" / "algorithm_revision" / "embedding_cache"
CLONES = HERE / "results" / "dandelion_clones"
OUT = HERE / "results" / "pcas_arm_analyses"

AIRR = {"malaria": DATA / "gse286215_malaria/exp1_bcr.tsv.gz",
        "malaria_late": DATA / "gse286215_malaria/exp2_bcr.tsv.gz"}

#: the same compartment vocabulary the figures use
COMPARTMENT = {"GC": "germinal centre", "PB": "plasmablast", "Memory": "memory"}
ARMS = ["germinal centre", "plasmablast", "memory"]

#: a convergent group needs the same V gene, the same J gene, the same CDR3 length and this much
#: amino-acid identity, which is the usual public-clonotype definition
MIN_IDENTITY = 0.80


def gene(call) -> str:
    """The gene name without the allele, from a possibly comma-separated AIRR call."""
    return str(call).split(",")[0].split("*")[0]


def clone_level(dataset: str) -> pd.DataFrame:
    """One row per clone: donor, arm, representative receptor, mutation load."""
    obs = __import__("anndata").read_h5ad(CACHE / f"{dataset}_embedding.h5ad").obs
    obs.index = obs.index.astype(str)
    ids = pd.read_csv(CLONES / f"{dataset}.csv").set_index("cell")["clone_id"]
    obs["clone_id"] = ids.reindex(obs.index).values
    obs = obs[obs["clone_id"].notna()].copy()

    t = pd.read_csv(AIRR[dataset], sep="\t", low_memory=False)
    t = t.loc[:, ~t.columns.duplicated()]
    t["cell"] = t["cell_id"].astype(str) + "-1"
    t = t[t["locus"].astype(str).eq("IGH") if "locus" in t.columns
          else t["v_call"].astype(str).str.startswith("IGH")]
    t = t[t["cell"].isin(obs.index)].drop_duplicates("cell", keep="first").set_index("cell")

    obs["junction_aa"] = t["junction_aa"].reindex(obs.index)
    obs["v_gene"] = t["v_call"].reindex(obs.index).map(gene)
    obs["j_gene"] = t["j_call"].reindex(obs.index).map(gene)
    obs["mu_freq"] = pd.to_numeric(t.get("v_identity"), errors="coerce").reindex(obs.index).rsub(1)

    arm = obs["cell_state"].astype(str).map(COMPARTMENT)
    obs["arm"] = arm
    g = obs.groupby("clone_id", observed=True)

    def modal(s):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else None

    out = pd.DataFrame({
        "donor": g["donor"].agg(modal),
        "n_cells": g.size(),
        "junction_aa": g["junction_aa"].agg(modal),
        "v_gene": g["v_gene"].agg(modal),
        "j_gene": g["j_gene"].agg(modal),
        "mu_freq": g["mu_freq"].mean(),
        "timepoint": g["timepoint"].agg(modal) if "timepoint" in obs else None,
        "n_arm_cells": g["arm"].agg(lambda s: s.notna().sum()),
        "arm": g["arm"].agg(modal),
    })
    return out[out["junction_aa"].notna() & out["junction_aa"].astype(str).str.len().gt(4)]


def convergent_groups(clones: pd.DataFrame) -> pd.Series:
    """Group id per clone, joining clones that share V gene, J gene, CDR3 length and >= 80% identity.

    Single-linkage within each (V, J, length) bucket, which is how public clonotypes are usually called.
    """
    group = pd.Series(pd.NA, index=clones.index, dtype="object")
    key = list(zip(clones["v_gene"], clones["j_gene"], clones["junction_aa"].str.len()))
    clones = clones.assign(_key=key)
    gid = 0
    for _, sub in clones.groupby("_key", observed=True):
        seqs = sub["junction_aa"].to_numpy()
        if len(sub) == 1:
            continue
        arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
        ident = (arr[:, None, :] == arr[None, :, :]).mean(axis=2)
        adj = ident >= MIN_IDENTITY
        # single-linkage components
        seen = np.zeros(len(sub), bool)
        for i in range(len(sub)):
            if seen[i]:
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                j = stack.pop()
                comp.append(j)
                for k in np.flatnonzero(adj[j] & ~seen):
                    seen[k] = True
                    stack.append(k)
            if len(comp) > 1:
                gid += 1
                group.iloc[[clones.index.get_loc(sub.index[c]) for c in comp]] = f"cg{gid:05d}"
    return group


def cross_animal(clones: pd.DataFrame, group: pd.Series) -> pd.Series:
    """True for clones in a convergent group that spans more than one animal."""
    donors = clones.assign(g=group).dropna(subset=["g"]).groupby("g")["donor"].nunique()
    multi = set(donors.index[donors > 1])
    return group.isin(multi)


def arm_enrichment(clones: pd.DataFrame, flag: pd.Series, n_perm: int, seed: int = 0) -> pd.DataFrame:
    """Is the flagged set over-represented in an arm, against a within-donor label permutation?

    The arm labels are shuffled among the clones of the same animal, so the null keeps each animal's
    arm composition and each animal's convergence rate fixed; only the pairing moves.
    """
    use = clones["arm"].notna() & clones["donor"].notna()
    arm, donor, f = clones["arm"][use], clones["donor"][use].astype(str), flag[use].to_numpy()
    obs = pd.crosstab(arm, f).reindex(ARMS).fillna(0)
    rate = (obs[True] / obs.sum(axis=1)) if True in obs.columns else pd.Series(0.0, index=ARMS)
    rng = np.random.default_rng(seed)
    codes = pd.Categorical(arm, categories=ARMS).codes
    order = np.argsort(donor.to_numpy(), kind="stable")
    bounds = np.flatnonzero(np.r_[True, donor.to_numpy()[order][1:] != donor.to_numpy()[order][:-1]])
    blocks = np.split(order, bounds[1:])
    null = np.zeros((n_perm, len(ARMS)))
    for p in range(n_perm):
        perm = codes.copy()
        for b in blocks:
            perm[b] = rng.permutation(codes[b])
        for a in range(len(ARMS)):
            m = perm == a
            null[p, a] = f[m].mean() if m.any() else np.nan
    return pd.DataFrame({
        "arm": ARMS,
        "n_clones": obs.sum(axis=1).to_numpy(),
        "n_convergent": (obs[True].to_numpy() if True in obs.columns else np.zeros(len(ARMS))),
        "rate": rate.to_numpy(),
        "null_mean": np.nanmean(null, axis=0),
        "p_greater": [(np.nanmean(null[:, a] >= rate.iloc[a])) for a in range(len(ARMS))],
        "p_less": [(np.nanmean(null[:, a] <= rate.iloc[a])) for a in range(len(ARMS))],
    })


def baseline_input(dataset: str, clones: pd.DataFrame) -> pd.DataFrame:
    """The AIRR columns shazam needs, one row per cell, carrying its clone's arm."""
    obs = __import__("anndata").read_h5ad(CACHE / f"{dataset}_embedding.h5ad").obs
    obs.index = obs.index.astype(str)
    ids = pd.read_csv(CLONES / f"{dataset}.csv").set_index("cell")["clone_id"]
    obs["clone_id"] = ids.reindex(obs.index).values
    t = pd.read_csv(AIRR[dataset], sep="\t", low_memory=False)
    t = t.loc[:, ~t.columns.duplicated()]
    t["cell"] = t["cell_id"].astype(str) + "-1"
    t = t[t["v_call"].astype(str).str.startswith("IGH")]
    t = t[t["cell"].isin(obs.index)].drop_duplicates("cell", keep="first")
    keep = ["sequence_id", "cell", "v_call", "j_call", "junction_length",
            "sequence_alignment", "germline_alignment"]
    t = t[[c for c in keep if c in t.columns]].copy()
    if "sequence_id" not in t.columns:
        t["sequence_id"] = t["cell"]
    t["clone_id"] = t["cell"].map(obs["clone_id"])
    t["arm"] = t["clone_id"].map(clones["arm"])
    t["donor"] = t["cell"].map(obs["donor"].astype(str))
    return t.dropna(subset=["clone_id", "arm", "sequence_alignment", "germline_alignment"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["malaria", "malaria_late"])
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for ds in a.datasets:
        print(f"\n=== {ds}", flush=True)
        clones = clone_level(ds)
        print(f"  {len(clones):,} clones with a heavy chain, {clones['arm'].notna().sum():,} with an arm",
              flush=True)

        group = convergent_groups(clones)
        clones["convergent_group"] = group
        clones["convergent"] = cross_animal(clones, group)
        n_groups = clones.loc[clones["convergent"], "convergent_group"].nunique()
        print(f"  {n_groups:,} cross-animal convergent groups covering "
              f"{int(clones['convergent'].sum()):,} clones", flush=True)

        enr = arm_enrichment(clones, clones["convergent"], a.n_perm)
        print(enr.round(4).to_string(index=False), flush=True)

        clones.to_csv(OUT / f"{ds}_clones.csv.gz")
        enr.to_csv(OUT / f"{ds}_convergence_by_arm.csv", index=False)
        bi = baseline_input(ds, clones)
        bi.to_csv(OUT / f"{ds}_baseline_input.tsv.gz", sep="\t", index=False)
        print(f"  wrote {len(bi):,} sequences for BASELINe", flush=True)

        sizes = clones["n_cells"]
        report[ds] = {
            "n_clones": int(len(clones)),
            "n_with_arm": int(clones["arm"].notna().sum()),
            "arm_counts": clones["arm"].value_counts().to_dict(),
            "n_convergent_groups": int(n_groups),
            "n_convergent_clones": int(clones["convergent"].sum()),
            "convergence_by_arm": enr.round(5).to_dict(orient="records"),
            "clone_sizes": {"max": int(sizes.max()), "ge5": int((sizes >= 5).sum()),
                            "ge8": int((sizes >= 8).sum()), "ge15": int((sizes >= 15).sum())},
            "lineage_trees": "not attempted: no clone reaches 15 cells, so tree topology has no signal",
            "identity_threshold": MIN_IDENTITY,
            "n_perm": a.n_perm,
        }

    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
