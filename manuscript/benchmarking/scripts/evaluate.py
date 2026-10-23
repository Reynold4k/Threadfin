#!/usr/bin/env python
"""Score every clone representation with the same readouts on the same clones.

Every method leaves either coordinates (``features.npy``) or a clone-by-clone distance matrix
(``distances.npy``; Clonotrace's long table) for the clones it represented. All readouts below use only
neighbour relations between clones, so coordinates and distance matrices are scored identically.

Readouts
--------
``annotation``      For a clone-level annotation that was not used to build any representation (sorting
                    gate, tissue, isotype, V-region mutation frequency, measured later fate ...), each
                    clone's value is predicted as the mean over its k = 10 nearest clones (leave-one-out
                    k-nearest-neighbour regression). Values are centred within donor first, so a method
                    gains nothing from separating donors. Reported: R^2, Spearman's rho, the permutation
                    null of R^2 (values permuted among the clones of each donor) and its excess.
``pairwise``        The same readout for each method against Threadfin's default on the clones that method
                    represents (so a method that drops clones does not shrink the other comparisons), with a
                    paired bootstrap interval of the difference and the R^2 on two-cell clones.
``annotation_disjoint``  The same, but the representation is computed from one half of every clone's
                    cells and the annotation from the other half, so no cell contributes to both.
``reproducibility`` Two runs on disjoint halves of every clone: Spearman correlation of the clone-clone
                    distances, overlap of the 10 nearest clones, and re-identification of each clone
                    across the halves from its distances to all other clones.
``linking``         LARRY: the rank of a barcode's day-6 clone among all day-6 clones, from its day-4 clone.
``simulation``      purity of the 10 nearest clones by true programme; separation of mixture from
                    intermediate clones (AUROC); excess of same-donor neighbours.

Comparisons are made on the clones that *every* compared method represents ("common" sets); coverage —
how many of the dataset's clones a method can represent at all — is reported separately.

Usage: python evaluate.py [annotation|pairwise|disjoint|reproducibility|larry|simulation|coverage|all]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from bench_paths import INPUTS, RESULTS, RUNS

K = 10
N_PERM = 199
SEEDS = (0, 1, 2)
RNG = np.random.default_rng(20261020)

# method id -> (run directory, how to read it)
METHODS = {
    "threadfin_kernel": ("threadfin_kernel", "features"),
    "threadfin_mean": ("threadfin_mean", "features"),
    "threadfin_kernel_nocentre": ("threadfin_kernel_nocentre", "features"),
    "threadfin_kernel_nosmooth": ("threadfin_kernel_nosmooth", "features"),
    "threadfin_kernel_noshrink": ("threadfin_kernel_noshrink", "features"),
    "mean_pca": ("mean_pca", "features"),
    "composition": ("composition", "features"),
    "conga_medoid": ("conga_medoid", "features"),
    "clone2vec_min3": ("clone2vec_min3", "features"),
    "clone2vec_min2": ("clone2vec_min2", "features"),
    "clonotrace_revised_min10": ("clonotrace_min10", "dis"),
    "clonotrace_raw_min10": ("clonotrace_min10", "dis_clonotrace"),
    "clonotrace_revised_min2": ("clonotrace_min2", "dis"),
    "clonotrace_raw_min2": ("clonotrace_min2", "dis_clonotrace"),
    "benisse_min2": ("benisse_min2", "distances"),
    "benisse_min10": ("benisse_min10", "distances"),
}
# Threadfin's default profile with one setting changed at a time (run_threadfin.py SENSITIVITY)
SENSITIVITY = ["threadfin_kernel_smooth5", "threadfin_kernel_smooth30", "threadfin_kernel_rff128",
               "threadfin_kernel_rff512", "threadfin_kernel_rff1024", "threadfin_kernel_bw0.5", "threadfin_kernel_bw2",
               "threadfin_kernel_pc10", "threadfin_kernel_pc50", "threadfin_kernel_seed1", "threadfin_kernel_seed2"]
METHODS.update({m: (m, "features") for m in SENSITIVITY})
SENS_SET = ["threadfin_kernel", "threadfin_kernel_nosmooth"] + SENSITIVITY
# each tool at its own default minimum clone size ("native"), and every tool forced to two cells
NATIVE = ["threadfin_kernel", "threadfin_mean", "clonotrace_revised_min10", "clonotrace_raw_min10",
          "clone2vec_min3", "benisse_min2", "benisse_min10", "mean_pca", "composition", "conga_medoid"]
ALL2 = ["threadfin_kernel", "threadfin_mean", "clonotrace_revised_min2", "clonotrace_raw_min2", "clone2vec_min2",
        "benisse_min2", "mean_pca", "composition", "conga_medoid"]
ABLATION = ["threadfin_kernel", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
            "threadfin_kernel_noshrink", "threadfin_mean", "mean_pca"]


@dataclass
class Target:
    name: str
    column: str
    kind: str            # "fraction" | "mean" | "clone"
    level: object = None  # value(s) counted for "fraction"
    tag: str = ""        # sorted | sequence | sampling | future | rna


T = Target
TARGETS = {
    "gut_gc_reporter": [T("GC-fate sort (tdTomato+)", "gc_sort", "fraction", "GC", "sorted"),
                        T("lamina propria", "tissue", "fraction", "LP", "sampling"),
                        T("IgA", "isotype", "fraction", "IGHA", "sequence"),
                        T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                        T("V-region mutation", "mutation_frequency", "mean", None, "sequence"),
                        T("plasma cell (RNA)", "celltype", "fraction", "plasma", "rna")],
    "mouse_rbd": [T("recently divided (mCherry-low)", "division_gate", "fraction", "mCherry-low", "sorted"),
                  T("RBD-bait+", "rbd_bait", "fraction", "RBD+", "sorted"),
                  T("dark-zone gate", "zone_gate", "fraction", "DZ", "sorted"),
                  T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                  T("V-region mutation", "mutation_frequency", "mean", None, "sequence")],
    "mouse_np": [T("recently divided (mCherry-low)", "division_gate", "fraction", "mCherry-low", "sorted"),
                 T("W33L", "w33l", "fraction", "W33L", "sequence"),
                 T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                 T("V-region mutation", "mutation_frequency", "mean", None, "sequence")],
    "malaria_late": [T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                     T("V-region mutation", "mutation_frequency", "mean", None, "sequence"),
                     T("GC B cell (RNA)", "cell_state", "fraction", "GC", "rna"),
                     T("plasmablast (RNA)", "cell_state", "fraction", "PB", "rna")],
    "malaria": [T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                T("V-region mutation", "mutation_frequency", "mean", None, "sequence"),
                T("GC B cell (RNA)", "cell_state", "fraction", "GC", "rna"),
                T("plasmablast (RNA)", "cell_state", "fraction", "PB", "rna")],
    "ln_vaccine": [T("blood (vs lymph node)", "tissue", "fraction", "blood", "sampling"),
                   T("spike binding", "spike_binding", "fraction", "S+", "sorted"),
                   T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                   T("IgA", "isotype", "fraction", "IGHA", "sequence"),
                   T("V-region mutation", "mutation_frequency", "mean", None, "sequence"),
                   T("days after vaccination", "timepoint", "mean", None, "sampling"),
                   T("GC B cell (RNA)", "state", "fraction", "GC", "rna")],
    "spont_gc": [T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                 T("IgA", "isotype", "fraction", "IGHA", "sequence"),
                 T("V-region mutation", "mutation_frequency", "mean", None, "sequence")],
    "gc_np_pc": [T("plasma-cell sort", "fate", "fraction", "plasma cell", "sorted"),
                 T("W33L", "w33l", "fraction", "W33L", "sequence"),
                 T("V-region mutation", "mutation_frequency", "mean", None, "sequence")],
    "covid_bcell": [T("IgG", "isotype", "fraction", "IGHG", "sequence"),
                    T("IgA", "isotype", "fraction", "IGHA", "sequence")],
    "larry_d2": [T("day-6 neutrophil fraction", "day6_neutrophil_fraction", "clone", None, "future"),
                 T("day-6 monocyte fraction", "day6_monocyte_fraction", "clone", None, "future"),
                 T("day-6 neutrophil share (of neu + mono)", "day6_neutrophil_share_of_neu_mono", "clone", None, "future")],
}
DAYS = {"d28": 28, "d35": 35, "d28+d35": 31.5, "d60": 60, "d110": 110, "d201": 201}


# --------------------------------------------------------------------------- loading


class Rep:
    """A method's representation of the clones of one run."""

    def __init__(self, clones, features=None, distances=None):
        self.clones = pd.Index(clones.astype(str))
        self.features = features
        self.distances = distances
        self.pos = pd.Series(np.arange(len(self.clones)), index=self.clones)

    def sub_distance(self, subset) -> np.ndarray:
        i = self.pos.loc[subset].to_numpy()
        if self.distances is not None:
            return self.distances[np.ix_(i, i)].astype(float)
        f = self.features[i].astype(float)
        sq = (f ** 2).sum(1)
        return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * f @ f.T, 0))

    def neighbours(self, subset, k=K) -> np.ndarray:
        """Indices (into ``subset``) of each clone's k nearest other clones of ``subset``."""
        k = min(k, len(subset) - 1)
        i = self.pos.loc[subset].to_numpy()
        if self.distances is None:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(self.features[i])
            _, idx = nn.kneighbors(self.features[i])
            out = np.empty((len(i), k), dtype=int)
            for r in range(len(i)):              # drop the clone itself (ties may put it later)
                row = idx[r][idx[r] != r]
                out[r] = row[:k]
            return out
        d = self.distances[np.ix_(i, i)].astype(float)
        np.fill_diagonal(d, np.inf)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, idx, 1), axis=1)
        return np.take_along_axis(idx, order, 1)


_CACHE: dict = {}


def load_rep(dataset, variant, method):
    key = (dataset, variant, method)
    if key in _CACHE:
        return _CACHE[key]
    sub, how = METHODS[method]
    d = RUNS / dataset / variant / sub
    rep = None
    try:
        info = json.loads((d / "run.json").read_text())
    except FileNotFoundError:
        info = {"status": "missing"}
    if info.get("status") == "completed":
        if how == "features":
            clones = pd.read_csv(d / "clones.tsv", sep="\t")["clone"].astype(str).to_numpy()
            rep = Rep(clones, features=np.load(d / "features.npy"))
        elif how == "distances":
            clones = pd.read_csv(d / "clones.tsv", sep="\t")["clone"].astype(str).to_numpy()
            rep = Rep(clones, distances=np.load(d / "distances.npy"))
        else:                                           # Clonotrace long table
            long = pd.read_csv(d / "clonotrace_distance_long.tsv.gz", sep="\t")
            clones = pd.read_csv(d / "clones_clonotrace.tsv", sep="\t")["clone"].astype(str).to_numpy()
            idx = pd.Series(np.arange(len(clones)), index=clones)
            m = np.zeros((len(clones), len(clones)))
            a = idx.loc[long["group1"].astype(str)].to_numpy()
            b = idx.loc[long["group2"].astype(str)].to_numpy()
            m[a, b] = m[b, a] = long[how].to_numpy(float)
            rep = Rep(clones, distances=m)
    _CACHE[key] = rep
    return rep


def run_info(dataset, variant, method):
    sub, _ = METHODS[method]
    p = RUNS / dataset / variant / sub / "run.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "missing"}


def load_cells(dataset, variant="full"):
    d = INPUTS / dataset
    cells = pd.read_csv(d / "cells.tsv.gz", sep="\t", index_col=0, low_memory=False)
    cells.index = cells.index.astype(str)
    if variant.startswith("half"):
        half, seed = variant[4], int(variant.split("_s")[1])
        split = pd.read_csv(d / f"split_seed{seed}.tsv.gz", sep="\t", index_col=0)["half"]
        split.index = split.index.astype(str)
        cells = cells.loc[(split.reindex(cells.index) == half).to_numpy()]
    return cells


def clone_values(dataset, cells, target: Target) -> pd.Series:
    if target.kind == "clone":
        t = pd.read_csv(INPUTS / dataset / "clone_targets.tsv", sep="\t", index_col=0)
        t.index = t.index.astype(str)
        return t[target.column].dropna()
    c = cells.dropna(subset=["clone"])
    v = c[target.column]
    if target.column == "timepoint":
        v = v.map(DAYS)
    if target.kind == "mean":
        v = pd.to_numeric(v, errors="coerce")
        return v.groupby(c["clone"].astype(str)).mean().dropna()
    ok = v.notna() & v.astype(str).ne("nan")
    hit = (v[ok].astype(str) == str(target.level)).astype(float)
    return hit.groupby(c.loc[ok, "clone"].astype(str)).mean()


def clone_donor(cells) -> pd.Series:
    c = cells.dropna(subset=["clone"])
    return c.groupby(c["clone"].astype(str))["donor"].first().astype(str)


def clone_sizes(cells) -> pd.Series:
    return cells["clone"].dropna().astype(str).value_counts()


# --------------------------------------------------------------------------- statistics


def centre(y: pd.Series, donor: pd.Series) -> np.ndarray:
    v = y.to_numpy(float)
    g = donor.loc[y.index].to_numpy()
    out = v.copy()
    for d in np.unique(g):
        m = g == d
        out[m] = v[m] - v[m].mean()
    return out


def permute_within(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    out = y.copy()
    for d in np.unique(groups):
        m = np.flatnonzero(groups == d)
        out[m] = y[rng.permutation(m)]
    return out


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else np.nan


def knn_regression(nbr, y, groups, n_perm=N_PERM, rng=RNG, eval_mask=None):
    """Leave-one-out kNN prediction of y from its neighbours; permutation null within groups."""
    yhat = y[nbr].mean(axis=1)
    m = np.ones(len(y), bool) if eval_mask is None else eval_mask
    obs = r2(y[m], yhat[m])
    rho = float(stats.spearmanr(yhat[m], y[m]).statistic) if m.sum() > 2 else np.nan
    null = np.array([r2(yp[m], yp[nbr].mean(axis=1)[m]) for yp in
                     (permute_within(y, groups, rng) for _ in range(n_perm))])
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return {"r2": obs, "spearman": rho, "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
            "excess_r2": float(obs - np.nanmean(null)), "p": p, "n_eval": int(m.sum()), "yhat": yhat}


def bootstrap_r2_ci(y, yhat, n_boot=1000, rng=RNG):
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = np.array([r2(y[i], yhat[i]) for i in idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def common_clones(reps: dict, base: pd.Index) -> pd.Index:
    out = base
    for rep in reps.values():
        out = out.intersection(rep.clones)
    return out


# --------------------------------------------------------------------------- readouts


def available(dataset, variant, methods):
    reps = {}
    for m in methods:
        r = load_rep(dataset, variant, m)
        if r is not None and len(r.clones) >= 3:
            reps[m] = r
    return reps


def annotation_rows(dataset, sets):
    """Same-cell annotation recovery on the full data."""
    rows = []
    cells = load_cells(dataset)
    sizes = clone_sizes(cells)
    donor = clone_donor(cells)
    for set_name, methods, min_size in sets:
        reps = available(dataset, "full", methods)
        if not reps:
            continue
        base = sizes.index[sizes >= min_size]
        common = common_clones(reps, base)
        for tgt in TARGETS.get(dataset, []):
            y_all = clone_values(dataset, cells, tgt)
            idx = common.intersection(y_all.index)
            if len(idx) < 20:
                continue
            y = centre(y_all.loc[idx], donor)
            groups = donor.loc[idx].to_numpy()
            two = (sizes.loc[idx] == 2).to_numpy()
            preds = {}
            for m, rep in reps.items():
                res = knn_regression(rep.neighbours(idx), y, groups)
                preds[m] = res.pop("yhat")
                lo, hi = bootstrap_r2_ci(y, preds[m])
                row = {"dataset": dataset, "set": set_name, "target": tgt.name, "tag": tgt.tag, "method": m,
                       "n_clones": len(idx), "r2_ci_low": lo, "r2_ci_high": hi, **res}
                if two.sum() >= 20:                     # how well the 2-cell clones are predicted
                    row["r2_two_cell_clones"] = r2(y[two], preds[m][two])
                    row["n_two_cell_clones"] = int(two.sum())
                rows.append(row)
            ref = preds.get("threadfin_kernel")
            if ref is not None:                       # paired bootstrap of the R^2 difference
                b = RNG.integers(0, len(y), size=(1000, len(y)))
                for m, pm in preds.items():
                    diff = np.array([r2(y[i], ref[i]) - r2(y[i], pm[i]) for i in b])
                    for row in rows[-len(preds):]:
                        if row["method"] == m and row["target"] == tgt.name and row["set"] == set_name:
                            row["delta_vs_threadfin_kernel"] = float(r2(y, ref) - r2(y, pm))
                            row["delta_ci_low"] = float(np.nanpercentile(diff, 2.5))
                            row["delta_ci_high"] = float(np.nanpercentile(diff, 97.5))
    return rows


def pairwise_rows(dataset, comparators, reference="threadfin_kernel", n_boot=1000):
    """Each method against the reference on the clones that method represents (pairwise common sets).

    The all-method common sets of ``annotation_rows`` shrink every comparison to the clones that the most
    restrictive method keeps (Clonotrace drops clones after label spreading); here each comparison keeps every
    clone that both methods represent.
    """
    rows = []
    cells = load_cells(dataset)
    sizes = clone_sizes(cells)
    donor = clone_donor(cells)
    ref = load_rep(dataset, "full", reference)
    if ref is None:
        return rows
    for m in comparators:
        rep = load_rep(dataset, "full", m)
        if rep is None or len(rep.clones) < 3:
            continue
        common = ref.clones.intersection(rep.clones).intersection(sizes.index[sizes >= 2])
        for tgt in TARGETS.get(dataset, []):
            y_all = clone_values(dataset, cells, tgt)
            idx = common.intersection(y_all.index)
            if len(idx) < 20:
                continue
            y = centre(y_all.loc[idx], donor)
            groups = donor.loc[idx].to_numpy()
            two = (sizes.loc[idx] == 2).to_numpy()
            res_ref = knn_regression(ref.neighbours(idx), y, groups)
            res_m = knn_regression(rep.neighbours(idx), y, groups)
            p_ref, p_m = res_ref.pop("yhat"), res_m.pop("yhat")
            b = RNG.integers(0, len(y), size=(n_boot, len(y)))
            diff = np.array([r2(y[i], p_m[i]) - r2(y[i], p_ref[i]) for i in b])
            row = {"dataset": dataset, "target": tgt.name, "tag": tgt.tag, "method": m, "reference": reference,
                   "n_clones": len(idx), "n_two_cell_clones": int(two.sum()),
                   "r2_reference": res_ref["r2"], "r2_method": res_m["r2"],
                   "null_mean_reference": res_ref["null_mean"], "null_mean_method": res_m["null_mean"],
                   "p_reference": res_ref["p"], "p_method": res_m["p"],
                   "delta_method_minus_reference": res_m["r2"] - res_ref["r2"],
                   "delta_ci_low": float(np.nanpercentile(diff, 2.5)),
                   "delta_ci_high": float(np.nanpercentile(diff, 97.5))}
            if two.sum() >= 20:
                row["r2_two_cell_reference"] = r2(y[two], p_ref[two])
                row["r2_two_cell_method"] = r2(y[two], p_m[two])
            rows.append(row)
    return rows


def disjoint_rows(dataset, sets, seeds=SEEDS):
    """Representation from one half of every clone, annotation from the other half."""
    rows = []
    for seed in seeds:
        for h, other in (("A", "B"), ("B", "A")):
            v, w = f"half{h}_s{seed}", f"half{other}_s{seed}"
            cells_v, cells_w = load_cells(dataset, v), load_cells(dataset, w)
            sv, sw = clone_sizes(cells_v), clone_sizes(cells_w)
            donor = clone_donor(load_cells(dataset))
            for set_name, methods, min_per_half in sets:
                reps = available(dataset, v, methods)
                if not reps:
                    continue
                base = sv.index[sv >= min_per_half].intersection(sw.index[sw >= min_per_half])
                common = common_clones(reps, base)
                for tgt in TARGETS.get(dataset, []):
                    if tgt.kind == "clone":
                        continue
                    y_all = clone_values(dataset, cells_w, tgt)
                    idx = common.intersection(y_all.index)
                    if len(idx) < 20:
                        continue
                    y = centre(y_all.loc[idx], donor)
                    groups = donor.loc[idx].to_numpy()
                    for m, rep in reps.items():
                        res = knn_regression(rep.neighbours(idx), y, groups, n_perm=99)
                        res.pop("yhat")
                        rows.append({"dataset": dataset, "set": set_name, "target": tgt.name, "tag": tgt.tag,
                                     "method": m, "seed": seed, "representation_half": h,
                                     "n_clones": len(idx), **res})
    return rows


def reidentify(da: np.ndarray, db: np.ndarray):
    """Rank of each clone's own row of half B among all rows of half B, matched on distance profiles."""
    n = len(da)
    a, b = da.copy(), db.copy()
    np.fill_diagonal(a, np.nan)
    np.fill_diagonal(b, np.nan)
    med_a, med_b = np.nanmedian(a, axis=1), np.nanmedian(b, axis=1)
    a[np.arange(n), np.arange(n)] = med_a
    b[np.arange(n), np.arange(n)] = med_b
    ra = stats.rankdata(a, axis=1)
    rb = stats.rankdata(b, axis=1)
    ra = (ra - ra.mean(1, keepdims=True)) / ra.std(1, keepdims=True)
    rb = (rb - rb.mean(1, keepdims=True)) / rb.std(1, keepdims=True)
    sim = ra @ rb.T / n                                  # Spearman correlation of distance profiles
    rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
    return {"reid_top1": float(np.mean(rank == 1)), "reid_top10": float(np.mean(rank <= 10)),
            "reid_mrr": float(np.mean(1 / rank)), "reid_median_rank": float(np.median(rank)),
            "reid_chance_top1": 1 / n}


def reproducibility_rows(dataset, sets, seeds=SEEDS):
    rows = []
    for seed in seeds:
        va, vb = f"halfA_s{seed}", f"halfB_s{seed}"
        sa, sb = clone_sizes(load_cells(dataset, va)), clone_sizes(load_cells(dataset, vb))
        for set_name, methods, min_per_half in sets:
            ra_ = available(dataset, va, methods)
            rb_ = available(dataset, vb, methods)
            both = [m for m in methods if m in ra_ and m in rb_]
            if not both:
                continue
            base = sa.index[sa >= min_per_half].intersection(sb.index[sb >= min_per_half])
            common = common_clones({m: ra_[m] for m in both}, base)
            common = common_clones({m: rb_[m] for m in both}, common)
            if len(common) < 20:
                continue
            if len(common) > 4000:
                common = pd.Index(RNG.choice(common, 4000, replace=False))
            iu = np.triu_indices(len(common), 1)
            for m in both:
                da, db = ra_[m].sub_distance(common), rb_[m].sub_distance(common)
                pairs = np.arange(len(iu[0]))
                if len(pairs) > 300000:
                    pairs = RNG.choice(pairs, 300000, replace=False)
                rho = float(stats.spearmanr(da[iu][pairs], db[iu][pairs]).statistic)
                k = min(K, len(common) - 1)
                na = np.argsort(da + np.diag(np.full(len(common), np.inf)), axis=1)[:, :k]
                nb = np.argsort(db + np.diag(np.full(len(common), np.inf)), axis=1)[:, :k]
                overlap = float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(common))]))
                rows.append({"dataset": dataset, "set": set_name, "method": m, "seed": seed, "n_clones": len(common),
                             "distance_spearman": rho, "knn_overlap": overlap,
                             "knn_overlap_chance": k / (len(common) - 1), **reidentify(da, db)})
    return rows


def larry_linking_rows(sets):
    rows = []
    cells = load_cells("larry_d46")
    sizes = clone_sizes(cells)
    for set_name, methods, min_size in sets:
        reps = available("larry_d46", "full", methods)
        if not reps:
            continue
        common = common_clones(reps, sizes.index[sizes >= min_size])
        bc = pd.Series(common.str.rsplit("|", n=1).str[0], index=common)
        day = pd.Series(common.str.rsplit("|", n=1).str[1], index=common)
        d4, d6 = common[day.eq("d4").to_numpy()], common[day.eq("d6").to_numpy()]
        partners = set(bc.loc[d4]) & set(bc.loc[d6])
        q = pd.Index([c for c in d4 if bc[c] in partners])
        if len(q) < 10:
            continue
        ranks = {}
        for m, rep in reps.items():
            D = rep.sub_distance(q.append(d6))
            dq = D[: len(q), len(q):]
            target = np.array([d6.get_loc(f"{bc[c]}|d6") for c in q])
            own = dq[np.arange(len(q)), target]
            ranks[m] = rank = (dq < own[:, None]).sum(1) + 1
            rows.append({"dataset": "larry_d46", "set": set_name, "method": m, "n_query_clones": len(q),
                         "n_candidate_day6_clones": len(d6), "top1": float(np.mean(rank == 1)),
                         "top10": float(np.mean(rank <= 10)), "mrr": float(np.mean(1 / rank)),
                         "median_rank": float(np.median(rank)), "chance_top1": 1 / len(d6),
                         "median_rank_chance": (len(d6) + 1) / 2})
        # paired bootstrap over query clones: reciprocal rank of the method minus that of Threadfin
        if "threadfin_kernel" in ranks:
            boot = np.random.default_rng(0).integers(0, len(q), size=(2000, len(q)))
            for row in rows[-len(ranks):]:
                d = 1 / ranks[row["method"]] - 1 / ranks["threadfin_kernel"]
                lo, hi = np.percentile(d[boot].mean(axis=1), [2.5, 97.5])
                row.update({"mrr_delta_vs_threadfin": float(d.mean()), "mrr_delta_ci_low": float(lo),
                            "mrr_delta_ci_high": float(hi)})
    return rows


def simulation_rows(datasets, sets):
    rows = []
    for ds in datasets:
        meta = json.loads((INPUTS / ds / "meta.json").read_text())
        cells = load_cells(ds)
        sizes = clone_sizes(cells)
        c = cells.dropna(subset=["clone"])
        prog = c.groupby(c["clone"].astype(str))["programme"].first()
        donor = clone_donor(cells)
        for set_name, methods, min_size in sets:
            reps = available(ds, "full", methods)
            if not reps:
                continue
            common = common_clones(reps, sizes.index[sizes >= min_size])
            if len(common) < 30:
                continue
            p = prog.loc[common].to_numpy()
            dn = donor.loc[common].to_numpy()
            share = pd.Series(p).value_counts(normalize=True).to_numpy()
            dshare = pd.Series(dn).value_counts(normalize=True).to_numpy()
            mi = np.isin(p, ["mixture_GC_PC", "intermediate"])
            two = (sizes.loc[common] == 2).to_numpy()
            for m, rep in reps.items():
                nbr = rep.neighbours(common)
                same = (p[nbr] == p[:, None]).mean(1)
                vote = (p[nbr] == "mixture_GC_PC").mean(1)
                auc = roc_auc_score(p[mi] == "mixture_GC_PC", vote[mi]) if mi.sum() >= 10 and len(set(p[mi])) == 2 else np.nan
                rows.append({"dataset": ds, "regime": meta["regime"], "batch": meta["batch"], "seed": meta["seed"],
                             "set": set_name, "method": m, "n_clones": len(common),
                             "programme_purity": float(same.mean()), "purity_chance": float((share ** 2).sum()),
                             "purity_two_cell_clones": float(same[two].mean()) if two.sum() >= 10 else np.nan,
                             "mixture_vs_intermediate_auroc": float(auc),
                             "same_donor_excess": float((dn[nbr] == dn[:, None]).mean() - (dshare ** 2).sum())})
    return rows


def not_run_reason(dataset, method, sizes):
    if method.startswith("benisse") and (dataset.startswith(("larry", "sim_"))):
        return "not applicable: no receptor sequences"
    if method.endswith("min10") and int((sizes >= 10).sum()) < 3:
        return "not applicable: fewer than 3 clones with >= 10 cells"
    return "not run"


def time_file(path):
    wall = rss = None
    if path.exists():
        for line in path.read_text().splitlines():
            if "Elapsed (wall clock)" in line:
                hms = line.rsplit(" ", 1)[1].split(":")
                wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(hms)))
            if "Maximum resident set size" in line:
                rss = int(line.rsplit(" ", 1)[1]) / 1024 ** 2
    return wall, rss


def coverage_rows(datasets):
    rows = []
    for ds in datasets:
        cells = load_cells(ds)
        sizes = clone_sizes(cells)
        for m in METHODS:
            info = run_info(ds, "full", m)
            rep = load_rep(ds, "full", m)
            n = 0 if rep is None else len(rep.clones)
            # Threadfin and the baselines run in one process per dataset; their own stage times are in run.json
            wall, rss = time_file(RUNS / ds / "full" / METHODS[m][0] / "time.txt")
            if info.get("status") == "missing":
                # a run that had to be stopped is kept next to its rerun as <sub>_..._stopped...
                stopped = sorted((RUNS / ds / "full").glob(f"{METHODS[m][0]}_*stopped*/run.json"))
                if stopped:
                    info = json.loads(stopped[-1].read_text())
                    wall, rss = time_file(stopped[-1].parent / "time.txt")
                    if (RUNS / ds / "full" / METHODS[m][0]).is_dir():
                        info["reason"] += " A rerun with more workers was still running when this table was made."
                else:
                    info = {"status": "not run", "reason": not_run_reason(ds, m, sizes)}
            elif info.get("graph_k_failed_attempts"):
                tried = ", ".join(a.split(":")[0] for a in info["graph_k_failed_attempts"])
                info["reason"] = (f"completed only with graph_k={info['graph_k_used']}: the default and the next settings "
                                  f"({tried}) gave a cell graph whose clone-bearing parts were disconnected")
            rows.append({"dataset": ds, "method": m, "status": info.get("status"), "reason": info.get("reason", ""),
                         "n_clones_represented": n, "n_clones_ge2": int((sizes >= 2).sum()),
                         "n_clones_ge10": int((sizes >= 10).sum()),
                         "coverage_of_ge2": n / max(int((sizes >= 2).sum()), 1),
                         "n_two_cell_clones_represented": 0 if rep is None else int((sizes.reindex(rep.clones) == 2).sum()),
                         "n_two_cell_clones": int((sizes == 2).sum()),
                         "wall_seconds_task": wall, "peak_rss_gb_task": rss,
                         "seconds_method": float(sum(info.get("seconds", {}).values())) if info.get("seconds") else None,
                         "peak_rss_gb_method": info.get("peak_rss_gb_self"),
                         "distance_workers": info.get("distance_workers", 1 if m.startswith("clonotrace")
                                                      and info.get("status") == "completed" else None),
                         "graph_k_used": info.get("graph_k_used"),
                         "seconds_stages": json.dumps(info.get("seconds", {}))})
    return rows


PROFILE = RUNS.parent / "profile_run"           # make_resource_jobs.py / profile_threadfin_run.py outputs
RESOURCE_TOOLS = {"clone2vec_min2": "clone2vec_min2", "clonotrace_revised_min2": "clonotrace_min2",
                  "benisse_min2": "benisse_min2"}
CORES = (1, 2, 4, 8, 16)


def resource_rows():
    """Time and memory with 1-16 cores (GSE287123 RBD) and on 5-50 % subsets of GSE195673 (8 cores), and whether
    each tool gives the same result with any number of cores."""
    rows = []
    plan = [("cores", "mouse_rbd", f"full_w{n}", n) for n in CORES]
    plan += [("size", f"ln_vaccine_f{f}", "full", 8) for f in ("0.05", "0.1", "0.25", "0.5")]
    for exp, ds, variant, cores in plan:
        if not (INPUTS / ds).is_dir():
            continue
        n2 = int((clone_sizes(load_cells(ds)) >= 2).sum())
        base = {"experiment": exp, "dataset": ds, "cores": cores, "n_clones_ge2": n2}
        prof = PROFILE / f"{ds}_{variant if exp == 'cores' else 'full_w8'}.json"
        if prof.exists():                       # the complete Threadfin analysis, step by step
            r = json.loads(prof.read_text())
            rows.append({**base, "tool": "threadfin_run", "status": "completed", "seconds": r["total_seconds"],
                         "peak_rss_gb": r["peak_rss_gb"],
                         "stages": json.dumps({k: v["seconds"] for k, v in r["steps"].items()})})
        if exp == "size":                       # Threadfin's clone representation alone, as timed for the others
            info = run_info(ds, "full", "threadfin_kernel")
            if info.get("status") == "completed":
                rows.append({**base, "tool": "threadfin_kernel", "status": "completed",
                             "seconds": float(sum(info["seconds"].values())), "peak_rss_gb": info.get("peak_rss_gb_self"),
                             "stages": json.dumps(info["seconds"])})
        for m, sub in RESOURCE_TOOLS.items():
            d = RUNS / ds / variant / sub
            if not (d / "run.json").exists():
                if (d / "time.txt").exists():
                    rows.append({**base, "tool": m, "status": "running or killed"})
                continue
            info = json.loads((d / "run.json").read_text())
            wall, rss = time_file(d / "time.txt")
            rows.append({**base, "tool": m, "status": info.get("status"), "seconds": wall, "peak_rss_gb": rss,
                         "stages": json.dumps(info.get("seconds", {}))})
    repro = []
    for m in RESOURCE_TOOLS:
        reps = {f"full_w{n}": load_rep("mouse_rbd", f"full_w{n}", m) for n in CORES}
        reps = {v: r for v, r in reps.items() if r is not None}
        if len(reps) < 2:
            continue
        ref_v = next(iter(reps))
        ref = reps[ref_v]
        for v, r in reps.items():
            if v == ref_v:
                continue
            common = ref.clones.intersection(r.clones)
            a, b = ref.sub_distance(common), r.sub_distance(common)
            iu = np.triu_indices(len(common), 1)
            repro.append({"tool": m, "reference_cores": int(ref_v.split("_w")[1]), "cores": int(v.split("_w")[1]),
                          "n_clones": len(common), "same_clones": len(common) == len(ref.clones) == len(r.clones),
                          "identical": bool(np.array_equal(a, b)), "max_abs_difference": float(np.abs(a - b).max()),
                          "distance_spearman": float(stats.spearmanr(a[iu], b[iu]).statistic)})
    for ds in ("mouse_np", "mouse_rbd"):        # Threadfin: verify_identical.py, released 4.3.1 vs n_jobs 1 and 8
        files = {k: PROFILE / f"verify_{ds}_{k}.json" for k in ("head", "new1", "new8")}
        if all(f.exists() for f in files.values()):
            r = {k: json.loads(f.read_text()) for k, f in files.items()}
            for k in ("new1", "new8"):
                same = (r[k]["labels"] == r["head"]["labels"] and r[k]["programmes"] == r["head"]["programmes"]
                        and r[k]["xy_sum"] == r["head"]["xy_sum"])
                repro.append({"tool": "threadfin_run", "dataset": ds, "reference_cores": 1, "cores": int(k[-1]),
                              "identical": bool(same), "note": "clone map and programmes vs released 4.3.1"})
    return rows, repro


def full_run_rows(datasets):
    """The complete Threadfin analysis (profile_threadfin_run.py: profiles, coherence test, tuned clone map,
    programmes, one label test) on every dataset, with the released 4.3.1 in one process."""
    rows = []
    for ds in datasets:
        f = PROFILE / f"{ds}_v431_seq.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows.append({"dataset": ds, "threadfin": r["threadfin"], "n_cells": r["n_cells"], "n_clones_ge2": r["n_clones_ge2"],
                     **{f"seconds_{k}": v["seconds"] for k, v in r["steps"].items()},
                     "total_seconds": r["total_seconds"], "peak_rss_gb": r["peak_rss_gb"]})
    return rows


def run_status_rows(datasets):
    """Every run of every tool on the full data and on the disjoint halves: did it finish, and with which settings."""
    rows = []
    for ds in datasets:
        for vdir in sorted(p for p in (RUNS / ds).iterdir() if p.is_dir()):
            variant = vdir.name
            if not (variant == "full" or variant.startswith("half")):
                continue
            for sub in sorted(set(s for s, _ in METHODS.values()) | {"clonotrace_min10", "clonotrace_min2"}):
                d = vdir / sub
                stopped = sorted(vdir.glob(f"{sub}_*stopped*/run.json"))
                if (d / "run.json").exists():
                    info = json.loads((d / "run.json").read_text())
                elif stopped:
                    info = json.loads(stopped[-1].read_text())
                    if d.is_dir():
                        info["reason"] += " A rerun with more workers was still running when this table was made."
                else:
                    continue
                rows.append({"dataset": ds, "variant": variant, "tool_run": sub, "status": info.get("status"),
                             "graph_k_used": info.get("graph_k_used"),
                             "graph_k_failed_attempts": len(info.get("graph_k_failed_attempts", [])),
                             "seconds": float(sum(info.get("seconds", {}).values())) if info.get("seconds") else None,
                             "reason": str(info.get("reason", ""))[:300]})
    return rows


REAL = ["gut_gc_reporter", "mouse_rbd", "mouse_np", "malaria_late", "malaria", "ln_vaccine", "spont_gc", "gc_np_pc",
        "covid_bcell"]
SIMS = [f"sim_{r}_b{b}_s{s}" for r in ("small", "large") for b in ("0", "1.5") for s in (1, 2, 3)]
SETS_FULL = [("native: clones >= 10 cells", NATIVE, 10), ("forced: clones >= 2 cells", ALL2, 2),
             ("ablation: clones >= 2 cells", ABLATION, 2), ("clone2vec default: clones >= 3 cells",
                                                            ["threadfin_kernel", "clone2vec_min3", "mean_pca"], 3)]
SETS_HALF = [("forced: >= 2 cells per half", ALL2, 2), ("native: >= 10 cells per half", NATIVE, 10),
             ("ablation: >= 2 cells per half", ABLATION, 2), ("sensitivity: >= 2 cells per half", SENS_SET, 2)]


def save(rows, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    print(f"wrote {name}.csv ({len(df)} rows)", flush=True)
    return df


def main(what):
    if what in ("coverage", "all"):
        save(coverage_rows(REAL + ["larry_d2", "larry_d46"] + SIMS), "coverage_runtime")
    if what in ("status", "coverage", "all"):
        save(run_status_rows(REAL + ["larry_d2", "larry_d46"]), "run_status")
    if what in ("resources", "all"):
        rows, repro = resource_rows()
        save(rows, "resources")
        save(repro, "resources_reproducibility")
        save(full_run_rows(REAL + ["larry_d2", "larry_d46"]), "threadfin_full_run")
    if what in ("annotation", "all"):
        save([r for ds in REAL + ["larry_d2"] for r in annotation_rows(ds, SETS_FULL)], "annotation_recovery")
    if what in ("pairwise", "all"):
        others = [m for m in METHODS if m != "threadfin_kernel"]
        save([r for ds in REAL + ["larry_d2"] for r in pairwise_rows(ds, others)], "annotation_pairwise")
    if what in ("disjoint", "all"):
        save([r for ds in REAL for r in disjoint_rows(ds, SETS_HALF)], "annotation_recovery_disjoint")
    if what in ("reproducibility", "all"):
        save([r for ds in REAL + ["larry_d2", "larry_d46"] for r in reproducibility_rows(ds, SETS_HALF)],
             "split_half_reproducibility")
    if what in ("larry", "all"):
        save(larry_linking_rows([("native: clones >= 10 cells", NATIVE, 10), ("forced: clones >= 2 cells", ALL2, 2)]),
             "larry_clone_linking")
    if what in ("simulation", "all"):
        save(simulation_rows(SIMS, [("native: clones >= 10 cells", NATIVE, 10), ("forced: clones >= 2 cells", ALL2, 2),
                                    ("ablation: clones >= 2 cells", ABLATION, 2),
                                    ("sensitivity: clones >= 2 cells", SENS_SET, 2)]), "simulation")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
