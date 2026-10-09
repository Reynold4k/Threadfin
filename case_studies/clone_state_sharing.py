#!/usr/bin/env python3
"""Within-mouse clone/state sharing analysis for the malaria case study.

This is an association analysis of co-observed cell states in same-day, same-mouse
BCR clones.  It deliberately contains no temporal or lineage-direction inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("THREADFIN_DATA", "/data/scratch/projects/punim1236/threadfin_data")) / "gse286215_malaria"
STATES = ("GC", "PB", "Memory")
PAIRS = (("GC", "PB"), ("GC", "Memory"), ("Memory", "PB"))


def clean_gene(value: object) -> str:
    """Keep the reported allele-level call for the deliberately strict control."""
    return "" if pd.isna(value) else str(value).strip()


def normalize_barcode(values: pd.Series) -> pd.Series:
    """Put Seurat and AIRR cell identifiers into the same 10x barcode form."""
    return (values.astype(str).str.replace(r"_contig_\d+$", "", regex=True)
            .str.replace(r"-\d+$", "", regex=True))


def valid_strict_value(value: object) -> bool:
    """Exact control excludes missing and multi/ambiguous AIRR assignments."""
    text = clean_gene(value)
    return bool(text) and "," not in text and "ambig" not in text.lower()


def first_mode(series: pd.Series) -> object:
    series = series.dropna()
    if series.empty:
        return np.nan
    counts = series.value_counts()
    return sorted(counts[counts == counts.max()].index.astype(str))[0]


def read_cohort(name: str, cells_path: Path, bcr_path: Path, metadata_path: Path | None) -> tuple[pd.DataFrame, dict]:
    cells = pd.read_csv(cells_path, compression="gzip")
    barcode_col = "Unnamed: 0" if "Unnamed: 0" in cells else cells.columns[0]
    cells = cells.rename(columns={barcode_col: "barcode"})
    # Seurat cell names retain the 10x ``-1`` suffix while AIRR ``cell_id`` does not.
    cells["barcode"] = normalize_barcode(cells["barcode"])
    cells["cohort"] = name
    cells["day"] = cells["timepoint"].astype(str)

    # Treatment is used only when directly present in the source cell metadata.
    treatment_evidence = {"available": False, "metadata_file": None, "values": []}
    if metadata_path is not None and metadata_path.exists():
        meta = pd.read_csv(metadata_path, compression="gzip", low_memory=False)
        meta_barcode = "Unnamed: 0" if "Unnamed: 0" in meta else meta.columns[0]
        if "treatment" in meta:
            treatment = meta[[meta_barcode, "treatment"]].rename(columns={meta_barcode: "barcode"})
            treatment["barcode"] = normalize_barcode(treatment["barcode"])
            if "treatment" in cells:
                cells = cells.drop(columns="treatment")
            cells = cells.merge(treatment, on="barcode", how="left", validate="one_to_one")
            vals = sorted(cells["treatment"].dropna().astype(str).unique().tolist())
            treatment_evidence = {"available": bool(vals), "metadata_file": str(metadata_path), "values": vals}
        else:
            cells["treatment"] = np.nan
    else:
        cells["treatment"] = np.nan

    header = pd.read_csv(bcr_path, sep="\t", compression="gzip", nrows=0)
    cols = [c for c in ["cell_id", "locus", "v_call", "j_call", "junction", "c_call", "productive"] if c in header]
    bcr = pd.read_csv(bcr_path, sep="\t", compression="gzip", usecols=cols, low_memory=False)
    bcr["cell_id"] = bcr["cell_id"].astype(str)
    if "productive" in bcr:
        bcr = bcr.loc[bcr["productive"].astype(str).str.upper().isin({"T", "TRUE", "1"})].copy()
    heavy = bcr.loc[bcr["locus"].eq("IGH")].copy()
    # Multiple IGH contigs cannot support an unambiguous exact-clonotype control.
    heavy_n = heavy.groupby("cell_id", observed=True).size()
    heavy = heavy.loc[heavy["cell_id"].isin(heavy_n[heavy_n == 1].index)].copy()
    strict_valid = heavy[["v_call", "j_call", "junction"]].map(valid_strict_value).all(axis=1)
    heavy = heavy.loc[strict_valid].copy()
    heavy["strict_igh"] = (heavy["v_call"].map(clean_gene) + "|" + heavy["j_call"].map(clean_gene)
                           + "|" + heavy["junction"].map(clean_gene))

    light = bcr.loc[bcr["locus"].isin(["IGK", "IGL"])].copy()
    # Paired-light exact controls have the same productive, unambiguous AIRR
    # requirement as the heavy-chain exact control.
    light_valid = light[["v_call", "j_call", "junction"]].map(valid_strict_value).all(axis=1)
    light = light.loc[light_valid].copy()
    light["light_signature"] = (light["locus"].map(clean_gene) + "|" + light["v_call"].map(clean_gene)
                                + "|" + light["j_call"].map(clean_gene) + "|" + light["junction"].map(clean_gene))
    light_n = light.groupby("cell_id", observed=True)["light_signature"].nunique()
    light = light.loc[light["cell_id"].isin(light_n[light_n == 1].index), ["cell_id", "light_signature"]].drop_duplicates("cell_id")
    heavy = heavy.merge(light, on="cell_id", how="left", validate="one_to_one")
    heavy["strict_igh_light"] = heavy["strict_igh"].where(heavy["light_signature"].notna()) + "|" + heavy["light_signature"]
    heavy = heavy.rename(columns={"cell_id": "barcode"})
    keep = ["barcode", "strict_igh", "strict_igh_light", "light_signature", "v_call", "j_call", "junction", "c_call"]
    cells = cells.merge(heavy[keep], on="barcode", how="left", validate="one_to_one")
    return cells, treatment_evidence


def occupancy(cells: pd.DataFrame, definition: str, key: str) -> pd.DataFrame:
    cols = ["cohort", "donor", "day", "treatment", "clone_id", "cell_state", "isotype", "mutation_frequency", "strict_igh", "strict_igh_light", "light_signature"]
    cells = cells.copy()
    for col in ("strict_igh", "strict_igh_light", "light_signature"):
        if col not in cells:
            cells[col] = np.nan
    if key not in cols:
        cols.append(key)
    use = cells.loc[cells[key].notna(), cols].copy()
    use["analysis_clone"] = use[key].astype(str)
    ident = ["cohort", "donor", "day", "treatment", "analysis_clone"]
    counts = (use.loc[use["cell_state"].isin(STATES)].groupby(ident + ["cell_state"], dropna=False, observed=True)
              .size().unstack("cell_state", fill_value=0).reindex(columns=STATES, fill_value=0).reset_index())
    total = use.groupby(ident, dropna=False, observed=True).size().rename("n_cells").reset_index()
    out = total.merge(counts, on=ident, how="left").fillna({s: 0 for s in STATES})
    descriptor = use.groupby(ident, dropna=False, observed=True).agg(
        clone_id=("clone_id", first_mode), isotype=("isotype", first_mode),
        mutation_frequency=("mutation_frequency", "mean"), strict_igh_n=("strict_igh", "nunique"),
        strict_igh_light_n=("strict_igh_light", "nunique"), paired_light_cells=("light_signature", "count"),
    ).reset_index()
    out = out.merge(descriptor, on=ident, how="left")
    out["clone_definition"] = definition
    out["n_states_observed"] = (out[list(STATES)] > 0).sum(axis=1)
    out["GC_bearing"] = out["GC"] > 0
    return out


def pair_stat(counts: np.ndarray, pair: tuple[str, str], threshold: int,
              total_sizes: np.ndarray | None = None, gc_conditional: bool = True) -> tuple[int, int, float]:
    a, b = STATES.index(pair[0]), STATES.index(pair[1])
    if total_sizes is None:
        total_sizes = counts.sum(axis=1)
    eligible = np.asarray(total_sizes) >= threshold
    # Optional secondary, conditional denominator for GC-containing contrasts.
    if gc_conditional and "GC" in pair:
        eligible &= counts[:, STATES.index("GC")] > 0
    shared = eligible & (counts[:, a] > 0) & (counts[:, b] > 0)
    den = int(eligible.sum())
    num = int(shared.sum())
    return num, den, num / den if den else np.nan


def permutation_rows(cells: pd.DataFrame, occ: pd.DataFrame, n_perm: int, seed: int, threshold: int = 2,
                     gc_conditional: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for (cohort, donor, day, treatment), g in occ.groupby(["cohort", "donor", "day", "treatment"], dropna=False, observed=True):
        # Use original per-cell isotypes.  Thus both nulls preserve clone size and
        # mouse state totals exactly; the second also preserves totals per isotype.
        mask = ((cells.cohort == cohort) & (cells.donor == donor) & (cells.day == day) & cells.clone_id.notna())
        if pd.isna(treatment):
            mask &= cells.treatment.isna()
        else:
            mask &= cells.treatment == treatment
        sub = cells.loc[mask, ["clone_id", "cell_state", "isotype"]].copy()
        codes, unique_clones = pd.factorize(sub.clone_id, sort=True)
        clone_cells = codes.astype(int)
        state_cells = sub.cell_state.fillna("Other").where(sub.cell_state.isin(STATES), "Other").to_numpy(dtype=object)
        iso_cells = sub.isotype.fillna("__missing__").astype(str).to_numpy(dtype=object)
        if len(unique_clones) != len(g):
            raise RuntimeError(f"clone occupancy mismatch for {cohort}/{donor}")
        observed = np.zeros((len(g), len(STATES)), dtype=int)
        for j, s in enumerate(STATES):
            observed[:, j] = np.bincount(clone_cells[state_cells == s], minlength=len(g))
        nulls = {"within_mouse": {p: [] for p in PAIRS}, "within_mouse_isotype": {p: [] for p in PAIRS}}
        for null_name in nulls:
            for _ in range(n_perm):
                perm = state_cells.copy()
                if null_name == "within_mouse":
                    perm = rng.permutation(perm)
                else:
                    for iso in np.unique(iso_cells):
                        ix = np.flatnonzero(iso_cells == iso)
                        perm[ix] = rng.permutation(perm[ix])
                c = np.zeros_like(observed)
                for j, s in enumerate(STATES):
                    c[:, j] = np.bincount(clone_cells[perm == s], minlength=len(g))
                for p in PAIRS:
                    nulls[null_name][p].append(pair_stat(c, p, threshold, g.n_cells.to_numpy(), gc_conditional)[2])
        for p in PAIRS:
            num, den, rate = pair_stat(observed, p, threshold, g.n_cells.to_numpy(), gc_conditional)
            if den == 0:
                continue
            for null_name, values in nulls.items():
                x = np.asarray(values[p], dtype=float)
                valid = x[np.isfinite(x)]
                # A null draw with no eligible clone is unavailable, not a zero effect.
                expected = float(valid.mean()) if len(valid) else np.nan
                rows.append(dict(cohort=cohort, donor=donor, day=day, treatment=treatment, clone_definition="threadfin", threshold=threshold,
                                 state_a=p[0], state_b=p[1], pair=f"{p[0]}+{p[1]}", observed_shared=num, denominator_clones=den,
                                 observed_rate=rate, n_clones=int(len(g)), n_cells=int(len(clone_cells)),
                                 null_type=null_name, null_model="cell_state_label_permutation", null="state_labels",
                                 denominator_definition="GC-bearing size>=2" if gc_conditional and "GC" in p else "all clones size>=2",
                                 null_model_available=True, null_mean_rate=expected,
                                 effect_rate_difference=float(rate - expected) if np.isfinite(expected) else np.nan,
                                 enrichment_p=float((1 + np.sum(valid >= rate)) / (1 + len(valid))) if len(valid) else np.nan,
                                 n_null_draws=int(len(x)), n_null_valid_draws=int(len(valid)),
                                 null_zero_denominator_fraction=float(1 - len(valid) / len(x))))
    return pd.DataFrame(rows)


def observed_rows(occ: pd.DataFrame, definition: str, threshold: int, include_zero: bool = False,
                  gc_conditional: bool = True) -> pd.DataFrame:
    rows = []
    for keys, g in occ.groupby(["cohort", "donor", "day", "treatment"], dropna=False, observed=True):
        counts = g.loc[:, list(STATES)].to_numpy(dtype=int)
        for p in PAIRS:
            num, den, rate = pair_stat(counts, p, threshold, g.n_cells.to_numpy(), gc_conditional)
            if den == 0 and not include_zero:
                continue
            rows.append(dict(cohort=keys[0], donor=keys[1], day=keys[2], treatment=keys[3], clone_definition=definition,
                             threshold=threshold, state_a=p[0], state_b=p[1], pair=f"{p[0]}+{p[1]}", observed_shared=num,
                             denominator_clones=den, observed_rate=rate, n_clones=len(g), n_cells=int(g.n_cells.sum()),
                             null_type="not_assessed", null_model="not_assessed", null="not_assessed", null_model_available=False,
                             null_mean_rate=np.nan, effect_rate_difference=np.nan, enrichment_p=np.nan,
                             n_null_draws=0, n_null_valid_draws=0, null_zero_denominator_fraction=np.nan))
    return pd.DataFrame(rows)


def summarize_pairs(rows: pd.DataFrame, occ: pd.DataFrame) -> pd.DataFrame:
    group = ["cohort", "day", "treatment", "clone_definition", "threshold", "pair", "state_a", "state_b", "null_type", "null_model", "null"]
    def agg(g: pd.DataFrame) -> pd.Series:
        shared = int(g.observed_shared.sum()); den = int(g.denominator_clones.sum())
        result = {"n_mice": int(g.donor.nunique()), "shared_clones": shared, "denominator_clones": den,
                  "pooled_rate": shared / den if den else np.nan, "mean_mouse_rate": g.observed_rate.mean()}
        result["null_model_available"] = bool(g.null_model_available.all())
        result["null_mean_rate"] = g.null_mean_rate.mean()
        result["mean_rate_difference"] = g.effect_rate_difference.mean()
        result["median_enrichment_p"] = g.enrichment_p.median()
        result["mean_null_zero_denominator_fraction"] = g.null_zero_denominator_fraction.mean()
        return pd.Series(result)
    summary = rows.groupby(group, dropna=False, observed=True).apply(agg, include_groups=False).reset_index()
    # Descriptives deliberately do not imply a directional relationship between states.
    desc = []
    for _, r in summary.iterrows():
        sub = occ[(occ.cohort == r.cohort) & (occ.day == r.day) & (occ.clone_definition == r.clone_definition)]
        if pd.notna(r.treatment): sub = sub[sub.treatment == r.treatment]
        elif sub.treatment.notna().any(): sub = sub[sub.treatment.isna()]
        a, b = r.state_a, r.state_b
        shared = sub[(sub[a] > 0) & (sub[b] > 0)]
        ref = sub[(sub[a] > 0) | (sub[b] > 0)]
        desc.append({"shared_mean_mutation_frequency": shared.mutation_frequency.mean(), "reference_mean_mutation_frequency": ref.mutation_frequency.mean(),
                     "shared_modal_isotype": first_mode(shared.isotype), "shared_clones_with_paired_light": int((shared.paired_light_cells > 0).sum())})
    return pd.concat([summary, pd.DataFrame(desc)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "case_studies/results/clone_state_sharing")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Directory containing exp1/exp2_bcr.tsv.gz and experiment metadata files.")
    ap.add_argument("--n-permutations", type=int, default=1999)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    if args.n_permutations < 499:
        raise ValueError("At least 499 permutations are required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = [("malaria_early", ROOT / "case_studies/results/malaria/cells.csv.gz", args.data_dir / "exp1_bcr.tsv.gz", args.data_dir / "experiment1_metadata.csv.gz"),
               ("malaria_late", ROOT / "case_studies/results/malaria_late/cells.csv.gz", args.data_dir / "exp2_bcr.tsv.gz", args.data_dir / "experiment2_metadata.csv.gz")]
    all_cells, evidence = [], {}
    for n, c, b, m in sources:
        x, e = read_cohort(n, c, b, m); all_cells.append(x); evidence[n] = e
    cells = pd.concat(all_cells, ignore_index=True)
    # Cohort stays in every key: Exp1 and Exp2 clonotypes are never merged.
    occ_threadfin = occupancy(cells, "threadfin", "clone_id")
    occ_strict = occupancy(cells, "strict_igh", "strict_igh")
    occ_light = occupancy(cells.loc[cells.strict_igh_light.notna()], "strict_igh_light", "strict_igh_light")
    occ = pd.concat([occ_threadfin, occ_strict, occ_light], ignore_index=True)
    occ.to_csv(args.output_dir / "clone_occupancy.csv.gz", index=False, compression="gzip")

    primary = permutation_rows(cells, occ_threadfin, args.n_permutations, args.seed, threshold=2)
    # Primary estimand: fixed denominator, all THREADFIN clones with >=2 cells.
    # The conditional GC-bearing analysis remains in mouse_pair_sharing as secondary.
    joint = permutation_rows(cells, occ_threadfin, args.n_permutations, args.seed, threshold=2, gc_conditional=False)
    joint.to_csv(args.output_dir / "primary_joint_sharing.csv", index=False)
    strict_rows = observed_rows(occ_strict, "strict_igh", threshold=2)
    light_rows = observed_rows(occ_light, "strict_igh_light", threshold=2)
    mouse = pd.concat([primary, strict_rows, light_rows], ignore_index=True, sort=False)
    summary = summarize_pairs(mouse, occ)
    mouse.to_csv(args.output_dir / "mouse_pair_sharing.csv", index=False)
    summary.to_csv(args.output_dir / "pair_summary.csv", index=False)
    observed_rows(occ_threadfin, "threadfin", 2, include_zero=True).to_csv(args.output_dir / "mouse_pair_coverage.csv", index=False)

    exact = pd.concat([
        observed_rows(occ_threadfin, "threadfin", 2, gc_conditional=False),
        observed_rows(occ_strict, "strict_igh", 2, gc_conditional=False),
        observed_rows(occ_light, "strict_igh_light", 2, gc_conditional=False),
    ], ignore_index=True)
    exact = (exact.groupby(["cohort", "clone_definition", "state_a", "state_b", "pair"], observed=True)
             .agg(n_mice=("donor", "nunique"), shared_clones=("observed_shared", "sum"),
                  denominator_clones=("denominator_clones", "sum"))
             .reset_index())
    exact["pooled_rate"] = exact.shared_clones / exact.denominator_clones
    exact["eligible_all_clones_ge2"] = exact.denominator_clones
    exact["denominator_definition"] = "all clones with >=2 BCR-bearing cells"
    exact["identity_note"] = "THREADFIN cluster identity; receptor similarity alone does not trace ancestry."
    exact.loc[exact.clone_definition.eq("strict_igh"), "identity_note"] = (
        "Exact reported heavy V/J/junction identity; this control alone does not trace ancestry.")
    exact.loc[exact.clone_definition.eq("strict_igh_light"), "identity_note"] = (
        "Exact paired heavy+light receptor identity; this control alone does not trace ancestry.")
    exact["null_note"] = "No permutation null was evaluated for exact receptor controls."
    exact.to_csv(args.output_dir / "exact_receptor_sharing.csv", index=False)

    sens = []
    for threshold in (2, 3):
        r = observed_rows(occ_threadfin, "threadfin", threshold)
        s = summarize_pairs(r, occ_threadfin)
        s["analysis"] = "observed_threshold_sensitivity"
        sens.append(s)
    pd.concat(sens, ignore_index=True).to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)

    selected = occ_threadfin[(occ_threadfin.GC.gt(0) & (occ_threadfin.PB.gt(0) | occ_threadfin.Memory.gt(0))) |
                             (occ_threadfin.Memory.gt(0) & occ_threadfin.PB.gt(0))].copy()
    selected.to_csv(args.output_dir / "selected_clones.csv", index=False)
    payload = {
        "analysis": "Same-day, same-mouse clone/state occupancy; no directionality, state transition, or GC re-entry is inferred.",
        "input_cohorts": {n: {"cells": str(c), "bcr": str(b), "metadata": str(m)} for n, c, b, m in sources},
        "clone_keying": "Cohort-qualified analysis; Exp1 and Exp2 are kept independent even when day/donor labels resemble each other.",
        "primary_clone_definition": "Existing THREADFIN clone_id, within donor; clone size is preserved by cell-label permutations.",
        "strict_control": "Exact donor + reported IGHV + IGHJ + junction on cells with a single IGH contig. Paired-light strict control uses uniquely observed IGK/IGL sequence only.",
        "denominator": "Primary joint sharing uses all THREADFIN clones with >=2 BCR-bearing cells. mouse_pair_sharing is a secondary GC-bearing conditional analysis for GC-containing pairs; Memory+PB uses all size-eligible clones.",
        "permutations": {"n": args.n_permutations, "seed": args.seed, "nulls": ["within_mouse", "within_mouse_isotype"], "preserved": "within-mouse clone sizes and state totals; the second null also preserves state totals within observed isotype strata"},
        "treatment_evidence": evidence,
        "limitations": ["Each mouse contributes one sampled time point, so co-occurrence cannot establish ancestry, migration, direction, or GC re-entry.", "State labels and receptor recovery are observational and incomplete.", "Strict exact clonotypes are a conservative control, not a replacement for lineage reconstruction."],
        "counts": {"cells": int(len(cells)), "threadfin_clone_rows": int(len(occ_threadfin)), "strict_igh_rows": int(len(occ_strict)), "strict_igh_light_rows": int(len(occ_light)), "selected_shared_clones": int(len(selected))},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
