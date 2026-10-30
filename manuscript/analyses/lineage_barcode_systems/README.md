# Lineage-barcode and T-cell receptor data

Threadfin needs only cell states and clone labels, so it also runs on clones defined by an experimental
lineage barcode or by a T-cell receptor. These analyses were made between 2026-10-07 and 2026-10-17. They
formed the 17 October version of Figure 4 ("measured validation"), which was replaced on 19 October by the
RBD division-reporter figure: LARRY and WATERMELON are not B-cell data, and the manuscript does not claim
prediction or compare methods. They are kept here because they test Threadfin where a later measurement on
*other* cells of the same clone exists — something no B-cell dataset provides.

All three datasets are public. Large intermediate tables (cell embeddings, all profiles) are not stored;
the scripts regenerate them.

## LARRY — in-vitro haematopoiesis ([`larry/`](larry))

**Data.** Weinreb *et al.* 2020 *Science* (GSE140802), the dataset of Clonotrace's tutorial: progenitors
tagged with heritable lineage barcodes and sampled on days 2, 4 and 6 of culture.

**Question.** Does a barcode's day-2 clone profile predict the fate its sister cells adopt by day 6?

**Design.** 172 barcodes with at least two day-2 cells and four day-6 cells (402 day-2 cells, 2,286 day-6
cells). Profiles were built from day-2 cells only (the embedding was also computed on day-2 cells). The
day-6 composition was predicted with ridge regression; folds hold out whole barcodes (3 repeats × 5 folds),
with ridge penalties tuned inside each training set, and every method used the same folds.

**Results** (barcode-equal out-of-fold R²; `future_scores.csv`, `summary.json`).

| Day-6 fate | RNA centroid | Threadfin kernel | paired ΔR² (kernel − centroid), 95 % barcode bootstrap |
|---|---|---|---|
| neutrophil fraction | 0.188 | 0.221 | +0.033 [−0.016, 0.081] |
| monocyte fraction | 0.131 | 0.170 | +0.040 [−0.002, 0.077] |

The kernel profile had the highest point estimate, but both intervals include zero: this evaluation does
not show that kernel profiles predict better than a plain centroid. Sister wells (211 barcodes) reproduced
each barcode's fate bias (ρ 0.73 and 0.69), showing that fate bias is a reproducible clone property in
this system. The unsupervised preprocessing used all day-2 cells (transductive), not only training cells.

The same data are used, with strict per-method protocols, in the [benchmark](../../benchmarking/README.md)
(day-2 representation → day-6 fate for every tool, and re-identification of a barcode's day-6 clone from its
day-4 clone, the demonstration of Clonotrace's tutorial).

## WATERMELON — persister cells under targeted therapy ([`watermelon/`](watermelon))

**Data.** Oren *et al.* 2021 *Nature* (GSE150949): PC9 lung-cancer cells carrying WATERMELON lineage barcodes
and a proliferation reporter, treated with osimertinib for 14 days.

**Question.** Is a clone's state during days 0–7 associated with the proliferation gate its cells occupy at
day 14?

**Design.** Profiles from 29,429 day 0/3/7 cells (2,089 clones with ≥ 2 cells); outcome = fraction of the
clone's day-14 cells in the mCherry-high (non-cycling) or mCherry-low (cycling) gate, for the 1,016 clones
with ≥ 2 day-14 cells. Association by `profile_association` (2,000 permutations).

**Results.** Excess R² for the non-cycling gate 0.075 with mean profiles and 0.052 with kernel profiles;
cycling gate 0.025 and 0.023 (P = 0.0005 for each). With Threadfin 4.6.0, which shuffles the label among clones
of similar size (`watermelon/summary_460.json`, re-derived on 2026-10-30): 0.075 and 0.052; 0.025 and 0.024
(P = 0.0005 for each). The association is real but weak, and the simpler mean
profile does at least as well as the kernel profile. Days 3 and 7 are already under treatment, so this is
not a pre-treatment prediction, and the gates are proliferation states among survivors, not persister versus
non-persister.

## NSCLC T cells — Clonotrace's longitudinal dataset ([`nsclc_tcr/`](nsclc_tcr))

**Data.** Peripheral T cells of 10 patients with non-small-cell lung cancer sampled over treatment cycles
(GSE266219), as analysed by Clonotrace; clones are exact paired TRA/TRB amino-acid sequences within a
patient.

**Question.** Does a T-cell clone keep its transcriptional profile between treatment cycles?

**Design.** 124,534 cells with an exact paired TCR; clone × cycle units with ≥ 4 cells; for each of 29
patient × cycle intervals, the distance between the same clone's profiles at the two cycles was compared
with distances between different clones of similar size (200 permutations).

**Results** (`longitudinal_distance_comparison.csv`). The same clone was closer to itself than to other
clones in 28 of 29 intervals for all three representations (median distance ratio: kernel 0.63, Threadfin
mean 0.66, RNA centroid 0.66; 1,262 clone pairs in total). T-cell clones keep their state across treatment
cycles, and the three representations agree.

> T-cell receptor support is outside the scope of the Threadfin package and manuscript; this analysis only
> shows that the clone-level machinery is not specific to B cells.

## Files

| Folder | Scripts | Results |
|---|---|---|
| `larry/` | `fetch_clonotrace_hematopoiesis.py` (download), `reanalyse_clonotrace_larry.py` (profiles, split-sample check), `reanalyse_larry_prediction.py` (day-2 → day-6 prediction) | `summary.json`, `future_scores.csv`, `future_folds.csv`, `future_predictions.csv.gz`, `day2_predictor_coverage.csv`, audits |
| `watermelon/` | `reanalyse_watermelon.py`, `watermelon_cell_vs_clone.py` | `summary.json`, `clone_table.csv`, `early_clone_table.csv`, `clone_map_mcherry.png` |
| `nsclc_tcr/` | `export_clonotrace_nsclc.R` (export from the authors' Seurat objects), `prepare_clonotrace_nsclc.py`, `reanalyse_clonotrace_nsclc.py` | `summary.json`, `per_patient_summary.csv`, `longitudinal_distance_comparison.csv`, audits |

The scripts were written to run from the repository's former `case_studies/` folder (they import
`datasets.py` and `run_case_study.py`); add `manuscript/case_studies` to `PYTHONPATH` to run them.
