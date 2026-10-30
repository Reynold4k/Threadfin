# Gut immunisation: all B-lineage cells, and what centring does to the plasma-cell branches

Data: Siniscalco *et al.* 2025 *Immunity* (GSE284622), three *S1pr2*-CreERT2 tdTomato GC-fate reporter mice
immunised orally with ovalbumin and cholera toxin; mesenteric lymph node (MLN), Peyer's patches (PP) and
lamina propria (LP); cells sorted into GC-fate-labelled (tdTomato⁺, "GC") and unlabelled ("nonGC") gates.
This is the dataset of Figure 3g–j and Supplementary Figs. 4–5. The analyses here (2026-10-12 to
2026-10-16) answered two questions raised while Figure 3 was built.

## 1. Does adding the non-GC B cells change the picture?

**What was done.** Threadfin was re-run on all B-lineage cells of the authors' annotation (B cells, GC B
cells and plasma cells; 15,432 cells, 11,714 with a BCR), instead of the subset used for Figure 3
(`allB_threadfin_run.py`, outputs in `allB_threadfin/`). Three centring variants were run: on each mouse
(as in the manuscript), none, and composition-balanced (each mouse's shift estimated from cell types present
in all mice).

**What was found.** Adding the non-GC B cells hardly changes the clones: 957 clones with ≥ 2 cells against
958, because 96 % of the BCR-carrying non-GC B cells are singletons and cannot be profiled. The
mouse-centred run reproduces Figure 3 (three programmes, stability 0.88 / 0.51 / 0.59; the GC-fate sort
fraction explains 7.7 % of profile variation, lamina-propria fraction 47 %, isotype 20 %; all P = 0.0005).
With Threadfin 4.6.0, which shuffles each label among clones of the same mouse and similar size, all nine label
tests of this run still give P = 0.0005 (`allB_threadfin/donor/label_effects_460.csv`, re-derived on
2026-10-30; the permutation means change by at most 0.0023, the excess R² of the plasma-cell fraction from 0.554 to 0.551).

## 2. Are the two plasma-cell branches two kinds of clones?

**What was done.** `check_plasma_arms.py` counted, for the 431 plasma-cell-dominated clones of the Figure 3
run, which mouse each clone came from in each region of the clone embedding, and compared the share of
nonGC-sorted plasma cells in each region with the share expected from the mice that contribute to it.
`plasma_arms_by_mouse.png` shows the result.

**What was found.**

| Region of the Figure 3 embedding | mouse 2 | mouse 3 | mouse 4 | nonGC share of plasma cells, observed | expected from the mice alone |
|---|---|---|---|---|---|
| bridging clones (P2) | 68 | 2 | 24 | 0.170 | 0.170 |
| lower branch (P2) | 12 | 136 | 0 | 0.015 | 0.017 |
| upper branch (P3) | 153 | 0 | 36 | 0.173 | 0.171 |

The lower branch is almost entirely mouse 3, whose sampled cells were 52 % plasma cells (22 % and 12 % in
the other mice) and only 3.6 % of whose plasma cells were nonGC-sorted. Because each clone is centred on
its own mouse, all of mouse 3's plasma-cell clones shift together into a branch of their own. In this
all-B run the two branches merge into one stable plasma-cell programme without centring (stability 0.86) and
with composition-balanced centring (0.81; the balanced run also returns an unstable third programme, 0.18).
The manuscript reports the same merge from the Figure 3 run (Jaccard 0.82 and 0.81). What survives every variant: the path from GC clones through bridging clones to plasma-cell clones,
and, within a mouse, the grouping of plasma-cell clones that contain nonGC-sorted plasma cells (Spearman's
ρ with the neighbourhood mean 0.70 in mouse 2 and 0.46 in mouse 4 without centring; P = 0.0005, also when
permuting within the lamina-propria stratum).

**How the manuscript uses this.** Figure 3 keeps the gut path; the Results state the mouse composition of
the two branches, and Supplementary Fig. 5 shows the centring variants. The general lesson is in
[Methods](../../../docs/METHODS.md), "Centring on the sampling context, and when it misleads".

## 3. Where the GC-fate-labelled cells sit on the cell-level embedding

`map_gc_derived_to_scrna_umap.py` projects the GC-sort label and each clone's GC-sorted fraction onto the
authors' cell embedding (`gc_derived_umap_summary.json`, `gc_sort_fraction_validation.csv`). The clone-level
GC-sorted fraction recomputed from the original metadata reproduced all 958 values used in the figures. The
GC/nonGC label is a sorting gate, not a continuous tdTomato intensity.

## Files

| File | Content |
|---|---|
| `allB_threadfin_run.py`, `allB_threadfin_run.sbatch` | the all-B re-run with three centring variants |
| `allB_story_analysis.py` | path and clone classes along the clone-profile axis, per variant |
| `check_plasma_arms.py`, `plasma_arms_by_mouse.py`, `plasma_arms_by_mouse.png` (`.pdf`) | mouse composition of the plasma-cell branches |
| `map_gc_derived_to_scrna_umap.py`, `gc_derived_umap_summary.json`, `gc_sort_fraction_validation.csv` | GC-sort label on the cell embedding; recomputation of the clone-level fraction |
| `scrna_umap_gc_sort_label.pdf`, `scrna_umap_gc_derived_fraction.pdf` | the authors' cell embedding coloured by the GC-sort label of each cell and by its clone's GC-sorted fraction |
| `embedding_by_*.png` | the Figure 3 clone embedding coloured by compartment, GC sort, isotype, tissue and mutation frequency |
| `embedding_display_windows.png` | the same clone map with clones of at least 2, 5 and 10 cells shown (display only; every clone is analysed) |
| `cells_context.png` | the 11,723 cells with a receptor on the cell embedding, coloured by the authors' cell-type annotation |
| `Figure_3_allB_candidate_numbers.json` | numbers of a candidate all-B Figure 3 (not adopted) |
| `allB_threadfin/<variant>/` | `summary.json`, programmes, label effects, clone tables of each centring variant |
| `allB_threadfin/story/` | per-variant path statistics and the plasma-branch check (`plasma_arms_check.json`) |

The scripts expect the project layout used to produce them (paths at the top of each script).
