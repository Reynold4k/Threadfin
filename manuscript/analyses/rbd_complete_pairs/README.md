# GSE287123 RBD cohorts: does Figure 4 depend on incompletely paired cells or on mixed-up contigs?

Two checks of the clone definition behind Figure 4 (division history in the RBD germinal-centre cohorts).
Both leave Figure 4 unchanged.

## 1. Cells without a complete heavy + light pair

**Question.** Dandelion's paired-chain clones also contain cells in which only one chain was recovered: such a
cell joins a clone on that chain alone. Of the 16,667 RBD cells with a clone, 797 have no productive light
chain (none lacks the heavy chain). Without them, 1,772 of the 1,887 profiled clones (≥ 2 cells) keep two or more
cells.

**What was done** ([`complete_pairs.py`](complete_pairs.py)). Every cell without at least one productive heavy
and one productive light chain lost its clone label, the clone profiles were recomputed (clones left with one
cell drop out) and the analyses of Figure 4 were repeated with the code that made the figure
(`figures/reserve/source/figure4_analysis_2026-10-19.py`). The full data were analysed the same way first and
reproduce the saved Figure 4 values.

| | all cells (Figure 4) | complete pairs only |
|---|---|---|
| profiled clones (cells) | 1,887 (9,825) | 1,772 (9,465) |
| recently divided fraction, excess R² of the clone profiles: both cohorts | 0.088 | 0.096 |
| … protein cohort / mRNA cohort | 0.092 / 0.069 | 0.098 / 0.080 |
| RBD bait (protein cohort) / dark-zone gate (mRNA cohort) | 0.023 / 0.037 | 0.025 / 0.045 |
| cosine of the division directions of the two cohorts (library crossover; −1 if only the libraries differed) | 0.887 | 0.884 |
| mice in which the clone axis rises with recent division | 10 of 10 | 10 of 10 |

Every association has P = 0.0005 (2,000 permutations within mice).

## 2. Contig ids reused across sequencing libraries

**Defect.** Cell Ranger names each contig after its cell barcode (`ACGGGTTTCATTAAGC-1_contig_1`). Each mouse of
GSE287123 was sequenced in several libraries, so two of its cells can carry the same barcode and therefore the
same contig ids. Dandelion indexes contigs by this id, and threadfin ≤ 4.3.1 passed the ids on unchanged, so
the chains of such cells were mixed: 18 of 21,939 cells in the RBD data, 8 of 4,855 in the NP data (and 20 of
51,837 in GSE171703; no other dataset of the manuscript has reused ids). The four RBD and one NP "light-chain-only
groups" of the earlier clone calls were this artefact: their cells do have heavy chains. threadfin 4.4.0 renames
duplicated ids after their cell.

**Effect** ([`contig_id_fix.py`](contig_id_fix.py), both calls in one Python session with `PYTHONHASHSEED=0`).

| | RBD | NP |
|---|---|---|
| adjusted Rand index between the clone calls before and after the fix | 0.9987 | 0.9993 |
| "light-chain-only" groups before → after | 4 → 0 | 1 → 0 |

Figure 4 with the corrected RBD clones (1,881 profiled clones): division excess R² 0.085 (protein 0.089, mRNA
0.069), RBD bait 0.022, dark-zone gate 0.037, crossover cosine 0.896, 10 of 10 mice — the same as the figure.

## Reproduce

```bash
python complete_pairs.py                       # results/summary.json
PYTHONHASHSEED=0 python contig_id_fix.py       # results/contig_id_fix.json; needs sc-dandelion
```

Both read the GSE287123 AIRR files through `case_studies/dandelion_clones.py` and the cached embedding of the
case study (`internal_validation/algorithm_revision/embedding_cache/mouse_rbd_embedding.h5ad`).
