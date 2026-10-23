# Analyses beyond Figures 1–4

Every analysis made for the manuscript that is not shown in the main figures. Each entry gives the question,
the data, what was done, what was found, the caveats and where the scripts and results are. Analyses whose
outputs had been removed in the 2026-10-13 clean-up (they used the earlier heavy-chain-only clone
definition) were re-run on the current paired-chain clones on 2026-10-20. The comparison with other tools is
in [`../benchmarking/`](../benchmarking/README.md); what did not hold is in
[`negative_and_withdrawn_results.md`](negative_and_withdrawn_results.md).

| # | Analysis | Data | One-line result | Location |
|---|---|---|---|---|
| 1 | [Export lag](#1-export-lag-how-far-behind-its-germinal-centre-is-a-clones-output) | *Plasmodium*, human lymph node | within a clone, output cells are less mutated than the clone's own GC cells (65 vs 8 clones in Plasmodium; 1,536 vs 821 in human lymph node) | [`export_lag/`](export_lag) |
| 2 | [Convergent receptors and the compartments of a clone](#2-convergent-receptors-in-plasmodium-infection) | *Plasmodium* | receptors shared between mice are enriched in plasmablast-dominated clones | [`../case_studies/results/pcas_arm_analyses/`](../case_studies/results/pcas_arm_analyses) |
| 3 | [States that co-occur within clones](#3-which-states-co-occur-within-a-clone-plasmodium) | *Plasmodium* | GC + plasmablast clones are enriched from day 10 on, depleted before | [`../case_studies/results/clone_state_sharing/`](../case_studies/results/clone_state_sharing) |
| 4 | [Clonal structure through infection](#4-clonal-structure-through-the-plasmodium-infection) | *Plasmodium* | the share of state explained by clone falls from 0.78 to 0.28 while clones diversify | [`../case_studies/results/malaria/timecourse.csv`](../case_studies/results/malaria/timecourse.csv) |
| 5 | [Selection on the antibody sequence](#5-selection-on-the-antibody-sequence) | 5 datasets | replacement mutations slightly above chance in model-antigen GCs; weakly related to sort gates | `../case_studies/results/<dataset>/sequence_selection.csv` |
| 6 | [Spike binding and mutation in the vaccine lymph node](#6-spike-binding-and-somatic-mutation-in-the-human-lymph-node) | human lymph node | spike-binding GC families are slightly more mutated at every visit; the label is clone-level, not affinity | [`../case_studies/results/spike_binding_audit/`](../case_studies/results/spike_binding_audit) |
| 7 | [Division, binding and zone with library controls](#7-model-antigen-germinal-centres-with-library-controls) | GSE287123 | with profiles centred within each mouse × gate library, division still explains clone profiles; bait and zone almost vanish | `../case_studies/results/mouse_gc_deep_dive/` |
| 8 | [Lineage trees](#8-somatic-hypermutation-lineage-trees) | GSE246382, NP–OVA, RBD | close relatives on the tree do not share state beyond the clone; plasma cells sit nearer the germline than GC cells of their clone | `../case_studies/results/<dataset>/lineage/` |
| 9 | [Clonal memory across time, tissue and gates](#9-clonal-memory-across-time-points-tissues-and-gates) | 11 datasets | clones keep their state across vaccine visits and between RBD-bait gates, not between lymph node and blood | [`cross_dataset/clonal_memory.csv`](cross_dataset/clonal_memory.csv) |
| 10 | [Clonally inherited gene sets](#10-clonally-inherited-genes) | 15 datasets | plasma-cell and GC programmes are inherited within clones above expression-matched genes | [`cross_dataset/geneset_heritability.csv`](cross_dataset/geneset_heritability.csv) |
| 11 | [All clone-level association tests](#11-every-clone-level-association-test) | 15 datasets | 58 annotation tests in one table | [`cross_dataset/label_effects.csv`](cross_dataset/label_effects.csv) |
| 12 | [Additional datasets](#12-additional-datasets) | 6 datasets | the pipeline on influenza, tonsil, EBV, bone marrow and bait-sorted cells (four re-run on 2026-10-20 without light-chain-only pseudo-clones) | `../case_studies/results/<dataset>/` |
| 13 | [Gut: all B cells and centring](gut_all_B_and_centring/README.md) | GSE284622 | the second plasma branch is one mouse; the GC → plasma path survives every centring | [`gut_all_B_and_centring/`](gut_all_B_and_centring) |
| 14 | [Lineage-barcode and TCR data](lineage_barcode_systems/README.md) | LARRY, WATERMELON, NSCLC T cells | day-2 profiles predict day-6 fate (kernel vs centroid not significantly different); T-cell clones keep their state across therapy cycles | [`lineage_barcode_systems/`](lineage_barcode_systems) |
| 15 | [Technical validation](technical_validation/README.md) | all | automatic clone-map settings (30/30 maps coherent); state-density model; runtimes; audits | [`technical_validation/`](technical_validation) |
| 16 | [Negative results and corrected defects](negative_and_withdrawn_results.md) | — | three falsified predictions, two withdrawn headlines, fourteen fixed defects | — |

---

## 1. Export lag: how far behind its germinal centre is a clone's output?

**Question.** Somatic hypermutation only accumulates, so within one clone an antibody-secreting cell can carry
at most the mutation load the clone's GC cells had when it left. Is the output of a clone systematically less
mutated than its own GC cells, and by how much?

**Data and method.** Class-switched cells of clones captured as both GC B cells and output cells
(plasmablasts; in the human lymph node also lymph-node plasma cells) at the same visit (and tissue). For each
clone, `threadfin.tl.export_lag` computes the mean mutation frequency of its output cells minus that of its
GC cells. The GC compartment's own maturation rate (a Theil–Sen fit on per-visit medians of switched GC cells)
converts the gap into days. Script: [`export_lag/export_lag_analysis.py`](export_lag/export_lag_analysis.py).

**Result.** (`export_lag/results/summary.json`, figure `export_lag/figures/export_lag.png`)

| Dataset | Clones in both compartments | Output less / more mutated / equal | Median gap (percentage points) [95 % CI] | GC clock (% per day) [95 % CI] | Median gap in days |
|---|---|---|---|---|---|
| *Plasmodium*, days 10–42 | 92 | 65 / 8 / 19 | −0.38 [−0.73, −0.37] | 0.101 [0.079, 0.135] | −3.7 |
| human lymph node | 2,564 | 1,536 / 821 / 207 | −0.19 [−0.19, −0.17] | 0.023 [0.002, 0.030] | −8.1 |
| *Plasmodium*, days 0–14 | 9 | 4 / 1 / 4 | 0.00 | not estimable | — |

Within a clone, the antibody-secreting cells are usually less mutated than the clone's own GC cells sampled at
the same time, as expected if output cells leave the GC before the GC cells sampled with them have finished
mutating. In *Plasmodium*, plasmablasts carry no mutations up to day 14 (extrafollicular output) and mutated
receptors from day 21, when the output tracks the GC with a lag. The human clock's interval nearly reaches
zero, so the conversion into days is very uncertain there. Numbers from the earlier heavy-chain clone
definition (122 and 1,946 clones; 4.9 and 10.5 days) are of the same direction and size.

**Caveats.** Descriptive: it does not show that the output cells sampled at a visit descend from the GC cells
sampled at that visit; converting mutations into days assumes every clone matures at the population's average
rate; mutation count is not affinity. Not in the manuscript for these reasons.

## 2. Convergent receptors in *Plasmodium* infection

**Question.** Different mice that respond to the same parasite with the same V gene, J gene and a
near-identical CDR3 have converged on one receptor solution — the strongest evidence of antigen specificity
available without a tetramer. Do such convergent clones sit preferentially in one compartment?

**Data and method.** GSE286215 (20 mice in experiment 1, 42 in experiment 2); convergent groups are
receptors with the same V and J genes and ≥ 80 % CDR3 identity found in at least two mice; each clone is
assigned to its dominant annotated compartment (GC B cell, plasmablast or memory); enrichment against
2,000 label permutations (`pcas_arm_analyses.py`).

**Result** (`summary.json`, `*_convergence_by_arm.csv`). Convergent receptors were enriched among
plasmablast-dominated clones in both experiments (days 0–14: 31.4 % vs 28.5 % expected; days 10–42: 40.8 %
vs 39.1 %; no permutation reached the observed value, P ≤ 0.0005) and depleted among GC-dominated clones (21.9 % vs 27.4 % and 31.7 % vs 33.3 %), with
memory-dominated clones depleted early and not different late. Lineage-tree topology was not attempted: no
clone reaches 15 cells (largest 8 and 9).

**Caveats.** Compartments are author annotations derived from RNA; convergence is evidence of shared
specificity, not of affinity. The BASELINe selection estimates (`*_baseline_input.tsv.gz` written for the
Immcantation R workflow) were not run.

## 3. Which states co-occur within a clone (*Plasmodium*)?

**Question.** In one mouse and one day, are pairs of states (GC + plasmablast, GC + memory, memory +
plasmablast) found together in the same clone more often than expected from how many cells of each state the
mouse had?

**Method.** For every mouse, clones with ≥ 2 cells; cell-state labels permuted within the mouse (1,999
permutations; also within isotype strata), which keeps each mouse's state totals and clone sizes; strict
exact-receptor controls (`clone_state_sharing.py`).

**Result** (`primary_joint_sharing.csv`, `pair_summary.csv`). From day 10 onwards (experiment 2, 40 mice,
1,015 clones), clones containing both GC B cells and plasmablasts were more frequent than expected (6.6 %
observed vs 3.9 % under permutation; 111 clones), whereas in days 0–14 (19 mice) they were less frequent
(2.7 % vs 4.9 %). GC + memory and memory + plasmablast pairs were rare and not enriched.

**Interpretation.** Consistent with plasmablasts coming increasingly from GC-experienced clones as the
infection proceeds (see 4). No direction, ancestry or GC re-entry can be inferred from one sampling per mouse.

## 4. Clonal structure through the *Plasmodium* infection

**Result** (`timecourse.csv`; clones with ≥ 2 cells per day). The share of transcriptional variance explained
by clone identity was 0.78 on day 4 and fell to 0.28 by day 42 (shuffled values 0.02–0.05; P = 0.005 at every
day), while the spread of cells within a clone roughly doubled (0.57 → 1.11 relative to unrelated cells) and
the share of clones containing two states peaked at days 14–21 (≈ 0.32). The median clone carried no somatic
mutation up to day 14 and 3–7 mutations from day 21. Early clones are homogeneous (extrafollicular
responses); later clones diversify across states as GC-derived cells appear.

## 5. Selection on the antibody sequence

**Question.** Is a clone's antibody under selection, measured from the receptor sequences alone, and does
selection relate to measured annotations?

**Method.** For each clone with enough mutations, the observed share of replacement mutations against the
share expected by randomly mutating the clone's own germline (`sequence_selection.py`); Spearman correlation
with clone-level gate fractions.

**Result.** Replacement share observed vs expected: NP–OVA 0.789 vs 0.775 (1,281 clones), RBD 0.781 vs 0.773
(3,623), GSE246382 0.798 vs 0.773 (312) — slight positive selection in the model-antigen GCs; *Plasmodium*
0.770 vs 0.788 (276) and 0.740 vs 0.784 (7,727). Correlations with the division, bait, zone and plasma-cell
gates were weak (|ρ| ≤ 0.09; none with P < 0.05). Selection on the sequence and the transcriptional profile
carry largely different information. The script is not configured for the gut and spontaneous-GC datasets,
and the human lymph-node deposit carries no V-region alignment (`sequence_alignment`, `germline_alignment`),
so selection cannot be scored there.

## 6. Spike binding and somatic mutation in the human lymph node

**Question.** What does the public spike-binding label of GSE195673 measure, and how does it relate to
mutation over time?

**Result** (`spike_binding_audit.py`, `spike_gc_trends.py`). The label (`s_pos_clone`) is a clone-level call
from probe sorting and recombinant-antibody ELISA, not a per-cell affinity; a "negative" is not a uniformly
assayed negative. Among GC cells, mutation rose from 1.5 % (spike-binding families) and 0.9 % (others) at day
28 to 5.3 % and 4.6 % at day 201 (donor-equal means), with spike-binding families slightly more mutated at
every visit. Descriptive only; the spike label is therefore used in the manuscript only as an annotation.

## 7. Model-antigen germinal centres with library controls

**Question.** In GSE287123, are the division, bait and zone associations library artefacts? Profiles were
re-estimated relative to each mouse × gate library and the associations re-tested; within-library
correlations; W33L and RBD-bait checks (`mouse_gc_deep_dive.py`).

**Result.** (`summary.json` and tables in `../case_studies/results/mouse_gc_deep_dive/`)

| Test | Clones | R² vs permutation mean | P |
|---|---|---|---|
| NP–OVA: division, profiles centred within each mouse × gate library | 471 | 0.073 vs 0.002 | 0.0002 |
| RBD protein: division, centred within library | 1,544 | 0.046 vs 0.001 | 0.0002 |
| RBD protein: RBD-bait binding, centred within library | 1,544 | 0.004 vs 0.001 | 0.0008 |
| RBD mRNA: division, centred within library | 343 | 0.016 vs 0.003 | 0.0002 |
| RBD mRNA: dark-zone gate, centred within library | 343 | 0.007 vs 0.003 | 0.04 |

Centring within each library removes any shift of a library's average; recent division keeps a clear
association in all three cohorts (smaller than with mouse centring, 0.088 overall), whereas the bait and
zone associations almost disappear — consistent with the main text, where only division is cleared of a
library explanation by the crossed design. Further checks: within clones, RBD binding and recent division are
anticorrelated (ρ = −0.30, 397 clones), but the sorting design alone produces ρ ≈ −0.22 (P vs design null
0.052); a ribosomal/translation score tracks division across clones (ρ = 0.27) but not within either gate
(ρ 0.01 and 0.08), so it reflects gate composition; NP clones carrying the high-affinity W33L mutation had
a higher recently divided fraction than germline clones (median 1.0 vs 0.91; 29 vs 29 clones; P = 0.078),
too few clones for a conclusion. Caveat: centring within library removes a library's mean but not effects
that differ between cell types, so it complements rather than replaces the crossed design of Fig. 4e.

## 8. Somatic-hypermutation lineage trees

**Question.** Inside one clone, do cells that sit close together on the clone's mutation lineage tree also
share their transcriptional state, and do sorted compartments differ in tree depth?

**Method.** GCtree trees (PHYLIP `dnapars`, 15-min limit per clone) for clones with ≥ 3 cells; cells placed on
their trees; within-clone comparisons of depth between compartments; within-clone Mantel correlation between
mutation and expression distances (`lineage_trees.py`, `run_gctree.sh`, `lineage_vs_state.py`). The pooled
phylogenetic mixed model (`threadfin.tl.lineage_heritability`, section 10 of [Methods](../../docs/METHODS.md))
found nothing beyond identical genotypes in earlier analyses (negative-results record, defects table).

**Result.** (trees in `../case_studies/results/<dataset>/lineage/`; `tree_status.csv`,
`cell_tree_positions.csv`, `within_clone_tree_comparisons.csv`)

Trees were built for 30 of 31 GSE246382 clones, 220 of 234 NP–OVA clones and 740 of 892 RBD clones (the rest
had no sequence difference or exceeded the time limit).

* **Relatives do not share state beyond the clone.** The within-clone correlation between mutation distance
  and expression distance had a median of −0.22 (GSE246382, 28 clones), −0.03 (NP–OVA, 224) and 0.00 (RBD,
  834), with combined P ≥ 0.99 in each: within a clone, how closely two cells are related on the mutation tree
  says nothing detectable about how similar their states are. This matches the earlier finding that the state
  is shared within a clone but cell-cycle position and zone are re-decided faster than one mutation
  accumulates.
* **Plasma cells sit nearer the germline than GC cells of the same clone** (GSE246382: depth −1.6 mutations,
  14 clones, Wilcoxon P = 0.011), consistent with the export lag of section 1.
* **Recently divided cells and tree depth:** slightly shallower within NP–OVA clones (−0.33 mutations, 118
  clones, P = 0.053) but deeper within RBD clones (+0.46, 467 clones, P = 0.016); the direction differs
  between the two systems and is not interpreted.

`threadfin.tl.lineage_heritability` (the pooled phylogenetic mixed model) gave the same picture in earlier
analyses: nothing beyond identical genotypes ([negative results](negative_and_withdrawn_results.md)).
Placing cells on trees needs `ete3` (installed with GCtree).

## 9. Clonal memory across time points, tissues and gates

For every clone seen in two time points, tissues or sort gates, the change of its profile is compared with
the distance to a random clone of the same donor (memory index 1: keeps everything; 0: no more similar to
itself than to a random clone). Selected results (`cross_dataset/clonal_memory.csv`):

| Dataset | Axis | Clones | Memory index [95 % CI] |
|---|---|---|---|
| influenza vaccination | day 0 → day 7 | 727 | 0.89 [0.80, 0.98] |
| RBD | RBD-bait gates | 484 | 0.84 [0.68, 1.01] |
| RBD | division gates | 798 | 0.61 [0.48, 0.75] |
| human lymph node | visits (days 28–201) | 1,736 | 0.55 [0.47, 0.64] |
| bone marrow | blood ↔ bone marrow | 1,843 | 0.26 [0.22, 0.29] |
| gut | GC ↔ nonGC sort | 126 | 0.22 [0.07, 0.37] |
| RBD | zone gates | 192 | 0.21 [0.03, 0.42] |
| EBV organoids | days 0–21 | 4,430 | 0.17 [0.15, 0.19] |
| gut | tissues | 311 | 0.10 [0.05, 0.14] |
| human lymph node | lymph node ↔ blood | 554 | −0.02 [−0.05, 0.02] |

Clones keep their state across vaccine visits and between sort gates that do not change the cell type
(bait, division), but not between compartments that do (lymph node vs blood, where the same clone is a GC B
cell or plasma cell in one and a circulating cell in the other).

## 10. Clonally inherited genes

The clonal ICC of every gene and of marker gene sets, compared with genes of matched expression level
(`gene_heritability.csv`, `geneset_heritability.csv` per dataset; all datasets in
`cross_dataset/geneset_heritability.csv`). In the gut cohort, for example, plasma-cell genes (median excess
ICC 0.099 vs 0.033 for matched genes; FDR 0.0009), GC genes (0.084 vs 0.018) and dark-zone/cycling genes
(0.076 vs 0.021) are inherited within clones above background, light-zone genes are not (FDR 0.16).

## 11. Every clone-level association test

[`cross_dataset/label_effects.csv`](cross_dataset/label_effects.csv) lists all 58 tests of annotations against
clone profiles across 14 datasets (R², permutation mean, excess R², P, number of clones, permutation
strata), including annotations derived from RNA (marked in the `note` column), which are not independent of
the profiles.

## 12. Additional datasets

Six further datasets went through the same pipeline (results in `../case_studies/results/<dataset>/`;
statistics in [`../dataset_overview.md`](../dataset_overview.md)): antigen-bait-sorted B cells after SARS-CoV-2
infection (592 clones; ICC 0.40), influenza vaccination (heavy chain only; 2,080 clones; ICC 0.56, not
comparable with paired datasets), influenza infection with haemagglutinin tetramers (201 clones; ICC 0.15;
a clone's cells in different haemagglutinin-binding categories have similar states, memory index 0.52), paediatric tonsil (361 clones;
ICC 0.26), EBV-infected tonsil organoids (6,916 clones; ICC 0.22; donors pooled) and human bone-marrow plasma
cells with blood (6,788 clones; ICC 0.40, with a large shuffled value of 0.23 because sorted compartments
differ in composition). The SARS-CoV-2, tonsil, EBV and bone-marrow results were re-run on 2026-10-20 after
Dandelion's light-chain-only groups were removed from their clones (they had made up 1,689 of 2,281, 419 of
780, 1,198 of 8,114 and 67 of 6,855 groups of two or more cells; see the
[record of defects](negative_and_withdrawn_results.md)); coherence rose in the SARS-CoV-2 and tonsil data once
these pseudo-clones were gone. They are not in the manuscript because they add breadth rather than new biology, and
three of them have design limitations (heavy chain only, pooled donors, sorting by compartment).
