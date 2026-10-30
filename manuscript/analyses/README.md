# Analyses beyond Figures 1–4

Every analysis made for the manuscript that is not shown in the main figures. Each entry gives the question,
the data, what was done, what was found, the caveats and where the scripts and results are. Analyses whose
outputs had been removed in the 2026-10-13 clean-up (they used the earlier heavy-chain-only clone
definition) were re-run on the current paired-chain clones on 2026-10-20. The comparison with other tools is
in [`../benchmarking/`](../benchmarking/README.md); what did not hold is in
[`negative_and_withdrawn_results.md`](negative_and_withdrawn_results.md).

| # | Analysis | Data | One-line result | Location |
|---|---|---|---|---|
| 1 | [Export lag](#1-export-lag-how-far-behind-its-germinal-centre-is-a-clones-output) | *Plasmodium*, human lymph node | the median within-clone mutation gap is negative, although output is more mutated in 8 of 92 mouse and 821 of 2,564 human clones | [`export_lag/`](export_lag) |
| 2 | [Convergent receptors and the compartments of a clone](#2-convergent-receptors-in-plasmodium-infection) | *Plasmodium* | receptors shared between mice are enriched in plasmablast-dominated clones | [`../case_studies/results/pcas_arm_analyses/`](../case_studies/results/pcas_arm_analyses) |
| 3 | [States that co-occur within clones](#3-which-states-co-occur-within-a-clone-plasmodium) | *Plasmodium* | GC + plasmablast clones are enriched from day 10 on, depleted before | [`../case_studies/results/clone_state_sharing/`](../case_studies/results/clone_state_sharing) |
| 4 | [Clonal structure through infection](#4-clonal-structure-through-the-plasmodium-infection) | *Plasmodium* | the share of state explained by clone falls from 0.78 to 0.28 while clones diversify | [`../case_studies/results/malaria/timecourse.csv`](../case_studies/results/malaria/timecourse.csv) |
| 5 | [Selection on the antibody sequence](#5-selection-on-the-antibody-sequence) | 5 datasets | replacement mutations slightly above chance in model-antigen GCs; weakly related to sort gates | `../case_studies/results/<dataset>/sequence_selection.csv` |
| 6 | [Spike binding and mutation in the vaccine lymph node](#6-spike-binding-and-somatic-mutation-in-the-human-lymph-node) | human lymph node | spike-binding GC families are slightly more mutated at every visit; the label is clone-level, not affinity | [`../case_studies/results/spike_binding_audit/`](../case_studies/results/spike_binding_audit) |
| 7 | [Division, binding and zone with library controls](#7-model-antigen-germinal-centres-with-library-controls) | GSE287123 | with profiles centred within each mouse × gate library, division still explains clone profiles; bait and zone almost vanish | `../case_studies/results/mouse_gc_deep_dive/` |
| 8 | [Lineage trees](#8-somatic-hypermutation-lineage-trees) | GSE246382, NP–OVA, RBD | close relatives on the tree do not share state beyond the clone; plasma cells sit nearer the germline than GC cells of their clone | `../case_studies/results/<dataset>/lineage/` |
| 9 | [Clonal memory across time, tissue and gates](#9-clonal-memory-across-time-points-tissues-and-gates) | 11 datasets | clones keep their state across vaccine visits and between RBD-bait gates; with Threadfin 4.6.0 the index between lymph node and blood is small but above zero (0.05 [0.02, 0.07]) and the gut-tissue index falls to 0.03 [−0.02, 0.09] | [`cross_dataset/clonal_memory_460.csv`](cross_dataset/clonal_memory_460.csv) (4.6.0), [`cross_dataset/clonal_memory.csv`](cross_dataset/clonal_memory.csv) |
| 10 | [Genes shared within clones](#10-genes-shared-within-clones) | 14 datasets | gene sets compared with expression-matched genes; with the 4.6.0 test, 57 of 95 sets are shared within clones more than matched genes (67 with the earlier test) | [`cross_dataset/geneset_heritability_460.csv`](cross_dataset/geneset_heritability_460.csv) (4.6.0), [`cross_dataset/geneset_heritability.csv`](cross_dataset/geneset_heritability.csv) |
| 11 | [All clone-level association tests](#11-every-clone-level-association-test) | 14 datasets | 58 annotation tests in one table; with Threadfin 4.6.0 (shuffles within donor × clone-size bins) one changes its call (GSE246382 dark-zone gate, P 0.03 → 0.17) | [`cross_dataset/label_effects_460.csv`](cross_dataset/label_effects_460.csv) (4.6.0), [`cross_dataset/label_effects.csv`](cross_dataset/label_effects.csv) |
| 12 | [Additional datasets](#12-additional-datasets) | 6 datasets | the pipeline on influenza, tonsil, EBV, bone marrow and bait-sorted cells (all six re-run on 2026-10-23 with threadfin 4.4 clone calls) | `../case_studies/results/<dataset>/` |
| 13 | [Gut: all B cells and centring](gut_all_B_and_centring/README.md) | GSE284622 | the second plasma branch is one mouse; the GC → plasma path survives every centring | [`gut_all_B_and_centring/`](gut_all_B_and_centring) |
| 14 | [Lineage-barcode and TCR data](lineage_barcode_systems/README.md) | LARRY, WATERMELON, NSCLC T cells | day-2 profiles predict day-6 fate (kernel vs centroid not significantly different); T-cell clones keep their state across therapy cycles | [`lineage_barcode_systems/`](lineage_barcode_systems) |
| 15 | [Heavy-chain vs paired-chain clones](spont_gc_paired_chains/README.md) | GSE203132 | adding the light chain splits 481 of 1,051 clones, but Figure 3d–f is unchanged (entropy vs clone axis ρ −0.747 → −0.754; 8 of 8 mice) | [`spont_gc_paired_chains/`](spont_gc_paired_chains) |
| 16 | [Incomplete chain pairs and reused contig ids (RBD)](rbd_complete_pairs/README.md) | GSE287123 | Figure 4 is unchanged with complete heavy + light pairs only (division excess R² 0.088 → 0.096; Threadfin 4.6.0: 0.087 → 0.095) and with corrected contig ids (0.085, 4.6.0: 0.085; cosine 0.896; 10 of 10 mice) | [`rbd_complete_pairs/`](rbd_complete_pairs) |
| 17 | [Technical validation](technical_validation/README.md) | all | the tutorial path on 16 datasets without an error; clone definition and embedding sensitivity; power of the coherence test; one million cells; T cells; automatic clone-map settings (30/30 maps coherent); state-density model; runtimes; audits | [`technical_validation/`](technical_validation) |
| 18 | [Negative results and corrected defects](negative_and_withdrawn_results.md) | — | three falsified predictions, two withdrawn headlines, eighteen fixed defects | — |

---

## 1. Export lag: how far behind its germinal centre is a clone's output?

**Question.** Is the median within-clone mutation gap between output cells and their own GC cells negative,
and how large is it? The gap can be positive in some clones.

**Data and method.** Class-switched cells of clones captured as both GC B cells and output cells
(ASCs) at the same visit (and tissue). For each clone, `threadfin.tl.export_lag` computes the mean mutation
frequency of its output cells minus that of its
GC cells. The estimate is the median gap across clones, with a sign-test interval and a sign test of a zero
median. The GC compartment's own maturation rate (a Theil–Sen fit on per-visit medians of switched GC cells)
converts the gap into days. Sen's interval estimates the rate, and the 97.5 % intervals of the gap and rate
give a days interval with at least 95 % coverage. Donor medians have their own sign-test intervals.
Script: [`export_lag/export_lag_analysis.py`](export_lag/export_lag_analysis.py).

**Result.** (Threadfin 4.6.0: `export_lag/results/summary_460.json`, per-donor tables
`export_lag/results/*_per_donor_460.csv`, re-derived on 2026-10-30 from the same clone pairs; the Threadfin 4.3
run with bootstrap intervals is `export_lag/results/summary.json`; figure `export_lag/figures/export_lag.png`)

The intervals below are 95 % intervals; pp means percentage points.

| Dataset | Median gap (pp) | Sign test P | Clock (%/day) | Days | Donors |
|---|---|---|---|---|---|
| *Plasmodium* d10–42 | −0.38 [−0.74, −0.37] | 3e-12 | 0.101 [0.079, 0.135] | −3.7 [−9.3, −2.8] | 13 of 16 mice negative; across donors −0.72 [−1.49, −0.28] |
| Human lymph node | −0.19 [−0.19, −0.16] | 1e-49 | 0.023 [0.002, 0.030] | −8.1 [−84, −5.4] | 8 of 8 participants negative; across donors −0.18 [−0.28, −0.06] |

The 92 *Plasmodium* clones include 65 with less mutated output, 8 with more mutated output and 19 ties.
The corresponding counts among 2,564 human lymph-node clones are 1,536, 821 and 207. The human interval in
days is wide because the d28→d35 segment of the clock is almost flat. In *Plasmodium* at days 0–14, nine
clones had a median gap of 0.00 pp [−1.12, 0.00] (4 less mutated, 1 more mutated, 4 tied; sign test P = 0.38;
3 of 7 mice negative), and the clock was not estimable (four time points; Sen's interval needs five). The
Threadfin 4.3 bootstrap intervals of the median gap were [−0.73, −0.37] and [−0.19, −0.17] pp; the medians,
counts, clocks and days are the same in both versions.
Numbers from the earlier heavy-chain clone definition (122 and 1,946 clones; 4.9 and 10.5 days) are of the
same direction and size.

A negative gap has four explanations that these data cannot separate:

* output that left the germinal centre before it accumulated further mutations;
* extrafollicular ASCs of the same clone;
* affinity-biased export of plasma cells;
* regulated hypermutation, in which high-affinity cells divide more often with fewer mutations per division.

**Caveats.** The gap does not establish a parent–offspring relation between particular cells. Converting
mutations into days assumes every clone mutates at the population's average rate and that the rate is
constant per day. Regulated hypermutation does not support that assumption. Mutation count is not affinity.
The analysis is not in the manuscript for these reasons.

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

**Result.** (`summary.json` and tables in `../case_studies/results/mouse_gc_deep_dive/`; Threadfin 4.6.0:
`summary_460.json`, re-derived on 2026-10-30 with the clones and embedding of the original run)

| Test | Clones | R² vs permutation mean | P | Threadfin 4.6.0: permutation mean; P |
|---|---|---|---|---|
| NP–OVA: division, profiles centred within each mouse × gate library | 471 | 0.073 vs 0.002 | 0.0002 | 0.002; 0.0002 |
| RBD protein: division, centred within library | 1,544 | 0.046 vs 0.001 | 0.0002 | 0.001; 0.0002 |
| RBD protein: RBD-bait binding, centred within library | 1,544 | 0.004 vs 0.001 | 0.0008 | 0.001; 0.005 |
| RBD mRNA: division, centred within library | 343 | 0.016 vs 0.003 | 0.0002 | 0.003; 0.0008 |
| RBD mRNA: dark-zone gate, centred within library | 343 | 0.007 vs 0.003 | 0.04 | 0.003; 0.03 |

Threadfin 4.6.0 shuffles each label among clones of the same mouse and similar size (5,000 permutations, as
here) and flags all five labels as fixed within each sample (each gate is one library of one mouse; 14 and 20
samples): every clone-level P value then compares samples, not independent clones. The conclusion is the same:
division keeps a clear association in all three cohorts; bait and zone explain 0.4 % and 0.7 % of the variation
between clone profiles.

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

For every clone seen in two time points, tissues or sort gates, the change of its profile between consecutive
levels is compared with the change to other clones of the same donor (memory index 1: keeps everything; 0: no
more similar to itself than to another clone). Threadfin 4.6.0 changed the index: a clone is now compared with
other clones of its donor seen at the same two levels (Threadfin 4.5 compared it with any clone of the later
level, so clones seen at two levels, a selected group, could read as having memory), and sampling noise is
removed with each snapshot's own variance (a one-cell snapshot borrows the mean variance of the multi-cell
snapshots of its level). The index is not bounded by 0 and 1.

Results with Threadfin 4.6.0 (re-derived on 2026-10-30 with the clones, profiles and settings of the original
runs; levels paired in the order of the original runs, e.g. LP → MLN → PP for the gut tissues;
`cross_dataset/clonal_memory_460.csv`, `../case_studies/results/<dataset>/memory_pairs_<axis>_460.csv`) next to
the Threadfin 4.2–4.4 results (`cross_dataset/clonal_memory.csv`). P values: 500 permutations (floor 0.002).

| Dataset | Axis | Clones | Memory index [95 % CI], 4.2–4.4 | Memory index [95 % CI]; P, 4.6.0 | 4.6.0, snapshots of ≥ 2 cells only (clones) |
|---|---|---|---|---|---|
| influenza vaccination | day 0 → day 7 | 726 | 0.89 [0.81, 0.98] | 0.82 [0.67, 1.05]; 0.002 | 0.66 [0.47, 0.81] (64) |
| RBD | RBD-bait gates | 484 | 0.84 [0.68, 1.01] | 0.78 [0.47, 1.18]; 0.002 | 0.43 [0.17, 0.72] (153) |
| RBD | division gates | 798 | 0.61 [0.48, 0.75] | 0.66 [0.46, 0.89]; 0.002 | 0.49 [0.33, 0.68] (272) |
| human lymph node | visits (days 28–201) | 1,736 (4.6.0: 1,733) | 0.55 [0.47, 0.64] | 0.71 [0.62, 0.81]; 0.002 | 0.23 [0.16, 0.32] (480) |
| bone marrow | blood ↔ bone marrow | 1,604 | 0.29 [0.26, 0.33] | 0.18 [0.13, 0.23]; 0.002 | 0.23 [0.09, 0.39] (111) |
| EBV organoids | days 0–21 | 730 | 0.28 [0.23, 0.33] | 0.25 [0.17, 0.34]; 0.002 | 0.25 [0.11, 0.38] (107) |
| gut | GC ↔ nonGC sort | 126 | 0.22 [0.07, 0.37] | 0.19 [0.01, 0.39]; 0.008 | 0.33 [0.04, 0.58] (34) |
| RBD | zone gates | 192 | 0.21 [0.03, 0.42] | 0.26 [−0.03, 0.56]; 0.026 | 0.45 [0.00, 0.77] (50) |
| gut | tissues | 311 | 0.10 [0.05, 0.14] | 0.03 [−0.02, 0.09]; 0.024 | 0.01 [−0.06, 0.09] (103; P = 0.33) |
| human lymph node | lymph node ↔ blood | 554 (4.6.0: 553) | −0.02 [−0.05, 0.02] | 0.05 [0.02, 0.07]; 0.002 | 0.02 [−0.01, 0.05] (155) |

With Threadfin 4.6.0, clones keep most of what distinguishes them across influenza vaccination (0.82), across
the visits after mRNA vaccination (0.71) and between sort gates that do not change the cell type (bait 0.78,
division 0.66); the index is low between compartments that do (bone marrow vs blood 0.18, GC-fate sort 0.19).
Two results change their reading. Between the gut tissues the index falls from 0.10 to 0.03, and its interval
now includes zero (the permutation P is 0.024). Between lymph node and blood it rises from −0.02 (P = 0.79) to
0.05 [0.02, 0.07] (P = 0.002): small, but no longer "no memory"; with snapshots of two or more cells it is
0.02 [−0.01, 0.05]. The intervals are wider than before, because each snapshot's noise is estimated. The
human-visit index depends on the variance assumed for one-cell snapshots: restricted to snapshots of two or more
cells (480 of 1,733 clones) it is 0.23 [0.16, 0.32]. With the 4.5 options (`reference="all", noise="pooled"`)
every index of the first table is reproduced exactly.

## 10. Genes shared within clones

The clonal ICC of every gene and of marker gene sets, compared with genes of matched expression level
(Threadfin 4.6.0, re-derived on 2026-10-30: `gene_heritability_460.csv`, `geneset_heritability_460.csv` per
dataset; all 14 datasets in `cross_dataset/geneset_heritability_460.csv`. The Threadfin 4.2–4.4 tables without
the suffix used the earlier test). In the gut cohort, for example, plasma-cell genes (median excess
ICC 0.099 vs 0.034 for matched genes; FDR 0.008), GC genes (0.084 vs 0.020; FDR 0.0005) and dark-zone/cycling
genes (0.076 vs 0.021; FDR 0.0015) are shared within clones more than matched genes; light-zone genes are not
(FDR 0.15).

The gene-set test was replaced in Threadfin 4.6.0 by one that allows for co-expressed genes: a competitive test
on Fisher's z of the clonal ICC whose error combines the spread between genes with a clone bootstrap. The
earlier Mann–Whitney test treated the genes of a set as independent; its P value is kept in the column
`pvalue_mw_genes_independent` and reproduces the earlier tables exactly. With the 4.6.0 test, 57 of the 95 sets
of the 14 datasets are shared within clones more than matched genes at FDR < 0.05, against 67 with the earlier
test: 13 calls are lost (the plasma-cell set of the NP–OVA division-reporter and Smart-seq2 cohorts; light-zone
genes in NP–OVA, *Plasmodium* days 10–42, influenza lung and bone marrow; interferon genes in the gut,
*Plasmodium* days 10–42 and influenza vaccination; dark-zone/cycling genes in EBV organoids and bone marrow;
memory genes in COVID-19 and bone marrow) and 3 are gained (naive genes in RBD, light-zone genes in the human
lymph node, dark-zone/cycling genes in spontaneous GCs). In the gut, the FDRs with the earlier test were 0.0009
(plasma cell), 0.0002 (GC), 0.0018 (dark zone/cycling) and 0.16 (light zone); the interferon set moved from
FDR 0.0499 to 0.078.

## 11. Every clone-level association test

[`cross_dataset/label_effects.csv`](cross_dataset/label_effects.csv) lists all 58 tests of annotations against
clone profiles across 14 datasets (R², permutation mean, excess R², P, number of clones, permutation
strata), including annotations derived from RNA (marked in the `note` column), which are not independent of
the profiles.

Threadfin 4.6.0 shuffles each annotation among clones of the same donor and of similar size (2, 3–4, 5–9 and
≥ 10 cells), because small clones' profiles are pulled towards the average profile, so an annotation that
follows clone size alone could explain profile variance. The 58 tests re-derived with 4.6.0 on 2026-10-30 are in
[`cross_dataset/label_effects_460.csv`](cross_dataset/label_effects_460.csv) (same clones and profiles; R² is
unchanged, the permutation mean and P change). 52 of 58 tests are significant at P < 0.05, against 53 before:
the dark-zone gate of the Smart-seq2 NP–OVA cohort (GSE246382) loses significance (excess R² 0.023 → 0.014;
P = 0.03 → 0.17). Excess R² changed by at most 0.021 (GSE246382 plasma-cell gate, 0.750 → 0.728); other P
values that moved are the NP–OVA isotype (0.007 → 0.024), NP–OVA mutation frequency (0.49 → 0.11, not
significant either way), RBD mutation frequency (0.0005 → 0.0015), GSE246382 W33L (0.019 → 0.009) and
influenza-lung isotype (0.005 → 0.001). With `size_strata=False` the earlier permutation means (to within 3 × 10⁻⁴) and P values are
reproduced.

## 12. Additional datasets

Six further datasets went through the same pipeline (results in `../case_studies/results/<dataset>/`;
statistics in [`../dataset_overview.md`](../dataset_overview.md)): antigen-bait-sorted B cells after SARS-CoV-2
infection (588 clones; ICC 0.40), influenza vaccination (heavy chain only; 2,082 clones; ICC 0.56, not
comparable with paired datasets), influenza infection with haemagglutinin tetramers (204 clones; ICC 0.16;
a clone's cells in different haemagglutinin-binding categories have similar states, memory index 0.63; with
Threadfin 4.6.0, 0.62 [0.16, 1.31], P = 0.002, 145 clones), paediatric
tonsil (361 clones; ICC 0.25), EBV-infected tonsil organoids (6,124 clones; ICC 0.16; donors pooled) and human
bone-marrow plasma cells with blood (6,583 clones; ICC 0.43, with a large shuffled value of 0.25 because sorted
compartments differ in composition). All six were re-run on 2026-10-23 with clones called by threadfin 4.4,
which leaves light-chain-only groups without a clone (they had made up 1,689 of 2,281, 419 of 780, 1,198 of
8,114 and 67 of 6,855 groups of two or more cells in the SARS-CoV-2, tonsil, EBV and bone-marrow data) and
renames contig ids that repeat across libraries. The second change mattered most for the EBV organoids, whose
libraries (one per time point) reuse barcodes: clones seen at two time points fell from 4,430 to 730 and the ICC
from 0.22 to 0.16 ([record of defects](negative_and_withdrawn_results.md)). They are not in the manuscript
because they add breadth rather than new biology, and
three of them have design limitations (heavy chain only, pooled donors, sorting by compartment).
