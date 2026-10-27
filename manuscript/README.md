# The Threadfin manuscript: status, data, analyses and open work

This folder holds everything behind the manuscript *"Threadfin infers relationships between B-cell clones
from paired single-cell RNA and receptor sequencing"*: the figures and the code that draws them, the
case-study analyses of every public dataset, the analyses that did not make it into the figures, a record of
negative results, and an end-to-end benchmark against related tools. This page explains where the
manuscript stands, what was tested on which data, what was found, and what is still worth doing.

**Contents**

1. [At a glance](#1-at-a-glance)
2. [Publication status](#2-publication-status-2026-10-20)
3. [What the manuscript claims, and the evidence for each claim](#3-what-the-manuscript-claims-and-the-evidence-for-each-claim)
4. [Datasets tested](#4-datasets-tested)
5. [Analyses](#5-analyses)
6. [Benchmarking against related tools](#6-benchmarking-against-related-tools)
7. [Methods and statistical conventions in brief](#7-methods-and-statistical-conventions-in-brief)
8. [Reproducing the analyses](#8-reproducing-the-analyses)
9. [What is still worth improving](#9-what-is-still-worth-improving)
10. [History of the manuscript](#10-history-of-the-manuscript)
11. [Layout of this folder](#11-layout-of-this-folder)

---

## 1. At a glance

| | |
|---|---|
| **Title** | Threadfin infers relationships between B-cell clones from paired single-cell RNA and receptor sequencing |
| **Authors** | Chen Zhu (Peter Doherty Institute, University of Melbourne), Yang Yang (Frazer Institute, University of Queensland; corresponding), Zewen Kelvin Tuong (Ian Frazer Centre for Children's Immunotherapy Research, University of Queensland; corresponding), Zhian Chen (affiliation to be provided) |
| **Article type and target** | Article, prepared for *Nature Methods* (format, word limits and reporting standards of that journal); *Cell Systems* and *Genome Biology* were assessed as alternatives |
| **Current version** | preprint draft of 2026-10-17, with the new Figure 4 added on 2026-10-19; package version used for the analyses 4.2.0, current package 4.5.0 |
| **Main figures** | 4 (concept; application to germinal-centre data; clone states; model-antigen germinal centres) |
| **Supplementary material** | 6 supplementary figures (Supplementary Fig. 1 in two sheets), 2 supplementary tables |
| **Datasets** | 9 paired scRNA-seq + scBCR-seq datasets in the manuscript, 6 further B-cell datasets analysed, 3 non-B-cell lineage datasets |
| **Where the text is** | kept private until submission (not in this repository); figures, legends' content and all code are here |

**The idea in one paragraph.** The cells of one B-cell clone share a receptor, which is inherited, but not a
transcriptional state, which changes. The states observed together within a clone are therefore a property
of the clone, not of any of its cells. Threadfin represents every clone by the distribution of its cells'
states — measured relative to the cells sampled with it, as a kernel mean embedding, shrunk according to the
number of cells sampled — and compares clones in a clone embedding and with permutation tests that use the
clone as the unit and respect the sampling design.

**The main findings** (each in Figures 2–4):

1. In eight of nine datasets clone identity explains a significant share of transcriptional state
   (intraclass correlation 0.12–0.74, against 0.006–0.28 for clone labels shuffled within samples).
2. Clone-level structure is detectable when most clones are captured as two cells (*Plasmodium* infection:
   830 of 1,015 profiled clones have two cells; ICC 0.45).
3. Plasmablasts that look alike belong to clones that differ in whether they also contain GC B cells — a
   clone property invisible on the cell-level embedding.
4. In spontaneous germinal centres, clones are ordered from single-state antibody-secreting clones to
   multi-state clones, and memory B cells occur mostly in the multi-state clones.
5. In gut-associated lymphoid tissues, the clone embedding resolves a continuous path from GC clones through
   clones spanning inductive sites and the lamina propria to IgA plasma-cell clones; clones whose plasma cells
   lacked GC-fate labelling group at the end of the path.
6. In RBD-immunised division-reporter mice, the main axis of clone profiles follows recent division in two
   cohorts whose sequencing libraries are crossed, which excludes a library artefact for division.

---

## 2. Publication status (2026-10-20)

### 2.1 The text

The draft is written in Markdown and built into two Word documents (main text; figures, legends and
supplementary tables) with Pandoc and a small build script; it is kept in `manuscript/preprint_2026-10-17/`,
which is excluded from the repository until submission.

| Part | State | Length |
|---|---|---|
| Title, authors, affiliations | complete except Zhian Chen's affiliation | — |
| Abstract | complete | 146 words (limit 150) |
| Main (introduction) | complete | part of main text |
| Results | complete; seven sections with claim-shaped headings | part of main text |
| Discussion | complete; one sentence must be updated (see 2.3) | part of main text |
| Main text total | — | 4,788 words (journal guideline about 3,000, maximum 5,000) |
| Methods | complete, including dataset-specific sections | 4,123 words |
| Figure legends | complete for all 4 main and 6 supplementary figures | 119–395 words each |
| Supplementary Tables 1–2 | complete (datasets and settings; coherence and association statistics) | — |
| References | 55, formatted | — |
| Data availability | accessions listed; Zenodo DOI for processed data still to be added | — |
| Code availability | repository given; Zenodo DOI for the archived code still to be added | — |
| Use of large language models | statement still to be written by the authors | — |
| Acknowledgements, author contributions, competing interests | to be completed | — |
| Reporting summary | to be completed at submission | — |

The seven Results headings:

1. Clone profiles represent which transcriptional states occur together within a clone
2. Clone identity explains transcriptional state across immunization, infection and vaccination
3. Clone-level structure is detectable when most clones are sampled as two cells
4. Plasmablasts in one transcriptional state belong to clones that differ in the states they contain
5. State diversity decreases towards antibody-secreting clones, and memory B cells occur in the most diverse clones
6. Clones spanning inductive sites and the lamina propria connect germinal-centre B cells with intestinal IgA plasma cells
7. Division, antigen binding and zone vary continuously between clones in model-antigen germinal centres

### 2.2 The figures

Final for the preprint: [Figures 1–4 and Supplementary Figures 1–6](figures/README.md). On 2026-10-17 the
authors fixed Figures 1–3; Figure 4 was rebuilt on 2026-10-19 around the RBD division-reporter cohorts (the
earlier Figure 4, which combined RBD with the non-B-cell LARRY and WATERMELON data and a method comparison,
was retired; its material is in [`analyses/lineage_barcode_systems/`](analyses/lineage_barcode_systems)).

### 2.3 Items that must be resolved before submission

| # | Item | Why | Suggested action |
|---|---|---|---|
| 1 | **Clone definition of the spontaneous-GC dataset (GSE203132)** — *resolved 2026-10-23* | The deposited AIRR table of GSE203132 holds heavy chains only (8,106 IGH rows), so Fig. 3a–f and Supplementary Fig. 3 use heavy-chain clones, while the Methods say paired chains for every dataset. The light chains were annotated from the deposited contigs with IgBLAST and the clones re-defined on heavy + light chains ([`analyses/spont_gc_paired_chains/`](analyses/spont_gc_paired_chains/README.md)): the light chain splits 481 of 1,051 clones, but the result of Fig. 3d–f is unchanged (entropy vs clone axis ρ −0.747 → −0.754; 8 of 8 mice negative; ICC 0.279 → 0.293). | Keep Figure 3 as fixed by the authors. In Methods, say that GSE203132 clones are heavy-chain clones because the deposited table has no light chains, and that clones re-defined on paired chains from the deposited contigs give the same result (numbers above; a supplementary note or one sentence). The sentence is in the local draft since 2026-10-23. |
| 2 | **"We did not compare these methods on common tasks"** (Discussion) — *replaced in the local draft and accepted by the authors 2026-10-23* | The benchmark in [`benchmarking/`](benchmarking/README.md) compares Threadfin with Clonotrace, clone2vec, Benisse, CoNGA's reduction and scirpy on common tasks. The sentence now summarises it (coverage and test calibration are where Threadfin differs; annotation recovery is similar to clone2vec and to the clone mean; clone2vec groups simulated discrete programmes better). | Optionally add the benchmark overview (its Fig. 0) as a supplementary figure. |
| 3 | **Library confound of the GSE287123 gates** — *approved by the authors 2026-10-23* | The Results paragraph that states that the bait and zone gates are fully determined by sequencing library (and that only the RBD division gate is crossed) stays in the draft. | — |
| 4 | **"Paired chains" in the RBD data** — *resolved 2026-10-23* | 797 RBD cells with a clone have no productive light chain; restricted to complete heavy + light pairs (1,772 of 1,887 profiled clones), Figure 4 is unchanged (division excess R² 0.088 → 0.096, crossover cosine 0.887 → 0.884, 10 of 10 mice). The one "light-chain-only" RBD group was an artefact of contig ids reused across libraries (fixed in threadfin 4.4.0; Figure 4 with the corrected clones: 0.085, cosine 0.896, 10 of 10 mice) — [`analyses/rbd_complete_pairs/`](analyses/rbd_complete_pairs/README.md). | One sentence in Methods: clones are Dandelion's paired-chain clones; cells missing one chain keep the clone of the other, and excluding them does not change the result (in the local draft since 2026-10-23). |
| 5 | Placeholders | affiliation (Zhian Chen), LLM statement, acknowledgements, author contributions, competing interests, Zenodo DOIs, reporting summary | Authors. |
| 6 | Package version | the analyses used 4.2.0; 4.3.0 only adds automatic clone-map settings, which change no statistic; 4.3.1 stops giving light-chain-only Dandelion groups a clone id (the manuscript's datasets were unaffected except one two-cell RBD group; four additional case studies were re-run without them), fixes the memory use of `tl.export_lag` and records the hash seed of clone calls; 4.4.0 adds `n_jobs` (same results on any number of cores), renames contig ids that repeat across libraries (Figure 4 unchanged, see 2.3-4) and is tested on Python 3.10–3.14; 4.4.1 installs on Windows and is tested on macOS and Windows; 4.5.0 adds T cells, scirpy input, gene-level summaries of the clone landscape and a Methods paragraph, and changes no published number | Keep the sentence in Methods; archive the versions on Zenodo. |
| 8 | **Clone calls depend on Python's hash seed** | Dandelion's clone assignment changes by a few clones between Python sessions unless `PYTHONHASHSEED` is fixed (GSE287123: 8,729–8,734 clones, 1,879–1,887 with ≥ 2 cells, from the same contigs). The manuscript's clone tables were computed once and are saved with the case-study results. | Deposit the clone tables on Zenodo with the processed data, and add one sentence to Methods (clone ids are provided; re-running Dandelion with `PYTHONHASHSEED=0` reproduces a table that differs by a handful of clones). |
| 7 | Word count | main text 4,844 words after the 2026-10-23 edits, within the 5,000-word maximum but above the 3,000-word guideline | Optional trimming of the Discussion. |

---

## 3. What the manuscript claims, and the evidence for each claim

| Claim | Evidence | Data | Main caveat stated in the text |
|---|---|---|---|
| Clone profiles capture which states co-occur in a clone | method; simulated example (Fig. 1) | — | proximity in the clone embedding is not ancestry |
| Clone identity explains state | ICC 0.12–0.74 vs 0.006–0.28 shuffled; P = 0.002 (500 permutations within samples) in 8 of 9 datasets; not significant in the 96-clone Smart-seq2 dataset (Supp. Table 2) | 9 datasets | the shuffled value shows how much sampling structure a naive test would call clonal |
| Two-cell clones contribute | *Plasmodium*: 830/1,015 clones have two cells, ICC 0.451 vs 0.015 (Fig. 2C) | GSE286215 | stability of P3 0.28: a continuum region, not a group |
| A clone property not visible on cells | P2 and P3 plasmablasts overlap on the cell embedding; P3 clones also contain GC cells (Fig. 2D) | GSE286215 | — |
| State diversity orders clones | entropy vs clone axis ρ = −0.742 (322 clones); memory-containing clones 35.8 % → 2.5 % across quartiles; negative in 8/8 mice; same on profile axes, after rarefaction and size adjustment (Fig. 3a–f; Supp. Fig. 3) | GSE203132 | annotations are RNA-derived; P3/P4 split is post hoc; clones are heavy-chain clones, and paired-chain clones give the same result (ρ −0.754, 8/8 mice; item 2.3-1) |
| A GC → plasma-cell clonal path in the gut | bridging clones: 131/146 contain GC B and plasma cells, 128/146 span inductive sites and LP; isotype IgG1 → IgA along the path (measured in BCR); profiles associated with IgA (R² 0.18), IgG1 (0.08), LP (0.51), GC sort (0.067) (Fig. 3g–j; Supp. Fig. 4) | GSE284622 | a static arrangement, not a measured trajectory; the second plasma branch is one mouse (Supp. Fig. 5) |
| Division history is written into clone profiles | excess R² 0.088 (1,887 clones); per cohort 0.092 and 0.069; cosine of division directions 0.89 vs −1 for a library effect; ρ > 0 in 10/10 mice (Fig. 4) | GSE287123 | bait and zone gates are library-confounded; bait is a gate, not affinity |

Full statistics: Supplementary Table 2 of the manuscript and the `summary.json` / `label_effects.csv` files
of each dataset in [`case_studies/results/`](case_studies/results).

---

## 4. Datasets tested

Every dataset analysed for the manuscript, built from the saved results by
[`dataset_overview.py`](dataset_overview.py) ([CSV](dataset_overview.csv)):

| Dataset | Study | Accession | Organism | System | Cells | Cells with BCR | Clones ≥ 2 cells | ≥ 10 cells | Clonal ICC (shuffled) | P | Programmes | Used in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `np_cgg_wt_d14` | Manakkat Vijay et al. 2026 Nat. Immunol. | GSE305877 | mouse | NP-CGG immunisation, one WT mouse, day 14 | 13,727 | 6,959 | 508 |  | 0.176 (0.006) | 0.002 |  | Fig. 2B; Supp. Fig. 1 (sheet 1) |
| `gut_gc_reporter` | Siniscalco et al. 2025 Immunity | GSE284622 | mouse | oral OVA + cholera toxin, GC-fate reporter, MLN/PP/LP | 16,619 | 11,723 | 958 | 119 | 0.401 (0.278) | 0.002 | 3 | Fig. 3g-j; Supp. Figs. 1, 2, 4, 5, 6 |
| `mouse_rbd` | Merkenschlager et al. 2025 Nature | GSE287123 | mouse | RBD protein / mRNA, H2B-mCherry division reporter | 36,188 | 16,667 | 1,887 | 160 | 0.169 (0.051) | 0.002 | 2 | Fig. 4; Supp. Fig. 1 (sheet 2) |
| `mouse_np` | Merkenschlager et al. 2025 Nature | GSE287123 | mouse | NP-OVA, H2B-mCherry division reporter | 10,109 | 3,894 | 471 | 41 | 0.120 (0.041) | 0.002 | 1 | Supp. Fig. 1 (sheet 2) |
| `gc_np_pc` | ElTanbouly et al. 2024 J. Exp. Med. | GSE246382 | mouse | NP-OVA day 14, sorted GC zones and plasma cells (Smart-seq2) | 884 | 762 | 96 | 1 | 0.048 (0.032) | 0.226 | 2 | Supp. Fig. 1 (sheet 1) |
| `malaria_late` | Skinner et al. 2026 Nat. Immunol. | GSE286215 (exp. 2) | mouse | Plasmodium chabaudi infection, days 10-42 | 72,359 | 59,922 | 1,015 | 0 | 0.451 (0.015) | 0.002 | 4 | Fig. 2B-D; Supp. Figs. 2, 6 |
| `malaria` | Skinner et al. 2026 Nat. Immunol. | GSE286215 (exp. 1) | mouse | Plasmodium chabaudi infection, days 0-14 | 40,623 | 21,145 | 523 | 0 | 0.736 (0.062) | 0.002 | 4 | Supp. Table 2 |
| `ln_vaccine` | Kim et al. 2022 Nature | GSE195673 | human | SARS-CoV-2 mRNA vaccination, lymph node FNA + blood | 193,442 | 153,049 | 11,141 | 1,243 | 0.326 (0.164) | 0.002 | 4 | Fig. 2B; Supp. Figs. 2, 6 |
| `spont_gc` | Castrillon et al. 2023 eLife | GSE203132 | mouse | spontaneous GCs, AID-EYFP+ splenic B cells | 22,898 | 8,105 | 1,051 | 127 | 0.279 (0.026) | 0.002 | 3 | Fig. 3a-f; Supp. Fig. 3 |
| `covid_bcell` | Dugan et al. 2021 Immunity | GSE171703 | human | antigen-bait-sorted B cells after SARS-CoV-2 infection | 61,423 | 39,503 | 588 | 31 | 0.400 (0.036) | 0.002 | 1 | additional (not in manuscript) |
| `flu` | Wang et al. 2023 | GSE175522 / GSE175523 | human | seasonal influenza vaccination, blood d0 and d7 (heavy chain only) | 123,693 | 84,037 | 2,082 | 26 | 0.561 (0.030) | 0.002 | 4 | additional (not in manuscript) |
| `flu_lung` | GEO series | GSE317692 | mouse | influenza A infection, lung and mediastinal LN, day 20, HA tetramers | 4,121 | 2,115 | 204 | 20 | 0.162 (0.052) | 0.002 | 1 | additional (not in manuscript) |
| `tonsil` | King et al. 2021 Sci. Immunol. | E-MTAB-9005 / 9003 | human | paediatric tonsil | 22,478 | 11,570 | 361 | 0 | 0.253 (0.033) | 0.002 | 2 | additional (not in manuscript) |
| `ebv` | Mitul et al. 2026 PNAS | GSE317492 | human | EBV infection of tonsil organoids (donors pooled) | 205,630 | 123,369 | 6,124 | 974 | 0.158 (0.043) | 0.002 | 2 | additional (not in manuscript) |
| `bone_marrow_pc` | GEO series | GSE253857 | human | bone-marrow plasma cells and memory B cells, with blood | 115,144 | 67,895 | 6,583 | 64 | 0.427 (0.245) | 0.002 | 3 | additional (not in manuscript) |

Clonal ICC: share of transcriptional variance explained by clone identity; in brackets the mean over 500 permutations of clone labels within samples; P = 0.002 is the floor for 500 permutations.

Notes on individual datasets:

* **NP–CGG (GSE305877).** Only one wild-type mouse (day 14) is individually identifiable in the public files;
  the other mice were pooled without per-cell donor labels and were not used. Analysed by its own script
  ([`figures/reserve/source/analyse_np_cgg_wt_d14.py`](figures/reserve/source/analyse_np_cgg_wt_d14.py)) with
  standard quality control (≥ 200 genes, ≥ 500 UMIs, < 20 % mitochondrial reads).
* **GSE287123** (RBD and NP–OVA). Mice were pooled into libraries with hashtag antibodies and demultiplexed
  (centred log-ratio, per-hashtag two-means). Each sort gate of the NP–OVA cohort and the bait/zone gates of the
  RBD cohorts were sequenced in their own libraries; only the RBD division gate is crossed with libraries.
* **GSE246382** (plate-based Smart-seq2, 884 cells, 11 mice). Receptors from TRUST4; W33L called after aligning
  query and germline. Clonal coherence is not significant (96 clones, mostly of two cells), but clone profiles
  are strongly associated with the plasma-cell sort (R² 0.78).
* **GSE286215** (*Plasmodium*). One sampling day per mouse (terminal sampling), so time is a donor-level
  label; experiment 1 (days 0–14) and experiment 2 (days 10–42) are analysed separately. No clone reaches ten
  cells.
* **GSE195673** (human lymph node). The kernel profiles, clone embedding, programmes and associations use the
  1,243 clones with ≥ 10 cells; coherence uses all 11,141 clones with ≥ 2 cells; Harmony integration over
  participants (from the original study). The blood/lymph-node label is a library label.
* **GSE203132** (spontaneous GCs). Heavy-chain clones (the deposited AIRR table has no light chains); clones
  re-defined on paired chains from the deposited contigs give the same result (see 2.3-1); states from
  marker-gene modules.
* **Additional datasets** (not in the manuscript): antigen-bait-sorted human B cells (GSE171703), influenza
  vaccination (GSE175522, heavy chain only, so its clones are not comparable with the paired datasets),
  influenza infection with haemagglutinin tetramers (GSE317692), tonsil (E-MTAB-9005), EBV-infected tonsil
  organoids (GSE317492; donors pooled, exploratory) and human bone-marrow plasma cells (GSE253857). All run
  through the same pipeline; results in [`case_studies/results/`](case_studies/results).
* **Removed:** the COVID-19 PBMC atlas subset distributed with scirpy (Stephenson *et al.* 2021): 83 clones,
  almost all single cells, no light chain.
* **Non-B-cell lineage data** (not in the manuscript): LARRY haematopoiesis, WATERMELON persister cells and
  Clonotrace's NSCLC T-cell data ([`analyses/lineage_barcode_systems/`](analyses/lineage_barcode_systems)).

---

## 5. Analyses

### 5.1 Analyses in the main and supplementary figures

| Figure | What was computed | Script | Results |
|---|---|---|---|
| Fig. 1 | schematic and simulated example | `figures/reserve/source/figure1_a4.py` | — |
| Fig. 2A | dataset and design table | `figures/reserve/source/polish_main_figure_text.py --figure 2` | `case_studies/results/*/summary.json` |
| Fig. 2B | clone embeddings of NP–CGG, *Plasmodium* (days 10–42), human lymph node | `case_studies/run_case_study.py`, `analyse_np_cgg_wt_d14.py` | `case_studies/results/{np_cgg_wt_d14,malaria_late,ln_vaccine}/` |
| Fig. 2C | clone sizes, coherence with within-mouse permutations, programme stability | same | `malaria_late/summary.json` |
| Fig. 2D | programmes projected onto cells; equal-weight composition | same | `malaria_late/cells.csv.gz`, `programme_composition.csv` |
| Fig. 3a–f | marker-module states, clone embedding, entropy along the first axis, memory by quartile, per-mouse correlations | `case_studies/spont_gc_fates.py`, `p3_subsplit_analysis.py` | `case_studies/results/spont_gc/` |
| Fig. 3g–j | gut clone embedding, cells by programme, GC sort on cells and clones | `case_studies/run_case_study.py`, `gut_plasma_branches.py` | `case_studies/results/gut_gc_reporter/` |
| Fig. 4 | design, division on cells and clones, PC1 deciles, crossover cosine, per-mouse ρ, decile cells, bait and zone | `figures/reserve/source/figure4_analysis_2026-10-19.py` | `figures/reserve/figure4_rbd_2026-10-19/` |
| Supp. Fig. 1 | NP model-antigen details; reporter, bait, zone and gut sorting annotations | `redesign_figure2_and_model_supps.py` | case-study results |
| Supp. Fig. 2 | cells and clone members on the cell embedding | `supp_clones_on_cells.py` | case-study results |
| Supp. Fig. 3 | state-diversity robustness | `revision_spont_gc_2026-10-17.py` | `figures/reserve/revision_2026-10-17/spont_gc/` |
| Supp. Figs. 4–5 | gut path and centring | `revision_gut_2026-10-17.py` | `figures/reserve/revision_2026-10-17/gut/` |
| Supp. Fig. 6 | dominant compartment per clone | `redesign_figure2_and_model_supps.py --only compartments` | case-study results |

### 5.2 Analyses beyond the figures

All of these are in [`analyses/`](analyses/README.md) or [`case_studies/results/`](case_studies/results),
with scripts. Those whose outputs had been removed in the 2026-10-13 clean-up of old (heavy-chain-only)
clone definitions were re-run on the current clones on 2026-10-20.

The full write-up of each (question, data, method, result with numbers, caveats) is in [`analyses/README.md`](analyses/README.md).

| # | Analysis | Data | One-line result | Location |
|---|---|---|---|---|
| 1 | [Export lag](analyses/README.md#1-export-lag-how-far-behind-its-germinal-centre-is-a-clones-output) | *Plasmodium*, human lymph node | within a clone, output cells are less mutated than the clone's own GC cells (65 vs 8 clones in Plasmodium; 1,536 vs 821 in human lymph node) | [`export_lag/`](analyses/export_lag) |
| 2 | [Convergent receptors and the compartments of a clone](analyses/README.md#2-convergent-receptors-in-plasmodium-infection) | *Plasmodium* | receptors shared between mice are enriched in plasmablast-dominated clones | [`case_studies/results/pcas_arm_analyses/`](case_studies/results/pcas_arm_analyses) |
| 3 | [States that co-occur within clones](analyses/README.md#3-which-states-co-occur-within-a-clone-plasmodium) | *Plasmodium* | GC + plasmablast clones are enriched from day 10 on, depleted before | [`case_studies/results/clone_state_sharing/`](case_studies/results/clone_state_sharing) |
| 4 | [Clonal structure through infection](analyses/README.md#4-clonal-structure-through-the-plasmodium-infection) | *Plasmodium* | the share of state explained by clone falls from 0.78 to 0.28 while clones diversify | [`case_studies/results/malaria/timecourse.csv`](case_studies/results/malaria/timecourse.csv) |
| 5 | [Selection on the antibody sequence](analyses/README.md#5-selection-on-the-antibody-sequence) | 5 datasets | replacement mutations slightly above chance in model-antigen GCs; weakly related to sort gates | `case_studies/results/<dataset>/sequence_selection.csv` |
| 6 | [Spike binding and mutation in the vaccine lymph node](analyses/README.md#6-spike-binding-and-somatic-mutation-in-the-human-lymph-node) | human lymph node | spike-binding GC families are slightly more mutated at every visit; the label is clone-level, not affinity | [`case_studies/results/spike_binding_audit/`](case_studies/results/spike_binding_audit) |
| 7 | [Division, binding and zone with library controls](analyses/README.md#7-model-antigen-germinal-centres-with-library-controls) | GSE287123 | with profiles centred within each mouse × gate library, division still explains clone profiles; bait and zone almost vanish | `case_studies/results/mouse_gc_deep_dive/` |
| 8 | [Lineage trees](analyses/README.md#8-somatic-hypermutation-lineage-trees) | GSE246382, NP–OVA, RBD | close relatives on the tree do not share state beyond the clone; plasma cells sit nearer the germline than GC cells of their clone | `case_studies/results/<dataset>/lineage/` |
| 9 | [Clonal memory across time, tissue and gates](analyses/README.md#9-clonal-memory-across-time-points-tissues-and-gates) | 11 datasets | clones keep their state across vaccine visits and between RBD-bait gates, not between lymph node and blood | [`cross_dataset/clonal_memory.csv`](analyses/cross_dataset/clonal_memory.csv) |
| 10 | [Clonally inherited gene sets](analyses/README.md#10-clonally-inherited-genes) | 15 datasets | plasma-cell and GC programmes are inherited within clones above expression-matched genes | [`cross_dataset/geneset_heritability.csv`](analyses/cross_dataset/geneset_heritability.csv) |
| 11 | [All clone-level association tests](analyses/README.md#11-every-clone-level-association-test) | 15 datasets | 58 annotation tests in one table | [`cross_dataset/label_effects.csv`](analyses/cross_dataset/label_effects.csv) |
| 12 | [Additional datasets](analyses/README.md#12-additional-datasets) | 6 datasets | the pipeline on influenza, tonsil, EBV, bone marrow and bait-sorted cells (four re-run on 2026-10-20 without light-chain-only pseudo-clones) | `case_studies/results/<dataset>/` |
| 13 | [Gut: all B cells and centring](analyses/gut_all_B_and_centring/README.md) | GSE284622 | the second plasma branch is one mouse; the GC → plasma path survives every centring | [`gut_all_B_and_centring/`](analyses/gut_all_B_and_centring) |
| 14 | [Lineage-barcode and TCR data](analyses/lineage_barcode_systems/README.md) | LARRY, WATERMELON, NSCLC T cells | day-2 profiles predict day-6 fate (kernel vs centroid not significantly different); T-cell clones keep their state across therapy cycles | [`lineage_barcode_systems/`](analyses/lineage_barcode_systems) |
| 15 | [Technical validation](analyses/technical_validation/README.md) | all | the tutorial path on 16 datasets without an error; clone definition and embedding sensitivity; power of the coherence test; one million cells; T cells; automatic clone-map settings (30/30 maps coherent); state-density model; runtimes; audits | [`technical_validation/`](analyses/technical_validation) |
| 16 | [Negative results and corrected defects](analyses/negative_and_withdrawn_results.md) | — | three falsified predictions, two withdrawn headlines, eighteen fixed defects | — |

### 5.3 Negative results and corrected defects

What did **not** hold, and the defects found in our own code and estimators, are recorded in
[`analyses/negative_and_withdrawn_results.md`](analyses/negative_and_withdrawn_results.md): the withdrawn
"GC continuum" dichotomy, the withdrawn "cell state collapses" headline, three falsified frozen predictions,
the artefactual 791-cell clone, the re-interpreted gut plasma branches, and eighteen code or estimator defects
with their fixes.

---

## 6. Benchmarking against related tools

The full benchmark — tools, versions, settings, fairness protocol, every result table and figure, and how to
rerun it — is in [`benchmarking/README.md`](benchmarking/README.md). In brief:

**What was compared.** Threadfin 4.3.1/4.4.0 against Clonotrace 1.2.1, clone2vec 0.1.1, Benisse 1.1.0, CoNGA's
one-cell-per-clone reduction, scirpy's clonotype modularity and two simple baselines (mean of the cells'
embedding coordinates; Leiden-cluster composition), on the nine B-cell datasets of the case studies plus
LARRY lineage barcoding (Clonotrace's own tutorial data) and 12 simulated repertoires with a known answer.
Every tool received the same embedding and clone ids and ran at its documented defaults, and again forced to
clones of ≥ 2 cells. Readouts: coverage; recovery of measured clone annotations that no tool used (sort gates,
isotype, mutation, tissue, later fate), with the same and with disjoint cells; split-half reproducibility;
LARRY clone linking; simulation; calibration and power of clonal tests; runtime and memory per dataset, with
1–16 cores and with growing data; runs that failed or were stopped; sensitivity of Threadfin to its settings;
and a usability checklist. A one-page overview (benchmark Fig. 0) compares every tool with Threadfin on each
of these, as better, no clear difference or worse.

**What was found.**

| Question | Answer |
|---|---|
| Which clones can each tool describe? | Threadfin and Benisse: every clone with ≥ 2 cells by default. clone2vec (default ≥ 3 cells) leaves out all two-cell clones, 28–86 % of the analysable clones; Clonotrace (default ≥ 10 cells) can describe only the 5–18 % of clones that reach ten cells and cannot run on 4 of 11 datasets; forced to two-cell clones it still drops 24–48 % of clones (39–84 % of two-cell clones) after label spreading. |
| Do two-cell clones carry information? | Yes (median R² ≈ 0.25 on two-cell clones alone) — and clone2vec forced to include them, or a plain mean, recovers about as much. "Only Threadfin uses two-cell clones" is true of the tools' defaults, not of the data. |
| Which representation recovers measured annotations best? | None uniformly. Against Threadfin on the same clones: clone2vec (forced) 5 better / 11 worse of 41 annotations, (default ≥ 3 cells) 10 / 7; plain mean 7 / 10; Clonotrace (forced, default distance) 2 / 13 of 41, (default ≥ 10 cells) 0 / 8 of 26; Benisse 3 / 22 of 31, winning only receptor-sequence annotations (bait binding, mutation, W33L). Threadfin's edge over the mean comes from centring on the donor. |
| Are the clonal tests calibrated? | Threadfin's coherence test: 0–8 % false positives at α = 5 % with real sampling structure kept; scirpy (≥ 1 clone flagged): 100 % in 4 of 9 datasets; cell-level ANOVA: 86–100 %. |
| Simulation with a known answer | clone2vec and cluster composition group clones by programme better than Threadfin, and separate "mixture" from "intermediate" clones better (AUROC 0.99 / 0.96 vs 0.88 with ≥ 10 cells; no method succeeds with two-cell clones). Centring removes an additive donor shift exactly but adds same-donor structure when donors differ in composition. |
| Cost | Threadfin profiles 0.2–43 s, < 2.5 GB, and the complete analysis 18 s–9.5 min; clone2vec 0.5–11 min (about 30 times Threadfin); Benisse up to 19 min and 32 GB; Clonotrace from a minute to more than four days (290–2,400 times Threadfin; 31 h for 6,174 LARRY clones on 6 workers; 101 h for the 11,141 lymph-node clones on 32 workers). |
| More cores, more data | 16 instead of 1 core: Threadfin's complete analysis 2.0 times faster (no gain beyond 8), clone2vec 1.2, Clonotrace 2.7 (from 4 to 16), Benisse 2.9 times *slower*. Ten times more clones: Threadfin's profiles 20 times longer, clone2vec 6, Clonotrace 164, Benisse 123 (and 35 GB at 5,570 clones). Threadfin and Clonotrace give identical results on any number of cores; clone2vec and Benisse differ slightly. |
| Runs that finished | Threadfin, clone2vec forced to two-cell clones and the baselines: all 77 (full data and halves). clone2vec at its default: 65 of 77 (fails on small inputs with an internal error). Benisse: 26 of 27. Clonotrace: 45 of 50 without help (five needed a larger cell graph, one of them after failing at k = 10, 30 and 50). |
| Sensitivity of Threadfin | unchanged by the number of random features, the seed, 5–30-cell smoothing or 50 principal components; a narrower kernel separates mixed from in-between clones better and a wider one worse; without smoothing measured annotations are recovered better but planted programmes worse. |
| Ease of use (checklist) | Threadfin 12 of 13 (not yet peer-reviewed); clone2vec 11 of 13; Clonotrace 10 of 13; Benisse and CoNGA 5.5; scirpy 11 of the 11 criteria assessed. |

**What this means for the manuscript.**

1. The Discussion sentence *"We did not compare these methods on common tasks"* (2.3-2) must be replaced;
   suggested content: coverage of small clones and calibration of the clonal test are where Threadfin
   differs; on annotation recovery it is comparable to clone2vec and to a mean of the embedding; on
   simulated discrete programmes clone2vec and cluster composition do better.
2. Any wording that implies other tools *cannot* use two-cell clones should say that they *exclude them by
   default* (clone2vec ≥ 3 cells, Clonotrace ≥ 10) and that Clonotrace still loses 39–84 % of two-cell clones
   when forced.
3. The donor-centring artefact seen in the gut data (Supp. Fig. 5) is reproduced in simulation; the Methods
   or Discussion should state that centring assumes donors differ by a shift, not by composition.
4. Benchmark figures suitable for a supplementary figure: the one-page overview (Fig. 0 of the benchmark),
   coverage (Fig. 1), two-cell clones (Fig. 4), test calibration (Fig. 5), simulation (Fig. 6) and computing
   resources (Figs. 11–12).

---

## 7. Methods and statistical conventions in brief

The model behind every step is described in [`../docs/METHODS.md`](../docs/METHODS.md) (with plain-language
summaries). Conventions used throughout the manuscript:

* **Clone definition.** Dandelion (sc-dandelion 0.5.7), paired heavy and light chains, within donors,
  junction amino-acid identity ≥ 85 % (GSE203132: heavy chain only, with a paired-chain sensitivity analysis;
  GSE175522: heavy chain only).
* **Embedding.** Library-size normalisation to 10,000, log1p, immunoglobulin and TCR genes removed,
  3,000 highly variable genes, scaling, 30 principal components; Harmony over participants for GSE195673 only.
* **Profiles.** Kernel mean embeddings (256 random Fourier features, median-heuristic bandwidth, 15-neighbour
  smoothing), centred on the donor, best linear unbiased predictor under a one-way random-effects model,
  30 principal components; clones with ≥ 2 cells (human lymph node: ≥ 10 for profiles); no reliability filter.
* **Coherence.** ICC from unsmoothed centred coordinates; 500 permutations of clone labels within samples.
* **Associations.** R² of clone profiles on a clone-level annotation (PERMANOVA / distance-based redundancy
  analysis); 2,000 permutations within donors; excess R² = R² − permutation mean.
* **P values** include a correction of one, so the floors are 0.002 (500 permutations) and 0.0005 (2,000);
  permutation P values for ICC, R², neighbour fractions, neighbourhood correlations and the cosine are
  one-sided; all others two-sided.
* **Seeds** fixed (0 unless stated) for every stochastic step.

---

## 8. Reproducing the analyses

**Environments** (Linux HPC; no GPU).

| Environment | Used for | Key versions |
|---|---|---|
| `threadfin_v4` | all Threadfin analyses and figures | Python 3.11, threadfin 4.2.0/4.3.0/4.3.1, numpy 2.2, scipy 1.17, pandas 2.3, anndata 0.12, scanpy 1.11, scikit-learn 1.9, umap-learn 0.5, leidenalg 0.12, harmonypy 2.0 |
| `dandelion` | clone definition | Python 3.11, sc-dandelion 0.5.7, matplotlib < 3.9 |
| `lineage` | somatic-hypermutation lineage trees | GCtree with PHYLIP `dnapars` |
| benchmark environments | competitor tools | see [`benchmarking/README.md`](benchmarking/README.md) |

**Order of steps.**

```bash
cd manuscript/case_studies
bash download.sh /path/to/data && export THREADFIN_DATA=/path/to/data   # public files + checksums
PYTHONHASHSEED=0 python dandelion_clones.py   # paired clones for every dataset (dandelion environment;
                                              # the fixed hash seed makes Dandelion's calls repeatable)
python cache_embedding.py <dataset>           # expression embeddings without immunoglobulin genes
python run_case_study.py <dataset>            # the Threadfin pipeline, one dataset at a time
python summarize.py                           # cross-dataset tables
python ../figures/reserve/source/polish_main_figure_text.py --figure 1   # ... 2, 3, 4: main figures
python ../dataset_overview.py                 # the dataset table of this page
```

Dataset-specific analyses and the scripts behind every supplementary figure are listed in
[`figures/README.md`](figures/README.md) and [`case_studies/README.md`](case_studies/README.md). Figure
scripts carry the absolute paths of the project layout in which they were run; adjust the constants at the top
of each script to your layout. A complete analysis of one dataset takes 0.4–15 minutes on one node.

---

## 9. What is still worth improving

Listed by priority. "Before submission" items are also in section 2.3.

### A. Before submission

1. ~~Correct or redo the GSE203132 clone definition~~ — done 2026-10-23: paired-chain clones give the same
   result; one Methods sentence remains (2.3-1).
2. **Bring the benchmark into the paper** (2.3-2): a supplementary figure with coverage of small clones,
   annotation recovery, split-half reproducibility, calibration of clone-level tests and runtime, and a revised
   Discussion sentence.
3. **Deposit code and processed data on Zenodo** (clone tables, profiles, source data for every figure) and
   complete the reporting summary.
4. ~~Approve the library-confound paragraph~~ (approved 2026-10-23) and ~~add the paired-chain completeness
   sentence for RBD~~ (in the draft since 2026-10-23).

### B. Strengthen the evidence

5. **Independent replication of the gut clonal path** in a second fate-mapped gut or mucosal dataset; the
   path rests on three mice, one of which dominates a branch.
6. **A dataset with crossed gates and libraries for the zone and bait gates**, or a re-analysis of other
   reporter datasets, so that antigen binding and zone can be tested free of library effects.
7. **Repeat the cross-animal prediction** of compartment composition in *Plasmodium*
   (section 1.8 of the negative-results record) on the paired clone definition.
8. **A calibrated test for the number of programmes.** The split test is conservative (it misses two groups
   four standard deviations apart); a calibrated alternative (e.g. SigClust-type null with a thresholded
   covariance, or a within-clone parametric bootstrap) would turn programme counts into evidence.
9. **Light chains in all analyses.** Some loaders keep only the heavy chain for per-cell annotations (e.g.
   W33L, mutation frequency); light-chain mutation and pairing information is unused.
10. **Functional validation.** The manuscript's interpretations (bridging clones, memory in multi-state clones,
    division along the clone axis) are hypotheses for experiments: monoclonal antibody expression and
    binding for clones at the two ends of an axis, photoconversion or fate-mapping to give the gut path a
    direction, and spatial receptor profiling.

### C. Package

11. **The default neighbour smoothing** (each cell averaged with its 15 nearest cells): without smoothing,
    measured annotations are recovered better (14 better, 1 worse of 35) but planted programmes worse in
    simulation (benchmark section 6.10), so the default was kept. Smoothing only clones below a size
    threshold remains worth testing.
12. **Automatic choice of the sampling context**: warn when `sample_key` is confounded with an annotation
    passed in `test`.
13. **Composition-balanced centring as an option** of `clone_profiles`, since it was needed to interpret the
    gut branches; the benchmark simulation shows the same artefact (centring adds same-donor structure when
    donors differ in clone composition).
14. **scverse integration** (MuData/scirpy input) and a PyPI release — *done in 4.5.0* (`pip install threadfin`).

### D. Beyond this manuscript

15. TCR data and lineage-barcode data run through the same machinery (shown in
    `analyses/lineage_barcode_systems/`) but are outside the scope of this paper.

---

## 10. History of the manuscript

| Date | Step |
|---|---|
| 2026-10-02 | version 4 of the package; repository reorganised around biological questions |
| 2026-10-03 | first case studies, lineage trees; journal assessment (Cell Systems / Genome Biology recommended, Nature Methods a stretch without benchmarks) |
| 2026-10-04 | draft v1; the "GC continuum" dichotomy withdrawn after audit |
| 2026-10-09–10 | lineage heritability and clone-level prediction studies; independent review; three frozen predictions falsified; "cell state collapses" withdrawn |
| 2026-10-11 | clones redefined with Dandelion (paired chains); reliability filter removed; influenza adapter defect fixed |
| 2026-10-13 | clean-up: everything computed with old clone definitions quarantined |
| 2026-10-15–16 | figures reorganised; gut centring artefact found and explained |
| 2026-10-17 | preprint draft for *Nature Methods*; Figures 1–3 fixed by the authors; supplementary analyses for Figs. S3–S5 |
| 2026-10-18 | every sentence checked against saved results (9 corrections); package 4.3.0 (automatic clone-map settings); Results headings rewritten as claims |
| 2026-10-19 | new Figure 4 (RBD division reporter, crossed libraries) |
| 2026-10-20 | this folder: all analyses published, orphaned analyses re-run on current clones, end-to-end benchmark, beginner documentation and a tested end-to-end tutorial notebook; package 4.3.1 |
| 2026-10-23 | package 4.4.0 (parallel `n_jobs` with identical results, contig-id fix, clearer errors, function reference, Python 3.10–3.14 and oldest dependencies tested); benchmark overview figure, computing-resource and sensitivity readouts, usability checklist; GSE203132 paired-chain and RBD complete-pair checks (Figures 3 and 4 unchanged) |
| 2026-10-24 | package 4.5.0 (T cells, scirpy input, `landscape_genes`, within-state markers, `result.methods()`); clone-definition and embedding sensitivity, power of the coherence test, one-million-cell run, T cells end to end; the tutorial path re-run on 16 datasets |
| 2026-10-27 | benchmark refreshed after Clonotrace's run on the 11,141 lymph-node clones finished (101 h on 32 workers): forced to two-cell clones, Clonotrace now runs on all 11 datasets; the reason texts of Clonotrace runs in `benchmarking/results/coverage_runtime.csv` corrected |

---

## 11. Layout of this folder

```
manuscript/
├── README.md                    this page
├── dataset_overview.{py,csv,md} the dataset table of section 4
├── figures/                     main and supplementary figures, their scripts and provenance
│   ├── README.md                every figure explained, and how to reproduce it
│   └── reserve/                 source/ (figure code), provenance/, revision_2026-10-17/, figure4_rbd_2026-10-19/
├── case_studies/                loaders, the per-dataset pipeline, dataset-specific analyses, results/
├── analyses/                    analyses beyond the figures, technical validation, negative results
└── benchmarking/                end-to-end comparison with related tools (scripts, results, figures)
```
