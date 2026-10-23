# Figures

Figures of the Threadfin manuscript (in preparation). Each figure is provided as a 600 dpi PNG and a PDF
(main figures also as editable SVG). The full legends are part of the manuscript; below, every figure is
explained in plain language — what question it answers, what each part shows and how to read it — followed
by the commands that reproduce it. For the status of the manuscript and every analysis behind it, see the
[manuscript overview](../README.md).

**Conventions used in every clone embedding.** One point is one clone (cells that share one paired
heavy- and light-chain rearrangement, within one individual). Point area grows with the number of cells
sampled. Points close together have similar *distributions* of cell states. Proximity does not mean shared
ancestry, and the axes have no units. Annotations shown in colour (sort gates, tissue, isotype) were **not**
used to build the clone profiles.

## Main figures

| Figure | Content | Files |
|---|---|---|
| Figure 1 | Threadfin: from paired single-cell RNA and BCR sequencing to the clone embedding; algorithm, clonal-coherence test and the kind of biological inference it supports | [PNG](Figure_1.png) · [PDF](Figure_1.pdf) · [SVG](Figure_1.svg) |
| Figure 2 | Application to germinal-centre datasets: datasets and designs, clone embeddings of three systems, diagnostics in *Plasmodium* infection and projection of programmes onto cells | [PNG](Figure_2_GC_overview.png) · [PDF](Figure_2_GC_overview.pdf) · [SVG](Figure_2_GC_overview.svg) |
| Figure 3 | Clone states: state diversity of spontaneous-GC clones and the clonal path from germinal-centre B cells to plasma cells in gut-associated lymphoid tissues | [PNG](Figure_3_clone_states.png) · [PDF](Figure_3_clone_states.pdf) · [SVG](Figure_3_clone_states.svg) |
| Figure 4 | Model-antigen germinal centres: recent division (H2B–mCherry reporter), RBD binding and zone along the clone profiles of two RBD cohorts whose sequencing libraries are crossed | [PNG](Figure_4_model_antigen_GC.png) · [PDF](Figure_4_model_antigen_GC.pdf) · [SVG](Figure_4_model_antigen_GC.svg) |

### Figure 1 — what Threadfin does

*Question:* how do paired single-cell RNA and BCR data become a picture of clones? **a**, the inputs (each
cell's expression state, and its heavy and light chains linked by the cell barcode; clones assigned within
each donor) and the output, a clone embedding with one point per clone. **b**, the workflow on simulated
cells: (1) cells in expression space, connected by clone; (2) each clone summarised by the mean of its cells'
kernel features (its "Fréchet mean" in feature space) after centring on the sample and shrinkage by clone
size; (3) clones placed by their profiles, so that clones made of intermixed cells can still separate;
(4) the coherence test — clonal intraclass correlation against labels shuffled within samples; (5–6) the kind
of inference it supports: continuous structures of clones that can be compared with annotations never used
to build them. All elements are schematic.

### Figure 2 — the method applied to germinal-centre datasets

*Question:* does Threadfin run on very different datasets, and what diagnostics does it report? **A**, the
eight datasets of this figure (model antigens, *Plasmodium* infection, human vaccination) and which cell
types each study sampled. **B**, clone embeddings of three of them (NP–CGG, 508 clones; *Plasmodium*,
1,015 clones; human lymph node, 1,243 clones with ≥ 10 cells), coloured by programme. **C**, the diagnostics
for *Plasmodium*: most profiled clones have only two cells, yet clone identity explains 45 % of state
(ICC 0.451 against 0.015 for shuffled labels) — shrinkage lets 2-cell clones count without being treated as
if sampled deeply; programme stability shows that one programme (P3) is not reproducible. **D**, programme
labels projected back onto cells: plasmablasts of programmes P2 and P3 occupy the same region of the cell
embedding, but P3 clones also contain germinal-centre B cells — a property of clones that a cell-level map
cannot show.

### Figure 3 — clone states

*Question:* what does a clone-level view show that a cell-level view does not? **a–f**, GC-experienced
B cells from mice with spontaneous germinal centres (GSE203132): the first axis of the clone embedding
orders clones from single-state antibody-secreting-cell clones to multi-state clones (Shannon entropy of
the clone's state composition, ρ = −0.74), and memory B cells occur mostly in the multi-state clones
(36 % of clones in the first quartile, 2.5 % in the last), consistently in each mouse. **g–j**, gut
immunisation with a GC-fate reporter (GSE284622): the clone embedding arranges clones along a continuous path
from germinal-centre clones, through "bridging" clones that contain both GC B cells and plasma cells and span
inductive sites and the lamina propria, to plasma-cell clones; clones whose plasma cells lacked GC-fate
labelling group at the end of the path. The isotype changes along the path (IgG1 → IgA) were measured in the
BCR sequences, not in the transcriptome used for the profiles.

### Figure 4 — division, antigen binding and zone in model-antigen germinal centres

*Question:* are measured properties of germinal-centre clones reflected in their profiles, and can a library
artefact be excluded? Data: GSE287123, H2B–mCherry division-reporter mice immunised with RBD protein (5 mice,
sorted by RBD-bait binding) or RBD mRNA (5 mice, sorted by light/dark zone); 36,188 cells; 1,887 clones with
≥ 2 cells. **a**, the design: four sequencing libraries, each holding one gate of each cohort, with the
recently divided (mCherry-low) cells in libraries A and C for one cohort and B and D for the other.
**b–c**, the division gate on the cell embedding and, per clone, on the clone embedding. **d**, the first
principal component of the clone profiles (42 % of their variation; it follows a dark-zone/cycling gene score)
orders clones by recent division in both cohorts (the recently divided fraction rises from about 0.1 to
0.7–0.9 across deciles). **e**, the key control: the profile directions associated with division agree
between the cohorts (cosine 0.89), whereas a library effect would make them opposite (−1). **f**, the same
correlation is positive in all ten mice. **g**, cells of clones in deciles 1, 5 and 10. **h–j**, RBD-bait
binding and the dark-zone gate also rise along the same axis, but both were sequenced in libraries C and D,
so for these two gates biology and library cannot be separated.

## Supplementary figures

| Figure | Content | Files |
|---|---|---|
| Supplementary Fig. 1, sheet 1 | NP–CGG immunization: cell states, sorting gates, W33L mutation and clone summaries | [PNG](Supplementary_model_antigen_NP.png) · [PDF](Supplementary_model_antigen_NP.pdf) |
| Supplementary Fig. 1, sheet 2 | division reporter, RBD bait and zone gates, and oral OVA immunization | [PNG](Supplementary_model_antigen_reporters.png) · [PDF](Supplementary_model_antigen_reporters.pdf) |
| Supplementary Fig. 2 | cells and clone members on the cell-level embedding (gut, *Plasmodium*, human vaccination) | [PNG](Supplementary_clones_on_cells.png) · [PDF](Supplementary_clones_on_cells.pdf) |
| Supplementary Fig. 3 | robustness of the state-diversity ordering of spontaneous-GC clones | [PNG](reserve/revision_2026-10-17/figures/Supplementary_Figure_3_state_diversity.png) |
| Supplementary Fig. 4 | the gut clonal path: composition, isotype, tissue and GC-fate labelling along the path | [PNG](reserve/revision_2026-10-17/figures/Supplementary_Figure_4_gut_clone_relationships.png) |
| Supplementary Fig. 5 | donor composition and context centring of the gut clone embedding | [PNG](reserve/revision_2026-10-17/figures/Supplementary_Figure_5_gut_centring.png) |
| Supplementary Fig. 6 | dominant cell compartment of each clone (gut, *Plasmodium*, human vaccination) | [PNG](Supplementary_clone_compartments.png) · [PDF](Supplementary_clone_compartments.pdf) |

In plain language: **S1** gives the cell-level detail behind the model-antigen datasets, including the
plate-based NP–OVA study in which clonal coherence was *not* significant (96 clones) although clone profiles
were strongly associated with the plasma-cell sorting gate — coherence and association are different
questions. **S2** shows, for three datasets, the cells of each clone connected by lines on the cell
embedding next to the clone embedding: clones whose cells are scattered across cell clusters become single
points. **S3** repeats the state-diversity result of Figure 3 on axes computed from the profiles rather
than the map, after rarefying every clone to five cells and after adjusting for clone size (ρ −0.75 to
−0.76). **S4** quantifies the gut path: composition, tissue, isotype and GC-fate labelling of clones in ten
equal groups along it. **S5** explains a second plasma-cell branch of the gut embedding: it consists almost
entirely of one mouse whose sampled response was dominated by lamina-propria plasma cells, and it merges with
the main branch without centring or with composition-balanced centring — a worked example of the centring
caveat in [Methods](../../docs/METHODS.md). **S6** colours each clone by its most frequent named compartment.

## Source code and reproduction

The scripts that draw the figures are in [`reserve/source/`](reserve/source); the numbers shown in the
figures are recorded in [`reserve/provenance/`](reserve/provenance) (paths in provenance files written before
2026-10-20 refer to the earlier layout, in which `manuscript/case_studies/` was `case_studies/` and this folder
was `paper/figure_plan/`). The scripts read the case-study results in
[`../case_studies/results/`](../case_studies/results) and expect the project layout used to produce them
(absolute paths at the top of each script).

| Figure | Entry point |
|---|---|
| Figures 1–3 | `python reserve/source/polish_main_figure_text.py --figure 1` (or `2`, `3`); Figure 1 alone: `reserve/source/figure1_a4.py` |
| Figure 4 | analysis: `reserve/source/figure4_analysis_2026-10-19.py` (outputs in [`reserve/figure4_rbd_2026-10-19/`](reserve/figure4_rbd_2026-10-19)); figure: `reserve/source/figure4_figure_2026-10-19.py`, or `polish_main_figure_text.py --figure 4` |
| Supplementary Figs. 1 and 6 | `python reserve/source/redesign_figure2_and_model_supps.py --only np` (or `reporters`, `compartments`) |
| Supplementary Fig. 2 | `reserve/source/supp_clones_on_cells.py` |
| Supplementary Figs. 3–5 | analyses: `reserve/source/revision_spont_gc_2026-10-17.py`, `revision_gut_2026-10-17.py` (outputs in [`reserve/revision_2026-10-17/`](reserve/revision_2026-10-17)); figures: `revision_supp_figures_2026-10-17.py` |
| NP–CGG analysis | `reserve/source/analyse_np_cgg_wt_d14.py` |

The clone embeddings in these figures were drawn with fixed UMAP settings, recorded per dataset in
`../case_studies/run_case_study.py` and in the provenance files. Threadfin 4.3 chooses these settings
automatically by default; see [Tutorial 3](../../docs/tutorials/03_clone_map.md).
