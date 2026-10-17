# Figures

Figures of the Threadfin manuscript (in preparation). Each figure is provided as a 600 dpi PNG and a PDF
(main figures also as editable SVG). Legends are part of the manuscript.

## Main figures

| Figure | Content | Files |
|---|---|---|
| Figure 1 | Threadfin: from paired single-cell RNA and BCR sequencing to the clone embedding; algorithm, clonal-coherence test and the kind of biological inference it supports | [PNG](Figure_1.png) · [PDF](Figure_1.pdf) · [SVG](Figure_1.svg) |
| Figure 2 | Application to germinal-centre datasets: datasets and designs, clone embeddings of three systems, diagnostics in *Plasmodium* infection and projection of programmes onto cells | [PNG](Figure_2_GC_overview.png) · [PDF](Figure_2_GC_overview.pdf) · [SVG](Figure_2_GC_overview.svg) |
| Figure 3 | Clone states: state diversity of spontaneous-GC clones and the clonal path from germinal-centre B cells to plasma cells in gut-associated lymphoid tissues | [PNG](Figure_3_clone_states.png) · [PDF](Figure_3_clone_states.pdf) · [SVG](Figure_3_clone_states.svg) |

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

## Source code

The scripts that draw the figures are in [`reserve/source/`](reserve/source); the numbers shown in the
figures are recorded in [`reserve/provenance/`](reserve/provenance). They read the case-study results in
[`case_studies/results/`](../../case_studies/results) and expect the project layout used to produce them
(absolute paths at the top of each script).

| Figure | Entry point |
|---|---|
| Figures 1–3 | `python reserve/source/polish_main_figure_text.py --figure 1` (or `2`, `3`); Figure 1 alone: `reserve/source/figure1_a4.py` |
| Supplementary Figs. 1 and 6 | `python reserve/source/redesign_figure2_and_model_supps.py --only np` (or `reporters`, `compartments`) |
| Supplementary Fig. 2 | `reserve/source/supp_clones_on_cells.py` |
| Supplementary Figs. 3–5 | analyses: `reserve/source/revision_spont_gc_2026-10-17.py`, `revision_gut_2026-10-17.py` (outputs in `reserve/revision_2026-10-17/`); figures: `revision_supp_figures_2026-10-17.py` |
| NP–CGG analysis | `reserve/source/analyse_np_cgg_wt_d14.py` |

The clone embeddings in these figures were drawn with fixed UMAP settings, recorded per dataset in
`case_studies/run_case_study.py` and in the provenance files. Threadfin 4.3 chooses these settings
automatically by default; see [Tutorial 3](../../docs/tutorials/03_clone_map.md).
