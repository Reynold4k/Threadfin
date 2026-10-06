# GC-focused figure plan

The main biological argument concerns captured clone-state organisation in GC
responses. The strongest non-GC gate validation is in Supplementary 5; five
further tested datasets have their own [coverage folder](tested_datasets/README.md).
The layout follows biological questions, actual study design and independent
measurement, rather than a sequence of statistical tests.

Current review outputs (6 October 2026): Figures 1–5 and Supplementary Figures
1–6 and 8 are available. Review the revised [Figure 1 PNG](Figure_1.png) or its
[PDF](Figure_1.pdf). Figure 6 and Supplementary Figure 7 are planned below;
their final outputs await the complete RBD native benchmark. NP results are
already available in the source tables. Pending models are not assigned zero
performance or included as partial comparisons.

| Main figure | Question and evidence |
|---|---|
| 1 | Three panels: **A**, Chen Satoshi’s original GC illustration; **B**, the same conceptual 18 paired cells, allocated to three sequence-defined families of six cells each, progressing from RNA states and BCR sequences to illustrative state-composition rings and a family map; **C**, four visual evidence cards for reporter division gates, same-mouse GC–PB sharing, repeated human capture and clone coherence versus shuffled family identity. BCR sequence calls families; context-adjusted receptor-excluded RNA kernel profiles describe captured family states. The conceptual spaces are not concatenated UMAPs or a learned BCR encoder; family-map proximity denotes state-profile similarity. |
| 2 | Model-antigen GC state nodes. **B** shows newly calculated GSE246382 clone embedding of 49 frozen same-mouse sequence-defined families (≥3 cells), coloured by three unsupervised Leiden clone clusters with an actual capture-size legend. Historical distance-row UMAP / Scanpy-graph settings are fixed before label inspection. Measured gates and Myc RNA annotate the GC selection-associated versus output-enriched groups. A,C/D retain independent reporter design and NP measurements; E–G show RBD reporter views. No direction or future fate is measured; GC interpretation is not extended to non-GC. |
| 3 | Same-mouse PcAS GC/output-like co-occupancy. Full experimental design, early PB and late GC clone maps, cell context, family overlays and mouse/isotype-preserving null. No consistent GC–PB positive excess after isotype control. |
| 4 | Repeated human GC capture and author-identified Spike binding. Clear cell/family maps, restored donor-stratified binding odds forest, repeated families, retention and donor-balanced **GC SHM** curves. Binding classification and SHM are separate measurements; neither is quantitative affinity. |
| 5 | Twelve dataset analyses: biological anchors, observed versus within-library shuffled clonal expression signal, reliable-profile coverage and gene-module contrasts. Analyses from one publication are not counted as independent studies. Module agreement describes expression and is not independent fate validation. |
| 6 | Task-aware tool comparison and native benchmark: official capabilities, common receptor-excluded input, fixed donor-private families and whole-mouse held-out reporter readout. Different output tasks and missing inputs are not zero performance. Native results are required for rendering; an empty score schema causes an error. |

| Supplementary figure | Evidence |
|---|---|
| 1 | Clone sizes, reliability and within-library coherence; low coverage and declined programme inference for GSE246382. |
| 2 | Division-gate family overlays, separate RBD protein binding/mRNA GC-zone views, measured-label retention and library-stratified associations. |
| 3 | PcAS treatment, rare GC/memory-like candidates, exact heavy/light matching, denominators and inference limits. |
| 4 | Human state composition, verified marrow donor/library mapping, descriptive module analysis and established within-family sequence/state null. |
| 5 | Revised marrow/blood validation: cell and family maps, pure-gate examples, exact productive heavy/light receptor identities across verified donors. |
| 6 | Author-clone/Threadfin crosswalk, donor programme binding coverage, individual-donor GC SHM curves and descriptive expression signatures. |
| 7 | Native benchmark family-size sensitivity, independent RBD-probe/GC-zone gate readout, measured execution stages and coverage/fold audits. |
| 8 | Original Top2a notebook output (exact cohort/day unresolved), current GSE246382 measured cell gates, new clone-averaged Myc and marker RNA by captured state bias. Public GSE180920 day7/day14 inputs are verified, but its saved image input chain is incomplete. |

## Reading the concepts and maps

A cell UMAP has one dot per captured cell. A clone UMAP has one dot per
receptor-defined family with an interpretable expression distribution.
Neighbouring family dots represent similar captured state distributions;
they do not establish ancestry between families. A same-donor sequence relation
supports within-family membership. Reporter/FACS/probe labels supply external
measurements, whereas gene signatures reuse the expression data.

Figure 1B’s scBCR space is conceptual, not a learned BCR encoder implemented by
Threadfin. The same 18 illustrative paired cells form three six-cell families
throughout the panel. The v4 package calls families from sequence within donors
and then summarises context-adjusted receptor-excluded RNA distributions with a
kernel profile and reliability. It does not concatenate two UMAPs. Ring charts,
positions and fractions are illustrative; one family-map point represents one
family and nearby points have similar captured state profiles. Figure 1A’s
arrows depict established GC biology, not directions estimated by the package.

## Reproduce and review

```bash
python case_studies/summarize_clonal_information.py
python case_studies/spike_gc_trends.py
python paper/figure_plan/make_biology_figures.py  # six main, eight supplements and tested pages
python paper/figure_plan/make_biology_figures.py --biological-only  # current Figure1–5 / S1–6,S8
python paper/figure_plan/make_figures.py          # main only; requires completed benchmark scores
python paper/figure_plan/make_supplementary.py     # supplements only
```

Saved source tables are under `case_studies/results/`. `figure_audit.json`
records maps, source paths and deterministic clone-example selection. Maps
reuse saved coordinates. The [Figure 3 review](review/Figure_3_map_reproduction.png)
and [analysis audit](../../case_studies/results/map_reproduction_audit/REANALYSIS_zh.md)
explain why swapping early/PB for late/GC changed appearance despite identical
coordinates. Dense cell clouds are rasterised in PDFs; new diagrams and text
remain editable vector artwork, while author-supplied Figure 1A is an embedded
raster original. S8A embeds the unmodified historical Top2a output. Figure 2B
and S8B–D use audited current GSE246382 data; their parameters and raw/source
hashes accompany [the new result tables](../../case_studies/results/gc_np_pc_clone_embedding/summary.json).
The [standalone clone embedding](review/GSE246382_clone_embedding.png), [three-preset review](review/GSE246382_reclustering_presets.png) and
[archived report reference](review/Figure_2_archived_report_reference.png)
are separate comparison artifacts. PNGs are review previews.

Interpretation, methods and revision decisions are in
[the manuscript](../MANUSCRIPT_draft_v2.md), [legends](../FIGURE_LEGENDS.md),
[biological inference review](../BIOLOGICAL_INFERENCE_REVIEW_zh.md),
[Spike audit](../SPIKE_BINDING_AUDIT_zh.md), [method comparison](../METHOD_COMPARISON.md)
and [revision logic](../REVISION_LOGIC_zh.md). Benchmark configuration and actual
execution states are in [BENCHMARK_DESIGN](../BENCHMARK_DESIGN.md) and its source
result tables; a completed encoder alone is not a completed Benisse model.
