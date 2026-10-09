# GC-focused figure plan

The main biological argument concerns captured clone-state organisation in GC
responses. The strongest non-GC gate validation is in Supplementary 5; five
further tested datasets have their own [coverage folder](tested_datasets/README.md).
The layout follows biological questions, actual study design and independent
measurement, rather than a sequence of statistical tests.

Current review (8 October 2026): six main figures and sixteen supplements.
Figures 4/5 are new LARRY and paired-TCR public-data analyses; former non-GC
and cross-dataset main figures are S14/S15. S13 shows all three infection
state arms. S16 tests the biological scope and capture coverage of Figure 2.
The native benchmark is complete and retained, including stronger RNA-mean baselines.

| Main figure | Question and evidence |
|---|---|
| 1 | One integrated concept with vector scientific diagrams and an approved raster fish: a compact three-cluster RNA cloud and membrane-BCR/H/L-contig diagram with three separately coloured receptor trees join an incoming fin filament of the approved cartoon threadfin fish. Its belly contains a double-peak sketch, Clone embeddings and Fréchet mean in feature space; two further filaments connect the GC selection-node map and vivid cycling-to-output gradient (right 67% of width). All clone dots are circles with area proportional to illustrative capture count. A gold outline emphasises the selection-associated node; the dashed GC return is a hypothesis. Coordinates, tree branches and sizes are synthetic, with scientific scope in the legend. |
| 2 | A: independent GSE287123 reporter cohorts. B: reconstructed GSE246382 map of 377 same-mouse V–D–J receptor groups (762 cells; 83 groups contain multiple junctions), with Leiden colour and capture-count area; k=40, min_dist=0.65, seed=123, graph k=15, Leiden resolution=0.3/seed=0. C: 36,188 RBD reporter cells. D/E: the same 381 reliable donor-private IGH sequence families, division/SHM colours, kernel UMAP k=15, min_dist=0.1, spread=1, seed=0. Parameter review supports stable division and weak global linear SHM association, with local SHM structure retained. No temporal direction, future fate or method superiority is assigned. |
| 3 | Clone-state variation in PcAS infection. A: compact design; B: enlarged early PB map; C: central, largest late GC map (1,183 families; k=50, min_dist=0.5, spread=1, seed=0); D: later cell context. E: cycling variation among 329 pure-PB families, negative within-mouse direction in 10/10 eligible mice. F: per-mouse GC-enriched/reliable family fraction, 36 infected mice and 1,174 families, separated by treatment. No cross-day family tracking or GC-origin inference. Isotype-controlled co-occupancy results, including no consistent positive excess, remain in text/source analyses. |
| 4 | LARRY: author cell SPRING context, Threadfin clone-day map, and day-2 to day-6 held-out-barcode prediction against RNA means, RNA means plus variances and capture-count controls. B is not a Clonotrace output. |
| 5 | Ten public NSCLC patients: exact paired-TCR identity, clone-cycle distributions and patient-level same/null temporal distances, plus descriptive RNA-module changes. No inferred response labels. |
| 6 | Matched native reporter prediction, four measured endpoints at the primary threshold, and disjoint-cell LARRY shrinkage error. Predictive accuracy and sampling denoising are separate claims. |

| Supplementary figure | Evidence |
|---|---|
| 1 | Capture depth, reliable-profile coverage, within-library clonal coherence and the sparse direct GC/output-sort limitation. |
| 2 | Reporter cross-gate examples, measured RBD binding/DZ fractions, retained clone states and within-library reporter associations. |
| 3 | PcAS treatment, rare GC/memory-like candidates, exact heavy/light matching, denominators and inference limits. |
| 4 | Human state composition, verified marrow donor/library mapping, descriptive module analysis and established within-family sequence/state null. |
| 5 | Revised marrow/blood validation: cell and family maps, pure-gate examples, exact productive heavy/light receptor identities across verified donors. |
| 6 | Author-clone/Threadfin crosswalk, donor programme binding coverage, individual-donor GC SHM curves and descriptive expression signatures. |
| 7 | Native benchmark family-size sensitivity, independent RBD-probe/GC-zone gate readout, measured execution stages and coverage/fold audits. |
| 8 | Original Top2a notebook output (exact cohort/day unresolved), reconstructed GSE246382 measured cell gates, new clone-averaged Myc and marker RNA by clone cluster, and measured compartment composition per cluster for the same 377 families as Figure 2B. Public GSE180920 day7/day14 inputs are verified, but its saved image input chain is incomplete. |
| 9 | Human GC repeated capture, binding labels and donor-weighted SHM. |
| 10 | Cell context for five exploratory non-GC analyses. |
| 11 | RBD measured/RNA annotations and UMAP sensitivity; local statistic is EV, not R². |
| 12 | Later-infection day, state and RNA context; terminal samples do not track individual families. |
| 13 | GC/PB/Memory fractions on identical Figure 3C coordinates, with per-mouse time summaries. |
| 14 | Pure marrow/blood receptor identity and non-GC coverage. |
| 15 | Capture coverage, clonal expression signal and RNA modules across twelve analyses. |
| 16 | RBD capture selection, within-mouse partial associations and matched family/gate SHM differences. |

## Reading the concepts and maps

A cell UMAP has one dot per captured cell. A clone UMAP has one dot per
receptor-defined family with an interpretable expression distribution.
Neighbouring family dots represent similar captured state distributions;
they do not establish ancestry between families. A same-donor sequence relation
supports within-family membership. Reporter/FACS/probe labels supply external
measurements, whereas gene signatures reuse the expression data.

Figure 1 is one continuous composition, without separate A/B/C panels or
lettered example glyphs. The compact RNA cloud has three state clusters;
three separately rooted, differently coloured receptor trees depict within-family
sequence relationships. These are neither measured phylogenies nor a one-to-one
mapping between RNA clusters and receptor families.
The v4 package calls families from sequence within donors and then summarises
context-adjusted receptor-excluded RNA distributions with a kernel profile and
reliability. The approved cartoon fish PNG replaces the integration box:
the two input paths join its incoming filament, while two outgoing filaments
connect the branched and continuous maps. Its belly retains two density peaks and labels the clone
embedding and feature-space Fréchet mean. This mean label refers specifically to
the unshrunk averaging stage, not the complete adjusted/shrunken profile or an
optimal-transport solver. The generated cartoon is embedded unchanged from
`assets/threadfin_fish_cartoon.png`, at its native aspect ratio and with alpha
transparency. The user supplied the reference fish and approved this generated
illustration. Tapered vector ribbons extend the existing fin tips, with matching
pale gold/blue-green shading and smoothly joined curves. The input fork and
both output connections use the same fin treatment. The asset checksum and
placement are recorded in `figure_audit.json`.
The workflow does not concatenate two
UMAPs or learn a BCR-sequence encoder. All positions and state fractions are
illustrative; a family-map point represents one family and nearby points have
similar captured profiles. A gold outline highlights the selection node; it is
an annotation, not a confidence region. The only biological direction arrow is the dashed,
explicitly hypothetical return from selection to GC cycling. Direction needs
additional evidence and does not establish memory re-entry. All clone points
are circles; no shape encodes a sampling date. Large central dots are a design
choice illustrating expansion, not a general rule relating UMAP position to
clone size. The lower map's vivid blue–teal–amber–rose gradient encodes hypothetical
GC/cycling versus output-associated programmes, with no imposed selection node
or time direction. Figure 2's model-antigen and Figure 3's infection questions
motivate separate conceptual geometries, not a pooled analysis or a common
inferred lineage. Figure 2B's historical centroid reconstruction remains a
separate method. Figure 1 scientific scope is recorded in its legend rather
than footer paragraphs. Original membrane-receptor and V(D)J vectors draw on
general motifs in Dandelion and scRepertoire 2 Figure 1; source links are in
the figure audit and the Chinese reading guide, with no article assets reused.

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
reuse saved coordinates. The current [Figure 2–3 review](../FIGURE_2_3_REVIEW_2026-10-07_zh.md)
records source corrections, parameter selection and inference limits.
[The RBD three-setting comparison](../../case_studies/results/mouse_rbd_embedding_audit/review/parameter_comparison.png)
uses measured division, DZ and SHM readouts, with exact parameters and input hashes in its review directory.
Figure 3's implementation is `figure3_infection.py`; its normal entry point is
`python paper/figure_plan/make_figures.py --figures 2 3`. E/F source tables are
refreshed in `case_studies/results/figure23_review/`, and the selected C coordinate
hash and layout dimensions are recorded in `figure_audit.json`. The [Figure 3 review](review/Figure_3_map_reproduction.png)
and [analysis audit](../../case_studies/results/map_reproduction_audit/REANALYSIS_zh.md)
explain why swapping early/PB for late/GC changed appearance despite identical
coordinates. Dense cell clouds are rasterised in PDFs. Figure 1 embeds the approved
fish PNG; its scientific diagrams, external labels and connectors remain vector
artwork. The earlier supplied GC raster remains archived
as an asset and is not embedded in the redesigned Figure 1. S8A embeds the unmodified historical Top2a output. Figure 2B
and S8B–D use audited current GSE246382 data; their parameters and raw/source
hashes accompany [the reconstruction audit](../../case_studies/results/gc_legacy_parameter_audit/RECONSTRUCTION_REVIEW_zh.md).
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

### Reproduce the current Figure 1

`python paper/figure_plan/make_figures.py --figures 1` overwrites the canonical
`Figure_1.png` and `Figure_1.pdf`. The full `make_figures.py` and
`make_biology_figures.py` workflows call the same `figure1()` implementation.
The artwork is implemented in `figure1_overview.py`; source and conceptual
membership details are recorded in `figure_audit.json`. No alternative layout
folder is required. The approved fish asset at `assets/threadfin_fish_cartoon.png`
is loaded by the normal workflow.
