# Threadfin links B-cell receptor families to germinal-centre cell-state distributions


Chen Zhu

Department of Microbiology and Immunology, Peter Doherty Institute for Infection and Immunity, The University of Melbourne, Melbourne, Australia

Draft 3, 6 October 2026. Additional authors, corresponding-author details, funding and declarations require author confirmation. This draft follows the formatting and numbered-reference conventions of the supplied progress-review document.

## Abstract

Germinal-centre B cells repeatedly alternate between selection and division,
while related cells may occupy different expression states. Paired single-cell
RNA and B-cell receptor sequencing records both receptor similarity and
current state, but these measurements need not describe the same biology.
Threadfin compares the distributions of states occupied by BCR-defined clones,
allowing clone-level biases to be interpreted without equating an expression
map with a lineage tree. Real-data reclustering of 49 same-mouse NP-OVA families
distinguishes Myc-positive light-zone-associated GC state bias from
plasma-cell output-associated bias. Measured gates and RNA markers anchor
the interpretation without assigning future fate. In NP-OVA and RBD division-reporter experiments,
clone profiles vary continuously and are associated most strongly with measured
division history; BCR mutation load provides different information. In
*Plasmodium* infection, same-mouse clones co-occupy author-annotated GC,
plasmablast and memory-like states. We evaluate that co-occupancy against
mouse-, clone-size- and isotype-preserving controls and conservative exact
heavy/light receptor matching. Shared-state families are observed, but GC–PB
co-occupancy does not show consistent enrichment beyond isotype-preserving
expectations. A longitudinal human vaccination cohort adds
repeat GC-containing clone sampling, while one marrow/blood model detects
identical heavy/light receptors across pure plasma-cell and memory-cell sorts.
These analyses distinguish clonal membership, phenotypic breadth and state
persistence. They prioritise biologically interpretable clone relationships;
future cell fate and memory GC re-entry remain hypotheses requiring direct
experimental testing. The selection/output interpretation is restricted to
model-antigen GC responses and is not transferred to non-GC samples.

## Introduction

The GC reaction couples receptor evolution to changing B-cell states. A selected
light-zone cell can enter a proliferative burst, whereas cells associated with
antibody-secreting and memory outputs express different programmes. Experimental
reporters and direct sorting resolve parts of this process {cite:reporter,output_sort,gc_permissive_selection}. An expression
snapshot alone, however, does not establish whether a cell will divide again,
leave the GC or participate in a later recall response.

A BCR-defined clone provides an additional unit of observation: a family of
cells whose receptor sequences support common ancestry within one individual.
Its captured cells may favour a particular state or span several states. This
family-wide distribution cannot always be represented by a single average cell.
It also separates two relationships that are often confused: related cells
within a BCR family, and distinct families with similar expression profiles.

We developed Threadfin to ask which states clones occupy, what independently
measured properties explain those distributions, and what persists when the
same clone is sampled again. The expression embedding excludes receptor genes;
clone membership is established separately from sequence. A clone UMAP
therefore visualises similarities between captured state distributions rather
than inferring ancestors or descendants from transcriptional proximity.

Existing tools already link repertoire information to expression. Benisse and
BiGCN learn joint representations; clone2vec summarises whole-clone variation
from expression neighbourhoods, including sparsely sampled clones
{cite:benisse,bigcn,clone2vec_preprint}. CoMBCR also co-learns paired receptor and
expression representations for cell-level functional tasks {cite:combcr}.
Repertoire workflows such as Dandelion,
Scirpy and Platypus provide complementary annotation and joint analysis
{cite:dandelion,scirpy,platypus}. Threadfin therefore focuses on an auditable GC
workflow: sequence-defined membership, sampling-adjusted state profiles,
profile reliability, conditional controls and measurements supplied by the
original experiments. A clone embedding alone is not the claimed contribution.

Here we concentrate on datasets with strong biological anchors: two GC
reporter cohorts from a model-antigen/vaccine study {cite:reporter}, a direct GC/plasma-cell
sort study {cite:output_sort}, a time-resolved *Plasmodium* infection with an anti-malarial
intervention {cite:malaria}, and repeated human GC sampling after mRNA vaccination {cite:human_gc}.
One non-GC marrow/blood study checks receptor identity across terminal-state
sorts {cite:marrow}. Other tested datasets document coverage rather than carrying the
main biological claims. This organisation distinguishes an interpretable
association from a demonstrated fate mechanism.

## Results

### Clone interpretation requires separate receptor, state and sampling records

Figure 1 defines the biological questions. Sequence similarity within a donor
supports clonal membership. Expression identifies a captured cell's state.
A clone profile describes the distribution of those states relative to the
cells sampled beside that clone, with reliability determined by captured size
and estimated within-clone resemblance. Similar clone profiles indicate similar
behaviour of different families, not a common ancestor of those families.

This distinction changes the reading of a clone map. A cluster of
GC-dominated profiles is a set of families with related GC-state distributions;
it does not identify a lineage that will remain in the GC. A family with both
GC and plasmablast cells demonstrates co-observed states within a sequence-defined
relationship; it does not identify which cell produced the other. A reliable
state-retention estimate requires separate snapshots of the same donor's
family. We use different experiments for these different questions.

The three panels of Figure 1 provide a continuous reading of those records.
Panel A sets the GC selection, expansion and output compartments. Panel B
follows the same illustrative cells: colour identifies cell state, receptor
sequence defines family membership, and a family profile retains the mixture
of states occupied by its members. One point in the clone map represents one
family; neighbouring points have similar captured RNA-state distributions.
The state-composition rings are teaching diagrams, rather than the numerical
kernel features. Panel C connects four possible uses to their experimental
anchors: division reporters, same-mouse GC/output co-occupancy, repeated
same-donor capture and a family-label shuffle control.

Profile reliability prevents very small repertoires from yielding unsupported
programme claims. The direct NP-OVA GC/plasma-cell dataset contains 884
captured cells, 762 with a called receptor, and 388 cells in 113 expanded
clones. Only three profiles reach reliability 0.5. Its coherence test is not
significant (p=0.224), and programme inference is declined. The direct sort
labels remain useful for descriptive clone examples, without validating a
future-fate model (Supplementary Figure 1). The centroid-based exploration shown next uses 49 of these same-mouse
families with at least three captured cells. Descriptive state bias in that
map is kept separate from reliability-filtered programme inference.

### In controlled GC models, clone reclustering distinguishes captured GC and output state biases

The independent day-14 NP-OVA GC/output-sort dataset, GSE246382, samples
light-zone, dark-zone, Myc-positive light-zone and plasma-cell compartments
{cite:output_sort}. Figure 2B now uses a new real-data centroid analysis rather
than reproducing the coordinates in the author's earlier report. Raw counts
are aligned to the frozen donor-restricted, sequence-defined family calls.
One point represents one of 49 families with at least three cells, covering
260 captured cells across the original 11-mouse experiment. A receptor-excluded
PCA supplies the centroids, and the continuous preset (20 neighbours,
min_dist 0.4, resolution 0.1, seed 123) is fixed before inspecting gate labels
or markers. The three presets and two seeds are retained for review.

The map distinguishes regions biased towards Myc-positive LZ capture and
plasma-cell capture, with LZ/DZ-associated families distributed through the
GC region. Eleven families are predominantly Myc-positive LZ, six predominantly
PC, 25 predominantly LZ and three predominantly DZ; four have tied predominant
gates and are shown as mixed. Clone-averaged Myc is 1.363 in the Myc-positive-LZ
predominant group versus 0.290 in the LZ group, in log-normalised RNA units.
The PC-predominant group has higher plasma-cell-module expression than the
GC-predominant groups (Supplementary Figure 8B–D). These observations anchor
a selection-associated GC state node and a contrasting captured output state.
They do not show that one clone traversed a selection junction and subsequently
took one of two fates. The primary Leiden graph gives one partition;
the discrete preset gives two, but neither constitutes validated programme
inference in this sparsely sampled dataset. The same receptor families show
consistent compartment/marker biases across differently arranged preset maps.
This model-antigen GC application is not used to assign selection nodes or
GC output routes in non-GC data.

The original notebook also preserves a Top2a-coloured clone map
(Supplementary Figure 8A). Public GSE180920 day-7/day-14 count and metadata
files were retrieved and checked, and the notebook explicitly contains a
concatenation of those two timepoints. Nevertheless, its non-sequential
interactive history does not bind this saved image to a complete input chain;
its exact cohort/day remains unresolved. It is shown as a separate historical
output and excluded from GSE246382 biological conclusions. The archived
report fork remains an accessible reference, not the source of the new
Figure 2B points or marker values.

The model-antigen reporter study provides independent measurements of what a
GC cell recently did {cite:reporter}. The NP-OVA cohort uses a 36-hour
H2B-mCherry dilution window before day-14 lymph-node collection. The two RBD
vaccine cohorts add antigen-probe binding or light/dark-zone gates. These are
separately sequenced sort libraries and independent protein/mRNA cohorts
(Figure 2A). Reporter maps compare the fraction captured in the mCherry-low
gate with V-region mutation frequency on the same saved family coordinates
(Figure 2C–G); their coordinates are not aligned to the independent
GSE246382 map.

Clone identity explains 9.3% and 14.5% of expression variation in the NP-OVA
and RBD datasets, compared with within-library shuffled values of 3.5% and
4.6% (Supplementary Figure 1). Both cohorts lack a significant discrete
programme split at the tested resolution: the result supports continuous
profile variation, without proving that all GC reactions must be continuous.
The centroid view and these context-adjusted profiles answer different
descriptive questions and use different clone representations.

Measured division history accounts for 17.6% and 8.0% of profile variation in
the two cohorts. The association survives stricter within-library comparisons
(16.0% in NP-OVA; 11.8% and 11.6% in the independent RBD protein and mRNA arms).
Dark-zone occupancy adds information, whereas antigen binding and mutation
frequency explain less of the global clone-profile variation. The interpretation
is a bias towards division-associated states, not a prediction that every
member of the family will divide or return to selection.

Related cells can occupy both division gates or both GC zones
(Supplementary Figure 2). In RBD experiments, clone-state retention is 0.47
across division gates, 0.49 across probe-binding status and 0.53 across LZ/DZ
sorts. These are cross-gate comparisons of the captured family; they do not
observe a round trip of one cell through the selection cycle.

The sequence record also differs from current state within a clone. Previously
completed, independently constructed mutation-tree analyses found no detectable
within-clone relationship between SHM distance and expression distance in the
two adequately powered reporter cohorts (923 tested clones). That null is
restricted to those datasets and the calibrated effects detectable there;
smaller GC/output datasets cannot exclude modest relationships. Clone-wide
bias and within-family mutation history should therefore be reported separately.

### *Plasmodium* reveals same-mouse GC/output-state relationships, not cross-day cell trajectories

The infection study provides biological complexity and an intervention {cite:malaria}.
Early samples cover the splenic B-cell landscape. Later samples enrich
IgD-low B cells and include an IgD-high naive spike-in, with separate infected
saline, anti-malarial and uninfected mice (Figure 3A). The experiments overlap
at days 10 and 14 but remain distinct cohorts and sampling regimes. Each mouse
is sampled terminally; there is no longitudinal tracking of an individual
mouse's clone from an early to a late day.

Author annotations identify GC, plasmablast (PB) and memory-like expression
states. Threadfin relates those labels to BCR families within each day and
mouse. Across early samples, 39 families contain both GC and PB cells, two
contain GC and memory-like cells, and 14 contain memory-like and PB cells.
Across the later experiment, the corresponding family counts are 145, 33 and
22. These are observed co-occupancies, not measured lineage transitions.
The very sparse early GC/memory-like overlap limits claims about early-memory
relationships to the GC.

Conservative sequence matching asks how much of this observation depends on
merging similar heavy-chain sequences into one family. Exact heavy/light
receptor groups observed in both states provide an identity-consistency check;
the control requires unambiguous productive heavy and light chains and does
not allow a missing light chain to stand in for a match. Exact matching loses
legitimate SHM-diverged relatives and is consequently a sensitivity analysis,
not a gold-standard definition of every GC family (Supplementary Figure 3).

The primary co-occupancy statistic uses all families with at least two captured
BCR-bearing cells as a fixed denominator. Mouse-specific label permutations
preserve family size and state totals; a second null also preserves totals
within each observed heavy-chain isotype. This tests whether family membership
adds state-association information beyond those measured sampling properties.
GC-bearing conditional fractions and a three-cell minimum are secondary
sensitivity analyses. Mice, rather than their cells, supply biological replicate
estimates. Figure 3F shows per-mouse deviations from the isotype-preserving
null with a fixed all-family denominator; its evidence is association, without assigning an output direction.

The choice of sampling control changes the biological reading. In the late
saline arm, 132 of 868 expanded families jointly contain GC and PB cells.
The equal-mouse mean observed rate is 11.91%, compared with 6.48% under a
mouse-only null and 17.37% when the null also preserves per-cell isotype.
Thus the apparent positive deviation from the mouse-only expectation reverses
after conditioning on isotype composition. Early infection likewise has an
equal-mouse observed rate of 3.86% versus 8.82% under the isotype-preserving
null. The late drug arm has 4.48% observed versus 5.73% expected. These are
descriptive mouse-averaged comparisons, not a pooled treatment test. They do
not establish consistent excess GC–PB mixing beyond measured isotype
composition. Instead, most sampled families favour separated states while a
minority supply receptor-linked cross-state candidates.

Those candidates include exact paired-heavy/light groups: 11 early and 41
late groups share GC and PB states; zero early and eight late groups share
GC and memory-like states. They demonstrate that some observed sharing
survives stringent identity matching. Counts under the two definitions have
different eligible denominators and are not compared as equivalent detection
rates. Exact identity strengthens selected membership examples, without
turning those examples into directed transitions or proving an enrichment.

This analysis adds a receptor-linked view to the authors' temporal cell-state
map. Clone maps and expression signatures alone could recapitulate annotation;
the additional question is whether particular pairs of states occur in the
same sequence-defined family, survive stricter sequence matching and exceed
an appropriately constrained sampling expectation. Only the latter tests add
evidence about clonal organisation. None establishes memory GC re-entry.

The treatment arm is shown by day and mouse in Supplementary Figure 3.
Differences are interpreted within the late experiment, with three infected
mice per arm/day. Uninfected controls and naive spike-in cells are not treated
as additional infected replicates. The dataset does not supply clone-specific
future outcomes or affinity measurements; it prioritises families for such
experiments rather than replacing them.

### Repeated human GC sampling tests clone persistence over a longer response

The vaccination cohort repeatedly samples draining lymph nodes and blood from
the same participants {cite:human_gc}. This differs from the terminal Plasmodium design:
a BCR family can genuinely be observed at multiple dates in one person.
Figure 4 shows the study design, captured GC/output compartments and an
independent spike-positive receptor label on the clone map.

GC-dominated and antibody-secreting profiles are distinct in this dataset.
The restored binding-enrichment plot compares author-identified Spike-positive
families within expression programmes against other families from the same donor
(Figure 4D). It reports binding-label odds rather than quantitative affinity.
The sequence crosswalk maps every Threadfin family to one author clone, with
623 author clones subdivided into multiple Threadfin families. Labels propagated
into such subdivisions are not new independent binding experiments
(Supplementary Figure 6). Two of four candidate programmes meet bootstrap
stability 0.75; programme names are not treated as guaranteed discrete fates.

Individual families with GC cells at three or more non-pooled dates illustrate
repeated GC-containing membership. We select these examples by measured
presence and captured size, not by a visually interesting UMAP location.
Bars display the recorded cells, including dates without captured members;
absence in a sample is not proof of biological extinction.

The existing clone-state retention estimate is 0.26 across dates (420 families;
95% interval 0.20–0.31), compared with approximately zero across LN/blood
snapshots (159 families). This shows that family membership and current
compartment are different pieces of information. It does not establish whether
a memory cell entered a later GC or whether a particular GC cell produced a
blood plasmablast. A study with prime/boost and fate mapping can address those
directions directly {cite:memory_reentry}.

The GC mutation trend is also shown explicitly (Figure 4G). We average within
family/date SHM, take a median across families for each donor/label, and give each
donor equal weight. Only donors represented in both label groups at that date
contribute (1, 1, 8, 6 and 4 donors at the five dates, respectively); the shaded
range is the donor interquartile range. The first two points therefore have no
across-donor replication. Later samples
have different captured families and sometimes different donors, so the curve
is not a paired estimate of maturation within the same family. The comparison
group is not identified as Spike-positive, rather than uniformly assayed
negative. The source paper includes functional antibody experiments, but our
current paired-cell input does not supply a quantitative affinity for every
Threadfin family. Neither binary binding labels nor SHM estimate affinity.

### Receptor-defined families explain expression organisation across tested models

Across twelve dataset analyses, clone identity is associated with more expression
variation than the 95th percentile of a within-library shuffled baseline in eleven
analyses (Figure 5B). The direct NP–OVA PC/GC dataset is the exception
(p=0.224). These are separate dataset analyses, not twelve independent studies:
the two reporter cohorts share a publication, and the two infection datasets
are separate experiments from one study. The comparison excludes receptor genes
from the expression embedding and preserves library composition and family sizes
under shuffle. It measures captured clonal expression organisation, rather than
antigen specificity, future fate or a benefit over another integration method.

The biological interpretation depends on the experimental anchor and sampled
repertoire (Figure 5A,C). Only 3 of 113 expanded families in the direct PC/GC
study reach profile reliability 0.5, compared with 69 of 373 NP reporter families,
381 of 1,414 RBD reporter families and 4,429 of 8,435 human vaccination families.
Library-stratified signal does not guarantee that individual profiles are
informative enough for programme inference. Clonal resemblance of GC, cycling,
light-zone, antibody-secreting, memory-associated and interferon modules varies
between models (Figure 5D). Those modules reuse the profile-building expression
matrix; expression-matched gene backgrounds aid interpretation but do not turn
module agreement into independent biological validation.

### Captured-state readout differs from repertoire-network refinement

Published tools address several related tasks: repertoire annotation, joint
sequence/expression representation, clonotype networks, CSR dynamics and whole-clone
state descriptions. We record their native capabilities and published validation
separately from measured performance. CoNGA, Ibex, scRepertoire and sciCSR
address distinct receptor/expression or class-switch tasks
{cite:conga,ibex,screpertoire2,scicsr}. We avoid assigning zero to an inapplicable
output (Figure 6; [source comparison](METHOD_COMPARISON.md)). Benisse {cite:benisse}, BiGCN {cite:bigcn}
and clone2vec {cite:clone2vec_preprint} are run through their official models; encoder-only outputs or
external proxy implementations are not substituted for these methods.

The controlled readout uses fixed donor-private families and a common
receptor-excluded PCA100 input. Seven representations are compared with a
training-mean control, retaining raw and donor-centred RNA centroids as essential
baselines. Representations are unsupervised and transductive; whole-mouse folds
withhold the measured labels only from the downstream ridge model. The endpoint
is the fraction of captured family cells in the measured mCherry-low gate,
not future fate, clonal ancestry or an affinity value.

In the completed NP–OVA comparison, all methods represent the same 373 expanded
families across seven mice. Median mouse-wise absolute error in gate fraction
is 0.197 for Threadfin mean profiles and 0.207 for kernel profiles, versus 0.314
for the training-mean control. Raw and donor-centred RNA centroids achieve 0.161
and 0.165, respectively; clone2vec, BiGCN and Benisse achieve 0.238, 0.291 and
0.314. Thus this readout does not demonstrate that Threadfin is the most accurate
representation. It tests one captured-state summary; profile reliability,
conditional controls and repeated-state comparisons are distinct outputs.
The size-threshold sensitivity retains at least five measured cells per family
(Supplementary Figure 7).

{RBD_BENCHMARK_RESULTS}

The reporter gate is associated with physical sequencing library in these
experiments. Whole-mouse label holdout retains shared libraries and a
transductive representation; the comparison cannot separate reporter biology
from all library effects or establish independent-library generalisation.

### Non-GC tests distinguish receptor identity from terminal-state annotation

Outside the GC-focused argument, the marrow/blood model {cite:marrow} supplies the
clearest measured-gate check (Supplementary Figure 5). A GEO manifest corrects
donor identities and retains 15 single-donor, single-tissue libraries from seven
donors, excluding five pooled or mixed libraries. The analysis contains 115,144
cells, 78,883 paired receptor records and 9,783 expanded families. Within pure
marrow plasma-cell and memory-cell gates, unique productive exact heavy/light
receptors are shared in 225/862 eligible repeated-receptor groups for donor 1681
and 31/677 for donor 1684; marrow PC–blood memory sharing is 124/1,012 and
115/934. These are identities across captured gates, without implying
transdifferentiation or a direction of differentiation. Influenza blood, influenza
lung, EBV blood, tonsil and COVID-19 blood data provide additional tested coverage
(1,997, 114, 6,923, 670 and 83 expanded families, respectively). Their clonal
expression signal is included in Figure 5, whereas individual maps remain in the
separate tested-dataset folder. Exploratory transcriptomic states and variable
sampling designs in those datasets do not establish independent fate mechanisms.
Thus one non-GC anchor supports interpretation of common receptor identity,
while the remaining analyses document where the workflow has been tested.

## Discussion

Threadfin's useful biological output is a description of clone-level state
bias and breadth, anchored to separately established sequence membership and
experimental labels. The reporter models show that expression distributions
record division-associated GC state beyond a family-average receptor history.
The infection model asks which GC/output-like states are co-observed in the
same mouse's families. The human cohort adds repeat GC membership, while the
single non-GC validation shows that common receptor identity can coexist with
distinct measured terminal-state gates.

These are different strengths of inference. Co-observed related cells support
shared clonal organisation. Repeated same-donor capture supports persistence.
Neither is equivalent to an observed parent–offspring relationship. Memory
re-entry into GC requires evidence of prior memory identity and later GC
participation; direct fate mapping and recall experiments illustrate the
necessary design {cite:memory_reentry}. Plasmodium samples from different terminal mice cannot
supply that history, however persuasive an embedding may appear.

The package complements receptor annotation, SHM phylogenies and cell-state
analysis. Sequence-only analysis cannot describe transcriptional breadth when
several family members have identical sequences; expression-only analysis
cannot establish a family link. This complementarity is meaningful without
claiming that no other combined workflow could calculate an occupancy table.
A claim of superior future-fate prediction would require a held-out outcome,
matched competitors and an independent validation cohort; it is not made here.

The closest whole-clone embedding comparator, clone2vec, already supports
continuous clone descriptions and clone-associated gene analysis
{cite:clone2vec_preprint}. The incremental contribution claimed here is the
combination of an explicit receptor-family definition, context-adjusted
profiles with reliability, matched sampling controls and GC experiments with
measured biological anchors. This is distinct from claiming a universally
superior representation; the completed NP reporter test favours ordinary RNA
family means on its selected endpoint. Additional independent experiments would
be needed to establish predictive advantages of the full state-distribution
representation.

The model-antigen GC map provides a concrete exploratory application:
clone reclustering organises Myc-positive-LZ-associated and plasma-cell-associated
family state biases using the same measured compartments as biological anchors.
Its value is in defining selection/output hypotheses that can be tested with
lineage tracing or independently measured output. The earlier report's fork
provided the motivation, but the present acceptance criterion is state
interpretability, not reproduction of that silhouette. Neither a chosen
UMAP preset nor a marker heatmap establishes two actual fates. The current
same-mouse clone calls prevent receptor-gene reuse across mice from creating
artificially large families, and the limited reliable-profile coverage remains
explicit. This GC-specific interpretation does not apply to the marrow/blood
non-GC validation, which tests receptor identity across measured terminal-state
gates.

Several limitations constrain the biological interpretation. Receptor recovery
is incomplete and may depend on cell state. Sparse captured families understate
phenotypic breadth. Heavy-chain similarity can merge unrelated receptors;
paired-light and exact-sequence controls reduce one source of uncertainty but
lose SHM-diverged family members. Alternative family callers and lineage
models address different assumptions {cite:partis,family_inference_evaluation,igphyml,tribal}.
Large-scale paired-chain data document chain-mixed groups and naive-like
pseudo-clonal clusters under heavy-chain-based inference {cite:paired_family_bias};
exact paired-chain controls cannot establish the sensitivity or specificity of
every primary family call. Pre-sorting constrains which state combinations
are observable. Author memory-like annotations are not functional recall
measurements. Expression signatures provide interpretation rather than external
validation. Antigen specificity and affinity cannot be inferred from generic
GC/PB/memory programmes.

The next decisive experiment is to test candidates ranked by robust
same-family GC/output co-occupancy or GC state bias against independently
measured antigen binding, recall or lineage tracing. The public-data results
provide a reproducible way to choose those candidates and a precise account
of what remains unknown.

## Methods

### Data and sampling design

Raw paired RNA/BCR files were obtained from GSE287123, GSE246382, GSE286215,
GSE195673/Zenodo 5895181 and GSE253857. Loader metadata records donor, library,
measured gates, time and tissue. The marrow mapping is committed in
`case_studies/bone_marrow_sample_manifest.tsv`; pooled and mixed libraries are
excluded by default. Plasmodium experiments are kept separate and `(day,
HTO mouse)` defines the biological individual. Treatment comes directly from
the deposited late-experiment metadata. The early d0 control is one naive
mouse incorporated in the d4 HTO pool; d4 therefore has four infected mice.
The late experiment has three treated, three saline and one naive mouse per
sampling date. The source study's anti-malarial regimen includes artesunate and
pyrimethamine; the deposited label `Artesunate` is retained in source tables.

### Clone membership and expression profiles

The case-study pipeline selects productive heavy chains and defines families
within actual donors using V/J gene calls, junction length and normalised
junction Hamming-distance single linkage. The automatic threshold is a density
valley when identifiable, with an explicit fallback. This is a sequence-defined
relationship and does not use expression proximity. Primary family calls use
heavy chains; conservative exact heavy/light analyses are reported separately.

Counts are normalised and log-transformed. Immunoglobulin V/D/J and constant
region genes are excluded from highly variable gene selection before the
30-component expression embedding. Harmony is applied only when the case
configuration specifies a batch key. The package describes a family's captured
expression distribution using random Fourier features, centres it against
its sampling context and estimates profile reliability from captured size and
variance components. Figure maps reuse saved coordinates. Their two-dimensional
distance is not used as evidence of ancestry or a future fate.

### Model-antigen centroid reclustering and historical provenance

`case_studies/run_gc_reclustering.py` aligns the GSE246382 MTX/features/barcodes
and deposited mouse/compartment metadata to the frozen 884-cell table and its
same-mouse sequence-defined family calls. Cell identities, mouse labels,
compartments and the original at-least-500-expressed-gene QC are checked.
Raw counts and source-table hashes accompany the results. Threadfin constructs
a receptor-excluded 3,000-HVG, 30-component PCA from log-normalised counts;
no gate labels, Myc values or signatures enter the embedding or graph.
Centroids retain families with at least three cells. Figure 2B uses the
continuous preset and seed 123, fixed before gate inspection. All three
presets and seeds 123/7 are saved, with requested/effective parameters and
package versions. New clone coordinates are computed; Supplementary Figure
8B reuses the previously saved cell coordinates with actual measured gates.

`tf.clonotype_recluster` provides cohesive (20 neighbours, resolution 0.3,
min_dist 0.1), continuous (20, 0.1, 0.4) and discrete (10, 0.8, 0.05)
starting configurations. All use spread 1 and learning_rate 1. Explicit values
override the preset; neighbour counts cap at retained clone count minus one.
Leiden uses original clone distances; UMAP embeds them as precomputed.
UMAP display controls can be adjusted independently without changing the
partition. The optional distance_profiles display treats distance-matrix
rows as Euclidean features, as in the historical notebook, but is not used
in the primary figure. Presets are not learned biological classes and do not
force a desired topology. The default filter is at least three cells;
strictly greater than three requires min_clone_size=4. Centroid exploration
remains separate from v4 sampling-adjusted profiles and programme validation.

Plot colours describe the largest measured-compartment fraction per family;
ties are labelled mixed. Myc and marker genes are averaged in log-normalised
RNA across member cells, then averaged equally across families for the
compartment-group heatmap. GC-identity, cycling and plasma-cell modules are
means of per-gene cell-standardised log RNA, using the recorded covered genes.
These reuse the transcriptome and provide descriptive state interpretation;
they are not held-out fate outcomes. No plot arrows assign temporal direction.

Supplementary Figure 8A preserves notebook cell 33 output 0 from commit
57a9565 with all saved data points and legends. Source hashes and provenance
limits are recorded in `paper/figure_plan/assets/legacy_gc_provenance.json`.
GSE180920 has 3,542 day-7 and 19,757 day-14 deposited cells; count-column and
metadata-cell IDs are checked in order, and Top2a is present. Its notebook
contains explicit day-7/day-14 RNA and BCR concatenation, but non-sequential
execution and missing original processed objects/receptor calls prevent an
exact attribution of the saved Top2a plot. The older GSE246382 report panels
are retained separately as a historical reference. Their notebook uses
UMAP on distance-matrix rows followed by Scanpy neighbours of clone-UMAP
coordinates. Executed clone filtering (at least one cell) and Leiden
resolution (0.3) differ from report methods (>3 cells, 0.1), so they are not
represented as a verified present-package rerun.

### Reporter labels, programmes and retention

Measured reporter/probe/FACS labels are compared with profiles separately from
expression signatures. Within-library label checks address pre-sorted sampling
in the reporter cohorts. Programme splitting is tested against a single-group
model; bootstrap stability is reported rather than treating every clustering
as a stable biological category. Clone-state retention compares separate
same-family snapshots against same-donor random-family comparisons, correcting
sampling noise. Its intervals resample clones and do not replace donor-level
population uncertainty. Full existing algorithms are specified in
[`docs/METHODS.md`](../docs/METHODS.md).

Mouse module scores use capitalisation-adjusted gene symbols, with per-dataset
coverage recorded; this is not a curated, gene-by-gene orthologue analysis.
Coverage is recorded for each signature. Clone-averaged scores and programme
scores reuse normalised expression and are descriptive; their agreement with an
expression-derived programme is not counted as independent validation.

### Same-mouse state sharing

`case_studies/clone_state_sharing.py` records GC/PB/memory-like counts within
each captured family. The primary rate counts jointly occupied families among
all families with at least two captured BCR-bearing cells, fixing the denominator
under permutation. The secondary GC-containing rate conditions on observed
GC membership and is explicitly labelled. Three-cell sensitivity is reported.

For each mouse, 1,999 state-label permutations preserve family sizes and the
mouse's state totals. A second null permutes within per-cell heavy-chain isotype
strata. This controls measured isotype/state composition but not annotation
error, state-dependent receptor recovery or unmeasured biology. Per-mouse
estimates are retained; no cell-level replicate treatment test is used in the
main argument. Undefined conditional denominators remain missing and their
coverage is recorded, rather than replacing missing estimates with zero.

Exact controls require productive, unambiguous receptor calls. Families are
replaced by exact reported IGHV/IGHJ/junction groups within the mouse; the
paired-light variant also requires a unique exact light-chain signature. Exact
shared-group counts test identity consistency and are not assigned an uncomputed
sampling-null p value. The marrow pure-gate check separately requires a unique
productive heavy/light pair and compares only verified donor-matched pure gates.

### Software implementation and input validation

Threadfin is a Python package using AnnData and Scanpy for expression data and
preprocessing {cite:scanpy}; Harmony is available for a supplied batch variable
{cite:harmony}. Users may provide a precomputed cell embedding or construct one
from finite non-negative counts while excluding receptor genes. Input checks
report missing metadata columns, incomplete donor/sample grouping labels,
invalid matrix dimensions and incompatible BCR barcodes in English, with
concrete correction steps. Missing measured state, probe or time labels remain
missing observations rather than being converted to negative calls. Unit tests,
wheel-installation tests and documented tutorials accompany the implementation.
The tested environment and exact verification scope are recorded separately
from the biological results.

### Native-method comparison and held-out labels

The two reporter cohorts use the same paired cells and receptor-excluded,
library-size-normalised/log-transformed PCA100 input for all applicable methods.
The saved sequence-defined families are held fixed within actual mice. Raw and
donor-centred RNA family centroids distinguish context adjustment from profile
modelling. Threadfin mean/kernel profiles use donor context, a minimum of two
cells, and seed 0; the kernel uses 256 Fourier features, 30 components and its
15-neighbour default. clone2vec 0.1.1 uses the native clone-neighbour graph (k=15)
and Skip-Gram (30 dimensions, maximum 500 iterations, seed 0), without any
reporter-derived cell composition in the model input.

Benisse uses its pretrained encoder and full official R graph model with the
README configuration. Reversible cell aliases address R column-name conversion.
BiGCN uses its native exact V–CDR3AA–J nodes and three official processing/graph/
training scripts; two audited CLI fixes retain the requested node count and
dataset. Neither model's equations are replaced. Their native nodes are mapped
back to cells and then averaged with cell-count weights within the fixed families.
Only fully represented families enter the common intersection.

Benisse returns squared latent distances. A complete cell-weighted family kernel
retains that geometry for prediction; a 30-component MDS representation is saved
only for audit and is not used in the benchmark. Other representations contribute
their full native dot-product kernels. The same linear-kernel ridge readout uses
training-fold centring, training-trace normalisation and an intercept. Alpha is
selected among 0.1, 1, 10, 100 and 1,000 in nested whole-mouse training folds;
the outer mouse's labels are never used in that choice. All native unsupervised
representations nevertheless see its unlabelled cells, so this is a label-held-out
transductive readout, not an inductive unseen-mouse model.

Targets use only measured gate labels; unmeasured RBD/zone gates stay missing.
The primary threshold requires two measured cells and the sensitivity requires
five. Folds require at least 20 training and three test families. Errors are
reported per mouse with equal mouse weighting; predictions are unbounded ridge
estimates and not calibrated probabilities. Configuration, complete-family
coverage, ID hashes, label denominators, fold prevalence, model versions and
measured execution scope accompany the source tables. These are single native
model configurations; BiGCN's official entry point does not set a fixed seed.
Saved outputs reproduce the readout, but no repeated-training superiority or
end-to-end package speed ranking is claimed. Whole-mouse folds retain shared
sort libraries: mCherry gates coincide with physical library groups in these
inputs, so residual library effects can accompany the measured biology. This
is not an independent-library generalisation test, and captured gate fractions
are not population-level fate or division probabilities. The full protocol is in
[BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md).

### Figure selection and reproducibility

Clone examples are selected by co-observed labels and descending captured size,
with deterministic ID ordering, not embedding location. Repeated human GC
examples require at least three separate non-pooled dates with GC members.
The figure audit records selected IDs and source paths. The main and focused supplementary figures regenerate with
`python paper/figure_plan/make_biology_figures.py`; weaker exploratory datasets
are rendered into `paper/figure_plan/tested_datasets/`.

## Data and code availability

All analysed data are public through the accessions above. The reproducible
analysis, source tables, figure generators and draft manuscript are available at
[Threadfin](https://github.com/Reynold4k/Threadfin). Source-study experimental
findings are credited to their original publications. The new public-data
analysis does not introduce unpublished functional or lineage-tracing outcomes.

## Author contributions

To be confirmed by the final author group.

## Acknowledgements and funding

To be confirmed by the authors; the reuse of public data does not imply endorsement by the original study authors.

## Competing interests

To be confirmed by the authors.

## References

{REFERENCES}
