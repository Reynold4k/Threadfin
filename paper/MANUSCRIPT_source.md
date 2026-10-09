# Threadfin links B-cell receptor families to germinal-centre cell-state distributions


Chen Zhu

Department of Microbiology and Immunology, Peter Doherty Institute for Infection and Immunity, The University of Melbourne, Melbourne, Australia

Draft 5, 9 October 2026. Additional authors, corresponding-author details, funding and declarations require author confirmation. This draft follows the formatting and numbered-reference conventions of the supplied progress-review document.

## Abstract

Germinal-centre B cells can share a receptor family while occupying different expression states. Threadfin summarises these captured state distributions using receptor-excluded expression profiles, context adjustment and sampling-dependent shrinkage. Applications to model-antigen and Plasmodium responses distinguish clone membership, state occupancy and measured division history without equating an expression map with a lineage tree. We then test the framework beyond BCR-defined families. In lineage-barcoded mouse haematopoiesis, a separate day-2 expression model predicts later observed state composition for 172 eligible barcodes, with modest improvements over RNA means on some endpoints. Longitudinal profiles of 124,534 cells with unambiguous paired TCRs from ten patients retain same-clone state similarity relative to patient-, interval- and capture-matched comparisons. Independent reporter readouts favour simple RNA means over Threadfin on the division endpoint, whereas disjoint-cell validation shows that shrinkage reduces low-capture profile estimation error by 27% when two cells are sampled. These complementary tests support sampling-aware descriptions of clone-state distributions and identify where additional representation complexity helps. They do not establish BCR differentiation direction, complete developmental potential or clinical response prediction.

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

Figure 1 makes this integration explicit in one concept diagram. A compact
three-cluster scRNA-seq map describes cellular expression states, while membrane
BCR, paired heavy/light contigs and separately rooted receptor-family trees
illustrate the sequence information. RNA-cluster colours and receptor-family
colours encode different memberships; they are not matched one-to-one. Both
measurements connect through the extended fin filaments of a cartoon
threadfin fish, which represents integrated clone embeddings. Its double-peak
sketch illustrates RNA distributions; the Fréchet-mean label describes the
unshrunk mean of RNA feature vectors under squared feature-space distance.
The implemented profiles additionally use context adjustment, reliability-dependent
shrinkage and projection; this is not an optimal-transport solver.
Two further filaments connect to complementary patterns. The upper pseudo-UMAP links
cycling, selection-associated and output-like family states, motivating the
model-antigen GC analysis in Figure 2. A gold outline highlights the selection
node. Its dashed return to cycling marks a recycling hypothesis; snapshots
alone cannot distinguish progression from return. The lower, more linear map
uses a vivid blue–teal–amber–rose gradient to illustrate continuous
cycling-to-output profile variation, motivating
Figure 3's maturation questions without transferring a selection-node
interpretation to infection data. All clone dots are circles, with synthetic
captured-cell counts controlling their area. Larger central dots are a
compositional choice, not a rule connecting UMAP centrality to clone size.
Every coordinate, count and tree branch is illustrative: the studies are not
pooled, and no phylogeny or temporal trajectory is estimated by this diagram.
The default numerical profiles use RNA kernel features with context adjustment
and reliability-dependent shrinkage.

Profile reliability prevents very small repertoires from yielding unsupported
programme claims. The direct NP-OVA GC/plasma-cell dataset contains 884
captured cells, 762 with a called receptor, and 388 cells in 113 expanded
clones. Only three profiles reach reliability 0.5. Its coherence test is not
significant (p=0.224), and programme inference is declined. The direct sort
labels remain useful for descriptive clone examples, without validating a
future-fate model (Supplementary Figure 1). The centroid-based exploration shown next uses 377 donor-restricted
V–D–J receptor groups with at least one captured cell from the same cohort.
Descriptive state bias in that
map is kept separate from reliability-filtered programme inference.

### In controlled GC models, clone reclustering distinguishes captured GC and output state biases

The independent day-14 NP-OVA GC/output-sort dataset, GSE246382, samples
light-zone, dark-zone, Myc-positive light-zone and plasma-cell compartments
{cite:output_sort}. Figure 2B shows a newly calculated reconstruction of the
historical clone-embedding workflow. Its 377 points are donor-restricted
productive IGH V–D–J receptor groups, retaining at least one captured cell
and covering 762 receptor-called cells from 11 mice. This historical-style
receptor grouping differs from the 487 stricter same-mouse V/J/junction-sequence
families used by the main case-study pipeline: 83 of the 377 V–D–J groups
contain more than one distinct junction nucleotide sequence. Accordingly,
these points are receptor groups and are not assumed to be individual
sequence-defined clonal lineages.

The cell UMAP is reconstructed from the notebook code; group centres are
means of those cell coordinates, accumulated in float64 by the package. Euclidean UMAP embeds the rows of their
pairwise distance matrix (40 neighbours, min_dist 0.65, seed 123), followed
by a Scanpy graph on group coordinates (15 neighbours) and Leiden at
resolution 0.3 (seed 0). Colours denote clone_cluster and point area is
proportional to captured cell count. This display was chosen through an
exploratory comparison of inputs and layout parameters. It retains one
15-neighbour graph component and six Leiden groups across seven UMAP seeds
(0/1/7/42/99/123/2024), with median pairwise adjusted Rand index 0.877
(range minimum 0.701). The minimum local neighbourhood trustworthiness across those seeds is
approximately 0.994. It is not the most stable tested configuration: the
historical 20-neighbour/min_dist 0.4 donor-restricted V–D–J control has
median adjusted Rand index 0.989. All 164 computed maps and the seed
comparisons are retained; no cluster-count target was used as a success criterion.

Post-clustering annotation identifies captured GC and output-associated
states (Supplementary Figure 8B–E). Clusters 0 and 1 have mean group-level
LZ fractions of 65% and 74%, respectively, with DZ fractions of about 26%.
Cluster 3 has 80% Myc-positive-LZ capture and the highest group-averaged Myc
RNA (1.36 log-normalised units); cluster 2 has 97% plasma-cell capture and
higher Prdm1/Xbp1/Jchain expression. Cluster 5 mixes Myc-positive-LZ (50%)
and plasma-cell (29%) capture, while the larger receptor groups in cluster 4
collect cells across all four measured compartments. These observations
support descriptive GC-state annotations, including a selection-associated
Myc-positive-LZ region. The RNA markers reuse the expression measurements
and are not independent validation. Neither the two-dimensional connections
nor these annotations establish a temporal junction, two future fates or
co-occupancy by one verified sequence-defined clone. This exploratory map is
separate from v4 reliability-filtered programme inference and its non-significant
coherence result in this sparsely sampled dataset. Selection/output
interpretation is restricted to this model-antigen GC application.

The original notebook also preserves a Top2a-coloured clone map
(Supplementary Figure 8A). Public GSE180920 day-7/day-14 count and metadata
files were retrieved and checked, and the notebook explicitly contains a
concatenation of those two timepoints. Nevertheless, its non-sequential
interactive history does not bind this saved image to a complete input chain;
its exact cohort/day remains unresolved. It is shown as a separate historical
output and excluded from GSE246382 biological conclusions. The archived
report fork remains an accessible reference, not the source of the new
Figure 2B points or marker values.

The model-antigen reporter study, GSE287123, provides independent measurements
of recent GC-cell division history {cite:reporter}. The NP-OVA cohort uses a
36-hour H2B-mCherry dilution window before day-14 lymph-node collection. The
two RBD vaccine cohorts add antigen-probe binding or light/dark-zone gates.
These are separately sequenced sort libraries and independent protein/mRNA
cohorts (Figure 2A). The RBD cell map supplies measured-gate context
(Figure 2C); the corresponding family maps show mCherry-low fraction and
V-region mutation frequency on the same saved coordinates (Figure 2D,E).
These 381 reliable families use donor-private IGH junction sequence similarity,
rather than the broader V–D–J grouping used in the independent GSE246382
reconstruction. The kernel-profile UMAP retains 15 neighbours, min_dist 0.1,
spread 1 and seed 0; the two studies' coordinates are not aligned.

Reconstruction reproduced the RBD default coordinates to within 4.8×10⁻⁷.
Across 85 UMAP settings, the linear association of map coordinates with the
measured division fraction remained modest but stable (R²=0.142–0.170).
SHM had a weaker global linear association (R²=0.008–0.040), while local
association was still detectable under 70 of the 85 configurations at
nominal p≤0.05. A weak global SHM gradient is therefore not evidence that
mutation history is unrelated to expression state. These exploratory
donor-stratified permutation tests are not corrected for parameter selection
and do not resolve gate-specific library effects. The maps illustrate
biological associations, without establishing an advantage over simpler
family expression summaries.

The biological interpretation depends on the timescale of each measurement.
H2B-mCherry dilution reports recent division over the 36-hour observation
window, whereas V-region mutation frequency measures accumulated sequence
divergence. The original reporter study identified mutation-free expansion
and regulated mutation per division using additional sequence and experimental
evidence {cite:reporter}. Our two colour overlays do not estimate that rate.
They show that captured proliferative state and accumulated receptor history
provide partly distinct information, without identifying a new mechanism.

We tested this interpretation outside the two-dimensional map. Among all
1,414 expanded RBD families, median capture was three cells; the 381 families
displayed after reliability filtering had a median of eleven. Conditioning on
mouse, capture size and the additional binding/zone gate retained a positive
dark-zone/cycling RNA association with division fraction in nine of ten mice
(median partial rank correlation 0.320; Supplementary Figure 16). This module
reuses the RNA input and the association was weaker in the mRNA arm.
For 218 families with at least two cells in each division gate within the same
additional sort gate, we compared mean SHM between the low and high mCherry
fractions. The equal-mouse difference was +0.083 percentage points in the
protein arm and −0.050 percentage points in the mRNA arm, with descriptive
mouse-bootstrap intervals spanning zero in both arms. These data do not establish
absence of a relationship, but do not support a consistent monotonic SHM
increase with recent division. Intermediate mCherry cells were not sampled
in this comparison, so captured gate fractions are not in-vivo population
proportions. Division gate and sequencing library remain confounded.

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
across division gates, 0.53 across probe-binding status and 0.51 across LZ/DZ
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

The early-infection clone map (Figure 3B) shows PB-biased profiles, while
the enlarged later-infection map (Figure 3C) displays GC-, PB- and
memory-enriched family regions against the cell-state context in Figure 3D.
The later display retains the selected 50-neighbour, min_dist 0.5 UMAP;
neighbourhoods represent profile similarity rather than ancestors or
descendants. Pure-state family clustering is an expected consequence of
shared expression, rather than an independent biological discovery.

Within 329 early families whose captured cells are all annotated PB,
distance from the two-dimensional centre of GC-enriched families is
negatively associated with the dark-zone/cycling module score (Figure 3E).
Of ten mice with at least five such families, all ten show a negative
within-mouse correlation (median Spearman ρ=−0.596). Day 7 contributes only
one family, compared with 227 at day 10 and 101 at day 14. This check
supports within-annotation variation beyond a pooled difference between
dates. Because the module and embedding reuse RNA, the relationship is
descriptive: it does not independently establish maturation time, a GC
origin or recent GC output. A threshold on two-dimensional distance is
not treated as a validated origin classifier.

Later-infection samples continue to contain GC-enriched families through
day 42 (Figure 3F). The fraction of reliable families with at least 50% GC
cells is calculated separately for each mouse and treatment arm, using
1,174 families from 36 infected mice. This denominator excludes the nine
families from four naive control mice retained in the 1,183-family display
in Figure 3C. Points are mice and lines connect group medians; the analysis
describes captured repertoire composition. Different terminal mice and
different early/later sampling regimes preclude interpreting these maps
as persistence of the same family across days, directed GC output or a
causal treatment response.

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
estimates. The accompanying source analyses retain per-mouse deviations
from the isotype-preserving null with a fixed all-family denominator;
these tests are distinct from the reliable-family composition displayed
in Figure 3F and do not assign an output direction.

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

The treatment arms are shown by day and mouse in Figure 3F and Supplementary Figure 3.
Differences are interpreted within the late experiment, with three infected
mice per arm/day. Uninfected controls and naive spike-in cells are not treated
as additional infected replicates. The dataset does not supply clone-specific
future outcomes or affinity measurements; it prioritises families for such
experiments rather than replacing them.

### Lineage barcodes test early-state associations with later observations

To separate clone-state visualisation from temporal validation, we reanalysed
the public in-vitro LARRY data used by Clonotrace
{cite:weinreb2020,clonotrace}. The 130,887-cell input contains 49,302 cells with
a unique lineage barcode. We estimated 7,575 expanded clone-day profiles in a
common receptor-excluded RNA space. Figure 4B shows the author-provided SPRING
cell landscape: a force-directed display of a cell-neighbour graph
{cite:spring2018}. Figure 4C summarises each clone-day cell distribution as a
Threadfin profile, with the same author state labels used for biological
interpretation. The comparison makes the change of analysis unit explicit:
from individual cell states to the state composition of many sampled clones.
Barcode-colouring on SPRING can already reveal where a clone's cells lie;
the paired maps demonstrate interpretability, not an advantage over SPRING.
The quantitative tests below assess whether estimated clone profiles add value
over simple summaries. Lines connect the same known barcode across measured
days; their direction is supplied by sampling dates and known barcode identity.

For prediction, we built a second expression representation using day-2 cells
only. Neither future cells nor their state annotations entered that
representation. Among 1,401 expanded day-2 barcodes, 172 also had at least four
day-6 cells and were eligible for the fixed comparison. The later target was the
observed fraction of cells in each author-annotated state, not a complete
assessment of developmental potential. All methods used the same three repeated
five-fold barcode partitions and training-fold-only ridge tuning.

The median fold R² for day-6 neutrophil fractions was 0.199 for Threadfin kernel
profiles, 0.163 for RNA means and 0.110 for concatenated RNA means and
variances; corresponding monocyte values were 0.163, 0.099 and 0.058
(Figure 4D,E). The mean-plus-variance baseline adds a simple measure of
within-clone spread using the same 30 early-cell PCs and identical readout folds. Absolute-error differences were smaller: for monocyte
fractions, kernel-profile MAE was 0.227 versus 0.225 for RNA means. Thus early
profiles contain information about later captured composition, but the added
value over a simple RNA mean is modest and depends on the endpoint. Adding
per-PC variance alone did not improve this baseline; that result does not isolate
the contribution of the kernel from shrinkage, smoothing or regularisation. Repeated
folds are not independent biological replicates, and eligibility selects a small
subset of the observed early barcodes.

### Longitudinal paired TCRs distinguish clone identity from changing state

We next analysed all ten public processed patient objects in GSE266219
{cite:mathew2024,clonotrace}. This analysis tests generality of the representation;
it does not reconstruct the eight-patient clinical-response selection in the
Clonotrace paper. Of 195,685 exported TCR-annotated cells, 124,534 had exactly
one TRA and one TRB CDR3 amino-acid sequence. Exact paired sequences within
patients define biological clones, with separate profiles for each observed
cycle. Patient-level centring retains temporal variation; cycle is not removed
as a nuisance covariate.

The analysis produced 10,428 expanded clone-cycle profiles (Figure 5A–C).
Expression-module colours describe a cytotoxic-to-memory-associated continuum,
using the same expression measurements that constructed the profiles. These
colours are descriptive rather than independent validation. Original cycle
labels, including cycles 3, 5 and 7 where present, were retained.

For 29 eligible patient–adjacent-observed-cycle comparisons, we compared the
same biological clones with different target clones within patient, interval
and target capture-count bin. All methods used identical pairs with at least
four cells at both dates. The median across patient-level median same/null
distance ratios was 0.608 for kernel profiles, 0.637 for Threadfin mean profiles
and 0.660 for RNA means (Figure 5D). Each representation retained same-clone
similarity, with no basis for calling this signal unique to Threadfin.
Within-clone module changes varied across patients and intervals (Figure 5E).
These observations support longitudinal state descriptions, but expansion,
contraction and migration can also change the captured blood distribution.
An audited clinical-response mapping is required before testing treatment
response associations.

### Captured-state readout differs from repertoire-network refinement

Published tools address several related tasks: repertoire annotation, joint
sequence/expression representation, clonotype networks, CSR dynamics and whole-clone
state descriptions. We record their native capabilities and published validation
separately from measured performance. CoNGA, Ibex, scRepertoire and sciCSR
address distinct receptor/expression or class-switch tasks
{cite:conga,ibex,screpertoire2,scicsr}. We avoid assigning zero to an inapplicable
output ([source comparison](METHOD_COMPARISON.md)). Benisse {cite:benisse}, BiGCN {cite:bigcn}
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
0.314. These measurements evaluate a specific captured-state readout. Ordinary RNA
means outperform Threadfin on the primary division endpoint, and sequence-native
tools need not be optimal for this task. The comparison does not establish a
universal method ranking. Profile reliability and repeated-state comparisons
require separate validation.
The size-threshold sensitivity retains at least five measured cells per family
(Supplementary Figure 7).

{RBD_BENCHMARK_RESULTS}

The reporter gate is associated with physical sequencing library in these
experiments. Whole-mouse label holdout retains shared libraries and a
transductive representation; the comparison cannot separate reporter biology
from all library effects or establish independent-library generalisation.

We therefore also tested the sampling estimator without a biological-label
prediction task. Calibration and evaluation used disjoint LARRY biological
barcodes, with 250 evaluation clone-day-well profiles and disjoint reference
cells. At 2, 4 and 8 query cells, shrinkage reduced pooled kernel-mean squared
error by 27.2%, 14.2% and 6.5%, respectively (Figure 6E,F). This is evidence
for denoising under this sampling design, not for absolute calibration of the
reliability number or for improved fate prediction. Smoothing was disabled;
the shared RNA preprocessing remains transductive.

### Repeated human GC sampling tests clone persistence over a longer response

The vaccination cohort repeatedly samples draining lymph nodes and blood from
the same participants {cite:human_gc}. This differs from the terminal Plasmodium design:
a BCR family can genuinely be observed at multiple dates in one person.
Supplementary Figure 9 shows the study design, captured GC/output compartments and an
independent spike-positive receptor label on the clone map.

GC-dominated and antibody-secreting profiles are distinct in this dataset.
The restored binding-enrichment plot compares author-identified Spike-positive
families within expression programmes against other families from the same donor
(Supplementary Figure 9D). It reports binding-label odds rather than quantitative affinity.
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

The corrected clone-state retention estimate is 0.25 across dates (420 families;
95% interval 0.20–0.31), compared with approximately zero across LN/blood
snapshots (159 families). This shows that family membership and current
compartment are different pieces of information. It does not establish whether
a memory cell entered a later GC or whether a particular GC cell produced a
blood plasmablast. A study with prime/boost and fate mapping can address those
directions directly {cite:memory_reentry}.

The GC mutation trend is also shown explicitly (Supplementary Figure 9G). We average within
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
analyses (Supplementary Figure 15B). The direct NP–OVA PC/GC dataset is the exception
(p=0.224). These are separate dataset analyses, not twelve independent studies:
the two reporter cohorts share a publication, and the two infection datasets
are separate experiments from one study. The comparison excludes receptor genes
from the expression embedding and preserves library composition and family sizes
under shuffle. It measures captured clonal expression organisation, rather than
antigen specificity, future fate or a benefit over another integration method.

The biological interpretation depends on the experimental anchor and sampled
repertoire (Supplementary Figure 15A,C). Only 3 of 113 expanded families in the direct PC/GC
study reach profile reliability 0.5, compared with 69 of 373 NP reporter families,
381 of 1,414 RBD reporter families and 4,429 of 8,435 human vaccination families.
Library-stratified signal does not guarantee that individual profiles are
informative enough for programme inference. Clonal resemblance of GC, cycling,
light-zone, antibody-secreting, memory-associated and interferon modules varies
between models (Supplementary Figure 15D). Those modules reuse the profile-building expression
matrix; expression-matched gene backgrounds aid interpretation but do not turn
module agreement into independent biological validation.

### Non-GC tests distinguish receptor identity from terminal-state annotation

Outside the GC-focused argument, the marrow/blood model {cite:marrow} supplies the
clearest measured-gate check (Supplementary Figures 5 and 14). A GEO manifest corrects
donor identities and retains 15 single-donor, single-tissue libraries from seven
donors, excluding five pooled or mixed libraries. The analysis contains 115,144
cells, 78,883 paired receptor records and 9,783 expanded families. Within pure
marrow plasma-cell and memory-cell gates, unique productive exact heavy/light
receptors are shared in 225/862 eligible repeated-receptor groups for donor 1681
and 31/677 for donor 1684; marrow PC–blood memory sharing is 124/1,012 and
115/934. These are identities across captured gates, without implying
transdifferentiation or a direction of differentiation. Influenza blood, influenza
lung, EBV-infected tonsil organoids, tonsil and COVID-19 blood data provide additional tested coverage
(1,997, 114, 6,923, 670 and 83 expanded families, respectively). Their clonal
expression signal is included in Supplementary Figure 15, whereas individual maps remain in the
separate tested-dataset folder. Exploratory transcriptomic states and variable
sampling designs in those datasets do not establish independent fate mechanisms.
Thus one non-GC anchor supports interpretation of common receptor identity,
while the remaining analyses document where the workflow has been tested.

## Discussion

Threadfin's useful biological output is a description of clone-level state
bias and breadth, anchored to separately established sequence membership and
experimental labels. The reporter models show that expression distributions
record division-associated GC state alongside a family-average receptor history.
The infection model asks which GC/output-like states are co-observed in the
same mouse's families. Lineage barcodes test early-to-later state associations, paired TCRs test
longitudinal state retention, and the human GC cohort adds repeated membership.
The marrow validation shows that common receptor identity can coexist with
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

Existing whole-clone methods, including clone2vec and the Clonotrace
preprint, already support continuous clone descriptions and clone-associated
gene analysis {cite:clone2vec_preprint,clonotrace}. Clonotrace compares smoothed
clone densities and relates clone profiles to measured temporal observations.
Using its public datasets does not by itself reproduce its method or establish
a direct performance comparison. Our Figure 4 contrasts cell and Threadfin
clone representations; neither panel is an official Clonotrace output. The incremental contribution claimed here is the
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

`case_studies/reproduce_gc_legacy_embedding.py` reconstructs the legacy-input
comparison; `case_studies/run_gc_reclustering.py` retains the earlier restricted
49-family control. Both align the GSE246382 MTX/features/barcodes
and deposited mouse/compartment metadata to the frozen 884-cell table and its
same-mouse sequence-defined family calls. Cell identities, mouse labels,
compartments and the original at-least-500-expressed-gene QC are checked.
Raw counts and source-table hashes accompany the results. Threadfin constructs
a receptor-excluded 3,000-HVG, 30-component PCA from log-normalised counts
for the separately retained PCA analysis. The primary clone-embedding run
instead reconstructs the notebook's executed cell pipeline (normalise to
10,000 counts, log1p, flag 5,500 highly variable genes without subsetting,
regress total counts and mitochondrial percentage, scale, PCA with the Scanpy
default highly-variable mask, 15-neighbour graph on 40 components, UMAP
min_dist 0.5, seed 0) and groups deposited productive TRUST4 IGH calls into
donor-restricted V–D–J receptor groups retained at at least one captured cell
(377 groups, 762 cells). These groups can merge distinct junctions within a
mouse; pooled cross-mouse V–D–J groups are diagnostic controls only. It follows notebook cell 27: pairwise Euclidean
centroid distances, UMAP on distance-matrix rows as Euclidean features
(40 neighbours, min_dist 0.65, spread 1, learning_rate 1, seed 123), then
Scanpy neighbours on clone UMAP (15) and Leiden (resolution 0.3, seed 0).
Gate labels and post hoc marker summaries are not supplied as clustering
covariates. Marker genes can contribute to the underlying RNA embedding. The full parameter
grid (10–80 UMAP neighbours × min_dist 0.1–0.85), seven seeds, three family
definitions, two cell bases and the all-genes sensitivity run are saved with
requested/effective parameters, source hashes and package versions. Selection
is exploratory and considers layout continuity and local fidelity; it is not
an optimisation for six groups or maximal seed stability. Median adjusted
Rand index is 0.877 for the display and 0.989 for the historical k=20,
min_dist=0.4 donor-restricted V–D–J control across seven seeds.
New clone coordinates are
computed; Supplementary Figure 8B reuses the reconstructed cell coordinates with
actual measured gates. The original notebook's processed objects and IgBLAST
tables are absent, so deposited TRUST4 calls serve as an independently recorded
receptor-call source; exact reproduction of the historical coordinates is not
claimed, and cross-mouse pooled VDJ groupings are reported as diagnostics only.

`tf.clonotype_recluster` provides cohesive (20 neighbours, resolution 0.3,
min_dist 0.1), continuous (20, 0.1, 0.4) and discrete (10, 0.8, 0.05)
starting configurations. All use spread 1 and learning_rate 1. Explicit values
override the preset; neighbour counts cap at retained clone count minus one.
By default, Leiden uses original clone distances and UMAP embeds them as
precomputed; display controls leave that partition unchanged. The primary
figure explicitly uses embedding_mode="distance_profiles" and
cluster_on="embedding" to match the notebook geometry and Scanpy graph.
In this mode, display parameters can affect the clustering; the graph and
UMAP neighbour counts are specified independently; cluster_random_state=0
separates graph/Leiden randomness from UMAP random_state=123. Presets are not learned
biological classes and do not
force a desired topology. The default filter is at least three cells;
strictly greater than three requires min_clone_size=4. Centroid exploration
remains separate from v4 sampling-adjusted profiles and programme validation.

Figure 2B colours unsupervised Leiden clone clusters. For their annotation,
measured-compartment fractions and log-normalised Myc/marker RNA are computed
over member cells, then averaged equally across families within each cluster.
Supplementary Figure 8D shows this cluster-level marker heatmap.
GC-identity, cycling and plasma-cell modules are
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
model. Its Gaussian null does not repeat the complete community, tree and
resolution selection, so its split p-values are approximate and do not establish
family-wise error control. Bootstrap stability is reported rather than treating
every clustering as a stable biological category. Clone-state retention compares separate
same-family snapshots against same-donor random-family comparisons, correcting
sampling noise. Its intervals resample clones and do not replace donor-level
population uncertainty. Full existing algorithms are specified in
[`docs/METHODS.md`](../docs/METHODS.md).

Mouse module scores use capitalisation-adjusted gene symbols, with per-dataset
coverage recorded; this is not a curated, gene-by-gene orthologue analysis.
Coverage is recorded for each signature. Clone-averaged scores and programme
scores reuse normalised expression and are descriptive; their agreement with an
expression-derived programme is not counted as independent validation.

### Conditional biological interpretation of Figure 2

We audited all 1,414 expanded RBD families and the 381 reliability-filtered
displayed families from the committed cell and family tables. Within each mouse,
we computed Spearman correlations of division fraction with RNA modules or mean
V mutation frequency. Partial rank correlations residualized both ranked
variables against ranked log capture count and the additional measured gate
fraction (RBD binding in the protein arm, DZ in the mRNA arm). At least eight
families and variation in both measurements were required. For paired SHM,
we retained family-by-additional-gate strata with at least two measured cells
in each mCherry extreme. We averaged stratum differences within family and
family differences within mouse; uncertainty was described by 10,000 mouse
bootstrap resamples within each arm (seed 20261008). The five mice per arm
limit interval precision. These exploratory tests do not remove physical-library
confounding or estimate mutation per division.

### Public lineage-barcode and longitudinal TCR validation

For GSE140802/GSM4185642, author-normalised cell-by-gene expression, lineage
membership and row-matched metadata were downloaded from GEO. Input dimensions
were 130,887 cells by 25,289 genes; 49,302 cells had exactly one of 5,864 barcode
labels, with no ambiguous multi-barcode rows. The expression matrix contains
normalised positive values, not raw integer UMI counts. We rescaled total
expression to 10,000 per cell, applied log1p, selected 3,000 receptor-excluded
variable genes, scaled with clipping at 10 and computed 30 PCs with seed 0.
The descriptive all-time map uses barcode-labelled cells and no context
subtraction. Each barcode-day is a distinct profile, requiring at least two
cells. Kernel profiles use 256 random Fourier features, median-distance
bandwidth, 15-neighbour smoothing and 30-component profile reduction; mean
profiles are unsmoothed. UMAP uses 15 neighbours, min_dist 0.3 and seed 0.
Author SPRING coordinates are used only for the cell-context panel.

The prediction model is independently preprocessed from day-2 cells only,
including unlabelled cells as expression context. Barcode-level targets are
day-6 state fractions for barcodes with at least two early and four later cells.
Unsupervised early PCA and profiles are transductive across the eligible
barcodes; no day-6 cells or labels enter them. Three five-fold outer partitions
(seeds 100–102) hold out biological barcode identities from the supervised
readout. A StandardScaler–ridge pipeline is tuned over alpha 0.1, 1, 10 and 100
using three training-only inner folds. Predictions are clipped to [0,1] and
normalised to sum to one. RNA means, concatenated means and population
variances of the same 30 PCs (ddof=0), log capture count, Threadfin mean/kernel
profiles and training-fold outcome means use identical outer partitions.
Saved tables contain each out-of-fold prediction and fold assignment.

For the shrinkage check, biological barcodes are deterministically split by
SHA256 parity into calibration and evaluation sets. Calibration cells alone
determine the PCA-space centring, kernel bandwidth, feature centring and
variance components; the common descriptive RNA PCA remains transductive.
The 256-feature kernel is unsmoothed. Evaluation units are barcode-day-well
profiles with at least 16 cells. For each of 20 random splits, one half provides
the disjoint reference mean, while 2, 4 or 8 cells from the other half provide
the unshrunk mean and the shrunken estimate. Error is the mean squared difference
across kernel features. Reported error reduction is one minus the ratio of
pooled mean shrunken to unshrunk error; it is not the average of per-profile
ratios. Shared barcodes across dates/wells are not independent biological
replicates, and this test does not calibrate the absolute reliability scale.

For GSE266219, all ten official processed patient objects were downloaded and
exported from Seurat with explicit cell-name alignment and the intersection
of 17,178 gene symbols. Temporary inspection files are excluded by the exact
filename pattern. Only cells with one TRA and one TRB CDR3 amino-acid sequence
enter the analysis. Canonical paired-chain strings are patient-private clone
identifiers; author clonotype IDs are audited but do not define cross-cycle
identity. Original PtCycle labels provide the cycle numbers. Receptor-excluded
30-PC RNA preprocessing uses the same normalisation settings and no Harmony;
profiles are centred by patient, with cycle preserved. Cell UMAP displays a
deterministic sample of 30,000 analysed cells. Mean per-gene z scores define
cytotoxic (NKG7, PRF1, GZMB, GNLY, CTSW), memory (IL7R, CCR7, TCF7, LEF1, LTB),
activation (IFNG, FOS, EGR1, CD69, TNFRSF9) and cycling (MKI67, TOP2A, TYMS,
STMN1) modules; these are expression descriptions.

Longitudinal comparisons require at least four captured cells per clone-cycle.
For each patient's consecutive observed cycles, target identities are permuted
200 times within floor(log2 target cell count) bins, using derangements without
fixed points. Singleton bins are excluded. Distances are Euclidean distances
in each method's feature space, not UMAP distances; method comparisons retain
exactly the same clone pairs. Ratios compare the median observed same-clone
distance with the median permuted median distance. Figure 5D first takes the
median across eligible intervals within each patient. No response categories
are inferred from patient numbers.

### Display-parameter and within-state analyses

The RBD parameter audit reconstructs the family features before comparing 85
UMAP settings. Global linear R² fits each readout to an intercept and the two
map coordinates. The local readout averages the ten nearest labelled families,
excluding the focal family. The saved field `knn_r2` is a legacy name for
explained variance, calculated as one minus residual variance divided by
readout variance; unlike standard predictive R², it does not penalise mean
prediction bias. Local excess is observed explained variance minus its mean
under 200 donor-stratified label permutations. These exploratory tests are
not held-out-mouse predictions and do not adjust for choosing among display
parameters. Pair-distance faithfulness is a Spearman correlation over sampled
family pairs in the feature space and the two-dimensional map.

For Figure 3E, pure-PB families contain only PB-labelled captured cells.
Euclidean distance uses their saved early-map coordinates and the mean
coordinates of reliable families with GC fraction at least 0.5. Within-mouse
Spearman correlations are summarised for mice with at least five pure-PB
families. Figure 3F uses the same 0.5 GC-fraction threshold without any
coordinate-distance rule, dividing by all reliable families in each infected
mouse and displaying treatment arms separately. The figure script regenerates
the family- and mouse-level source tables with the normal drawing workflow.

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
