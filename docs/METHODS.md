# How Threadfin works

This page describes each step of `threadfin.run()` and the matching
functions in `threadfin.tl`, with the defaults and the reasoning behind them.
References are listed at the end.

**How to read this page.** Every section starts with *In plain words*, a summary without formulas, followed
by the details. If you only want to use Threadfin, the plain-words paragraphs are enough; the details matter
when you write a methods section or change a default.

**The whole method in one paragraph.** Cells are grouped into clones by their receptor sequences. Each cell
has coordinates in an expression space built without immunoglobulin genes. A clone is described by the
distribution of its cells in that space, measured relative to the other cells sampled with it and shrunk
towards the average profile of all profiled clones when the clone was seen in few cells (the *clone profile*).
Clone profiles are then compared: in a two-dimensional *clone map*, in groups of similar clones (*programmes*,
kept only if they pass a split test, each reported with its stability when cells are resampled), and in
permutation tests that ask whether cells of one clone are more alike than cells of different clones
(*coherence*) and whether an annotation explains differences between clones (*association*). Annotations are
tested with clones, not cells, as the unit, and every permutation shuffles labels only within the strata of the
sampling design: clone labels within sampling units (libraries, sort gates), annotations within donors.

**Symbols used below.** *i* a cell; *c(i)* its clone; *s(i)* its sampling context; *x* a cell's embedding
coordinates; *n_c* the number of cells of clone *c*; *τ²* variance between clones; *σ²* variance within
clones.

## 1. Clones (`define_clones_dandelion`)

A clone is the family of B cells descended from one V(D)J recombination event — one
rearrangement of the heavy **and** the light chain. Threadfin delegates clone calling to
**Dandelion** (a maintained, cited implementation of the paired definition;
sc-dandelion; Suo et al. 2024), through :func:`threadfin.define_clones_dandelion`: contig-level AIRR tables
are grouped within each donor with ``dandelion.tl.find_clones`` (default junction identity
0.85 for BCR and 1.0 for TCR), and the composite heavy+light clone id is returned per cell. Dandelion is an optional
dependency: only clone calling needs it, and clone ids made by any other caller can be supplied
directly in ``adata.obs``. Install it with ``pip install "threadfin[dandelion]"`` (Python 3.10–3.14) or
``pip install "sc-dandelion>=0.5.7,<1"``. Threadfin is validated with Dandelion 0.5.x; Dandelion 1.0 calls clones
differently. Dandelion 0.5.7 imports nxviz, which uses ``matplotlib.cm.get_cmap``, a function removed in
matplotlib 3.9. Importing Threadfin restores it, so import Threadfin first if you call Dandelion yourself
with matplotlib 3.9 or later.

Dandelion's assignment depends on Python's string-hash seed (set iteration order). Threadfin calls clones
under seed 0 in every session, in a separate Python process when Python was not started with
``PYTHONHASHSEED=0`` (about 2 s more per call). Clone ids are therefore identical run after run, with no setting
needed. ``hash_seed=None`` (``clone_options={"hash_seed": None}`` in ``tf.run``) keeps the earlier in-session
behaviour. The seed and whether a separate process was used are recorded as ``ids.attrs["dandelion_hash_seed"]``
and ``ids.attrs["dandelion_subprocess"]``. After ``tf.run(bcr=...)``, both are in
``result.settings["clone_calling"]``, and ``result.methods()`` states the seed.

*In plain words.* Two cells belong to the same clone when they carry the same heavy-chain and the same
light-chain rearrangement (same V and J genes, same junction length, junctions ≥ 85 % identical, which
allows for somatic hypermutation). Clones are called inside each individual, so no clone can contain cells
of two people or animals.

*T cells.* TCR contigs are handled the same way, with the beta (or delta) chain in the place of the heavy chain
and the alpha (or gamma) chain in the place of the light chain. T-cell receptors do not mutate, so T-cell
clones need identical junctions (Dandelion's default identity of 1.0 for TCR). In both cases, cells without a
heavy (beta) chain are left without a clone (light and alpha chains alone are not diverse enough, and some alpha
chains, such as those of MAIT cells, are shared by unrelated T cells).

*Cells with two chains of one kind.* A cell with two productive light chains or two heavy chains (allelic
inclusion, ambient RNA, or a doublet) receives from Dandelion a clone id that joins both chains, so it usually
forms a clone of its own; Threadfin reports how many cells this affects (about 4–25 % of cells in the public
datasets, mostly two light chains). Optionally (`check_contigs=True`), Dandelion's contig check runs first: a cell with
two heavy chains keeps the one that clearly dominates in UMIs or is set aside, while two light chains are allowed
(Suo et al. 2024). The case studies use the default; the clone-definition sensitivity analysis
(`manuscript/analyses/technical_validation/clone_definition`) compares both, and junction identities from 0.75
to 1.0.

(Threadfin's earlier built-in caller grouped cells by the heavy chain alone with a
data-driven Hamming threshold, light chains only as an optional post-hoc split. It is
deprecated: two cells whose heavy chains are similar but whose light chains are unrelated
are not the same clone, and roughly a quarter of analysable multi-cell clones are lost under
the stricter paired definition — a reported, intended cost.)

## 2. Expression embedding (`threadfin.pp.prepare_embedding`)

*In plain words.* Each cell gets 30 coordinates that summarise its gene expression, computed after removing
the receptor genes so that cells of one clone do not look alike just because they express the same
antibody transcripts.

Standard single-cell processing (normalisation, 3,000 highly variable genes,
30 principal components, optional Harmony integration over technical batches;
Korsunsky et al. 2019) with one important change: **immunoglobulin and TCR
genes are removed before choosing variable genes**. Cells of one clone share
their receptor transcripts, so leaving them in would make clonal relatives
look alike for a trivial reason, and would make isotype a circular label.
Integrate only over technical factors (donor, library), never over the biology
under study (tissue, time point, sort gate).

## 3. Clone profiles (`clone_profiles`)

*In plain words.* A clone profile answers "how are this clone's cells distributed over cell states,
compared with the other cells sampled with them?". It is measured relative to the sampling context, it
describes the whole distribution (so a clone split between two states differs from a clone halfway between
them), and it is pulled towards the average profile of all profiled clones when only a few cells of the clone
were seen. The pull is strong for a 2-cell clone and weak for a 50-cell clone, which is how small clones are
kept without letting their noise dominate.

Each clone is described by where its cells sit **relative to the other cells
sampled with them** (same sample, or same donor), so that differences between
samples or batches are not mistaken for differences between clones.

Clone profiles come from a random-effects model (Searle et al. 1992): every
cell's position is its context's average, plus a clone effect, plus cell-level
noise, for each coordinate separately. From it Threadfin obtains

* the **clonal intraclass correlation (ICC)**: the share of the variance of cell
  states (the context-centred expression embedding) that is explained by clone
  identity;
* a **shrunken profile** for each clone (its best linear unbiased predictor):
  the clone's mean is pulled towards the average profile of the profiled clones
  (the mean of all their context-centred cells), strongly for small clones and
  weakly for large ones;
* a **reliability** for each clone, which grows with the number of cells
  sampled (Spearman-Brown): the expected squared correlation between the
  estimated and the true profile. Reliability sets how strongly a clone takes
  part in defining programmes; no clone is excluded for low reliability. A clone
  needs at least two cells, because a single cell cannot show whether cells
  of the same clone share a state.

By default (`representation="kernel"`), the profile that the clone map, the
programmes and the clone-level tests use describes the **whole distribution**
of a clone's cells, not only their average (a kernel mean embedding; Rahimi &
Recht 2007; Muandet et al. 2017). A clone split between two states is then
different from a clone sitting in between. For this profile each cell is first
averaged over its 15 nearest cells (itself included) to reduce noise, and the
shrinkage and the reliabilities are computed from these smoothed kernel
features. The reliability is exact for independent cells; for smoothed cells it
is slightly optimistic (in simulations by 0.02–0.09, more for larger clones),
and the summary says that it refers to the smoothed profile. The ICC is
computed on the unsmoothed embedding coordinates, and the clonal memory index
(section 8) on unsmoothed kernel features.

## 4. Clonal coherence (`clonal_coherence`)

*In plain words.* "Are cells of one clone more alike than cells of different clones?" The answer is the
share of the variance of cell states (the expression embedding) that clone identity explains — 0 if clones
are random collections of cells, 1 if all cells of a clone are identical — compared with what shuffled clone
labels give. For example, an ICC of 0.27 against 0.002 for shuffled labels means that clone identity explains
about a quarter of the variance of cell states, 27 percentage points more than the sampling design alone.

The clonal ICC is compared with the same quantity after shuffling clone labels
**among the cells of one sampling unit** (`strata_key`). The shuffled clones
keep their sizes and every sampling unit keeps its cell-state composition, so
differences between sampling units cannot create apparent clonality. The
strata should therefore be the finest sampling unit: the library, sort gate or
tissue sample. By default they are the sampling context (`sample_key` in
`threadfin.run`). When the context is coarser — for example the donor, in a
design that compares tissues or sort gates within donors — name the libraries
with `strata_key`: cells of one clone are sequenced together, so shuffling
across libraries would count the differences between libraries as clonal.
Strata never cross donors, because a clone lies within one donor: when a
library pools cells of several donors (hashtag-multiplexed samples, for
example), clone labels are shuffled within each donor × library combination,
with a warning. The
result is reported as the variance explained by clone, the shuffled value,
their difference (the excess, which the summary gives in percentage points,
together with the strata used) and a permutation p-value (Phipson & Smyth 2010).

*Power.* How many clones of two or more cells the test needs depends on how
many dimensions of the embedding carry the clonal signal. In simulations a
clonal ICC of 0.1 was detected with 80 % power from about 10 clones when the
signal was spread over 30 dimensions, and from about 20 clones when it was
spread over 10; an ICC of 0.05 needed about 40 clones or more.

## 5. The clone map (`embed_clones`, `tune_clone_map`)

*In plain words.* A picture with one dot per clone, in which clones with similar profiles are close. It is
for looking, not for testing; Threadfin chooses its layout settings so that a population of clones that is
connected in profile space is not broken into islands.

*How similar are clones to one another?* Every clone with a profile is placed on a two-dimensional
map with UMAP (McInnes et al. 2018), one point per clone. Clones whose cells are distributed alike sit
close together, so gradients in the composition of clones, for example from clones of germinal-centre B
cells to clones of ASCs, appear as paths. The map is a display: profiles, coherence, programmes and all tests
are computed in profile space and do not depend on it. No clone is removed for low reliability. Because the
layout is chosen to keep connected what the profiles connect (below), a connected map is not evidence of a
continuum: discrete types of clones joined by clones that mix their states also form one connected piece.

Two UMAP settings decide what the map looks like: the number of neighbours each clone is tied to
(`n_neighbors`) and how closely clones may pack (`min_dist`). Too few neighbours or too little spacing
breaks a connected population of clones into islands; too many neighbours smooth away local structure. Because the
best values depend on the number of clones and on how they are distributed, Threadfin chooses them for
each dataset by default (`n_neighbors="auto"`, `min_dist="auto"`); numbers set them by hand.

**What is optimised.** Each candidate layout is scored, without any annotation, on

* *coherence*: the fraction of clones in the largest connected piece of the map, where two clones are
  connected when they lie within 1.5 *D*/√*N* of each other (*D*, extent of the map; *N*, number of
  clones). At Scanpy's default point size this is about the distance at which drawn points touch, so a
  piece is what a reader sees as one island;
* *local fidelity*: the fraction of each clone's 30 nearest neighbours in profile space that remain among
  its 30 nearest neighbours on the map;
* *global fidelity*: the Spearman correlation between profile-space and map distances over random pairs
  of clones (both fidelity measures as in Kobak & Berens 2019).

The score is the mean of the two fidelities minus 5 × the shortfall of coherence below a target. The
target comes from the profiles themselves: the fraction of clones in the largest connected component of
the mutual 15-nearest-neighbour graph of profiles (the graph scale used for programmes), capped at 0.99.
A map is therefore not allowed to be more fragmented than the profiles, while a small group of clones
that is genuinely separate in profile space may stay apart.

**How settings are searched.** A UMAP layout is random, and the settings that are most faithful in one
layout are often those closest to breaking apart in another. The search is therefore built on the worst
case. (i) Clones are laid out at every combination of six neighbour counts (10, 18, 32, 56, 100, 180; at
most half the number of clones) and four `min_dist` values (0.3, 0.55, 0.8, 0.95), each with two random
seeds shared by all settings; a setting is credited with its mean fidelity and its lowest coherence.
(ii) Scores are averaged with those of neighbouring settings (Gaussian weights), so that settings
surrounded by good settings are preferred. (iii) The best settings are laid out with four more seeds.
(iv) Among the settings that held together in all six layouts, the best smoothed score wins; ties go to
the more coherent setting. (v) If no setting held together, the most coherent settings seen are
confirmed as well and, if necessary, laid out with weaker repulsion between clones (UMAP
`repulsion_strength` 0.5, then 0.25), which moves UMAP towards more continuous layouts (Böhm, Berens &
Kobak 2022). During the search, layouts use 200 optimisation epochs. Above 3,000 clones the search runs
on a random subset, `n_neighbors` is scaled to the full data in proportion to the number of clones, and
the best settings are confirmed again on all clones before one is chosen.
Finally, the map that is returned is laid out with the chosen settings and UMAP's default number of
epochs, and checked: if it falls short
of the target, two further seeds and then the next confirmed settings are tried, and the most coherent
layout is kept.

Everything the search did is stored in `adata.uns['threadfin']['profiles']['map_params']`: the chosen
settings, the quality of the returned map, every layout scored and every candidate confirmed. The choice
is the same for any `n_jobs`; it can change with the number of BLAS threads (see Reproducibility, below).
`tl.map_quality` scores any map, including one made by hand. An alternative search, `method="adam"`,
follows a stochastic gradient instead of the ladder (Adam, Kingma & Ba 2015, with simultaneous-
perturbation gradient estimates, Spall 1992) and ends with the same confirmation and checks.

## 6. Clonal programmes (`find_programmes`)

*In plain words.* Groups of clones with similar profiles — but only splits that pass a test against one
Gaussian group of clones. How often each group comes back when the cells are resampled is reported as its
stability. If no split passes the test, Threadfin reports a single programme instead of inventing groups. A
programme is a partition of the clone profiles: graded variation also departs from one Gaussian group, so several
programmes are not proof of discrete types of clones, and one programme is not proof of a continuum.

*Which clones behave alike?* Clones are connected to their 15 most similar
clones, with each connection weighted by the reliability of the two clones,
and grouped by Leiden community detection (Traag et al. 2019). Among the
resolutions that give at least two groups of five or more clones, with at most
10 % of clones in smaller groups, the one with the most groups whose
clone-weighted mean stability under resampling is at least 0.75 is used. If no
resolution reaches 0.75, the most stable of them is used, with a warning when
more than one programme remains and not all of them are stable; if no
resolution gives two such groups, the first resolution scanned (the lowest, by
default) is used.
Two safeguards keep the groups meaningful:

* **Split test.** Community detection always returns groups, even when clones
  vary continuously. Every split is therefore tested against a single Gaussian
  group fitted to the same clones: the 2-means cluster index of the clones is
  compared with its distribution in Gaussian samples of the same mean and
  covariance (Liu et al. 2008), along the tree of groups, with a hierarchical
  threshold allocation inspired by sc-SHC (Grabski et al. 2023). Groups that are
  not significantly separated are merged. The null repeats the 2-means search
  but not the selection of the Leiden partition, the tree and the resolution,
  and its p-values come from a normal fit, so the node thresholds are nominal:
  family-wise error control is not established. The test detects any departure
  from one Gaussian group, which graded variation along a path or a curve also
  produces. If no split survives, Threadfin reports a single programme (see the
  note at the end of this page).
* **Stability.** Stability is a cell-resampling stability. The cells of every
  clone are resampled (a Poisson bootstrap, 30 replicates by default) while the
  set of clones, the smoothing neighbours, the variance components and the
  principal axes of the profiles are held fixed; the clone graph (with the
  original reliabilities as edge weights), Leiden clustering at the chosen
  resolution and the split test (with a quarter of the null samples, at least
  20) are repeated. Each programme's stability is its mean Jaccard overlap with
  the best-matching group of the resamples, as in Hennig (2007), who resamples
  the objects themselves; here the clones are fixed and their cells resampled.
  Stability therefore measures how robust a programme is to the sampling of
  cells, not whether it would recur in other clones or donors. The thresholds
  are Hennig's guide values: stability ≥ 0.75 is stable, 0.6–0.75 present but
  blurred, and below 0.6 a programme should not be interpreted. The summary
  marks programmes below 0.6 as unstable, on the label-test lines too.
  Programmes are not removed for low stability.

Clones left out when the programmes are defined (below `min_reliability`, or in
Leiden groups of fewer than five clones) are afterwards assigned to the most
likely programme, taking their size into account, when the posterior
probability is at least 0.7 (`clone_programme_assigned`).
Marker genes of each programme (`programme_markers`) are found by comparing
clone averages, so a large clone counts once. Programmes usually differ in which cell states their clones
contain, so these markers are largely the markers of those states; with `state_key` and `state`, each clone is
summarised by its cells in one state only (for example its germinal-centre B cells), centred on the cells of
that state in its context, which asks what differs between programmes within a state. The P values and FDRs of
markers are descriptive: the programmes were defined from the same expression, and testing genes between groups
that were found from those genes gives P values that are too small (selective inference; Gao, Bien & Witten
2024). Use them to rank and name genes.

**State composition (`programme_composition`).** With a cell-state annotation (`state_key`), each programme is
described by its composition, counted in one of two ways (`weight=` of `tl.programme_composition` and
`pl.programme_composition`; `composition_weight=` of `threadfin.run`). `weight="cells"`, the default, gives the
fraction of the programme's cells in each state, so an expanded clone weighs in proportion to its sampled cells.
`weight="clones"` counts every clone once: each clone's share of cells in each state, averaged over the
programme's clones. Use the cell weighting for questions about output and abundance (how many of the programme's
cells are ASCs), and the clone weighting to describe a typical clone and to name programmes, which are groups of
clones. When the two differ, a few large clones drive the cell-weighted view. In sorted designs neither equals the
composition *in vivo*, because the sort set how many cells of each state were sequenced. Only cells with both a
programme and a state are counted.

**Genes along the landscape (`landscape_genes`).** With a single programme, or with unstable programmes, there
are no groups to compare. Each profiled clone is then summarised by the mean context-centred log expression of its
cells, the clones are linked to their 15 nearest neighbours in profile space, and each gene is scored by Moran's
I of the clone means on this graph (Moran 1950): positive when neighbouring clones express the gene alike, about
−1/(n−1) when the gene ignores the landscape. P values come from the normal approximation under randomisation
(Cliff & Ord 1981), with Benjamini–Hochberg correction; in simulations the analytic mean and variance of I
matched 20,000 permutations. The FDR is descriptive when clone sizes differ: the means of small clones are
noisier, and small clones sit close together in profile space because they are shrunk towards the average
profile, so noise alone can follow the graph. Rank genes by Moran's I. The rank correlation of each gene with the
first principal component of the clone profiles gives the direction of a gradient. Because the profiles are built
from the same expression, this names what varies across clones rather than testing an independent hypothesis.

## 7. What distinguishes clones (`profile_association`, `association_test`)

*In plain words.* "Does this annotation — antigen binding, isotype, a sort gate — line up with how clones
differ?" The annotation is summarised once per clone, and its association with the profiles is compared
with the association obtained after shuffling the annotation among clones of the same donor.

Cell-level labels (antigen binding, isotype, mutation load, sort gate, tissue)
are first summarised per clone (majority label, mean value, or the fraction
of a clone's cells in a gate). A clone without a majority level (for example a
two-cell clone with one cell of each kind) has no majority label and is left
out of that test; the summary says how many clones this concerns. Two questions
are then asked, always with **clones as the unit** and labels shuffled among
clones of the same donor and of similar size (2, 3–4, 5–9 or ≥ 10 cells;
among all clones of that size when no donor is given):

* **Does the label explain how clones differ?** (`profile_association`) The
  share of the variation between clone profiles explained by the label
  (as in PERMANOVA for categories, Anderson 2001; as in distance-based
  redundancy analysis for numbers, McArdle & Anderson 2001). This works
  whether or not there are distinct programmes.
* **Which programmes differ?** (`association_test`, when there are at least
  two programmes) For categories, the odds ratio of each programme for each
  level, pooled within donor × clone-size bins (Mantel & Haenszel 1959), with a 95 % interval
  (Robins, Breslow & Greenland 1986); for numbers, rank effects; with false
  discovery rate control (Benjamini & Hochberg 1995). The summary shows the
  pooled shares of clones next to each odds ratio and notes when they point
  the other way, which happens when donors differ both in the sizes of the
  programmes and in how often the label occurs.

**Why clone size is a stratum.** Profiles are shrunk towards the average
profile, the more the fewer cells a clone has, so small clones lie near the
centre of the profile space and programmes partly sort clones by size (in
the manuscript's data 44–74 % of a programme's clones have two cells). A
label that follows clone size alone, such as antigen binding when bound
clones are larger, then differs between programmes without any difference
in state. Shuffling labels only among clones of similar size closes this
path: in simulations, a label that followed only clone size was called
significant in up to 42 % of data sets when shuffled within donors and in
3–8 % when shuffled within donor × size bins, while labels that follow the
programmes kept their power (0–7 points lost). A label that never varies
within any donor × size bin cannot be tested this way and is reported as
such. `size_strata=False` shuffles within donors only, as before version
4.6.0; when every large clone falls into one programme, size bins leave
little to compare and the tests lose power. For numeric labels the rank
effect of each programme is compared with its expectation under these
shuffles. A two-level label gives each programme effect once.

Labels that are fixed for a donor (age, sex, disease severity) cannot be
separated from other differences between donors: shuffling them within donors
changes nothing, so these tests are not run, and the summary lists them with
this reason. Compare such labels across donors, with one value per donor.
A label made from the same expression (a cell-type annotation) is not
independent of the profiles: it describes them but does not validate them, and
the summary marks the test of the `state_key` annotation as a positive control.

**Labels fixed within each sample.** A label with one value among all
cells of every sample (a tissue, sort gate or visit sequenced in its own
library) varies only between samples, so differences between libraries
count as differences between its levels and its replicates are samples,
not clones. Threadfin detects such labels, with the sampling unit
(`strata_key`, else `sample_key`), warns, and the summary says that the
comparison is between samples and gives their number. The test is still
run, because samples can be numerous and a design can separate the label
from the library (Tutorial 5, Step 13).

## 8. Clonal memory (`clonal_memory`)

*In plain words.* "Is a clone still recognisably itself at a later time point, or in another tissue?" A clone
seen at two levels is compared with itself and with other clones of the same donor seen at both.

*Do clones keep their state over time or across tissues?* Threadfin profiles each clone separately at every
level of `time_key` (a time point, tissue or sort gate) at which it was sampled, and pairs each clone's
consecutive levels. For snapshots *a* and *b* with *n_a* and *n_b* cells and context-centred mean profiles *m_a*
and *m_b* (unsmoothed features), the change is D_same = ‖m_a − m_b‖² − v_a/n_a − v_b/n_b. Here *v* is the
variance of a snapshot's cells summed over features, so *v/n* removes the change expected from sampling a few
cells. A snapshot of two or more cells uses its own unbiased variance. A one-cell snapshot uses the mean
variance of the multi-cell snapshots at the same level from clones of the same kind (seen at one level, or
at several); when there are fewer than five of those, it uses that kind at every level. D_other is the same
quantity with *b* replaced by the later snapshot of another clone of the same donor seen at the same two
levels, averaged exactly over those clones. The **memory index** is M = 1 − mean(D_same)/mean(D_other). It is
1 when clones keep everything that distinguishes them from each other, and 0 when a clone is no more similar
to its earlier self than to another clone seen at the same levels. Without sampling noise, M estimates
Lin's concordance correlation coefficient (Lin 1989) between the clone profiles at the two levels, summed
over features and pooled over donors and pairs of levels. A change shared by all these clones, such as a
tissue effect that the context centring did not remove, therefore lowers M, and its size is reported.
Because the noise is estimated and subtracted, M is not bounded by 0 and 1.

The p-value permutes the later snapshots among the pairs of the same donor and the same two levels
(500 permutations; Phipson & Smyth 2010). The noise terms are the same in every permutation, so the p-value
does not depend on the noise correction. The 95% interval is a percentile bootstrap over clones
(500 resamples) that also re-estimates the variance of one-cell snapshots. If a resample leaves no positive
distance between clones, the interval is reported as missing and the index is not interpreted.

*Why compare only clones seen at both levels?* Clones found at two levels are a selected group, and they can
share a state that clones found at one level lack. Against all clones of the later level, as in Threadfin
4.5 (`reference="all"`), that shared state alone reads as memory: in simulations with such selection but no
clone memory, the 4.5 index was 0.81 and its test rejected in 100 of 100 data sets, against −0.01 and 5 of 100
now. For the same reason a pair is compared only with pairs between the same two levels; matching on the
later level alone rejected in 100% of null data sets with three levels, against 4.5% for the current strata.
A variance pooled over all snapshots (`noise="pooled"`, as in 4.5) over-corrects when the compared snapshots
vary less than the rest and under-corrects when they vary more. With a true index of 0.5 it gave 1.12, and an
index above 1 in 77% of data sets. The variance of one-cell snapshots cannot be measured; if it differs from
that of larger snapshots, the index is biased while the p-value is not. With `min_cells=2`, every snapshot
uses its own variance.

Each clone's consecutive levels are paired, so the order of the levels matters.
It is the order given with `order=`; otherwise the order of an ordered pandas
Categorical column; otherwise the numbers in the labels (`d7` before `d28`);
otherwise alphabetical, with a warning. The summary names the order used.

## 9. Genes shared within clones (`gene_heritability`)

*In plain words.* The coherence question asked gene by gene: which genes are more alike among cells of one
clone than among unrelated cells? A gene shared within clones may reflect a state fixed in the clone, but
also a shared niche, sampling time or antigen, or sister cells sampled together; the statistic does not
separate these.

The same random-effects model is fitted gene by gene to context-centred log expression. A gene's clonal
ICC is the share of its variance explained by clone. Each gene is tested by shuffling clone labels among
cells within strata. Genes are ranked by their excess ICC, the ICC minus its mean over the shuffles.

Gene sets, such as plasma-cell or germinal-centre genes, are compared with an expression-matched background:
every other tested gene in the same decile of mean expression, each decile weighted by its share of the set,
because highly expressed genes are measured more precisely. The comparison uses Fisher's z of the clonal
ICC, ½·log of the between- to within-clone mean-square ratio, whose sampling variance hardly depends on the
ICC (Fisher 1925). It is not truncated at zero, and it is centred on its value for the mean squares expected
under the shuffles, which is exact and free of Monte Carlo error.

The null hypothesis is competitive: the set's genes are like other genes of the same expression level.
The variance of the set-minus-background difference therefore has two parts:

* the spread of true values between genes of a decile, estimated as their variance minus their mean sampling
  variance;
* the error from sampling clones, estimated by resampling clones with replacement within strata
  (200 replicates) and recomputing every gene on the same resampled clones. This keeps the correlation
  between genes, whether it comes from co-expression within cells or from programmes shared within clones.

P-values are two-sided (normal reference) and adjusted across sets by the Benjamini–Hochberg procedure.
Inference is conditional on the sampled samples or donors.

Treating the genes of a set as independent, as the Mann–Whitney test used up to version 4.5.3 did, rejected
16 % of null sets at α = 0.05 when genes were co-expressed, and 50–60 % when they also shared clonal programmes
(Goeman & Bühlmann 2007; Wu & Smyth 2012). Shuffling clone labels is no reference for this comparison: it
removes every gene's clonal component, and with it the variation that comes from sampling clones
(47–81 % false positives when every gene was equally clonal). The test used here rejected 4.4–6.3 % of null
sets in simulations with co-expressed and clonal gene modules, and 3.8–5.1 % of random gene sets of the gut
case study.

## 10. Heritability along the receptor lineage tree (`lineage_heritability`)

Clone profiles describe how clones differ from each other. This asks the complementary question inside a
clone: after removing everything a clone's cells share, do cells that sit close together on the clone's
somatic-hypermutation tree still resemble each other?

For a state feature `y`, cell `i` of clone `c` sampled in block `b`:

    y_i = mu_{c,b} + x_i' beta + g_i + e_i,    g ~ N(0, sigma_g^2 K),  e ~ N(0, sigma_e^2 I)

`mu_{c,b}` is a free intercept per clone x block unit, so no clone-wide or block-wide difference can
contribute. `x_i` holds cell covariates, and the cell's root-to-tip mutation depth is one of them by default:
distance on a tree grows with depth, so without that term a state that simply varies with mutation load is
reported as heritability. `K` is a kinship kernel built from the tree — `exp(-d/l)` in mutations, identical
genotype, identical mutated genotype, or Brownian.

`h2 = sigma_g^2 / (sigma_g^2 + sigma_e^2)` is the share of within-unit variation structured by the lineage.
Each unit is projected onto its Helmert contrasts and rotated into the eigenbasis of the projected kernel, so
the restricted likelihood is a sum over rows and is maximised by a one-dimensional search. The test is a REML
score statistic whose null distribution comes from permuting the residuals of the null model among the cells of
each unit (Freedman & Lane 1983); it stays valid when units differ in variance. The profile-likelihood interval
of `h2` and the boundary likelihood-ratio test are diagnostics only: in simulations on real repertoire trees the
interval covered the true value in 65–87 % of datasets instead of 95 %, and the likelihood-ratio test rejected
in 16–38 % of datasets at a nominal 5 % when residuals were heavy-tailed or differed in variance between units.
Report the permutation p-value, and for an interval a bootstrap over clones (and over donors).

Because `h2` is a variance share and not a test statistic, `lineage_power` plants a component of known size on
the same trees and reports what the data could have detected.

## 11. Export lag (`export_lag`)

*In plain words.* "Are a clone's ASCs less mutated than its own germinal-centre cells, and by how much?"
The answer is the median across clones of a within-clone mutation gap. It says how much less mutated output
cells typically are, not why.

Which clones and cells are compared:

* clones captured in both compartments, the germinal-centre states and the output states named by the user;
* by default class-switched cells only, because switching is an irreversible event that mutation counts
  do not describe;
* cells matched on the columns given in `within`, such as visit and tissue.

For each such clone, the gap is the mean V-region mutation frequency of its output cells minus that of its
germinal-centre cells. Each clone is compared with itself, so differences between donors, visits and sorts
do not enter the gap. The gap varies in sign between clones: output cells were the more mutated in 8 of 92
mouse and 821 of 2,564 human lymph-node clones.

The estimate is the median gap across clones. Its 95 % interval is the pair of order statistics that inverts
the sign test (Hollander, Wolfe & Chicken 2014). It covers the median with probability at least 95 % for any
distribution, including the many tied values that mutation counts produce, and the sign test of a zero
median is reported with it. Clones are the unit, so the interval is conditional on the donors sampled.
With `donor_key`, each donor's median and interval are reported. With at least six donors, the median of the
donor medians receives its own sign-test interval, which describes the population of donors.

A negative gap has several explanations that the data cannot separate:

* output that left the germinal centre before it accumulated further mutations;
* extrafollicular antibody-secreting cells of the same clone (Jacob & Kelsoe 1992);
* affinity-biased export of plasma cells (Phan et al. 2006; Kräutler et al. 2017);
* regulated hypermutation, in which high-affinity cells divide more often with fewer mutations per division
  (Merkenschlager et al. 2025).

With a time column, the germinal centre's rate of mutation converts the gap into days.

* **Rate:** the Theil–Sen slope through the per-time-point medians of the germinal-centre cells
  (Theil 1950; Sen 1968). Each time point counts once, because cells of one visit are not independent
  observations of the trend.
* **Rate interval:** Sen's exact distribution-free interval, which needs at least five time points.
* **Days interval:** the range of gap / rate over the 97.5 % intervals of both, so it covers at least 95 %.
  It is not reported when the rate's interval does not exclude zero.
* **Assumptions:** every clone mutates at the population's average rate, and this rate is constant per day,
  which regulated hypermutation does not support. Days are a scale, not dates.
* **`clock="cells"`** keeps the earlier fit over all cells, which treats cells as independent and has no
  interval.

The gap does not establish a parent–offspring relation between particular cells.

### Centring on the sampling context, and when it misleads

Centring removes each context's average before clones are compared. This removes technical differences
between samples, but it has a side effect when contexts differ in their *cell-type composition*: if one
donor's sampled cells are mostly ASCs, that donor's average lies close to the ASC state, and subtracting it
shifts all of that donor's clones together. In a gut immunisation dataset this placed the ASC clones of one
mouse, whose sampled response was dominated by lamina-propria ASCs, in a branch of their own (manuscript,
Supplementary Fig. 5). Three habits avoid misreading such a result:

1. examine programmes and map regions **by donor** (`tf.pl.clone_map(adata, color="donor")`);
2. repeat the analysis **without centring** (`context_key=None`; in `threadfin.run`, `sample_key=None` with
   the libraries as `strata_key`, so that the coherence test still shuffles within libraries) and keep only
   conclusions that survive;
3. choose as the context a **nuisance** variable (donor, library), never the biology under study.

### A note on counting programmes

`find_programmes` reports one programme when no split is significant. The split test rarely rejects a single
Gaussian group (it is conservative under its own null), and its power depends on the direction of a separation:
the null is drawn with the covariance of the clones themselves, which a separation along their main direction of
variation inflates. **A single programme is therefore not evidence of a continuum.** Conversely, graded variation
along a path or a curve departs from a single Gaussian group, so **several programmes are not evidence of discrete
types of clones**: a programme is a partition of the clone profiles, to be read with its stability. See
`split_test` for the measured behaviour of both available nulls.

### Reproducibility

A run gives the same result for any `n_jobs`, and repeating it on the same data with the same `random_state` gives
the same result. The following settings also matter for reproducibility:

* **Clone calling.** Threadfin calls Dandelion under hash seed 0 in every session, in a separate Python process
  when needed, so no setting is needed for identical clone ids. `hash_seed=None` keeps the earlier in-session
  behaviour (section 1).
* **BLAS threads.** With another number of BLAS threads (`OMP_NUM_THREADS`), the computations (from the expression
  embedding on) differ at floating-point precision. Every statistic agrees to about 1e-7, but the automatically
  chosen clone map can
  change; keep the thread count the same between runs that must give the same map.
* **Order of the cells.** Random draws are made by position (the pairs of cells that set the kernel bandwidth,
  the bootstrap weights), so the same cells in another order can give different programmes. Sort the cells by
  name (`adata = adata[adata.obs_names.sort_values()].copy()`) when results must not depend on their order.

## References

* Anderson MJ. A new method for non-parametric multivariate analysis of variance. *Austral Ecol* 2001;26:32-46.
* Böhm JN, Berens P, Kobak D. Attraction-repulsion spectrum in neighbor embeddings. *J Mach Learn Res* 2022;23:1-32.
* Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57:289-300.
* Cliff AD, Ord JK. *Spatial Processes: Models and Applications*. Pion, 1981.
* Fisher RA. *Statistical Methods for Research Workers*. Oliver & Boyd, 1925.
* Freedman D, Lane D. A nonstochastic interpretation of reported significance levels. *J Bus Econ Stat* 1983;1:292-298.
* Gao LL, Bien J, Witten D. Selective inference for hierarchical clustering. *J Am Stat Assoc* 2024;119:332-342.
* Goeman JJ, Bühlmann P. *Bioinformatics* 2007;23:980-987.
* Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods* 2023;20:1196-1202.
* Gupta NT et al. Hierarchical clustering can identify B cell clones with high confidence in Ig repertoire sequencing data. *J Immunol* 2017;198:2489-2499.
* Hennig C. Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* 2007;52:258-271.
* Hollander M, Wolfe DA, Chicken E. *Nonparametric Statistical Methods*, 3rd ed. Wiley, 2014.
* Jacob J, Kelsoe G. *J Exp Med* 1992;176:679-687.
* Kingma DP, Ba J. Adam: a method for stochastic optimization. *ICLR* 2015.
* Kobak D, Berens P. The art of using t-SNE for single-cell transcriptomics. *Nat Commun* 2019;10:5416.
* Korsunsky I et al. Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat Methods* 2019;16:1289-1296.
* Kräutler NJ et al. *J Exp Med* 2017;214:1259-1267.
* Lin LI. A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 1989;45:255-268.
* Liu Y, Hayes DN, Nobel A, Marron JS. Statistical significance of clustering for high-dimension, low-sample size data. *J Am Stat Assoc* 2008;103:1281-1293.
* Mantel N, Haenszel W. Statistical aspects of the analysis of data from retrospective studies of disease. *J Natl Cancer Inst* 1959;22:719-748.
* McArdle BH, Anderson MJ. Fitting multivariate models to community data: a comment on distance-based redundancy analysis. *Ecology* 2001;82:290-297.
* McInnes L, Healy J, Saul N, Großberger L. UMAP: uniform manifold approximation and projection. *J Open Source Softw* 2018;3:861.
* Merkenschlager J et al. *Nature* 2025;641:495.
* Moran PAP. Notes on continuous stochastic phenomena. *Biometrika* 1950;37:17-23.
* Muandet K, Fukumizu K, Sriperumbudur B, Schoelkopf B. Kernel mean embedding of distributions: a review and beyond. *Found Trends Mach Learn* 2017;10:1-141.
* Phan TG et al. *J Exp Med* 2006;203:2419-2424.
* Phipson B, Smyth GK. Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 2010;9:39.
* Rahimi A, Recht B. Random features for large-scale kernel machines. *NeurIPS* 2007.
* Robins J, Breslow N, Greenland S. Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. *Biometrics* 1986;42:311-323.
* Searle SR, Casella G, McCulloch CE. *Variance Components*. Wiley, 1992.
* Sen PK. Estimates of the regression coefficient based on Kendall's tau. *J Am Stat Assoc* 1968;63:1379-1389.
* Spall JC. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Trans Autom Control* 1992;37:332-341.
* Suo C et al. Dandelion uses the single-cell adaptive immune receptor repertoire to explore lymphocyte developmental origins. *Nat Biotechnol* 2024;42:40-51.
* Theil H. A rank-invariant method of linear and polynomial regression analysis. *Proc K Ned Akad Wet* 1950;53:386-392, 521-525, 1397-1412.
* Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. *Sci Rep* 2019;9:5233.
* Wu D, Smyth GK. *Nucleic Acids Res* 2012;40:e133.

## Optional frozen-reference state distributions

See [STATE_DENSITY.md](STATE_DENSITY.md) for the separate region-count model,
conditional credible intervals, exact sampling intervals, regional annotation
and train-only RNA projection. The default mean/kernel workflow remains intact.
