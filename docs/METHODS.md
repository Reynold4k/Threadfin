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
towards the sample average when the clone was seen in few cells (the *clone profile*). Clone profiles are then
compared: in a two-dimensional *clone map*, in groups of similar clones (*programmes*, kept only if they pass a
test and are reproducible), and in permutation tests that ask whether clone identity explains cell state
(*coherence*) and whether an annotation explains differences between clones (*association*). Every test uses
clones, not cells, as the unit, and shuffles labels only within the strata of the sampling design.

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
directly in ``adata.obs``. Dandelion 0.5.7 uses ``matplotlib.cm.get_cmap``, which matplotlib 3.9
removed, so install it with ``pip install "threadfin[dandelion]"`` (or ``pip install sc-dandelion
"matplotlib<3.9"``) or run it in its own environment. Dandelion's assignment depends on Python's string-hash
seed (set iteration order), so clone ids are identical between sessions only when ``PYTHONHASHSEED`` is fixed;
the seed in use is recorded with the result.

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
forms a clone of its own; Threadfin reports how many cells this affects (4–20 % of cells in the public datasets,
mostly two light chains). Optionally (`check_contigs=True`), Dandelion's contig check runs first: a cell with
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
them), and it is pulled towards the context average when only a few cells of the clone were seen. The pull
is strong for a 2-cell clone and weak for a 50-cell clone, which is how small clones are kept without
letting their noise dominate.

Each clone is described by where its cells sit **relative to the other cells
sampled with them** (same sample, or same donor), so that differences between
samples, tissues or batches are not mistaken for differences between clones.

Clone profiles come from a random-effects model (Searle et al. 1992): every
cell's position is its sample's average, plus a clone effect, plus cell-level
noise. From it Threadfin obtains

* the **clonal intraclass correlation (ICC)**: the share of transcriptional
  variation within samples that is explained by clone identity;
* a **shrunken profile** for each clone: small clones are pulled towards their
  sample's average, large clones keep their own;
* a **reliability** for each clone, which grows with the number of cells
  sampled (Spearman-Brown). Reliability sets how strongly a clone takes part
  in defining programmes; no clone is excluded for low reliability. A clone
  needs at least two cells, because a single cell cannot show whether cells
  of the same clone share a state.

By default for programmes (`representation="kernel"`), a clone's profile
describes the **whole distribution** of its cells, not only their average
(a kernel mean embedding; Rahimi & Recht 2007; Muandet et al. 2017). A clone
split between two states is then different from a clone sitting in between.
Each cell is first averaged with its 15 nearest cells to reduce noise.

## 4. Clonal coherence (`clonal_coherence`)

*In plain words.* "Do cells of the same clone resemble each other more than chance?" The answer is the
share of variation in cell state that clone identity explains — 0 if clones are random collections of
cells, 1 if all cells of a clone are identical — compared with what shuffled clone labels give. For
example, an ICC of 0.27 against 0.002 for shuffled labels means clone identity explains about a quarter of
the cell-to-cell variation within samples.

*Does clone identity shape cell state?* The clonal ICC is compared with the
same quantity after shuffling clone labels **among cells of the same sample**.
Shuffling within samples keeps every sample's cell-type composition, so
differences between samples cannot create apparent clonality. The result is
reported as the variance explained by clone, the shuffled value and a
permutation p-value (Phipson & Smyth 2010).

## 5. The clone map (`embed_clones`, `tune_clone_map`)

*In plain words.* A picture with one dot per clone, in which clones with similar profiles are close. It is
for looking, not for testing; Threadfin chooses its layout settings so that a continuous population of
clones is not broken into islands.

*How are clones related to one another?* Every clone with a profile is placed on a two-dimensional
map with UMAP (McInnes et al. 2018), one point per clone. Clones whose cells are distributed alike sit
close together, so continuous transitions between clones, for example from germinal-centre clones to
plasma-cell clones, appear as paths. The map is a display: profiles, coherence, programmes and all tests
are computed in profile space and do not depend on it. No clone is removed for low reliability.

Two UMAP settings decide what the map looks like: the number of neighbours each clone is tied to
(`n_neighbors`) and how closely clones may pack (`min_dist`). Too few neighbours or too little spacing
breaks a continuum of clones into islands; too many neighbours smooth away local structure. Because the
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
Finally, the map that is returned is laid out at UMAP's default settings and checked: if it falls short
of the target, two further seeds and then the next confirmed settings are tried, and the most coherent
layout is kept.

Everything the search did is stored in `adata.uns['threadfin']['profiles']['map_params']`: the chosen
settings, the quality of the returned map, every layout scored and every candidate confirmed.
`tl.map_quality` scores any map, including one made by hand. An alternative search, `method="adam"`,
follows a stochastic gradient instead of the ladder (Adam, Kingma & Ba 2015, with simultaneous-
perturbation gradient estimates, Spall 1992) and ends with the same confirmation and checks.

## 6. Clonal programmes (`find_programmes`)

*In plain words.* Groups of clones with similar profiles — but only groups that are better than one big
group (a split test) and that come back when the cells are resampled (stability). If clones vary along a
continuum, Threadfin says so instead of inventing groups.

*Which clones behave alike?* Clones are connected to their 15 most similar
clones, with each connection weighted by the reliability of the two clones,
and grouped by Leiden community detection (Traag et al. 2019); the resolution
is the finest one whose programmes are stable under resampling. Two
safeguards keep the groups meaningful:

* **Approximate split diagnostics.** Community detection always returns groups,
  even when clones vary continuously. Every split is therefore tested
  against a single-group model fitted to the same clones (a Gaussian
  cluster-index test, Liu et al. 2008, applied along the tree of groups with
  a hierarchical threshold allocation inspired by sc-SHC, Grabski et al. 2023).
  The full selected Leiden/tree/resolution process is not repeated in the null;
  normal-tail calibration is approximate, so FWER control is not established. Groups that
  are not significantly separated are merged. If no split survives, Threadfin
  reports a **continuum**: clones differ, but not as distinct groups.
* **Stability.** The cells are resampled 30-50 times and the whole procedure
  repeated; each programme's stability is how well it is recovered (mean
  Jaccard overlap; Hennig 2007). Programmes with stability >= 0.75 are
  stable; below 0.6 they should not be interpreted.

When programmes are defined on a subset of clones (`min_reliability` above
0), the remaining clones are afterwards assigned to the most likely programme,
taking their size into account (`clone_programme_assigned`).
Marker genes of each programme (`programme_markers`) are found by comparing
clone averages, so a large clone counts once. Programmes usually differ in which cell states their clones
contain, so these markers are largely the markers of those states; with `state_key` and `state`, each clone is
summarised by its cells in one state only (for example its germinal-centre B cells), centred on the cells of
that state in its context, which asks what differs between programmes within a state.

**Genes along the landscape (`landscape_genes`).** When clones vary continuously there are no programmes to
compare. Each profiled clone is then summarised by the mean context-centred log expression of its cells, the
clones are linked to their 15 nearest neighbours in profile space, and each gene is scored by Moran's I of the
clone means on this graph (Moran 1950): positive when neighbouring clones express the gene alike, about
−1/(n−1) when the gene ignores the landscape. P values come from the normal approximation under randomisation
(Cliff & Ord 1981), with Benjamini–Hochberg correction; in simulations the analytic mean and variance of I
matched 20,000 permutations. The rank correlation of each gene with the first principal component of the clone
profiles gives the direction of a gradient. Because the profiles are built from the same expression, this
names what varies across clones rather than testing an independent hypothesis.

## 7. What distinguishes clones (`profile_association`, `association_test`)

*In plain words.* "Does this annotation — antigen binding, isotype, a sort gate — line up with how clones
differ?" The annotation is summarised once per clone, and its association with the profiles is compared
with the association obtained after shuffling the annotation among clones of the same donor.

Cell-level labels (antigen binding, isotype, mutation load, sort gate, tissue)
are first summarised per clone (majority label, mean value, or the fraction
of a clone's cells in a gate). Two questions are then asked, always with
**clones as the unit** and labels shuffled among clones of the same donor:

* **Does the label explain how clones differ?** (`profile_association`) The
  share of the variation between clone profiles explained by the label
  (as in PERMANOVA; Anderson 2001). This works whether or not there are
  distinct programmes.
* **Which programmes differ?** (`association_test`, when there are at least
  two programmes) Odds ratios per programme across donors (Mantel & Haenszel
  1959) for categories, or rank effects for numbers, with false discovery
  rate control (Benjamini & Hochberg 1995).

Labels that are fixed for a donor (age, sex, disease severity) cannot be
separated from other differences between donors and are reported as such.

## 8. Clonal memory (`clonal_memory`)

*In plain words.* "Is a clone still recognisably itself at a later time point, or in another tissue?" A clone
seen twice is compared with itself and with a random other clone of the same donor.

*Do clones keep their state over time or across tissues?* For every clone seen
at two time points (or tissues, or sort gates), Threadfin measures how much
its profile changed, removes the part expected from sampling only a few
cells, and compares the change with the distance to a random other clone of
the same donor. The **memory index** is 1 when clones keep everything that
distinguishes them and 0 when a clone is no more similar to its earlier self
than to a random clone.

## 9. Clonally inherited genes (`gene_heritability`)

*In plain words.* The coherence question asked gene by gene: which genes are more similar among cells of one
clone than among unrelated cells?

The same random-effects model is fitted gene by gene: a gene's clonal ICC is
the share of its expression variation explained by clone. Genes are ranked by
how much they exceed shuffled clones, and gene sets (for example plasma-cell
or germinal-centre genes) are compared with genes of similar expression level,
because highly expressed genes are measured more precisely.

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
the restricted likelihood is a sum over rows and is maximised by a one-dimensional search; the interval comes
from the profile likelihood and the p-value from permuting the null residuals inside each unit (Freedman-Lane),
which stays valid when units differ in variance.

Because `h2` is a variance share and not a test statistic, `lineage_power` plants a component of known size on
the same trees and reports what the data could have detected.

### Centring on the sampling context, and when it misleads

Centring removes each context's average before clones are compared. This removes technical differences
between samples, but it has a side effect when contexts differ in their *cell-type composition*: if one
donor's sampled cells are mostly plasma cells, that donor's average lies close to the plasma-cell state, and
subtracting it shifts all of that donor's clones together. In a gut immunisation dataset this placed the
plasma-cell clones of one mouse, whose sampled response was dominated by lamina-propria plasma cells, in a
branch of their own (manuscript, Supplementary Fig. 5). Three habits avoid misreading such a result:

1. examine programmes and map regions **by donor** (`tf.pl.clone_map(adata, color="donor")`);
2. repeat the analysis **without centring** (`context_key=None`) or with a composition-balanced reference,
   and keep only conclusions that survive;
3. choose as the context a **nuisance** variable (donor, library), never the biology under study.

### A note on counting programmes

`find_programmes` reports one programme when no split is significant. The split test is conservative: with a
two-group structure planted in real clone profiles it did not detect a separation of four within-group
standard deviations. **A single programme is therefore not evidence of a continuum**; it means the data did not
pass a demanding test. See `split_test` for the measured calibration of both available nulls.

## References

* Anderson MJ. A new method for non-parametric multivariate analysis of variance. *Austral Ecol* 2001;26:32-46.
* Böhm JN, Berens P, Kobak D. Attraction-repulsion spectrum in neighbor embeddings. *J Mach Learn Res* 2022;23:1-32.
* Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57:289-300.
* Cliff AD, Ord JK. *Spatial Processes: Models and Applications*. Pion, 1981.
* Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods* 2023;20:1196-1202.
* Gupta NT et al. Hierarchical clustering can identify B cell clones with high confidence in Ig repertoire sequencing data. *J Immunol* 2017;198:2489-2499.
* Hennig C. Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* 2007;52:258-271.
* Kingma DP, Ba J. Adam: a method for stochastic optimization. *ICLR* 2015.
* Kobak D, Berens P. The art of using t-SNE for single-cell transcriptomics. *Nat Commun* 2019;10:5416.
* Korsunsky I et al. Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat Methods* 2019;16:1289-1296.
* Liu Y, Hayes DN, Nobel A, Marron JS. Statistical significance of clustering for high-dimension, low-sample size data. *J Am Stat Assoc* 2008;103:1281-1293.
* Mantel N, Haenszel W. Statistical aspects of the analysis of data from retrospective studies of disease. *J Natl Cancer Inst* 1959;22:719-748.
* McInnes L, Healy J, Saul N, Großberger L. UMAP: uniform manifold approximation and projection. *J Open Source Softw* 2018;3:861.
* Moran PAP. Notes on continuous stochastic phenomena. *Biometrika* 1950;37:17-23.
* Muandet K, Fukumizu K, Sriperumbudur B, Schoelkopf B. Kernel mean embedding of distributions: a review and beyond. *Found Trends Mach Learn* 2017;10:1-141.
* Phipson B, Smyth GK. Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 2010;9:39.
* Rahimi A, Recht B. Random features for large-scale kernel machines. *NeurIPS* 2007.
* Searle SR, Casella G, McCulloch CE. *Variance Components*. Wiley, 1992.
* Spall JC. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Trans Autom Control* 1992;37:332-341.
* Suo C et al. Dandelion uses the single-cell adaptive immune receptor repertoire to explore lymphocyte developmental origins. *Nat Biotechnol* 2024;42:40-51.
* Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. *Sci Rep* 2019;9:5233.

## Optional frozen-reference state distributions

See [STATE_DENSITY.md](STATE_DENSITY.md) for the separate region-count model,
conditional credible intervals, exact sampling intervals, regional annotation
and train-only RNA projection. The default mean/kernel workflow remains intact.
