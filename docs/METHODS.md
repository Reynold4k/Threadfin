# How Threadfin works

This page describes each step of `threadfin.run()` and the matching
functions in `threadfin.tl`, with the defaults and the reasoning behind them.
References are listed at the end.

## 1. Clones (`define_clones`)

A clone is the family of B cells descended from one V(D)J recombination event. Clones are defined from heavy-chain sequences **within each donor**
(two people never share a clone). Two cells can belong to the same clone only
if they use the same IGHV and IGHJ genes and their junctions have the same
length; within such a group, junctions that differ by less than a threshold
(fraction of differing nucleotides) are joined, and chains of such links form
one clone (single linkage), so that hypermutated relatives stay together
(Gupta et al. 2017).

The threshold is read from the data: the distance of every junction to its
closest relative has two peaks, one for clonal relatives near zero and one
for unrelated sequences, and the threshold is placed at the valley between
them (as in the SHazaM toolkit). Values outside the usual range (0.02-0.25)
are not accepted; the default 0.15 is used and reported instead. Light chains
can optionally split heavy-chain clones (`light_chain=True`).

## 2. Expression embedding (`threadfin.pp.prepare_embedding`)

Standard single-cell processing (normalisation, 3,000 highly variable genes,
30 principal components, optional Harmony integration over technical batches;
Korsunsky et al. 2019) with one important change: **immunoglobulin and TCR
genes are removed before choosing variable genes**. Cells of one clone share
their receptor transcripts, so leaving them in would make clonal relatives
look alike for a trivial reason, and would make isotype a circular label.
Integrate only over technical factors (donor, library), never over the biology
under study (tissue, time point, sort gate).

## 3. Clone profiles (`clone_profiles`)

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
  sampled (Spearman-Brown); clones with reliability >= 0.5 ("core" clones)
  are used to define programmes.

By default for programmes (`representation="kernel"`), a clone's profile
describes the **whole distribution** of its cells, not only their average
(a kernel mean embedding; Rahimi & Recht 2007; Muandet et al. 2017). A clone
split between two states is then different from a clone sitting in between.
Each cell is first averaged with its 15 nearest cells to reduce noise.

## 4. Clonal coherence (`clonal_coherence`)

*Does clone identity shape cell state?* The clonal ICC is compared with the
same quantity after shuffling clone labels **among cells of the same sample**.
Shuffling within samples keeps every sample's cell-type composition, so
differences between samples cannot create apparent clonality. The result is
reported as the variance explained by clone, the shuffled value and a
permutation p-value (Phipson & Smyth 2010).

## 5. Clonal programmes (`find_programmes`)

*Which clones behave alike?* Core clones are connected to their most similar
clones and grouped by Leiden community detection (Traag et al. 2019). Two
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

Clones too small to be core are afterwards assigned to the most likely
programme, taking their size into account (`clone_programme_assigned`).
Marker genes of each programme (`programme_markers`) are found by comparing
clone averages, so a large clone counts once.

## 6. What distinguishes clones (`profile_association`, `association_test`)

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

## 7. Clonal memory (`clonal_memory`)

*Do clones keep their state over time or across tissues?* For every clone seen
at two time points (or tissues, or sort gates), Threadfin measures how much
its profile changed, removes the part expected from sampling only a few
cells, and compares the change with the distance to a random other clone of
the same donor. The **memory index** is 1 when clones keep everything that
distinguishes them and 0 when a clone is no more similar to its earlier self
than to a random clone.

## 8. Clonally inherited genes (`gene_heritability`)

The same random-effects model is fitted gene by gene: a gene's clonal ICC is
the share of its expression variation explained by clone. Genes are ranked by
how much they exceed shuffled clones, and gene sets (for example plasma-cell
or germinal-centre genes) are compared with genes of similar expression level,
because highly expressed genes are measured more precisely.

## 9. Heritability along the receptor lineage tree (`lineage_heritability`)

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

### A note on counting programmes

`find_programmes` reports one programme when no split is significant. The split test is conservative: with a
two-group structure planted in real clone profiles it did not detect a separation of four within-group
standard deviations. **A single programme is therefore not evidence of a continuum**; it means the data did not
pass a demanding test. See `split_test` for the measured calibration of both available nulls.

## References

* Anderson MJ. A new method for non-parametric multivariate analysis of variance. *Austral Ecol* 2001;26:32-46.
* Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57:289-300.
* Grabski IN, Street K, Irizarry RA. Significance analysis for clustering with single-cell RNA-sequencing data. *Nat Methods* 2023;20:1196-1202.
* Gupta NT et al. Hierarchical clustering can identify B cell clones with high confidence in Ig repertoire sequencing data. *J Immunol* 2017;198:2489-2499.
* Hennig C. Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* 2007;52:258-271.
* Korsunsky I et al. Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat Methods* 2019;16:1289-1296.
* Liu Y, Hayes DN, Nobel A, Marron JS. Statistical significance of clustering for high-dimension, low-sample size data. *J Am Stat Assoc* 2008;103:1281-1293.
* Mantel N, Haenszel W. Statistical aspects of the analysis of data from retrospective studies of disease. *J Natl Cancer Inst* 1959;22:719-748.
* Muandet K, Fukumizu K, Sriperumbudur B, Schoelkopf B. Kernel mean embedding of distributions: a review and beyond. *Found Trends Mach Learn* 2017;10:1-141.
* Phipson B, Smyth GK. Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 2010;9:39.
* Rahimi A, Recht B. Random features for large-scale kernel machines. *NeurIPS* 2007.
* Searle SR, Casella G, McCulloch CE. *Variance Components*. Wiley, 1992.
* Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. *Sci Rep* 2019;9:5233.

## Optional frozen-reference state distributions

See [STATE_DENSITY.md](STATE_DENSITY.md) for the separate region-count model,
conditional credible intervals, exact sampling intervals, regional annotation
and train-only RNA projection. The default mean/kernel workflow remains intact.
