# Threadfin methods (v4)

This document specifies every statistical step in Threadfin, the defaults, and
why each choice was made. It is written to be cited in a Methods section.
Function names refer to `threadfin.tl` unless stated otherwise.

---

## 1. Input and notation

Paired single-cell RNA-seq and BCR-seq data: for every cell *i* an expression
profile, a heavy-chain V(D)J rearrangement (optionally a light chain), the
donor *d(i)* and the sample *s(i)* (one library / sort gate / tissue / time
point of one donor). A **clone** *c* is a set of cells that descend from one
V(D)J recombination event. *n_c* is the number of sampled cells of clone *c*.

## 2. Clone definition (`define_clones`)

Clones are defined from heavy-chain sequences **within each donor** (two people
never share a clone; identical sequences across donors are convergent). Two
sequences can be related only if they share the IGHV gene, the IGHJ gene
(first call, allele removed) and the junction length. Within such a partition,
sequences closer than a threshold *t* in length-normalised Hamming distance
(nucleotides by default) are joined by single linkage (Gupta et al. 2017;
Nouri & Kleinstein 2018).

`threshold="auto"` estimates *t* from the distribution of each unique
sequence's distance to its nearest neighbour in its partition, which is
bimodal (clonal relatives near zero, unrelated sequences far away). A Gaussian
kernel density is evaluated on [0, 0.6]; *t* is the density minimum between the
first mode below 0.35 and the next mode, constrained to [0.02, 0.35] (the
"density" method of SHazaM). If no valley is found the default is 0.15 for
nucleotides and 0.20 for amino acids, with a warning. Optionally
(`light_chain=True`), heavy-chain clones are split by light chain (V gene, J
gene, junction length); cells without a light chain join the clone's most
common light-chain group.

## 3. Expression embedding (`threadfin.pp.prepare_embedding`)

Counts are normalised to 10,000 per cell and log1p-transformed. Highly variable
genes (3,000, `flavor="seurat"`, batch-aware) are selected **after removing
immunoglobulin and TCR genes** (V/D/J segments, constant regions IGHM, IGHD,
IGHG1-4, IGHA1/2, IGHE, IGKC, IGLC1-7, surrogate light chains; human and mouse
symbols; genes such as IGHMBP2 or JCHAIN are kept). Receptor transcripts are
identical within a clone, so leaving them in makes clonally related cells look
transcriptionally similar for a trivial reason and turns isotype into a
circular validation label. Genes are scaled (max 10), 30 principal components
are computed, and optionally integrated over technical batches with Harmony
(Korsunsky et al. 2019). Integrate over technical factors only (donor,
library), never over biology to be studied (tissue, time, sort gate).

## 4. The clone-state model (`clone_profiles`)

For a feature *j* of the embedding (or of a feature map, §5), Threadfin fits the
one-way random-effects model

    x_ij = mu_{s(i) j} + u_{c(i) j} + e_ij ,   u_cj ~ N(0, tau2_j),   e_ij ~ N(0, sigma2_j)

* **Context centring.** `mu_s` is the mean of *all* cells of context *s*
  (`context_key`, e.g. the donor or the sample). A clone is described
  relative to the cells it was sampled with, which removes technical shifts
  between libraries and sampling differences that would otherwise be
  attributed to clone identity.
* **Variance components.** With residuals *r_i = x_i - mu_s(i)* and the
  clones with at least two cells, the ANOVA (method-of-moments) estimators for
  an unbalanced one-way design are used (Searle, Casella & McCulloch 1992):
  `sigma2 = MS_within`, `tau2 = max(0, (MS_between - MS_within) / n0)` with
  `n0 = (N - sum n_c^2 / N) / (C - 1)`.
* **Clonal intraclass correlation (ICC).**
  `ICC = sum_j tau2_j / sum_j (tau2_j + sigma2_j)`: the fraction of
  within-context transcriptional variance explained by clone identity.
* **Shrunken clone profiles (BLUPs).** For clone *c*,
  `u_hat_cj = lambda_cj * (rbar_cj - rbar_j)` with
  `lambda_cj = n_c tau2_j / (n_c tau2_j + sigma2_j)`. Small clones are pulled
  towards their context; large clones keep their own mean; features without
  clonal variance (`tau2_j = 0`) are removed.
* **Reliability.** `R_c = sum_j lambda_cj tau2_j / sum_j tau2_j`. For a common
  ICC rho this is the Spearman-Brown formula `n rho / (1 + (n - 1) rho)`; it
  converts the arbitrary "minimum clone size" of other tools into a
  data-driven quantity: the clone size at which a profile is reliable is
  reported for every dataset.

## 5. Distributional (kernel) profiles

A clone whose cells are split between two states has the same centroid as a
clone sitting between them. With `representation="kernel"` (the default for
programmes) every cell is mapped to *D* = 256 random Fourier features of its
context-centred embedding position (Rahimi & Recht 2007), approximating a
Gaussian kernel with bandwidth equal to the median distance of a cell to its
30th nearest neighbour (a local scale that matches the width of one state;
the global median-distance heuristic blurs neighbouring states). The features
are centred again by context and passed through the same random-effects model,
so a clone's profile is a shrunken difference between the kernel mean
embedding of its cells and that of its context; Euclidean distances between
profiles approximate the maximum mean discrepancy between cell distributions.
Profiles are reduced to 30 principal components.

## 6. Clonal programmes (`find_programmes`)

Clones with reliability >= 0.5 are linked in a k-nearest-neighbour graph
(*k* = 15, Gaussian weights with a per-clone bandwidth equal to the distance to
the *k*-th neighbour, symmetrised by the maximum) and partitioned with Leiden
(RB configuration model; Traag et al. 2019).

**Stability.** For *B* = 30 (pipelines: 50) Poisson bootstrap replicates, every
cell receives a Poisson(1) weight, clone profiles are re-estimated with the
fitted variance components, the graph is rebuilt and re-partitioned. Each
reference programme is matched to the replicate cluster with maximal Jaccard
overlap; the programme's stability is the mean best Jaccard (Hennig 2007).
Following Hennig, programmes with stability >= 0.75 are stable and those below
0.6 are flagged and should not be interpreted. A clone's confidence is the
fraction of replicates in which it falls in the matched cluster.

**Resolution.** Resolutions 0.2-2.0 are scanned; groups with fewer than five
clones are not programmes; resolutions leaving more than 10% of clones in such
groups are skipped. The finest resolution whose size-weighted stability is at
least 0.75 is selected (otherwise the most stable one, with a warning). Fewer
than 30 reliable clones: programmes are not defined and only the coherence
test is reported.

## 7. Statistical tests

**The clone is the unit of replication.** Cells of one clone share ancestry
and usually a sample; testing at cell level (as Fisher tests on cell counts do)
counts one expanded clone hundreds of times.

**Clonal coherence (`clonal_coherence`).** Statistic: the ICC (§4) on the
mean representation. Null: clone labels are permuted among BCR+ cells *within*
strata (`strata_key`, default the sample), which preserves each stratum's
cell-state composition and clone-size distribution and breaks only the
cell-to-clone link. P-value: `(1 + #{null >= observed}) / (B + 1)`
(Phipson & Smyth 2010), *B* = 200-500. A null that shuffles across samples
(or only within donor) reports clonality whenever clones are confined to
samples with different compositions — which they nearly always are.

**Programme-label associations (`association_test`).** Cell-level labels are
collapsed per clone (majority with a unique mode covering >= 50% of the clone's
labelled cells; mean for numbers; `fraction:<level>` for the share of a
clone's cells in a gate). *Categorical labels*: for every programme x level,
the observed number of clones, its expectation under within-stratum
permutation, the Mantel-Haenszel odds ratio across strata (donors) with a
Robins-Breslow-Greenland 95% confidence interval, and a two-sided permutation
p-value from 2,000 within-donor permutations of clone labels. *Numeric labels*:
per programme, the median in and out of the programme, a rank effect (mean
rank difference / (N/2)) and a two-sided stratified permutation p-value; an
omnibus Kruskal-Wallis statistic with the same permutation scheme.
Benjamini-Hochberg FDR across all rows of a label. Donor-level labels (age,
severity) cannot be separated from donor effects by within-donor permutation
and are reported as such.

## 8. Clonal memory (`clonal_memory`)

For clones observed at two or more levels of `time_key` (time points, tissues,
sort gates) with at least three cells each, every (clone, level) *snapshot*
gets an unshrunk mean profile *m*. For consecutive snapshots *a*, *b* of the
same clone

    D_same = ||m_a - m_b||^2 - sum_j sigma2_j (1/n_a + 1/n_b)

where `sigma2` is the *within-snapshot* variance, so that *D* estimates the
change of the true state with sampling noise removed. `D_random` replaces *b*
by a snapshot of a different clone from the same donor and level. The
**memory index** is `M = 1 - mean(D_same) / mean(D_random)`: 1 when clones
retain everything that distinguishes them, 0 when a clone is no closer to its
former self than to a random clone (a test-retest reliability of clone state).
P-value: `mean(D_same)` against 500 random re-pairings; 95% CI: bootstrap over
clones. A descriptive programme transition table (nearest programme centroid
per snapshot, with its expectation under random pairing) is also reported.

Labels that are assigned per clone from all of its cells (any clone-level
clustering) are constant across a clone's snapshots by construction; their
"transition matrix" is diagonal whatever the biology, which is why
`community_transition` warns in that case.

## 9. Gene-level clonal heritability (`gene_heritability`)

The model of §4 is fitted gene by gene to context-centred log-normalised
expression, using cells of clones with >= 2 cells and genes expressed in >= 5%
of them; receptor genes are excluded from the scan and their median ICC is
reported as a positive control (they are clonal by definition). P-values come
from 100 within-stratum permutations of clone labels shared across genes
(vectorised); BH FDR. Gene-set summaries compare a set's ICCs with all other
tested genes (Mann-Whitney).

## 10. Simulation framework (`threadfin.sim`)

Cells are generated from donors, contexts with technical mean shifts (clones
nested in one context by default), cell states (Gaussian centres in a
20-dimensional embedding), clonal programmes (Dirichlet compositions over
states), clone-level compositions `Dirichlet(15 * theta_g)`, a heritable
clone offset, truncated-Zipf clone sizes (most clones are singletons) and,
optionally, several time points with a known probability of keeping the
programme. Scenarios: default, strong/no batch shift, clones spread over
contexts, small clones, *bifurcation* (two programmes with identical
centroids) and *null* (no clonal signal beyond sampling context).

## 11. Defaults at a glance

| step | default | rationale |
|---|---|---|
| clone threshold | density valley of distance-to-nearest (nt) | data-driven, standard in Immcantation |
| HVGs | 3,000, receptor genes excluded | removes trivial clonal similarity |
| embedding | 30 PCs, Harmony over technical batch | standard scverse practice |
| context | donor (pipelines), sample in `run()` | compare clones with co-sampled cells |
| programme representation | kernel, 256 features, local bandwidth | resolves mixtures of states |
| programme eligibility | reliability >= 0.5 | data-driven minimum clone size |
| stability | 30-50 Poisson bootstraps, Jaccard >= 0.75 | Hennig 2007 |
| coherence null | within-sample permutation | preserves sampling composition |
| association test | clone-level, within-donor permutation | no pseudoreplication |
| memory | noise-corrected snapshot distances | not circular |

## References

* Gupta NT et al. Hierarchical clustering can identify B cell clones with high confidence in Ig repertoire sequencing data. *J Immunol* 2017.
* Hennig C. Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* 2007.
* Korsunsky I et al. Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat Methods* 2019.
* Nouri N, Kleinstein SH. A spectral clustering-based method for identifying clones from high-throughput B cell repertoire sequencing data. *Bioinformatics* 2018.
* Phipson B, Smyth GK. Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 2010.
* Rahimi A, Recht B. Random features for large-scale kernel machines. *NeurIPS* 2007.
* Searle SR, Casella G, McCulloch CE. *Variance Components*. Wiley 1992.
* Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden. *Sci Rep* 2019.
