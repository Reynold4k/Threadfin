# Biological interpretation of Threadfin

*Audience: this document explains what Threadfin means biologically. It is
written for both computational biologists and experimental immunologists; no
implementation details are required to read it.*

---

## What a normal clonotype network tells you

Classic single-cell BCR repertoire analysis (Cell Ranger + scirpy,
scRepertoire, Immcantation, dandelion) is built around the **sequence** of
the B-cell receptor:

* each **node** is a cell or a clonotype (a group of cells with the same or
  near-identical V(D)J rearrangement);
* **edges** mean *sequence relatedness* — same CDR3, same V/J genes, few
  somatic mutations apart;
* the typical questions are: *Which cells are clonally related? How expanded
  is each clone? Which clones share CDR3 motifs? What do lineage trees look
  like?*

This view is powerful for **ancestry and specificity**, but it is blind to
**behaviour**. Two clones with completely different receptors may be doing
exactly the same thing (e.g. both becoming plasmablasts), and one expanded
clone can contain cells in very different states. A sequence network cannot
see any of that, because it never looks at the transcriptome.

## What Threadfin adds

Threadfin asks a different question. Instead of *"who is sequence-related?"*
it asks:

> **"What are the clones doing — and which clones are doing the same thing?"**

Concretely, Threadfin:

1. places every clonotype at the **average transcriptional position of its
   cells** (its "centroid") in a gene-expression space;
2. connects clonotypes that sit in the **same region of cell-state space**,
   forming **clone communities**;
3. labels every cell with the community of its clone, so the community
   structure can be inspected on the normal cell UMAP;
4. can order clone communities along **clone-level pseudotime**, describing
   how lineages traverse transcriptional states;
5. can optionally blend CDR3 sequence distance into the clone distance
   (`cdr3_weight`), testing whether sequence-similar clones also behave
   similarly.

### Reading a Threadfin clone map

| Feature of the map | Biological meaning |
|---|---|
| one point | one clonotype (a genetically distinct B-cell lineage) |
| point size | clone expansion (number of cells) |
| point position | the transcriptional state of that clone's cells |
| clone cluster (colour) | a **community of clones converging on the same state/fate** |

### Questions Threadfin answers that a sequence network cannot

| Question | How Threadfin addresses it | Where implemented |
|---|---|---|
| Which genetically **distinct** clones converge on the same B-cell state or fate? | clone communities = clusters of centroids in expression space | `clonotype_recluster` |
| Which **expanded** clones behave similarly despite different BCR sequences? | large points grouped by community, independent of sequence | clone map (size = expansion) |
| Which sequence-related clones **diverge** transcriptionally? | same VDJ key / similar CDR3 but different community; or cells of one clone spanning communities | `metrics.clone_state_purity` (entropy), `cdr3_weight` comparison |
| Which clones are transcriptionally **stable vs plastic**? | per-clone purity/entropy across cell states | `metrics.clone_state_purity` |
| Which communities occupy GC / memory / plasma / activated states? | Fisher-exact enrichment of curated or de-novo states per community | `metrics.state_enrichment` |
| Do some communities appear **earlier or later** in a response? | enrichment of timepoint labels per community; clone-level pseudotime | `state_enrichment` on timepoint; `clonal_pseudotime` |
| Does **sequence similarity agree with fate similarity** — or do they decouple? | compare expression-only vs sequence-only vs blended clone groupings | `cdr3_weight` grid in the validation pipeline |

### Worked example from the bundled benchmark

On 5,000 B-lineage cells from COVID-19 patients (Stephenson et al. 2021),
Threadfin finds a clone community that is **~198× enriched for the
plasmablast state** (FDR ≈ 6e-34) and expresses plasmablast/interferon
programmes. A sequence network of the same data shows which clones are
*related*; Threadfin shows which clones are *doing the same thing* — here,
mounting the antibody-secreting response. (Note: the bundled 5k subset is
sparse, so exact-CDR3 clonotypes are mostly singletons; the benchmark config
uses `min_clone_size=2` and the community count is small — see the dataset
report for an honest discussion.)

## Evidence across five public datasets

All numbers below come from `benchmarks/biological_validation/results/`
(per-dataset `summary.json` / `report.md`; cross-dataset table in
`results/cross_dataset_summary.md`). Every dataset was processed with the
same pipeline; dataset-specific parameters are declared in the JSON configs.

| dataset | cells | clones | expanded (>=3) | null1 purity vs null (p) | held-out vs chance | basis ARI |
|---|---|---|---|---|---|---|
| Stephenson 2021 COVID PBMC (5k subset) | 5,000 | 4,823 | 22 | 0.973 vs 0.817 (0.005) | n/a (only 2 communities) | 0.944 |
| Flu vaccine PBMC (Wang 2023, GSE175522) | 123,693 | 80,682 | 499 | 0.883 vs 0.356 (0.005) | 0.97 vs 0.26 | 0.313 |
| Tonsil (King 2021, E-MTAB-9005/9003) | 22,478 | 10,473 | 200 | 0.662 vs 0.489 (0.005) | 0.99 vs 0.66 | 0.000 |
| EBV tonsil organoid (Mitul 2026, GSE317492) | 205,630 | 89,757 | 4,564 | 0.791 vs 0.314 (0.005) | 0.985 vs 0.101 | 0.352 |
| LN vaccine GC (Kim 2022, GSE195673) | 193,442 | 92,761 | 4,396 | 0.849 vs 0.478 (0.005) | 0.928 vs 0.087 | 0.234 |

Reading of the table:

* **Null 1 (within-donor clone-label permutation) is rejected on every
  dataset** (p <= 0.005): clones are far more transcriptionally pure than
  chance. This is the core claim — clonal identity carries transcriptional
  information — and it holds across tissues, diseases and sequencing
  protocols.
* **Held-out split-clone validation is strong wherever more than a handful
  of communities exist**: sibling halves of the same clone re-co-cluster at
  0.93–0.99 vs chance 0.09–0.26 (flu/EBV/LN). On Stephenson the test is
  uninformative (two coarse communities make the chance level ~1.0) and we
  say so rather than quoting the number.
* **Fine-grained community boundaries are basis-dependent** (clone-level ARI
  0.23–0.35 between PCA and UMAP bases on the three large datasets; 0.0 on
  tonsil). We therefore state conclusions at the level of *enrichments*,
  not boundaries — and the headline enrichments do replicate across bases
  (e.g. tonsil plasmablast community: OR 5.8–26.9, FDR <= 1e-13 in all
  runs that resolve the community).

### Per-dataset biology

* **LN vaccine (Kim 2022)** — the flagship dataset (8 donors, 193k B cells,
  GC + blood + bone-marrow compartments). Communities are strongly enriched
  for author-annotated states (17–37 significant enrichments per run; null 3
  p = 0.015). Communities have coherent **isotype** compositions (e.g. one
  community is 80% IGHG, another 88% IGHG, others IGHM/IGHD-rich), and a
  **clonal-state transition matrix across the vaccination time course is
  essentially diagonal**: a clone's community membership at d28 predicts its
  membership at d60/d110 — clonal transcriptional identity is stable across
  the response (with the caveat that communities are coarse).
* **Flu vaccine (Wang 2023)** — 711 clones are observed at both d0 and d7;
  expanded d7 plasmablast clones concentrate in a small number of
  communities (top enrichment OR 290, FDR ≈ 0). Clone state-purity is the
  highest of all datasets relative to its null (0.883 vs 0.356).
* **Tonsil (King 2021)** — a methodological finding in itself: exact
  cellranger clonotypes in these libraries are almost all singletons (13
  clones with >=3 cells), and only Threadfin's **sequence-similarity clone
  definition** (`define_clones`, merging SHM variants within the same V/J
  into lineages) recovers an analysable clonal landscape (200 expanded
  lineages). A plasmablast community is recovered under both bases.
* **EBV organoid (Mitul 2026)** — highest concordance with de-novo states
  (NMI 0.67) and the largest expanded-clone set; the joint GEX+BCR
  embedding finds 20 communities with NMI 0.59.

### What the BCR modality does — and does not — add at clone level

The v2 joint embedding quantifies each modality's role honestly:

* The clone-level BCR graph is **sparse by biology, not by bug**: distinct
  clonotypes sharing >=0.85 CDR3 identity within the same V/J are rare
  (0–1,389 edges per dataset). Consequently the joint embedding is
  GEX-dominated on all five datasets (modality contribution <= 19% BCR).
  On tonsil there are *zero* inter-clone BCR edges — after SHM merging,
  lineages are sequence-islands.
* Where enough BCR edges exist to judge (EBV: 1,389), latent distances
  correlate positively with BCR similarity (testcor_bcr = 0.31); on LN
  the correlation is weakly negative (-0.12). We report these as-is: on
  current public data, sequence convergence between *distinct* clones is
  too rare to reweight clone-level geometry. The BCR modality's real value
  in Threadfin is **upstream** (sequence-aware clone definition, which
  rescued the tonsil dataset) and in **annotation** (isotype/SHM per
  community), not in blending distances.
* The v1 `cdr3_weight` blend illustrates why naive distance mixing fails:
  on LN it collapses concordance (NMI 0.196 -> 0.016) because z-scored
  near-degenerate sequence distances dominate the blend. It is kept for
  backward compatibility; the graph-based joint embedding is the
  recommended path.

## What Threadfin does **not** claim

* **No lineage inference.** Threadfin is not a phylogenetic method and does
  not replace Immcantation-style lineage trees. A clone community is *not* a
  family of shared ancestry — it is a group of clones with similar behaviour.
* **No causality.** Enrichment of a state in a community is associational.
* **Communities are resolution- and basis-dependent.** We therefore report a
  robustness analysis (PCA vs UMAP basis) and null models for every dataset;
  communities that appear only under one embedding should be treated with
  caution.
* **Small clones are noisy.** A centroid of 1–2 cells is unreliable, hence
  the `min_clone_size` filter; conclusions rest on expanded clones.

## How the claims are validated

The multi-dataset validation framework in
[`benchmarks/biological_validation/`](../benchmarks/biological_validation/)
operationalizes every claim above:

| Claim | Validation |
|---|---|
| Clone communities reflect real transcriptional structure | **Null 1**: clone-label permutation within donor (preserves size distribution) |
| Communities correspond to biology, not community-size artefacts | **Null 3**: size-matched random communities |
| Sequence-only grouping does not explain communities | **Null 2**: `cdr3_weight = 1.0` (sequence-only) run compared to expression-based runs |
| Communities are not an artefact of the specific cells sampled | **Held-out split-clone validation**: cells of each large clone are split into discovery/validation halves treated as independent pseudo-clones; sibling halves must re-co-cluster above chance |
| Communities are not a UMAP artefact | **Basis robustness**: PCA- vs UMAP-based runs compared at clone level |
| Communities replicate across donors | per-donor re-runs where donor counts allow |
| Communities reflect experimentally defined biology | enrichment of **external labels not used to build the graph**: vaccination timepoint, EBV GFP+ infection status, author-annotated cell types, tissue compartment |

Negative results are reported as-is: a dataset with too few expanded clones
or weak transcriptional heterogeneity is expected to yield weak communities,
and we say so in the per-dataset report.
