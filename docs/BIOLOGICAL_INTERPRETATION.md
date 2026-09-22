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
Threadfin finds four clone communities. Community 3 is an island of large,
expanded clones that is **22.7× enriched for the plasmablast state**
(FDR ≈ 1e-111) and expresses plasmablast/interferon programmes. A sequence
network of the same data shows which clones are *related*; Threadfin shows
which clones are *doing the same thing* — here, mounting the antibody-secreting
response.

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
