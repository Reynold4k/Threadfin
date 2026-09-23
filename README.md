# Threadfin

**Clonotype-level inference of B-cell transcriptional states and fates from paired scRNA-seq + scBCR-seq data.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org)

Threadfin integrates paired single-cell BCR and transcriptome data to place
every B-cell clonotype into a transcriptional state space. Classic repertoire
tools ask *"which cells share a receptor sequence?"* — Threadfin asks the
orthogonal question:

> **"What are the clones doing — and which genetically distinct clones are
> doing the same thing?"**

Each clonotype is placed at the centroid of its member cells in a
gene-expression embedding; clonotypes are then clustered in that space into
**clone communities** — groups of genetically distinct clones that converge
on the same transcriptional state or fate. Communities are mapped back to
cells for inspection, and clonal state transitions are inferred across real
timepoints or along a diffusion-based ordering of the clone graph. Version 2
adds a BCR-sequence layer (BLOSUM62-aware CDR3 similarity, sequence-aware
lineage grouping, and a coupled GEX+BCR graph embedding inspired by Benisse)
so the two modalities can be integrated rigorously — plus diagnostics that
state honestly when one modality contributes nothing.

## Why infer states at the level of clones?

Two properties of real B-cell repertoires motivate Threadfin's design.

**1. Repertoires are dominated by singletons; lineage trees cover a small,
unrepresentative minority.** In every dataset analysed below, the median
clonotype size is exactly one cell, and clonotypes expanded to ≥3 cells make
up only 0.5–5% of the repertoire. This is not specific to our data:
re-analysis of large immune repertoires likewise found that *"the majority
of clonotypes were singletons and only 9–18% of patients' clonotypes were
clonally expanded"* ([Sturm et al. 2020](https://doi.org/10.1093/bioinformatics/btaa611)).
Phylogenetic methods in the Immcantation/SCOPer tradition
([Gupta et al. 2015](https://doi.org/10.1093/bioinformatics/btv359);
[Nouri & Kleinstein 2018](https://www.frontiersin.org/articles/10.3389/fimmu.2018.00682/full))
can only build lineage trees for that small expanded fraction — and even
where a tree can be inferred, it encodes the **ancestry of the receptor
sequence** (which somatic mutation descended from which), not what the cells
are doing. A clone tree cannot, by construction, tell you whether a clone is
becoming a plasmablast or a memory cell.

**2. Receptor identity ≠ cell state, in both directions.** Cells of the same
clonal group routinely occupy different transcriptional clusters — *"even
when cells belonged to the same BCR or TCR clonal group, they could still be
found in different transcriptional clusters … lymphocytes sharing the same
immune receptor specificity may still undergo very different cell fates and
functions in the course of an immune response"* ([Yermanos et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046018/)).
Conversely, genetically unrelated clones converge on the same state during a
response: affinity-matured sister clones diverge in sequence while sharing
fate. A sequence network is blind to both directions because it never looks
at the transcriptome; a cell-level clustering cannot see that this
convergence happens **clone by clone**, not cell by cell.

Threadfin therefore treats the **clone as the unit of inference in
transcriptional state space** — the same conceptual question Benisse posed
(*"refined analyses of BCRs guided by single-cell gene expression"*,
[Zhang et al. 2022](https://www.nature.com/articles/s42256-022-00492-6)),
operationalized inside the scanpy ecosystem with explicit permutation nulls,
held-out replication, and basis-robustness checks for every claim.

---

## What Threadfin finds in real data

Everything in this section is reproduced end-to-end by
[`benchmarks/biological_validation/`](benchmarks/biological_validation/)
(per-dataset configs, loaders, reports and figures are committed; dataset
sources and checksums in [`benchmarks/datasets_manifest.tsv`](benchmarks/datasets_manifest.tsv)).

### 1. Lymph-node germinal-centre response to mRNA vaccination
*[Kim et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04527-1) — GSE195673 / [Zenodo 5895181](https://zenodo.org/records/5895181) (8 donors, 193,442 B cells, LN FNA + blood + bone marrow, d28–d201)*

This study tracked the SARS-CoV-2 mRNA-vaccine germinal-centre (GC) reaction
in draining axillary lymph nodes by serial fine-needle aspiration over
months, showing that vaccine-induced GCs are durable and keep producing
affinity-matured clones long after the blood response wanes. It is the
richest setting here: 92,761 clonotypes (median size 1; 4,396 expanded to
≥3 cells) across author-annotated GC, memory (RMB), naive, plasmablast (PB)
and lymph-node plasma-cell (LNPC) states.

Threadfin resolves **11 clone communities** over the 4,396 expanded clones
(up to 24 at higher resolution), and the communities are biologically
coherent in three independent ways:

* **Cell-state composition.** 17–37 significant enrichments per parameter
  setting against the authors' annotations (size-matched permutation null
  p = 0.015) — e.g. the largest community maps onto GC cells (17,464 cells,
  OR 2.4, FDR ≈ 0), and at higher resolution a dedicated plasmablast
  community emerges (OR 156, FDR ≈ 0).
* **Class-switch status.** Isotype was never used to build the graph, yet
  communities have internally consistent isotype composition: the three
  largest communities are 80%, 88% and 83% **IGHG** (class-switched), while
  others are IGHM/IGHD-rich. Community membership and somatic class-switch
  history — two independent molecular layers — agree.
* **Temporal stability of clonal identity.** For the 1,016 clone transitions
  observed between successive timepoints (d28 → d60 → d110), the community
  transition matrix is **perfectly diagonal (1,016/1,016)**: the community a
  clone occupies early in the response is exactly the community it occupies
  months later. Clonal transcriptional identity is thus a stable, heritable
  property of the lineage across the GC reaction — consistent with the
  durable, continuously maturing GC clones reported by the original study.

Statistical support: clone state-purity 0.849 vs 0.478 for the within-donor
permutation null (p = 0.005); held-out split-clone co-clustering **0.93 vs
0.09 chance** (1,517 clones whose two halves were re-derived independently).

![LN vaccine five-panel](benchmarks/biological_validation/results/ln_vaccine_gse195673/figures/comparison_five_panel.png)

*A: cells by author state. B: the classic view — top-20 sequence-defined
clonotypes on the cell UMAP (each is a scattered point cloud; sequence
networks say nothing about behaviour). C: Threadfin clone map — one point
per clonotype, size = expansion, colour = community. D: cells coloured by
their clone's community. E: gene signatures per community (the plasmablast
programme lights up exactly one community).*

### 2. Influenza vaccination, young vs older adults
*[Wang et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10564424/) — GSE175522/GSE175523 (6 donors × d0/d7, 123,693 cells)*

A seasonal-influenza-vaccine study designed to compare young and older
adults — the setting where age-related blunting of the humoral response
should be visible at clonal resolution. Because the authors tracked
`clone_id` across both timepoints, **711 clones are observed at both d0 and
d7**, enabling genuine clonal fate tracking rather than cross-sectional
comparison.

Threadfin finds that the day-7 plasmablast burst is carried by a
**restricted set of clonal lineages**: expanded d7 plasmablast clones
concentrate in a small number of communities (top enrichments OR 290 and,
at higher resolution, OR 1,341; both FDR ≈ 0). The communities are cleanly
stratified by class-switch status — one community is 75% IGHM / 21% IGHD
(unswitched), another 52% IGHG / 25% IGHA (switched, vaccine-responsive) —
and the d0→d7 community transition matrix is perfectly diagonal (272/272
clone transitions): a clone's pre-vaccination state predicts its day-7
state. Clone–state coupling is the strongest of all five datasets
(state-purity 0.883 vs 0.356 null, p = 0.005; held-out co-clustering 0.97
vs 0.26 chance), as expected for an acute recall response where what a
clone does is maximally determined by which clone it is. This dataset also
exercises `tf.clones.clone_fate_table` and `community_transition` directly,
and its young-vs-older design makes community-level stratification by age
group directly testable with `state_enrichment`.

### 3. Human tonsil B-cell maturation
*King et al. 2021 — [E-MTAB-9005](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9005)/E-MTAB-9003 (6 donors, 22,478 B-lineage cells)*

A tonsil B-cell maturation atlas with 15 author-annotated states spanning
the full differentiation axis (naive → preGC → dark/light-zone GC → cycling
→ memory, including FCRL4⁺ memory → pre-plasmablast → plasmablast). This
dataset is a methodological finding in itself: exact cellranger clonotypes
here are **almost all singletons** (13 clonotypes with ≥3 cells out of
10,473), because ongoing somatic hypermutation in GCs fragments every
lineage into many unique sequences. Any clone-level analysis built on exact
clonotypes is dead on arrival in exactly the tissue where B-cell biology is
most interesting.

Threadfin's sequence-similarity clone definition (`define_clones`, merging
SHM variants within the same V/J into lineages) recovers **200 expanded
lineages**, and a plasmablast community is recovered under both embedding
bases (OR 5.8–26.9, FDR ≤ 1e-13). We report the limits honestly: with only
200 lineages the clone graph resolves coarse structure (community
boundaries are unstable across bases, ARI ≈ 0; the enrichment-count null is
not rejected), yet the clone-level signal itself is real — state-purity
0.662 vs 0.489 null (p = 0.005) and held-out co-clustering 0.99 vs 0.66
chance. The headline is the rescue: sequence-aware clone definition *inside*
the package turns an unusable dataset into an analysable one.

### 4. Primary EBV infection in tonsil organoids
*[Mitul et al. 2026, PNAS](https://www.pnas.org/doi/10.1073/pnas.2603586123) — GSE317492 (d0–d21, GFP± sorted, 205,630 cells)*

A primary-EBV-infection time course in tonsil organoids, with GFP reporter
sorting at d14/d21 marking experimentally infected cells — an external,
non-transcriptomic label that the communities are scored against but never
see during construction. This dataset contributes the **largest expanded
clone set** (4,564 clones ≥3 cells of 89,757 clonotypes) and the **highest
concordance with de-novo cell states** (NMI 0.67, ARI 0.51; top
community–state pair OR 5,547, FDR ≈ 0). Held-out co-clustering is 0.985 vs
0.101 chance; state-purity 0.791 vs 0.314 null (p = 0.005).

EBV is also the **one dataset where the BCR modality is measurably
informative at clone level**: latent distances in the joint GEX+BCR
embedding correlate positively with inter-clone BCR similarity
(testcor_bcr = 0.31 over 1,389 inter-clone sequence edges), whereas on the
other datasets the clone-level BCR graph is too sparse to matter. The
biological reading: acute viral infection drives rapid expansion of
recently activated, low-SHM clonal families, so sequence relatedness and
cell state transiently align — exactly the regime where sequence-guided
integration (Benisse-style) earns its keep. One honest caveat: the
enrichment-count null is *not* rejected on this dataset (32 observed vs
41.2 expected under size-matched random communities), so the claims here
rest on concordance, purity and held-out replication rather than on
enrichment counts.

### 5. COVID-19 PBMC (the bundled quickstart dataset)
*[Stephenson et al. 2021, Nat Med](https://www.nature.com/articles/s41591-021-01329-2) — 5,000 BCR+ B-lineage cells*

This COVID-19 PBMC atlas established that severe disease is marked by an
expanded, interferon-activated plasmablast compartment (923 of the 5,000
B-lineage cells here). Threadfin recovers that response without ever
looking at receptor sequences: one clone community is **~198× enriched for
plasmablasts** (193 cells, FDR 6e-34) and expresses the expected
plasmablast/interferon programme. We report the caveats openly: the 5k
subset is sparse (median clone size 1; only 22 clonotypes with ≥3 cells, so
the config uses `min_clone_size=2`), and with only two communities the
held-out test is uninformative — this dataset is a reference and tutorial,
not evidence of large-scale structure.

### Cross-dataset summary

| dataset | cells | clones | expanded (≥3) | clone purity vs permutation null (p) | held-out vs chance |
|---|---|---|---|---|---|
| Stephenson 2021 COVID PBMC (5k) | 5,000 | 4,823 | 22 | 0.97 vs 0.82 (0.005) | n/a (2 communities) |
| Flu vaccine (Wang 2023) | 123,693 | 80,682 | 499 | 0.88 vs 0.36 (0.005) | 0.97 vs 0.26 |
| Tonsil (King 2021) | 22,478 | 10,473 | 200 | 0.66 vs 0.49 (0.005) | 0.99 vs 0.66 |
| EBV organoid (Mitul 2026) | 205,630 | 89,757 | 4,564 | 0.79 vs 0.31 (0.005) | 0.985 vs 0.101 |
| LN vaccine GC (Kim 2022) | 193,442 | 92,761 | 4,396 | 0.85 vs 0.48 (0.005) | 0.93 vs 0.09 |

Machine-readable:
[`results/cross_dataset_summary.md`](benchmarks/biological_validation/results/cross_dataset_summary.md).

## The biological interpretation — and its limits

Reading a Threadfin clone map: one point = one clonotype (a genetically
distinct lineage); size = expansion; position = the average transcriptional
state of that clone's cells; colour = a **community of clones converging on
the same state/fate**. The full guide is
[`docs/BIOLOGICAL_INTERPRETATION.md`](docs/BIOLOGICAL_INTERPRETATION.md).

We also report what does **not** hold, because it disciplines the claims:

* **Fine-grained community boundaries are basis-dependent** (clone-level ARI
  0.23–0.35 between PCA and UMAP bases on the large datasets). Conclusions
  should be stated at the level of *enrichments*, which do replicate across
  bases — not at the level of exact boundary assignments.
* **The clone-level BCR graph is sparse by biology, not by bug**: distinct
  clonotypes sharing ≥0.85 CDR3 identity within the same V/J are rare
  (0–1,389 edges per dataset), so the joint embedding is GEX-dominated
  (BCR modality ≤ 19%) and on tonsil contributes literally zero edges. The
  BCR modality's real value is upstream (sequence-aware clone definition)
  and in annotation (isotype/SHM per community), not in reweighting
  clone-level geometry. The v1 `cdr3_weight` blend illustrates the failure
  mode of naive distance mixing (on LN it collapses NMI 0.196 → 0.016); it
  is kept for backward compatibility, and the graph-based joint embedding is
  the recommended path.
* **Small clones are noisy.** Conclusions rest on expanded clones
  (`min_clone_size`); the Stephenson 5k subset is included as a reference,
  not as evidence of large-scale structure.

## Installation

```bash
pip install "git+https://github.com/Reynold4k/Threadfin.git"
pip install igraph leidenalg umap-learn   # graph clustering + embedding
pip install parasail                       # optional: C-accelerated CDR3 alignment
```

or from a clone: `pip install -e ".[leiden,seq,test]"`.

## Quick start

```python
import threadfin as tf

# 1) per-cell BCR table (10x filtered_contig_annotations.csv or AIRR TSV)
bcr = tf.read_10x_vdj("filtered_contig_annotations.csv")
bcr = tf.build_clone_key(bcr, strategy="cdr3")

# 2) attach to an existing scanpy object (needs adata.obsm['X_umap'])
adata = tf.attach_bcr(adata, bcr)               # adds obs['clone_id']

# 3) infer clone communities in transcriptional state space
adata = tf.clonotype_recluster(adata, min_clone_size=3, resolution=0.3)
adata = tf.clonal_pseudotime(adata)             # diffusion ordering of the clone graph

# 3b) v2: joint GEX+BCR latent embedding (coupled graph, Benisse-inspired)
tf.joint_embedding(adata, basis="X_pca", lam=0.5)
adata = tf.clonotype_recluster(adata, basis="joint", key_added="joint_cluster")
print(tf.integration_diagnostics(adata))

# 4) evaluate + visualize
print(tf.metrics.state_concordance(adata, "clone_cluster", "leiden"))
print(tf.metrics.state_enrichment(adata, "clone_cluster", "leiden"))
tf.plotting.clone_map(adata, color="clone_cluster", save="clone_map.png")
```

A runnable end-to-end example on public data is in
[`examples/quickstart.py`](examples/quickstart.py) (~1 minute).

## Scaling (synthetic data, up to 200k cells / 10k clones)

| cells | clones | attach_bcr | centroids | recluster | pseudotime | peak RSS |
|---|---|---|---|---|---|---|
| 2,000 | 200 | 0.06 s | 0.003 s | 18.5 s\* | 1.5 s | 0.46 GB |
| 10,000 | 833 | 0.38 s | 0.007 s | 1.3 s | 0.04 s | 0.47 GB |
| 50,000 | 3,333 | 1.2 s | 0.014 s | 5.9 s | 0.21 s | 0.50 GB |
| 200,000 | 10,000 | 4.4 s | 0.07 s | 30.8 s | 26.4 s | 0.59 GB |

\*first call pays numba JIT compilation. Runtime scales with the number of
*clones* (graph size), not cells. Reproduce with
`python benchmarks/run_scaling_benchmark.py benchmarks/results`.

## API overview

| Function | Purpose |
|---|---|
| `tf.read_10x_vdj` / `tf.read_airr` | read Cell Ranger / AIRR-format BCR tables (nt-junction translation when `junction_aa` is empty) |
| `tf.build_clone_key` | build clonotype keys (`vdj` / `clonotype_id` / `cdr3`) |
| `tf.define_clones` | sequence-similarity lineage grouping via the BCR graph |
| `tf.bcr_similarity_graph` | sparse clone×clone BCR similarity graph (V/J-blocked, BLOSUM62-aware) |
| `tf.attach_bcr` | attach BCR annotations to `AnnData` (vectorized) |
| `tf.clone_centroids` | per-clone centroids in any embedding (optionally weighted) |
| `tf.clonotype_recluster` | the core: infer clone communities in transcriptional state space (incl. `basis="joint"`) |
| `tf.joint_embedding` | coupled GEX+BCR graph embedding of clones (Benisse-inspired, ADMM-free) |
| `tf.integration_diagnostics` | latent-vs-GEX / latent-vs-BCR correlations, modality contribution |
| `tf.clonal_pseudotime` | diffusion-pseudotime ordering of the clone graph (clonal state-transition inference) |
| `tf.clones.clone_isotype_summary` / `clone_shm_summary` | isotype / SHM per clone or community |
| `tf.clones.clone_fate_table` / `community_transition` | clone fate tracking across timepoints |
| `tf.metrics.*` | NMI/ARI concordance, state enrichment (Fisher + FDR), clone purity/entropy |
| `tf.sequence.cdr3_similarity` / `atchley_embedding` | BLOSUM62-aware CDR3 similarity; deterministic physicochemical embedding |
| `tf.plotting.clone_map` / `cells` / `signature_heatmap` | publication-quality figures |

## Comparison with existing tools

Threadfin is **complementary, not competing**: run your favourite V(D)J
pipeline upstream, then use Threadfin to interpret clones in transcriptional
space. The closest published methods either define/analyse clonotypes
sequence-first, or integrate sequence with expression but do not infer
transcriptional-state communities of clones:

| Tool (original publication) | What it does with clonotypes | Infers clone communities by transcriptional state | Clonal trajectory inference | Sequence×expression integration |
|---|---|---|---|---|
| [scirpy](https://scirpy.scverse.org/) ([Sturm et al. 2020, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btaa611)) | CDR3-similarity clonotypes, repertoire stats, UMAP overlays | ✗ | ✗ | ✗ |
| [dandelion](https://github.com/zktuong/dandelion) ([Suo et al. 2024, *Nat Biotechnol*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10791579/)) | VDJ reannotation, clonal networks, V(D)J feature-space trajectory | ✗ | ✗ (cell-level only) | ✗ |
| [scRepertoire](https://github.com/ncborcherding/scRepertoire) ([Borcherding et al. 2021, *F1000Research*](https://f1000research.com/articles/10-230/v1)) | clonotype counting/overlap/gene usage on Seurat objects | ✗ | ✗ | ✗ |
| [Immcantation / SCOPer](https://immcantation.readthedocs.io) ([Gupta et al. 2015, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btv359); [Nouri & Kleinstein 2018, *Front Immunol*](https://www.frontiersin.org/articles/10.3389/fimmu.2018.00682/full)) | clonal-family definition & lineage trees from sequences | ✗ | ✗ (sequence lineage trees) | ✗ |
| [Benisse](https://github.com/wooyongc/Benisse) ([Zhang et al. 2022, *Nat Mach Intell*](https://www.nature.com/articles/s42256-022-00492-6)) | BCR embedding + sparse-graph integration with GEX | ✗ (graph components on clones) | ✗ | ✓ (ADMM graph learning) |
| [CoNGA](https://github.com/phbradley/conga) ([Schattgen et al. 2022, *Nat Methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8832949/)) | GEX–TCR neighbour-graph overlap scores per clonotype | ✗ (scores individual clonotypes) | ✗ | ✓ (graph-overlap test, TCR) |
| [TESSA](https://github.com/jcao89757/TESSA) ([Zhang et al. 2021, *Nat Methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7799492/)) | weighted TCR embedding networks constrained by GEX | ✗ | ✗ | ✓ (Bayesian, TCR) |
| [mvTCR](https://github.com/SchubertLab/mvTCR) ([Drost et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220149/)) | multi-view VAE joint GEX+TCR latent space | ✗ | ✗ | ✓ (deep generative, TCR) |
| **Threadfin** (this repo) | **clone centroids in GEX space → clone communities** | **✓** | **✓ (diffusion-based ordering of the clone graph)** | **✓ (coupled-graph joint embedding; BCR lineage grouping)** |

Notes on the comparison: Benisse is the conceptual parent of our joint
embedding — Threadfin replaces its ADMM on dense n×n matrices (and its
torch-based sequence encoder) with a sparse coupled-Laplacian eigenmap and a
deterministic BLOSUM62/Atchley sequence model, so it runs inside the scanpy
ecosystem without a GPU. CoNGA/TESSA/mvTCR target T cells and do not model
SHM-based B-cell lineages; Threadfin's `define_clones` is built for exactly
that (see the tonsil case study above).

## Citing Threadfin

If you use Threadfin, please cite this repository
(https://github.com/Reynold4k/Threadfin). A manuscript is in preparation.

## License

[MIT](LICENSE) © 2026 Chen Satoshi (Reynold4k)
