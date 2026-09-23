# Threadfin

**Clonotype-level inference of B-cell transcriptional states and fates from paired scRNA-seq + scBCR-seq data.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org)
[![tests](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml/badge.svg)](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml)

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
timepoints or along a diffusion-based ordering of the clone graph. The BCR
sequence layer (BLOSUM62-aware CDR3 similarity, sequence-aware lineage
grouping, coupled GEX+BCR graph embedding inspired by Benisse) is used where
it provably helps — and the v3 discovery layer connects communities to
antigen specificity, affinity maturation, tissue migration and gene
programmes, with null-model validation throughout.

## The gap Threadfin closes

A systematic review of the published tools and their public issue trackers
(full evidence with quotes and links: [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md)):

* **No tool groups clones by transcriptional state.** Benisse's graph is
  hard-constrained by V/J sequence; CoNGA tests sequence↔expression
  correlation but does not group clones; dandelion's V(D)J feature space is
  pseudobulk V/J usage; mvTCR embeds cells, not clones; scirpy/scRepertoire/
  Immcantation are sequence-first toolboxes.
* **No tool validates clone–state coupling statistically.** We ship
  within-donor permutation nulls, size-matched community nulls, held-out
  split-clone replication and cross-basis robustness — run on 5 public
  datasets, with failures reported as-is.
* **Clone fate/migration tracking is chronically unmet** — scirpy's
  STARTRAC-indices issue has been open since 2020
  ([#36](https://github.com/scverse/scirpy/issues/36)); users repeatedly ask
  how to follow clones across clusters, timepoints and tissues
  (scRepertoire [#428](https://github.com/BorchLab/scRepertoire/issues/428),
  [#585](https://github.com/BorchLab/scRepertoire/issues/585)).
* **Real repertoires break sequence-first tools**: the median clonotype has
  exactly 1 cell in every dataset we analysed; cell-level clone networks die
  past ~20k cells ([dandelion #235](https://github.com/tuonglab/dandelion/issues/235))
  and lineage-tree builders can be left with zero clones
  ([dowser #38](https://github.com/immcantation/dowser/issues/38)).
  Threadfin's clone-as-node design scales with clone count, not cell count.
* **Pure Python, scanpy-native, CPU-only** — no GPU (mvTCR), no R+torch
  two-language stack (Benisse/TESSA), no C++ deps (CoNGA).

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

## Walkthrough: five public datasets, end to end

Everything below is reproduced by two committed pipelines —
[`benchmarks/biological_validation/`](benchmarks/biological_validation/)
(clone-community inference + null models + held-out replication) and
[`benchmarks/discovery/`](benchmarks/discovery/) (specificity, SHM,
migration, gene programmes). Dataset sources and checksums:
[`benchmarks/datasets_manifest.tsv`](benchmarks/datasets_manifest.tsv).

**How to read a Threadfin clone map:** one point = one clonotype (a
genetically distinct lineage); point size = expansion (cells per clone);
position = the clone's average transcriptional state; colour = inferred
**clone community**. Communities are scored against labels the method never
saw (author cell states, antigen specificity, infection status, severity),
with permutation nulls for every claim.

### 1. Lymph-node germinal-centre response to mRNA vaccination
*[Kim et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04527-1) — GSE195673 / [Zenodo 5895181](https://zenodo.org/records/5895181) (8 donors, 193,442 B cells, LN FNA + blood + bone marrow, d28–d201)*

The study tracked the SARS-CoV-2 mRNA-vaccine GC reaction in draining lymph
nodes for months and showed vaccine-induced GCs keep producing
affinity-matured clones long after the blood response wanes. 92,761
clonotypes (median size 1; 4,396 expanded to ≥3 cells) over author-annotated
GC / memory / naive / plasmablast / LN-plasma-cell states.

**Threadfin resolves 11 clone communities** over the 4,396 expanded clones,
coherent in three independent ways:

* **cell states** — 17–37 significant enrichments vs the authors'
  annotations per parameter setting (size-matched null p = 0.015); the
  largest community maps onto GC cells (OR 2.4, FDR ≈ 0), a plasmablast
  community emerges at higher resolution (OR 156);
* **class-switch history** — never used to build the graph, yet the three
  largest communities are 80% / 88% / 83% IGHG while others are
  IGHM/IGHD-rich;
* **temporal stability** — the community transition matrix of clones
  followed across d28 → d60 → d110 is **perfectly diagonal (1,016/1,016)**:
  a clone's early-response community predicts its community months later.

![LN vaccine five-panel](benchmarks/biological_validation/results/ln_vaccine_gse195673/figures/comparison_five_panel.png)

*Walk through the panels left to right: A — cells by author state. B — the
classic view: top-20 sequence-defined clonotypes on the cell UMAP (each a
scattered point cloud — sequence networks say nothing about behaviour).
C — Threadfin clone map (one point per clone; size = expansion; colour =
community). D — the same communities projected back onto cells. E — gene
signatures per community: the plasmablast programme lights up exactly one
community.*

**Held-out biology the method never saw.** The authors' BCR tables carry
per-clone spike-specificity calls (1,368 S+ clones) and 1,350 ELISA-validated
monoclonal antibodies, plus per-sequence SHM frequencies:

* **Specificity maps onto communities**: one community is 86.6%
  spike-specific (OR 4.8, FDR ≈ 0; 9,013 cells); three more are S+-enriched
  (OR 1.7–1.8, FDR ≤ 1e-51).
* **Affinity maturation, reconstructed**: median SHM of S+ clones rises
  monotonically 0.75% (d28) → 1.5% (d35) → 2.2% (d60) → 3.7% (d110) →
  5.2% (d201), above S− clones from d35 on (p ≤ 1e-81) — the paper's
  central finding in one curve.
* **GC clones emigrate to blood**: S+ expanded clones span LN+blood 2.8×
  more often than S− clones (25.1% vs 9.1%; per-clone migration index
  0.032 vs 0.017).

![LN discovery](benchmarks/discovery/results/ln_vaccine_gse195673/figures/discovery_ln.png)

*Left: per-community enrichment of spike-specific clones (red = FDR < 0.05).
Middle: affinity maturation of S+ vs S− clones across the five-month
response. Right: gene programmes per community (GC, plasmablast and naive
programmes occupy distinct communities).*

Statistics: clone state-purity 0.849 vs 0.478 within-donor permutation null
(p = 0.005); held-out split-clone co-clustering **0.93 vs 0.09 chance**
(1,517 clones, sibling halves re-derived independently).
Reproduce: [`configs/ln_vaccine_gse195673.json`](benchmarks/biological_validation/configs/ln_vaccine_gse195673.json) →
[`results/ln_vaccine_gse195673/`](benchmarks/biological_validation/results/ln_vaccine_gse195673/) +
[`discovery results`](benchmarks/discovery/results/ln_vaccine_gse195673/).

### 2. Influenza vaccination, young vs older adults
*[Wang et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10564424/) — GSE175522/GSE175523 (6 donors × d0/d7, 123,693 cells)*

A seasonal-flu vaccine study comparing young and older adults. The authors
tracked `clone_id` across both timepoints, so **711 clones are observed at
both d0 and d7** — genuine clonal fate tracking, not cross-sectional
comparison.

**What Threadfin finds:** the day-7 plasmablast burst is carried by a
restricted set of clonal lineages — expanded d7 plasmablast clones
concentrate in a few communities (top enrichments OR 290, and OR 1,341 at
higher resolution; FDR ≈ 0). Communities stratify by class-switch status
(one 75% IGHM / 21% IGHD unswitched; another 52% IGHG / 25% IGHA switched,
vaccine-responsive), and the d0→d7 community transition matrix is perfectly
diagonal (272/272). Clone–state coupling is the strongest of all five
datasets (purity 0.883 vs 0.356 null, p = 0.005; held-out 0.97 vs 0.26
chance).

| clone map (499 clones) | communities on the cell UMAP |
|---|---|
| ![flu clone map](benchmarks/biological_validation/results/flu_gse175522/figures/clone_map.png) | ![flu cells](benchmarks/biological_validation/results/flu_gse175522/figures/cells_clone_cluster.png) |

*Left: every point is a clone; the orange community on the right is the
vaccine-responsive switched compartment. Right: the same communities
projected back onto 123,693 cells (grey = no BCR).*

![flu discovery](benchmarks/discovery/results/flu_gse175522/figures/discovery_flu.png)

*Left: clonal expansion indices rise from d0 (blue) to d7 (red) in both age
groups. Middle: clone-level migration between timepoints (59.2 cross-terms =
clones shared d0↔d7). Right: the plasmablast programme lights up community 1
— matching the community where d7 plasmablast clones concentrate.*

The repertoire is almost entirely private (1 public clone across 6 donors).
Reproduce: [`configs/flu_gse175522.json`](benchmarks/biological_validation/configs/flu_gse175522.json) →
[`results/flu_gse175522/`](benchmarks/biological_validation/results/flu_gse175522/) +
[`discovery results`](benchmarks/discovery/results/flu_gse175522/).

### 3. Human tonsil B-cell maturation
*King et al. 2021 — [E-MTAB-9005](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9005)/E-MTAB-9003 (6 donors, 22,478 B-lineage cells)*

A tonsil atlas with 15 author states spanning the full maturation axis
(naive → preGC → dark/light-zone GC → cycling → memory incl. FCRL4⁺ →
pre-plasmablast → plasmablast). **A methodological finding in itself**:
exact cellranger clonotypes here are almost all singletons (13 of 10,473
have ≥3 cells) — ongoing SHM fragments every lineage, so any exact-clonotype
analysis is dead on arrival in exactly the tissue where B-cell biology is
most interesting.

Threadfin's sequence-similarity clone definition (`define_clones`, merging
SHM variants within the same V/J into lineages) recovers **200 expanded
lineages**, and a plasmablast community is recovered under both embedding
bases (OR 5.8–26.9, FDR ≤ 1e-13). Limits reported honestly: with 200
lineages the graph resolves coarse structure (boundaries unstable across
bases; enrichment-count null not rejected), but clone-level signal is real
(purity 0.662 vs 0.489 null; held-out 0.99 vs 0.66).

| lineage clone map (200 lineages, resolution 0.8) | plasmablast programme per community |
|---|---|
| ![tonsil clone map](benchmarks/discovery/results/tonsil_king2021/figures/clone_map_discovery.png) | ![tonsil discovery](benchmarks/discovery/results/tonsil_king2021/figures/discovery_tonsil.png) |

*Left: the rescued lineage landscape — each point is an SHM-merged lineage,
not an exact clonotype. Right: the plasmablast gene programme lights up
exactly one lineage community (score 0.85 vs ≤ 0.10 elsewhere).*

Reproduce: [`configs/tonsil_king2021.json`](benchmarks/biological_validation/configs/tonsil_king2021.json) →
[`results/tonsil_king2021/`](benchmarks/biological_validation/results/tonsil_king2021/) +
[`discovery results`](benchmarks/discovery/results/tonsil_king2021/).

### 4. Primary EBV infection in tonsil organoids
*[Mitul et al. 2026, PNAS](https://www.pnas.org/doi/10.1073/pnas.2603586123) — GSE317492 (d0–d21, GFP± sorted, 205,630 cells)*

A primary-EBV-infection time course with GFP reporter sorting at d14/d21
marking experimentally infected cells — an external, non-transcriptomic
label communities are scored against but never see. Largest expanded-clone
set (4,564 of 89,757 clonotypes) and highest concordance with de-novo states
(NMI 0.67; top community–state pair OR 5,547, FDR ≈ 0; held-out 0.985 vs
0.101; purity 0.791 vs 0.314 null).

**The dominant clone community is 39.7× enriched for experimentally
infected (GFP+) cells** (10,812 cells, FDR ≈ 0) — clone communities recover
infection status measured by viral reporter sorting. EBV is also the one
dataset where the clone-level BCR graph is measurably informative
(testcor_bcr = 0.31 over 1,389 inter-clone edges): acute infection expands
recently activated, low-SHM families, so sequence and state transiently
align — the regime where Benisse-style sequence-guided integration earns its
keep. Honest caveat: the enrichment-count null is not rejected here (32
observed vs 41.2 under size-matched random communities); claims rest on
concordance, purity and held-out replication.

| clone map (4,564 clones) | communities recover experimental infection |
|---|---|
| ![ebv clone map](benchmarks/biological_validation/results/ebv_organoid_gse317492/figures/clone_map.png) | ![ebv discovery](benchmarks/discovery/results/ebv_organoid_gse317492/figures/discovery_ebv.png) |

*Left: the clonal landscape of the infection time course (13 communities).
Right: per-community enrichment for GFP+ experimentally infected cells
(red = FDR < 0.05) and gene programmes per community (community 12 = GC
programme).*

Reproduce: [`configs/ebv_organoid_gse317492.json`](benchmarks/biological_validation/configs/ebv_organoid_gse317492.json) →
[`results/ebv_organoid_gse317492/`](benchmarks/biological_validation/results/ebv_organoid_gse317492/) +
[`discovery results`](benchmarks/discovery/results/ebv_organoid_gse317492/).

### 5. COVID-19 PBMC (the bundled quickstart dataset)
*[Stephenson et al. 2021, Nat Med](https://www.nature.com/articles/s41591-021-01329-2) — 5,000 BCR+ B-lineage cells*

The atlas that established the expanded, interferon-activated plasmablast
compartment of severe COVID-19 (923 of 5,000 B-lineage cells here).
Threadfin recovers that response without ever looking at receptor sequences:
one clone community is **~198× enriched for plasmablasts** (193 cells, FDR
6e-34) with the expected plasmablast/interferon programme — and the
community **stratifies clinical severity** (enriched for severe disease,
OR 3.5, FDR 0.003; the B-cell community is enriched for asymptomatic/mild
donors, OR 32, FDR 7e-7). Reference matching against
[CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/) flags
SARS-CoV-2-binding clones and places them in the plasmablast community
(small numbers on the 5k subset; reported as-is). Caveats we state openly:
the subset is sparse (median clone size 1; 22 clones ≥3 cells, so
`min_clone_size=2`), and with two communities the held-out test is
uninformative — a reference and tutorial, not evidence of large-scale
structure.

| clone map (75 clones) | permutation null vs observed purity | community × severity |
|---|---|---|
| ![stephenson clone map](benchmarks/biological_validation/results/stephenson2021/figures/clone_map.png) | ![stephenson null](benchmarks/biological_validation/results/stephenson2021/figures/null_purity.png) | ![stephenson discovery](benchmarks/discovery/results/stephenson2021/figures/discovery_stephenson.png) |

*Left: clone map with the plasmablast community (blue) separated from B-cell
clones. Middle: observed clone state-purity (red line) vs the within-donor
permutation null (grey). Right: community × severity odds ratios
(* = FDR < 0.05) and gene programmes per community.*

Reproduce: [`configs/stephenson2021.json`](benchmarks/biological_validation/configs/stephenson2021.json) →
[`results/stephenson2021/`](benchmarks/biological_validation/results/stephenson2021/) +
[`discovery results`](benchmarks/discovery/results/stephenson2021/).

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

The full interpretation guide is
[`docs/BIOLOGICAL_INTERPRETATION.md`](docs/BIOLOGICAL_INTERPRETATION.md).
We report what does **not** hold, because it disciplines the claims:

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
* **Reference-database specificity hits are evidence, not proof**:
  heavy-chain-only matching can collide across antigens for public CDR3s;
  treat `annotate_specificity` reference-mode hits as hypotheses to
  validate, and prefer author-validated labels where they exist.

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

# 3b) joint GEX+BCR latent embedding (coupled graph, Benisse-inspired)
tf.joint_embedding(adata, basis="X_pca", lam=0.5)
adata = tf.clonotype_recluster(adata, basis="joint", key_added="joint_cluster")
print(tf.integration_diagnostics(adata))

# 4) discovery layer: specificity, migration, programmes
adata = tf.annotate_specificity(adata, labels=clone_labels,
                                label_col="spike_specific")
print(tf.specificity_enrichment(adata))          # Fisher per community
print(tf.migration_index(adata, group_key="tissue"))   # STARTRAC-style
print(tf.community_markers(adata))               # marker genes per community

# 5) evaluate + visualize
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
| `tf.annotate_specificity` / `tf.specificity_enrichment` | clone-level antigen specificity (labels or CoV-AbDab-style reference matching) + per-community enrichment |
| `tf.migration_index` / `transition_index` / `expansion_index` / `clone_distribution` | STARTRAC-style clonal migration/transition/expansion indices |
| `tf.community_markers` / `tf.community_score` | marker genes and gene-programme scores per community |
| `tf.clones.clone_isotype_summary` / `clone_shm_summary` / `shm_gradient_test` | isotype / SHM per clone or community, maturation-gradient tests |
| `tf.clones.clone_fate_table` / `community_transition` / `public_clone_summary` | clone fate tracking; cross-donor public clones |
| `tf.metrics.*` | NMI/ARI concordance, state enrichment (Fisher + FDR), clone purity/entropy |
| `tf.sequence.cdr3_similarity` / `atchley_embedding` | BLOSUM62-aware CDR3 similarity; deterministic physicochemical embedding |
| `tf.plotting.clone_map` / `cells` / `signature_heatmap` | publication-quality figures |

## Comparison with existing tools

Threadfin is **complementary, not competing**: run your favourite V(D)J
pipeline upstream, then use Threadfin to interpret clones in transcriptional
space. The closest published methods either define/analyse clonotypes
sequence-first, or integrate sequence with expression but do not infer
transcriptional-state communities of clones:

| Tool (original publication) | What it does with clonotypes | Infers clone communities by transcriptional state | Clonal trajectory inference | Clone migration indices | Sequence×expression integration |
|---|---|---|---|---|---|
| [scirpy](https://scirpy.scverse.org/) ([Sturm et al. 2020, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btaa611)) | CDR3-similarity clonotypes, repertoire stats, UMAP overlays | ✗ | ✗ | ✗ ([open issue since 2020](https://github.com/scverse/scirpy/issues/36)) | ✗ |
| [dandelion](https://github.com/zktuong/dandelion) ([Suo et al. 2024, *Nat Biotechnol*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10791579/)) | VDJ reannotation, clonal networks, V(D)J feature-space trajectory | ✗ | ✗ (cell-level only) | ✗ | ✗ |
| [scRepertoire](https://github.com/ncborcherding/scRepertoire) ([Borcherding et al. 2021, *F1000Research*](https://f1000research.com/articles/10-230/v1)) | clonotype counting/overlap/gene usage on Seurat objects | ✗ | ✗ | ✓ (STARTRAC wrapper) | ✗ |
| [Immcantation / SCOPer](https://immcantation.readthedocs.io) ([Gupta et al. 2015, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btv359); [Nouri & Kleinstein 2018, *Front Immunol*](https://www.frontiersin.org/articles/10.3389/fimmu.2018.00682/full)) | clonal-family definition & lineage trees from sequences | ✗ | ✗ (sequence lineage trees) | ✗ | ✗ |
| [Benisse](https://github.com/wooyongc/Benisse) ([Zhang et al. 2022, *Nat Mach Intell*](https://www.nature.com/articles/s42256-022-00492-6)) | BCR embedding + sparse-graph integration with GEX | ✗ (graph components on clones) | ✗ | ✗ | ✓ (ADMM graph learning) |
| [CoNGA](https://github.com/phbradley/conga) ([Schattgen et al. 2022, *Nat Methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8832949/)) | GEX–TCR neighbour-graph overlap scores per clonotype | ✗ (scores individual clonotypes) | ✗ | ✗ | ✓ (graph-overlap test, TCR) |
| [TESSA](https://github.com/jcao89757/TESSA) ([Zhang et al. 2021, *Nat Methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7799492/)) | weighted TCR embedding networks constrained by GEX | ✗ | ✗ | ✗ | ✓ (Bayesian, TCR) |
| [mvTCR](https://github.com/SchubertLab/mvTCR) ([Drost et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220149/)) | multi-view VAE joint GEX+TCR latent space | ✗ | ✗ | ✗ | ✓ (deep generative, TCR) |
| **Threadfin** (this repo) | **clone centroids in GEX space → clone communities** | **✓** | **✓ (diffusion-based ordering of the clone graph)** | **✓ (STARTRAC-style, clone-level)** | **✓ (coupled-graph joint embedding; BCR lineage grouping; specificity annotation)** |

Notes on the comparison: Benisse is the conceptual parent of our joint
embedding — Threadfin replaces its ADMM on dense n×n matrices (and its
torch-based sequence encoder) with a sparse coupled-Laplacian eigenmap and a
deterministic BLOSUM62/Atchley sequence model, so it runs inside the scanpy
ecosystem without a GPU. CoNGA/TESSA/mvTCR target T cells and do not model
SHM-based B-cell lineages; Threadfin's `define_clones` is built for exactly
that (see the tonsil case study above).

## Citing Threadfin

If you use Threadfin, please cite this repository
(https://github.com/Reynold4k/Threadfin; see [`CITATION.cff`](CITATION.cff)).
A manuscript is in preparation.

## License

[MIT](LICENSE) © 2026 Chen Satoshi (Reynold4k)
