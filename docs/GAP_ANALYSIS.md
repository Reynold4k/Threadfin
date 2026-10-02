# Where Threadfin fits: the gap it closes

*Updated 2 October 2026 for v4. Every tool below was checked against its
publication (and, where relevant, its code). Benchmark numbers refer to
`benchmarks/simulation` and `benchmarks/public_datasets` in this repository.*

---

## The question

A paired single-cell RNA-seq + BCR-seq experiment measures, for every B cell,
**which clone it belongs to** (its receptor sequence) and **what it is doing**
(its transcriptome). Immunologists want to know, at the level of clones:

1. Does a clone's identity shape what its cells do, beyond where and when the
   cells were sampled?
2. Which genetically unrelated clones behave alike (for example, clones biased
   towards plasma-cell output versus clones retained in the germinal centre)?
3. Are such behaviours linked to the antigen specificity, mutation load,
   isotype or selection history of the clones?
4. Do clones keep their behaviour over time or across tissues?

Answering these needs (i) B-cell clones defined from hypermutated sequences,
(ii) clone-level estimates that are honest about how few cells most clones
have, and (iii) statistics that count clones, not cells, and respect the
sampling design. No existing tool provides this combination.

## The landscape (2018-2026)

### 1. Immune-repertoire toolkits: sequence first, expression as overlay

| Tool | What it does with clones and expression | What it does not do |
|---|---|---|
| **scirpy** (Sturm et al. 2020, *Bioinformatics*; scverse) | clonotype definition, repertoire statistics; `clonotype_modularity` scores whether a clonotype's cells are connected in the expression kNN graph | no grouping of clones by state; per-clonotype scores without a null that respects samples; no time/tissue memory |
| **Dandelion** (Suo et al. 2024, *Nat Biotechnol*) | improved V(D)J annotation; V(D)J-usage feature space on pseudobulks of cells; trajectories | pseudobulks are groups of cells, not clones; no clone-level state inference |
| **scRepertoire 2** (Yang et al. 2025, *PLoS Comput Biol*) | clonotype tracking, diversity, STARTRAC indices on Seurat objects | descriptive; indices without clone-level nulls |
| **Platypus** (Yermanos et al. 2021, *NAR Genom Bioinform*); **crocketa** (2026, *BMC Genomics*) | end-to-end workflows joining repertoire and expression | workflows, not statistical models of clone state |
| **Immcantation** (Change-O/SCOPer; **dowser**, Hoehn et al. 2022, *PLoS Comput Biol*) | clonal families and lineage trees; phylogenetic tests of migration, differentiation and isotype switching *within* clones | requires trees (expanded clones); models sequence ancestry, not between-clone state structure |
| **HILARy** (Spisak et al. 2024, *eLife*); **TRIBAL** (Weber et al. 2024, *Cell Genomics*) | high-precision clonal families; isotype-aware lineage trees | clone definition / phylogeny only |
| **AMULETY** (Wang et al. 2026, *Immunoinformatics*) | language-model embeddings of receptor sequences | sequence representation only |

### 2. Receptor-transcriptome integration: does a *similar receptor* mean a similar state?

| Tool | Unit and estimand | Relation to Threadfin |
|---|---|---|
| **CoNGA** (Schattgen et al. 2022, *Nat Biotechnol*) | per clonotype: overlap of receptor-similarity and expression neighbourhoods (T cells; experimental B-cell mode) | asks whether sequence-similar clonotypes share expression; Threadfin asks whether the *same* clone shares a state and which *unrelated* clones behave alike |
| **TESSA** (Zhang et al. 2021, *Nat Methods*), **Benisse** (Zhang et al. 2022, *Nat Mach Intell*) | receptor embeddings regularised by expression (TCR / BCR) | sequence-similarity graphs; no clone-level state statistics |
| **mvTCR** (Drost et al. 2024, *Nat Commun*), **CoMBCR** (Zou et al. 2026, *Bioinformatics*) | cell-level joint embeddings of receptor and expression (TCR / BCR) | cell-level representations; no clone-level inference, nulls or memory |
| **tcrpheno / TiRP / TCR-mem** (Lagattuta et al. 2022, *Nat Immunol*; 2025, *Cell Rep*) | receptor sequence features that predict T-cell fate | sequence-to-fate prediction, complementary |
| **STARTRAC** (Zhang et al. 2018, *Nature*); **TCRi** (Ceglia et al. 2022, bioRxiv) | clone-sharing indices between clusters/tissues; information-theoretic clonotype-phenotype metrics | descriptive indices over predefined cell clusters |

### 3. Clone-level analysis in lineage tracing: closest in spirit, built for synthetic barcodes

| Tool | What it does | Why it does not transfer to B-cell repertoires |
|---|---|---|
| **clone2vec** (Isaev, Erickson, Adameyko & Kharchenko 2026, bioRxiv) | learns clone embeddings with a skip-gram model over the cell kNN graph; applied to lineage barcodes and to TCR clones in tumours | exact barcodes (no hypermutation-aware clone definition); no model of sampling context; no per-clone reliability; no clone-level tests or nulls; no memory index. In our simulations clones nested in shifted samples reduce it to ARI 0.10 (default scenario) |
| **ClonoCluster** (Richman et al. 2023, *Cell Genomics*) | hybrid *cell* clusters weighting clone identity | clusters cells, not clones |
| **CoSpar** (Wang et al. 2022, *Nat Biotechnol*), **moslin** (Lange et al. 2024, *Genome Biol*), **CellRank 2** (Weiler et al. 2024, *Nat Methods*), **DestinyNet** (2026, *Patterns*) | fate maps and transition probabilities from barcoded time courses | designed for prospective barcoding with planned sampling; not for donor-private, mostly singleton, hypermutated repertoires |

### 4. The biology says the question is real

Clonal state inheritance is now well documented: B-cell clones have restricted
fate sets and transcriptional memory (Swift, Horns & Quake 2023, *Life Sci
Alliance*); non-genetic B-cell states stay stable within germinal-centre clonal
bursts (Xiang et al. 2026, *Cell Syst*); germinal centres output plasma cells
across affinities (Sprumont et al. 2023, *Cell*); high-affinity clones divide
more but mutate less per division (Merkenschlager et al. 2025, *Nature*); and
receptor sequence biases T-cell fate (Mantena & Raychaudhuri 2026, *Immunol
Rev*). What is missing is a calibrated, clone-level method to measure these
effects in ordinary paired single-cell data.

## The gap, stated precisely

| Requirement | Repertoire toolkits | Receptor-expression integrators | Lineage-tracing clone tools | **Threadfin v4** |
|---|---|---|---|---|
| B-cell clones from hypermutated sequences, within donors | yes (Immcantation, Dandelion, scirpy) | partly | no (exact barcodes) | **yes** (`define_clones`) |
| Clone as the unit of inference | no | no (cells or clonotype pairs) | yes | **yes** |
| Sampling context removed from clone profiles | no | no | no | **yes** (context-centred random-effects model) |
| Per-clone reliability / uncertainty | no | no | no | **yes** (BLUP shrinkage, Spearman-Brown reliability, posterior assignment) |
| Distributional clone profiles (mixed vs intermediate clones) | no | no | partly (clone2vec, implicitly) | **yes** (kernel mean embeddings) |
| Calibrated test that clone identity explains state | no (cell-level or global nulls) | no | no | **yes** (within-sample permutation of the clonal ICC) |
| Clone-level association tests (specificity, isotype, SHM, gates) | no (cell-level counts) | no | partly | **yes** (stratified permutation, Mantel-Haenszel) |
| Non-circular clonal memory across time/tissue | no | no | trajectory-based | **yes** (noise-corrected memory index) |
| Which genes are clonally inherited | no | no | partly | **yes** (`gene_heritability`) |
| Ground-truth simulator with B-cell-like sampling | no | no | partly | **yes** (`threadfin.sim`) |

## Why the statistics matter: what goes wrong without them

These failures are not hypothetical; the v3 release of this package itself
made them, and the v4 benchmarks quantify them:

* **Global or donor-level nulls.** With no clonal signal at all but clones
  confined to samples that differ technically, shuffling clone labels across
  samples (or only within donor) reports "significant clonality" in every
  simulation; shuffling within samples is calibrated.
* **Cell-level tests.** Testing whether a programme is enriched for a label by
  counting cells calls a coin flipped per clone "significant" in essentially
  every programme, because one expanded clone contributes hundreds of
  correlated cells.
* **Immunoglobulin genes in the embedding.** Receptor transcripts are
  identical within a clone; including them inflated the clonal ICC by about a
  quarter on real data and makes isotype a circular validation label.
* **Clone-level labels compared over time.** A programme label computed from
  all of a clone's cells is the same at every time point, so its "transition
  matrix" is diagonal whatever happens biologically; a snapshot-based,
  noise-corrected memory index recovers the true switching rate.

## Threadfin's contribution in one paragraph

Threadfin treats each B-cell clone as the unit of inference. It defines clones
within donors from hypermutated sequences, describes each clone by a shrunken,
context-adjusted profile of its cells (a centroid or a kernel embedding of its
cell distribution) with an explicit reliability, measures how much of
transcriptional variation is clonally inherited with a sampling-aware null,
groups clones into stability-assessed programmes, tests those programmes
against clone-level labels counting clones rather than cells, quantifies
clonal memory across time points and tissues without circularity, and ranks
genes by clonal heritability — in pure Python, AnnData-native, with a single
`threadfin.run()` entry point and a ground-truth simulator for benchmarking.
