# Where Threadfin fits

*Updated October 2026. Each tool below was checked against its publication.*

## The biological questions

A paired single-cell RNA-seq + BCR-seq experiment tells you, for every B cell,
**which clone it belongs to** (its receptor sequence) and **what it is doing**
(its transcriptome). Immunologists want to answer questions about clones:

1. **Is B-cell fate inherited within clones?** Do the cells of one clone
   resemble each other more than cells of unrelated clones sampled at the
   same place and time?
2. **Which clones behave alike?** Are there groups of unrelated clones with a
   shared fate bias (for example plasma-cell output versus germinal-centre
   retention), or do clones vary along a continuum?
3. **What makes clones differ?** Antigen binding, affinity mutations, isotype,
   division history, tissue, time after vaccination or infection.
4. **Do clones keep their state** over weeks and months, or across tissues?
5. **Which genes are clonally inherited**, and which are shared by all clones
   (for example the germinal-centre dark-zone / light-zone cycle)?

Answering these needs B-cell clones defined from hypermutated sequences,
clone-level estimates that account for how few cells most clones have, and
comparisons that count clones rather than cells and respect how samples were
collected.

## Existing tools and the questions they answer

### Immune-repertoire toolkits: sequence first, expression as an overlay

| Tool | Answers | Does not answer |
|---|---|---|
| **scirpy** (Sturm et al. 2020, *Bioinformatics*) | clonotypes, repertoire diversity; whether a clonotype's cells are neighbours in expression space | which clones behave alike; what explains differences between clones; persistence over time |
| **Dandelion** (Suo et al. 2024, *Nat Biotechnol*) | V(D)J annotation; V(D)J usage along cell trajectories | clone-level state questions |
| **scRepertoire 2** (Yang et al. 2025, *PLoS Comput Biol*) | clonotype tracking, diversity, sharing between clusters | clone-level state questions |
| **Platypus** (Yermanos et al. 2021, *NAR Genom Bioinform*) | end-to-end repertoire + expression workflows | statistical questions about clone states |
| **Immcantation** (Change-O, SCOPer; dowser, Hoehn et al. 2022, *PLoS Comput Biol*) | clonal families; lineage trees; migration, differentiation and switching *within* a clone's tree | what distinguishes *different* clones; clones too small for trees |
| **HILARy** (Spisak et al. 2024, *eLife*); **TRIBAL** (Weber et al. 2024, *Cell Genomics*) | accurate clonal families; isotype-aware lineage trees | expression |

### Receptor-transcriptome integration: does a similar receptor mean a similar state?

| Tool | Answers | Relation to Threadfin |
|---|---|---|
| **CoNGA** (Schattgen et al. 2022, *Nat Biotechnol*) | whether receptor-similar clonotypes share expression neighbourhoods (mainly T cells) | asks about *similar sequences*; Threadfin asks about *the same clone* and about unrelated clones with shared fates |
| **TESSA** (Zhang et al. 2021, *Nat Methods*); **Benisse** (Zhang et al. 2022, *Nat Mach Intell*) | receptor embeddings shaped by expression | sequence-similarity structure, not clone fates |
| **mvTCR** (Drost et al. 2024, *Nat Commun*) | joint cell embeddings of receptor and expression | cell-level, not clone-level |
| **STARTRAC** (Zhang et al. 2018, *Nature*) | clone sharing between predefined clusters and tissues | descriptive indices over predefined cell clusters |

### Clone-level analysis from lineage tracing: closest in spirit

| Tool | Answers | Why it does not transfer to B-cell repertoires |
|---|---|---|
| **clone2vec** (Isaev et al. 2026, bioRxiv) | embeddings of clones from the cell neighbourhood graph | built for exact synthetic barcodes; no hypermutation-aware clones, no adjustment for where clones were sampled, no clone-level tests |
| **ClonoCluster** (Richman et al. 2023, *Cell Genomics*) | cell clusters informed by clone identity | clusters cells, not clones |
| **CoSpar** (Wang et al. 2022, *Nat Biotechnol*); **moslin** (Lange et al. 2024, *Genome Biol*); **CellRank 2** (Weiler et al. 2024, *Nat Methods*) | fate maps from barcoded time courses | designed for prospective barcoding, not donor-private, mostly small, hypermutated repertoires |

### The biology says the questions are real

B-cell clones have restricted fate sets and transcriptional memory (Swift,
Horns & Quake 2023, *Life Sci Alliance*); germinal centres produce plasma
cells across a range of affinities (Sprumont et al. 2023, *Cell*);
high-affinity germinal-centre clones divide more but mutate less per division
(Merkenschlager et al. 2025, *Nature*). These results come from dedicated
experimental systems. What has been missing is a way to ask the same questions
in ordinary paired single-cell data.

## The gap Threadfin fills

| Question | Repertoire toolkits | Receptor-expression integration | Lineage-tracing clone tools | **Threadfin** |
|---|---|---|---|---|
| Clones from hypermutated sequences, within donors | yes | partly | no | **yes** |
| Is fate inherited within clones, beyond where cells were sampled? | no | no | no | **yes** (clonal coherence) |
| Which clones behave alike, or is it a continuum? | no | no | partly | **yes** (programmes with significant splits only) |
| What explains differences between clones (antigen, isotype, gates, tissue)? | no (cell counts) | no | no | **yes** (clone-level label tests) |
| Do clones keep their state over time or across tissues? | no | no | trajectory-based | **yes** (clonal memory) |
| Which genes are clonally inherited? | no | no | partly | **yes** (gene-level clonal heritability) |
| How reliable is each clone's profile, given its size? | no | no | no | **yes** (per-clone reliability) |

## In one paragraph

Threadfin treats each B-cell clone as the unit of analysis. It defines clones
within donors from hypermutated sequences, describes each clone relative to
the cells it was sampled with (so that sample and batch differences are not
mistaken for clonal ones), measures how much of B-cell state is clonally
inherited, groups clones into programmes only when the groups are genuinely
distinct, tests what distinguishes clones with clones rather than cells as
replicates, measures whether clones keep their state over time and across
tissues, and ranks genes by clonal inheritance, all from one `threadfin.run()`
call on standard AnnData objects.
