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
| **BiGCN** (Liu et al. 2026, *Small Methods*) | a joint embedding of transcriptome and BCR, learned with a graph network, for annotating B-cell states | the unit is the **cell**: it produces a better cell representation. Threadfin asks questions about clones and reports how certain each answer is |
| **CoNGA** (Schattgen et al. 2022, *Nat Biotechnol*) | whether receptor-similar clonotypes share expression neighbourhoods | asks about *similar sequences*; Threadfin asks about *the same clone*, and about unrelated clones with shared behaviour |
| **sciCSR** (Ng et al. 2023, *Nat Methods*) | B-cell state transitions and class-switch dynamics from expression | transitions between cell states; no clone-level estimates |
| **Clonotrace** (2025, preprint) | state transitions and fate bias from expression plus clonotype (mainly T cells) | overlaps "which clones behave alike"; trajectory-shaped, and does not model where a clone was sampled |
| **TESSA** (Zhang et al. 2021, *Nat Methods*); **Benisse** (Zhang et al. 2022, *Nat Mach Intell*) | receptor embeddings shaped by expression | sequence-similarity structure, not clone behaviour |
| **SeQuoIA** (2025, preprint) | selection pressure in germinal centres from somatic mutation patterns alone | **complementary**: it reads selection from the sequences, Threadfin reads clone behaviour from expression. The two can be checked against each other |
| **STARTRAC** (Zhang et al. 2018, *Nature*) | clone sharing between predefined clusters and tissues | descriptive indices over predefined cell clusters |

### Clone-level variation in other systems

| Tool | Answers | Why it does not transfer directly |
|---|---|---|
| **Mold et al. 2024, *Cell Systems*** | established that gene expression is **clonally heritable**, by comparing variation within and between clones of cultured human lymphocytes and mouse brain cells | the measurement Threadfin builds on. It was made where clone membership is known and all cells come from one culture. In a repertoire, clones are inferred from hypermutating receptors, are mostly one to three cells, and are confounded with the sample, tissue or sort gate they came from - which is the problem Threadfin solves |
| **clone2vec** (Isaev et al. 2026, preprint) | embeddings of clones learned from the cell neighbourhood graph | built for exact synthetic barcodes; no hypermutation-aware clones, no adjustment for where a clone was sampled, no clone-level tests |
| **ClonoCluster** (Richman et al. 2023, *Cell Genomics*) | cell clusters informed by clone identity | clusters cells, not clones |
| **CoSpar** (Wang et al. 2022, *Nat Biotechnol*); **moslin** (Lange et al. 2024, *Genome Biol*); **CellRank 2** (Weiler et al. 2024, *Nat Methods*) | fate maps from barcoded time courses | designed for prospective barcoding with planned sampling, not donor-private, mostly small, hypermutated repertoires |

### The biology says the questions are real

B-cell clones have restricted fate sets and transcriptional memory (Swift,
Horns & Quake 2023, *Life Sci Alliance*); germinal centres produce plasma cells
across a range of affinities (Sprumont et al. 2023, *Cell*); high-affinity
germinal-centre clones divide more but mutate less per division (Merkenschlager
et al. 2025, *Nature*). A mathematical model of the germinal centre (Xiang et
al. 2026, *Cell Systems*) predicts that non-genetic cell states are variable
while B cells search for help but stable within proliferative clonal bursts,
and that this stability speeds up affinity maturation. That prediction is about
a quantity nobody measures directly from data: how much of a clone's state
survives a round of selection. Threadfin measures it.

## The gap Threadfin fills

| Question | Repertoire toolkits | Receptor-expression integration | Clone-level tools elsewhere | **Threadfin** |
|---|---|---|---|---|
| Clones from hypermutated sequences, within donors | yes | partly | no | **yes** |
| Is state inherited within clones, beyond where the cells were sampled? | no | no | measured, but only where clones are known and co-cultured | **yes**, with the sampling design accounted for |
| Which clones behave alike - **or is it a continuum?** | no | no (clusters are always returned) | no | **yes**: groups are reported only when the split is real |
| What explains differences between clones, with clones as replicates? | no (cell counts) | no | no | **yes** |
| Do clones keep their state over time, tissue or selection round? | no | trajectory-shaped | no | **yes** |
| How reliable is each clone's answer, given how few cells it has? | no | no | no | **yes** |

## In one paragraph

Treating a clone as the unit of analysis is not new - gene expression was shown
to be clonally heritable in cultured lymphocyte clones (Mold et al. 2024) - but
doing it on an immune repertoire is, because there the clones are inferred from
hypermutating receptors, most of them hold one to three cells, and which clone
a cell belongs to is tangled up with which sample, tissue or sort gate it came
from. Threadfin makes that measurement valid: it defines clones within donors,
describes each clone relative to the cells it was sampled with, reports how
reliable each clone's answer is given its size, reports groups of clones only
when the split between them is real, tests what explains the differences
between clones with clones as the replicates, and measures how much of a
clone's state survives over time, across tissues or across a round of
selection - all from one `threadfin.run()` call on standard AnnData objects.
