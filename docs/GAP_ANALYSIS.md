# Gap analysis: what is missing in single-cell BCR repertoire analysis

*This document is the evidence base for Threadfin's design. Every claim is
backed by a primary quote (paper or GitHub issue, with link). Written
2026-09-23; issue states may have changed since.*

---

## The question

Paired scRNA-seq + scBCR-seq measures, for each B cell, both **what receptor
it carries** (the V(D)J sequence) and **what it is doing** (the
transcriptome). The biological questions that matter for clonal research —
*which clones respond to the antigen, which clones become plasma cells or
memory, how clones move between tissues over time* — live at the
intersection of the two modalities. We asked: **which existing tool can
answer "what are the clones doing, and which clones are doing the same
thing?" at clonal resolution, with statistical support?**

## Gap 1 — No tool infers transcriptional-state communities of clones

| Tool | What it actually does with clones | Why it is not clone-level state inference |
|---|---|---|
| **Benisse** ([Zhang et al. 2022, *Nat Mach Intell*](https://www.nature.com/articles/s42256-022-00492-6)) | Embeds BCR CDR3H (Atchley factors, torch encoder) and learns a sparse BCR graph guided by GEX via ADMM | The graph is **constrained by sequence**: *"two BCR clones share the same V/J genes and are connected in the crude graph"* (Supplementary Note). It cannot group sequence-unrelated clones that converge on the same state — precisely the biology of convergent responses. No fate tracking, no clonal diffusion ordering. |
| **CoNGA** ([Schattgen et al. 2022, *Nat Biotechnol*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8832949/)) | Tests per-clonotype overlap between GEX and TCR neighbour graphs | A **correlation test**, not a grouping: it scores whether a clonotype's sequence neighbours are also expression neighbours. TCR-only (tcrdist does not model SHM/CSR); dandelion's authors measured that *"Both CoNGA and mvTCR failed to preserve the intercellular relationships"* on their data. |
| **TESSA** ([Zhang et al. 2021, *Nat Methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7799492/)) | TCR embedding network reweighted by GEX | Networks are defined by sequence similarity to estimate **antigen specificity**, not transcriptional state. CDR3β only; T cells only; repo unmaintained since 2021-10. |
| **mvTCR** ([Drost et al. 2024, *Nat Commun*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220149/)) | Cell-level multi-view VAE embedding of GEX+TCR | Embedding is **cell-level** (*"Even within cells of a clonotype, mvTCR expressed the variability of the transcriptome"*) — no clone-as-unit concept. BCR explicitly deferred: *"an in-depth benchmark must be conducted upon the availability of large-scale B cell datasets with specificity annotation."* Requires GPU. |
| **dandelion** ([Suo et al. 2024, *Nat Biotechnol*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10791579/)) | V(D)J feature space = V/J **usage frequencies** over cell pseudobulks; Palantir trajectory in that space | Pseudobulk-of-cells, not clone-level; no CDR3 similarity in the embedding; trajectory is developmental, not clonal fate. |
| **scirpy** ([Sturm et al. 2020](https://doi.org/10.1093/bioinformatics/btaa611)) | Clonotype definition, repertoire stats, `clonotype_modularity` | `clonotype_modularity` measures whether cells of one clone cluster in GEX — a single scalar per comparison, no grouping of clones into communities, no fate. |
| **scRepertoire** ([Borcherding et al. 2021](https://f1000research.com/articles/10-230/v1)) | Clonotype counting/overlap overlaid on Seurat clusters | Descriptive overlays; authors' stated scope is *"processing …, descriptive statistics, clonal comparisons, and repertoire diversity"*. |
| **Immcantation/dowser** ([Gupta et al. 2015](https://doi.org/10.1093/bioinformatics/btv359)) | Clonal families and lineage trees from sequences | Trees encode receptor ancestry. The 2024 review notes the suite *"do not explicitly interact with the scRNA-seq analysis tool kits"* ([Irac et al., *Nat Methods* 2024](https://doi.org/10.1038/s41592-024-02243-4)). |

## Gap 2 — No tool ships statistical validation of clone–state coupling

The closest statistical devices in the literature answer *different*
questions: CoNGA's score tests sequence-neighbourhood/expression-neighbourhood
overlap per clonotype; TESSA randomized cluster labels 10,000× to validate
that its networks capture antigen specificity. **No published tool tests
whether clonal identity carries transcriptional information at all** (a
clone-label permutation null), **whether clone groupings replicate on
held-out cells**, or **whether they survive a change of embedding basis**.
Threadfin runs all three on every dataset, and reports failures as-is.

## Gap 3 — Clone fate tracking across time and tissue is chronically unmet

Public evidence that users need this and cannot get it:

* scirpy issue [#36](https://github.com/scverse/scirpy/issues/36) — the
  maintainers proposed STARTRAC-style migration/expansion/transition indices
  in **April 2020; still open**.
* scRepertoire issue [#585](https://github.com/BorchLab/scRepertoire/issues/585)
  (2026-08) — users get a STARTRAC table but *cannot interpret it*;
  [#428](https://github.com/BorchLab/scRepertoire/issues/428),
  [#399](https://github.com/BorchLab/scRepertoire/issues/399) — longitudinal
  clone tracking requested repeatedly.
* scRepertoire [#578](https://github.com/BorchLab/scRepertoire/issues/578)
  (2026-06) — users need dowser-style lineage information attached to
  expression analysis.
* dandelion's trajectory workflow requires users to supply their own scVI
  embedding ([discussion #303](https://github.com/tuonglab/dandelion/discussions/303))
  and operates on pseudobulks, not clones.

## Gap 4 — Real repertoires break sequence-first tools

The dominant feature of real data is singletons (median clone size = 1 in
all five of our validation datasets; *"the majority of clonotypes were
singletons and only 9–18% of patients' clonotypes were clonally expanded"* —
[Sturm et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7751015/)).
Consequences visible in public issue trackers:

* dowser [#38](https://github.com/immcantation/dowser/issues/38) — after
  clone filtering, *zero* clones remain for tree building.
* dandelion [#235](https://github.com/tuonglab/dandelion/issues/235) —
  kernel dies building a cell-level clone network past ~20k cells;
  [#517](https://github.com/tuonglab/dandelion/issues/517) — cells with >1
  VDJ chain silently dropped from clone calling.
* CoNGA [#49](https://github.com/phbradley/conga/issues/49) — 7,744 of
  7,846 clones are singletons; downstream plots produce nothing.
* scirpy [#201](https://github.com/scverse/scirpy/issues/201) —
  `clonotype_network` errors out on a repertoire of ~12k near-all-singleton
  clones.

Cell-level sequence networks are quadratic in the wrong variable. Treating
the **clone as the node** (a few hundred to a few thousand expanded clones)
is the scalable design — and, as our tonsil case study shows,
sequence-*similarity* clone definition is often required before any
clone-level analysis is possible at all.

## Gap 5 — The integration tools that exist are hard to run

Benisse: PyTorch encoder + R core, pinned `torch==2.2.2`, 9 manual
hyperparameters, pretrained model undocumented
([#14](https://github.com/wooyongc/Benisse/issues/14)); installation issues
[#5](https://github.com/wooyongc/Benisse/issues/5),
[#6](https://github.com/wooyongc/Benisse/issues/6),
[#12](https://github.com/wooyongc/Benisse/issues/12). TESSA: pinned Python
3.6.4/Keras 2.2.4/R 3.5.1, unmaintained since 2021. mvTCR: requires GPU;
roughly a third of its issues are install/dependency failures
([#10](https://github.com/SchubertLab/mvTCR/issues/10),
[#21](https://github.com/SchubertLab/mvTCR/issues/21)). CoNGA: C++ TCRdist +
ImageMagick/Inkscape. **A pure-Python, scanpy-native, CPU-only package is
itself a contribution.**

## Why the field agrees this gap matters

* *"All current analytical approaches for BCRs solely investigate the BCR
  sequences and ignore their correlations with the transcriptomics of the
  B cells, yielding conclusions of unknown functional relevance"* — Benisse
  abstract, 2022.
* *"a fundamental limitation of these approaches is that all conclusions are
  drawn based on solely interrogating the TCR sequences"* — TESSA, 2021.
* *"T cells sharing the same TCR … distribute non-randomly across gene
  expression-based clusters … a layer of T cell transcriptional diversity is
  clonally inherited"* — mvTCR, 2024 (the clonal-state layer is real
  biology).
* *"systematic approaches that can identify previously unknown populations
  … by correlating gene expression and TCR sequence have not been reported"*
  — CoNGA, 2022.
* Irac et al., *Nat Methods* 2024 review: integration methods exist but
  *"none of the current scTCR/BCR-seq analysis tool kits … provide native
  implementations or direct linkage to these methods"* — the toolchain
  fracture is explicit.
* LIBRA-seq (Setliff et al., *Cell* 2019): *"BCR sequencing … provides
  limited information about the antigen specificity of the sequenced BCRs"* —
  the specificity-annotation gap that reference-database matching addresses.

## Threadfin's position

Threadfin is built to close exactly these gaps:

1. **Clone communities in transcriptional state space** (Gap 1) —
   `clonotype_recluster`, with the BCR layer used where it helps
   (`define_clones` upstream, isotype/SHM annotation downstream, coupled
   graph embedding where sequence edges exist).
2. **Validation as part of the method** (Gap 2) — within-donor permutation
   nulls, size-matched community nulls, held-out split-clone replication,
   cross-basis robustness, all shipped in
   `benchmarks/biological_validation/` and run on 5 public datasets.
3. **Clonal fate tracking** (Gap 3) — `clone_fate_table`,
   `community_transition`, migration/transition indices, and
   diffusion-based ordering of the clone graph.
4. **Clone-as-node scalability and sequence-aware clone definition**
   (Gap 4) — runtime scales with clone count, not cell count; tonsil case
   study (13 usable exact clonotypes → 200 lineages).
5. **Pure Python / scanpy-native / CPU-only** (Gap 5) — AnnData in, AnnData
   out; optional parasail acceleration; no GPU, no R, no external binaries.

And the v3 discovery layer (below) goes from *description* to *biological
reconstruction*: antigen-specificity annotation against reference databases
(CoV-AbDab) or author-validated labels, per-community gene programmes, SHM
gradients, and cross-tissue/ cross-time migration indices.
