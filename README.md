# Threadfin

**Clone-level analysis of B-cell states from paired single-cell RNA-seq and BCR-seq.**

[![tests](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml/badge.svg)](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org)

A B-cell clone is a family of cells descended from one ancestor that share a
B-cell receptor. Paired single-cell sequencing tells you, for every cell, which
clone it belongs to and what it is doing. Threadfin uses the **clone as the
unit of analysis**.

This matters most where the biology is a cycle. In a germinal centre, B cells
divide and mutate their receptor, test it, and are then either sent back for
another round or leave as plasma or memory cells. Nothing in that cycle is a
beginning or an end, so ordering single cells along a pseudotime asks a
question the biology does not answer. A clone is different: all of its cells
descend from one ancestor, so a clone has a history even when its cells do not
have an order. Threadfin therefore compares clones with one another, and asks
which clones look as though they were recently selected and which look as
though they were sent back to divide again.

It answers four questions:

1. **Is B-cell state inherited within clones?** How much of the variation in
   gene expression is explained by which clone a cell belongs to, beyond when
   and where the cells were sampled.
2. **Which clones behave alike?** Clones are grouped only when the split
   between groups is real; otherwise they are reported as a continuum.
3. **What explains the differences between clones?** Clone-level tests against
   anything you measured: antigen binding, isotype, mutation load, division
   history, sort gate, infection, tissue, time.
4. **Do clones keep their state** over time, across tissues, or across the
   compartments of a germinal centre?

Every result comes with an honest measure of uncertainty, every test counts
clones rather than cells, and the output is a hypothesis about clones that
experiments still have to confirm.

---

## Installation

```bash
pip install "git+https://github.com/Reynold4k/Threadfin.git"
```

Python >= 3.10. Scanpy, Leiden and Harmony are installed automatically; no GPU
and no R are needed.

## Quick start

```python
import scanpy as sc
import threadfin as tf

adata = sc.read_h5ad("b_cells.h5ad")          # raw counts; one row per cell

result = tf.run(
    adata,
    bcr="filtered_contig_annotations.csv",    # Cell Ranger VDJ output (or an AIRR .tsv)
    donor_key="donor",                        # which person each cell came from
    sample_key="sample",                      # which sample / library each cell came from
)

print(result.summary())                       # plain-language report
result.plot("threadfin_overview.pdf")         # overview figure
```

That is all. `tf.run` defines clones from the BCR sequences (within each
donor), builds an expression embedding that ignores immunoglobulin genes,
tests whether clones are transcriptionally coherent, groups clones into
programmes and reports how stable they are.

**Add your own labels** to test them against the programmes, and a time column
to ask whether clones keep their state:

```python
result = tf.run(adata, bcr="filtered_contig_annotations.csv",
                donor_key="donor", sample_key="sample",
                state_key="cell_type",                    # your cell annotation (optional)
                test=["antigen_binding", "isotype"],      # any obs columns (optional)
                time_key="timepoint")                     # optional
```

**No data at hand?** Try it on simulated data with a known answer:

```python
adata = tf.sim.simulate_repertoire()
result = tf.run(adata, donor_key="donor", sample_key="context", basis="X_pca")
print(result.summary())
```

### What you get back

| Output | Where | Meaning |
|---|---|---|
| clone of every cell | `adata.obs["clone_id"]` | defined within donors, hypermutated relatives merged |
| programme of every cell | `adata.obs["clone_programme"]` | the programme of the cell's clone (reliable clones) |
| extended programme labels | `adata.obs["clone_programme_assigned"]` | also smaller clones, assigned with posterior >= 0.7 |
| per-clone table | `result.clones` | size, reliability, programme, bootstrap confidence, posterior |
| clonal coherence | `result.coherence` | share of variance explained by clone, null, p-value |
| programme summary | `result.programmes` | size and bootstrap stability of each programme |
| label vs clone states | `result.profile_tests[label]` | share of clone-profile variance explained by the label, permutation p |
| label vs programmes | `result.tests[label]` | per-programme odds ratios / effects, permutation p, FDR (needs >= 2 programmes) |
| clonal memory | `result.memory` | memory index (1 = clones keep their state, 0 = no memory) |

## How it works (one paragraph per step)

1. **Clones.** Cells are grouped into clones within each donor: same IGHV and
   IGHJ genes, same junction length, and junction sequences closer than a
   threshold found automatically from the data, so that hypermutated relatives
   stay together.
2. **Clone profiles.** Each clone is described relative to the cells it was
   sampled with. Small clones are shrunk towards their sample, large clones
   keep their own profile, and each clone gets a *reliability* score; the
   profile can describe the whole distribution of a clone's cells, so a clone
   split between two states is not mistaken for one sitting in between.
3. **Coherence test.** The share of transcriptional variance explained by
   clone identity is compared with clones shuffled *within samples*, so
   differences between samples are never mistaken for clonality.
4. **Programmes.** Reliable clones are grouped by similarity of their
   profiles. Every split between groups must pass a significance test against
   a single-group model, so clones that only vary along a continuum are
   reported as one group instead of being cut into artificial programmes.
   Resampling the cells 30-50 times tells how stable each programme is
   (Jaccard >= 0.75 = stable). Smaller clones are then assigned to programmes
   with a probability.
5. **Tests and memory.** A label is first tested for whether it explains how
   clone profiles differ at all (this works even without distinct
   programmes), then compared between programmes; clones are the replicates
   and labels are shuffled within donors. Clonal memory compares each clone's
   snapshots at different time points with snapshots of random clones, after
   removing sampling noise.

Full statistical details: [`docs/METHODS.md`](docs/METHODS.md).

## Step by step (advanced)

`tf.run` chains functions you can also call yourself, for example to change
the sampling context, the number of permutations or the programme resolution:

```python
bcr = tf.read_bcr("filtered_contig_annotations.csv")
bcr["donor"] = adata.obs["donor"].reindex(bcr.index)
bcr = tf.define_clones(bcr, donor_key="donor")             # 1. clones
tf.attach_bcr(adata, bcr)
tf.pp.prepare_embedding(adata, batch_key="donor")         # embedding without IG genes
tf.tl.clone_profiles(adata, basis="X_threadfin",          # 2. clone profiles
                     context_key="sample", donor_key="donor")
tf.tl.clonal_coherence(adata)                             # 3. coherence test
tf.tl.find_programmes(adata)                              # 4. programmes
tf.tl.profile_association(adata, "isotype")               # 5. label vs clone states
tf.tl.association_test(adata, "isotype")                  #    label vs programmes
tf.tl.programme_markers(adata)                            #    genes, clones as replicates
tf.tl.clonal_memory(adata, "timepoint")                   #    memory over time
tf.tl.gene_heritability(adata)                            #    which genes are clonally inherited
```

Figures: `tf.pl.clone_map`, `tf.pl.coherence`, `tf.pl.stability`,
`tf.pl.programme_composition`, `tf.pl.association`, `tf.pl.memory`,
`tf.pl.heritability`, `tf.pl.overview`.

## What Threadfin found in public data

Eight published datasets, human and mouse, were re-analysed with the same
script (`case_studies/`). A few of the findings:

**Germinal centres have no beginning and no end**, so ordering single cells
along a pseudotime asks a question the biology does not answer. Comparing
*clones* works instead, because a clone has a history even when its cells do
not have an order. In two mouse experiments with a model antigen, where cells
were sorted by how often they had divided and by germinal-centre zone:

* clone identity explained 9-15% of B-cell state, far more than shuffled
  clones (p = 0.002), but never most of it: a clone biases what its cells do;
* clones did **not** fall into distinct programmes - they differ along a
  continuum;
* how many times a clone's cells had divided was the strongest explanation of
  how clones differ (18% and 8%), with the dark/light-zone sort comparable
  (9%), while antigen binding and mutation load explained much less;
* about half of what distinguishes a clone was still recognisable when its
  cells were caught in a different part of the cycle.

**In a vaccinated human lymph node**, spike-binding clones were concentrated in
the germinal-centre and one antibody-secreting group of clones (odds ratios
around 3 across donors). Clones kept their state over months (memory index
0.26), but the same clone's cells in blood and lymph node did not resemble each
other at all - where a cell is matters more than which clone it came from.

**Seven days after influenza vaccination**, the antibody-secreting burst came
from class-switched, mutated clones, as expected if it is recall of existing
memory rather than a new response.

These are associations in observational data: Threadfin produces hypotheses
about clones that need experiments to confirm. The step-by-step walkthroughs
are in [`report/PUBLIC_DATASETS_REPORT.md`](report/PUBLIC_DATASETS_REPORT.md)
and, for the germinal centre,
[`report/GERMINAL_CENTRE_CASE_STUDIES.md`](report/GERMINAL_CENTRE_CASE_STUDIES.md);
the analysis code is in [`case_studies/`](case_studies/); how Threadfin relates
to other tools is in [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md); the
manuscript figure plan is in [`paper/figure_plan/`](paper/figure_plan/).

## Upgrading from v3

Version 4 replaces the v3 statistics, several of which were not valid (for
example, community "transitions" over time were diagonal by construction and
enrichment tests counted cells of the same clone as independent). The v3
functions remain importable; see [`docs/MIGRATION_V3_TO_V4.md`](docs/MIGRATION_V3_TO_V4.md)
for what changed and why. The v3 code and results are preserved at tag `v3.0.0`.

## Citation

If you use Threadfin, please cite this repository (see [`CITATION.cff`](CITATION.cff)).
A manuscript is in preparation.

## License

[MIT](LICENSE) © 2026 Chen Satoshi (Reynold4k)
