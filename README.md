# Threadfin

**Clone-level analysis of B-cell states from paired single-cell RNA-seq and BCR-seq.**

[![tests](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml/badge.svg)](https://github.com/Reynold4k/Threadfin/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org)

A B-cell clone is a family of cells descended from one ancestor that share a
B-cell receptor. Paired single-cell sequencing tells you, for every cell,
which clone it belongs to and what it is doing. Threadfin uses the **clone as
the unit of analysis** and answers three questions:

1. **Does clone identity shape cell state?** How much of the variation in gene
   expression is explained by which clone a cell belongs to, beyond when and
   where the cells were sampled.
2. **Which clones behave alike?** Genetically unrelated clones are grouped into
   *clonal programmes* by the states of their cells (for example clones biased
   towards plasma-cell output, or clones retained in the germinal centre).
3. **What distinguishes the programmes?** Clone-level tests against anything
   you measured: antigen binding, isotype, mutation load, infection, sort
   gate, and whether clones keep their state over time or across tissues.

Every result comes with an honest measure of uncertainty, and every test
counts clones, not cells.

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
| clone of every cell | `adata.obs["clone_id"]` | defined within donors, hypermutated variants merged |
| programme of every cell | `adata.obs["clone_programme"]` | the programme of the cell's clone (reliable clones) |
| extended programme labels | `adata.obs["clone_programme_assigned"]` | also smaller clones, assigned with posterior >= 0.7 |
| per-clone table | `result.clones` | size, reliability, programme, bootstrap confidence, posterior |
| clonal coherence | `result.coherence` | share of variance explained by clone, null, p-value |
| programme summary | `result.programmes` | size and bootstrap stability of each programme |
| label tests | `result.tests[label]` | clone-level odds ratios / effects, permutation p, FDR |
| clonal memory | `result.memory` | memory index (1 = clones keep their state, 0 = no memory) |

## How it works (one paragraph per step)

1. **Clones.** Cells are grouped into clones within each donor: same IGHV and
   IGHJ genes, same junction length, and junction sequences closer than a
   threshold found automatically from the data.
2. **Clone profiles.** Each clone is described relative to the cells it was
   sampled with. Small clones are shrunk towards their sample, large clones
   keep their own profile, and each clone gets a *reliability* score; the
   profile can describe the whole distribution of a clone's cells, so a clone
   split between two states is not mistaken for one sitting in between.
3. **Coherence test.** The share of transcriptional variance explained by
   clone identity is compared with clones shuffled *within samples*, so
   differences between samples are never mistaken for clonality.
4. **Programmes.** Reliable clones are grouped by similarity of their profiles;
   resampling the cells 30-50 times tells how stable each programme is
   (Jaccard >= 0.75 = stable). Smaller clones are then assigned to programmes
   with a probability.
5. **Tests and memory.** Labels are compared between programmes with clones as
   replicates and permutations within donors; clonal memory compares each
   clone's snapshots at different time points with snapshots of random clones,
   after removing sampling noise.

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
tf.tl.association_test(adata, "isotype")                  # 5. clone-level tests
tf.tl.programme_markers(adata)                            #    genes, clones as replicates
tf.tl.clonal_memory(adata, "timepoint")                   #    memory over time
tf.tl.gene_heritability(adata)                            #    which genes are clonally inherited
```

Figures: `tf.pl.clone_map`, `tf.pl.coherence`, `tf.pl.stability`,
`tf.pl.programme_composition`, `tf.pl.association`, `tf.pl.memory`,
`tf.pl.heritability`, `tf.pl.overview`.

## Validation

* **Simulations with a known answer** (`benchmarks/simulation/`): programme
  recovery against eight approaches, calibration of every test under the
  null, the memory index against the true switching rate, and scaling to
  millions of cells.
* **Seven public datasets** (`benchmarks/public_datasets/`), human and mouse,
  including a germinal-centre experiment with a fluorescent division reporter
  and antigen-bait sorting that provide experimental ground truth.

A step-by-step, biologist-friendly walkthrough of every dataset is in
[`report/PUBLIC_DATASETS_REPORT.md`](report/PUBLIC_DATASETS_REPORT.md); how
Threadfin relates to other tools is in [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md);
the figure plan for the manuscript is in [`paper/figure_plan/`](paper/figure_plan/).

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
