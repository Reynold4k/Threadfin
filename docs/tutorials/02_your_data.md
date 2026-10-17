# Tutorial 2 — Your own data

Threadfin needs two things for each cell: its gene expression and the B-cell receptor (BCR) that tells
which clone it belongs to. This tutorial goes from Cell Ranger output to results.

## 1. What to prepare

| Input | Format | Notes |
|---|---|---|
| expression | `AnnData`, one row per cell, raw counts in `X` (or `layers["counts"]`) | B cells only; doublets and low-quality cells removed |
| `obs["donor"]` | the individual each cell came from | clones are defined and compared within individuals |
| `obs["sample"]` | the library, sort or time point each cell was sequenced in | each clone is compared with the cells sampled alongside it |
| BCR | Cell Ranger `filtered_contig_annotations.csv`, an AIRR `.tsv`, or a contig-level `DataFrame` | one row per receptor chain, so that heavy and light chains can be paired |

Clones are defined with Dandelion's paired heavy- and light-chain definition (Suo et al. 2024), which
needs the `sc-dandelion` package. Dandelion pins older plotting libraries, so you may prefer to define
clones in a separate environment and pass the clone ids in `adata.obs["clone_id"]` instead (section 4).

## 2. One sample

```python
import scanpy as sc
import threadfin as tf

adata = sc.read_10x_h5("sample1/filtered_feature_bc_matrix.h5")
adata.obs["donor"] = "donor1"
adata.obs["sample"] = "sample1"

result = tf.run(adata, bcr="sample1/filtered_contig_annotations.csv",
                donor_key="donor", sample_key="sample")
print(result.summary())
```

`tf.run` defines the clones, builds an expression embedding without immunoglobulin genes (so that cells of
one clone do not look alike merely because they share receptor transcripts), estimates one profile per
clone, tests clonal coherence, lays out the clone map, finds programmes and reports their stability.

## 3. Several samples

Barcodes repeat across 10x libraries, so give every cell a unique name in both tables, for example by
prefixing the sample:

```python
import anndata as ad
import pandas as pd

samples = {"d0_LN": "donor1", "d7_LN": "donor1", "d0_blood": "donor2"}
parts, contigs = [], []
for sample, donor in samples.items():
    a = sc.read_10x_h5(f"{sample}/filtered_feature_bc_matrix.h5")
    a.obs_names = [f"{sample}_{b}" for b in a.obs_names]
    a.obs["sample"], a.obs["donor"] = sample, donor
    parts.append(a)
    c = pd.read_csv(f"{sample}/filtered_contig_annotations.csv")
    c["barcode"] = sample + "_" + c["barcode"].astype(str)
    contigs.append(c)

adata = ad.concat(parts)
contigs = pd.concat(contigs, ignore_index=True)

result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                batch_key="sample")          # integrate over technical batches only
```

Use `batch_key` only for technical batches (libraries, sequencing runs). Do not integrate over the biology
you want to study — tissue, time point or sort gate — because integration would remove the differences you
are asking about.

## 4. Clones defined elsewhere

If clone ids come from Dandelion run in its own environment, from Immcantation or from your own pipeline,
put them in `obs` and omit `bcr`:

```python
ids = pd.read_csv("clones.csv", index_col=0)["clone_id"]       # one row per cell
adata.obs["clone_id"] = ids.reindex(adata.obs_names).to_numpy()
result = tf.run(adata, donor_key="donor", sample_key="sample")
```

Clone ids must not be shared across donors. With Dandelion, `tf.define_clones_dandelion(contigs,
donor_key="donor")` prefixes the donor to every id.

## 5. Your own labels

Anything measured per cell can be tested against the clones: antigen binding, isotype, mutation load, sort
gate, tissue or time point.

```python
result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                state_key="cell_type",                         # your annotation, to describe programmes
                test=["antigen_binding", "isotype"],           # labels to test, clones as the unit
                time_key="timepoint")                          # do clones keep their state over time?
print(result.summary())
result.plot("threadfin_overview.pdf")
```

For each label, Threadfin first asks how much of the difference between clone profiles the label explains
(`result.profile_tests`), which needs no programmes, and then which programmes differ
(`result.tests`). Labels are shuffled among the clones of each donor, so a label that differs only between
donors cannot appear clonal. Labels that are constant within a donor (age, sex, disease group) cannot be
separated from other differences between donors; compare donors for those.

## 6. Saving

```python
result.clones.to_csv("threadfin_clones.csv")       # one row per clone
adata.write_h5ad("threadfin_result.h5ad")          # everything, including adata.uns["threadfin"]
```

Next: [Tutorial 3 — the clone map](03_clone_map.md).
