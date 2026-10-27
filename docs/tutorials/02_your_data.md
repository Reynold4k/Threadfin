# Tutorial 2 — Your own data

Threadfin needs two things for each cell: its **gene expression** and its **B-cell receptor (BCR)**, which
tells which clone it belongs to. T cells work the same way with their TCR (section 6). This tutorial goes
from Cell Ranger output to results, explains each decision you have to make, and lists the errors you are
most likely to meet and how to fix them. For a
complete worked example on a published dataset, see [Tutorial 5](05_end_to_end_tutorial.ipynb).

---

## 0. Before you start: a checklist

| Question | Why it matters |
|---|---|
| Do I have **gene expression** and **BCR** for the *same* cells (5′ 10x, or plate-based with TRUST4/MiXCR)? | Clones come from the BCR; states come from the RNA; the two are linked by the cell barcode. |
| Do I have the **contig-level** BCR table — one row per receptor chain? | Clones are defined from heavy *and* light chain; a per-cell summary cannot be paired. Cell Ranger's `filtered_contig_annotations.csv` and `airr_rearrangement.tsv` are contig-level. |
| Do I know which **individual** (donor, mouse) each cell came from? | Clones are defined within individuals, and the tests of your labels shuffle them within individuals. |
| Do I know the **sample** (library / sort / time point) of each cell? | Each clone is compared with the cells sampled with it. |
| Are there enough clones with **two or more cells**? | With 40 such clones the coherence test detects a clonal ICC of 0.05 nine times in ten, and about 10 clones suffice for an ICC of 0.1 ([power simulation](../../manuscript/analyses/technical_validation/power/README.md)); programmes and label tests ask for more, typically hundreds. The 15 public datasets analysed with Threadfin have 96–11,141. |
| Are **plasma cells and memory B cells** included, if they were sampled? | Clones span these states; removing them removes exactly the clone structure Threadfin describes. |

> **Rule of thumb for experiment design.** Threadfin learns about clones from *repeated sampling of the same
> clone*. Sorting a single compartment (for example only germinal-centre B cells) keeps clones but removes
> the states they connect; sampling a whole response (GC, plasma, memory) gives clones more to say.

---

## 1. What to prepare

| Input | Format | Notes |
|---|---|---|
| expression | `AnnData`, one row per cell, **raw counts** in `X`, or in a layer named with `counts_layer=` (for example `tf.run(..., counts_layer="counts")`) | B cells (including plasma cells); doublets and low-quality cells removed |
| `obs["donor"]` | the individual each cell came from | mouse id, participant id |
| `obs["sample"]` | the library, sort or time point each cell was sequenced in | used as the sampling context — see section 5 |
| BCR | Cell Ranger `filtered_contig_annotations.csv`, an AIRR `.tsv`, a contig-level `DataFrame`, or a scirpy object | one row per receptor chain, so that heavy and light chains can be paired |

Clones are defined with Dandelion's paired heavy- and light-chain definition (Suo et al. 2024), which needs
the `sc-dandelion` package. Dandelion 0.5.7 uses a matplotlib function that was removed in matplotlib 3.9,
so install both together: `pip install "threadfin[dandelion]"`
or `pip install "sc-dandelion>=0.5.7,<1" "matplotlib<3.9"` (Threadfin works with matplotlib 3.7 and later; it is
validated with Dandelion 0.5.x, and Dandelion 1.0 calls clones differently). If you prefer,
define clones in a separate environment or with another clone caller and pass the clone ids directly
(section 6).

*Reproducible clone ids.* Dandelion's clone assignment depends on the order in which Python iterates over
sets of sequences, which is randomised in every new Python session. Repeated runs can therefore differ by a
few clones out of thousands (the conclusions should not change). Start Python with `PYTHONHASHSEED=0`
(in a terminal: `PYTHONHASHSEED=0 python my_analysis.py`; for Jupyter, register the kernel with
`python -m ipykernel install --user --name threadfin --env PYTHONHASHSEED 0`) when the ids must be identical
every time; the seed in use is stored in `clone_ids.attrs["python_hash_seed"]`.

---

## 2. Quality control and the right cells

**What.** Remove empty droplets, damaged cells and doublets, and keep the B-lineage cells.

**Why.** Damaged cells (few genes, many mitochondrial reads) and doublets (two cells in one droplet) create
artificial "states". A doublet of a germinal-centre B cell and a plasma cell would even look like a clone
spanning both states.

```python
import scanpy as sc

adata = sc.read_10x_h5("sample1/filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], multi_panel=True)

# starting points: inspect the plots and adapt them to your tissue and chemistry
adata = adata[(adata.obs["n_genes_by_counts"] >= 200) &
              (adata.obs["total_counts"] >= 500) &
              (adata.obs["pct_counts_mt"] < 15)].copy()
```

* *Thresholds* depend on tissue and chemistry; read them off the violin plots rather than copying them.
  Plasma cells have many immunoglobulin transcripts and sometimes fewer detected genes — do not set the
  gene threshold so high that they disappear.
* *Doublets*: if your samples were hashtagged, remove cells positive for two hashtags (Tutorial 5, Step 3);
  otherwise a doublet detector such as Scrublet (`sc.pp.scrublet`) helps.
* *B-lineage cells*: keep cells annotated as B cells, germinal-centre B cells, plasmablasts, plasma cells
  and memory B cells. Marker genes that help: *CD19*, *MS4A1*, *CD79A* (B cells), *BCL6*, *AICDA* (germinal
  centre), *PRDM1*, *XBP1*, *JCHAIN*, *SDC1* (plasma cells). Plasma cells have little *MS4A1*/CD20, so do not
  select B cells on *MS4A1* alone.
* Keep the **raw counts**: Threadfin normalises them itself.

---

## 3. One sample

```python
import threadfin as tf

adata.obs["donor"] = "donor1"
adata.obs["sample"] = "sample1"

result = tf.run(adata, bcr="sample1/filtered_contig_annotations.csv",
                donor_key="donor", sample_key="sample")
print(result.summary())
```

`tf.run` (1) defines the clones; (2) builds an expression embedding without immunoglobulin genes, so that
cells of one clone do not look alike merely because they share receptor transcripts; (3) estimates one
profile per clone; (4) tests clonal coherence; (5) lays out the clone map; (6) finds programmes and their
stability; and (7) runs the tests you asked for. [Tutorial 1](01_getting_started.md#2-one-call) explains how
to read the summary.

**What to look for.** The first line of the summary says how many cells have a BCR and how many clones have
at least three cells. If almost no cell has a BCR, the barcodes did not match (section 4).

---

## 4. Several samples: make every barcode unique

**What.** Load each sample, prefix every barcode with its sample name in *both* the expression and the BCR
table, and concatenate.

**Why.** 10x barcodes are drawn from a fixed whitelist, so the same barcode occurs in different libraries.
Without a prefix, cells of different samples would be merged, and BCRs would be attached to the wrong cells.

```python
import anndata as ad
import pandas as pd

samples = {"d0_LN": "donor1", "d7_LN": "donor1", "d0_blood": "donor2"}
parts, contigs = [], []
for sample, donor in samples.items():
    a = sc.read_10x_h5(f"{sample}/filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    a.obs_names = [f"{sample}_{b}" for b in a.obs_names]
    a.obs["sample"], a.obs["donor"] = sample, donor
    parts.append(a)
    c = pd.read_csv(f"{sample}/filtered_contig_annotations.csv")
    c["barcode"] = sample + "_" + c["barcode"].astype(str)
    contigs.append(c)

adata = ad.concat(parts)
contigs = pd.concat(contigs, ignore_index=True)

# check the match before running anything else
matched = contigs["barcode"].isin(adata.obs_names)
print(f"{matched.mean():.0%} of BCR contigs match a cell of the expression data")

result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                batch_key="sample")          # integrate over technical batches only
```

**What to look for.** In the 14 public datasets tested with this tutorial, between a quarter and all of the
contigs matched a cell; the rest belong to droplets removed by QC, hashtag demultiplexing or the selection of
B cells, so a low share is not by itself a problem. A match near 0 % means that the prefixes or the `-1`
suffixes differ between the two tables.

---

## 5. Choosing `donor_key`, `sample_key` and `batch_key`

These three arguments decide what Threadfin treats as *nuisance* — differences that should not count as
differences between clones.

| Argument | What it does | Use | Never use |
|---|---|---|---|
| `donor_key` | clones are defined within individuals; tests shuffle labels among clones of one individual | the individual | — |
| `sample_key` | each clone is described relative to the cells of its context; coherence shuffles clone labels within it | the donor, a library, or donor × time point when time is *not* what you study | anything you want to compare between clones: tissue, sort gate, treatment, time point |
| `batch_key` | Harmony integration when the embedding is built | technical batches (libraries, runs) | biology |

**Why the context must be a nuisance variable.** Threadfin subtracts each context's average before
comparing clones. If the context were the tissue, every clone would be compared only with the cells of its
own tissue, and differences between tissues — the thing you wanted to see — would be removed. Two examples:

* *Lymph node and blood, several visits per donor.* Use `sample_key="donor"` if you want to see lymph-node
  versus blood clones; test `tissue` afterwards with `test=["tissue"]`.
* *Cells sorted into gates (for example GC versus plasma cells, or a division reporter), each gate a library.*
  Use the donor as the context, and keep in mind that a gate sorted into its own library cannot be separated
  from differences between libraries (Tutorial 5, Step 13).

**Why the coherence test shuffles within samples.** Cells of one clone are usually sequenced together. If
samples differ in their cell-type composition, any analysis that ignores samples would call that clonal.
Shuffling clone labels within samples keeps every sample's composition and destroys only the link between a
cell's clone and its state.

---

## 6. How clones are called

**The default.** With `bcr=`, clones are called with Dandelion within each donor: cells belong to one clone
when they use the same V and J genes on both chains (heavy and light; for T cells, beta and alpha), with
junctions of the same length that are at least 85 % identical in amino acids (B cells, to allow for somatic
hypermutation) or identical (T cells, which do not mutate). These are Dandelion's own defaults. Cells without
a heavy (beta) chain are left without a clone, because light chains alone are not diverse enough to define
one.

**Changing the settings.** `clone_options` passes settings to `tf.define_clones_dandelion`:

```python
result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                clone_options={"identity": 0.9})           # stricter junction identity
result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                clone_options={"check_contigs": True})     # Dandelion's contig check first
```

**Cells with two chains of one kind.** In droplet data 4–20 % of cells carry two productive light chains or,
less often, two heavy chains: allelic inclusion, ambient RNA or doublets. Dandelion gives such a cell a clone
id that joins both chains, so the cell usually forms a clone of its own and adds nothing to an analysis that
compares the cells within clones; the log says how many cells this affects (when it is more than 2 %), and how
many of them have two heavy chains. `check_contigs=True` runs Dandelion's contig check first: a cell with two
heavy chains keeps the one that clearly dominates in UMIs, or is set aside when none does; two light chains are
allowed, as in allelic inclusion. The case studies use the default. On two of them, the check and junction
identities from 0.75 to 1.0 left the conclusions unchanged
([clone-definition sensitivity](../../manuscript/analyses/technical_validation/clone_definition/README.md)).

**T cells.** Pass the TCR contigs (Cell Ranger's `filtered_contig_annotations.csv` from the TCR library, or
an AIRR table) as `bcr`, exactly as for B cells. TCR genes are left out of the expression embedding, as
immunoglobulin genes are for B cells, so that cells of one clone do not look alike merely because they
express the same receptor.

**scirpy users.** Receptor data already loaded with [scirpy](https://scirpy.scverse.org) (in
`obsm["airr"]`) can be passed directly:

```python
import scirpy as ir

airr = ir.io.read_10x_vdj("filtered_contig_annotations.csv")     # AnnData with obsm["airr"]
result = tf.run(adata, bcr=airr, donor_key="donor", sample_key="sample")
# or, from a MuData that holds both: tf.run(mdata["gex"], bcr=mdata, ...)
```

### Clones defined elsewhere

If clone ids come from Dandelion run separately, from Immcantation or from your own pipeline, put them in
`obs` and omit `bcr`:

```python
ids = pd.read_csv("clones.csv", index_col=0)["clone_id"]       # one row per cell
adata.obs["clone_id"] = ids.reindex(adata.obs_names).to_numpy()
result = tf.run(adata, donor_key="donor", sample_key="sample")
```

Clone ids must not be shared across donors. To call them yourself with Threadfin's settings (for example to
inspect or save them before the analysis), give every contig its donor and let `tf.define_clones_dandelion`
prefix the donor to every id:

```python
contigs = tf.read_contigs(contigs)                    # a path or the table of section 4; AIRR column names
contigs = contigs[contigs["cell_id"].isin(adata.obs_names)]
contigs["donor"] = contigs["cell_id"].map(adata.obs["donor"].astype(str))
ids = tf.define_clones_dandelion(contigs, donor_key="donor")      # one clone id per cell
adata.obs["clone_id"] = ids.reindex(adata.obs_names).to_numpy()
result = tf.run(adata, donor_key="donor", sample_key="sample")
print(result.methods(ids))                            # describes the clone calling too
```

> **Why paired chains?** Defining clones on the heavy chain alone merges unrelated cells that happen to use
> similar heavy chains. In the published datasets analysed with Threadfin, adding the light chain split
> over-merged clones (one 791-cell "clone" became clones of at most 254 cells) and increased clonal coherence
> in every dataset that had light chains.

---

## 7. Your own labels

Anything measured per cell can be tested against the clones: antigen binding, isotype, mutation load, sort
gate, tissue or time point. `test=` takes columns of `adata.obs`, so add them before the run. Isotype, for
example, comes from the constant gene of the heavy-chain contig (Cell Ranger's `c_gene`, AIRR `c_call`: IGHM,
IGHA1, IGHG2B …); the first four letters give the class. `tf.read_contigs` gives the table AIRR column names,
whether or not you ran the snippet of section 6:

```python
airr = tf.read_contigs(contigs)
heavy = airr[airr["locus"] == "IGH"].drop_duplicates("cell_id").set_index("cell_id")
adata.obs["isotype"] = heavy["c_call"].reindex(adata.obs_names).str[:4].to_numpy()   # IGHM, IGHA, IGHG ...
```

```python
result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="sample",
                state_key="cell_type",                         # your annotation, to describe programmes
                test=["antigen_binding", "isotype"],           # labels to test, clones as the unit
                time_key="timepoint")                          # do clones keep their state over time?
print(result.summary())
result.plot("threadfin_overview.pdf")
```

For each label, Threadfin first asks how much of the difference between clone profiles the label explains
(`result.profile_tests`), which needs no programmes, and then which programmes differ (`result.tests`).
Labels are shuffled among the clones of each donor, so a label that differs only between donors cannot
appear clonal.

* A **categorical label** (isotype) is summarised per clone as its majority value; use
  `tf.tl.profile_association(adata, "isotype", how="fraction:IGHA")` to test the fraction of one level (the
  level must be spelled as in the column; Threadfin warns when it never occurs).
* A **numeric label** (mutation frequency) is summarised as the clone mean.
* Labels **constant within a donor** (age, sex, disease group) cannot be separated from other differences
  between donors; compare donors for those.
* Labels derived from the same **RNA** (a cell-type annotation) are not independent of the clone profiles;
  they describe programmes but do not validate them.

**Reading a test result.** `r2` is the share of variation between clone profiles explained by the label;
`null_mean` is what shuffled labels explain; `excess_r2` is their difference; `p_value` is the permutation
P value (with 2,000 permutations it cannot be smaller than 0.0005). Small R² values are normal: a label is
one coarse measurement, the profile describes the whole transcriptome.

---

## 8. Saving

```python
result.clones.to_csv("threadfin_clones.csv")       # one row per clone with >= 2 cells
adata.write_h5ad("threadfin_result.h5ad")          # everything, including adata.uns["threadfin"]
print(result.methods())                            # a Methods paragraph with the settings and versions used
```

`result.methods()` describes what was run — clone calling, embedding, profiles, tests — with the settings and
package versions of this analysis and the references it relies on, as a starting point for the Methods section
of a paper. Check it against your own description of the data.

---

## 9. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `No barcode of the receptor data matches adata.obs_names` | the barcode formats differ (prefix, `-1` suffix) | give the `barcode` (or AIRR `cell_id`) column of the contig table the same prefix or suffix as `adata.obs_names`, as in section 4, and pass the table |
| `A per-cell BCR table cannot be used for clone definition` | a one-row-per-cell table was passed | pass the contig-level table (`filtered_contig_annotations.csv` or AIRR) |
| `Only N clone(s) have two or more cells` | too few cells per clone were captured, or BCRs were not attached | check the match rate (section 4); check that clones were called within donors; a dataset with very few repeated clones cannot be profiled |
| `N clone id(s) occur in more than one donor` | clone ids were not made donor-private | prefix the donor (the message shows the one-line fix), or call clones with `donor_key` |
| `Counts contain NaN, infinite, or negative values` | `adata.X` holds normalised or scaled data | put raw counts in `X`, or name the layer that holds them with `counts_layer=` (Threadfin 4.5.2 or later), or pass your own embedding with `basis=` |
| `'X_...' has NaN or infinite values in N of M cells` | your own embedding (`basis=`) has missing values | drop those cells first; the message shows the line of code |
| `'donor' not found in adata.obs` (or any other column) | a column name is misspelled or missing | the message names the column; check `adata.obs.columns` |
| `Clone calling needs the Dandelion package` | Dandelion is not installed | `pip install "threadfin[dandelion]"`, or pass clone ids yourself (section 6) |
| `sc-dandelion 1.x is installed; Threadfin … is validated with sc-dandelion 0.5.x` | Dandelion 1.0 calls clones differently | `pip install "sc-dandelion<1"`; results then match the documentation |
| `cannot import name 'get_cmap' from 'matplotlib.cm'` | Dandelion 0.5.7 with matplotlib ≥ 3.9 | `pip install "matplotlib<3.9"` |
| `No resolution reached a clone-weighted mean stability of 0.75` | the programmes are, on average, less stable than the threshold | judge each programme by its own stability (`result.programmes`); the summary marks those below 0.6 as unstable |
| coherence high but the shuffled value is also high (> 10 %) | samples differ strongly in composition | expected; Threadfin accounts for it — make sure `sample_key` is set |
| a programme is dominated by one donor | donors differ in their sampled composition, and centring shifts all clones of a donor together | look at programmes by donor; compare with `sample_key=None` (no centring); see Methods, "centring on the sampling context" |

Next: [Tutorial 3 — the clone map](03_clone_map.md).
