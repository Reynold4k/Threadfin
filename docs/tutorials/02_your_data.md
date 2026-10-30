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
| Are there enough clones with **two or more cells**? | The coherence test detects a clonal ICC of 0.1 with 80 % power from about 10–20 such clones (fewer when the clonal signal is spread over many dimensions of the embedding), and an ICC of 0.05 from about 40 or more ([power simulation](../../manuscript/analyses/technical_validation/power/README.md)); programmes and label tests ask for more, typically hundreds. The 15 public datasets analysed with Threadfin have 96–11,141. |
| Are **ASCs (plasmablasts and plasma cells) and memory B cells** included, if they were sampled? | Clones span these states; removing them removes exactly the clone structure Threadfin describes. |

> **Rule of thumb for experiment design.** Threadfin learns about clones from *repeated sampling of the same
> clone*. Sorting a single compartment (for example only germinal-centre B cells) keeps clones but removes
> the states they connect; sampling a whole response (GC B cells, ASCs, memory B cells) gives clones more to
> say.

---

## 1. What to prepare

| Input | Format | Notes |
|---|---|---|
| expression | `AnnData`, one row per cell, **raw counts** in `X`, or in a layer named with `counts_layer=` (for example `tf.run(..., counts_layer="counts")`) | B-lineage cells (including ASCs); doublets and low-quality cells removed |
| `obs["donor"]` | the individual each cell came from | mouse id, participant id |
| `obs["sample"]` | the library, sort or time point each cell was sequenced in | used as the sampling context — see section 5 |
| BCR | Cell Ranger `filtered_contig_annotations.csv`, an AIRR `.tsv`, a contig-level `DataFrame`, or a scirpy object | one row per receptor chain, so that heavy and light chains can be paired |

Clones are defined with Dandelion's paired heavy- and light-chain definition (Suo et al. 2024), which needs
the `sc-dandelion` package: `pip install "threadfin[dandelion]"` (Python 3.10–3.14) or
`pip install "sc-dandelion>=0.5.7,<1"`. Threadfin is validated with Dandelion 0.5.x; Dandelion 1.0 calls clones
differently. Dandelion 0.5.7 imports a matplotlib function that matplotlib 3.9 removed (`matplotlib.cm.get_cmap`);
importing Threadfin restores it, so if you use Dandelion yourself with matplotlib 3.9 or later, import Threadfin
first. If you prefer,
define clones in a separate environment or with another clone caller and pass the clone ids directly
(section 6).

*Reproducible clone ids.* Dandelion's clone assignment depends on the order in which Python iterates over
sets of sequences, which is randomised in every new Python session: on the dataset of Tutorial 5, sessions with
different seeds moved the cells of about one expanded clone in eight and changed nearly every clone id.
Threadfin therefore calls clones under Python's hash seed 0 in every session: when Python was not started with
`PYTHONHASHSEED=0`, Dandelion runs in a separate Python process that is (about 2 s more per call). The ids are
identical run after run, in scripts and notebooks alike, and the same as in these tutorials and the manuscript;
you do not need to set anything. The seed is stored with the clone ids (`ids.attrs["dandelion_hash_seed"]`;
`ids.attrs["dandelion_subprocess"]` says whether a separate process was used; after `tf.run(bcr=...)` both are in
`result.settings["clone_calling"]`), and `result.methods()` states it. `hash_seed=None`
(`clone_options={"hash_seed": None}` in `tf.run`) calls clones in your own session with its own seed instead, as
Threadfin 4.5.3 and earlier did.

---

## 2. Quality control and the right cells

**What.** Remove empty droplets, damaged cells and doublets, and keep the B-lineage cells.

**Why.** Damaged cells (few genes, many mitochondrial reads) and doublets (two cells in one droplet) create
artificial "states". A doublet of a germinal-centre B cell and a plasma cell would even look like a clone
spanning both states.

```python
import scanpy as sc

adata = sc.read_10x_h5("sample1/filtered_feature_bc_matrix.h5")
# or, for a folder with matrix.mtx.gz, features.tsv.gz and barcodes.tsv.gz:
# adata = sc.read_10x_mtx("sample1/filtered_feature_bc_matrix")
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
  ASCs have many immunoglobulin transcripts and sometimes fewer detected genes — do not set the gene
  threshold so high that they disappear.
* *Doublets*: if your samples were hashtagged, remove cells positive for two hashtags (Tutorial 5, Step 3);
  otherwise a doublet detector such as Scrublet (`sc.pp.scrublet`) helps.
* *B-lineage cells*: keep cells annotated as B cells, germinal-centre B cells, plasmablasts, plasma cells
  and memory B cells. Marker genes that help: *CD19*, *MS4A1*, *CD79A* (B cells), *BCL6*, *AICDA* (germinal
  centre), *PRDM1*, *XBP1*, *JCHAIN*, *SDC1* (ASCs). ASCs have little *MS4A1*/CD20, so do not select B cells
  on *MS4A1* alone.
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
two or more cells, the clones that are profiled. If almost no cell has a BCR, the barcodes did not match
(section 4).

---

## 4. Several samples: make every barcode unique

**What.** Load each sample, prefix every barcode with its sample name in *both* the expression and the BCR
table, and concatenate.

**Why.** 10x barcodes are drawn from a fixed whitelist, so the same barcode occurs in different libraries.
Without a prefix, cells of different samples would be merged, and BCRs would be attached to the wrong cells.

```python
import anndata as ad
import pandas as pd
import scanpy as sc
import threadfin as tf

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
# apply the quality control and the selection of B-lineage cells of section 2 to adata here

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

<a id="5-choosing-donor_key-sample_key-and-batch_key"></a>

## 5. Choosing `donor_key`, `sample_key`, `strata_key` and `batch_key`

These four arguments decide what Threadfin treats as *nuisance* — differences that should not count as
differences between clones.

| Argument | What it does | Use | Never use |
|---|---|---|---|
| `donor_key` | clones are defined within individuals; tests of your labels shuffle them among clones of one individual | the individual | — |
| `sample_key` | each clone is described relative to the cells of its context; by default, coherence shuffles clone labels within it | the donor, a library, or donor × time point when time is *not* what you study | anything you want to compare between clones: tissue, sort gate, treatment, time point |
| `strata_key` | the coherence test shuffles clone labels within it (default: `sample_key`) | the finest sampling unit — the library, sort gate or tissue sample — whenever `sample_key` is coarser | a column coarser than the libraries, such as the donor when each donor has several libraries |
| `batch_key` | Harmony integration when the embedding is built | technical batches (libraries, runs) | biology |

**Why the context must be a nuisance variable.** Threadfin subtracts each context's average before
comparing clones. If the context were the tissue, every clone would be compared only with the cells of its
own tissue, and differences between tissues — the thing you wanted to see — would be removed. Two examples:

* *Lymph node and blood, several visits per donor.* Use `sample_key="donor"` if you want to see lymph-node
  versus blood clones, name the libraries as the strata of the coherence test, and test `tissue` afterwards:

  ```python
  result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key="donor", strata_key="sample",
                  test=["tissue"])
  ```
* *Cells sorted into gates (for example GC B cells versus ASCs, or a division reporter), each gate a library.*
  Use the donor as the context and the library as the strata (`sample_key="donor", strata_key="sample"`), and
  keep in mind that a gate sorted into its own library cannot be separated from differences between libraries
  (Tutorial 5, Step 13).

Such a tissue or gate label is then fixed within each sample. `test=["tissue"]` still tests it, and the summary
says that the comparison is between samples (and how many), not between independent clones, because each
sample is also a library.

**Why the coherence test shuffles within sampling units.** Cells of one clone are usually sequenced together. If
samples differ in their cell-type composition, any analysis that ignores samples would call that clonal.
Shuffling clone labels within each sampling unit keeps every unit's composition and destroys only the link
between a cell's clone and its state. The strata must therefore be the sampling units themselves: with
`sample_key="donor"` and no `strata_key`, clone labels would be shuffled across a donor's libraries, and the
differences between those libraries (tissues, gates, visits) would count as clonal. If a library pools several
donors (hashtag multiplexing), Threadfin shuffles within each donor × library combination and warns. The summary
and panel b of the overview name the strata used.

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

**Cells with two chains of one kind.** In droplet data about 4–25 % of cells carry two productive light chains
or, less often, two heavy chains: allelic inclusion, ambient RNA or doublets. Dandelion gives such a cell a clone
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
`obsm["airr"]`) can be passed directly. scirpy is not installed with Threadfin (`pip install scirpy`). With
several samples, give the receptor data the same barcode prefixes as the expression data (section 4):

```python
import anndata as ad
import scirpy as ir

parts = []
for sample in samples:                                              # the samples of section 4
    a = ir.io.read_10x_vdj(f"{sample}/filtered_contig_annotations.csv")   # AnnData with obsm["airr"]
    a.obs_names = [f"{sample}_{b}" for b in a.obs_names]           # the prefix of the expression data
    parts.append(a)
airr = ad.concat(parts)
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

A cell without a clone must have a missing value (NaN) as its clone id, not a text such as `"nan"`, `"None"` or
`"donor1|nan"`: each such id would join all of its cells into one large clone, and Threadfin warns when it sees
them.
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
example, comes from the constant gene of each cell's productive heavy-chain contig with the most UMIs (Cell
Ranger's `c_gene`, AIRR `c_call`: IGHM, IGHA1, IGHG2B …); the first four letters give the class.
`tf.read_contigs` gives the table AIRR column names, whether or not you ran the snippet of section 6:

```python
airr = tf.read_contigs(contigs)
chain = airr["locus"] if "locus" in airr.columns else airr["v_call"].astype(str).str[:3]   # IGH, IGK, IGL
productive = airr["productive"].astype(str).str.upper().isin(["T", "TRUE"])
umis = next(c for c in ("umi_count", "duplicate_count") if c in airr.columns)              # UMIs per contig
heavy = (airr[chain.eq("IGH") & productive]
         .sort_values(umis, ascending=False)                  # each cell's heavy chain with most UMIs first
         .drop_duplicates("cell_id").set_index("cell_id"))
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
appear clonal, and among clones of similar size (2, 3–4, 5–9 or ≥ 10 cells): small clones' profiles are pulled
towards the average profile, so a label that follows only clone size would otherwise seem to separate clones
(`size_strata=False` shuffles within donors only).

* A **categorical label** (isotype) is summarised per clone as its majority value; clones without a majority
  level (a two-cell clone with one IgM and one IgG cell) are left out, and the summary says how many. Use
  `tf.tl.profile_association(adata, "isotype", how="fraction:IGHA")` to test the fraction of one level (the
  level must be spelled as in the column; Threadfin warns when it never occurs).
* A **numeric label** (mutation frequency) is summarised as the clone mean.
* Labels **constant within a donor** (age, sex, disease group) cannot be separated from other differences
  between donors; compare donors for those.
* Labels derived from the same **RNA** (a cell-type annotation) are not independent of the clone profiles;
  they describe programmes but do not validate them. The summary marks the test of the `state_key` annotation
  as a positive control.
* **Programme composition.** With `state_key`, each programme is described by the share of its cells in each
  state (`composition_weight="cells"`, the default), which suits questions about output and abundance.
  `composition_weight="clones"` counts every clone once and describes a typical clone, which suits naming
  programmes. In sorted designs neither equals the composition in the animal, because the sort decided how
  many cells of each state were sequenced.
* Results of `tf.tl.profile_association` and `tf.tl.association_test` are stored in `adata.uns["threadfin"]`
  under the label's name, so a later call with the same label (for example the `how="fraction:IGHA"` test
  above) replaces the result that `tf.run` stored there; `result.profile_tests` and `result.tests` keep the
  results of the run.

* A label that is **fixed within each sample** (a tissue, sort gate or visit sequenced in its own library) is
  compared between samples, not between independent clones: the summary says so and gives the number of
  samples, and `result.profile_tests[label]["sample_level"]` is `True`.

**Reading a test result.** `r2` is the share of variation between clone profiles explained by the label;
`null_mean` is what shuffled labels explain; `excess_r2` is their difference; `p_value` is the permutation
P value (with 2,000 permutations it cannot be smaller than 0.0005). Small R² values are normal: a label is
one coarse measurement, the profile describes the whole transcriptome.

---

## 8. Saving

```python
from pathlib import Path

result.clones.to_csv("threadfin_clones.csv")       # one row per clone with >= 2 cells
Path("threadfin_summary.txt").write_text(result.summary())
Path("threadfin_methods.txt").write_text(result.methods())   # a Methods paragraph with the settings and versions used
adata.write_h5ad("threadfin_result.h5ad")          # the AnnData, with the results of every step in adata.uns["threadfin"]
```

`result.methods()` describes what was run — clone calling, embedding, profiles, tests — with the settings and
package versions of this analysis and the references it relies on, as a starting point for the Methods section
of a paper. Check it against your own description of the data. The h5ad file also keeps the record of the run
(settings, analyses not run and why, package versions), so the result can be rebuilt later:

```python
import anndata

result = tf.ThreadfinResult.from_adata(anndata.read_h5ad("threadfin_result.h5ad"))
print(result.summary())                            # the same summary as before saving
```

---

## 9. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `No barcode of the receptor data matches adata.obs_names` | the barcode formats differ (prefix, `-1` suffix) | give the `barcode` (or AIRR `cell_id`) column of the contig table the same prefix or suffix as `adata.obs_names`, as in section 4, and pass the table; for scirpy objects, prefix the `obs_names` of each sample's receptor data (section 6) |
| `A per-cell BCR table cannot be used for clone definition` | a one-row-per-cell table was passed, for example the output of `tf.read_bcr` or `tf.read_10x_vdj` (columns ending in `_light`, or `n_contigs`) | pass the contig-level table (`filtered_contig_annotations.csv` or AIRR) |
| `Only N clone(s) have two or more cells` | fewer than 3 clones have two or more cells: too few cells per clone were captured, or BCRs were not attached | check the match rate (section 4); check that clones were called within donors; a dataset with very few repeated clones cannot be profiled |
| `No programmes: only N clones have two or more cells (need 30)` (in the summary) | programmes need at least 30 profiled clones | the coherence test and the label tests still apply |
| `N clone id(s) occur in more than one donor` | clone ids were not made donor-private | make them donor-private with the line the message shows (it keeps cells without a clone id missing), or call clones with `donor_key` |
| `N cells have a clone id that looks like a missing value` | cells without a clone carry a text id such as `"nan"`, `"None"` or `"donor1\|nan"`, and each such id becomes one large clone | set those ids to missing (NaN) in `adata.obs["clone_id"]`; when you build ids by joining text, end with `.where(adata.obs["clone_id"].notna())` |
| `Counts contain NaN, infinite, or negative values` | `adata.X` holds normalised or scaled data | put raw counts in `X`, or name the layer that holds them with `counts_layer=` (Threadfin 4.5.2 or later), or pass your own embedding with `basis=` |
| `adata.X does not look like raw counts (it has non-integer values)` | `adata.X` holds normalised values | as above; ignore it only when the counts are corrected but not normalised (for example after ambient-RNA removal) |
| `using the existing obsm['X_threadfin']` (in the log) | an embedding from an earlier run on the same AnnData is reused | expected when nothing changed; Threadfin builds it again by itself when `counts_layer` or `batch_key` differ from those it was built with (the log says so), and `del adata.obsm["X_threadfin"]` forces a new one |
| `'X_...' has NaN or infinite values in N of M cells` | your own embedding (`basis=`) has missing values | drop those cells first; the message shows the line of code (it needs `import numpy as np`) |
| `'donor' not found in adata.obs` (or any other column) | a column name is misspelled or missing | the message names the column; check `adata.obs.columns` |
| `Clone calling needs the Dandelion package` | Dandelion is not installed | `pip install "threadfin[dandelion]"`, or pass clone ids yourself (section 6) |
| `sc-dandelion 1.x is installed; Threadfin … is validated with sc-dandelion 0.5.x` | Dandelion 1.0 calls clones differently | `pip install "sc-dandelion<1"`; results then match the documentation |
| `cannot import name 'get_cmap' from 'matplotlib.cm'` | `import dandelion` before `import threadfin`, with matplotlib ≥ 3.9 | import threadfin first, or let Threadfin call Dandelion |
| `'csr_array' object has no attribute 'getnnz'` from `sc.pp.neighbors` | pynndescent 0.7 (October 2026) returns a sparse type that scanpy's neighbour search (for 8,192 or more cells) cannot yet read | `pip install "pynndescent<0.7"`; Threadfin 4.5.3 and later install it |
| `No resolution reached a clone-weighted mean stability of 0.75` | the programmes are, on average, less stable than the threshold | judge each programme by its own stability (`result.programmes`); the summary marks those below 0.6 as unstable, also on the lines of the label tests |
| `The levels of 'tissue' (…) have no order, so they are paired in alphabetical order` | `time_key` names a column whose levels carry no numbers and no order | give the order: `adata.obs["tissue"] = pd.Categorical(adata.obs["tissue"], categories=["PP", "MLN", "LP"], ordered=True)` before `tf.run`, or `tf.tl.clonal_memory(adata, "tissue", order=["PP", "MLN", "LP"])` |
| coherence high but the shuffled value is also high (> 10 %) | samples differ strongly in composition | expected; Threadfin accounts for it — make sure the strata are the sampling units (`strata_key`, by default `sample_key`; section 5) |
| a programme is dominated by one donor | donors differ in their sampled composition, and centring shifts all clones of a donor together | look at programmes by donor; compare with `sample_key=None, strata_key="sample"` (no centring; the coherence test still shuffles within samples); see Methods, "centring on the sampling context" |

Next: [Tutorial 3 — the clone map](03_clone_map.md).
