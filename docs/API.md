# API reference (threadfin 4.4.1)

Generated from the docstrings by `python docs/make_api_reference.py`; do not edit by hand. Start with [`tf.run`](#tfrun): it runs every step below with the recommended settings. The tutorials in [`tutorials/`](tutorials/) show each function on real data.

- [One call](#one-call)
- [Reading data and calling clones](#reading-data-and-calling-clones)
- [Preprocessing: tf.pp](#preprocessing-tfpp)
- [Analysis steps: tf.tl](#analysis-steps-tftl)
- [Figures: tf.pl](#figures-tfpl)
- [Simulated data: tf.sim](#simulated-data-tfsim)
- [Earlier (v3) functions](#earlier-v3-functions)

## One call

The complete analysis and its result.

| function | what it does |
|---|---|
| [`tf.run`](#tfrun) | Run the complete Threadfin analysis in one call. |
| [`tf.ThreadfinResult`](#tfthreadfinresult) | Everything :func:`run` computed. Print :meth:`summary` for a readable report. |
| [`ThreadfinResult.summary`](#threadfinresultsummary) | Plain-language summary of the results. |
| [`ThreadfinResult.plot`](#threadfinresultplot) | Overview figure (clone map, coherence test, programme composition, stability). |

### tf.run

```python
tf.run(adata, bcr=None, *, donor_key: 'str | None' = None, sample_key: 'str | None' = None,
    batch_key: 'str | None' = None, state_key: 'str | None' = None, time_key: 'str | None' = None,
    test=None, clone_key: 'str' = 'clone_id', basis: 'str | None' = None, representation: 'str' =
    'kernel', map_n_neighbors: 'int | str' = 'auto', map_min_dist: 'float | str' = 'auto', n_perm:
    'int' = 200, n_boot: 'int' = 30, random_state: 'int' = 0, n_jobs: 'int | None' = 1, verbose:
    'bool' = True) -> 'ThreadfinResult'
```

```text
Run the complete Threadfin analysis in one call.

Parameters
----------
adata
    AnnData of B cells (one row per cell). Needs raw counts in ``X`` (or
    ``layers['counts']``) unless you pass a ready-made ``basis``.
bcr
    BCR data: a path to a Cell Ranger ``filtered_contig_annotations.csv``
    or an AIRR ``.tsv``, or a contig-level DataFrame (one row per receptor
    chain). Clones are defined with Dandelion's paired heavy + light
    definition (:func:`threadfin.define_clones_dandelion`), within each
    donor; this needs the sc-dandelion package. Leave ``None`` if
    ``adata.obs[clone_key]`` already holds clone ids.
donor_key
    ``obs`` column naming the individual each cell came from. Strongly
    recommended: clones are defined within donors and tests are
    stratified by donor.
sample_key
    ``obs`` column naming the sample/library of each cell. Clones are
    compared with the other cells of their own sample, and the null
    model shuffles clone labels only within samples.
batch_key
    Optional ``obs`` column to integrate over with Harmony when the
    embedding is built (technical batches only).
state_key
    Optional cell-type/state annotation, used to describe programmes.
time_key
    Optional time point (or tissue) column; enables the clonal-memory test.
test
    Optional list of ``obs`` columns to test against programmes at clone
    level (e.g. ``["isotype", "antigen_binding"]``).
clone_key
    Name of the clone-id column (created when ``bcr`` is given).
basis
    Existing ``obsm`` embedding to use instead of building one. It
    should exclude immunoglobulin genes (see
    :func:`threadfin.pp.ig_gene_mask`).
representation
    How each clone is summarised for grouping into programmes:
    ``"kernel"`` (default; the whole distribution of the clone's cells,
    so two clones with the same average but different spreads differ)
    or ``"mean"`` (the clone centroid only). The coherence test always
    uses clone means.
map_n_neighbors, map_min_dist
    UMAP settings of the clone map. ``"auto"`` (default) chooses them
    for this dataset by optimisation (:func:`threadfin.tl.tune_clone_map`),
    so that clones that vary continuously are drawn as one connected
    picture; pass numbers to set them by hand. The map is a display only:
    no statistic depends on it.
n_perm, n_boot
    Permutations for the coherence test; bootstrap replicates for
    programme stability.
n_jobs
    Processes for the two most time-consuming steps, the automatic choice
    of the clone-map settings and the bootstrap of programme stability
    (``-1``: all available cores). Results are identical for any value;
    only the run time changes.

Returns
-------
:class:`ThreadfinResult`. Also adds ``obs[clone_key]`` (if ``bcr`` is
given), ``obs['clone_programme']`` and results under
``adata.uns['threadfin']``.

Examples
--------
>>> import threadfin as tf
>>> result = tf.run(adata, bcr="filtered_contig_annotations.csv",
...                 donor_key="donor", sample_key="sample")
>>> print(result.summary())
>>> result.plot("threadfin_overview.pdf")
```

### tf.ThreadfinResult

```python
tf.ThreadfinResult(adata: 'object', coherence: 'dict', programmes: 'pd.DataFrame', clones:
    'pd.DataFrame', composition: 'pd.DataFrame | None' = None, tests: 'dict' = <factory>,
    profile_tests: 'dict' = <factory>, memory: 'dict | None' = None, settings: 'dict' = <factory>,
    skipped: 'dict' = <factory>) -> None
```

```text
Everything :func:`run` computed. Print :meth:`summary` for a readable report.
```

### ThreadfinResult.summary

```python
summary(self) -> 'str'
```

```text
Plain-language summary of the results.
```

### ThreadfinResult.plot

```python
plot(self, save: 'str | Path | None' = None)
```

```text
Overview figure (clone map, coherence test, programme composition, stability).
```

## Reading data and calling clones

Receptor tables in, one clone id per cell out.

| function | what it does |
|---|---|
| [`tf.read_bcr`](#tfread_bcr) | Read a BCR file into one row per cell. |
| [`tf.read_10x_vdj`](#tfread_10x_vdj) | Read a 10x Cell Ranger ``filtered_contig_annotations.csv`` file. |
| [`tf.read_airr`](#tfread_airr) | Read an AIRR rearrangement table (IgBlast fmt19 / Change-O TSV). |
| [`tf.attach_bcr`](#tfattach_bcr) | Attach per-cell BCR annotations to ``adata.obs`` (vectorized). |
| [`tf.read_contigs`](#tfread_contigs) | Contig-level AIRR table from a path or a DataFrame, for :func:`define_clones_dandelion`. |
| [`tf.define_clones_dandelion`](#tfdefine_clones_dandelion) | Call clones with Dandelion and return one clone id per cell. |
| [`tf.define_clones`](#tfdefine_clones) | Group cells into B-cell clones (lineages) from their receptor sequences. |

### tf.read_bcr

```python
tf.read_bcr(path, *, barcode_prefix: 'str' = '', barcode_suffix: 'str' = '') -> 'pd.DataFrame'
```

```text
Read a BCR file into one row per cell.

Accepts the two common formats and detects which one by extension:

* Cell Ranger ``filtered_contig_annotations.csv`` (``.csv`` / ``.csv.gz``);
* AIRR rearrangement table (``.tsv`` / ``.tsv.gz``), e.g. Cell Ranger's
  ``airr_rearrangement.tsv`` or an Immcantation ``*_db-pass.tsv``.

The table is indexed by cell barcode (with optional prefix/suffix, to
match ``adata.obs_names`` of multi-sample objects) and contains the
heavy-chain V/D/J/C calls, junction or CDR3 sequences, light-chain
calls and, for AIRR input with alignments, ``mutation_frequency``.
```

### tf.read_10x_vdj

```python
tf.read_10x_vdj(path: 'str', productive_only: 'bool' = True, high_confidence_only: 'bool' = True) ->
    'pd.DataFrame'
```

```text
Read a 10x Cell Ranger ``filtered_contig_annotations.csv`` file.

One cell can carry multiple contigs (heavy + light chain). The contig with
the highest UMI count is kept per chain class.

Parameters
----------
path
    Path to the CSV (optionally gzipped).
productive_only
    Keep only productive contigs.
high_confidence_only
    Keep only high-confidence contigs.

Returns
-------
DataFrame indexed by cell barcode with columns
``v_call, d_call, j_call, c_call, cdr3, cdr3_nt, umis, clonotype_id,
v_call_light, cdr3_light, n_contigs``.
```

### tf.read_airr

```python
tf.read_airr(path: 'str', cell_col: 'str | None' = None) -> 'pd.DataFrame'
```

```text
Read an AIRR rearrangement table (IgBlast fmt19 / Change-O TSV).

Parameters
----------
path
    Path to the (optionally gzipped) TSV file.
cell_col
    Column holding the cell barcode. When ``None``, ``cell_id`` is used if
    present, otherwise the trailing ``_contig_<n>`` suffix of
    ``sequence_id`` is stripped to recover the barcode.

Returns
-------
DataFrame indexed by cell barcode with columns
``v_call, d_call, j_call, cdr3`` (whichever are present).
```

### tf.attach_bcr

```python
tf.attach_bcr(adata, bcr_table: 'pd.DataFrame', clone_col: 'str' = 'clone_id', prefix: 'str' =
    'bcr_', summarize: 'bool' = True, verbose: 'bool' = True) -> "'object'"
```

```text
Attach per-cell BCR annotations to ``adata.obs`` (vectorized).

Cells without a BCR get ``NaN``. Every column of ``bcr_table`` is added
to ``obs`` with ``prefix`` (the clone column keeps its own name).
Columns from a previous call with the same prefix are replaced.

Parameters
----------
adata
    :class:`anndata.AnnData` with ``obs_names`` matching the BCR barcodes.
bcr_table
    Per-cell BCR table indexed by barcode, with a ``clone_col`` column
    (see :func:`threadfin.define_clones` or :func:`build_clone_key`).
clone_col
    Column of ``bcr_table`` holding the clone id.
prefix
    Prefix for the added ``obs`` columns.
summarize
    Also store a clone-level summary (modal V/D/J calls and CDR3 per
    clone) in ``adata.uns['threadfin_clones']``; only needed by the legacy
    sequence-blending functions (``cdr3_weight``, ``joint_embedding``).

Returns
-------
The modified ``adata`` (in place and returned).
```

### tf.read_contigs

```python
tf.read_contigs(bcr) -> 'pd.DataFrame'
```

```text
Contig-level AIRR table from a path or a DataFrame, for :func:`define_clones_dandelion`.

Accepts a Cell Ranger ``filtered_contig_annotations.csv`` (``.csv`` / ``.csv.gz``), an AIRR
rearrangement ``.tsv`` / ``.tsv.gz``, or an already contig-level DataFrame (one row per chain,
with ``cell_id``/``barcode``, ``v_call``/``v_gene``, ``j_call``/``j_gene`` and
``junction``/``cdr3_nt``). A per-cell table (one row per cell, e.g. the output of
:func:`threadfin.read_bcr`) is refused: the paired heavy + light clone definition needs the
individual chains.
```

### tf.define_clones_dandelion

```python
tf.define_clones_dandelion(contigs: 'pd.DataFrame', *, use_light_chain: 'bool' = True, identity:
    'float' = 0.85, donor_key: 'str | None' = None, cell_key: 'str' = 'cell_id', productive_only:
    'bool' = True, require_heavy_chain: 'bool' = True, out_col: 'str' = 'clone_id', verbose: 'bool'
    = True) -> 'pd.Series'
```

```text
Call clones with Dandelion and return one clone id per cell.

Parameters
----------
contigs
    **Contig-level** AIRR table — one row per receptor chain, not per cell — with at least
    ``cell_id``, ``v_call``, ``j_call`` and ``junction``. This is what Cell Ranger's
    ``airr_rearrangement.tsv`` and ``filtered_contig_annotations.csv`` provide.
use_light_chain
    ``True`` (default) defines a clone on the heavy **and** light chain, which is Dandelion's own
    definition. ``False`` keeps only IGH contigs, for deposits with no light chain; the choice is
    recorded in ``Series.attrs['use_light_chain']``.
identity
    Junction identity threshold passed to ``dandelion.tl.find_clones``.
donor_key
    Column naming the individual. When given, clones are called separately within each donor, so no
    clone can span two animals or participants; the donor is prefixed to the id.
productive_only
    Drop non-productive rearrangements before calling.
require_heavy_chain
    ``True`` (default) returns clone ids only for cells with a (productive) heavy chain. Dandelion
    also groups cells whose heavy chain was not recovered, by their light chain alone (ids such as
    ``B_VJ_...``). Light-chain rearrangements are far less diverse than heavy-chain ones, so these
    groups join unrelated cells and are not clones; in GSE287123 they were 5,276 of 21,939 cells and
    740 of 2,626 groups of two or more cells. ``False`` keeps them.

Returns
-------
Series indexed by cell id. ``attrs`` records the settings, the resulting clone counts and the
``PYTHONHASHSEED`` in use (``None`` when unset; clone calls are then not bit-reproducible between Python
sessions, see the module notes).
```

### tf.define_clones

```python
tf.define_clones(bcr: 'pd.DataFrame', *, donor_key: 'str | None' = 'donor', v_col: 'str' = 'v_call',
    j_col: 'str' = 'j_call', junction_col: 'str | None' = None, threshold: 'float | str' = 'auto',
    light_chain: 'bool' = False, light_v_col: 'str' = 'v_call_light', light_j_col: 'str' =
    'j_call_light', light_junction_col: 'str | None' = None, out_col: 'str' = 'clone_id', verbose:
    'bool' = True) -> 'pd.DataFrame'
```

```text
Group cells into B-cell clones (lineages) from their receptor sequences.

A clone is the family of cells descended from one V(D)J recombination.
B cells hypermutate their receptor, so relatives are joined when their
junctions are similar enough (single linkage below ``threshold``).

Parameters
----------
bcr
    Per-cell table (one row per cell, e.g. from :func:`threadfin.read_bcr`)
    with V gene, J gene and junction (or CDR3) sequence of the heavy chain.
donor_key
    Column identifying the individual. Clones are never formed across
    donors. ``None`` treats all cells as one donor.
junction_col
    Heavy-chain junction/CDR3 column. Default: the first available of
    ``junction``, ``cdr3_nt`` (nucleotides, preferred) and
    ``junction_aa``, ``cdr3`` (amino acids).
threshold
    Maximum normalised Hamming distance for single linkage, or ``"auto"``
    (density valley, :func:`find_threshold`; defaults 0.15 for
    nucleotides and 0.20 for amino acids if no valley is found).
light_chain
    Split heavy-chain clones whose cells carry different light chains
    (light V gene, light J gene and junction length). Cells without a
    light chain join the clone's most common light-chain group.
out_col
    Name of the new clone-id column. Ids look like ``"<donor>|C00001"``,
    numbered by decreasing size within each donor.

Returns
-------
Copy of ``bcr`` with ``out_col`` (NaN where V, J or junction is
missing). ``.attrs['clone_definition']`` records the threshold, the
sequence type and the number of clones.

.. deprecated::
    This heavy-chain-only caller is superseded by Dandelion's paired
    heavy + light clone definition (:func:`threadfin.define_clones_dandelion`),
    which is Threadfin's clone definition.
```

## Preprocessing: tf.pp

The cell embedding from which clone profiles are built.

| function | what it does |
|---|---|
| [`tf.pp.ig_gene_mask`](#tfppig_gene_mask) | Boolean mask of immunoglobulin (and optionally TCR) genes. |
| [`tf.pp.exclude_ig_genes`](#tfppexclude_ig_genes) | Remove IG (and optionally TR) genes from an existing HVG selection. |
| [`tf.pp.prepare_embedding`](#tfppprepare_embedding) | Standard expression embedding for clone-level analysis. |

### tf.pp.ig_gene_mask

```python
tf.pp.ig_gene_mask(var_names, *, constant: 'bool' = True, tr_genes: 'bool' = False) -> 'np.ndarray'
```

```text
Boolean mask of immunoglobulin (and optionally TCR) genes.

Parameters
----------
var_names
    Gene symbols (``adata.var_names``); human and mouse casing supported.
constant
    Also flag constant-region genes (IGHM, IGHG1, IGKC, ...). Keep
    ``True`` when isotype is used as an external validation label.
tr_genes
    Also flag TCR segment genes (TRAV, TRBV, ...), useful when T cells
    contaminate a B-cell dataset.

Notes
-----
Real genes with IG-like prefixes (e.g. ``IGHMBP2``, ``IGLON5``, ``JCHAIN``)
are *not* flagged.
```

### tf.pp.exclude_ig_genes

```python
tf.pp.exclude_ig_genes(adata, *, hvg_key: 'str' = 'highly_variable', constant: 'bool' = True,
    tr_genes: 'bool' = False, verbose: 'bool' = True)
```

```text
Remove IG (and optionally TR) genes from an existing HVG selection.

Sets ``adata.var[hvg_key]`` to ``False`` for receptor genes and records
the mask in ``adata.var['threadfin_receptor_gene']``. Re-run PCA after
calling this.
```

### tf.pp.prepare_embedding

```python
tf.pp.prepare_embedding(adata, *, counts_layer: 'str | None' = None, batch_key: 'str | None' = None,
    n_top_genes: 'int' = 3000, n_comps: 'int' = 30, exclude_receptor_genes: 'bool' = True,
    integrate: 'str | None' = 'harmony', key_added: 'str' = 'X_threadfin', random_state: 'int' = 0,
    verbose: 'bool' = True)
```

```text
Standard expression embedding for clone-level analysis.

normalize_total (1e4) -> log1p -> HVG (``flavor='seurat'``, batch-aware,
receptor genes excluded) -> scale -> PCA -> optional Harmony on
``batch_key``. The log-normalised matrix is kept in
``adata.layers['log_norm']`` and ``adata.X`` is left untouched.

Parameters
----------
counts_layer
    Layer holding raw counts; ``None`` uses ``adata.X``.
batch_key
    ``obs`` column for batch-aware HVG selection and Harmony
    integration. Integrate over *technical* batches (donor, library),
    not over biology you want to study (tissue, timepoint, sort gate).
integrate
    ``"harmony"`` (requires ``harmonypy``) or ``None``.
key_added
    ``obsm`` key for the final embedding (the unintegrated PCA is also
    stored as ``X_pca``).

Returns
-------
``adata`` (modified in place).
```

## Analysis steps: tf.tl

The steps that `tf.run` calls, and further analyses, one by one.

| function | what it does |
|---|---|
| [`tf.tl.association_test`](#tftlassociation_test) | Clone-level association between programmes and a label. |
| [`tf.tl.clonal_coherence`](#tftlclonal_coherence) | Test whether clone identity explains transcriptional state. |
| [`tf.tl.clonal_memory`](#tftlclonal_memory) | Do clones keep their state between time points (or tissues)? |
| [`tf.tl.clone_densities`](#tftlclone_densities) | AnnData wrapper requiring an explicitly fitted StateDensityModel. |
| [`tf.tl.clone_labels`](#tftlclone_labels) | Collapse a cell-level ``obs`` column to one value per clone. |
| [`tf.tl.clone_profiles`](#tftlclone_profiles) | Estimate a shrunken, context-adjusted state profile for every clone. |
| [`tf.tl.define_clones`](#tftldefine_clones) | Group cells into B-cell clones (lineages) from their receptor sequences. |
| [`tf.tl.define_clones_dandelion`](#tftldefine_clones_dandelion) | Call clones with Dandelion and return one clone id per cell. |
| [`tf.tl.embed_clones`](#tftlembed_clones) | Lay the clone profiles out in two dimensions — **every** profiled clone, by default. |
| [`tf.tl.export_lag`](#tftlexport_lag) | Date each clone's output against its own germinal centre. |
| [`tf.tl.find_programmes`](#tftlfind_programmes) | Group clones into clonal programmes, with bootstrap stability. |
| [`tf.tl.find_threshold`](#tftlfind_threshold) | Threshold at the density valley of a distance-to-nearest distribution. |
| [`tf.tl.gene_heritability`](#tftlgene_heritability) | Clonal ICC of every gene, with a within-stratum permutation p-value. |
| [`tf.tl.geneset_heritability`](#tftlgeneset_heritability) | Summarise :func:`gene_heritability` over named gene sets. |
| [`tf.tl.layout_clone_map`](#tftllayout_clone_map) | Lay the clones out at given settings, at UMAP's default number of epochs, and check the result. |
| [`tf.tl.lineage_forest`](#tftllineage_forest) | Group cells into clone x block units and collect their lineage-tree distances. |
| [`tf.tl.lineage_heritability`](#tftllineage_heritability) | Estimate and test lineage heritability of cell state inside clones. |
| [`tf.tl.map_quality`](#tftlmap_quality) | Score a clone map: coherence, local and global fidelity, and the tuner's combined score. |
| [`tf.tl.mutation_frequency`](#tftlmutation_frequency) | V-region somatic mutation frequency from AIRR alignments. |
| [`tf.tl.profile_association`](#tftlprofile_association) | Does a clone-level label explain how clones differ? (no programmes needed) |
| [`tf.tl.programme_composition`](#tftlprogramme_composition) | Fraction of each programme's cells in each cell state (rows sum to 1). |
| [`tf.tl.programme_markers`](#tftlprogramme_markers) | Genes that distinguish programmes, tested with clones as replicates. |
| [`tf.tl.tune_clone_map`](#tftltune_clone_map) | Choose UMAP ``n_neighbors`` and ``min_dist`` for a clone map. |

### tf.tl.association_test

```python
tf.tl.association_test(adata, label_key: 'str', *, programme_key: 'str' = 'clone_programme',
    clone_key: 'str | None' = None, strata_key: 'str | None' = 'auto', how: 'str' = 'auto',
    min_frac: 'float' = 0.5, n_perm: 'int' = 2000, random_state: 'int' = 0, verbose: 'bool' = True)
    -> 'pd.DataFrame'
```

```text
Clone-level association between programmes and a label.

Each clone contributes once. Cell-level labels are first collapsed per
clone (:func:`clone_labels`). P-values come from permuting clone labels
within strata (default: donor), which keeps every donor's label
frequencies fixed; Benjamini-Hochberg FDR is computed across all rows.

*Categorical labels* (e.g. antigen specificity, isotype, infection
status) give one row per (programme, level) with the observed and
expected number of clones, the Mantel-Haenszel odds ratio across strata
with a 95% confidence interval, and a two-sided permutation p-value.

*Numeric labels* (e.g. mutation frequency, fraction of a clone's cells in
a sort gate) give one row per programme with the median in vs out of the
programme, a rank-based effect (difference in mean ranks scaled to
[-1, 1]) and a two-sided permutation p-value; an omnibus Kruskal-Wallis
test with the same permutation scheme is stored in ``.attrs``.

Parameters
----------
label_key
    ``obs`` column to test.
strata_key
    Permutation strata. ``"auto"`` uses the donor recorded by
    :func:`threadfin.tl.clone_profiles` (if any); ``None`` permutes
    globally.
how, min_frac
    Passed to :func:`clone_labels`.
```

### tf.tl.clonal_coherence

```python
tf.tl.clonal_coherence(adata, *, strata_key: 'str | None' = None, n_perm: 'int' = 200, random_state:
    'int' = 0, verbose: 'bool' = True) -> 'dict'
```

```text
Test whether clone identity explains transcriptional state.

The statistic is the clonal ICC (fraction of embedding variance explained
by clone, see :mod:`threadfin.profiles`). The null distribution comes from
shuffling clone labels among BCR+ cells *within* ``strata_key`` (for
example sample, or donor x tissue x timepoint), which keeps each
stratum's cell-state composition and clone-size distribution intact and
destroys only the link between a cell's clone and its state.

Parameters
----------
strata_key
    ``obs`` column defining permutation strata. Defaults to the
    ``context_key`` used for :func:`threadfin.tl.clone_profiles`; with
    neither, labels are shuffled globally (not recommended).
n_perm
    Number of permutations.

Returns
-------
dict with the observed ``icc``, null mean/sd/95th percentile,
``excess`` (observed - null mean), ``p_value`` and bookkeeping; also stored
in ``adata.uns['threadfin']['coherence']``.
```

### tf.tl.clonal_memory

```python
tf.tl.clonal_memory(adata, time_key: 'str', *, order=None, programme_key: 'str' = 'clone_programme',
    min_cells: 'int' = 1, min_pairs: 'int' = 10, n_null: 'int' = 500, n_boot: 'int' = 500,
    random_state: 'int' = 0, verbose: 'bool' = True) -> 'dict'
```

```text
Do clones keep their state between time points (or tissues)?

Parameters
----------
time_key
    ``obs`` column with the time point (or tissue) of each cell.
order
    Order of the levels of ``time_key``; inferred from numbers in the
    labels (``d7`` < ``d28``) when omitted. Each clone's consecutive
    observed levels are paired.
programme_key
    Programme column of the clone table (for the descriptive transition
    table); skipped when absent.
min_cells
    Minimum cells per snapshot.
min_pairs
    Fewest snapshot pairs for which an index is reported. Below this the
    bootstrap interval collapses and the estimate says nothing, so the
    function raises instead of returning a number.
n_null, n_boot
    Random re-pairings for the p-value; bootstrap replicates for the CI.

Returns
-------
dict with ``memory_index`` (see module docstring), its bootstrap
``memory_index_ci``, ``p_value``, ``n_clones``, ``n_pairs``, the
per-pair table ``pairs`` and, when programmes exist, programme
``persistence`` with ``persistence_null`` and the ``transitions`` /
``transitions_expected`` tables. Stored in
``adata.uns['threadfin']['memory'][time_key]``.
```

### tf.tl.clone_densities

```python
tf.tl.clone_densities(adata, *, reference_model, clone_key='clone_id', basis='X_pca',
    context_key=None, credible_level=0.95, key_added='densities')
```

```text
AnnData wrapper requiring an explicitly fitted StateDensityModel.

Cells without a clone are omitted. Reference and query must share feature
coordinates. Results are separate from existing mean/kernel profiles.
```

### tf.tl.clone_labels

```python
tf.tl.clone_labels(adata, label_key: 'str', *, clone_key: 'str | None' = None, how: 'str' = 'auto',
    min_frac: 'float' = 0.5) -> 'pd.Series'
```

```text
Collapse a cell-level ``obs`` column to one value per clone.

Parameters
----------
how
    ``"majority"``: modal value, kept only if it covers at least
    ``min_frac`` of the clone's labelled cells (else NaN);
    ``"mean"`` / ``"median"``: for numeric columns;
    ``"fraction:<level>"``: fraction of the clone's labelled cells equal
    to ``<level>`` (numeric output);
    ``"auto"``: ``"mean"`` for numeric columns, else ``"majority"``.
```

### tf.tl.clone_profiles

```python
tf.tl.clone_profiles(adata, *, clone_key: 'str' = 'clone_id', basis: 'str' = 'X_pca', context_key:
    'str | None' = None, donor_key: 'str | None' = None, representation: 'str' = 'mean', n_features:
    'int' = 256, n_components: 'int' = 30, bandwidth: 'float | str' = 'median', smooth: 'int | None'
    = None, min_cells: 'int' = 2, random_state: 'int' = 0, verbose: 'bool' = True) -> 'pd.DataFrame'
```

```text
Estimate a shrunken, context-adjusted state profile for every clone.

Parameters
----------
adata
    AnnData with ``obs[clone_key]`` and an expression embedding.
clone_key
    ``obs`` column with clone ids (missing = no BCR). Clone ids must be
    unique across donors (see :func:`threadfin.define_clones`).
basis
    ``obsm`` key of the embedding. Use a batch-integrated embedding built
    without immunoglobulin genes (:func:`threadfin.pp.prepare_embedding`).
context_key
    ``obs`` column of sampling contexts whose mean is removed before
    profiling (e.g. sample, or donor x timepoint). Choose *nuisance*
    strata only: biology you want to compare between clones (tissue,
    sort gate, infection status) must not be a context.
donor_key
    Optional donor column, recorded per clone for stratified tests.
representation
    ``"mean"`` (shrunken centroid) or ``"kernel"`` (shrunken kernel mean
    embedding from ``n_features`` random Fourier features, reduced to
    ``n_components`` principal components). :func:`threadfin.run` uses
    ``"kernel"``; this function keeps ``"mean"`` as its default so that
    code written for earlier versions gives the same result.
bandwidth
    Kernel bandwidth: ``"median"`` (median distance between cells, the
    standard heuristic), ``"local"`` (distance to the 30th nearest
    neighbour; sharper, but needs larger clones) or a number.
smooth
    Average each cell with its ``smooth`` nearest cells before profiling
    (denoises small clones by borrowing strength from their local state).
    Default: 15 for ``"kernel"``, 0 for ``"mean"``. The coherence test and
    clonal memory always use unsmoothed cells.
min_cells
    Clones with fewer cells are not profiled. The default of 2 is **not** a quality filter --
    Threadfin never filters on reliability -- it is what the clone map is *of*. Singleton clones
    are 75 to 98 % of clones in these deposits, so including them turns the clone embedding back
    into the cell UMAP it is supposed to summarise. Set 1 to include them; they are then drawn as
    triangles rather than circles so a reader can see how much of a map rests on a single cell.
    Threadfin does not filter clones. Every clone, and every cell in it, is measured data, and
the profiles are already **shrunk**: a clone seen in few cells has its profile pulled towards the
population mean by the variance-component model, so its influence is down-weighted in proportion to
how little was seen. Cutting on reliability *on top of* that shrinkage discards the same information
twice, and it discards it silently — on GSE287123 a 0.5 floor removes 75 % (RBD) and 90 % (NP-OVA) of
profiled clones, and on GSE246382 it removes all of them, which is how that dataset came to look as
though it had no clone structure when its map separates the sorted gate at p < 0.001.

Returns
-------
Clone table (one row per profiled clone) with ``n_cells``,
``reliability`` and, when available, ``donor`` and ``n_contexts``.
Profiles and model parameters go to ``adata.uns['threadfin']['profiles']``.
```

### tf.tl.define_clones

```python
tf.tl.define_clones(bcr: 'pd.DataFrame', *, donor_key: 'str | None' = 'donor', v_col: 'str' =
    'v_call', j_col: 'str' = 'j_call', junction_col: 'str | None' = None, threshold: 'float | str' =
    'auto', light_chain: 'bool' = False, light_v_col: 'str' = 'v_call_light', light_j_col: 'str' =
    'j_call_light', light_junction_col: 'str | None' = None, out_col: 'str' = 'clone_id', verbose:
    'bool' = True) -> 'pd.DataFrame'
```

```text
Group cells into B-cell clones (lineages) from their receptor sequences.

A clone is the family of cells descended from one V(D)J recombination.
B cells hypermutate their receptor, so relatives are joined when their
junctions are similar enough (single linkage below ``threshold``).

Parameters
----------
bcr
    Per-cell table (one row per cell, e.g. from :func:`threadfin.read_bcr`)
    with V gene, J gene and junction (or CDR3) sequence of the heavy chain.
donor_key
    Column identifying the individual. Clones are never formed across
    donors. ``None`` treats all cells as one donor.
junction_col
    Heavy-chain junction/CDR3 column. Default: the first available of
    ``junction``, ``cdr3_nt`` (nucleotides, preferred) and
    ``junction_aa``, ``cdr3`` (amino acids).
threshold
    Maximum normalised Hamming distance for single linkage, or ``"auto"``
    (density valley, :func:`find_threshold`; defaults 0.15 for
    nucleotides and 0.20 for amino acids if no valley is found).
light_chain
    Split heavy-chain clones whose cells carry different light chains
    (light V gene, light J gene and junction length). Cells without a
    light chain join the clone's most common light-chain group.
out_col
    Name of the new clone-id column. Ids look like ``"<donor>|C00001"``,
    numbered by decreasing size within each donor.

Returns
-------
Copy of ``bcr`` with ``out_col`` (NaN where V, J or junction is
missing). ``.attrs['clone_definition']`` records the threshold, the
sequence type and the number of clones.

.. deprecated::
    This heavy-chain-only caller is superseded by Dandelion's paired
    heavy + light clone definition (:func:`threadfin.define_clones_dandelion`),
    which is Threadfin's clone definition.
```

### tf.tl.define_clones_dandelion

```python
tf.tl.define_clones_dandelion(contigs: 'pd.DataFrame', *, use_light_chain: 'bool' = True, identity:
    'float' = 0.85, donor_key: 'str | None' = None, cell_key: 'str' = 'cell_id', productive_only:
    'bool' = True, require_heavy_chain: 'bool' = True, out_col: 'str' = 'clone_id', verbose: 'bool'
    = True) -> 'pd.Series'
```

```text
Call clones with Dandelion and return one clone id per cell.

Parameters
----------
contigs
    **Contig-level** AIRR table — one row per receptor chain, not per cell — with at least
    ``cell_id``, ``v_call``, ``j_call`` and ``junction``. This is what Cell Ranger's
    ``airr_rearrangement.tsv`` and ``filtered_contig_annotations.csv`` provide.
use_light_chain
    ``True`` (default) defines a clone on the heavy **and** light chain, which is Dandelion's own
    definition. ``False`` keeps only IGH contigs, for deposits with no light chain; the choice is
    recorded in ``Series.attrs['use_light_chain']``.
identity
    Junction identity threshold passed to ``dandelion.tl.find_clones``.
donor_key
    Column naming the individual. When given, clones are called separately within each donor, so no
    clone can span two animals or participants; the donor is prefixed to the id.
productive_only
    Drop non-productive rearrangements before calling.
require_heavy_chain
    ``True`` (default) returns clone ids only for cells with a (productive) heavy chain. Dandelion
    also groups cells whose heavy chain was not recovered, by their light chain alone (ids such as
    ``B_VJ_...``). Light-chain rearrangements are far less diverse than heavy-chain ones, so these
    groups join unrelated cells and are not clones; in GSE287123 they were 5,276 of 21,939 cells and
    740 of 2,626 groups of two or more cells. ``False`` keeps them.

Returns
-------
Series indexed by cell id. ``attrs`` records the settings, the resulting clone counts and the
``PYTHONHASHSEED`` in use (``None`` when unset; clone calls are then not bit-reproducible between Python
sessions, see the module notes).
```

### tf.tl.embed_clones

```python
tf.tl.embed_clones(adata, *, n_neighbors: 'int | str' = 'auto', min_dist: 'float | str' = 'auto',
    spread: 'float' = 1.0, min_reliability: 'float' = 0.0, coherence_target: 'float | str' = 'auto',
    tune_kwargs: 'dict | None' = None, random_state: 'int' = 0, n_jobs: 'int | None' = 1, verbose:
    'bool' = True) -> 'pd.DataFrame'
```

```text
Lay the clone profiles out in two dimensions — **every** profiled clone, by default.

This is the clone map the rest of the package is built around, and it is deliberately separate from
:func:`find_programmes`. Programme discovery is a clustering problem and needs reasonably estimated
profiles, so it keeps its ``min_reliability`` gate. The map is a *display* of the clones that exist,
and filtering it by reliability throws away the data the map is meant to show: on GSE287123 the 0.5
gate removes 75 % (RBD) and 90 % (NP-OVA) of profiled clones, and on GSE246382 it removes all of
them, which is why that dataset appeared to have no clone map at all.

The map does not change any statistic: profiles, coherence, programmes and tests are computed in
profile space, never on the map.

Parameters
----------
n_neighbors, min_dist
    UMAP settings. ``"auto"`` (the default for both) chooses them per dataset with
    :func:`threadfin.tl.tune_clone_map`: a ladder of settings, each laid out with several seeds and
    smoothed over its neighbours, finds the most faithful layout that is no more fragmented than the
    clone profiles themselves, so a continuum of clones is drawn as one connected picture. Pass numbers to set them
    by hand (``n_neighbors=50, min_dist=0.8`` reproduces the fixed default of Threadfin 4.2); a
    number for one and ``"auto"`` for the other tunes only the second.
spread
    UMAP ``spread``.
min_reliability
    Reliability floor. ``0.0`` (the default) keeps every clone that has a profile; raise it only to
    reproduce an older filtered layout.
coherence_target, tune_kwargs
    Passed to :func:`threadfin.tl.tune_clone_map` when a setting is ``"auto"``.
random_state
    Seed for the tuner and for the final layout.
n_jobs
    Processes for the layouts of the automatic search (``-1``: all available cores). The chosen
    settings and the map do not depend on it.

Returns
-------
The clone table with ``x`` and ``y`` added. ``adata.uns['threadfin']['profiles']['map_params']``
records the settings, how they were chosen (``"tuning"``, with the full optimisation history), the
quality of the final map (``"quality"``: coherence, local and global fidelity) and how many clones
were laid out.
```

### tf.tl.export_lag

```python
tf.tl.export_lag(adata, *, state_key: 'str', source_states, output_states, clone_key: 'str' =
    'clone_id', mutation_key: 'str' = 'mutation_frequency', within=(), time_key: 'str | None' =
    None, isotype_key: 'str | None' = 'isotype', switched_only: 'bool' = True, min_cells: 'int' = 1)
    -> 'dict'
```

```text
Date each clone's output against its own germinal centre.

Parameters
----------
state_key
    ``obs`` column holding the captured cell state.
source_states, output_states
    Which levels of ``state_key`` are the germinal centre and which are its output, for example
    ``source_states=["GC"]`` and ``output_states=["PB", "LNPC"]``.
within
    ``obs`` columns that must match between the two cells being compared, on top of the clone itself.
    Pass the visit and the tissue here: a clone's germinal-centre cell from one visit should not be
    compared with its plasma cell from another.
time_key
    Optional numeric (or ``d28``-style) column. When given, the source compartment's own maturation
    slope converts each clone's lag into days.
switched_only
    Compare only class-switched cells. Class switching is irreversible, so an IgM cell and an IgG cell
    of one clone are separated by an event that mutation count does not describe.
min_cells
    Cells of the clone required on each side of a comparison.

Returns
-------
dict with

``pairs``
    one row per clone (and stratum): ``source``, ``output``, ``lag``, cell counts, and ``lag_days``
    when ``time_key`` is given.
``n_clones``, ``n_output_earlier``, ``n_output_later``
    how many clones were compared and how the lag fell.
``median_lag``
    median lag in mutation-frequency units, with a clone bootstrap interval.
``maturation_slope``
    mutation per unit time in the source compartment, when ``time_key`` is given.

Examples
--------
>>> res = export_lag(adata, state_key="cell_state", source_states=["GC"],
...                  output_states=["PB"], within=("timepoint",), time_key="timepoint")
>>> res["median_lag"], res["n_output_earlier"], res["n_clones"]
```

### tf.tl.find_programmes

```python
tf.tl.find_programmes(adata, *, resolution: 'float | str' = 'auto', resolutions=(0.2, 0.4, 0.6, 0.8,
    1.0, 1.4, 2.0), n_neighbors: 'int' = 15, min_reliability: 'float' = 0.0, weight_by_reliability:
    'bool' = True, n_boot: 'int' = 30, stability_threshold: 'float' = 0.75, min_programme_size:
    'int' = 5, min_clones: 'int' = 30, test_splits: 'bool' = True, alpha: 'float' = 0.05, n_null:
    'int' = 100, split_null: 'str' = 'total', assign_remaining: 'bool' = True, min_posterior:
    'float' = 0.7, random_state: 'int' = 0, n_jobs: 'int | None' = 1, key_added: 'str' =
    'clone_programme', verbose: 'bool' = True) -> 'pd.DataFrame'
```

```text
Group clones into clonal programmes, with bootstrap stability.

Parameters
----------
adata
    AnnData after :func:`threadfin.tl.clone_profiles`.
resolution
    Leiden resolution, or ``"auto"`` to scan ``resolutions`` and keep the
    finest partition whose size-weighted stability is at least
    ``stability_threshold`` (falling back to the most stable partition).
n_neighbors
    Neighbours per clone in the clone graph.
weight_by_reliability
    Scale each graph edge by ``sqrt(rel_i * rel_j)`` so that every clone is clustered but a
    thinly observed one pulls weakly. This is what makes ``min_reliability=0`` safe: on a
    simulated one-programme continuum of 3,000 clones with a median of one cell each, an
    unweighted unfiltered run invents four programmes at bootstrap stability 0.78-0.86, while the
    weighted run recovers the single true programme.
min_reliability
    Reliability floor for clustering; ``0.0`` (the default) clusters every profiled clone.
    Threadfin does not filter clones. Every clone, and every cell in it, is measured data, and
the profiles are already **shrunk**: a clone seen in few cells has its profile pulled towards the
population mean by the variance-component model, so its influence is down-weighted in proportion to
how little was seen. Cutting on reliability *on top of* that shrinkage discards the same information
twice, and it discards it silently — on GSE287123 a 0.5 floor removes 75 % (RBD) and 90 % (NP-OVA) of
profiled clones, and on GSE246382 it removes all of them, which is how that dataset came to look as
though it had no clone structure when its map separates the sorted gate at p < 0.001.
n_boot
    Bootstrap replicates for stability (0 disables stability, which is
    only allowed with a fixed ``resolution``).
min_programme_size
    Groups with fewer clones are not reported as programmes (their clones
    stay unassigned); resolutions that leave more than 10% of clones in
    such groups are not selected.
min_clones
    Minimum number of reliable clones; with fewer, programme detection is
    refused (report the coherence test only).
test_splits, alpha, n_null
    Merge communities that are not significantly better described as two
    groups than as one Gaussian (see the module docstring); ``alpha`` is
    a nominal hierarchical threshold (not a validated family-wise error guarantee) and ``n_null`` the
    number of Gaussian null samples per test. The node tests are stored
    in ``adata.uns['threadfin']['programmes']['split_tests']``. A single
    programme means "no distinct programmes".
assign_remaining
    Programmes are *defined* on reliable ("core") clones only. With this
    option every other profiled clone is then *assigned* to the most
    likely programme by a Gaussian classifier whose noise term shrinks
    with clone size (see ``_assign_remaining``); assignments with
    posterior >= ``min_posterior`` go to ``<key_added>_assigned``.
n_jobs
    Processes for the Leiden runs on the bootstrap graphs (``-1``: all
    available cores). Programmes and stabilities are identical for any
    value; only the run time changes.
key_added
    ``obs`` column receiving each cell's programme (cells of clones that
    were not clustered get NaN).

Returns
-------
Programme summary: ``n_clones``, ``n_cells``, ``stability`` (mean
bootstrap Jaccard) and ``mean_reliability`` per programme. Clone-level
assignments (with per-clone bootstrap ``confidence``) are added to
``adata.uns['threadfin']['profiles']['clone_table']``.
```

### tf.tl.find_threshold

```python
tf.tl.find_threshold(distances, *, default: 'float' = 0.15, lo: 'float' = 0.02, hi: 'float' = 0.25,
    max_first_mode: 'float' = 0.12) -> 'dict'
```

```text
Threshold at the density valley of a distance-to-nearest distribution.

Clonally related sequences sit close to their nearest relative, unrelated
sequences far away, so the distance-to-nearest distribution is bimodal;
the valley between the modes separates them. The first ("related") mode
must lie below ``max_first_mode`` and the valley inside ``[lo, hi]``;
otherwise (for example in repertoires dominated by unmutated naive B
cells, where no clean related mode exists) ``default`` is returned. The
upper bound follows common practice for nucleotide junction distances
(thresholds of roughly 0.05-0.20; Gupta et al. 2017).
```

### tf.tl.gene_heritability

```python
tf.tl.gene_heritability(adata, *, clone_key: 'str | None' = None, context_key: 'str | None' =
    'auto', strata_key: 'str | None' = None, layer: 'str | None' = 'log_norm', genes=None,
    min_frac_expressed: 'float' = 0.05, exclude_receptor_genes: 'bool' = True, n_perm: 'int' = 100,
    chunk_size: 'int' = 1000, random_state: 'int' = 0, verbose: 'bool' = True) -> 'pd.DataFrame'
```

```text
Clonal ICC of every gene, with a within-stratum permutation p-value.

Parameters
----------
clone_key, context_key
    Default to the values used by :func:`threadfin.tl.clone_profiles`
    (``context_key="auto"``); pass ``None`` to disable context centring.
strata_key
    Permutation strata (defaults to the context).
layer
    Log-normalised expression layer (``None`` uses ``adata.X``).
genes
    Genes to test (default: genes expressed in at least
    ``min_frac_expressed`` of the analysed cells).
exclude_receptor_genes
    Leave immunoglobulin/TCR genes out of the scan. They encode the
    receptor itself, so they are clonal by definition; their median ICC is
    still reported (``uns[...]['receptor_gene_icc']``) as a positive
    control for the statistic.
n_perm
    Permutations (shared across genes, so the cost is ``n_perm`` matrix
    products per chunk of genes).

Returns
-------
DataFrame indexed by gene: ``icc``, ``tau2``, ``sigma2``, ``mean``,
``frac_expressed``, ``null_mean``, ``pvalue``, ``fdr`` and ``excess_icc``
(``icc - null_mean``, the effect size), sorted by ``excess_icc``. In large
datasets nearly every gene passes FDR < 0.05; interpret ``excess_icc``.
Also stored in ``adata.uns['threadfin']['gene_heritability']``.
```

### tf.tl.geneset_heritability

```python
tf.tl.geneset_heritability(table: 'pd.DataFrame', gene_sets: 'dict[str, list[str]]', *, n_bins:
    'int' = 10, seed: 'int' = 0) -> 'pd.DataFrame'
```

```text
Summarise :func:`gene_heritability` over named gene sets.

Highly expressed genes are measured with less noise and therefore have
higher ICCs, so each set is compared with an **expression-matched**
background: other tested genes drawn from the same mean-expression deciles
as the set's genes (up to 20 background genes per set gene). Reports the
set's median ICC, the matched background median, the fraction of set genes
significant at FDR < 0.05 and a Mann-Whitney p-value (set vs matched
background).
```

### tf.tl.layout_clone_map

```python
tf.tl.layout_clone_map(features, *, n_neighbors: 'int', min_dist: 'float', repulsion_strength:
    'float' = 1.0, spread: 'float' = 1.0, coherence_target: 'float | str' = 'auto', tuning:
    'MapTuning | None' = None, max_retries: 'int' = 4, random_state: 'int' = 0)
```

```text
Lay the clones out at given settings, at UMAP's default number of epochs, and check the result.

When the settings came from :func:`tune_clone_map` (``tuning``) and this layout falls short of the
coherence target, up to ``max_retries`` further layouts are tried: two more seeds, then the next
confirmed settings that held together in every layout of the search. The most coherent layout is
returned. With settings given by hand, the layout is returned as it is.

Returns
-------
``(xy, quality, setting, attempts)``: the coordinates, :func:`map_quality` of the returned layout,
the ``(n_neighbors, min_dist, repulsion_strength, seed)`` that produced it, and every layout tried.
```

### tf.tl.lineage_forest

```python
tf.tl.lineage_forest(clones, pairs, *, blocks=None, depth=None, min_cells: 'int' = 2, max_depth:
    'float | None' = None) -> 'LineageForest'
```

```text
Group cells into clone x block units and collect their lineage-tree distances.

Parameters
----------
clones
    Clone label per data row (length ``n_cells``).
pairs
    Long table of within-clone cell pairs with integer row indices ``i``, ``j``, the tree-path
    distance ``d`` (mutations between the two cells' genotypes) and, optionally, ``shared``
    (mutations from the root to their common ancestor). Pairs across clones are ignored.
blocks
    Sampling block per row. Units are clone x block. Default: one block, i.e. units are whole clones.

    **Use the finest separately processed unit — the library or sort gate, not the animal.** A clone x
    gate interaction (one clone behaving differently in two sorted gates of the same animal, which
    physical sorting all but guarantees) is *not* removed by fitting the gate as a fixed effect. On real
    germinal-centre trees that omitted interaction, with no lineage component present at all, produced
    h2 = 0.09-0.16 and rejected 49-88% of the time. Blocking by animal when gates are nested inside
    animals is therefore not a valid test of lineage heritability.
depth
    Root-to-tip depth per row. If omitted it is taken from ``pairs`` columns ``depth_i`` and
    ``depth_j``.
min_cells
    Minimum cells for a unit to be kept (at least 2).
max_depth
    Drop cells whose depth exceeds this (alignment or assignment artefacts).
```

### tf.tl.lineage_heritability

```python
tf.tl.lineage_heritability(y, forest: 'LineageForest', *, covariates=None, depth_effect: 'bool' =
    True, kernel: 'str' = 'exponential', length_scale: 'float' = 2.0, n_perm: 'int' = 999,
    feature_names=None, random_state: 'int' = 0, verbose: 'bool' = True) -> 'LineageHeritability'
```

```text
Estimate and test lineage heritability of cell state inside clones.

Parameters
----------
y
    ``(n_cells,)`` or ``(n_cells, k)`` state features, row-aligned with the data the forest
    indexes (module scores, principal components, or 0/1 recorded labels). Rows outside the
    forest are ignored; rows inside must be finite (use :meth:`LineageForest.subset`).
forest
    Output of :func:`lineage_forest`.
covariates
    Optional ``(n_cells, p)`` cell-level fixed effects (log sequencing depth, quality metrics,
    one-hot sample indicators...). Clone x block intercepts are always included.
depth_effect
    Add the cell's root-to-tip mutation depth as a fixed effect (recommended: it separates "more
    mutated cells differ" from "relatives resemble each other").
kernel, length_scale
    Kinship kernel (``"exponential"``, ``"identical"``, ``"identical_mutated"``, ``"brownian"``)
    and, for the exponential kernel, its length scale in mutations.
n_perm
    Freedman-Lane permutations for the score test (0 to skip).

Returns
-------
:class:`LineageHeritability` with, per feature, the REML ``h2`` and its 95% profile interval,
the boundary likelihood-ratio test, the score statistic and its permutation p-value, and a
combined permutation p-value over all features (sum of standardised scores).
```

### tf.tl.map_quality

```python
tf.tl.map_quality(features, xy, *, coherence_target: 'float | str' = 'auto', penalty: 'float' = 5.0,
    random_state: 'int' = 0) -> 'dict'
```

```text
Score a clone map: coherence, local and global fidelity, and the tuner's combined score.

Parameters
----------
features
    Clone profiles, one row per clone (``adata.uns['threadfin']['profiles']['features']``).
xy
    The map, one row per clone in the same order.
coherence_target, penalty
    As in :func:`tune_clone_map`.

Returns
-------
``dict`` with ``coherence`` (fraction of clones in the largest piece), ``local_fidelity`` (30-NN
preservation), ``global_fidelity`` (Spearman correlation of distances), ``score`` and the
``coherence_target`` used.
```

### tf.tl.mutation_frequency

```python
tf.tl.mutation_frequency(airr: 'pd.DataFrame', *, sequence_col: 'str' = 'sequence_alignment',
    germline_col: 'str' = 'germline_alignment', align: 'str' = 'auto') -> 'pd.Series'
```

```text
V-region somatic mutation frequency from AIRR alignments.

Counts substitutions between the query and its germline over the V
segment (:func:`v_region_windows`), ignoring gaps (``-``, ``.``) and
ambiguous ``N``; returns mismatches / compared positions per row (NaN
when the V region cannot be located).

With ``align="auto"`` (the default) the two strings are first put in a
common coordinate system, because pipelines differ in how much upstream
sequence each carries; without this, cells of such a dataset all look
hypermutated. ``align="none"`` compares them from their starts, as the
AIRR standard specifies. Insertions and deletions are not modelled, so a
clone with an indel in its V region can look more mutated than it is.
The windows used are reported in ``.attrs['windows']``.
```

### tf.tl.profile_association

```python
tf.tl.profile_association(adata, label_key: 'str', *, clone_key: 'str | None' = None, strata_key:
    'str | None' = 'auto', how: 'str' = 'auto', min_frac: 'float' = 0.5, min_reliability: 'float' =
    0.0, n_perm: 'int' = 2000, random_state: 'int' = 0, verbose: 'bool' = True) -> 'dict'
```

```text
Does a clone-level label explain how clones differ? (no programmes needed)

Programme tests (:func:`association_test`) need distinct programmes. When
clones vary along a continuum there are none, but a label (division
history, antigen binding, isotype) can still track where a clone sits.
This test asks how much of the variation between clone profiles is
explained by the label: the share of the profile sum of squares between
label groups (categorical labels, as in PERMANOVA; Anderson 2001) or
captured by a linear trend in the label (numeric labels, as in
distance-based redundancy analysis; McArdle & Anderson 2001). The
p-value comes from permuting labels among clones within strata (donors).

Parameters
----------
label_key, how, min_frac
    Cell-level label and how it is collapsed per clone (see
    :func:`clone_labels`).
strata_key
    Clone-table column whose levels define permutation strata;
    ``"auto"`` uses the donor when available.
min_reliability
    Reliability floor; ``0.0`` (the default) uses every profiled clone.
    Threadfin does not filter clones. Every clone, and every cell in it, is measured data, and
the profiles are already **shrunk**: a clone seen in few cells has its profile pulled towards the
population mean by the variance-component model, so its influence is down-weighted in proportion to
how little was seen. Cutting on reliability *on top of* that shrinkage discards the same information
twice, and it discards it silently — on GSE287123 a 0.5 floor removes 75 % (RBD) and 90 % (NP-OVA) of
profiled clones, and on GSE246382 it removes all of them, which is how that dataset came to look as
though it had no clone structure when its map separates the sorted gate at p < 0.001.

Returns
-------
dict with ``r2`` (fraction of profile variance explained by the label),
the permutation ``null_mean``, ``excess_r2`` (``r2 - null_mean``),
``p_value``, ``n_clones``, ``kind`` and, for every level of a categorical
label (or for the numeric label), Spearman correlations with the first
three principal axes of the profiles (``axis_correlation``). Stored in
``adata.uns['threadfin']['profile_associations'][label_key]``.
```

### tf.tl.programme_composition

```python
tf.tl.programme_composition(adata, state_key: 'str', *, key: 'str | None' = None) -> 'pd.DataFrame'
```

```text
Fraction of each programme's cells in each cell state (rows sum to 1).
```

### tf.tl.programme_markers

```python
tf.tl.programme_markers(adata, *, programme_key: 'str' = 'clone_programme', layer: 'str | None' =
    'log_norm', context_key: 'str | None' = 'auto', min_frac_expressed: 'float' = 0.05, n_top: 'int'
    = 25, verbose: 'bool' = True) -> 'pd.DataFrame'
```

```text
Genes that distinguish programmes, tested with clones as replicates.

Every clone is summarised by the mean context-centred log expression of
its cells; for each programme, clones inside are compared with clones in
other programmes by a Mann-Whitney test per gene (BH FDR). Testing across
clones avoids the pseudoreplication of cell-level marker tests, where one
expanded clone contributes hundreds of correlated cells.

Returns
-------
Tidy DataFrame ``[programme, gene, mean_diff, auc, pvalue, fdr, rank]``
with the top ``n_top`` up-regulated genes per programme.
```

### tf.tl.tune_clone_map

```python
tf.tl.tune_clone_map(features, *, n_neighbors: 'int | str' = 'auto', min_dist: 'float | str' =
    'auto', spread: 'float' = 1.0, method: 'str' = 'ladder', coherence_target: 'float | str' =
    'auto', penalty: 'float' = 5.0, tolerance: 'float' = 0.005, ladder_seeds: 'int' = 2, top: 'int'
    = 4, n_layouts: 'int' = 6, rescue: 'int' = 3, allow_weaker_repulsion: 'bool' = True, smoothing:
    'float' = 0.7, search_epochs: 'int | None' = 200, max_clones: 'int | None' = 3000, full_layouts:
    'int' = 2, adam_options: 'dict | None' = None, random_state: 'int' = 0, n_jobs: 'int | None' =
    1, verbose: 'bool' = True) -> 'MapTuning'
```

```text
Choose UMAP ``n_neighbors`` and ``min_dist`` for a clone map.

Looks for the most faithful layout that keeps at least ``coherence_target`` of the clones in one
connected piece in every layout tried, which makes the choice robust to the randomness of UMAP (see
the module documentation for the score and the search). A setting passed as a number is held fixed
and only the other is tuned.

Parameters
----------
features
    Clone profiles, one row per clone.
n_neighbors, min_dist
    ``"auto"`` to tune the setting, or a number to hold it fixed.
spread
    UMAP ``spread``, held fixed.
method
    ``"ladder"`` (default; a fixed ladder of settings, smoothed and confirmed) or ``"adam"``
    (stochastic-gradient search; options in ``adam_options``).
coherence_target
    Fraction of clones that must lie in one connected piece of the map. ``"auto"`` (default) takes
    it from the profiles: the fraction of clones in the largest component of their mutual
    15-nearest-neighbour graph, capped at 0.99, so the map is not more fragmented than the profiles.
    Pass a number (e.g. 0.99 or 1.0) to demand a single piece regardless.
penalty
    Score lost per unit of coherence below the target. The default (5) makes the target binding:
    one percentage point of coherence outweighs any realistic gain in fidelity.
tolerance
    Confirmed settings whose score is within this distance of the best are treated as equally
    faithful, and the most coherent of them is chosen (seed-to-seed noise is about 0.005).
ladder_seeds, top, n_layouts
    Layouts per ladder setting; how many of the best settings are confirmed; how many layouts each
    confirmed setting is scored on in total.
rescue, allow_weaker_repulsion
    When the winner falls short of the target, how many of the most coherent settings seen are also
    confirmed, and whether weaker repulsion may then be tried.
smoothing
    Width (in ladder steps) of the Gaussian that averages each ladder setting with its neighbours.
search_epochs
    UMAP optimisation epochs for layouts during the search (``None``: UMAP's default).
max_clones, full_layouts
    Above ``max_clones`` clones, the search runs on a random subset of this size, ``n_neighbors`` is
    scaled up to the full data in proportion to the number of clones, and the best settings are then
    confirmed on every clone with ``full_layouts`` layouts each. ``None`` always searches on every
    clone.
adam_options
    For ``method="adam"``: ``n_iter`` (30), ``patience`` (8), ``learning_rate`` (0.25),
    ``perturbation`` (0.35), ``betas`` ((0.8, 0.99)), ``max_warmup`` (6) and ``start`` (default
    ``sqrt(N)`` neighbours between 15 and 100, and ``min_dist`` 0.3).
random_state
    Seed for layout seeds and the clone subset; the result is reproducible for a given seed.
n_jobs
    Processes used to compute layouts in parallel (``-1``: all available cores). Every layout depends
    only on its setting and seed, so the chosen settings are identical for any ``n_jobs``; only the
    time changes.

Returns
-------
:class:`MapTuning` with the chosen settings (including ``repulsion_strength``, 1.0 unless the rescue
needed weaker repulsion), their confirmed scores, every layout scored (``history``) and the confirmed
candidates (``candidates``).
```

## Figures: tf.pl

Every figure takes the AnnData that the analysis wrote to.

| function | what it does |
|---|---|
| [`tf.pl.set_style`](#tfplset_style) | Nature-style defaults: small sans text, hairline axes, editable PDF text. |
| [`tf.pl.category_colors`](#tfplcategory_colors) | Fixed-order colours for categories; beyond eight, grey 'other'. |
| [`tf.pl.clone_map`](#tfplclone_map) | Scatter of clones (one point per clone; size = number of cells). |
| [`tf.pl.coherence`](#tfplcoherence) | Observed clonal ICC next to the within-stratum permutation null. |
| [`tf.pl.programme_composition`](#tfplprogramme_composition) | Stacked bars: fraction of each programme's cells in each cell state. |
| [`tf.pl.stability`](#tfplstability) | Bootstrap stability (Jaccard) of each programme, with the 0.75 / 0.6 guides. |
| [`tf.pl.association`](#tfplassociation) | Forest plot of clone-level programme effects from :func:`threadfin.tl.association_test`. |
| [`tf.pl.memory`](#tfplmemory) | Clonal memory: how much a clone changes between snapshots, vs random clones. |
| [`tf.pl.label_effects`](#tfpllabel_effects) | How much of the difference between clones each label explains. |
| [`tf.pl.heritability`](#tfplheritability) | Genes ranked by clonal ICC; optional gene sets highlighted (<= 3 sets). |
| [`tf.pl.overview`](#tfploverview) | Four-panel summary of a :func:`threadfin.run` result. |
| [`tf.pl.cells`](#tfplcells) | Cells on an embedding coloured by an ``obs`` column (cells without a value in grey). |
| [`tf.pl.signature_heatmap`](#tfplsignature_heatmap) | Mean expression of gene signatures per group (z-scored across groups). |

### tf.pl.set_style

```python
tf.pl.set_style(font_size: 'float' = 7.0) -> 'None'
```

```text
Nature-style defaults: small sans text, hairline axes, editable PDF text.
```

### tf.pl.category_colors

```python
tf.pl.category_colors(categories) -> 'dict'
```

```text
Fixed-order colours for categories; beyond eight, grey 'other'.
```

### tf.pl.clone_map

```python
tf.pl.clone_map(adata, color: 'str | None' = None, *, size_range: 'tuple' = (2, 60), facet: 'bool |
    None' = None, ncols: 'int' = 4, ax=None, fig=None, title: 'str | None' = None, save: 'str |
    None' = None, **legacy_kwargs)
```

```text
Scatter of clones (one point per clone; size = number of cells).

With :func:`threadfin.tl.find_programmes` results, points are placed by
the UMAP of clone profiles and ``color`` defaults to ``clone_programme``.
More than three categories are drawn as small multiples (one panel per
category, highlighted on grey) unless ``facet=False``. Numeric columns use
a single-hue blue scale.

Falls back to the v3 layout (``uns['threadfin']['clone_map']`` from
:func:`threadfin.clonotype_recluster`) when no v4 clone table exists.
```

### tf.pl.coherence

```python
tf.pl.coherence(adata, *, ax=None, save=None)
```

```text
Observed clonal ICC next to the within-stratum permutation null.

Two bars: the share of variance explained by clone identity, and the same
statistic after shuffling clone labels within strata (whisker = 95% of
permutations). The excess of the first over the second is the clonal
signal beyond sampling.
```

### tf.pl.programme_composition

```python
tf.pl.programme_composition(adata, state_key: 'str', *, programme_key: 'str' = 'clone_programme',
    ax=None, save=None, legend: 'bool' = True)
```

```text
Stacked bars: fraction of each programme's cells in each cell state.
```

### tf.pl.stability

```python
tf.pl.stability(adata, *, ax=None, save=None)
```

```text
Bootstrap stability (Jaccard) of each programme, with the 0.75 / 0.6 guides.
```

### tf.pl.association

```python
tf.pl.association(table: 'pd.DataFrame', *, level=None, ax=None, save=None, title: 'str | None' =
    None)
```

```text
Forest plot of clone-level programme effects from :func:`threadfin.tl.association_test`.

Categorical labels: Mantel-Haenszel odds ratios with 95% CIs (log scale)
for one ``level``; numeric labels: rank effects. Filled markers mark
FDR < 0.05.
```

### tf.pl.memory

```python
tf.pl.memory(res: 'dict', *, ax=None, save=None)
```

```text
Clonal memory: how much a clone changes between snapshots, vs random clones.

Bars show the mean noise-corrected squared change of the same clone and of
random same-donor clones; the title gives the memory index with its 95% CI.
```

### tf.pl.label_effects

```python
tf.pl.label_effects(results: 'dict', *, ax=None, save=None)
```

```text
How much of the difference between clones each label explains.

``results`` maps label names to :func:`threadfin.tl.profile_association`
results. Blue: share of clone-profile variance explained by the label;
grey: the same with labels shuffled among clones of the same donor.
```

### tf.pl.heritability

```python
tf.pl.heritability(table: 'pd.DataFrame', *, highlight: 'dict | None' = None, n_label: 'int' = 12,
    ax=None, save=None)
```

```text
Genes ranked by clonal ICC; optional gene sets highlighted (<= 3 sets).
```

### tf.pl.overview

```python
tf.pl.overview(adata, *, state_key: 'str | None' = None, save=None)
```

```text
Four-panel summary of a :func:`threadfin.run` result.
```

### tf.pl.cells

```python
tf.pl.cells(adata, color: 'str' = 'clone_cluster', basis: 'str' = 'X_umap', ax=None, palette: 'dict
    | None' = None, s: 'float' = 10, title: 'str | None' = None, save: 'str | None' = None)
```

```text
Cells on an embedding coloured by an ``obs`` column (cells without a value in grey).
```

### tf.pl.signature_heatmap

```python
tf.pl.signature_heatmap(adata, signatures: 'dict', groupby: 'str' = 'clone_cluster', layer: 'str |
    None' = None, z_score: 'bool' = True, ax=None, cmap: 'str' = 'viridis', save: 'str | None' =
    None)
```

```text
Mean expression of gene signatures per group (z-scored across groups).

Only the signature genes are densified, so this is safe on large datasets.
```

## Simulated data: tf.sim

Repertoires with a known answer, for testing and teaching.

| function | what it does |
|---|---|
| [`tf.sim.simulate_repertoire`](#tfsimsimulate_repertoire) | Simulate an AnnData with ground-truth clones, programmes and states. |

### tf.sim.simulate_repertoire

```python
tf.sim.simulate_repertoire(*, scenario: 'str' = 'default', n_donors: 'int' = 4, contexts_per_donor:
    'int' = 3, n_programmes: 'int' = 4, n_states: 'int' = 6, n_clones: 'int' = 3000,
    clone_size_exponent: 'float' = 2.2, max_clone_size: 'int' = 300, dim: 'int' = 20,
    state_separation: 'float' = 1.5, programme_sparsity: 'float' = 0.4, clone_concentration: 'float
    | None' = 15.0, clonal_offset: 'float' = 0.4, noise: 'float' = 1.0, context_shift: 'float' =
    1.0, clone_nesting: 'str' = 'nested', n_timepoints: 'int' = 1, memory: 'float' = 1.0,
    random_state: 'int' = 0)
```

```text
Simulate an AnnData with ground-truth clones, programmes and states.

Parameters
----------
scenario
    ``"default"``, ``"null"`` or ``"bifurcation"`` (see module docstring).
clone_size_exponent
    Zipf exponent (larger = more singletons).
state_separation
    Standard deviation of state centres (larger = more distinct states).
programme_sparsity
    Dirichlet concentration of programme compositions (smaller = each
    programme dominated by fewer states).
clone_concentration
    How closely clones follow their programme (``None`` = exactly).
clonal_offset
    SD of the clone-specific continuous offset.
noise, context_shift
    SD of cell noise and of context mean shifts.
clone_nesting
    ``"nested"`` (each clone sampled in one context) or ``"spread"``.
n_timepoints, memory
    Number of sampling time points and the probability that a clone keeps
    its programme from one time point to the next (see module docstring).

Returns
-------
AnnData with ``X`` = ``obsm['X_pca']`` = the embedding and ``obs`` columns
``clone_id``, ``donor``, ``context``, ``true_programme``, ``true_state``
and ``clone_size``; parameters in ``uns['simulation']``.
```

## Earlier (v3) functions

Kept importable for analyses made with Threadfin 3; new analyses should use the functions above.

| function | what it does |
|---|---|
| `tf.annotate_specificity` | Annotate clones with antigen specificity. |
| `tf.bcr_reclustering` | Backward-compatible alias for the original Threadfin API. |
| `tf.bcr_similarity_graph` | Sparse clone x clone BCR similarity adjacency. |
| `tf.clonal_pseudotime` | Diffusion pseudotime over the clonotype graph. |
| `tf.clone_centroids` | Compute the centroid of every clonotype in a transcriptional embedding. |
| `tf.clone_distribution` | Clones x groups matrix of within-clone cell fractions. |
| `tf.clonotype_recluster` | Cluster clonotypes by the transcriptional state of their member cells. |
| `tf.community_markers` | Differentially expressed genes per clone community. |
| `tf.community_score` | Mean per-community score of named gene programmes. |
| `tf.expansion_index` | Per-group clonal expansion index (STARTRAC-expa analogue). |
| `tf.integration_diagnostics` | Benisse testCor analogues and a heuristic modality-contribution split. |
| `tf.joint_embedding` | Coupled-Laplacian eigenmap embedding of clones. |
| `tf.migration_index` | Pairwise group migration index (STARTRAC-migr analogue). |
| `tf.transition_index` | Pairwise state transition index (STARTRAC-tran analogue). |
| `tf.specificity_enrichment` | Fisher-exact enrichment of specificities in clone communities. |
