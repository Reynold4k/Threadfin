# Changelog

Each version lists what changed and, under *What this means for you*, what you need to know or do.

## 4.6.0 — 2026-10-30

Five independent reviews of the package (software, statistics, concept, biology and benchmark), and every tutorial
followed in new environments, led to this release.

**Unchanged for the same settings:** clone calls, profiles, the clone map, programmes and the coherence ICC.

**Changed, where a review showed a test answering a different question from the one documented:**
- the null of the coherence test (with `strata_key`);
- the clonal memory index;
- the label tests;
- the gene-set test;
- the intervals of the export lag.

*What this means for you:*
- **Several samples per donor.** If your cells come from several samples, libraries, sort gates or tissues per donor and you centre on the donor, set `strata_key` to the sample column (Tutorial 2, section 5). The coherence test then shuffles clone labels within samples, as the manuscript's analyses did.
- **Clone calls no longer need `PYTHONHASHSEED`.**
- **Label tests, memory indices and gene-set tests** can give different numbers from 4.5. The previous estimators remain available for comparison (below).

### New

- **Composition counted per clone.** `tf.run(..., composition_weight="clones")` (and `weight=` in `tl.programme_composition` and `pl.programme_composition`) averages each clone's share of cells in each state over the programme's clones, so every clone counts once. The default, `"cells"`, is unchanged. The summary, the Methods paragraph and panel d of the overview say which was used.
- **`strata_key`** in `tf.run`: the column within which the coherence test shuffles clone labels (default: `sample_key`). Strata stay within donors: when a library pools several donors, labels are shuffled within each donor × library combination, with a warning (also in `tl.gene_heritability`). The summary and panel b name the strata and give the excess of the observed ICC over shuffled clones, in percentage points.
- **Clone calls are identical in every Python session.** `define_clones_dandelion` calls Dandelion under Python's hash seed 0; when Python was not started with `PYTHONHASHSEED=0`, it does so in a separate process (about 2 s more). `hash_seed=None` (`clone_options={"hash_seed": None}` in `tf.run`) keeps the old behaviour. Without a fixed seed, about one expanded clone in eight had changed its cells between sessions.
- **The Dandelion extra installs on Python 3.10–3.14.** Importing threadfin restores `matplotlib.cm.get_cmap`, which Dandelion 0.5.7 needs. Up to Python 3.12 the extra keeps the validated matplotlib < 3.9. Clone calls are identical on Python 3.11, 3.13 and 3.14.
- **The run is recorded in the AnnData** (`adata.uns['threadfin']['run']`): settings, analyses not run and why, package versions and BLAS threads. `ThreadfinResult.from_adata(adata)` rebuilds the result after `write_h5ad`/`read_h5ad`; `summary()`, `methods()` and `plot()` give the same output as before saving.

### Statistics

- **Coherence without samples** (`tl.clonal_coherence`, `tl.gene_heritability`). With a `donor_key` but no `sample_key` or `strata_key`, clone labels are now shuffled within donors. They were shuffled across all cells, so that differences between donors counted as clonal: in simulations without any clonal structure, every test gave the smallest possible P value.
- **Clonal memory** (`tl.clonal_memory`).
  - Each clone is now compared with other clones of its donor seen at the same two levels. It was compared with any clone of the later level, so clones found at two levels, a selected group that can share a state, read as having memory: in simulations with selection but no memory, the old index was 0.81 and its test rejected in 100 of 100 data sets, against −0.01 and 5 of 100 now.
  - Sampling noise is removed with each snapshot's own variance instead of one pooled variance, which put the index above 1 in 77% of data sets with a true index of 0.5.
  - The p-value permutes the later profiles among the clones of each donor seen at the same two levels.
  - `reference="all", noise="pooled"` reproduces 4.5. The index is not bounded by 0 and 1, and an index without an interval is reported as not interpretable.
- **Label tests** (`tl.profile_association`, `tl.association_test`, and `test=` in `tf.run`).
  - Labels are shuffled among clones of the same donor and of similar size (2, 3–4, 5–9, ≥ 10 cells). Small clones' profiles are pulled towards the average profile, so a label that follows only clone size was called significant in up to 42% of simulated data sets; now 3–8%, with 0–7 points of power lost. `size_strata=False` reproduces 4.5.
  - A label fixed within each sample (a tissue, sort gate or visit sequenced in its own library) is flagged: the comparison is between samples, not between independent clones. `tf.run` says so once per label.
  - Each programme effect is shown once: a two-level label, or any label tested on two programmes, gives mirrored rows (same P value, inverse odds ratio), which are marked (`mirror`) and left out of the summary.
  - Strata from a column that varies within clones put each clone in its most common level (a tie in the first level). Tied clones were pooled into one stratum across levels, and under pandas 3 the test failed.
  - The numeric programme test is centred within strata.
  - Levels that look like missing values ("nan", ".") give a warning.
- **Gene sets** (`tl.geneset_heritability`). The new competitive test on Fisher's z of the clonal ICC has an error term that combines the spread between genes with a clone bootstrap. The Mann–Whitney test treated co-expressed genes as independent (16% false positives with co-expressed null genes; 50–60% when they also shared clonal programmes). It is kept only as the column `pvalue_mw_genes_independent`. `tl.gene_heritability` gains `n_boot` and the column `excess_fisher_z`; genes that share a name keep their own rows, and the gene-set test asks for unique names.
- **Export lag** (`tl.export_lag`).
  - It is described as a within-clone mutation gap; its sign varies between clones.
  - The median gap has a sign-test interval (at least 95% coverage for any distribution, ties included) instead of a bootstrap.
  - The clock goes through per-time-point medians with Sen's exact interval, as in the manuscript's analysis.
  - Days come with an interval.
  - `donor_key` adds per-donor results; clones without a donor stay in the overall result.

### Fixed

- **A run that could not make programmes left those of an earlier run** in `adata.obs` and `adata.uns['threadfin']`, where plots and `ThreadfinResult.from_adata` found them. They are now removed.
- **A fix suggested by an error message created false clones.** For clone ids shared between donors, the suggested line turned cells without a clone into one clone per donor ("BCP005|nan"). It now keeps missing ids missing, and clone ids that look like missing values give a warning.
- **`counts_layer=` and `batch_key=` were ignored on a second run** when `obsm['X_threadfin']` already existed. The embedding is now rebuilt when they differ, and the log says which embedding was used.
- **No programmes, and the wrong reason.** When the programme step could not run (for example with `n_boot` below 5), the summary said there were too few clones; it now gives the actual reason. `n_jobs`, `map_n_neighbors`, `map_min_dist` and `random_state` are checked before any work starts.
- **The summary contradicted itself** where an odds ratio within donors and the pooled shares point in opposite directions. It now says so, prints the 95% interval of each odds ratio, and bounds odds ratios from sparse tables ("> 100").
- **Clonal memory paired tissues in alphabetical order without saying so.** It now uses the order of an ordered Categorical, warns when it falls back to alphabetical order, and the summary names the order.
- **Smaller fixes.**
  - A per-cell table from `read_bcr` was not recognised as per-cell.
  - The pre-check of `tf.run` asked for 2 expanded clones although 3 are needed.
  - Harmony's batch-aware gene selection warned about an AnnData view.
  - `random_state=None` failed late.
  - `define_clones_dandelion(donor_key=..., check_contigs=True)` failed on contigs without read counts.
  - Methods and markers functions now say which layer is missing instead of raising `KeyError: 'log_norm'`.

### Reporting

- **The summary:**
  - asks "Are cells of one clone more alike than cells of different clones?" and describes the ICC as the share of the variance of cell states (the expression embedding);
  - counts clones "with two or more cells", the ones profiled;
  - says that reliability refers to the smoothed profile;
  - says "every split passed the split test";
  - flags unstable programmes on the label-test lines;
  - reports clones left out of a label test for lack of a majority level, and labels that are the state annotation itself (positive controls);
  - says that programmes are descriptive when coherence is not significant.
- **Warnings** when no immunoglobulin or TCR genes are recognised (gene names that are not symbols), and when `sample_key` is not given.
- **Docstrings that disagreed with the code were corrected:** the split test, the reliability weighting of the clone graph, the shrinkage target, the reliability under smoothing, and the clone-map tuner.

### Tests

- **New tests of statistical properties:**
  - calibration of the coherence, profile-association and association tests under nested nulls;
  - permutation p-values and BH on hand-worked examples;
  - variance components against the closed-form ANOVA;
  - the Mantel–Haenszel odds ratio;
  - no receptor genes among the highly variable genes;
  - the smoothing neighbours;
  - the BCR-file path of `tf.run`;
  - plots of test results;
  - the new memory, association, gene-set, export-lag and Dandelion behaviour.
- **Mutation test.** The independent review planted 25 realistic one-line errors in the statistics. The test suite now detects all 25, against 8 before.
- **The Dandelion job of continuous integration** runs on Python 3.12, 3.13 and 3.14, without `PYTHONHASHSEED`.
- **The sdist's tests** skip the checks of files it does not contain.

### Documentation

- README, Methods and Tutorials 1–5 were revised with the reviews:
  - `strata_key` and when to set it;
  - composition weighting;
  - run time and memory measured up to 205,630 cells, with the scope of the million-cell figure;
  - reproducibility across `n_jobs`, BLAS threads and cell order;
  - the split test detects any departure from one Gaussian group;
  - stability is a cell-resampling stability;
  - marker statistics are descriptive;
  - an expected-squared-correlation reading of reliability;
  - ASC wording;
  - the export lag as a mutation gap, with the alternatives to its interpretation;
  - genes "shared within clones" instead of "inherited";
  - corrected and completed shown outputs.
- **Tutorial 2:** a corrected isotype snippet (productive contigs, the heavy chain with the most UMIs), the scirpy multi-sample loop, and new troubleshooting rows.
- **Tutorial notebook:** conda-forge only, a complete `environment.yml` block, no seed setting needed, and `requirements-lock.txt` with every package at the version of the published run. A weekly job of continuous integration runs the notebook on the GEO data with the newest allowed dependencies.
- **The figure gallery:** every figure of the 16 validation datasets (`manuscript/analyses/technical_validation/tutorial_path/figures/`).

## 4.5.3 — 2026-10-28

A dependency released on 27 October 2026 stopped new installations from running the tutorial notebook and from
reproducing the published numbers.

*What this means for you:* if you installed Threadfin on or after 27 October 2026, run `pip install -U threadfin`
(or `pip install "pynndescent<0.7"`); results then match the documentation again. Nothing else changes.

### Fixed

- **pynndescent 0.7** (27 October 2026) broke two things, so Threadfin now requires `pynndescent<0.7`:
  - scanpy's neighbour search (`sc.pp.neighbors`, which the tutorial notebook calls, as do many analyses) uses
    pynndescent for 8,192 cells or more and stopped with "'csr_array' object has no attribute 'getnnz'": scanpy
    1.11.5 and 1.12.4 cannot read the sparse arrays that pynndescent 0.7 returns;
  - pynndescent 0.7 finds different approximate neighbours (on 60,000 simulated cells with the same seed, 22% of
    the rows of the 15-nearest-neighbour graph were identical; 90.3% and 91.3% of the exact neighbours were
    found), which changes Threadfin's smoothing of datasets with more than 50,000 cells and the clone-map layout
    of more than 4,096 clones. numba 0.68.1 instead of 0.68.0 changed nothing.
- **The Dandelion extra needs Python 3.10–3.12.** On Python 3.13 and 3.14, `pip install "threadfin[dandelion]"`
  fails while building matplotlib: Dandelion 0.5 imports nxviz, which needs matplotlib < 3.9, and matplotlib < 3.9
  has no packages for those versions. The README, Tutorial 2 and the Methods now say so. Threadfin without
  Dandelion runs on 3.10–3.14 (Tutorials 1, 3 and 4 print the same numbers on Python 3.11 and 3.14).

### Documentation

- Tutorial notebook, Step 0: conda takes packages from conda-forge only (`--override-channels`, and `nodefaults`
  in `environment.yml`), so newer conda versions do not ask you to accept the terms of Anaconda's own channels;
  the download command for `environment.yml` is given; the first cell also prints the pynndescent version and
  checks for Threadfin 4.5.3 or later.
- Tutorial 2, troubleshooting: what to do about the `getnnz` error and about the matplotlib build failure on
  Python 3.13 or later.

## 4.5.2 — 2026-10-27

Fixes found by checking the documentation, sentence by sentence, against the code.

*What this means for you:* no number changes. Two messages do: a run with a single programme no longer describes
the clones as a continuum, and a label that is fixed within every donor (age, sex, disease group) is listed as not
testable instead of being reported as not significant with p = 1. Raw counts kept in a layer can now be passed to
`tf.run` with `counts_layer=`.

### Fixed

- **Labels fixed within every donor** were tested by shuffling them among the clones of each donor, which changes
  nothing, so `tf.run` reported "does not significantly explain clone states" with p = 1. `tl.profile_association`
  and `tl.association_test` now raise a `ValueError` that says why, and `tf.run` lists the label under "Not run" in
  the summary, as the Methods page already said.
- **One programme is not a continuum.** When no split passed the split test, the summary and the log said that the
  clones vary "along a continuum". The split test is conservative (Methods, "A note on counting programmes"), so
  they now say that no split into separate groups was significant.
- **`tf.run(..., counts_layer="counts")`.** The documentation said raw counts could be kept in `layers["counts"]`,
  but `tf.run` could not read them; it now passes `counts_layer` to `tf.pp.prepare_embedding`.
- **A programme test that cannot be run** (`association_test` raising `ValueError`) no longer stops `tf.run`; like
  the other optional tests it is listed under "Not run".
- The summary numbers its optional sections in order (clonal memory was numbered 4 even without label tests).

### Documentation

- Methods: how the Leiden resolution is chosen (the most programmes whose clone-weighted mean stability reaches
  0.75; otherwise the most stable partition, with a warning); programmes are kept for passing the split test and
  reported with their stability, not removed for low stability; the bootstrap holds the fitted profile model
  fixed; which clones are assigned to a programme afterwards; the ICC uses unsmoothed coordinates, the
  reliabilities the smoothed kernel features; the returned map uses the chosen settings; a new section on the
  export lag; an option that does not exist (a composition-balanced reference) removed.
- Programme composition (summary, `tl.programme_composition`, overview panel d) is the share of each programme's
  cells in each state; Tutorials 1 and 4 said that clones were weighted equally.
- The coherence test shuffles clone labels within samples; only the tests of your labels shuffle within donors
  (Tutorials 1 and 2).
- Tutorial 2: the isotype snippet works whether or not the contig table was converted to AIRR names in section 6.
- Tutorial 4: `association_test` needs the programmes; only coherence, programmes and memory detect changed
  inputs; the lineage functions take trees built elsewhere.
- README and Tutorial 2: profiles, reliabilities and `result.clones` cover clones with two or more cells; panel d
  of `result.plot()` shows programme sizes unless `state_key` is given; the licence link works on PyPI.
- Tutorial 5: reliability is the expected *squared* correlation; the genes named for the main axis are among
  those printed; the division curve rises mostly over the first three deciles.
- Compatibility: the verification of 4.5.1.
- Continuous integration also installs Dandelion and scirpy, so the README's main path (clones called from
  contigs) is tested on every change, and runs every week so that a change in a dependency is noticed.

## 4.5.1 — 2026-10-25

Fixes found by a first-time user following the README and every tutorial in new environments installed from PyPI.

*What this means for you:* if you installed `threadfin[dandelion]` on or after 2026-10-25, check
`pip show sc-dandelion`. Version 1.0 calls clones differently from the 0.5.x that Threadfin is validated with (on
the GSE287123 tutorial data: 7,208 instead of 8,736 clones, largest clone 494 instead of 252 cells), and every
downstream number changes with it. Run `pip install "sc-dandelion<1"`, or upgrade Threadfin, and call the clones
again. Results obtained with Dandelion 0.5.x are unchanged.

### Fixed

- **Dandelion is pinned below 1.0** in the `[dandelion]` extra (`sc-dandelion>=0.5.7,<1`), and clone calling warns
  when Dandelion 1.x is installed.
- **T-cell contig files passed as a path.** `tf.run(adata, bcr="tcr_filtered_contig_annotations.csv", ...)` called
  the clones and then stopped with "No productive BCR contigs left after filtering": the per-cell summary that is
  attached after clone calling read B-cell chains only. T-cell files now keep the clone ids only, as DataFrames and
  scirpy objects already did.
- **`result.plot()` panel b** showed the most recent coherence test run on the object (in the end-to-end tutorial,
  the additional test of Step 11) instead of `result.coherence`. `plot.overview` takes `coherence_result=`.
- **`result.plot()` without programmes** (too few reliably profiled clones) raised a KeyError; it now draws the
  clone map coloured by clone size and says that there are no programmes.
- **`tf.define_clones_dandelion`** accepts Cell Ranger's column names (`barcode`, `v_gene`, `j_gene`, `cdr3_nt`)
  as its docstring said, by passing the table through `tf.read_contigs`.
- **`clone_labels(..., how="fraction:<level>")`** warns when the level never occurs (for example `IGHA` in a column
  holding `IGHA1` and `IGHA2`), instead of returning 0 for every clone without a word.
- The summary warns when fewer than 10 clones have two or more cells, and says "1 cell" rather than "1 cells".
- The warning "No resolution reached stability" now says it refers to the clone-weighted mean over programmes and
  that single programmes can still be stable.
- Docstrings of `clone_profiles`, `profile_association`, `find_programmes` and `embed_clones` no longer contain a
  mis-indented paragraph of development history.

### Documentation

- README: the simulated-data call works as written (`sample_key="context", basis="X_pca"`); `sample_key` must
  never be a sort gate, tissue or time point; `n_jobs=8` appears in the example whose timing it describes.
- Tutorial 2: calling clones yourself uses `tf.read_contigs` and a donor column; how to make an `isotype` column
  from the heavy-chain constant gene; barcode mismatches are fixed in the contig table, not with `read_bcr`; new
  troubleshooting rows for Dandelion 1.x and the stability warning.
- Tutorial 5: how to get the notebook without cloning the repository; the real barcode, doublet and negative counts
  per tube; map settings and stabilities can differ slightly between computers; panel b shows Step 10's test.
- Tutorial 1: the link to the installation section works.
- Threadfin is on PyPI: `pip install threadfin` (with Dandelion: `pip install "threadfin[dandelion]"`). The
  README, the tutorials, the tutorial environment and the install hint shown when Dandelion is missing use it.

## 4.5.0 — 2026-10-24

New features answer the questions a specialist in clonal analysis would ask first: does it work for T cells, does it
take my scirpy object, how are doublets and cells with two light chains handled, and which genes vary between clones
when there are no distinct programmes? The fixes were found by following the tutorials on all 14 public BCR datasets
of the case studies and on the two LARRY datasets, in a new environment, as a user would
(`manuscript/analyses/technical_validation/tutorial_path/`). Results of the published analyses are unchanged.

*What this means for you:* existing analyses give the same results. T cells run with the same call; scirpy users
pass their object as `bcr`; `print(result.methods())` gives a Methods paragraph with the settings used. Read the new
log line on cells with two chains of one kind: when many of them have two heavy chains, compare with
`clone_options={"check_contigs": True}`.

### New

- **T cells.** `tf.run(adata, bcr=tcr_contigs, ...)` and `tf.define_clones_dandelion` accept TCR contigs: clones
  are called on the beta (or delta) and alpha (or gamma) chains with identical junctions (Dandelion's default for
  TCRs, which do not mutate; B cells keep 0.85). `identity` can be one number or a dict per chain type. Tested end
  to end on scirpy's example data (Wu et al. 2020, 3,000 T cells).
- **scirpy objects as input.** `bcr=` takes a scirpy AnnData or MuData (receptor chains in `obsm["airr"]`), as well
  as files and contig tables.
- **Clone-calling settings in `tf.run`**: `clone_options={...}` is passed to `define_clones_dandelion`, for example
  `{"identity": 0.9}` or `{"check_contigs": True}`, which runs Dandelion's contig check first (a cell with two heavy
  chains keeps the one that dominates in UMIs, or is set aside when none does).
- **Cells with two chains of one kind are reported.** Dandelion gives such cells (4–20 % in the public datasets,
  mostly two light chains) a clone id that joins both chains, so most of them form a clone of their own. The log now
  says how many cells this affects and how many have two heavy chains; the counts are in the returned `attrs`.
- **`tl.landscape_genes`**: genes whose clone-average expression changes smoothly across the clones (Moran's I on the
  clone-profile graph, analytic P values, BH correction), with the direction along the main axis of the profiles.
  For datasets in which clones vary along a continuum and there are no programmes to compare.
- **`tl.programme_markers(..., state_key=, state=)`** compares programmes within one cell state (for example their
  germinal-centre B cells only), so the markers are not simply those of the states the programmes contain.
- **`result.methods()`** writes a Methods paragraph for the analysis just run — clone calling, embedding, profiles,
  coherence test, clone map, programmes and label tests — with the settings and package versions that were used and
  numbered references, as a starting point for a paper.
- **How much do the results depend on the clone definition?**
  `manuscript/analyses/technical_validation/clone_definition/`: junction identities 0.75–1.0 and the contig check on
  two case studies; every conclusion holds.
- **How many clones are needed?** `manuscript/analyses/technical_validation/power/`: the coherence test is calibrated
  and detects a clonal ICC of 0.1 with about 10 clones of two or more cells, 0.05 with about 40 and 0.02 with about
  160–320. Tutorial 2's checklist now quotes these numbers.
- **Further checks** in the same folder: the conclusions hold for 2,000–5,000 variable genes and 20–50 principal
  components, and leaving the receptor genes in would inflate clonal coherence (`embedding/`); a million cells take
  about an hour on 8 cores and 8.3 GB (`scalability/`); T cells run end to end from a scirpy object (`t_cells/`).

### Fixes

- **A subset made without `.copy()` works.** `adata[mask]` is a view; Threadfin's results written to a view
  were lost, and `threadfin.run` then stopped with the misleading "Run threadfin.tl.clone_profiles() first".
  `run`, `tl.clone_profiles` and `pp.prepare_embedding` now turn a view into its own object first (as AnnData
  does on any change) and say so.
- **Contig tables with both a 10x `barcode` and an AIRR `cell_id` column** (for example a deposited AIRR table
  to which harmonised barcodes were added) no longer end up with two `cell_id` columns; the column that names
  the cells of `adata` is used.
- A cell that Dandelion cannot place receives no clone instead of the empty clone id `""`, which several such cells
  of one donor would have shared (one cell of GSE246382 had it; no clone of two or more cells was affected).
- A `productive` column read as numbers (1.0/0.0, as pandas reads a column with gaps) is understood; such contigs
  were all dropped before.
- The Dandelion version recorded with the clone ids is that of the installed package; Dandelion's own
  `__version__` reports the version of whichever git repository the current directory is in.

### Clearer messages

- **A misspelled column name is answered with the closest existing names** ("Did you mean 'sort_gate'?").
- **Normalised data passed as counts is flagged**: `pp.prepare_embedding` (and `run` without `basis=`) warns when
  the counts have non-integer values, because Threadfin normalises and log-transforms them itself.
- The message for a missing Dandelion installation gives the exact install command; the message for an
  undefined clonal-memory interval says what it means and that the estimate and P value are unaffected.
- **Fewer alarming warnings that were not about the user's data**: the notice that nxviz prints when Dandelion is
  imported is no longer shown, the expression matrix is made dense before scaling (scanpy warned that it would
  densify it), and duplicated gene names are made unique inside the embedding step without a warning per copy.
  Dandelion's notice that a table "needs to undergo sanitization", printed twice per donor, is hidden.
- **One clonal ICC in the log.** The clone-profile step printed a second "clonal ICC" (of the smoothed profile
  features, which only sets the shrinkage); only the tested ICC of the coherence step is reported now.
- **The warning that no resolution reached the stability threshold** is given only when two or more programmes
  remain after the split test; when the clones form a continuum, the log already says so.
- **Receptor-neutral wording**: the summary says "Cells with a BCR", "a TCR" or "a clone id" according to the data,
  and the messages about unmatched barcodes name the receptor data rather than the BCR.

### Documentation

- Tutorial 2: a section on how clones are called (defaults, `clone_options`, cells with two chains of one kind,
  T cells, scirpy objects); the troubleshooting table quotes the current messages; the expected share of matching
  BCR contigs is taken from the 14 datasets tested. Tutorial 1: time and memory on real data. Tutorial 4: the new
  gene-level functions. Tutorial 5 (notebook): a step on the genes along the clone landscape; needs
  threadfin ≥ 4.5.0 and explains the renamed contig ids. Methods: T cells, the contig check, within-state markers
  and landscape genes.
- `CITATION.cff` carries the current version; a test keeps it, `pyproject.toml`, `threadfin.__version__` and the
  function reference in step.

### Known issue

- The pattern for TCR genes, which are left out of the expression embedding, also matches *TRADD*, which is not a
  receptor gene. Correcting it would change 2–4 of the 3,000 variable genes in 7 of the 14 public datasets, and with
  them the published embeddings, so it is kept until the next major version.

## 4.4.1 — 2026-10-23

- **Installs on Windows.** harmonypy 2.x publishes no Windows wheels and its source does not compile with
  Microsoft's compiler, so `pip install threadfin` failed on Windows. Windows now receives the pure-Python
  harmonypy (< 2); Linux and macOS are unchanged. Continuous integration now also runs on macOS and Windows.
- The repeatability test of the legacy `clonotype_recluster` no longer fails with pandas 3. The function was
  always deterministic; the test compared the labels of cells without a clone as NaN == NaN, which pandas 3
  keeps missing in `astype(str)` (issue #2).

## 4.4.0 — 2026-10-23

### New

- **Several CPU cores, the same result.** `threadfin.run`, `tl.embed_clones`, `tl.tune_clone_map` and
  `tl.find_programmes` take `n_jobs` (default 1; `-1` uses every core the job may use). The clone-map search
  lays out its candidate maps in parallel and the programme search runs its bootstrap replicates in parallel.
  The chosen map, its coordinates and the programmes are identical for any `n_jobs` (verified against 4.3.1 on
  GSE287123). On GSE287123 RBD (1,887 clones) 8 cores made the clone-map search 2.4 times and the bootstrap
  2.6 times faster; more than 8 cores gave nothing further.
  *What this means for you:* on large data, `tf.run(..., n_jobs=-1)`.
- **Analyses that could not be run are listed in the summary.** A requested label test whose label has no
  usable values, or a clonal-memory axis on which no clone is seen twice, was skipped with a log line only. It
  is now recorded in `result.skipped` and listed under "Not run" in `result.summary()`; everything else is
  reported as before.
- **A function reference**, [`docs/API.md`](docs/API.md), generated from the docstrings (a test keeps it in
  step with the code).

### Fixes

- **Contig ids reused across libraries.** Cell Ranger names contigs after the cell barcode
  (`ACGG...-1_contig_1`), so when one donor was sequenced in several libraries two of its cells could carry the
  same contig ids, and Dandelion mixed their chains. `define_clones_dandelion` now renames duplicated ids after
  their cell. In GSE287123 this affected 18 of 21,939 cells (RBD) and 8 of 4,855 (NP); the RBD and NP
  "light-chain-only groups" counted in 4.3.1 were this artefact. Figure 4 of the manuscript is unchanged
  (`manuscript/analyses/rbd_complete_pairs/`).
  *What this means for you:* if your donors span several 10x libraries, re-call clones with 4.4.0.
- **`define_clones_dandelion(cell_key=...)`** checked for a `cell_id` column before using `cell_key`, so a
  table whose barcode column had another name was rejected; the required columns now follow `cell_key`.
- **Clearer messages** for clone ids shared between donors (how many, an example, and the one-line fix), for an
  embedding with missing or infinite values (how many cells, and how to drop them) and for data in which no
  clone has two cells.

### Installation

- **Python 3.10 to 3.14.** The test suite and the tutorial pass on Python 3.13 and 3.14 with current releases of
  every dependency, and on Python 3.10 with the oldest versions `pyproject.toml` allows; continuous integration
  now runs all of these.
- **Minimum versions raised:** umap-learn 0.5.5 and pynndescent 0.5.8. Older releases import `pkg_resources`,
  which new Python environments no longer include, so an installation that picked them could not draw the clone
  map. `joblib` (already installed with scikit-learn) is now declared.

## 4.3.1 — 2026-10-20

### Fixes

- **Cells without a heavy chain no longer receive a clone.** `threadfin.define_clones_dandelion` (and
  therefore `threadfin.run(bcr=...)`) returned Dandelion's light-chain-only groups (ids such as `B_VJ_...`)
  as clones. Light-chain rearrangements are far less diverse than heavy-chain ones, so these groups join
  unrelated cells: in GSE287123 they were 5,276 of 21,939 cells and 740 of 2,626 groups of two or more cells.
  Such cells are now left without a clone (`require_heavy_chain=True`, the new default); the number is logged
  and stored in `ids.attrs["n_cells_without_heavy_chain"]`. The nine datasets of the manuscript were essentially
  unaffected (their loaders keep only cells with a heavy chain; one two-cell group among the 1,887 profiled
  RBD clones was light-chain-only). Four additional case studies whose loaders read 10x per-cell tables were
  affected — GSE171703 (1,689 of 2,281 groups of two or more cells were light-chain-only), the paediatric
  tonsil (419 of 780), EBV-infected tonsil organoids (1,198 of 8,114) and bone-marrow plasma cells (67 of
  6,855) — and were re-run with these groups removed (`manuscript/case_studies/run_case_study.py` now drops
  them as well).
  *What this means for you:* if you ran `threadfin.run(bcr=...)` or `define_clones_dandelion` with 4.3.0 or
  earlier, re-run it; expect fewer clones with two or more cells, and more reliable ones.
- **`tl.export_lag` works on large datasets.** The maturation slope used to convert mutation gaps into days was
  computed over every pair of germinal-centre cells, which needs 8 × 10⁸ slopes for 40,000 cells. Above five
  million pairs it is now the median over a fixed-seed random sample of five million pairs, which agrees with
  the exact value to within 5 × 10⁻⁵ in tests.
  *What this means for you:* `export_lag(..., time_key=...)` no longer runs out of memory on large data;
  results on small data are unchanged.
- **Clone calls record the hash seed.** Dandelion's clone assignment depends on the order in which Python
  iterates over sets of sequences, which changes in every new Python session unless `PYTHONHASHSEED` is fixed:
  the same GSE287123 contigs gave 8,729 clones in one session and 8,731 in another. `define_clones_dandelion`
  now stores the seed in use in `ids.attrs["python_hash_seed"]` and logs a note when it is not set.
  *What this means for you:* for clone ids that are identical run after run, start Python (or register your
  Jupyter kernel) with `PYTHONHASHSEED=0`; conclusions should not depend on it.

### Installation

- New optional extra: `pip install "threadfin[dandelion] @ git+https://github.com/Reynold4k/Threadfin.git"`
  installs Dandelion together with a compatible matplotlib. Dandelion 0.5.7 calls `matplotlib.cm.get_cmap`,
  which matplotlib 3.9 removed, but does not declare the limit, so a plain `pip install sc-dandelion` can leave
  `import dandelion` failing.
  *What this means for you:* use the extra, or `pip install sc-dandelion "matplotlib<3.9"`.

### Documentation

- New **end-to-end tutorial notebook** (`docs/tutorials/05_end_to_end_tutorial.ipynb`): environment, download of
  a published dataset from GEO (GSE287123), hashtag demultiplexing, clone definition, embedding, `threadfin.run`,
  every output explained, clone-level tests with the library caveat, and the main result of the manuscript's
  Figure 4 — tested end to end in a fresh environment before release. `docs/tutorials/environment.yml`
  recreates the tested environment.
- Tutorials 1–4, Methods, the case-study guide and the figure guide were expanded for beginners: what each step
  does and why, how to read every output, common misreadings, a checklist for preparing your own data, how to
  choose `donor_key` / `sample_key` / `batch_key`, a clone-map trust checklist and a troubleshooting table.
- The homepage images that a `.gitignore` pattern had hidden are published.

### Repository

- Everything behind the manuscript is now in `manuscript/`: figures and their code (`manuscript/figures/`,
  formerly `paper/figure_plan/`), the case studies (`manuscript/case_studies/`, formerly `case_studies/`), every
  analysis beyond the figures and the technical validation (`manuscript/analyses/`), a record of negative
  results, and an end-to-end benchmark against Clonotrace, clone2vec, Benisse, CoNGA's reduction and scirpy
  (`manuscript/benchmarking/`). `manuscript/README.md` describes the state of the manuscript.
  *What this means for you:* update links and paths that pointed to `case_studies/` or `paper/figure_plan/`.

## 4.3.0 — 2026-10-18

### The clone map chooses its own settings

- `threadfin.tl.embed_clones` and `threadfin.run` now choose the UMAP settings of the clone map for each
  dataset (`n_neighbors="auto"`, `min_dist="auto"`, the new default). Numbers still set them by hand, and
  `n_neighbors=50, min_dist=0.8` reproduces the fixed default of 4.2. In `threadfin.run` the settings are
  `map_n_neighbors` and `map_min_dist`.
- New `threadfin.tl.tune_clone_map`: every candidate layout is scored for coherence (the fraction of
  clones in one connected piece of the map) and for local and global fidelity to the clone profiles. The
  most faithful setting that keeps at least the target fraction of clones in one piece *in every layout
  tried* is chosen. The target comes from the profiles themselves (the largest component of their mutual
  15-nearest-neighbour graph, capped at 0.99), so a map is not made more fragmented than the profiles. The
  search lays the clones out over a ladder of settings with shared seeds, smooths the scores over
  neighbouring settings, confirms the best settings on six layouts, and if necessary tries weaker repulsion
  between clones. Above 3,000 clones the search runs on a random subset and the best settings are
  confirmed again on all clones. `method="adam"` follows a stochastic gradient instead (Adam with simultaneous-
  perturbation gradient estimates).
- New `threadfin.tl.map_quality` scores any clone map, and `threadfin.tl.layout_clone_map` lays out and
  checks the final map: if it falls short of the target, further seeds and the next confirmed settings are
  tried. With settings given by hand, `embed_clones` reports when the map is more fragmented than the
  profiles.
- `adata.uns["threadfin"]["profiles"]["map_params"]` records the chosen settings, the quality of the map,
  every layout tried and the full search; it is stored as tables, so results can be written to `.h5ad`.
- The map is a display only: profiles, coherence, programmes and all tests are unchanged.

### Fixes

- Results can be written with `adata.write_h5ad` after a full analysis. The split-test table of
  `find_programmes` held Python lists and the clonal-memory interval was a tuple, which AnnData cannot
  write; stored results are now converted to writable types (lists of communities become comma-separated
  text, with their counts in `n_communities_left` / `n_communities_right`). Returned objects are unchanged.
- The Dandelion adapter recognises missing junction amino-acid sequences under pandas 3, where missing
  values in a text column are no longer turned into the string "nan".
- `examples/tutorial.py` runs its step-by-step section on a copy, so the overview figure of `tf.run` can
  still be drawn at the end.

### Documentation

- New tutorials (`docs/tutorials/`): getting started, your own data, the clone map, and step by step.
- `docs/METHODS.md` describes the clone map and corrects the description of reliability: no clone is
  excluded for low reliability; reliability weights a clone's part in programme discovery.
- New README, case-study overview and figure index.

### Repository

- Benchmarks, technical validation and analyses of non-B-cell lineage data (Clonotrace LARRY and NSCLC,
  WATERMELON) are no longer part of the repository.

## 4.2.0 — 2026-10-10

### New analysis: heritability of cell state along the receptor lineage tree

- `threadfin.tl.lineage_forest` and `threadfin.tl.lineage_heritability` (module `threadfin.phylo`) ask, inside
  a clone, whether cells that are close relatives on the somatic-hypermutation lineage tree occupy more similar
  transcriptional states than distant relatives of the same clone sampled in the same block.
- The estimator is a phylogenetic mixed model fitted by REML and pooled over the whole forest, which makes it
  usable in the regime an immune repertoire produces: thousands of trees of two to eight cells, nested in
  animals and sort gates. Each clone x block unit gets a free intercept, so clone-wide and block-wide state
  cannot contribute; the cell's root-to-tip mutation depth is a fixed effect, which separates "more mutated
  cells differ" from "relatives resemble each other".
- Four kinship kernels (exponential in tree distance, identical genotype, identical mutated genotype,
  Brownian), a profile-likelihood interval, a boundary likelihood-ratio test and a Freedman-Lane permutation
  score test that stays valid when units differ in variance.
- `simulate_lineage_trait` and `lineage_power` plant a component of known size on the real trees and report
  bias, type I error and power, so that a null result can be stated as "a heritability above x would have been
  detected".

### Calibration of the clonal-programme split test

- `split_test` now takes `null=`, and its documentation reports measured behaviour for both settings. The
  shipped default (`"total"`, unchanged) is valid but conservative: with a two-group structure planted in real
  clone profiles it detected nothing up to a separation of four within-group standard deviations. An
  alternative that estimates the null covariance within the candidate groups (`"within"`) was implemented,
  measured to have a false-positive rate of 1.0 at zero separation, and is therefore documented as invalid and
  left non-default.
- Consequence for users: a non-significant split from `find_programmes` is **not** evidence that clones form a
  continuum. The documentation now says so.

## 4.1.0 — 2026-10-09

### New optional analyses

- StateDensityModel provides fixed, clone-balanced reference regions, context-aware
  leave-own-clone-out backgrounds, count-based shrinkage, conditional posterior
  intervals, separate simultaneous sampling intervals, regional contrasts and
  clone-balanced gene/module descriptions.
- Query counts can be accumulated in batches. Numeric NPZ model serialization
  does not require pickle.
- FrozenExpressionModel fits receptor-excluded HVGs, scaling and PCA on reference
  cells only, then projects held-out cells without refitting.

### Correctness and reproducibility

- Kernel profile reconstruction preserves the random-feature basis after bandwidth
  estimation. Legacy models are matched to saved features or rejected; modified
  cells, labels or coordinates invalidate the cached fit. Older AnnData files that
  omit None-valued context parameters are supported.
- The minimum capture count now solves the actual weighted reliability function.
- AIRR heavy/light masks remain aligned after UMI sorting. The NumPy affine-gap
  global alignment fallback penalizes leading gaps.
- Memory null matches never borrow another donor. Insufficient matched pairs and
  unidentified noise-corrected ratios are reported explicitly.
- Degenerate Gaussian split nulls do not yield false p=0 results. Split tests remain
  approximate because they do not repeat the full model-selection procedure.
- Additional checks reject ambiguous IDs, missing strata, invalid weights/counts,
  and undefined zero-variation or empty-data inference.

### Migration and scientific interpretation

The existing mean/kernel pipeline remains the default. Recompute downstream
programme stability, assignments and memory when upgrading a 4.0 analysis:
those routines may have reconstructed a different random-feature basis.
The initial fitted profile formulas are unchanged. No automatic conversion of
an unreproducible legacy fit is attempted.

The optional state model is not a validated replacement for every endpoint.
Simulations retain prior-misspecification failures; whole-mouse reporter tests
often favour RNA means. Its data_weight is not the legacy profile reliability,
and posterior direction probabilities are not p-values. See the
[state-distribution documentation](docs/STATE_DENSITY.md) (the algorithm review of 2026-10-09 cited here at
the time is no longer part of the repository).

### Paper and reproducibility assets

Twelve case studies were rerun and checked for identical cells, clone membership,
capture counts and display eligibility. Publication display coordinates are
retained; corrected statistics and programme annotations are regenerated.
The LARRY future-state comparison now includes RNA mean plus variance.
Figure 4B is the original cell SPRING layout; Figure 4C is Threadfin, and the
map comparison alone does not establish method superiority. Figure 2's reporter
and SHM interpretation is supported by conditional and within-family checks,
with capture and physical-library limitations stated.
