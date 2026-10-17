# Changelog

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
and posterior direction probabilities are not p-values. See
[the algorithm review](docs/ALGORITHM_REVIEW_2026-10-09_zh.md) and
[state-distribution documentation](docs/STATE_DENSITY.md).

### Paper and reproducibility assets

Twelve case studies were rerun and checked for identical cells, clone membership,
capture counts and display eligibility. Publication display coordinates are
retained; corrected statistics and programme annotations are regenerated.
The LARRY future-state comparison now includes RNA mean plus variance.
Figure 4B is the original cell SPRING layout; Figure 4C is Threadfin, and the
map comparison alone does not establish method superiority. Figure 2's reporter
and SHM interpretation is supported by conditional and within-family checks,
with capture and physical-library limitations stated.
