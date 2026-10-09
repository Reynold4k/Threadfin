# Changelog

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
