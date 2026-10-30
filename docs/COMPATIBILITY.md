# Verified compatibility

## Version 4.6.0 — 30 October 2026

* **The user's path.** A new environment was built as Tutorial 2 describes (Python 3.11, `pip install
  "threadfin[dandelion]"` from this version: pynndescent 0.6.0, numba 0.68.1, scanpy 1.11.5, sc-dandelion 0.5.7,
  matplotlib 3.8.4), and `tf.run` was called on the same 16 datasets, with `strata_key="sample"` for the 14 BCR
  datasets: no error, and every clone table — clone calls, reliabilities, clone-map coordinates and programmes — is
  identical, byte for byte, to that of 4.5.2, as is every clonal ICC. The shuffled ICC of the coherence test now
  comes from shuffles within samples; every test that was significant stays at P = 0.005 (200 permutations). In
  GSE171703 three libraries pool several donors; the strata there are donor × library (a fix this run found).
  Every code block of Tutorials 1, 3 and 4 ran (23 of 23) and printed what the tutorials show, apart from timings.
* **The tutorial notebook** ran end to end in a separately built conda environment without `PYTHONHASHSEED`:
  no error, 5.6 minutes on 8 cores including the download, 7.2 GB; the same 8,736 clones, 1,881 of them profiled,
  and the same ICC (0.169) as with 4.5.3. Every package version of that run is in
  `docs/tutorials/requirements-lock.txt`.
* **Tests.** In that environment 359 passed and 1 was skipped (scirpy was not installed). With the newest releases
  of the dependencies (Python 3.12, pandas 3.1, numpy 2.5, anndata 0.13, scanpy 1.12), 345 passed and 15 that need
  Dandelion or scirpy were skipped; continuous integration passed on Python 3.10–3.14, macOS, Windows, the oldest
  allowed dependencies and with Dandelion on Python 3.12–3.14.
* **Python 3.13 and 3.14 with Dandelion.** The Dandelion extra no longer asks for matplotlib < 3.9 on Python 3.13
  and later, where importing Threadfin restores the `matplotlib.cm.get_cmap` that Dandelion 0.5.7 needs. With
  sc-dandelion 0.5.7 and matplotlib 3.11.2, clone calls on three datasets (GSE246382, GSE287123 RBD, GSE286215
  d0-14) are identical on Python 3.11, 3.13 and 3.14, and identical with and without `PYTHONHASHSEED=0`; the
  Dandelion job of continuous integration installs `threadfin[dandelion]` on Python 3.12, 3.13 and 3.14.

## Version 4.5.3 — 28 October 2026

* **Why this version.** Every tutorial was followed again in new environments that saw nothing of the testing
  account (no settings, caches or packages; Python packages from PyPI, tutorial files from GitHub). The notebook
  stopped at Step 8: pynndescent 0.7.0, released the day before, returns sparse arrays that scanpy 1.11.5 cannot
  read (`sc.pp.neighbors` on 8,192 or more cells). On the same 60,000 simulated cells and seed, pynndescent 0.7.0
  also returned other approximate neighbours than 0.6.0 (22% of the rows of the 15-neighbour graph identical) and
  another UMAP layout; with pynndescent 0.6.0, numba 0.68.0 and 0.68.1 gave identical results. Threadfin now
  requires pynndescent < 0.7.
* **The release candidate.** A new environment (Python 3.11 from conda-forge only, `pip install
  "threadfin[dandelion]"` from this version: pynndescent 0.6.0, numba 0.68.1, scanpy 1.11.5, sc-dandelion 0.5.7,
  matplotlib 3.8.4) gave clone tables and summaries identical, byte for byte, to those of 4.5.2 for the three
  datasets whose smoothing uses pynndescent (61,423, 72,359 and 115,144 cells; the last also lays out 6,583 clones,
  for which UMAP uses pynndescent too); the tests passed (268, 1 skipped without scirpy);
  every code block of Tutorials 1, 3 and 4 ran (22 of 22); the regenerated notebook ran end to end (7.8 minutes on
  8 cores including the download, 7.6 GB) and printed the same numbers as with 4.5.2.
* **Python 3.14.** `pip install threadfin` installed on Python 3.14 (numpy 2.5, pandas 3.1, anndata 0.13, scanpy
  1.12, matplotlib 3.11); every code block of Tutorials 1, 3 and 4 and `examples/tutorial.py` ran and printed the
  same numbers as on Python 3.11. `pip install "threadfin[dandelion]"` failed there, because matplotlib < 3.9,
  which Dandelion 0.5 needs, has no packages for Python 3.13 or later: the Dandelion extra needs Python 3.10–3.12.

## Version 4.5.2 — 27 October 2026

* **The user's path.** A new environment was built as Tutorial 2 describes (Python 3.11,
  `pip install "threadfin[dandelion]"` from this version: sc-dandelion 0.5.7, numpy 2.4, pandas 2.3, anndata 0.12,
  scanpy 1.11, matplotlib 3.8), and `tf.run` was called on the same 16 datasets: every clone table is identical,
  byte for byte, to that of 4.5.1; the summaries differ only in the sentence about a single programme (3 datasets),
  and the Methods paragraphs only in the version numbers. Every code block of Tutorials 1, 3 and 4 ran (22 of 22).
* **The tutorial notebook** ran end to end in a separately built conda environment: no error, 6.0 minutes on 8
  cores, 7.6 GB; every printed number is the same as with 4.5.1.
* **Tests.** In that environment 268 passed and 1 was skipped (scirpy was not installed). With Python 3.12,
  sc-dandelion 0.5.7 and scirpy 0.25.1, the setting of the new continuous-integration job, all 269 passed and the
  tutorial on simulated data ran.

## Version 4.5.1 — 25 October 2026

* **The user's path, repeated with Dandelion pinned below 1.0.** A new environment was built as Tutorial 2
  describes (Python 3.11, `pip install "threadfin[dandelion]"`, which now installs sc-dandelion 0.5.7 rather than
  1.0; numpy 2.4, pandas 2.3, scipy 1.17, scikit-learn 1.9, anndata 0.12, scanpy 1.11, umap-learn 0.5.12, harmonypy
  2.1, matplotlib 3.8), and `tf.run` was called on the same 16 datasets: every clone table is identical, byte for
  byte, to that of 4.5.0, and the summaries differ only where 4.5.0 wrote "1 cells". Every code block of Tutorials
  1, 3 and 4 ran (22 of 22).
* **The tutorial notebook** ran end to end in a separately built conda environment: no error, 5.5 minutes on 8
  cores, 7.2 GB; 8,736 clones, as with Dandelion 0.5.x before.
* **Tests.** 266 passed; 1 skipped (scirpy was not installed in that environment).
* **Continuous integration** (Linux with Python 3.10–3.14 and with the oldest allowed dependencies, macOS,
  Windows): passed.

## Version 4.5.0 — 24 October 2026

* **The user's path on real data.** A new environment was built as Tutorial 2 describes (Python 3.11,
  `pip install "threadfin[dandelion]"` from this version: numpy 2.4, pandas 2.3, scipy 1.17, scikit-learn 1.9,
  anndata 0.12, scanpy 1.11, umap-learn 0.5.12, harmonypy 2.1, sc-dandelion 0.5.7, matplotlib 3.8), and `tf.run`
  was called on the 14 public BCR datasets of the case studies and the two LARRY datasets with every setting at
  its default, followed by `summary()`, `methods()`, `plot()` and saving: no dataset raised an error, and every
  clone count and clonal ICC equals that of 4.4.2. Every code block of Tutorials 1, 3 and 4 ran (22 of 22)
  ([`tutorial_path`](../manuscript/analyses/technical_validation/tutorial_path/README.md)).
* **The tutorial notebook** (`docs/tutorials/05_end_to_end_tutorial.ipynb`, which downloads GSE287123 from GEO)
  was executed end to end in a separately built conda environment: no error, 5.6 minutes on 8 cores, 7.6 GB.
* **Tests.** The full suite (261 tests) passed with none skipped in one environment that has both optional
  dependencies (Python 3.11, sc-dandelion 0.5.7, scirpy 0.22.5).
* **T cells and scirpy.** scirpy's example T-cell dataset ran end to end from the scirpy MuData
  ([`t_cells`](../manuscript/analyses/technical_validation/t_cells/README.md)).

## Version 4.4.1 — 23 October 2026

Continuous integration now also installs the package and runs the test suite and the tutorial on macOS and
Windows (Python 3.12). Windows found one problem: harmonypy 2.x publishes no Windows wheels and its source does
not compile with Microsoft's compiler, so the installation failed; Windows now receives the pure-Python
harmonypy (< 2). The Harmony step is tested with harmonypy 0.0.9, 0.2.0 and 2.1.0.

## Version 4.4.0 — 23 October 2026

The package was installed from the source tree with `uv` into new, isolated environments (no inherited
`PYTHONPATH` or user site-packages), and the full test suite and the tutorial on simulated data were run in
each (2 tests are skipped when an optional dependency is absent):

| Python | dependency versions | tests | tutorial |
|---|---|---|---|
| 3.10 | the **oldest** versions `pyproject.toml` allows: numpy 1.24.1, pandas 2.0.0, scipy 1.10.0, scikit-learn 1.3.0, anndata 0.10.0, scanpy 1.10.0, matplotlib 3.7.0, umap-learn 0.5.5, pynndescent 0.5.8, igraph 0.10.0, leidenalg 0.10.0, joblib 1.2.0, harmonypy 0.0.9 | 245 passed | passed |
| 3.11 | the benchmark environment (numpy 2.2, pandas 2.3, scanpy 1.11) | 245 passed | passed |
| 3.13 | current releases | 244 passed, 1 expected failure (the legacy v3 repeatability check) | passed |
| 3.14 | current releases: numpy 2.5, pandas 3.1, scipy 1.18, scikit-learn 1.9, anndata 0.13, scanpy 1.12, matplotlib 3.11, numba 0.68 | 244 passed, the same expected failure | passed |

The expected failure was a fault of the test, not of the package, and is fixed since 4.4.0: with pandas 3, `astype(str)` keeps missing values missing, and the test compared the labels of cells without a clone as NaN == NaN. `clonotype_recluster` gives identical clusters on every stack tested (issue #2).

This check found two problems, both fixed in 4.4.0: umap-learn < 0.5.5 and pynndescent < 0.5.8 import
`pkg_resources`, which new environments do not include (the minimum versions were raised), and a
manuscript script used `DataFrame.map`, which pandas 2.0 does not have. Continuous integration now runs
Python 3.10–3.14 and the oldest-versions installation for every change to `main`.

The tutorial notebook (`docs/tutorials/05_end_to_end_tutorial.ipynb`), which downloads GSE287123 from GEO, was
executed end to end in a new conda environment built as the notebook describes, with this version installed.

`n_jobs` (clone-map search and programme bootstrap) gives the same clone map and programmes as one process and
as the released 4.3.1; this is checked in `tests/test_parallel_and_robustness.py` and was checked on GSE287123.

## Version 4.3.0 — 18 October 2026

The package was installed with `pip` from the source tree into a new, isolated Python 3.11 environment
and run outside the repository: `threadfin.run()` completed on a simulated repertoire, including the
automatic choice of the clone-map settings, and the result was written to and read from `.h5ad`. The full
test suite (232 tests) passed in the development environment (Python 3.11). Continuous integration runs
the suite and the tutorial on Python 3.10, 3.11 and 3.12 for every change to `main`.

## Version 4.1.0 — 9 October 2026

The final 4.1.0 wheel was installed into the three isolated Linux environments
and tested from outside the repository. All 25 package modules matched the
source, wheel and installed copies byte-for-byte. This check found and fixed
an older AnnData serialization case in which an absent None-valued context
parameter prevented model reconstruction.

| Python | Installed-wheel full suite | Dependency check | Tutorial | Simulated quick start |
|---|---|---|---|---|
| 3.10.22 | 198 passed | Passed | Passed | Passed |
| 3.11.7 | 198 passed | Passed | Passed | Passed |
| 3.12.15 | 197 passed, 1 expected failure | Passed | Passed | Passed |

The separate current-development-environment suite also passed all 198 tests.
The Python 3.12 expected failure remains the legacy-v3 repeatability check,
not a passing test or a new state-density failure. Scientific warnings were
retained (33–37 per installed suite).
See [the 4.1.0 verification record](compatibility_results_2026-10-09.json)
for hashes, dependencies and every stage exit code.

## Historical check — 6 October 2026

Checked on 6 October 2026 using clean Linux environments without inherited
system-site packages. Tests imported the installed wheel from `site-packages`
outside the repository. All 23 package Python modules were also
compared byte-for-byte with the wheel and all three installed copies.

| Python | Full test suite | Dependency check | Tutorial | Simulated quick start |
|---|---|---|---|---|
| 3.10.22 | 161 passed | Passed | Passed | Passed |
| 3.11.7 | 161 passed | Passed | Passed | Passed |
| 3.12.15 | 160 passed, 1 expected failure | Passed | Passed | Passed |

The Python 3.12 expected failure is the existing legacy-v3
`clonotype_recluster` repeatability check. It is not counted as a passing test.
The v4 `tf.run` tests and examples passed on all three environments. The suites
emit 33–37 warnings, including scientific warnings about unstable programme
partitions; those are retained rather than hidden.

All 17 reclustering-control tests passed. They cover explicit overrides,
unchanged default partitions, independent display controls, AnnData round-trip
serialization, missing clone IDs, invalid parameter/distance rejection and
exact agreement with the historical Scanpy graph/Leiden recipe, including
independent UMAP and graph/Leiden random seeds. The optional
embedding graph is tested alongside unchanged default distance partitions.

Exact dependency versions, the tested wheel's SHA-256 and stage exit codes are
in [the verification record](compatibility_results_2026-10-06.json). The same
tests, optional sequence dependencies and simulated quick start are covered
by the repository's Python 3.10/3.11/3.12 GitHub Actions matrix.

## User-facing errors

Input validation now checks requested metadata columns before analysis,
rejects missing or blank donor/sample/batch labels, reports duplicate or
unmatched BCR barcodes and unusable heavy-chain calls, and checks finite
cell-by-feature embeddings and non-negative counts. Requested test columns
are not silently skipped. PCA dimensions are capped to the selected matrix
size. Errors are in English and include concrete correction steps.

See the [troubleshooting table of Tutorial 2](tutorials/02_your_data.md#9-troubleshooting) for examples. Missing measured
state, probe or time labels remain missing observations; they are not filled
as negative measurements.

## Verification limits

This verifies the tested dependency stacks (Linux for the real-data runs; Linux, macOS and Windows in
continuous integration since 4.4.1) and the documented example paths, not every real-data configuration.
Record package versions alongside scientific outputs; since 4.6.0, `threadfin.run` records them in
`adata.uns['threadfin']['run']`.
