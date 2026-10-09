# Verified compatibility

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

See [Troubleshooting](TROUBLESHOOTING.md) for examples. Missing measured
state, probe or time labels remain missing observations; they are not filled
as negative measurements.

## Verification limits

This verifies the tested Linux dependency stacks and documented example paths.
It does not establish Windows/macOS compatibility, minimum dependency-version
support, Python 3.13+ support or every real-data configuration. For reproducible
legacy-v3 reclustering, use the tested Python 3.10/3.11 stack or retain the
computed labels. Record package versions alongside scientific outputs.
