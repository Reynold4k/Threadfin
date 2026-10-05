# Troubleshooting Threadfin inputs

## `No BCR barcodes match adata.obs_names`

The BCR table and `adata.obs_names` use different barcode formats. Compare a
few values from each. Multi-sample AnnData objects often add a sample prefix
or suffix that is absent from Cell Ranger output. Read the BCR file with, for
example, `tf.read_bcr(path, barcode_prefix="sample1_")`, or make the two
indexes identical before `tf.run`. Do not remove the donor key to force a
match: clone definitions remain donor-specific.

## `Matched BCR barcodes contain no usable heavy-chain clone definition`

Threadfin defines clones from heavy-chain V and J calls plus a non-empty
junction/CDR3. Check that the input is a BCR/VDJ annotation, that productive
IGH contigs are retained, and that its columns are present. A cell with only a
light chain is retained as an annotation but cannot define a clone.

## Missing `obs` columns

The names passed as `donor_key`, `sample_key`, `state_key`, `time_key`,
`batch_key`, or `test` must be columns in `adata.obs`. Inspect
`adata.obs.columns`, correct the spelling, or omit an optional argument.
`donor_key` is strongly recommended when defining clones from BCR data so
identical receptors from different people are never joined.

Every value in a provided `donor_key`, `sample_key`, or `batch_key` must also
be present and non-blank. Threadfin uses these labels to define groups; it
will not turn unknown values into a shared string label. Fill the metadata or
remove those cells before analysis. Missing values in `state_key`, `test`, and
`time_key` remain valid observation-level missing data.

## Too few expanded clones

Clone profiles require at least three clones with at least two cells by
default. Singletons remain in the AnnData object but cannot estimate a
within-clone component. Add more BCR-matched cells, analyze a larger dataset,
or lower `min_cells` only when one-cell clone profiles are scientifically
defensible for the analysis.

## Small or sparse expression matrices

When no `basis` is supplied, Threadfin builds a PCA embedding from finite,
non-negative counts in `adata.X` or a requested layer. It automatically caps
PCA dimensions to the available matrix dimensions. If no non-receptor variable genes remain,
use a larger expression matrix, revisit gene filtering, or pass an existing
cell-by-feature embedding through `basis`. The embedding must have one finite
row per cell.

## Optional dependency errors

Harmony integration needs `harmonypy`; install it with `pip install
harmonypy`, or call `tf.pp.prepare_embedding(..., integrate=None)`. The
legacy sequence-alignment features need the optional `seq` extra:
`pip install "threadfin[seq]"`.

An installation smoke test that uses system-site packages verifies that the
wheel is importable and runnable, but it does not verify a clean dependency
resolution. Use a fresh environment without inherited packages to test that
separately.

## Legacy `clonotype_recluster` reproducibility on Python 3.12

The legacy v3 `tf.clonotype_recluster` path can assign different Leiden
labels across repeated calls with the same `random_state` on the current
Python 3.12 dependency stack (`igraph` 1.0.0 and `leidenalg` 0.12.0). This
does not affect the v4 `tf.run` workflow. For a reproducible legacy v3
analysis, use the tested Python 3.10 or 3.11 environment, or record and reuse
the resulting clone labels rather than rerunning the clustering.
