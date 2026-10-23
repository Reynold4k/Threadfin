#!/usr/bin/env Rscript
# Export the public GSE266219 processed Seurat objects to a common sparse
# count matrix plus cell metadata.  Inputs are double-gzipped by GEO; the
# outer layer is unpacked by the caller, leaving an inner RDS gzip stream.
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(SeuratObject))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: export_clonotrace_nsclc.R INPUT_DIR OUT_DIR")
input <- normalizePath(args[[1]]); out <- args[[2]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
files <- sort(list.files(input, pattern = "^GSE266219_Merged_p[0-9]+_CD8_sorted_and_PBMC_extracted_inner.rds.gz$", full.names = TRUE))
if (length(files) != 10) stop("Expected 10 public patient objects; found ", length(files))

objects <- lapply(files, function(path) {
  patient <- sub("^.*Merged_(p[0-9]+)_CD8.*$", "\\1", basename(path))
  x <- readRDS(path)
  counts <- GetAssayData(x, assay = "RNA", layer = "counts")
  if (is.null(counts)) stop("No RNA counts layer: ", path)
  meta <- x@meta.data[colnames(counts), , drop = FALSE]
  stopifnot(identical(colnames(counts), rownames(meta)))
  keep <- !is.na(meta$cdr3s_aa) & nzchar(as.character(meta$cdr3s_aa)) &
    !is.na(meta$PtCycle) & nzchar(as.character(meta$PtCycle))
  counts <- counts[, keep, drop = FALSE]
  meta <- meta[keep, , drop = FALSE]
  ids <- paste(patient, colnames(counts), sep = "|")
  colnames(counts) <- ids; rownames(meta) <- ids
  meta$patient <- patient
  meta$cell_id <- ids
  cat(patient, ncol(counts), "TCR-annotated cells\n")
  list(counts = counts, meta = meta)
})

# Seurat objects share almost all features but order is not assumed.  Retain
# exact common symbols so every exported cell is represented in one gene space.
genes <- Reduce(intersect, lapply(objects, function(z) rownames(z$counts)))
if (length(genes) < 1000) stop("Unexpectedly small common feature set: ", length(genes))
counts <- do.call(cbind, lapply(objects, function(z) z$counts[genes, , drop = FALSE]))
fields <- Reduce(intersect, lapply(objects, function(z) colnames(z$meta)))
meta <- do.call(rbind, lapply(objects, function(z) z$meta[, fields, drop = FALSE]))
meta <- meta[colnames(counts), , drop = FALSE]
stopifnot(!anyDuplicated(meta$cell_id), !anyDuplicated(genes), ncol(counts) == nrow(meta))
writeMM(counts, file.path(out, "counts.mtx"))
write.table(genes, file.path(out, "features.tsv"), quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t")
write.table(colnames(counts), file.path(out, "barcodes.tsv"), quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t")
write.csv(meta, file.path(out, "cell_metadata.csv"), row.names = FALSE, quote = TRUE)
write.csv(data.frame(patient = names(table(meta$patient)), cells = as.integer(table(meta$patient))),
          file.path(out, "export_counts_by_patient.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "R_sessionInfo.txt"))
cat(sprintf("Exported %d cells x %d common genes from %d patients\n", ncol(counts), nrow(counts), length(files)))

writeLines("All 10 patient objects exported successfully", file.path(out, "EXPORT_COMPLETE"))
