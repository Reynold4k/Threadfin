#!/usr/bin/env Rscript
# Clonotrace (Fu et al. 2025, bioRxiv 10.1101/2025.09.01.673503), run as in its vignette (version 1.2.1).
#
#   1. cell kNN graph on the embedding (k = 30, connectivity) -> transition matrix
#   2. clone labels of the clones with >= min_size cells (others unlabelled), label spreading with
#      bootstrap (alpha 0.6, sample rate 0.8, 48 replicates, blocked backend)
#   3. cells with bootstrap deviance >= 0.3 removed, cell-to-clone probabilities kept at 90 % row and
#      column mass and renormalised
#   4. clone distances on the cell graph, nearest-neighbour approximation (the vignette default for
#      large data); `revised = TRUE` (the 1.2.1 default) blends the distance with the clone-centroid
#      distance, and the unrevised (published) distance is returned alongside in `dis_clonotrace`
#      (the vignette passes `revised = FALSE` to reproduce its published figure); clones whose cells all
#      fall below the 0.1 probability threshold are removed and counted
#   5. Leiden clustering + UMAP of clones (k = 20, resolution 0.5), as in the vignette
#
# Steps 2 and 4 run on `workers` parallel R sessions (future::plan(multisession)), as the vignette sets the
# plan before both label_spreading_bootstrap() and clone_disance(). The first full-data runs of this
# benchmark switched back to one session before step 4; their distances are identical (the approximate
# distance is deterministic) but their clone-distance times are single-process. run.json records
# `distance_workers` for every run made with this version.
#
# With `exact` as the 8th argument, step 4 uses the vignette's exact optimal-transport distance instead
# (`exact = TRUE`, disk backend, `force_exact = TRUE`, checkpoints in <graph dir>/exact_ot_checkpoints; still
# `revised = TRUE`, the unrevised exact distance in `dis_clonotrace`). This is the call of the exact-OT runs of
# 2026-10-25 (its arguments are stored in each run's graph_k*/clone_graph_dis_exact.rds); run.json then
# records "distance": "exact OT".
#
# Note: the memory and CPU that /usr/bin/time reports for this script cover the main R session only; the
# parallel sessions are not waited-for children of it, so whole-job peaks come from the scheduler (sacct).
#
# Usage: Rscript run_clonotrace.R <dataset> <variant> <min_size> <workers> <inputs_dir> <out_dir> [graph_k ladder] [exact]
#   graph_k ladder: comma-separated graph sizes to try for clone distances, default "10,30,50"

suppressPackageStartupMessages({
  library(Clonotrace); library(Matrix); library(future); library(future.apply); library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]; variant <- args[2]; min_size <- as.integer(args[3]); workers <- as.integer(args[4])
inputs <- file.path(args[5], dataset); out <- args[6]
ladder <- if (length(args) >= 7) as.integer(strsplit(args[7], ",")[[1]]) else c(10L, 30L, 50L)
exact_ot <- length(args) >= 8 && identical(args[8], "exact")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(future.globals.maxSize = 64 * 1024^3)
set.seed(0)

read_npy <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  magic <- readBin(con, "raw", 6); ver <- readBin(con, "raw", 2)
  hlen <- if (as.integer(ver[1]) == 1L) readBin(con, "integer", 1, size = 2, endian = "little", signed = FALSE)
          else readBin(con, "integer", 1, size = 4, endian = "little")
  header <- rawToChar(readBin(con, "raw", hlen))
  shape <- as.integer(strsplit(gsub(".*'shape': \\(([0-9, ]+)\\).*", "\\1", header), ",\\s*")[[1]])
  stopifnot(grepl("'<f4'", header), grepl("'fortran_order': False", header))
  v <- readBin(con, "numeric", prod(shape), size = 4, endian = "little")
  matrix(v, nrow = shape[1], ncol = shape[2], byrow = TRUE)
}

stamp <- list(); t_all <- Sys.time(); graph_k_used <- NA; graph_attempts <- character()
tick <- function(name, t0) stamp[[name]] <<- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 3)
write_status <- function(status, extra = list()) {
  rec <- c(list(status = status, method = "clonotrace", software = paste("Clonotrace", packageVersion("Clonotrace")),
                dataset = dataset, variant = variant, min_size = min_size, workers = workers,
                seconds = stamp, R = R.version.string), extra)
  writeLines(jsonlite_like(rec), file.path(out, "run.json"))
}
jsonlite_like <- function(x) {   # small JSON writer (jsonlite is not installed in this environment)
  enc <- function(v) {
    if (is.list(v)) {
      if (!is.null(names(v)) && all(names(v) != "")) {
        paste0("{", paste(sprintf("\"%s\": %s", names(v), vapply(v, enc, "")), collapse = ", "), "}")
      } else paste0("[", paste(vapply(v, enc, ""), collapse = ", "), "]")
    } else if (is.character(v)) {
      if (length(v) == 1) sprintf("\"%s\"", gsub("\"", "'", v)) else paste0("[", paste(sprintf("\"%s\"", v), collapse = ", "), "]")
    } else if (length(v) == 1) { if (is.na(v)) "null" else as.character(v) }
    else paste0("[", paste(v, collapse = ", "), "]")
  }
  enc(x)
}

t0 <- Sys.time()
X <- read_npy(file.path(inputs, "X.npy"))
zread <- function(path, ...) fread(cmd = sprintf("zcat '%s'", path), sep = "\t", ...)   # fread needs R.utils for .gz
cells <- zread(file.path(inputs, "cells.tsv.gz"), colClasses = list(character = c("cell", "clone", "donor", "sample")))
rownames(X) <- cells$cell
if (startsWith(variant, "half")) {          # other variant names (full, full_w8, ...) are the full data
  half <- substr(variant, 5, 5); seed <- sub(".*_s", "", variant)
  split <- zread(file.path(inputs, sprintf("split_seed%s.tsv.gz", seed)), colClasses = list(character = c("cell", "half")))
  keep <- cells$cell %in% split$cell[split$half == half]
  X <- X[keep, , drop = FALSE]; cells <- cells[keep]
}
lab <- cells$clone; lab[lab == ""] <- NA
sizes <- table(lab)
expanded <- names(sizes)[sizes >= min_size]
labels <- ifelse(lab %in% expanded, lab, NA_character_)
tick("load", t0)
if (length(expanded) < 3) {
  write_status("failed", list(reason = sprintf("only %d clones with >= %d cells", length(expanded), min_size),
                              n_clones_eligible = length(expanded)))
  quit(save = "no", status = 0)
}

result <- tryCatch({
  t0 <- Sys.time()
  cell_knn <- embedding2knn(embedding = X, k = 30, mode = "connectivity", if_self = FALSE)
  cell_knn <- compute_transition(cell_knn)
  tick("cell_graph", t0)

  t0 <- Sys.time()
  plan(multisession, workers = workers)
  spread <- label_spreading_bootstrap(adj = cell_knn, labels = labels, alpha = 0.6, sample_rate = 0.8,
                                      sample_n = 48, backend = "blocked", block_size = 128,
                                      tmpdir = file.path(out, "tmp"), verbose = FALSE)
  prob <- spread[[1]]; deviance <- spread[[2]]
  rownames(prob) <- cells$cell
  colnames(prob) <- levels(as.factor(labels))
  tick("label_spreading", t0)

  t0 <- Sys.time()
  prob <- prob[deviance < 0.3, , drop = FALSE]
  prob <- prob / rowSums(prob)
  prob <- prob[is.finite(rowSums(prob)), , drop = FALSE]
  prob <- mat_sparsify(mat = prob, row_mass = 0.9, col_mass = 0.9)
  prob <- prob / rowSums(prob)
  prob <- prob[is.finite(rowSums(prob)) & rowSums(prob) > 0, , drop = FALSE]
  prob <- Matrix(prob, sparse = TRUE)
  mass <- Matrix::colSums(prob)
  lost <- colnames(prob)[mass <= 0]
  prob <- prob[, mass > 0, drop = FALSE]
  # clone_disance() (approximate mode) stops if any clone has no cell with probability >= 0.1
  # (its prob_thresh). Small clones in crowded regions can lose all such cells after propagation
  # and sparsification; they are removed here and counted as lost coverage.
  prob <- methods::as(prob, "CsparseMatrix")
  colmax <- vapply(seq_len(ncol(prob)), function(j) {
    if (prob@p[j + 1] > prob@p[j]) max(prob@x[(prob@p[j] + 1):prob@p[j + 1]]) else 0
  }, numeric(1))
  lost_thresh <- colnames(prob)[colmax < 0.1]
  prob <- prob[, colmax >= 0.1, drop = FALSE]
  if (ncol(prob) < 3) stop(sprintf("only %d clones keep a cell with probability >= 0.1", ncol(prob)))
  tick("sparsify", t0)

  t0 <- Sys.time()
  emb <- X[rownames(prob), , drop = FALSE]
  observed <- data.frame(cell = cells$cell, clone = labels)[!is.na(labels), ]
  donor <- tapply(cells$donor[!is.na(labels)], labels[!is.na(labels)], function(v) v[1])
  clone_sample <- donor[colnames(prob)]
  # package defaults: centroids from the cell-to-clone probabilities (clone_labels = NULL) and no
  # sample restriction (sample = NULL), exactly as clone_disance() is called in the vignette. The default
  # cell graph (graph_k = 10) can be disconnected; Clonotrace then stops and recommends a larger graph_k,
  # which is what is done here (30, then 50 by default; a longer ladder can be given), and recorded.
  dis <- NULL; attempts <- character()
  for (gk in ladder) {
    gdir <- file.path(out, sprintf("graph_k%d", gk))
    attempt <- if (exact_ot) {
      tryCatch(clone_disance(emb, prob, outpath = gdir, graph_k = gk, exact = TRUE, force_exact = TRUE,
                             revised = TRUE, checkpoint_dir = file.path(gdir, "exact_ot_checkpoints"),
                             verbose = FALSE), error = function(e) e)
    } else {
      tryCatch(clone_disance(emb, prob, outpath = gdir, graph_k = gk,
                             exact = FALSE, revised = TRUE, verbose = FALSE), error = function(e) e)
    }
    if (!inherits(attempt, "error")) { dis <- attempt; graph_k_used <<- gk; break }
    attempts <- c(attempts, sprintf("graph_k=%d: %s", gk, conditionMessage(attempt)))
    if (!grepl("disconnected", conditionMessage(attempt))) stop(conditionMessage(attempt))
  }
  plan(sequential)
  if (is.null(dis)) stop(paste(attempts, collapse = " | "))
  graph_attempts <<- attempts
  tick("clone_distance", t0)

  t0 <- Sys.time()
  sq <- long2square(long = as.data.frame(dis)[, c("group1", "group2", "dis")], row_names_from = "group1",
                    col_names_from = "group2", values_from = "dis", symmetric = TRUE)
  sq <- as.matrix(sq); diag(sq) <- 0
  # long2square returns the clones in the order of the long table's identifiers
  if (is.null(rownames(sq))) rownames(sq) <- colnames(sq) <- colnames(prob)
  kk <- min(20L, nrow(sq) - 1L)
  clusters <- leiden_dis(dismat = sq, k = kk, resolution = 0.5, if_umap = TRUE)
  tick("leiden_umap", t0)

  fwrite(as.data.frame(dis), file.path(out, "clonotrace_distance_long.tsv.gz"), sep = "\t")
  fwrite(data.frame(clone = colnames(prob), mass = as.numeric(Matrix::colSums(prob)),
                    n_cells = as.integer(sizes[colnames(prob)]), donor = as.character(clone_sample)),
         file.path(out, "clones_clonotrace.tsv"), sep = "\t")
  fwrite(cbind(clone = rownames(clusters), as.data.frame(clusters)), file.path(out, "clone_umap_leiden.tsv"), sep = "\t")
  list(n_eligible = length(expanded), n_lost = length(lost), n_lost_thresh = length(lost_thresh), n_out = ncol(prob),
       n_cells_kept = nrow(prob), n_cells = nrow(X),
       mean_weight = if (is.null(attr(dis, "mean_weight"))) NA else attr(dis, "mean_weight"))
}, error = function(e) e)

unlink(file.path(out, "tmp"), recursive = TRUE)
if (inherits(result, "error")) {
  write_status("failed", list(reason = conditionMessage(result), n_clones_eligible = length(expanded)))
} else {
  write_status("completed", c(list(n_clones_eligible = result$n_eligible, n_clones_lost_after_filtering = result$n_lost,
                                   n_clones_lost_below_prob_thresh = result$n_lost_thresh,
                                   n_clones = result$n_out, n_cells_input = result$n_cells,
                                   n_cells_after_deviance_filter = result$n_cells_kept,
                                   revised_mean_weight = result$mean_weight,
                                   graph_k_used = graph_k_used, distance_workers = workers),
                              if (exact_ot) list(distance = "exact OT"),
                              list(graph_k_failed_attempts = if (length(graph_attempts)) graph_attempts else "none",
                                   total_seconds = round(as.numeric(difftime(Sys.time(), t_all, units = "secs")), 3))))
}
