#!/bin/bash
# Annotate the deposited 10x contigs of GSE203132 (heavy AND light chains) with IgBLAST, one AIRR table per mouse.
#
# The deposit's AIRR table (GSE203132_10X_clone-pass_germ-pass.tsv) holds heavy chains only, but each mouse's
# filtered_contig.fasta holds every assembled contig. IgBLAST 1.22.0 (NCBI release) with the IMGT mouse IG
# reference directory (V, D, J of IGH, IGK, IGL; built with edit_imgt_file.pl + makeblastdb) annotates them.
#
# Usage: IGBLAST_DIR=<ncbi-igblast-1.22.0> IGBLAST_DB=<dir with mouse_V/D/J> DATA=<GSE203132 dir> \
#        bash annotate_contigs.sh <out_dir>
set -euo pipefail
OUT=$1; mkdir -p "$OUT"
export IGDATA=$IGBLAST_DIR
for fa in "$DATA"/GSM*_M*_filtered_contig.fasta.gz; do
  mouse=$(basename "$fa" | cut -d_ -f2)
  zcat "$fa" > "$OUT/$mouse.fasta"
  "$IGBLAST_DIR/bin/igblastn" -germline_db_V "$IGBLAST_DB/mouse_V" -germline_db_D "$IGBLAST_DB/mouse_D" \
      -germline_db_J "$IGBLAST_DB/mouse_J" -auxiliary_data "$IGBLAST_DIR/optional_file/mouse_gl.aux" \
      -organism mouse -ig_seqtype Ig -domain_system imgt -outfmt 19 -num_threads "${THREADS:-8}" \
      -query "$OUT/$mouse.fasta" -out "$OUT/${mouse}_airr.tsv"
  echo "$mouse: $(($(wc -l < "$OUT/${mouse}_airr.tsv") - 1)) contigs annotated"
done
