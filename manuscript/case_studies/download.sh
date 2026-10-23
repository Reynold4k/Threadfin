#!/usr/bin/env bash
# Download the public datasets used in the Threadfin case studies.
#
#   bash download.sh [DATA_DIR]        (default: $THREADFIN_DATA or ./data)
#
# The loaders in datasets.py read from the same folder (set THREADFIN_DATA to
# it). Sources and checksums are listed in datasets_manifest.tsv and
# md5_manifest.txt. About 12 GB in total.
set -uo pipefail
BASE="${1:-${THREADFIN_DATA:-$(pwd)/data}}"
mkdir -p "$BASE"

dl() {  # dl <url> <output file>; skips files that already exist, retries a few times
  local url=$1 out=$2 i
  [ -s "$out" ] && { echo "skip $(basename "$out")"; return 0; }
  for i in 1 2 3 4 5; do
    curl -sfL --retry 2 -o "$out.tmp" "$url" && [ -s "$out.tmp" ] && { mv "$out.tmp" "$out"; echo "ok   $(basename "$out")"; return 0; }
    echo "retry ($i) $(basename "$out")"; sleep $((i * 15))
  done
  echo "FAILED $(basename "$out")"; return 1
}

# 1. Human lymph node + blood after SARS-CoV-2 mRNA vaccination (Kim et al. 2022, Nature; Zenodo 5895181)
D="$BASE/gse195673_ln_vaccine"; mkdir -p "$D"
Z="https://zenodo.org/api/records/5895181/files"
dl "$Z/WU368_kim_et_al_nature_2022_meta.tsv/content" "$D/bcr_meta.tsv"
dl "$Z/WU368_kim_et_al_nature_2022_bcr_heavy.tsv.gz/content" "$D/bcr_heavy.tsv.gz"
dl "$Z/WU368_kim_et_al_nature_2022_bcr_light.tsv.gz/content" "$D/bcr_light.tsv.gz"
dl "$Z/WU368_kim_et_al_nature_2022_gex_b_cells.h5ad/content" "$D/gex_b_cells.h5ad"

# 2. Mouse germinal centres with a division reporter (Merkenschlager et al. 2025, Nature; GEO GSE287123)
D="$BASE/gse287123_np"; mkdir -p "$D/extracted"
dl "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE287nnn/GSE287123/suppl/GSE287123_RAW.tar" "$D/GSE287123_RAW.tar" \
  && tar -xf "$D/GSE287123_RAW.tar" -C "$D/extracted"

# 3. Human blood before and 7 days after influenza vaccination (Wang et al. 2023; GEO GSE175522 / GSE175523)
D="$BASE/gse175522_flu"; mkdir -p "$D"
for e in GSM5340834_120648_0 GSM5340835_120648_7 GSM5340836_120667_0 GSM5340837_120667_7 GSM5340838_141393_0 \
         GSM5340839_141393_7 GSM5340840_141394_0 GSM5340841_141394_7 GSM5340842_141409_0 GSM5340843_141409_7 \
         GSM5340844_141415_0 GSM5340845_141415_7; do
  gsm="${e%%_*}"; dl "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5340nnn/$gsm/suppl/${e}_cellranger.tar.gz" "$D/${e}_cellranger.tar.gz"
done
for e in GSM5340846_120648_0 GSM5340847_120648_7 GSM5340848_120667_0 GSM5340849_120667_7 GSM5340850_141393_0 \
         GSM5340851_141393_7 GSM5340852_141394_0 GSM5340853_141394_7 GSM5340854_141409_0 GSM5340855_141409_7 \
         GSM5340856_141415_0 GSM5340857_141415_7; do
  gsm="${e%%_*}"; dl "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5340nnn/$gsm/suppl/${e}_clone_pass_fil_airr.txt.gz" "$D/${e}_clone_pass_fil_airr.txt.gz"
done

# 4. Human tonsil organoids infected with EBV (Mitul et al. 2026, PNAS; GEO GSE317492)
D="$BASE/gse317492_ebv"; mkdir -p "$D"
S=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9473nnn
for p in "GSM9473008 d0" "GSM9473012 d4" "GSM9473016 d7" "GSM9473020 d14-gfp-neg-b" "GSM9473023 d14_gfp-pos-b" \
         "GSM9473027 d21-gfp-neg-b" "GSM9473030 d21_gfp-pos-b"; do
  set -- $p
  for part in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do dl "$S/$1/suppl/${1}_${2}_gex_${part}" "$D/${1}_${2}_gex_${part}"; done
done
for p in "GSM9473009 d0" "GSM9473013 d4" "GSM9473017 d7" "GSM9473021 d14-gfp-neg-b" "GSM9473024 d14_gfp-pos-b" \
         "GSM9473028 d21-gfp-neg-b" "GSM9473031 d21_gfp-pos-b"; do
  set -- $p
  dl "$S/$1/suppl/${1}_${2}_vdjB_filtered_contig_annotations.csv.gz" "$D/${1}_${2}_vdjB_filtered_contig_annotations.csv.gz"
done

# 5. Human paediatric tonsil (King et al. 2021, Sci Immunol; ArrayExpress E-MTAB-9005 / E-MTAB-9003)
D="$BASE/king2021_tonsil"; mkdir -p "$D"
for donor in BCP003 BCP004 BCP005 BCP006 BCP008 BCP009; do
  dl "https://www.ebi.ac.uk/biostudies/files/E-MTAB-9005/${donor}_Total_5GEX_filtered_feature_bc_matrix.tar.gz" "$D/${donor}_Total_5GEX.tar.gz"
  dl "https://www.ebi.ac.uk/biostudies/files/E-MTAB-9003/${donor}_Total_scVDJ_filtered_contigs.tar.gz" "$D/${donor}_Total_scVDJ.tar.gz"
done
dl "https://www.ebi.ac.uk/biostudies/files/E-MTAB-9005/CellTypeMetaData.txt" "$D/CellTypeMetaData.txt"

# 6. Human blood in COVID-19 (Stephenson et al. 2021, Nat Med; 5,000-cell subset prepared by scirpy)
dl "https://exampledata.scverse.org/scirpy/stephenson2021_5k.h5mu" "$BASE/stephenson2021_5k.h5mu"

echo "done: $BASE  (export THREADFIN_DATA=$BASE before running the case studies)"
