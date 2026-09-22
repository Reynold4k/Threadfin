#!/usr/bin/env bash
# Download the public benchmark dataset (Stephenson et al. 2021, Nature Medicine;
# paired GEX+BCR COVID-19 PBMC, 5,000 BCR-containing cells, prepared by scirpy).
set -euo pipefail
OUT="${1:-$(dirname "$0")/../data}"
mkdir -p "$OUT"
cd "$OUT"
curl -sSL -o stephenson2021_5k.h5mu \
  "https://exampledata.scverse.org/scirpy/stephenson2021_5k.h5mu"
echo "6ea26f9d95525371ff9028f8e99ed474  stephenson2021_5k.h5mu" | md5sum -c
echo "Done: $OUT/stephenson2021_5k.h5mu"
