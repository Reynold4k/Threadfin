#!/bin/bash
# Build one somatic-hypermutation lineage tree with GCtree.
#
#   bash run_gctree.sh <clone>          # expects <clone>.fasta in the current folder
#   ls *.fasta | sed 's/.fasta//' | xargs -P 8 -n1 bash run_gctree.sh
#
# The first record of each FASTA must be the germline root, named "naive"
# (written by case_studies/lineage.py). Results go to trees/<clone>/, where
# gctree.out.inference.1.nk is the best tree. Needs gctree and PHYLIP dnapars
# on PATH (or GCTREE_BIN set to the folder holding them).
set -u
P=${GCTREE_BIN:-$(dirname "$(command -v gctree)")}
export PATH="$P:$PATH" PYTHONNOUSERSITE=1 QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp

c=$1
d=trees/$c
mkdir -p "$d"
cd "$d" || exit 1

deduplicate "../../$c.fasta" --root naive --abundance_file abundances.csv --idmapfile idmap.txt \
  > deduplicated.phylip 2> dedup.log || { echo "$c dedup-fail"; exit 0; }
n=$(head -1 deduplicated.phylip | awk '{print $1}')
if [ "$n" -lt 3 ]; then echo "$c trivial ($n genotypes)"; exit 0; fi

# exhaustive parsimony grows quickly with the number of distinct sequences, so give it a time limit
mkconfig deduplicated.phylip dnapars > dnapars.cfg && rm -f outfile outtree
timeout "${DNAPARS_TIMEOUT:-900}" dnapars < dnapars.cfg > dnapars.log 2>&1
[ -s outfile ] || { echo "$c dnapars-timeout ($n genotypes)"; exit 0; }
gctree infer outfile abundances.csv --root naive --idmapfile idmap.txt --verbose > gctree.log 2>&1 \
  && echo "$c ok" || echo "$c gctree-fail"
