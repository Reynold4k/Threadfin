#!/bin/bash
#SBATCH --job-name=tf_discovery
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=discovery_%j.out
set -euo pipefail

# Usage: sbatch run_discovery.sbatch <config.json> [<config2.json> ...]
VENV=/data/scratch/projects/punim1236/threadfin_data/venv/bin/python
HERE=/data/scratch/projects/punim1236/Threadfin/benchmarks/discovery
cd "$HERE"
for cfg in "$@"; do
  echo "[sbatch] $(date) running $cfg"
  "$VENV" run_discovery.py "$cfg"
done
echo "[sbatch] $(date) all done"
