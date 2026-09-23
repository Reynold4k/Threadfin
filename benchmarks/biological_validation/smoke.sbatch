#!/bin/bash
#SBATCH --job-name=tf_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=smoke_%j.out
set -uo pipefail

VENV=/data/scratch/projects/punim1236/threadfin_data/venv/bin/python
HERE=/data/scratch/projects/punim1236/Threadfin/benchmarks/biological_validation
cd "$HERE"
echo "[sbatch] $(date) smoke testing: $*"
PYTHONPATH=. "$VENV" smoke_loaders.py "$@"
echo "[sbatch] $(date) smoke done"
