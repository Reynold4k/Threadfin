#!/bin/bash
#SBATCH --job-name=threadfin_bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.out
set -euo pipefail

REPO=/data/scratch/projects/punim1236/Threadfin
DATA=/data/scratch/projects/punim1236/threadfin_data
VENV=$DATA/venv

echo "[sbatch] node: $(hostname), start: $(date)"
"$VENV/bin/python" "$REPO/benchmarks/run_real_benchmark.py" \
    "$DATA/stephenson2021_5k.h5mu" "$REPO/benchmarks/results"
"$VENV/bin/python" "$REPO/benchmarks/run_scaling_benchmark.py" \
    "$REPO/benchmarks/results"
echo "[sbatch] done: $(date)"
