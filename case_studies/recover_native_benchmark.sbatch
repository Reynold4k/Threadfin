#!/bin/bash
#SBATCH --job-name=threadfin_bigcn_recovery
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/native_benchmark_retry_%j.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/native_benchmark_retry_%j.err

set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONPATH
ROOT=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
export THREADFIN_DATA=/data/scratch/projects/punim1236/threadfin_data
cd "$ROOT"
/usr/bin/time -v "$PY" case_studies/recover_native_benchmark.py mouse_rbd --original-job 32288767_2
