#!/bin/bash
#SBATCH --job-name=threadfin_benchmark_finalize
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/benchmark_finalize_%j.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/benchmark_finalize_%j.err
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
cd /data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python case_studies/finalize_native_benchmark.py
