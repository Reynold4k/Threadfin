#!/bin/bash
#SBATCH --account=punim1236
#SBATCH --partition=long
#SBATCH --time=31-00:00:00
#SBATCH --job-name=gut_allB_tf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=allB_threadfin/run_%j.log

# Threadfin on every B-lineage cell of GSE284622 (activated/non-GC B + GC B + plasma).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"   # submit from gut_gc_candidates/
mkdir -p allB_threadfin
export PYTHONNOUSERSITE=1 THREADFIN_DATA=/data/scratch/projects/punim1236/threadfin_data
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python allB_threadfin_run.py
