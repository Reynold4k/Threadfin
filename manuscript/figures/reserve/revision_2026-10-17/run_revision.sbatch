#!/bin/bash
#SBATCH --account=punim1236
#SBATCH --partition=long
#SBATCH --time=31-00:00:00
#SBATCH --job-name=tf_rev1017
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1 THREADFIN_DATA=/data/scratch/projects/punim1236/threadfin_data
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
SRC=../source
$PY "$SRC/revision_${1}_2026-10-17.py"
