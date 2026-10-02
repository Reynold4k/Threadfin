#!/bin/bash
#SBATCH --job-name=tf_data
#SBATCH --partition=long
#SBATCH --time=31-00:00:00
#SBATCH --output=logs/%x_%j.out
# Threadfin v4 analysis of one public dataset: sbatch -J tf_<name> --cpus-per-task=N --mem=XG run_dataset.sbatch <name>
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] $1 on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python run_dataset.py "$1"
echo "[$(date)] done"
