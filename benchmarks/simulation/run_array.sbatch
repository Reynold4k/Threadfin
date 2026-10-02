#!/bin/bash
#SBATCH --job-name=tf_sim
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-109%40
#SBATCH --output=logs/sim_%A_%a.out
# Threadfin v4 simulation benchmark: one task per line of tasks.txt
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
cd "$SLURM_SUBMIT_DIR"
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.txt)
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${TASK} on $(hostname)"
$PY run_simulation_benchmark.py ${TASK}
echo "[$(date)] done"
