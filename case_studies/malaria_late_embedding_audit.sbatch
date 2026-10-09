#!/bin/bash
#SBATCH --job-name=mal_late_embed_audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin/case_studies/results/malaria_late_embedding_audit/audit_%A.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin/case_studies/results/malaria_late_embedding_audit/audit_%A.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ROOT=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
cd "$ROOT"
export THREADFIN_DATA=/data/scratch/projects/punim1236/threadfin_data
mkdir -p "$ROOT/case_studies/results/malaria_late_embedding_audit"
"$PY" case_studies/malaria_late_embedding_audit.py
