#!/bin/bash
#SBATCH --job-name=rbd_embed_gallery
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin/case_studies/results/mouse_rbd_embedding_audit/gallery_%A.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin/case_studies/results/mouse_rbd_embedding_audit/gallery_%A.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ROOT=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
cd "$ROOT"
export THREADFIN_DATA=/data/scratch/projects/punim1236/threadfin_data
"$PY" case_studies/mouse_rbd_embedding_audit_gallery.py
