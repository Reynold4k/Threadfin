#!/bin/bash
#SBATCH --job-name=tf_nsclc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/nsclc_%j.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/nsclc_%j.err
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export MPLBACKEND=Agg
cd /data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
export PYTHONPATH="$PWD"
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python -u case_studies/prepare_clonotrace_nsclc.py --stage "${1:-export}"
