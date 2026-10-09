#!/bin/bash
#SBATCH --job-name=tf_fig456
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/figures_%j.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/paper_673503/figures_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MPLBACKEND=Agg
cd /data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
export PYTHONPATH="$PWD"
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python paper/figure_plan/make_figures.py --figures 4 5 6
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python paper/figure_plan/make_supplementary.py --figures 10 11 12 13 14 15 16

/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/docx_env/bin/python paper/export_manuscript.py
