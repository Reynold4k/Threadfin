#!/bin/bash
#SBATCH --account=punim1236
#SBATCH --partition=long
#SBATCH --job-name=tf_valid5
#SBATCH --time=31-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --output=/data/scratch/projects/punim1236/AAA_chen_here/tmp/claude-17221/-data-scratch-projects-punim1236/a4e79915-7eab-43db-9192-bd02add3f2b8/scratchpad/tuner/valid5_%j.out
cd /data/scratch/projects/punim1236/AAA_chen_here/tmp/claude-17221/-data-scratch-projects-punim1236/a4e79915-7eab-43db-9192-bd02add3f2b8/scratchpad/tuner
export PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python validate5.py
