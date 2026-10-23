#!/bin/bash
#SBATCH --account=punim1236
#SBATCH --partition=long
#SBATCH --job-name=tf_ep
#SBATCH --time=31-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/data/scratch/projects/punim1236/AAA_chen_here/tmp/claude-17221/-data-scratch-projects-punim1236/a4e79915-7eab-43db-9192-bd02add3f2b8/scratchpad/tuner/ep_%j_%x.out
cd /data/scratch/projects/punim1236/AAA_chen_here/tmp/claude-17221/-data-scratch-projects-punim1236/a4e79915-7eab-43db-9192-bd02add3f2b8/scratchpad/tuner
export PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python epochs_test.py $1 $2
