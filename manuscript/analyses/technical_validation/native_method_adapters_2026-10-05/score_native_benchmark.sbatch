#!/bin/bash
#SBATCH --job-name=threadfin_clone_profiles_native
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/clone_profiles_benchmark_%A_%a.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/clone_profiles_benchmark_%A_%a.err
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
task_root=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
task_python=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
dataset=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$task_root/case_studies/native_benchmark_datasets.txt")
cd "$task_root"
/usr/bin/time -v "$task_python" case_studies/score_native_benchmark.py "$dataset" --stage profiles
/usr/bin/time -v "$task_python" case_studies/score_native_benchmark.py "$dataset" --stage clone2vec
