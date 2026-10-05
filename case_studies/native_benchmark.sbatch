#!/bin/bash
#SBATCH --job-name=threadfin_benisse_native
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/native_benchmark_%A_%a.out
#SBATCH --error=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/native_benchmark_%A_%a.err

set -euo pipefail
module load GCC/11.3.0 OpenMPI/4.1.4 R/4.3.2
# R's module pulls a Python-3.10 SciPy path; this task's environment is 3.11.
unset PYTHONPATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ROOT=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
DATA=/data/scratch/projects/punim1236/threadfin_data
DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/case_studies/native_benchmark_datasets.txt")
STAGE=${STAGE:-encode}
cd "$ROOT"
export THREADFIN_DATA="$DATA"
export R_LIBS_USER=/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/internal_validation/Rlibs
/usr/bin/time -v "$PY" case_studies/native_benchmark.py "$DATASET" --stage "$STAGE" --python "$PY"
