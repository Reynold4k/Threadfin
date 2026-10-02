#!/bin/bash
#SBATCH --job-name=tf_sim_all
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sim_all_%j.out
# Threadfin v4 simulation benchmark on one 8-CPU node: 4 tasks in parallel, 2 threads each.
set -uo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
cd "$SLURM_SUBMIT_DIR"
PY=/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python
grep -v '^scaling' tasks.txt | xargs -P 4 -I{} sh -c "$PY run_simulation_benchmark.py {} > logs/task_\$(echo {} | tr ' ' '_').log 2>&1 && echo 'ok {}' || echo 'FAILED {}'"
grep '^scaling' tasks.txt | while read -r t; do $PY run_simulation_benchmark.py $t > "logs/task_$(echo $t | tr ' ' '_').log" 2>&1 && echo "ok $t" || echo "FAILED $t"; done
echo "[$(date)] all done"
