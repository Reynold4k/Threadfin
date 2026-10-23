#!/bin/bash
# Run a benchmark task file in parallel on one node: run_tasks.sh <task_file> <n_parallel>
# Each line is a self-contained shell command (see make_tasks.py); failures are recorded by the tools
# themselves in run.json and do not stop the other tasks.
set -u
TASKS="$1"; NPAR="${2:-8}"
echo "[$(date)] $(wc -l < "$TASKS") tasks, $NPAR in parallel on $(hostname)"
tr '\n' '\0' < "$TASKS" | xargs -0 -P "$NPAR" -I{} bash -c '{} || echo "task exited with status $?: {}" >&2'
echo "[$(date)] done"
