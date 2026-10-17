"""Execute downstream story scripts in the user-specified Threadfin environment."""
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parent.parent
env = os.environ.copy()
env.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(PROJECT),
           THREADFIN_DATA="/data/scratch/projects/punim1236/threadfin_data",
           OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           NUMBA_NUM_THREADS="8", MPLCONFIGDIR=str(PROJECT / "case_studies/results/storyline_verification_2026-10-14/matplotlib"))
logs = PROJECT / "case_studies/results/storyline_verification_2026-10-14/execution_logs"
logs.mkdir(parents=True, exist_ok=True)
log_path = logs / (Path(sys.argv[1]).stem + ".log")
print("Execution log:", log_path, flush=True)
with log_path.open("w") as log:
    proc = subprocess.Popen(
        ["/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python", *sys.argv[1:]],
        cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    code = proc.wait()
raise SystemExit(code)
