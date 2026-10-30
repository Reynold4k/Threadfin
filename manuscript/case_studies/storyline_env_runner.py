"""Execute downstream story scripts in the user-specified Threadfin environment."""
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(PROJECT),
           THREADFIN_DATA=os.environ.get("THREADFIN_DATA", str(PROJECT / "manuscript" / "case_studies" / "data")),
           OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           NUMBA_NUM_THREADS="8", MPLCONFIGDIR=str(PROJECT / "manuscript/case_studies/results/storyline_verification_2026-10-14/matplotlib"))
logs = PROJECT / "manuscript/case_studies/results/storyline_verification_2026-10-14/execution_logs"
logs.mkdir(parents=True, exist_ok=True)
log_path = logs / (Path(sys.argv[1]).stem + ".log")
print("Execution log:", log_path, flush=True)
with log_path.open("w") as log:
    proc = subprocess.Popen(
        [os.environ.get("THREADFIN_PYTHON", sys.executable), *sys.argv[1:]],
        cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    code = proc.wait()
raise SystemExit(code)
