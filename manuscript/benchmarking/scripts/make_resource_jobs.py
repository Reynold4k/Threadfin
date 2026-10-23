#!/usr/bin/env python
"""Write (and optionally submit) the SLURM jobs of the computational-resource readouts.

``workers``  every tool on GSE287123 RBD (clones of >= 2 cells) with 1, 2, 4, 8 and 16 CPU cores
             (Threadfin: processes for the clone-map search plus BLAS threads; clone2vec: torch threads;
             Clonotrace: parallel R sessions; Benisse: BLAS threads). Each configuration runs alone in a job
             that holds 16 cores, one after the other, so timings are not disturbed by the others.
``size``     every tool with 8 cores on random 5, 10, 25 and 50 % subsets of GSE195673 (make_scaling_inputs.py);
             the full dataset comes from the main runs.

Jobs are spread over the user's SLURM accounts (fair share). Usage: python make_resource_jobs.py [--submit]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bench_paths import HERE, INPUTS, RUNS, WORK

ENVS = "/data/scratch/projects/punim1236/python_envs"
TF, C2V = f"{ENVS}/threadfin_v4/bin/python", f"{ENVS}/clone2vec/bin/python"
RSCRIPT = f"{ENVS}/clonotrace_r/bin/Rscript"
RLIB = WORK.parent / "envs" / "R_clonotrace_1.2.1"
CT_R = WORK / "frozen" / "run_clonotrace_v4.R"
LOGS = WORK.parent / "logs"
PROF = WORK / "profile_run"
PKG_SNAPSHOT = WORK / "frozen" / "threadfin_package"     # copy of threadfin/ taken when the jobs are submitted


def env(threads):
    return (f"PYTHONNOUSERSITE=1 OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
            f"NUMBA_NUM_THREADS={threads} MPLBACKEND=Agg")


def timed(run: Path, cmd: str, threads: int) -> str:
    return (f"mkdir -p '{run}' && {env(threads)} /usr/bin/time -v -o '{run}/time.txt' {cmd} > '{run}/stdout.log' 2>&1 "
            f"|| echo 'failed: {run}'")


def threadfin_profile(ds, variant, n):
    out = PROF / f"{ds}_{variant}.json"
    return (f"{env(n)} /usr/bin/time -v -o '{PROF}/{ds}_{variant}.time' {TF} profile_threadfin_run.py {ds} {n} '{out}' "
            f"> '{PROF}/{ds}_{variant}.log' 2>&1 || echo 'failed: {out}'")


def clone2vec(ds, variant, n):
    return timed(RUNS / ds / variant / "clone2vec_min2", f"{C2V} run_clone2vec.py {ds} {variant} 2 clone2vec_min2", n)


def clonotrace(ds, variant, n):
    run = RUNS / ds / variant / "clonotrace_min2"
    return timed(run, f"env R_LIBS='{RLIB}' {RSCRIPT} '{CT_R}' {ds} {variant} 2 {n} '{INPUTS}' '{run}'", 1)


def benisse(ds, variant, n):
    return timed(RUNS / ds / variant / "benisse_min2", f"{TF} run_benisse.py {ds} {variant} 2 benisse_min2", n)


def representation(ds, variant, n):
    return timed(RUNS / ds / variant / "_threadfin_kernel_task", f"{TF} run_threadfin.py {ds} {variant} threadfin_kernel", n)


JOBS = []                                   # (name, account, partition, cpus, mem, time, commands)
W = (16, 8, 4, 2, 1)
JOBS.append(("res_tf_workers", "punim1445", "long", 16, "48G", "31-00:00:00",
             [threadfin_profile("mouse_rbd", f"full_w{n}", n) for n in W]))
JOBS.append(("res_c2v_workers", "uom00103", "cascade", 16, "48G", "1-00:00:00", [clone2vec("mouse_rbd", f"full_w{n}", n) for n in W]))
JOBS.append(("res_ct_workers", "uom00060", "cascade", 16, "160G", "4-00:00:00",
             [clonotrace("mouse_rbd", f"full_w{n}", n) for n in W]))
JOBS.append(("res_bn_workers", "punim2394", "cascade", 16, "64G", "2-00:00:00",
             [benisse("mouse_rbd", f"full_w{n}", n) for n in (16, 4, 1)]))
F = ("0.05", "0.1", "0.25", "0.5")
JOBS.append(("res_tf_size", "uom00103", "long", 8, "96G", "31-00:00:00",
             [c for f in F for c in (threadfin_profile(f"ln_vaccine_f{f}", "full_w8", 8),
                                     representation(f"ln_vaccine_f{f}", "full", 8))]))
JOBS.append(("res_c2v_size", "uom00103", "cascade", 8, "48G", "2-00:00:00",
             [clone2vec(f"ln_vaccine_f{f}", "full", 8) for f in F]))
JOBS.append(("res_ct_size_small", "uom00060", "cascade", 8, "96G", "4-00:00:00",
             [clonotrace(f"ln_vaccine_f{f}", "full", 8) for f in ("0.05", "0.1")]))
JOBS.append(("res_ct_size_f025", "punim2394", "cascade", 8, "160G", "7-00:00:00", [clonotrace("ln_vaccine_f0.25", "full", 8)]))
JOBS.append(("res_ct_size_f05", "uom00103", "bigmem", 8, "760G", "7-00:00:00", [clonotrace("ln_vaccine_f0.5", "full", 8)]))
JOBS.append(("res_bn_size_small", "punim2394", "cascade", 8, "96G", "2-00:00:00",
             [benisse(f"ln_vaccine_f{f}", "full", 8) for f in ("0.05", "0.1")]))
JOBS.append(("res_bn_size_f025", "uom00060", "bigmem", 8, "760G", "3-00:00:00", [benisse("ln_vaccine_f0.25", "full", 8)]))
JOBS.append(("res_bn_size_f05", "uom00103", "bigmem", 8, "760G", "3-00:00:00", [benisse("ln_vaccine_f0.5", "full", 8)]))


def main(submit: bool, only=None):
    PROF.mkdir(parents=True, exist_ok=True)
    for name, account, partition, cpus, mem, wall, cmds in JOBS:
        if only and name not in only:
            continue
        path = LOGS / f"{name}.sbatch"
        body = "\n".join(f"cd '{HERE}' && {c}" for c in cmds)
        path.write_text(f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={wall}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={LOGS}/{name}_%j.out
# Computational-resource readout of the benchmark (make_resource_jobs.py); configurations run one after another.
unset PYTHONPATH
{"export PYTHONPATH=" + str(PKG_SNAPSHOT) if name.startswith("res_tf") else ""}
{body}
echo done
""")
        if submit:
            r = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
            print(name, account, partition, r.stdout.strip() or r.stderr.strip())
        else:
            print("wrote", path)


if __name__ == "__main__":
    main("--submit" in sys.argv, [a for a in sys.argv[1:] if not a.startswith("--")])
