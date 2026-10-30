#!/usr/bin/env python
"""Write the benchmark task list: one shell command per (tool, dataset, variant, setting).

Each command runs under ``/usr/bin/time -v`` so that wall-clock time and peak memory are recorded in the
run directory (``time.txt``) together with the tool's own log. ``run_tasks.sh`` executes a task file
in parallel on one node.

Usage: python make_tasks.py <phase> > tasks_<phase>.txt
  phase ``full``    every dataset, all cells
  phase ``halves``  every real dataset, the two disjoint halves of split seed 0 (and seeds 1-2 for the fast
                    tools); the simulations are not split (their readouts use the full data only)
  phase ``sensitivity``  Threadfin's default profile with one setting changed at a time (run_threadfin.py
                    SENSITIVITY), on the full data, the six halves of the real datasets and the simulations
  phase ``sim_seeds``  simulation seeds 4-10 (simulate_inputs.py): Threadfin, its ablations and sensitivity
                    settings, the baselines and clone2vec; not Clonotrace (hours per large simulation) and not
                    Benisse (no receptor sequences)
  phase ``exact_ot``   Clonotrace with the exact optimal-transport distance (``exact`` as the 8th argument of
                    run_clonotrace.R), 8 R sessions per run, as run on 2026-10-25

The programs are read from environment variables (the defaults are the programs on the PATH):
``THREADFIN_PYTHON`` (Threadfin, the baselines and Benisse's encoder; default: this interpreter),
``CLONE2VEC_PYTHON``, ``CLONOTRACE_RSCRIPT``, ``CLONOTRACE_R_LIBS`` (the library holding Clonotrace 1.2.1),
``CLONOTRACE_R`` (a frozen copy of run_clonotrace.R) and, optionally, ``THREADFIN_PACKAGE`` (a folder holding
the ``threadfin`` package of the version to run, put first on PYTHONPATH; the benchmark used 4.3.1).
"""
from __future__ import annotations

import json
import os
import sys

from bench_paths import HERE, INPUTS, RUNS

TF = os.environ.get("THREADFIN_PYTHON", sys.executable)
C2V = os.environ.get("CLONE2VEC_PYTHON", "python")
RSCRIPT = os.environ.get("CLONOTRACE_RSCRIPT", "Rscript")
RLIB = os.environ.get("CLONOTRACE_R_LIBS", "")
TF_PKG = os.environ.get("THREADFIN_PACKAGE", "")
# Rscript reads its script while it runs, so jobs use a frozen copy (editing the repository copy cannot
# disturb a running job); `cp run_clonotrace.R <work>/frozen/run_clonotrace_par.R` before generating tasks
CLONOTRACE_R = os.environ.get("CLONOTRACE_R", str(RUNS.parent / "frozen" / "run_clonotrace_par.R"))
REAL = ["gut_gc_reporter", "mouse_rbd", "mouse_np", "malaria_late", "malaria", "ln_vaccine", "spont_gc",
        "gc_np_pc", "covid_bcell"]
LARRY = ["larry_d2", "larry_d46"]
SIMS = [f"sim_{r}_b{b}_s{s}" for r in ("small", "large") for b in ("0", "1.5") for s in (1, 2, 3)]
SIMS_EXTRA = [f"sim_{r}_b{b}_s{s}" for r in ("small", "large") for b in ("0", "1.5") for s in range(4, 11)]


def r_env():
    return f"env R_LIBS='{RLIB}' " if RLIB else ""


def tf_cmd(args):
    """A Threadfin command, with the frozen package first on PYTHONPATH when THREADFIN_PACKAGE is set."""
    return (f"env PYTHONPATH='{TF_PKG}' " if TF_PKG else "") + f"{TF} {args}"
# Clonotrace runs that are not repeated on the split halves: forced to two-cell clones on the two largest
# datasets (11,141 and 6,174 clones) the full-data run alone takes many hours
SKIP_CLONOTRACE = {"half": {("ln_vaccine", 2), ("larry_d46", 2)}}
BENISSE = {"gut_gc_reporter": 2, "mouse_rbd": 2, "mouse_np": 2, "malaria_late": 2, "malaria": 2, "spont_gc": 2,
           "gc_np_pc": 2, "ln_vaccine": 10, "covid_bcell": 2}


def timed(run, cmd, threads):
    env = (f"PYTHONNOUSERSITE=1 OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads} "
           f"NUMBA_NUM_THREADS={threads} MPLBACKEND=Agg")
    return (f"mkdir -p '{run}' && cd '{HERE}' && {env} /usr/bin/time -v -o '{run}/time.txt' {cmd} "
            f"> '{run}/stdout.log' 2>&1")


def n_eligible(dataset, k):
    meta = json.loads((INPUTS / dataset / "meta.json").read_text())
    if f"n_clones_ge{k}" in meta:
        return meta[f"n_clones_ge{k}"]
    # simulations record only n_clones (every simulated clone has >= 2 cells) and n_clones_ge10
    return meta["n_clones"] if k == 2 else 0


def tasks(variants_by_tool, datasets=REAL + LARRY + SIMS):
    out = []
    for ds in datasets:
        for v in variants_by_tool.get("threadfin", []):
            run = RUNS / ds / v / "_threadfin_task"
            out.append(timed(run, tf_cmd(f"run_threadfin.py {ds} {v}"), 4))
        for v in variants_by_tool.get("clone2vec", []):
            for k in (3, 2):
                run = RUNS / ds / v / f"clone2vec_min{k}"
                out.append(timed(run, f"{C2V} run_clone2vec.py {ds} {v} {k} clone2vec_min{k}", 4))
        for v in variants_by_tool.get("clonotrace", []):
            for k in (10, 2):
                if n_eligible(ds, k) < 3:
                    continue
                run = RUNS / ds / v / f"clonotrace_min{k}"
                if (ds, k) in SKIP_CLONOTRACE.get(v.split("_")[0][:4], set()):
                    continue
                out.append(timed(run, f"{r_env()}{RSCRIPT} '{CLONOTRACE_R}' {ds} {v} {k} 4 '{INPUTS}' '{run}'", 4))
        if ds in BENISSE:
            k = BENISSE[ds]
            for v in variants_by_tool.get("benisse", []):
                run = RUNS / ds / v / f"benisse_min{k}"
                out.append(timed(run, tf_cmd(f"run_benisse.py {ds} {v} {k} benisse_min{k}"), 4))
    return out


# the exact optimal-transport runs of 2026-10-25 (one SLURM job each, 8 R sessions, graph ladder 10,30,50,100)
EXACT_OT = [("covid_bcell", 10), ("covid_bcell", 2), ("gc_np_pc", 2), ("gut_gc_reporter", 10), ("gut_gc_reporter", 2),
            ("larry_d2", 2), ("larry_d46", 10), ("ln_vaccine", 10), ("malaria", 2), ("malaria_late", 2),
            ("mouse_np", 10), ("mouse_np", 2), ("mouse_rbd", 10), ("mouse_rbd", 2), ("spont_gc", 10), ("spont_gc", 2)] + \
           [(f"sim_large_b{b}_s{s}", 10) for b in ("0", "1.5") for s in (1, 2, 3)] + \
           [(f"sim_small_b{b}_s{s}", 2) for b in ("0", "1.5") for s in (1, 2, 3)]


def main(phase):
    if phase == "full":
        lines = tasks({t: ["full"] for t in ("threadfin", "clone2vec", "clonotrace", "benisse")})
    elif phase == "rerun_failed":
        lines = []
        for ds, k in (("gut_gc_reporter", 10), ("gut_gc_reporter", 2), ("mouse_rbd", 10), ("malaria_late", 2)):
            run = RUNS / ds / "full" / f"clonotrace_min{k}"
            lines.append(f"rm -rf '{run}' && " + timed(run, f"{r_env()}{RSCRIPT} '{CLONOTRACE_R}' {ds} full {k} 4 '{INPUTS}' '{run}'", 4))
    elif phase == "halves":
        s0 = ["halfA_s0", "halfB_s0"]
        more = s0 + ["halfA_s1", "halfB_s1", "halfA_s2", "halfB_s2"]
        lines = tasks({"threadfin": more, "clone2vec": more, "clonotrace": s0, "benisse": s0}, REAL + LARRY)
    elif phase == "sensitivity":
        lines = []
        for ds in REAL + LARRY + SIMS + SIMS_EXTRA:
            variants = ["full"] if ds.startswith("sim_") else ["full", "halfA_s0", "halfB_s0", "halfA_s1", "halfB_s1",
                                                               "halfA_s2", "halfB_s2"]
            for v in variants:
                run = RUNS / ds / v / "_threadfin_sensitivity_task"
                lines.append(timed(run, tf_cmd(f"run_threadfin.py {ds} {v} sensitivity"), 2))
    elif phase == "sim_seeds":
        lines = tasks({"threadfin": ["full"], "clone2vec": ["full"]}, SIMS_EXTRA)
        for ds in SIMS_EXTRA:
            run = RUNS / ds / "full" / "_threadfin_sensitivity_task"
            lines.append(timed(run, tf_cmd(f"run_threadfin.py {ds} full sensitivity"), 2))
    elif phase == "exact_ot":
        lines = []
        for ds, k in EXACT_OT:
            run = RUNS / ds / "full" / f"clonotrace_exact_min{k}"
            lines.append(timed(run, f"{r_env()}{RSCRIPT} '{CLONOTRACE_R}' {ds} full {k} 8 '{INPUTS}' '{run}' "
                                    f"10,30,50,100 exact", 1))
    else:
        raise SystemExit(f"unknown phase {phase}")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
