#!/usr/bin/env python
"""Where each timed run ran, and what the scheduler measured for its job (cluster-specific provenance).

The runs were started by SLURM jobs on the University of Melbourne's Spartan cluster, many of them several at a
time inside one job (run_tasks.sh). This script maps every run directory of the full data, of the core-count and
data-size experiments and of the exact-OT Clonotrace runs to the job that produced it, using the job scripts and task
lists in the benchmark's log folder and SLURM's accounting (``sacct``, ``scontrol``):

* the node, its kind (Spartan feature ``sr``: 128-core Intel Xeon Sapphire Rapids; ``physg5``: the 72-core nodes of
  the older ``cascade`` generation), the partition and the account;
* how many runs the job held, and whether it held this run alone;
* the job's peak memory (``MaxRSS`` of its batch step, which covers every process of the job, so for Clonotrace it
  includes the parallel R sessions that /usr/bin/time misses) and its CPU time (``TotalCPU``), both attributable
  to a run only when the job held that run alone.

It needs the cluster (sacct) and the job logs, so it is run once and its output, ``results/run_hosts.csv``, is kept
with the other tables. Usage: python record_run_hosts.py [log folder]   (default: <work>/../logs)
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bench_paths import RESULTS, RUNS, WORK

LOGS = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK.parent / "logs"
RUN_RE = re.compile(r"work/runs/([^/'\"\s]+)/([^/'\"\s]+)/([^/'\"\s]+)")
TF_DEFAULT = ["threadfin_kernel", "threadfin_mean", "threadfin_kernel_nocentre", "threadfin_kernel_nosmooth",
              "threadfin_kernel_noshrink", "mean_pca", "composition", "conga_medoid"]
TF_SENS = ["threadfin_kernel_smooth5", "threadfin_kernel_smooth30", "threadfin_kernel_rff128", "threadfin_kernel_rff512",
           "threadfin_kernel_rff1024", "threadfin_kernel_bw0.5", "threadfin_kernel_bw2", "threadfin_kernel_pc10",
           "threadfin_kernel_pc50", "threadfin_kernel_seed1", "threadfin_kernel_seed2"]
# task lists that ran as `srun --overlap` steps inside another job (log name -> task list, host job)
HOSTED = {"halves_ct_in_main.out": ("tasks_halves_clonotrace.txt", "33480629"),
          "halves_bn_in_heavy.out": ("tasks_halves_benisse.txt", "33480630"),
          "halves_ct_redo.out": ("tasks_halves_ct_redo.txt", "33480630"),
          "covid_heavychain_in_heavy.out": ("tasks_covid_heavychain.txt", "33480630")}
KIND = {"sr": "Sapphire Rapids, 128 cores", "physg5": "72-core cascade generation"}


def expand(ds, variant, sub):
    """Run directories that one task wrote (a Threadfin task writes one directory per method)."""
    if sub == "_threadfin_task":
        return [(ds, variant, m) for m in TF_DEFAULT]
    if sub == "_threadfin_sensitivity_task":
        return [(ds, variant, m) for m in TF_SENS]
    if sub == "_threadfin_kernel_task":
        return [(ds, variant, "threadfin_kernel")]
    if sub.endswith("_par"):                 # parallel twins were renamed to the run's own name once finished
        return [(ds, variant, sub[:-4])]
    return [(ds, variant, sub)]


PROFILE_RE = re.compile(r"profile_run/([\w.-]+?)_(full_w\d+|v431_seq)\.json")


def runs_in(text):
    out = []
    for ds, variant, sub in RUN_RE.findall(text):
        for r in expand(ds, variant, sub):
            if r not in out:
                out.append(r)
    for ds, variant in PROFILE_RE.findall(text):  # the complete Threadfin analysis (profile_threadfin_run.py)
        r = (ds, variant, "threadfin_run")
        if r not in out:
            out.append(r)
    return out


def parse_time(s):
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def log_window(path):
    """Start and end of a hosted task list, from run_tasks.sh's own lines."""
    stamps = re.findall(r"^\[(.+?)\] ", path.read_text(errors="replace"), flags=re.M)
    ts = []
    for s in stamps:
        for fmt in ("%a %b %d %H:%M:%S %Z %Y", "%a %d %b %Y %H:%M:%S %Z"):
            try:
                ts.append(dt.datetime.strptime(s, fmt))
                break
            except ValueError:
                continue
    return (min(ts), max(ts)) if ts else (None, None)


def jobs_from_logs():
    """job id -> {name, runs, parallel, start/end (hosted only)}."""
    jobs = {}
    for out in sorted(list(LOGS.glob("*_[0-9]*.out")) + list((LOGS / "exact_ot").glob("*_[0-9]*.out"))):
        m = re.match(r"(.+)_(\d{7,9})\.out$", out.name)
        if not m:
            continue
        name, jid = m.groups()
        sb = out.parent / f"{name}.sbatch"
        if not sb.exists():
            sb = out.parent / f"{name.replace('run_', '', 1)}.sbatch" if (out.parent / f"{name.replace('run_', '', 1)}.sbatch").exists() else sb
        body = sb.read_text(errors="replace") if sb.exists() else ""
        tasks = re.search(r"run_tasks\.sh\s+\S*?(tasks_[\w.-]+\.txt)\s+(\d+)", body)
        if tasks:
            tf = LOGS / tasks.group(1)
            runs = runs_in(tf.read_text(errors="replace")) if tf.exists() else []
            parallel = int(tasks.group(2))
        else:
            runs = runs_in(body)
            parallel = 1
        if runs:
            jobs[jid] = {"job_name": name, "runs": runs, "parallel": parallel, "window": None}
    for log, (tasks, host) in HOSTED.items():
        p = LOGS / log
        if p.exists() and (LOGS / tasks).exists():
            jobs[f"{host}:{log}"] = {"job_name": log.replace(".out", ""), "runs": runs_in((LOGS / tasks).read_text()),
                                     "parallel": 4, "window": log_window(p), "host": host}
    return jobs


def sacct(ids):
    fmt = "JobID,JobName,Partition,Account,AllocCPUS,Elapsed,TotalCPU,MaxRSS,NodeList,Start,End,State"
    out = subprocess.run(["sacct", "-j", ",".join(sorted(set(ids))), f"--format={fmt}", "-P", "-n"],
                         capture_output=True, text=True).stdout
    rows = [dict(zip(fmt.split(","), line.split("|"))) for line in out.splitlines() if line.strip()]
    alloc = {r["JobID"]: r for r in rows if "." not in r["JobID"]}
    batch = {r["JobID"].split(".")[0]: r for r in rows if r["JobID"].endswith(".batch")}
    return alloc, batch


def seconds(s):
    """SLURM durations: [D-]HH:MM:SS or MM:SS.mmm."""
    if not s:
        return None
    d, _, rest = s.rpartition("-")
    parts = [float(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def gib(s):
    if not s:
        return None
    unit = s[-1].upper()
    v = float(s[:-1]) if unit in "KMGT" else float(s)
    return v * {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1, "T": 1024}.get(unit, 1 / 1024 ** 3)


def node_info(nodes):
    info = {}
    for n in sorted(set(nodes)):
        text = subprocess.run(["scontrol", "show", "node", n], capture_output=True, text=True).stdout
        feat = re.search(r"ActiveFeatures=(\S+)", text)
        cpus = re.search(r"CPUTot=(\d+)", text)
        f = [x for x in (feat.group(1).split(",") if feat else []) if x != "avx512"]
        info[n] = {"node_feature": f[0] if f else "", "node_cores": int(cpus.group(1)) if cpus else None}
    return info


def finished_at(run):
    ds, variant, sub = run
    if sub == "threadfin_run":
        p = RUNS.parent / "profile_run" / f"{ds}_{variant}.json"
        return dt.datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None
    for f in ("run.json", "time.txt"):
        p = RUNS / ds / variant / sub / f
        if p.exists():
            return dt.datetime.fromtimestamp(p.stat().st_mtime)
    return None


def main():
    jobs = jobs_from_logs()
    ids = [j.split(":")[0] for j in jobs] + [j["host"] for j in jobs.values() if "host" in j]
    alloc, batch = sacct(ids)
    for jid, j in jobs.items():
        a = alloc.get(j.get("host", jid))
        if a and j["window"] is None:
            j["window"] = (parse_time(a["Start"]), parse_time(a["End"]))
    nodes = node_info(a["NodeList"] for a in alloc.values() if a["NodeList"].startswith("spartan"))
    # every run directory -> the job whose task list names it and whose time window contains the run's end
    rows = []
    wanted = set()
    for j in jobs.values():
        wanted.update(j["runs"])
    for run in sorted(wanted):
        ds, variant, sub = run
        if not (variant in ("full", "v431_seq") or variant.startswith("full_w")) or sub.startswith("_"):
            continue
        end = finished_at(run)
        if end is None:
            continue
        slack = dt.timedelta(minutes=5)
        cands = [(jid, j) for jid, j in jobs.items() if run in j["runs"] and j["window"][0] is not None
                 and j["window"][0] - slack <= end <= (j["window"][1] or dt.datetime.max) + slack]
        if not cands:
            rows.append({"dataset": ds, "variant": variant, "tool_run": sub, "job_id": None})
            continue
        jid, j = max(cands, key=lambda c: c[1]["window"][0])     # the latest job that names it
        real_id = j.get("host", jid.split(":")[0])
        a, b = alloc.get(real_id, {}), batch.get(real_id, {})
        node = a.get("NodeList", "")
        ni = nodes.get(node, {})
        alone = len(j["runs"]) == 1
        rows.append({"dataset": ds, "variant": variant, "tool_run": sub, "job_id": real_id, "job_name": j["job_name"],
                     "node": node.replace("spartan-", ""), "node_feature": ni.get("node_feature"),
                     "node_kind": KIND.get(ni.get("node_feature"), ni.get("node_feature")),
                     "node_cores": ni.get("node_cores"), "partition": a.get("Partition"), "account": a.get("Account"),
                     "runs_in_job": len(j["runs"]), "parallel_in_job": j["parallel"], "alone_in_job": alone,
                     "job_alloc_cpus": int(a["AllocCPUS"]) if a.get("AllocCPUS") else None,
                     "job_elapsed_hours": (seconds(a.get("Elapsed")) or 0) / 3600 if a else None,
                     "job_max_rss_gb": gib(b.get("MaxRSS")) if alone else None,
                     "job_cpu_hours": (seconds(b.get("TotalCPU")) or 0) / 3600 if alone and b else None})
    df = pd.DataFrame(rows).sort_values(["dataset", "variant", "tool_run"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "run_hosts.csv", index=False)
    print(f"wrote run_hosts.csv ({len(df)} runs, {int(df['job_id'].isna().sum())} without a job)")
    print(df.groupby(["node_kind"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
