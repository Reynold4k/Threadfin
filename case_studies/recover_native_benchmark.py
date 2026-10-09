#!/usr/bin/env python3
"""Skip a completed native run; retry an allocation timeout without concurrency.

Submit only with an afterany dependency on the original job. A model failure
other than TIMEOUT stops here for diagnosis rather than silently repeating it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from native_benchmark import OUT, stamp


def recover(dataset: str, original_job: str):
    out = OUT / dataset
    manifest = json.loads((out / 'run.json').read_text())
    if manifest.get('bigcn_status') == 'completed':
        print('Original BiGCN run completed; recovery skips model execution.', flush=True)
        return
    result = subprocess.run(
        ['sacct', '-j', original_job, '--parsable2', '--noheader',
         '--format=JobID,State'], check=True, text=True, capture_output=True)
    states = [row.split('|')[1] for row in result.stdout.splitlines()
              if row.split('|')[0] == original_job]
    if states != ['TIMEOUT']:
        raise RuntimeError(f'{original_job}: incomplete native run has states {states}; diagnose before retrying')
    job = os.environ['SLURM_JOB_ID']
    work = out / 'BiGCN_official'
    archive = out / f'BiGCN_attempt_{original_job}_timeout'
    if archive.exists():
        raise RuntimeError(f'Recovery already attempted; preserve {archive} and diagnose')
    if work.exists():
        work.rename(archive)
    stamp(out / 'run.json', bigcn_retry_of=original_job,
          bigcn_initial_slurm_state='TIMEOUT', bigcn_attempts=2,
          bigcn_recovery_job=job,
          bigcn_resource_log=f'native_benchmark_retry_{job}.err',
          bigcn_retry_scope='same upstream 1500-epoch configuration; allocation timeout, not score-driven selection')
    subprocess.run([sys.executable, str(Path(__file__).with_name('native_benchmark.py')),
                    dataset, '--stage', 'bigcn', '--python', sys.executable], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mouse_np', 'mouse_rbd'])
    parser.add_argument('--original-job', required=True)
    args = parser.parse_args()
    recover(args.dataset, args.original_job)
