#!/usr/bin/env python
"""Smoke-test each loader: load, print shapes, clone counts, barcode overlap."""
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from loaders import LOADERS  # noqa: E402


def smoke(cfg_name: str) -> bool:
    cfg = json.load(open(HERE / "configs" / cfg_name))
    print(f"\n===== {cfg_name} (loader={cfg['loader']}) =====", flush=True)
    t0 = time.time()
    try:
        adata, bcr = LOADERS[cfg["loader"]](cfg)
    except Exception:
        traceback.print_exc()
        return False
    overlap = adata.obs_names.isin(bcr.index).mean()
    print(f"  adata: {adata.shape}, obs cols: {list(adata.obs.columns)[:12]}", flush=True)
    print(f"  bcr:   {bcr.shape}, cols: {list(bcr.columns)}", flush=True)
    if "clone_id" in bcr.columns:
        print(f"  clones: {bcr['clone_id'].nunique()}", flush=True)
    print(f"  barcode overlap: {overlap:.3f} ({int(overlap * adata.n_obs)}/{adata.n_obs})",
          flush=True)
    print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)
    return overlap > 0.01


if __name__ == "__main__":
    names = sys.argv[1:]
    results = {n: smoke(n) for n in names}
    print("\n===== SUMMARY =====", flush=True)
    for n, ok in results.items():
        print(f"  {n}: {'OK' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if all(results.values()) else 1)
