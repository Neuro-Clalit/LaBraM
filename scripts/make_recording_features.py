#!/usr/bin/env python
"""Per-recording slow amplitude dynamics (``labram/features/slow_dynamics.py``) of every recording of
a cohort, written into the cohort folder as ``<name>.npz`` (``stems``, ``features [n, 228]``,
``names``), where ``mil.recording_features=<name>.npz`` finds it.

Windows ``--start`` .. ``--start + --windows`` of each recording are used (the evaluation's trim and a
5-minute cap); recordings with fewer than ``--min-windows`` get NaN (the training mean in the model).

    python scripts/make_recording_features.py --cohort <cohort folder> --start 6 --windows 30
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from labram.features.slow_dynamics import feature_names, referenced_segment, slow_dynamics

CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
            "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
ARGS = {}


def one(stem):
    arr = np.load(os.path.join(ARGS["cohort"], "recordings", f"{stem}.npy"), mmap_mode="r")
    if arr.shape[0] // 2000 - ARGS["start"] < ARGS["min_windows"]:
        return np.full(len(CHANNELS) * 12, np.nan, np.float32)
    x, present = referenced_segment(arr, ARGS["stored"], CHANNELS, ARGS["start"], ARGS["windows"])
    return slow_dynamics(x, present)


def init(args):
    ARGS.update(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--min-windows", type=int, default=12)
    ap.add_argument("--name", default="slow_dynamics_v1")
    ap.add_argument("--jobs", type=int, default=7)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.cohort, "manifest.json")))
    recs = man["recordings"]
    stems = sorted(r["stem"] for r in (recs if isinstance(recs, list) else recs.values()))
    shared = {"cohort": args.cohort, "start": args.start, "windows": args.windows,
              "min_windows": args.min_windows, "stored": man["channels"]}
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(shared,)) as ex:
        feats = list(ex.map(one, stems, chunksize=32))
    out = os.path.join(args.cohort, f"{args.name}.npz")
    np.savez(out, stems=np.array(stems), features=np.stack(feats).astype(np.float32),
             names=np.array(feature_names(CHANNELS)),
             params=json.dumps({"start_window": args.start, "windows": args.windows, "min_windows": args.min_windows}))
    f = np.stack(feats)
    print(f"{out}: {len(stems)} recordings, {f.shape[1]} features, all-NaN rows {int(np.isnan(f).all(1).sum())}, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
