#!/usr/bin/env python
"""Check the npy TUAB format against the window pickles on a real dataset.

1. Split: build the TUAB_AGE bundle in both formats with a run's window
   selection and compare recordings / window counts per split -- with each
   other and, optionally, with a run's recorded ``data_split.json``.
2. Items: compare sampled items bit for bit (input tensor and target).
3. Throughput: time a DataLoader over the training split in each format
   (cold page cache when ``--drop-caches`` and passwordless sudo are available).

    python scripts/verify_npy_format.py \\
        --data-path /data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf \\
        --config labram/configs/defaults/finetune_tuab_age.json \\
        --data-split path/to/data_split.json --drop-caches
"""
import argparse
import json
import os
import subprocess
import time
from collections import Counter

import numpy as np
import torch

from labram.eval.loading import load_run_config
from labram.data import get_dataset_bundle
from labram.data.window_selection import WindowSelection, _leaves, apply_window_selection
from labram.data.tuh_metadata import recording_stem


def build(fmt, data_path, data_cfg):
    bundle = get_dataset_bundle("TUAB_AGE", data_path, data_format=fmt)
    return apply_window_selection(bundle, WindowSelection.from_data_config(data_cfg))


def split_summary(dataset):
    files = [f for leaf in _leaves(dataset) for f in leaf.files]
    return Counter(recording_stem(f) for f in files), files


def compare_splits(pkl, npy, recorded):
    ok = True
    for name in ("train", "val", "test"):
        (cp, fp), (cn, fn) = split_summary(getattr(pkl, name)), split_summary(getattr(npy, name))
        same = fp == fn
        line = f"{name:5s} recordings={len(cn)} windows={sum(cn.values())} pickle==npy: {same}"
        if recorded and name in recorded:
            rec = recorded[name]
            rec_ids = {os.path.basename(r) for r in rec["recordings"]}
            same_rec = rec_ids == set(cn) and rec["n_windows"] == sum(cn.values())
            line += (f" | recorded: recordings={rec['n_recordings']} windows={rec['n_windows']}"
                     f" match: {same_rec}")
            same = same and same_rec
        print(line)
        ok = ok and same
    return ok


def compare_items(pkl, npy, n, seed=0):
    rng = np.random.default_rng(seed)
    ok = True
    for name in ("train", "val", "test"):
        a, b = getattr(pkl, name), getattr(npy, name)
        idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
        bad = 0
        for i in idx:
            (xa, ya), (xb, yb) = a[int(i)][:2], b[int(i)][:2]
            if xa.dtype != xb.dtype or xa.shape != xb.shape or not torch.equal(xa, xb) or ya != yb:
                bad += 1
        print(f"{name:5s} {len(idx)} sampled items, mismatches: {bad} (shape {tuple(xb.shape)}, {xb.dtype})")
        ok = ok and bad == 0
    return ok


def drop_caches():
    subprocess.run(["sync"], check=True)
    r = subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
    return r.returncode == 0


def throughput(dataset, batches, batch_size, workers, cold):
    if cold and not drop_caches():
        print("  (could not drop page cache; numbers are warm-cache)")
    g = torch.Generator().manual_seed(0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=workers, generator=g, drop_last=True)
    t0, n = time.perf_counter(), 0
    for i, batch in enumerate(loader):
        if i == 0:
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
            continue
        n += batch[0].shape[0]
        if i >= batches:
            break
    dt = time.perf_counter() - t0
    return n / dt, t_first


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--config", default="labram/configs/defaults/finetune_tuab_age.json")
    ap.add_argument("--data-split", default=None, help="a run's recorded data_split.json")
    ap.add_argument("--items", type=int, default=2000, help="sampled items per split")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--drop-caches", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true")
    args = ap.parse_args()

    data_cfg = load_run_config(args.config).data
    recorded = json.load(open(args.data_split)) if args.data_split else None
    t = time.perf_counter(); pkl = build("pickle", args.data_path, data_cfg)
    print(f"pickle bundle built in {time.perf_counter() - t:.1f}s")
    t = time.perf_counter(); npy = build("npy", args.data_path, data_cfg)
    print(f"npy bundle built in {time.perf_counter() - t:.1f}s")
    print(f"target_stats pickle={pkl.target_stats} npy={npy.target_stats}")

    print("\n== split ==")
    ok = compare_splits(pkl, npy, recorded) and pkl.target_stats == npy.target_stats
    print("\n== items ==")
    ok = compare_items(pkl, npy, args.items) and ok
    if not args.skip_throughput:
        print(f"\n== training loader throughput ({args.batches} batches x {args.batch_size}, "
              f"{args.workers} workers, shuffled) ==")
        for fmt, b in (("pickle", pkl), ("npy", npy)):
            rate, first = throughput(b.train, args.batches, args.batch_size, args.workers,
                                     args.drop_caches)
            print(f"{fmt:6s} {rate:8.0f} samples/s  ({args.batch_size / rate:.3f} s/batch; "
                  f"first batch {first:.1f}s)")
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
