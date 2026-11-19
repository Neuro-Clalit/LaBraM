#!/usr/bin/env python
"""Build (or reuse) the MIL Stage 1 feature cache of trained runs, and mirror it to S3.

    AWS_PROFILE=mfa python scripts/mil_cache.py \\
        --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --run E2L=/data/runs/heedb/e2l/finetune_... --run E2=/data/runs/heedb/e2_r2/finetune \\
        --out-dir /data/runs/heedb/mil_cache \\
        --s3 s3://eeg-data-public/HEEDB/v4.1/derived/mil_cache

A split is skipped when every run already has it (``--force`` rebuilds). The
S3 upload refuses unless the bucket blocks public access (HEEDB DUA).
"""
import argparse
import os

from labram.mil.cache import build_cache, sync_to_s3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--n-train-windows", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test: first N windows per split")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--s3", default=None, help="mirror the cache to this s3:// prefix")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    args = ap.parse_args()
    runs = dict(spec.split("=", 1) for spec in args.run)
    todo = [s for s in args.splits if args.force or not all(
        os.path.exists(os.path.join(args.out_dir, r, s, "h.npy")) for r in runs)]
    if todo:
        build_cache(runs, args.data_path, args.out_dir, splits=todo,
                    n_train_windows=args.n_train_windows, batch_size=args.batch_size,
                    num_workers=args.num_workers, max_windows=args.max_windows)
    else:
        print("cache complete; nothing to build")
    if args.s3:
        for name in runs:
            sync_to_s3(os.path.join(args.out_dir, name), f"{args.s3.rstrip('/')}/{name}", args.profile)
            print(f"synced {name} -> {args.s3.rstrip('/')}/{name}")


if __name__ == "__main__":
    main()
