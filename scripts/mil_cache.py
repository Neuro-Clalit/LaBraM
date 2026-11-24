#!/usr/bin/env python
"""Build (or reuse) the MIL Stage 1 feature cache of trained runs, and mirror it to S3.

    AWS_PROFILE=mfa python scripts/mil_cache.py \\
        --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --run E2L=/data/runs/heedb/e2l/finetune_... --run E2=/data/runs/heedb/e2_r2/finetune \\
        --out-dir /data/runs/heedb/mil_cache \\
        --s3 s3://eeg-data-public/HEEDB/v4.1/derived/mil_cache

A split is skipped when every run already has it (``--force`` rebuilds). The
S3 upload refuses unless the bucket blocks public access (HEEDB DUA).

Frozen embeddings of a pretrained, not fine-tuned encoder (the input of
``scripts/fm_probe.py``): ``--pretrained NAME=CONFIG`` takes the data, window
selection and input options of a run config and the weights of its
``finetune_checkpoint.finetune``; ``--override NAME:key=value`` changes a field
for that encoder only.

    python scripts/mil_cache.py --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \
        --pretrained CBraMod=/data/runs/heedb/mil_phased_cv/P4_L1/fold_0/cv/fold_0/run_config.yaml \
        --override CBraMod:model.model=cbramod_base_patch200 \
        --override CBraMod:finetune_checkpoint.finetune=./checkpoints/cbramod-base.pth \
        --recordings probe_recordings.json --no-descriptors --out-dir /data/runs/fm_probe/cache
"""
import argparse
import json
import os

from labram.mil.cache import build_cache, sync_to_s3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR")
    ap.add_argument("--pretrained", action="append", default=[], metavar="NAME=CONFIG",
                    help="a pretrained, not fine-tuned encoder: run config (yaml / json) whose "
                         "finetune_checkpoint.finetune holds the released weights")
    ap.add_argument("--override", action="append", default=[], metavar="NAME:KEY=VALUE",
                    help="config override for one --pretrained encoder (repeatable)")
    ap.add_argument("--recordings", default=None,
                    help="JSON list of recording stems, or {split: [stems]} (a split it does not "
                         "name is cached whole; the file of `scripts/fm_probe.py sample` works): "
                         "cache only these")
    ap.add_argument("--recording-zscore", nargs="*", default=[], metavar="NAME",
                    help="encoders whose input is z-scored per recording and channel, clipped at 15 SD")
    ap.add_argument("--no-descriptors", action="store_true", help="skip the window descriptors")
    ap.add_argument("--readout", choices=["pooled", "channel_mean"], default="pooled",
                    help="pretrained encoders: the pooled head input, or the per-channel mean of the "
                         "patch tokens concatenated over channels (keeps the topography)")
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
    pretrained = {name: (cfg, []) for name, cfg in (spec.split("=", 1) for spec in args.pretrained)}
    for spec in args.override:
        name, override = spec.split(":", 1)
        if name not in pretrained:
            ap.error(f"--override {spec}: {name} is not a --pretrained encoder")
        pretrained[name][1].append(override)
    names = list(runs) + list(pretrained)
    if not names:
        ap.error("give at least one --run or --pretrained")
    recordings = None
    if args.recordings:
        with open(args.recordings) as fh:
            recordings = json.load(fh)
        if isinstance(recordings, dict):          # scripts/fm_probe.py sample: {"cache": {split: stems}}
            recordings = recordings.get("cache", recordings)
    todo = [s for s in args.splits if args.force or not all(
        os.path.exists(os.path.join(args.out_dir, r, s, "h.npy")) for r in names)]
    if todo:
        build_cache(runs, args.data_path, args.out_dir, splits=todo,
                    n_train_windows=args.n_train_windows, batch_size=args.batch_size,
                    num_workers=args.num_workers, max_windows=args.max_windows,
                    pretrained=pretrained, recordings=recordings,
                    recording_zscore=args.recording_zscore, descriptors=not args.no_descriptors,
                    readout=args.readout)
    else:
        print("cache complete; nothing to build")
    if args.s3:
        for name in names:
            sync_to_s3(os.path.join(args.out_dir, name), f"{args.s3.rstrip('/')}/{name}", args.profile)
            print(f"synced {name} -> {args.s3.rstrip('/')}/{name}")


if __name__ == "__main__":
    main()
