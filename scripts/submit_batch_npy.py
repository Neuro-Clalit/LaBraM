#!/usr/bin/env python
"""Convert a TUH corpus EDF -> npy as an AWS Batch Spot array job.

Each array child converts one duration-balanced shard of the corpus scan
(``make_TUH_npy.py convert``, shard = ``AWS_BATCH_JOB_ARRAY_INDEX``) from the
corpus' EDFs on S3 to ``<corpus prefix>/<dst>/`` on S3. A rerun skips finished
recordings, so resubmitting after failures only redoes what is missing.
Afterwards ``make_TUH_npy.py merge`` writes the manifest (run it locally).
Infrastructure: infra/batch_dataprep (deploy.sh).

    python scripts/submit_batch_npy.py submit --corpus tueg \\
        --scan /data/datasets/EEG-public/TUH_EEG/_state/tueg/recordings.parquet \\
        --notch 50,60 --line-removal interp --shards 64 --dry-run
    python scripts/submit_batch_npy.py status --job-id <parent job id>
    # HEEDB (BIDS labels, flat windows dropped), after download_heedb.py:
    python scripts/submit_batch_npy.py submit --corpus heedb \\
        --scan /data/datasets/HEEDB/v4.1/scan/convert_scan.parquet --notch 50,60 \\
        --line-removal interp --channel-names bids --min-channels 19 --drop-flat-std 0.5 \\
        --shards 64 --dry-run
"""
import argparse
import datetime as dt
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_maker.download_tuh import BUCKET, CORPORA  # noqa: E402

QUEUE = "labram-dataprep"
JOB_DEFINITION = "labram-dataprep-tuh-npy"
# Non-NEDC corpora converted by the same job: S3 prefix under BUCKET and the
# subdir the scan's paths are relative to. HEEDB (Harvard EEG Database, BDSP
# DUA): copied by dataset_maker/download_heedb.py into the key-only bucket.
EXTRA_PREFIXES = {"heedb": ("HEEDB/v4.1", "raw")}


def corpus_prefix(name):
    """(S3 prefix, default EDF subdir) of a corpus."""
    if name in EXTRA_PREFIXES:
        return EXTRA_PREFIXES[name]
    return CORPORA[name].s3_prefix, ""


def default_dst(notch, dtype, method="notch"):
    tag = "none" if notch.lower() == "none" else notch.replace(",", "-")
    return f"processed_npy_{method}{tag}_{dtype}"


def cmd_submit(args):
    prefix, default_subdir = corpus_prefix(args.corpus)
    base = f"s3://{BUCKET}/{prefix}"
    dst = f"{base}/{args.dst or default_dst(args.notch, args.dtype, args.line_removal)}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    scan_key = f"{prefix}/_work/scan-{stamp}.parquet"
    subdir = args.edf_subdir if args.edf_subdir is not None else default_subdir
    edf_root = f"{base}/{subdir}" if subdir else base
    command = ["convert", "--scan", f"s3://{BUCKET}/{scan_key}", "--edf-root", edf_root,
               "--dst", dst, "--num-shards", str(args.shards), "--notch", args.notch,
               "--line-removal", args.line_removal,
               "--dtype", args.dtype, "--threads", str(args.threads)]
    # Only when set, so TUH commands still run on images built before these flags.
    if args.channel_names != "tuh":
        command += ["--channel-names", args.channel_names]
    if args.min_channels > 1:
        command += ["--min-channels", str(args.min_channels)]
    if args.drop_flat_std > 0:
        command += ["--drop-flat-std", str(args.drop_flat_std)]
    print(f"{args.corpus}: {args.shards} Spot array children -> {dst}")
    print("command:", " ".join(command))
    if args.dry_run:
        return
    boto3.client("s3").upload_file(args.scan, BUCKET, scan_key)
    job = boto3.client("batch").submit_job(
        jobName=f"{args.corpus}-npy-{stamp}", jobQueue=QUEUE, jobDefinition=JOB_DEFINITION,
        arrayProperties={"size": args.shards}, containerOverrides={"command": command},
        tags={"Project": "labram", "Purpose": "dataprep", "Corpus": args.corpus})
    print(f"submitted {job['jobName']}: {job['jobId']}")
    print(f"  status: python scripts/submit_batch_npy.py status --job-id {job['jobId']}")
    print(f"  merge:  python -m dataset_maker.make_TUH_npy merge --scan {args.scan} --dst {dst}")


def cmd_status(args):
    batch = boto3.client("batch")
    job = batch.describe_jobs(jobs=[args.job_id])["jobs"][0]
    summary = job.get("arrayProperties", {}).get("statusSummary", {})
    print(f"{job['jobName']}: {job['status']}  " + "  ".join(f"{k}={v}" for k, v in sorted(summary.items()) if v))
    if summary.get("FAILED"):
        pages = batch.get_paginator("list_jobs").paginate(arrayJobId=args.job_id, jobStatus="FAILED")
        for page in pages:
            for j in page["jobSummaryList"][:20]:
                print(f"  failed child {j['arrayProperties']['index']}: {j.get('statusReason', '')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--corpus", required=True, choices=sorted(set(CORPORA) | set(EXTRA_PREFIXES)))
    s.add_argument("--scan", required=True, help="local corpus scan (work list), .parquet")
    s.add_argument("--notch", required=True, help="mains frequencies: '50,60' (recommended with "
                   "interp), '50' (TUAB papers), '60', 'none'")
    s.add_argument("--line-removal", default="interp", choices=["notch", "interp"])
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.add_argument("--shards", type=int, default=64)
    s.add_argument("--threads", type=int, default=4, help="recordings in flight per child")
    s.add_argument("--dst", default=None, help="output dir under the corpus prefix "
                   "(default processed_npy_notch<notch>_<dtype>)")
    s.add_argument("--edf-subdir", default=None, help="dir under the corpus prefix the scan's "
                   "paths are relative to ('' for download_tuh scans, 'edf' for a scan made at edf/; "
                   "default '' / 'raw' for heedb)")
    s.add_argument("--channel-names", default="tuh", choices=["tuh", "bids"])
    s.add_argument("--min-channels", type=int, default=1)
    s.add_argument("--drop-flat-std", type=float, default=0.0, help="µV; 0 = keep flat windows")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_submit)
    t = sub.add_parser("status")
    t.add_argument("--job-id", required=True)
    t.set_defaults(func=cmd_status)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
