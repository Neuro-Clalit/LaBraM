#!/usr/bin/env python
"""Collect the per-epoch metrics of the line-removal A/B runs into one JSON.

Seed 0 ran locally (``checkpoints/ab_<task>_<variant>_*/log.txt``); further
seeds ran on SageMaker (``scripts/submit_line_removal_ab.sh``, job names
``labram-ab-<task>-<variant>-s<seed>-...``), whose ``finetune/log.txt`` is read
from the job's ``model.tar.gz``. The output feeds Section 9 of
``notebooks/tuh_preprocessing_analysis.ipynb``.

    AWS_PROFILE=mfa python scripts/collect_line_removal_ab.py \\
        --out docs/data/line_removal_ab.json
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

import boto3

VARIANTS = ("A_notch50", "B_notch60", "C_interp5060")
KEEP = {
    "age": ["train_mae", "val_mae", "test_mae", "test_rmse", "test_pearson_r", "test_normal_mae",
            "test_abnormal_mae", "test_age_bias_slope"],
    "abn": ["val_accuracy", "val_balanced_accuracy", "test_accuracy", "test_balanced_accuracy",
            "test_roc_auc", "test_pr_auc", "test_window_balanced_accuracy", "test_window_roc_auc",
            "test_window_pr_auc"],
}
JOB_RE = re.compile(r"^labram-ab-(age|abn)-([ABC])-([a-z0-9]+)-s(\d+)-")


def epochs_from_log(text, task):
    rows = []
    for line in text.splitlines():
        if line.startswith("{"):
            e = json.loads(line)
            rows.append({"epoch": e["epoch"], **{k: e.get(k) for k in KEEP[task]}})
    return rows


def local_runs(checkpoints):
    for task in KEEP:
        for v in VARIANTS:
            logs = sorted(glob.glob(os.path.join(checkpoints, f"ab_{task}_{v}_*", "log.txt")))
            if logs:
                yield {"task": task, "variant": v, "seed": 0, "source": "local g5.2xlarge",
                       "run": os.path.basename(os.path.dirname(logs[-1])),
                       "epochs": epochs_from_log(open(logs[-1]).read(), task)}


def sagemaker_runs():
    sm, s3 = boto3.client("sagemaker"), boto3.client("s3")
    pages = sm.get_paginator("list_training_jobs").paginate(NameContains="labram-ab-",
                                                            StatusEquals="Completed")
    for page in pages:
        for job in page["TrainingJobSummaries"]:
            m = JOB_RE.match(job["TrainingJobName"])
            if not m:
                continue
            task, letter, rest, seed = m.groups()
            variant = next(v for v in VARIANTS if v.startswith(f"{letter}_"))
            uri = sm.describe_training_job(
                TrainingJobName=job["TrainingJobName"])["ModelArtifacts"]["S3ModelArtifacts"]
            bucket, key = uri[5:].split("/", 1)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                text = tar.extractfile("finetune/log.txt").read().decode()
            yield {"task": task, "variant": variant, "seed": int(seed),
                   "source": "SageMaker ml.g5.2xlarge", "run": job["TrainingJobName"],
                   "epochs": epochs_from_log(text, task)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default="docs/data/line_removal_ab.json")
    p.add_argument("--no-sagemaker", action="store_true")
    args = p.parse_args(argv)
    runs = list(local_runs(args.checkpoints))
    if not args.no_sagemaker:
        runs += list(sagemaker_runs())
    # one run per (task, variant, seed): the latest wins
    latest = {}
    for r in runs:
        latest[(r["task"], r["variant"], r["seed"])] = r
    runs = [latest[k] for k in sorted(latest)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"variants": {"A_notch50": "50 Hz notch (LaBraM / published)",
                                "B_notch60": "60 Hz notch",
                                "C_interp5060": "50 + 60 Hz spectrum interpolation"},
                   "runs": runs}, fh, indent=1)
    for task in KEEP:
        for v in VARIANTS:
            seeds = sorted(r["seed"] for r in runs if r["task"] == task and r["variant"] == v)
            print(f"{task} {v}: seeds {seeds}")
    print(f"wrote {len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
