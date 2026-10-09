#!/usr/bin/env python
"""Run dataset_maker/make_TUAB_npy.py convert as a SageMaker processing job.

On-demand CPU instances read the window pickles straight from S3 and write one
float32 .npy per recording to the destination prefix; each instance takes a
size-balanced shard of the recordings (by its host index). Re-running skips
recordings already written. Afterwards run ``make_TUAB_npy.py merge`` locally.

    python scripts/submit_tuab_npy_conversion.py --dry-run
    python scripts/submit_tuab_npy_conversion.py --instances 4
"""
import argparse
import os

import boto3

ROLE = "arn:aws:iam::574441342949:role/SageMakerExecutionRole"
SRC = "s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed"
DST = "s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed_npy"
CODE_BUCKET = "sagemaker-us-east-1-574441342949"
CODE_KEY = "labram/tuab-npy/code/make_TUAB_npy.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--instance-type", default="ml.c5.4xlarge")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--max-runtime-h", type=float, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script = os.path.join(os.path.dirname(__file__), "..", "dataset_maker", "make_TUAB_npy.py")
    arguments = ["convert", "--src", args.src, "--dst", args.dst,
                 "--recording-threads", "16", "--read-threads", "64",
                 "--num-shards", str(args.instances)]
    print(f"{args.instances} x {args.instance_type} on-demand, {args.src} -> {args.dst}")
    if args.dry_run:
        print("arguments:", " ".join(arguments))
        return
    # Upload the script ourselves (plain S3, bucket-default encryption) rather
    # than letting the SDK pick an upload location/KMS key.
    boto3.client("s3").upload_file(script, CODE_BUCKET, CODE_KEY)
    from sagemaker.sklearn.processing import SKLearnProcessor
    proc = SKLearnProcessor(
        framework_version="1.2-1", role=ROLE, instance_type=args.instance_type,
        instance_count=args.instances, volume_size_in_gb=30,
        max_runtime_in_seconds=int(args.max_runtime_h * 3600),
        base_job_name="tuab-npy-convert")
    # Each host derives its shard from its index; --num-shards must match.
    proc.run(code=f"s3://{CODE_BUCKET}/{CODE_KEY}", arguments=arguments, wait=False, logs=False)
    job = proc.latest_job.job_name
    print(f"processing job: {job}\n  https://us-east-1.console.aws.amazon.com/sagemaker/home"
          f"?region=us-east-1#/processing-jobs/{job}")


if __name__ == "__main__":
    main()
