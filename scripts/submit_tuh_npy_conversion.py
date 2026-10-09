#!/usr/bin/env python
"""Run dataset_maker/make_TUH_npy.py convert as a SageMaker processing job.

CPU instances read EDFs straight from S3 and write one float16 .npy per
recording; each instance takes a duration-balanced shard (by host index), and a
rerun skips recordings already written. The scan (work list) and the skip list
are uploaded with the code. Afterwards run ``make_TUH_npy.py merge`` locally.

    python scripts/submit_tuh_npy_conversion.py --corpus tueg \\
        --scan /data/datasets/EEG-public/TUH_EEG/_state/tueg/recordings.parquet \\
        --skip-stems /data/datasets/EEG-public/TUH_EEG/_state/tuab/recordings.parquet --dry-run
"""
import argparse
import os
import shutil
import tarfile
import tempfile

import boto3

ROLE = "arn:aws:iam::574441342949:role/SageMakerExecutionRole"
BUCKET = "eeg-data-public"
CODE_BUCKET = "sagemaker-us-east-1-574441342949"
CORPORA = {"tueg": "TUH_EEG/tueg/v2.0.2", "tuep": "TUH_EEG/tuep/v3.1.0"}
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# make_TUH_npy falls back to flat imports of these when labram is not installed.
CODE = ["dataset_maker/make_TUH_npy.py", "dataset_maker/make_TUAB_npy.py",
        "labram/data/tuh_metadata.py"]
# Installed by the entry script; the SKLearn image has numpy/pandas/boto3 only.
REQUIREMENTS = ["mne==1.4.2", "pyarrow"]

ENTRY = """\
import subprocess, sys, tarfile
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *{reqs!r}])
tarfile.open("/opt/ml/processing/input/archive/src.tar.gz").extractall("/opt/ml/processing")
sys.path.insert(0, "/opt/ml/processing/src")
sys.argv = ["make_TUH_npy.py", *sys.argv[1:]]
import make_TUH_npy
make_TUH_npy.main()
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--scan", required=True, help="local corpus scan (.parquet)")
    ap.add_argument("--skip-stems", nargs="*", default=[], help="local scans to skip (e.g. TUAB)")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--instance-type", default="ml.c5.4xlarge")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--max-runtime-h", type=float, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prefix = CORPORA[args.corpus]
    edf_root, dst = f"s3://{BUCKET}/{prefix}/edf", f"s3://{BUCKET}/{prefix}/processed_npy"
    staging = f"labram/tuh-npy/{args.corpus}"
    inputs_dir = "/opt/ml/processing/src"
    skip = [f"{inputs_dir}/skip_{i}.parquet" for i in range(len(args.skip_stems))]
    arguments = ["convert", "--scan", f"{inputs_dir}/scan.parquet", "--edf-root", edf_root,
                 "--dst", dst, "--dtype", args.dtype, "--threads", str(args.threads),
                 "--num-shards", str(args.instances)]
    if skip:
        arguments += ["--skip-stems", *skip]
    print(f"{args.instances} x {args.instance_type}: {edf_root} -> {dst}")
    print("arguments:", " ".join(arguments))
    if args.dry_run:
        return

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        for rel in CODE:
            shutil.copy(os.path.join(REPO, rel), src)
        shutil.copy(args.scan, os.path.join(src, "scan.parquet"))
        for i, path in enumerate(args.skip_stems):
            shutil.copy(path, os.path.join(src, f"skip_{i}.parquet"))
        tar = os.path.join(tmp, "src.tar.gz")
        with tarfile.open(tar, "w:gz") as fh:
            fh.add(src, arcname="src")
        entry = os.path.join(tmp, "entry.py")
        with open(entry, "w") as fh:
            fh.write(ENTRY.format(reqs=REQUIREMENTS))
        s3.upload_file(tar, CODE_BUCKET, f"{staging}/src.tar.gz")
        s3.upload_file(entry, CODE_BUCKET, f"{staging}/entry.py")

    from sagemaker.processing import ProcessingInput
    from sagemaker.sklearn.processing import SKLearnProcessor
    proc = SKLearnProcessor(
        framework_version="1.2-1", role=ROLE, instance_type=args.instance_type,
        instance_count=args.instances, volume_size_in_gb=50,
        max_runtime_in_seconds=int(args.max_runtime_h * 3600),
        base_job_name=f"{args.corpus}-npy-convert")
    # The archive arrives packed; the entry script unpacks it to /opt/ml/processing/src.
    proc.run(code=f"s3://{CODE_BUCKET}/{staging}/entry.py",
             inputs=[ProcessingInput(source=f"s3://{CODE_BUCKET}/{staging}/src.tar.gz",
                                     destination="/opt/ml/processing/input/archive")],
             arguments=arguments, wait=False, logs=False)
    job = proc.latest_job.job_name
    print(f"processing job: {job}\n  https://us-east-1.console.aws.amazon.com/sagemaker/home"
          f"?region=us-east-1#/processing-jobs/{job}")


if __name__ == "__main__":
    main()
