#!/usr/bin/env python
"""Submit eval-only SageMaker jobs that re-score trained runs (no training).

Each job runs ``scripts/eval_runs.py`` on a shard of ``--manifest`` (a JSON
list of ``{"name", "model"}``): it downloads each run's ``model.tar.gz`` from
S3, rebuilds the model and data from the run's own config and scores
``checkpoint-best.pth`` on train / val / test (MAE, RMSE, R², r per cohort),
checks the rebuilt train / val / test recordings against the run's recorded
``data_split.json``, and writes ``metrics.csv``, ``split_check.csv`` and
``predictions.csv.gz`` as the job's ``model.tar.gz``.

* Runs whose ``model`` is a local directory are tarred (checkpoint-best.pth,
  run_config.yaml, data_split.json, log.txt) and uploaded first.
* The dataset channel (``--data-uri``, default the TUAB v3.0.0 npy export the
  phase-1 runs trained on) is delivered in File mode: every model reads all
  training windows, so one local copy per job beats streaming it 15 times.
* SageMaker settings (role, image, region, KMS key, profile) come from
  ``--config`` / ``--set`` exactly as for ``labram.runs.submit_sagemaker``.

    python scripts/submit_eval_runs.py --manifest runs.json --shards 3 --dry-run
    python scripts/submit_eval_runs.py --manifest runs.json --shards 3 \\
        --set sagemaker.role=arn:aws:iam::<acct>:role/SageMakerExecutionRole
"""
import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from labram.aws.sagemaker import SageMakerJobSpec, SageMakerLauncher  # noqa: E402
from labram.configs.run_configs import FinetuneRunConfig  # noqa: E402
from labram.configs.utils_conf import parse_overrides  # noqa: E402
from labram.runs import submit_sagemaker as ss  # noqa: E402

logger = logging.getLogger("submit_eval_runs")
DATA_URI = "s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed_npy/"
TOKENIZER_URI = "s3://eeg-data-public/models/labram/vqnsp.pth"
RUN_FILES = ("checkpoint-best.pth", "run_config.yaml", "data_split.json", "log.txt")


def tar_run_dir(run_dir: str, dest: str) -> str:
    """``dest`` = a SageMaker-style ``model.tar.gz`` (``finetune/<file>``) of the
    files of ``run_dir`` the evaluator needs."""
    with tarfile.open(dest, "w:gz") as tar:
        for f in RUN_FILES:
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                tar.add(p, arcname=f"finetune/{f}")
    return dest


def shard(items, n):
    """``n`` round-robin shards of ``items`` (balanced sizes)."""
    return [items[k::n] for k in range(n) if items[k::n]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--config", default="labram/configs/defaults/finetune_tuab_age.json")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--data-uri", default=DATA_URI)
    ap.add_argument("--tokenizer-uri", default=TOKENIZER_URI)
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--volume-gb", type=int, default=200)
    ap.add_argument("--max-run-hours", type=float, default=8.0)
    ap.add_argument("--name", default="labram-eval-p1", help="job name prefix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.manifest) as f:
        manifest = json.load(f)
    config = FinetuneRunConfig.load_config(args.config, **parse_overrides(list(args.overrides)))
    sm = config.sagemaker
    shards = shard(manifest, args.shards)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    logger.info("%d runs in %d shard(s): %s", len(manifest), len(shards), [len(s) for s in shards])
    if args.dry_run:
        for k, s in enumerate(shards):
            logger.info("[dry-run] shard %d: %s", k, [m["name"] for m in s])
        return

    profile = ss.resolve_aws_profile(config)
    launcher = SageMakerLauncher(region=sm.region or None, default_role=sm.role, profile=profile or None)
    role = launcher.resolve_role(sm.role)
    identity = launcher.caller_identity()
    mismatch = ss.cross_account_role_error(role, identity, profile)
    if mismatch:
        raise SystemExit(mismatch)
    session = launcher._get_session()
    bucket = session.default_bucket()
    prefix = f"labram-eval/{args.name}-{stamp}"
    extra = ss.source_upload_extra_args(config)

    # 1. local runs -> S3 tarballs
    with tempfile.TemporaryDirectory(prefix="eval_runs_upload_") as tmp:
        for m in manifest:
            if os.path.isdir(m["model"]):
                tar = tar_run_dir(m["model"], os.path.join(tmp, f"{m['name']}.tar.gz"))
                m["model"] = ss._upload_data(session, tar, f"{prefix}/models", extra).rstrip("/")
                if not m["model"].endswith(".tar.gz"):
                    m["model"] = f"{m['model']}/{m['name']}.tar.gz"
                logger.info("uploaded %s -> %s", m["name"], m["model"])
                os.remove(tar)
        # 2. shard manifests -> S3
        shard_uris = []
        for k, s in enumerate(shards):
            p = os.path.join(tmp, f"shard_{k}.json")
            with open(p, "w") as f:
                json.dump(s, f, indent=1)
            uri = ss._upload_data(session, p, f"{prefix}/manifests/shard_{k}", extra).rstrip("/")
            shard_uris.append(uri if uri.endswith(".json") else f"{uri}/shard_{k}.json")
    # 3. source
    git_info = ss.utils.collect_git_info(ss.repo_root())
    with ss.staged_source_dir(config) as source_dir:
        source_uri = ss.package_and_upload_source(launcher, config, source_dir, git_info)
    # 4. one job per shard
    for k, uri in enumerate(shard_uris):
        spec = SageMakerJobSpec(
            entry_point="scripts/eval_runs.py", source_dir=source_uri, role=role,
            instance_type=args.instance_type, volume_size_gb=args.volume_gb,
            max_run_sec=int(args.max_run_hours * 3600), framework_version=sm.framework_version,
            py_version=sm.py_version, image_uri=sm.image_uri,
            hyperparameters={"manifest": f"/opt/ml/input/data/manifest/{os.path.basename(uri)}",
                             "data-path": "/opt/ml/input/data", "out-dir": "/opt/ml/model",
                             "data-format": "npy",
                             "tokenizer-weight": "/opt/ml/input/data/tokenizer/vqnsp.pth"},
            environment=dict(sm.environment), tags={**sm.tags, "project": "LaBraM", "phase": "eval"},
            inputs={"processed_npy": args.data_uri, "manifest": uri, "tokenizer": args.tokenizer_uri},
            input_mode="File", output_path=sm.output_path, code_location=sm.code_location,
            output_kms_key=sm.output_kms_key, base_job_name=f"{args.name}-s{k}")
        launcher.submit(spec, wait=False, stream_logs=False,
                        on_submitted=lambda name, k=k: logger.info("submitted shard %d: %s", k, name))
    logger.info("Submitted %d eval job(s); manifests under s3://%s/%s", len(shard_uris), bucket, prefix)


if __name__ == "__main__":
    main()
