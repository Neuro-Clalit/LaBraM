# Submitting training to AWS SageMaker

LaBraM fine-tuning can be dispatched as managed **SageMaker training jobs**: a
single fine-tune becomes one job; a cross-validation study becomes **one job per
fold**, each job name embedding the fold number. The submission layer is
`labram.runs.submit_sagemaker`; the generic SageMaker SDK wrapper lives in
`labram.aws.sagemaker` (vendored from the shared `common` repo, mirroring
`labram/file_system`); and the in-container entry point is
`labram.runs.sagemaker_entry`.

SageMaker is only used on the *submitting* side — it never affects in-container
training — so `sagemaker.enabled` defaults to `false`.

## Quick start

```bash
# Preview the job plan without contacting AWS:
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_cv.json --dry_run

# Submit one job per CV fold:
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set sagemaker.enabled=true \
        sagemaker.role=arn:aws:iam::123456789012:role/SageMakerRole \
        sagemaker.instance_type=ml.g5.2xlarge \
        sagemaker.output_path=s3://my-bucket/labram/finetune_tuab_cv5 \
        data.data_path=s3://my-bucket/data/TUAB
```

## How a job runs

1. The submit CLI resolves the `FinetuneRunConfig`, uploads it to S3, and mounts
   it in-container via the **`config` input channel**
   (`/opt/ml/input/data/config/run_config.yaml`).
2. For CV it plans one job per fold (or a single fold when
   `cross_validation.fold >= 0`); otherwise a single job. Each job's name is
   `‹job_name_prefix›-fold-‹k›` (or just `‹job_name_prefix›`).
3. SageMaker launches `labram/runs/sagemaker_entry.py --config <mounted> --fold <k>`.
   The entry point points outputs at `/opt/ml/model` (auto-uploaded to
   `sagemaker.output_path` at job end) and dispatches to the cross-validation
   runner or a plain fine-tune.

## Configuration (`sagemaker`)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Consulted by the submit CLI (bypass with `--dry_run`). |
| `role` | `''` | Execution role ARN; `''` → `sagemaker.get_execution_role()`. |
| `instance_type` / `instance_count` | `ml.g4dn.xlarge` / `1` | Compute per job. |
| `volume_size_gb` | `100` | EBS volume per instance. |
| `max_run_sec` | `345600` | Hard wall-clock cap per job. |
| `use_spot` / `max_wait_sec` | `false` / `0` | Managed spot training (`0` → reuse `max_run_sec`). |
| `framework_version` / `py_version` | `2.4.1` / `py311` | Managed PyTorch DLC selectors. |
| `image_uri` | `''` | Explicit training image (overrides the managed DLC). |
| `entry_point` / `source_dir` | `labram/runs/sagemaker_entry.py` / repo root | Training script and packaged code. |
| `job_name_prefix` | `labram-finetune` | Prefix; the fold number is appended for CV. |
| `region` | `''` | AWS region; `''` → boto3 default. |
| `output_path` / `code_location` | `''` | S3 prefixes for model artifacts / packaged source. |
| `config_channel` | `''` | Pre-uploaded config S3 uri; `''` → the CLI uploads it. |
| `environment` / `hyperparameters` / `tags` | `{}` | Extra container env vars / hyperparameters / job tags. |
| `wait` | `false` | Block until the (last) job finishes. |

## Requirements

The submitting machine needs the `sagemaker` and `boto3` SDKs and AWS
credentials with permission to create training jobs, plus an execution role. The
training container uses the managed PyTorch Deep Learning Container for
`framework_version`; `requirements.txt` is installed from the packaged
`source_dir`. The generic wrapper imports the SDK lazily, so `--dry_run` and the
unit tests need neither the SDK nor credentials.

## Design notes

- **`labram.aws.sagemaker`** — `SageMakerJobSpec` + `estimator_kwargs()` (a pure
  spec→kwargs mapping) + `SageMakerLauncher` (lazy SDK). Vendored verbatim from
  `common/sagemaker.py` so both repos share one implementation.
- **`labram.runs.submit_sagemaker`** — builds specs from `FinetuneRunConfig`,
  plans the fold jobs (`plan_jobs`, unit-tested via `--dry_run`), uploads the
  config, and submits.
- **`labram.runs.sagemaker_entry`** — the in-container dispatcher.
