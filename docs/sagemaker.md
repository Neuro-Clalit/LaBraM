# Submitting training to AWS SageMaker

**Any** LaBraM trainer can be dispatched as a managed **SageMaker training job**
— VQNSP tokenizer, masked pre-training, or fine-tuning — selected with
`--phase {vqnsp,pretrain,finetune}` (default `finetune`). A single training run
becomes one job; a fine-tune with cross-validation enabled becomes **one job per
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

# Smoke test — run a real but tiny fine-tune job in DEBUG mode to verify the whole
# pipeline (image, pip install, data access, training, model upload) end-to-end.
# trainer.debug shrinks epochs/samples; a cheap single-GPU spot instance keeps it
# fast/cheap; wait=true blocks so you see it finish and can read the job log.
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab.json \
  --set sagemaker.enabled=true \
        sagemaker.role=arn:aws:iam::123456789012:role/SageMakerRole \
        sagemaker.instance_type=ml.g5.xlarge \
        sagemaker.use_spot=true sagemaker.wait=true \
        sagemaker.job_name_prefix=labram-debug-smoke \
        sagemaker.output_path=s3://my-bucket/labram/debug \
        data.data_path=s3://my-bucket/data/TUAB \
        finetune_checkpoint.finetune=s3://my-bucket/checkpoints/labram-base.pth \
        trainer.debug=true \
        clearml.enabled=false

# Submit one job per CV fold:
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set sagemaker.enabled=true \
        sagemaker.role=arn:aws:iam::123456789012:role/SageMakerRole \
        sagemaker.instance_type=ml.g5.2xlarge \
        sagemaker.output_path=s3://my-bucket/labram/finetune_tuab_cv5 \
        data.data_path=s3://my-bucket/data/TUAB

# Submit a different trainer (VQNSP tokenizer / pre-training):
python -m labram.runs.submit_sagemaker --phase vqnsp \
  --config labram/configs/defaults/vqnsp.json --dry_run
python -m labram.runs.submit_sagemaker --phase pretrain \
  --config labram/configs/defaults/pretrain.json \
  --set sagemaker.enabled=true sagemaker.role=arn:aws:iam::123:role/SM
```

`scripts/submit_paper_experiments.sh` bundles the paper experiment set (CV on the
paper config + gradient-clip / codebook / LaBraM++ runs that reuse one recorded
`data_split.json`) behind env-var knobs — `DRY_RUN=1 scripts/submit_paper_experiments.sh`
to preview, or pass `ROLE`/`DATA`/`CKPT`/`VQNSP`/`OUT` to submit (optionally a
subset, e.g. `scripts/submit_paper_experiments.sh cv codebook`).

## How a job runs

1. The submit CLI resolves the phase's run config (`VQNSPRunConfig` /
   `PretrainRunConfig` / `FinetuneRunConfig`), uploads it to S3, and mounts it
   in-container via the **`config` input channel**
   (`/opt/ml/input/data/config/run_config.yaml`).
2. For a fine-tune with CV it plans one job per fold (or a single fold when
   `cross_validation.fold >= 0`); every other trainer is a single job. Each
   job's name is `‹job_name_prefix›-fold-‹k›` (or just `‹job_name_prefix›`).
3. SageMaker launches
   `labram/runs/sagemaker_entry.py --phase <phase> --config <mounted> [--fold <k>]`.
   The entry point points outputs at `/opt/ml/model` (auto-uploaded to
   `sagemaker.output_path` at job end) and dispatches to the matching runner:
   `run_vqnsp` / `run_pretrain` / (`finetune_cv` or plain `run_finetune`).

## Configuration (`sagemaker`)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Consulted by the submit CLI (bypass with `--dry_run`). |
| `role` | `''` | Execution role ARN; `''` → `sagemaker.get_execution_role()`. |
| `instance_type` / `instance_count` | `ml.g5.2xlarge` / `1` | Compute per job (matches a g5.2xl EC2 box; use `ml.g5.xlarge` for the cheaper single-GPU option). |
| `volume_size_gb` | `100` | EBS volume per instance. |
| `max_run_sec` | `345600` | Hard wall-clock cap per job. |
| `use_spot` / `max_wait_sec` | `false` / `0` | Managed spot training (`0` → reuse `max_run_sec`). |
| `framework_version` / `py_version` | `2.4.0` / `py311` | Managed PyTorch DLC selectors (2.4.0 is a published DLC; 2.4.1 is not). |
| `image_uri` | `''` | Explicit training image (overrides the managed DLC). |
| `entry_point` / `source_dir` | `labram/runs/sagemaker_entry.py` / repo root | Training script and packaged code. |
| `job_name_prefix` | `labram-finetune` | Prefix; the fold number is appended for CV. |
| `region` | `''` | AWS region; `''` → boto3 default. |
| `output_path` / `code_location` | `''` | S3 prefixes for model artifacts / packaged source. |
| `config_channel` | `''` | Pre-uploaded config S3 uri; `''` → the CLI uploads it. |
| `environment` / `hyperparameters` / `tags` | `{}` | Extra container env vars / hyperparameters / job tags. |
| `wait` | `false` | Block until the (last) job finishes. |
| `input_mode` | `File` | S3 dataset delivery: `File` (download) or `FastFile` (stream on demand). |
| `subnets` / `security_group_ids` | `[]` | VPC placement (required for an EFS/FSx data mount). |
| `data_fs_id` | `''` | EFS/FSx file-system id to mount as the read-only `dataset` channel (empty → use S3). |
| `data_fs_type` / `data_fs_dir` / `data_fs_access` | `EFS` / `/` / `ro` | File-system type (`EFS`\|`FSxLustre`), directory to mount, access mode. |

## IAM execution role (required)

SageMaker training jobs **must** run under an IAM *execution role* (the role the
training container assumes — distinct from the identity you submit with). So
`sagemaker.role` is required when you submit from a plain EC2 box or laptop.
If `role` is empty the launcher calls `sagemaker.get_execution_role()`, which
only works *inside* a SageMaker-managed environment (a notebook / training job)
and otherwise raises a clear error asking for a role ARN.

```bash
--set sagemaker.role=arn:aws:iam::<account-id>:role/<SageMakerExecutionRole>
```

### A. Create the execution role (AWS Console, one-time)

1. **IAM → Roles → Create role**.
2. **Trusted entity type**: *AWS service*. **Use case**: choose **SageMaker**
   (this sets the trust policy so `sagemaker.amazonaws.com` can assume the role).
   Next.
3. AWS attaches **`AmazonSageMakerFullAccess`** automatically. Keep it. Next.
4. **Name** it e.g. `LaBraMSageMakerExecutionRole` → **Create role**.
5. Give it access to **your data/output bucket** (SageMakerFullAccess only covers
   buckets named `*sagemaker*`). Open the role → **Add permissions → Create
   inline policy**, JSON:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
       "Resource": ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"]
     }]
   }
   ```
6. Copy the role **ARN** (`arn:aws:iam::<account-id>:role/LaBraMSageMakerExecutionRole`)
   — that is `sagemaker.role`. (CLI equivalent: `aws iam create-role
   --role-name LaBraMSageMakerExecutionRole --assume-role-policy-document …` then
   `aws iam attach-role-policy --policy-arn
   arn:aws:iam::aws:policy/AmazonSageMakerFullAccess`.)

### B. Give the *submitting* machine credentials + permission to launch

The machine that runs `submit_sagemaker` authenticates as **you** (a different
identity from the execution role) and needs permission to create training jobs
and to hand the execution role to SageMaker.

1. **Credentials** — how boto3 finds them, in order:
   - **On the EC2 box (recommended):** attach an **instance profile** (IAM →
     Roles → an EC2-trusted role, then EC2 → *Actions → Security → Modify IAM
     role*). No keys to manage.
   - **Laptop / elsewhere:** `aws configure` (writes `~/.aws/credentials`), or
     export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION`.
   - Verify with `aws sts get-caller-identity`.
2. **Permissions** for that submitting identity — attach a policy allowing job
   creation and **passing the execution role**:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {"Effect": "Allow",
        "Action": ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
                   "sagemaker:StopTrainingJob", "sagemaker:AddTags"],
        "Resource": "*"},
       {"Effect": "Allow", "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/LaBraMSageMakerExecutionRole",
        "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
       {"Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
        "Resource": ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"]}
     ]
   }
   ```
   `iam:PassRole` is the one people miss — without it `CreateTrainingJob` fails
   with *"not authorized to perform iam:PassRole"* even though the role exists.
3. **Submit**, passing the role ARN from step A:
   ```bash
   python -m labram.runs.submit_sagemaker \
     --config labram/configs/defaults/finetune_tuab_cv.json \
     --set sagemaker.enabled=true \
           sagemaker.role=arn:aws:iam::<account-id>:role/LaBraMSageMakerExecutionRole
   ```
   Confirm the plan first with `--dry_run` (needs no AWS calls). The launcher
   logs the resolved role and training image before it submits.

### C. Configure a default role locally (skip `--set sagemaker.role` every time)

Pick whichever fits how you work — resolution order is: explicit
`sagemaker.role` (config / `--set`) → env var → `get_execution_role()`.

- **Environment variable (recommended, not committed):** export the ARN once and
  every submit picks it up. Checked in order: `LABRAM_SAGEMAKER_ROLE`,
  `SAGEMAKER_ROLE`, `AWS_SAGEMAKER_ROLE`.
  ```bash
  # add to ~/.bashrc / ~/.zshrc (or the EC2 box's shell profile):
  export LABRAM_SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/LaBraMSageMakerExecutionRole
  # then simply:
  python -m labram.runs.submit_sagemaker --config … --set sagemaker.enabled=true
  ```
- **In a config file (committed default):** set `sagemaker.role` in a JSON config
  (e.g. `finetune_tuab_cv.json`) so the repo carries the default. Fine for a
  private repo; avoid committing the ARN if the repo is public.
- **Inside SageMaker/EC2 with an attached role:** leave `sagemaker.role` empty and
  `get_execution_role()` resolves it automatically — no config needed.

## Training image

By default no `image_uri` is set, so the SDK resolves the **managed PyTorch Deep
Learning Container** for `framework_version=2.4.0`, `py_version=py311` and the
chosen GPU instance (CUDA 12.4, compatible with the g5/A10G family). `2.4.0` is
used because it is a *published* SageMaker DLC tag — `2.4.1` (the version used
for bare-metal installs) has **no** managed DLC. The submit CLI logs the exact
resolved image before launching (`SageMaker training image: …`), and
`SageMakerLauncher.resolve_image_uri(spec)` returns it for verification. Set
`sagemaker.image_uri` to override with your own ECR image (then
`framework_version`/`py_version` are ignored).

## Making the training dataset accessible to the job

The TUAB/TUEV loaders read from a **local directory** (`os.listdir`), so the
dataset must be presented to the container as a mounted path — it can't be read
straight from `s3://` at runtime. There are three supported ways, in increasing
suitability for large datasets:

### 1. S3 + `File` mode (default) — download to the instance
Any `s3://` value in `data.data_path`, `finetune_checkpoint.finetune`, or
`model.codebook_reg.tokenizer_weight` is turned into a SageMaker **input channel**
(`dataset` / `pretrained` / `tokenizer`, mounted under `/opt/ml/input/data/...`)
and the uploaded config is rewritten to the mount path — just pass the S3 URIs on
the normal fields. In `File` mode SageMaker **downloads the whole prefix to the
instance's EBS volume before training starts**, so size `sagemaker.volume_size_gb`
to fit the dataset. Simple; fine for small/medium data.

```bash
--set data.data_path=s3://eeg-data-public/TUAB sagemaker.volume_size_gb=300
```

### 2. S3 + `FastFile` mode — stream on demand (recommended for large S3 data)
`--set sagemaker.input_mode=FastFile` mounts the S3 prefix as a FUSE filesystem
and fetches objects lazily on first access — **no full download, small EBS
volume, faster job start**. `os.listdir` works, so the loaders are unchanged.
Best default for a big bucket like `s3://eeg-data-public`.

```bash
--set data.data_path=s3://eeg-data-public/TUAB sagemaker.input_mode=FastFile
```

### 3. Elastic file system (EFS / FSx for Lustre) — read-only mount
For data that already lives on a shared elastic filesystem, mount it directly
(read-only) as the `dataset` channel instead of copying through S3. Set
`sagemaker.data_fs_id` (+ type/dir) and the **required VPC** placement; the CLI
adds a `FileSystemInput` and points `data.data_path` at the mount. FSx for Lustre
(optionally backed by the S3 bucket) gives the highest throughput for repeated /
multi-fold training.

```bash
--set sagemaker.data_fs_id=fs-0abc123 \
      sagemaker.data_fs_type=FSxLustre \
      sagemaker.data_fs_dir=/eeg/TUAB \
      sagemaker.data_fs_access=ro \
      sagemaker.subnets=subnet-0aaa,subnet-0bbb \
      sagemaker.security_group_ids=sg-0ccc
```
An EFS/FSx mount **requires** `sagemaker.subnets` + `sagemaker.security_group_ids`
(SageMaker runs the job in your VPC to reach the filesystem); the submit CLI errors
early if they're missing. The security group must allow NFS (EFS) / Lustre (FSx)
traffic to the filesystem's mount targets.

### `data.split_json` — read directly from S3
`data.split_json` is the exception to mounting: it is left as an `s3://` URI and
read directly in-container via the shared `FileSystem` (the execution role's S3
access), so **reusing a recorded split** across runs is just
`--set data.split_json=s3://…/data_split.json` (see
[`cross_validation.md`](cross_validation.md) for how the split is applied).

## Dependencies / `pip install` in the job

The estimator packages `source_dir` (the repo root by default) and uploads it;
the SageMaker training toolkit then runs **`pip install -r requirements.txt`**
from that directory before invoking the entry point. LaBraM's `requirements.txt`
deliberately does **not** pin `torch`, so the container keeps the DLC's CUDA
torch build and only the remaining libraries (timm, mne, pyhealth, scikit-learn,
`boto3`/`s3fs` for S3 data, `clearml` for tracking, …) are installed at start-up.
To add job-only dependencies, either extend `requirements.txt` or point
`sagemaker.source_dir` at a directory whose `requirements.txt` you control.

## ClearML logging from SageMaker

ClearML logging works inside the job: `clearml` is in `requirements.txt`, and the
per-fold tasks group under `‹project›/‹experiment›` exactly as for a local run.
It only needs credentials in the container. When `clearml.enabled` is set, the
submit CLI **forwards the submitter's `CLEARML_*` environment variables**
(`CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`,
`CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`) into the job's environment (without
overwriting any you set explicitly via `sagemaker.environment`). Provide them on
the submitting machine (or pass them through `sagemaker.environment`) and the
fold jobs will report to your ClearML server; the fold-number parsing in
`cv_report` tolerates the `append_timestamp` task-name suffix.

## Requirements (submitting machine)

The `sagemaker` and `boto3` SDKs, AWS credentials able to create training jobs,
and an execution role (above). The generic wrapper imports the SDK lazily, so
`--dry_run` and the unit tests need neither the SDK nor credentials.

## Design notes

- **`labram.aws.sagemaker`** — `SageMakerJobSpec` + `estimator_kwargs()` (a pure
  spec→kwargs mapping) + `SageMakerLauncher` (lazy SDK). Vendored verbatim from
  `common/sagemaker.py` so both repos share one implementation.
- **`labram.runs.submit_sagemaker`** — builds specs from `FinetuneRunConfig`,
  plans the fold jobs (`plan_jobs`, unit-tested via `--dry_run`), uploads the
  config, and submits.
- **`labram.runs.sagemaker_entry`** — the in-container dispatcher.
