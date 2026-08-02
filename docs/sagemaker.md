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

A complete single-job TUAB abnormal/normal fine-tune on spot, streaming the
corpus with `FastFile` and reporting to ClearML — `trainer.debug=true` makes it a
few-batch smoke test, drop it for the real run:

```bash
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab.json \
  --set sagemaker.enabled=true \
        sagemaker.role=arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        sagemaker.instance_type=ml.g5.xlarge \
        sagemaker.input_mode=FastFile sagemaker.use_spot=true sagemaker.wait=true \
        sagemaker.job_name_prefix=labram-abnormal \
        data.dataset=TUAB \
        data.data_path=s3://<bucket>/TUH_Abnormal/v3.0.0/edf/processed/ \
        output.output_dir= output.log_dir= \
        clearml.enabled=true clearml.project_name=eeg/abnormal \
        clearml.task_name=finetune_tuab_abnormal \
        trainer.debug=true
```

Empty `output.output_dir` / `output.log_dir` are filled in by the container
entry point with `/opt/ml/model/finetune` and `…/finetune/tensorboard`, so
checkpoints and TensorBoard events end up in the job's `model.tar.gz`. The
shipped `./checkpoints/labram-base.pth` is **not** re-uploaded: it has an S3
mirror in `sagemaker.weight_s3_uris`, so it is mounted as the `pretrained`
channel straight from S3 (see [S3 data & checkpoints](#s3-data--checkpoints-input-channels)).

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
| `weight_s3_uris` | `{./checkpoints/labram-base.pth → s3://eeg-data-public/models/labram/labram-base.pth, ./checkpoints/vqnsp.pth → s3://…/vqnsp.pth}` | Local weight paths whose bytes already live in S3; the mirror is mounted as a channel instead of the local file being uploaded (see below). |
| `input_mode` | `File` | Channel delivery: `File`, `FastFile` or `Pipe` (see below). |
| `environment` / `hyperparameters` / `tags` | `{}` | Extra container env vars / hyperparameters / job tags. |
| `wait` | `false` | Block until the (last) job finishes. |

Unknown `--set` keys are rejected, so a typo in one of these names fails the
submission instead of being silently dropped.

### `input_mode`: use `FastFile` for the TUH corpora

`File` (the default) downloads every object in every channel onto the instance's
EBS volume before training starts. The preprocessed TUAB corpus is ~400k small
pickles, so that means a long idle download and a `volume_size_gb` big enough to
hold the whole corpus — on a spot instance, paid for and repeated after every
interruption.

`FastFile` streams the channel from S3 through a FUSE mount instead: training
starts immediately and the volume only holds checkpoints. The window loaders'
access pattern (`os.listdir` + open-one-pickle-per-item) is exactly what it
suits, so prefer it for any real dataset channel:

```bash
--set sagemaker.input_mode=FastFile
```

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

## S3 data & checkpoints (input channels)

The TUAB/TUEV loaders read from a local directory (`os.listdir`), so the dataset
must be **mounted**, not read from S3 at runtime. The submit CLI handles this
automatically: any `s3://` value in `data.data_path`,
`finetune_checkpoint.finetune`, `model.codebook_reg.tokenizer_weight` (finetune),
or `model.tokenizer.tokenizer_weight` (the frozen VQNSP used by pre-training) is
turned into a SageMaker **input channel** (`dataset` / `pretrained` /
`tokenizer`, mounted under `/opt/ml/input/data/...`) and the uploaded config is
rewritten to the in-container mount path. Just pass the S3 URIs on the normal
config fields.

Nothing on the submitting machine's filesystem exists inside the container, so
**local** paths are handled rather than passed through:

- `finetune_checkpoint.finetune` / `model.codebook_reg.tokenizer_weight` /
  `model.tokenizer.tokenizer_weight` — a local file is **uploaded** to the
  session bucket and mounted on its channel, *unless* the path has an S3 mirror
  in `sagemaker.weight_s3_uris` (see below), in which case the mirror is mounted
  and nothing is uploaded. A path that is neither mirrored nor an existing local
  file fails the submission immediately. `https://` URLs are left alone
  (`torch.hub` fetches them in-container).
- `data.data_path` — **rejected**. Corpora are far too large to upload as part
  of a submission; put the preprocessed data in S3 first (see
  `scripts/upload_tuab_to_s3.sh`) and pass the uri.

### `weight_s3_uris`: don't re-upload the checkpoints in git

The shipped `./checkpoints/labram-base.pth` and `./checkpoints/vqnsp.pth` are
~95 MB each and version controlled, so uploading them to S3 on **every**
submission is pure waste. `sagemaker.weight_s3_uris` maps a local weight path to
the S3 copy of the same bytes; when a weight field points at a mapped path, the
submission mounts that S3 object as the channel and skips the upload. The default
covers the two shipped checkpoints:

```json
{
  "./checkpoints/labram-base.pth": "s3://eeg-data-public/models/labram/labram-base.pth",
  "./checkpoints/vqnsp.pth":       "s3://eeg-data-public/models/labram/vqnsp.pth"
}
```

So the shipped fine-tune / pre-train / codebook configs submit with **no weight
upload** out of the box — the `pretrained` and `tokenizer` channels come straight
from `s3://eeg-data-public/...`. Paths are matched normalized (`./checkpoints/x.pth`
== `checkpoints/x.pth`), and a mirrored path need not exist locally at all.

- Point the mirror at your own bucket: `--set
  sagemaker.weight_s3_uris='{"./checkpoints/labram-base.pth": "s3://my-bucket/labram-base.pth"}'`
  (config-file edit is easier for multi-entry maps).
- Force a fresh local checkpoint to upload again: clear the map with
  `--set sagemaker.weight_s3_uris='{}'`, or just point the weight field at a path
  that isn't in the map.

This is a submit-side convenience only — `weight_s3_uris` never affects
in-container or local (non-SageMaker) training.

`data.split_json` is the exception — it is left as an `s3://` URI and read
directly in-container via the shared `FileSystem` (the execution role's S3
access), so **reusing a recorded split** across runs is just
`--set data.split_json=s3://…/data_split.json` (see
[`cross_validation.md`](cross_validation.md) for how the split is applied).

## What gets packaged as `source_dir`

The estimator tars and uploads the whole `source_dir`, and the repo root also
holds `.venv/`, downloaded `checkpoints/` and local `log/` output — gigabytes
that have no business in a code upload. So when `sagemaker.source_dir` is empty
the CLI builds the package from the **git-tracked files** (working-tree content,
so uncommitted edits ship) into a temp directory, minus model weights
(`*.pth`/`*.pt`/`*.ckpt`/`*.h5`/`*.pkl`) — those travel as input channels. The
temp directory is removed once the jobs are dispatched.

Untracked `.py` files under `labram/` are **not** packaged; the CLI warns about
them by name, so `git add` anything the job needs. Set `sagemaker.source_dir`
explicitly to bypass all of this and upload a directory verbatim.

## Dependencies / `pip install` in the job

The SageMaker training toolkit runs **`pip install -r requirements.txt`** from
the packaged source dir before invoking the entry point. LaBraM's
`requirements.txt`
deliberately does **not** pin `torch`, so the container keeps the DLC's CUDA
torch build and only the remaining libraries (timm, mne, pyhealth, scikit-learn,
`boto3`/`s3fs` for S3 data, `clearml` for tracking, …) are installed at start-up.
To add job-only dependencies, either extend `requirements.txt` or point
`sagemaker.source_dir` at a directory whose `requirements.txt` you control.

## ClearML logging from SageMaker

ClearML logging works inside the job: `clearml` is in `requirements.txt`, and the
per-fold tasks group under `‹project›/‹experiment›` exactly as for a local run.
It only needs credentials in the container. When `clearml.enabled` is set, the
submit CLI **forwards the submitter's ClearML credentials**
(`CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`,
`CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`) into the job's environment, resolving
each from — in order — an explicit `sagemaker.environment` entry, the matching
environment variable, then the local **`clearml.conf`** (which is where
`clearml-init` puts them, so the common setup needs no extra work). If the
access key, secret key and API host still cannot be resolved the CLI warns
before submitting, because the job would otherwise fail to report. The
fold-number parsing in `cv_report` tolerates the `append_timestamp` task-name
suffix.

## Requirements (submitting machine)

AWS credentials able to create training jobs, an execution role (above), and the
SageMaker SDK — which is **not** in `requirements.txt`, because that file is also
what the training container installs and the SDK's dependency tree (mlflow, onnx,
tritonclient, `urllib3>=2`) conflicts with `pyhealth` and does not belong in the
job. Install it separately on the submitting machine:

```bash
pip install -r requirements-sagemaker.txt
```

The generic wrapper imports the SDK lazily, so `--dry_run` and the unit tests
need neither the SDK nor credentials.

## Design notes

- **`labram.aws.sagemaker`** — `SageMakerJobSpec` + `estimator_kwargs()` (a pure
  spec→kwargs mapping) + `SageMakerLauncher` (lazy SDK). Vendored verbatim from
  `common/sagemaker.py` so both repos share one implementation.
- **`labram.runs.submit_sagemaker`** — builds specs from `FinetuneRunConfig`,
  plans the fold jobs (`plan_jobs`, unit-tested via `--dry_run`), uploads the
  config, and submits.
- **`labram.runs.sagemaker_entry`** — the in-container dispatcher.
