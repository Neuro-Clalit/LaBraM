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
        sagemaker.input_mode=FastFile sagemaker.use_spot=true \
        sagemaker.max_wait_min=1500 sagemaker.wait=true \
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
| `max_run_sec` | `86400` (24h) | Hard wall-clock cap per job — SageMaker **stops** the job at the cap, bounding what one submission can cost. Raise it explicitly for long pre-training runs. |
| `use_spot` / `max_wait_min` | `false` / `0.0` | Managed spot training. `max_wait_min` is a whole or fractional number of minutes (`90` and `90.0` are both valid; `1.5` = 90 seconds) and covers queue time **and** run time; `0` → reuse `max_run_sec`. |
| `on_demand_fallback` | `false` | When a spot job ends with `MaxWaitTimeExceeded`, resubmit it on-demand. The submitter stays attached to observe the result; this option is ignored with `--detach`. |
| `framework_version` / `py_version` | `2.4.0` / `py311` | Managed PyTorch DLC selectors (2.4.0 is a published DLC; 2.4.1 is not). |
| `image_uri` | `''` | Explicit training image (overrides the managed DLC). |
| `entry_point` / `source_dir` | `labram/runs/sagemaker_entry.py` / repo root | Training script and packaged code. |
| `job_name_prefix` | `labram-finetune` | Prefix; the fold number is appended for CV. |
| `region` | `''` | AWS region; `''` → boto3 default. |
| `profile` | `''` | AWS credential profile to submit from; `''` → the checkout's `.aws-profile` file if it has one, else boto3's own resolution. Must name the same account as `role` — see [Cross-account role](#cross-account-role). |
| `output_path` / `code_location` | `''` | S3 prefixes for model artifacts / packaged source. |
| `output_kms_key` | `''` | KMS key for the S3 objects the submission writes (model output + the code/config/weight uploads). `''` → plain uploads and the SDK/account default for the job output (see [KMS-encrypted buckets](#kms-encrypted-buckets--mfa-enforced-accounts)). |
| `config_channel` | `''` | Pre-uploaded config S3 uri; `''` → the CLI uploads it. |
| `weight_s3_uris` | `{./checkpoints/labram-base.pth → s3://eeg-data-public/models/labram/labram-base.pth, ./checkpoints/vqnsp.pth → s3://…/vqnsp.pth}` | Local weight paths whose bytes already live in S3; the mirror is mounted as a channel instead of the local file being uploaded (see below). |
| `input_mode` | `File` | Channel delivery: `File`, `FastFile` or `Pipe` (see below). |
| `environment` / `hyperparameters` / `tags` | `{}` | Extra container env vars / hyperparameters / job tags. |
| `wait` | `false` | Block until the (last) job finishes. The job runs on AWS either way — waiting only keeps the local process attached. |
| `stream_logs` | `true` | While waiting, stream the job's CloudWatch logs locally. `false` waits quietly; `--detach` turns this *and* `wait` off. |

Unknown `--set` keys are rejected, so a typo in one of these names fails the
submission instead of being silently dropped.

## On-demand vs. spot training

The hardware is identical; the purchase contract is not.

| | On-demand | Managed spot |
|---|---|---|
| Allocation | A dedicated instance from AWS's regular pool. | AWS spare capacity. |
| Start | Usually immediate (an on-demand job can acquire hardware in seconds). | Only when capacity exists. A low placement score (for example `1/10` for `g5.2xlarge`) means it may wait for hours or never start. |
| During training | The instance remains yours until the job ends. | AWS can interrupt it with two minutes' notice. Persist checkpoints frequently so training can resume from the latest checkpoint. |
| `ml.g5.2xlarge`, `us-east-1` | About `$1.52/hr`. | Usually about 60–70% cheaper (roughly `$0.45–0.60/hr`); time spent waiting is free. |

Use on-demand when a prompt start or uninterrupted run matters:

```bash
python -m labram.runs.submit_sagemaker --phase finetune \
  --config labram/configs/defaults/finetune_tuab.json \
  --set sagemaker.enabled=true sagemaker.role=arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        sagemaker.instance_type=ml.g5.2xlarge sagemaker.use_spot=false
```

For spot, choose a window that includes both the maximum training duration and
the queue delay you are willing to tolerate. This example permits a 24-hour run
plus six hours of capacity wait (`1800` minutes):

```bash
python -m labram.runs.submit_sagemaker --phase finetune \
  --config labram/configs/defaults/finetune_tuab.json \
  --set sagemaker.enabled=true sagemaker.role=arn:aws:iam::<account-id>:role/SageMakerExecutionRole \
        sagemaker.instance_type=ml.g5.2xlarge sagemaker.use_spot=true \
        sagemaker.max_wait_min=1800 sagemaker.on_demand_fallback=true
```

With `on_demand_fallback=true`, a `MaxWaitTimeExceeded` spot result submits the
same job again on-demand. It cannot work with `--detach`, because the local
submitter must wait long enough to observe the spot result. This fallback only
handles failure to obtain spot capacity; it does not turn an in-progress spot
job into on-demand after an interruption.

SageMaker rejects a positive `max_wait_min` that is shorter than
`max_run_sec`: it is the **total** spot-job window, not a separate queue-only
timeout. With the default 24-hour `max_run_sec=86400`, the minimum is `1440`.
For a 24-hour run plus up to 90 minutes for capacity, use
`sagemaker.max_wait_min=1530`. The submitter checks this before it uploads code
or creates a job and reports the required value.

When splitting a command across lines, the `\` must be the final character on
the line. For example:

```bash
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set sagemaker.enabled=true \
        sagemaker.role=arn:aws:iam::574441342949:role/SageMakerExecutionRole \
        sagemaker.instance_type=ml.g5.2xlarge \
        sagemaker.input_mode=FastFile sagemaker.use_spot=true \
        sagemaker.max_wait_min=1530 sagemaker.on_demand_fallback=true \
        sagemaker.job_name_prefix=labram-brain-age sagemaker.wait=true \
        data.data_path=s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed/ \
        output.output_dir= output.log_dir= \
        clearml.enabled=true clearml.project_name=eeg/brain_age \
        clearml.task_name=finetune_tuab_age
```

### Check capacity and quotas before waiting

`scripts/check_sagemaker_capacity.py` is read-only: it reports SageMaker
on-demand/spot quotas, EC2's 1–10 spot placement score, available AZ offerings,
and recent training-job outcomes. It cannot reserve capacity, but it is the
best preflight signal before submitting a spot job.

```bash
python scripts/check_sagemaker_capacity.py \
  --types ml.g5.2xlarge,ml.g6.2xlarge --profile neuro --region us-east-1
```

If the report shows a quota of zero (or you need parallel jobs), request an
increase using the quota code printed by the script:

```bash
aws service-quotas request-service-quota-increase \
  --service-code sagemaker --quota-code <code-reported-by-script> \
  --desired-value <concurrent-instances> --profile neuro --region us-east-1
```

To inspect the same quotas directly with the AWS CLI (without running the
helper), list all SageMaker quotas or filter for a specific training instance
type. The matching on-demand and spot rows include the `QuotaCode` needed for a
request:

```bash
aws service-quotas list-service-quotas \
  --service-code sagemaker --region us-east-1 --profile neuro \
  --query "Quotas[?contains(QuotaName, 'ml.g5.2xlarge')].[QuotaName,Value,QuotaCode,Adjustable]" \
  --output table
```

The AWS console's SageMaker quota page is also available at
`https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas`
(select the same region). To open it from macOS:

```bash
open 'https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas?region=us-east-1'
```

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

#### Single-object channels are always delivered as `File`

Fast file mode "supports S3 prefixes **only**": it mounts the channel's S3 uri as
a *prefix* and exposes the keys **beneath** it. Two of the channels address a
single object rather than a prefix — `config` (the uploaded `run_config.yaml`) and
the weight channels (`pretrained` / `tokenizer`) — so under `FastFile` their
mounts come up **empty** and the job dies on a file that was never there:

```
RuntimeError: Non valid  path: /opt/ml/input/data/config/run_config.yaml
```

The submitter prevents this: channels that address one object get a per-channel
`File` `InputMode` (via `TrainingInput`), which restores the
`‹mount›/‹basename›` layout the container expects. They are small, so this costs
one quick download, and the big `dataset` channel keeps streaming — which is the
point of `FastFile`. `channel_input_modes()` decides this, and the submit log
names the channels it overrode. Nothing to configure.

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

### Cross-account role

The execution role must live in the **same AWS account as the credentials you
submit with**. `CreateTrainingJob` refuses otherwise:

```
ClientError (ValidationException) … CreateTrainingJob: RoleArn: Cross-account
pass role is not allowed.
```

No trust policy can grant this — it is a hard SageMaker restriction, not a
permission you are missing. The usual cause is a shell whose `AWS_PROFILE` is
unset (so boto3 silently uses `default`) while `sagemaker.role` names a second
account. Before uploading anything, the CLI now logs the caller identity and
fails with both accounts spelled out:

```
[INFO] SageMaker execution role: arn:aws:iam::574441342949:role/SageMakerExecutionRole
[INFO] AWS caller identity: arn:aws:iam::660185423351:user/leon (account 660185423351)
Cross-account SageMaker execution role.
  role    arn:aws:iam::574441342949:role/SageMakerExecutionRole
          -> account 574441342949
  caller  arn:aws:iam::660185423351:user/leon
          -> account 660185423351 (from your current credentials)
```

Fix it by submitting from the role's account, either per-shell or pinned in the
config so it cannot drift:

```bash
AWS_PROFILE=neuro python -m labram.runs.submit_sagemaker --config …
# or, travelling with the config:
python -m labram.runs.submit_sagemaker --config … --set sagemaker.profile=neuro
```

`sagemaker.profile` is passed to `boto3.Session(profile_name=…)`, so it also
determines which account's default bucket (`sagemaker-<region>-<account>`)
receives the config, code and weight uploads. If STS cannot be reached the check
is skipped rather than blocking the submission.

#### `.aws-profile`: the checkout's account

When `sagemaker.profile` is empty, the submitter walks up from the repo root for
a `.aws-profile` file holding a profile name and uses that, logging where it
came from:

```
[INFO] Using AWS profile 'neuro' from /path/to/LaBraM/.aws-profile (set sagemaker.profile to override).
```

This is the same convention as a `chpwd` shell hook that exports `AWS_PROFILE`
per repository, but reading the file directly also covers the contexts where
such a hook never runs — PyCharm run configurations, cron, `ssh host 'cmd'`.

It is also strictly more reliable than `AWS_PROFILE`. **Exported
`AWS_ACCESS_KEY_ID` / `AWS_SESSION_TOKEN` credentials outrank `AWS_PROFILE`** in
botocore's chain, so an MFA session minted from another account silently wins —
`aws sts get-caller-identity` then contradicts the `AWS_PROFILE` you set. A
profile handed to `boto3.Session(profile_name=…)` does not have that problem:
botocore drops the environment provider when the profile is set programmatically.
If you keep per-account MFA sessions, cache them per profile
(`session-<profile>.env`) and clear a session belonging to a different profile
when you switch.

### KMS-encrypted buckets / MFA-enforced accounts

Some accounts attach a policy that **denies `kms:GenerateDataKey`** unless the
session is MFA-backed (e.g. an `Admin-MFA-Enforcement` policy), and set a default
KMS key on the SageMaker session bucket via `sagemaker.config`. A submission then
fails with:

```
S3UploadFailedError: Failed to upload …/source.tar.gz … An error occurred
(AccessDenied) when calling the PutObject operation: User … is not authorized to
perform: kms:GenerateDataKey … with an explicit deny in an identity-based
policy: …/Admin-MFA-Enforcement
```

The tell is that the code, config and weight uploads **succeed** and only the
estimator's own source-code upload fails — because that upload inherits the
account-default `output_kms_key`, while the CLI's uploads use no customer key.
LaBraM already sidesteps this by packaging and uploading the code itself (so the
estimator gets an `s3://` `source_dir` and skips its KMS upload), so a plain
`submit_sagemaker` no longer hits this. If you still see a `kms:GenerateDataKey`
`AccessDenied` (e.g. your output/session bucket *mandates* a CMK), fix it one of
these ways:

- **Submit with MFA-backed credentials** — assume a role / get session-token
  credentials that satisfy the MFA condition, then submit.
- **Use a key you are allowed to use** — `--set sagemaker.output_kms_key=arn:aws:kms:…:key/…`.
  It is applied to *all* the submission's S3 writes (code, config, weights, and
  the job's model output).
- **Write to a bucket without a mandatory CMK** — point
  `--set sagemaker.output_path=s3://my-bucket/…` at a bucket whose default
  encryption you can satisfy.

Note the job's **model output** is written by the *execution role* at runtime,
not by you, so an account-default output key it can use is fine and left in place.

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

The code is tarred and uploaded before the job starts, and the repo root also
holds `.venv/`, downloaded `checkpoints/` and local `log/` output — gigabytes
that have no business in a code upload. So when `sagemaker.source_dir` is empty
the CLI builds the package from the **git-tracked files** (working-tree content,
so uncommitted edits ship) into a temp directory, minus model weights
(`*.pth`/`*.pt`/`*.ckpt`/`*.h5`/`*.pkl`) — those travel as input channels. The
temp directory is removed once the tarball is uploaded.

Untracked `.py` files under `labram/` are **not** packaged; the CLI warns about
them by name, so `git add` anything the job needs. Set `sagemaker.source_dir`
explicitly to bypass all of this and upload a directory verbatim.

The CLI packages that directory into `sourcedir.tar.gz` and **uploads it itself**
(same S3 path as the config and weights), then hands the estimator an
`s3://…/sourcedir.tar.gz` `source_dir` — the SDK then skips its own code upload.
This is deliberate: the SDK's built-in upload inherits the estimator's
`output_kms_key`, which an account-level `sagemaker.config` default can set to a
customer KMS key the *submitting* identity isn't allowed to use, so it would fail
where the config/weight uploads succeed (see
[KMS-encrypted buckets](#kms-encrypted-buckets--mfa-enforced-accounts)).

## Dependencies / `pip install` in the job

The SageMaker training toolkit runs **`pip install -r requirements.txt`** from
the packaged source dir before invoking the entry point. LaBraM's
`requirements.txt`
deliberately does **not** pin `torch`, so the container keeps the DLC's CUDA
torch build and only the remaining libraries (timm, mne, pyhealth, scikit-learn,
`boto3`/`s3fs` for S3 data, `clearml` for tracking, …) are installed at start-up.
To add job-only dependencies, either extend `requirements.txt` or point
`sagemaker.source_dir` at a directory whose `requirements.txt` you control.

## After submission: the job is independent of your terminal

As soon as a job is created the CLI prints a banner with its **real** (timestamped)
job name, the instance, the runtime cap, the git commit, and a console link — before
any log streaming, so the name is on screen even if the wait is later interrupted.

The banner exists to make one thing unmissable: **the job runs on AWS, not in your
terminal.** With `sagemaker.wait=true` the local process is only tailing CloudWatch;
Ctrl-C, closing the terminal, or losing the connection stops the *log tail* only —
training continues and still writes checkpoints to S3 and metrics to ClearML. An
interrupt during a wait prints a second banner repeating that, with the job names
still running. To actually stop a job:

```bash
aws sagemaker stop-training-job --training-job-name ‹name›
```

### Submit and exit: `--detach`

```bash
python -m labram.runs.submit_sagemaker --config ‹cfg› --detach --set ...
```

`--detach` creates the job(s), prints the banner, and returns — it waits for
nothing and streams no logs, overriding `sagemaker.wait`. Use it for long runs, or
from a script that should not hold a terminal open. To wait but without the log
firehose, keep `wait=true` and set `--set sagemaker.stream_logs=false`.

### Runtime cap

`max_run_sec` defaults to **24h** and SageMaker *stops* the job when it is reached,
so a hung or diverging run cannot burn a GPU for days. Long pre-training needs it
raised explicitly (`--set sagemaker.max_run_sec=345600` for 4 days). With
`use_spot`, `max_wait_min=0` reuses the same value.

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

### Git provenance (branch / commit / uncommitted changes)

ClearML fills an experiment's *code* section by finding a `.git` next to the
running script. A SageMaker job has none — the packaged source is built from
`git ls-files`, so `.git` is deliberately absent — which would leave every
submitted experiment with **no branch, no commit and no record of uncommitted
edits**.

So the submitter captures that metadata from your checkout
(`labram/utils/git_info.py`), ships it inside the source tarball as
`labram_git_info.json`, and the in-container run replays it onto the task with
`Task.set_script()`, so ClearML shows:

- **repository** (with any embedded credentials stripped from the URL), **branch**
  and **full commit sha**;
- the **uncommitted diff** — because the job runs your *working tree*, not the
  commit — capped at 256 KiB, plus the modified/untracked file lists as a
  searchable `git` parameter section.

The submit log states the commit and warns when the tree is dirty. Local runs are
untouched: they have a real `.git`, ship no metadata file, and keep ClearML's own
detection. Tracking never fails a run — every step degrades to a warning.

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
  config, weights and code tarball (`package_and_upload_source`, so the estimator
  skips its `output_kms_key`-inheriting upload), and submits.
  `_reraise_kms_access_denied` turns a residual `kms:GenerateDataKey` denial into
  an actionable error.
- **`labram.runs.sagemaker_entry`** — the in-container dispatcher.
