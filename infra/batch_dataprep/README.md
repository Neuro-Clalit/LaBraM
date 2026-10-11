# batch-dataprep: AWS Batch on Spot for EEG data preparation

Converts EEG corpora EDF -> per-recording npy (`dataset_maker/make_TUH_npy.py`) as
**array jobs on EC2 Spot**: one child per duration-balanced shard of a corpus scan,
reading EDFs from and writing npy to `s3://eeg-data-public/...`. The compute
environment scales to zero vCPUs when idle, so it costs nothing between runs.

Why Batch rather than SageMaker Processing: Processing jobs cannot use Spot; Batch
runs Spot natively, retries children whose Spot host is reclaimed, and has no
SageMaker price premium. TUEG (~20k hours of EEG) is about 100 vCPU-hours.

## Resources (`template.yaml`, stack `batch-dataprep`)

| resource | purpose |
|---|---|
| ECR `labram-dataprep` | the data-prep image (`Dockerfile`: python 3.10, mne 1.4.2, numpy 1.26.4, scipy 1.15.3 -- same filter implementation as the training environment) |
| compute env `labram-dataprep-spot` | EC2 Spot, `SPOT_PRICE_CAPACITY_OPTIMIZED`, r/m families, 0..`MaxVcpus` (256) vCPUs |
| job queue `labram-dataprep` | |
| job definition `labram-dataprep-tuh-npy` | 4 vCPU / 30 GB per child (the longest TUEG recordings need ~24 GB while filtering; the converter filters them one at a time), 8 h per attempt, up to 5 attempts on Spot reclaim |
| IAM roles `labram-dataprep-{instance,job,execution}` | ECS host role; job role limited to `s3:GetObject/PutObject/ListBucket` on the data bucket; ECR pull + logs |

## Deploy (needs an MFA session)

The account's `Admin-MFA-Enforcement` policy denies CloudFormation and the IAM/ECR/EC2
calls it makes on the caller's behalf unless the session has MFA:

```bash
MFA_SERIAL=arn:aws:iam::574441342949:mfa/<device> infra/batch_dataprep/deploy.sh
IMAGE_ONLY=1 infra/batch_dataprep/deploy.sh      # after converter code changes
```

## Run

```bash
python scripts/submit_batch_npy.py submit --corpus tueg \
    --scan /data/datasets/EEG-public/TUH_EEG/_state/tueg/recordings.parquet \
    --notch 60 --shards 64                       # --dry-run to preview
python scripts/submit_batch_npy.py status --job-id <parent id>
python -m dataset_maker.make_TUH_npy merge \
    --scan /data/datasets/EEG-public/TUH_EEG/_state/tueg/recordings.parquet \
    --dst s3://eeg-data-public/TUH_EEG/tueg/v2.0.2/processed_npy_notch60_float16
```

Output goes to `processed_npy_notch<notch>_<dtype>/` so recipes never mix. A rerun
skips recordings already written with the same notch; per-recording failures are
listed in the manifest's `errors`.
