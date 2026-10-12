#!/usr/bin/env bash
# TUH cohort brain-age experiments on SageMaker, M1 baseline (scenario D +
# mixup 0.4, 4 epochs, 1 warmup). Cohorts are built by
# dataset_maker/make_cohort_dataset.py and copied to s3://.../TUH_EEG/cohorts/.
#   exp2: normal_tueg_v1 -- healthy TUEG patients outside TUAB, train + val, no test
#   exp3: all_both_v1    -- every recording, TUAB + TUEG, test = TUAB eval
#   exp3b: exp3 on a longer schedule (10 epochs, 3 warmup)
#   DRY_RUN=1 scripts/submit_tuh_cohorts.sh exp2 exp3
set -euo pipefail
ROLE="${ROLE:-arn:aws:iam::574441342949:role/SageMakerExecutionRole}"
PY="${PY:-python}"
S3=s3://eeg-data-public/TUH_EEG/cohorts
declare -A COHORT=([exp2]=normal_tueg_v1 [exp3]=all_both_v1 [exp3b]=all_both_v1)
declare -A EPOCHS=([exp2]=4 [exp3]=4 [exp3b]=10)
declare -A WARMUP=([exp2]=1 [exp3]=1 [exp3b]=3)
declare -A VOLUME=([exp2]=120 [exp3]=590 [exp3b]=590)          # GB: File mode copies the cohort first
# exp3 (459 GB) exceeds the 450 GB local disk of ml.g5.2xlarge; ml.g5.4xlarge has the
# same A10G GPU and 600 GB of NVMe.
declare -A INSTANCE=([exp2]=ml.g5.2xlarge [exp3]=ml.g5.4xlarge [exp3b]=ml.g5.4xlarge)
mode=(--detach); [[ "${DRY_RUN:-0}" == "1" ]] && mode=(--dry_run)
for exp in "$@"; do
  cohort=${COHORT[$exp]}
  name="tuh_${exp}_${cohort%_v1}_M1_ep${EPOCHS[$exp]}"
  "${PY}" -m labram.runs.submit_sagemaker --config labram/configs/defaults/finetune_tuab_age.json "${mode[@]}" --set \
    sagemaker.enabled=true sagemaker.role="${ROLE}" sagemaker.instance_type="${INSTANCE[$exp]}" \
    sagemaker.input_mode=File sagemaker.volume_size_gb="${VOLUME[$exp]}" \
    sagemaker.wait=false sagemaker.stream_logs=false \
    sagemaker.job_name_prefix="labram-tuh-${exp}-${cohort//_/-}-ep${EPOCHS[$exp]}" \
    data.data_format=npy data.data_path="${S3}/${cohort}/" \
    trainer.epochs="${EPOCHS[$exp]}" optimizer.warmup_epochs="${WARMUP[$exp]}" \
    mixup.enabled=true mixup.alpha=0.4 mixup.sigma=0.0 \
    output.output_dir= output.log_dir= \
    clearml.enabled=true clearml.project_name=LaBraM/brain_age clearml.task_name="${name}" \
    "clearml.tags=[\"brain_age\",\"tuh_cohorts\",\"${cohort%_v1}\",\"M1\",\"ep${EPOCHS[$exp]}\",\"sagemaker\"]" \
    2>&1 | grep -v -i warn | tail -n 2
done
