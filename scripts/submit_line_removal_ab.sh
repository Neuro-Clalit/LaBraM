#!/usr/bin/env bash
# Line-removal A/B on TUAB v3.0.1 as SageMaker jobs: the same short schedule as
# the local seed-0 runs (4 epochs, 1 warmup), only the npy preprocessing and the
# seed differ. A = 50 Hz notch (LaBraM / published), B = 60 Hz notch,
# C = 50 + 60 Hz spectrum interpolation (notebooks/tuh_preprocessing_analysis.ipynb).
#
#   SEEDS="1 2" scripts/submit_line_removal_ab.sh          # submit (max MAX_INFLIGHT at once)
#   DRY_RUN=1 SEEDS=1 scripts/submit_line_removal_ab.sh    # preview
set -euo pipefail
ROLE="${ROLE:-arn:aws:iam::574441342949:role/SageMakerExecutionRole}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g5.2xlarge}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
SEEDS="${SEEDS:-1 2}"
DRY_RUN="${DRY_RUN:-0}"
S3=s3://eeg-data-public/TUH_EEG/tuab/v3.0.1/edf
declare -A DATA=([A_notch50]=${S3}/processed_npy/ [B_notch60]=${S3}/processed_npy_notch60/
                 [C_interp5060]=${S3}/processed_npy_interp50-60/)
declare -A CONFIG=([age]=labram/configs/defaults/finetune_tuab_age.json
                   [abn]=labram/configs/defaults/finetune_tuab.json)
PY="${PY:-python}"

wait_for_slot() {
  [[ "${DRY_RUN}" == "1" ]] && return
  while true; do
    n=$(aws sagemaker list-training-jobs --status-equals InProgress --no-paginate \
          --query "length(TrainingJobSummaries)" --output text 2>/dev/null || echo 99)
    n=${n//[^0-9]/}; n=${n:-99}
    [[ "${n}" -lt "${MAX_INFLIGHT}" ]] && return
    sleep 120
  done
}

for seed in ${SEEDS}; do
  for task in age abn; do
    for v in A_notch50 B_notch60 C_interp5060; do
      name="ab_${task}_${v}_seed${seed}"
      wait_for_slot
      echo "[$(date -u +%FT%TZ)] submitting ${name}"
      mode=(--detach); [[ "${DRY_RUN}" == "1" ]] && mode=(--dry_run)
      "${PY}" -m labram.runs.submit_sagemaker --config "${CONFIG[$task]}" "${mode[@]}" --set \
        sagemaker.enabled=true sagemaker.role="${ROLE}" sagemaker.instance_type="${INSTANCE_TYPE}" \
        sagemaker.input_mode=File sagemaker.wait=false sagemaker.stream_logs=false \
        sagemaker.job_name_prefix="labram-ab-${task}-${v//_/-}-s${seed}" \
        data.data_format=npy data.data_path="${DATA[$v]}" \
        trainer.epochs=4 optimizer.warmup_epochs=1 distributed.seed="${seed}" \
        output.output_dir= output.log_dir= \
        clearml.enabled=true clearml.project_name=LaBraM/preprocessing_ab clearml.task_name="${name}" \
        "clearml.tags=[\"line_removal_ab\",\"${task}\",\"${v}\",\"seed${seed}\",\"sagemaker\"]" \
        2>&1 | grep -v -i "warn" | tail -n 3
      [[ "${DRY_RUN}" == "1" ]] || sleep 60     # let it register as InProgress
    done
  done
done
echo "[$(date -u +%FT%TZ)] all line-removal A/B jobs submitted"
