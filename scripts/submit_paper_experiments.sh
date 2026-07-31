#!/usr/bin/env bash
# --------------------------------------------------------
# Submit the paper fine-tuning experiments to AWS SageMaker (epochs=20):
#   1. cv          — K-fold cross-validation on the paper config (one job/fold)
#   2. paper_clip  — paper config + gradient clipping
#   3. codebook    — codebook-regularised fine-tune
#   4. labram_pp   — LaBraM++ scenario
#
# Runs 2/3/4 reuse the SAME recorded data_split.json ($SPLIT_JSON) so the models
# train on an identical train/val/test split and are directly comparable.
#
# Usage:
#   # Preview every job plan without contacting AWS:
#   DRY_RUN=1 scripts/submit_paper_experiments.sh
#
#   # Fill in the S3/role settings and submit all experiments:
#   ROLE=arn:aws:iam::123:role/SM DATA=s3://b/TUAB CKPT=s3://b/labram-base.pth \
#   VQNSP=s3://b/vqnsp.pth OUT=s3://b/out \
#   scripts/submit_paper_experiments.sh
#
#   # Submit only some of them:
#   scripts/submit_paper_experiments.sh cv codebook
#
# All knobs are environment variables (defaults shown below).
# --------------------------------------------------------
set -euo pipefail

# ---- settings (override via environment) --------------------------------
ROLE="${ROLE:-}"                                  # SageMaker execution role ARN (required unless DRY_RUN)
DATA="${DATA:-s3://CHANGE-ME/TUAB}"               # TUAB data prefix (train/val/test dirs)
CKPT="${CKPT:-s3://CHANGE-ME/checkpoints/labram-base.pth}"   # pre-trained checkpoint
VQNSP="${VQNSP:-s3://CHANGE-ME/checkpoints/vqnsp.pth}"       # tokenizer (codebook run only)
OUT="${OUT:-s3://CHANGE-ME/labram/out}"           # model output prefix
SPLIT_JSON="${SPLIT_JSON:-s3://clearml-eeg/eeg/finetune_tuab_base_paper_20260725_155351_022.23f8562a2c7b412a86834a7ba910e807/artifacts/data_split/data_split.json}"

EPOCHS="${EPOCHS:-20}"
N_FOLDS="${N_FOLDS:-5}"
CLIP_GRAD="${CLIP_GRAD:-3.0}"                      # grad-clip value for the paper_clip run
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g5.2xlarge}"
CLEARML_PROJECT="${CLEARML_PROJECT:-LaBraM}"
DRY_RUN="${DRY_RUN:-0}"                            # 1 -> print the plan, no AWS calls

CONFIG_DIR="labram/configs/defaults"

# ---- helpers ------------------------------------------------------------
DRY_FLAG=()
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_FLAG=(--dry_run)
elif [[ -z "${ROLE}" ]]; then
  echo "ERROR: set ROLE=arn:aws:iam::<acct>:role/<name> (or DRY_RUN=1 to preview)." >&2
  exit 1
fi

# Common --set tokens shared by every job, one per line (read into an array).
common_sets() {  # common_sets <output_subdir> <job_name_prefix>
  cat <<EOF
sagemaker.enabled=true
sagemaker.role=${ROLE}
sagemaker.instance_type=${INSTANCE_TYPE}
sagemaker.output_path=${OUT}/${1}
sagemaker.job_name_prefix=${2}
data.data_path=${DATA}
finetune_checkpoint.finetune=${CKPT}
trainer.epochs=${EPOCHS}
clearml.enabled=true
clearml.project_name=${CLEARML_PROJECT}
EOF
}

submit() {   # submit <config> <set-token...>
  local config="$1"; shift
  echo "=============================================================="
  echo ">> ${config}"
  echo "=============================================================="
  python -m labram.runs.submit_sagemaker --config "${config}" --set "$@" "${DRY_FLAG[@]}"
}

run_cv() {
  local sets
  mapfile -t sets < <(common_sets "paper_cv${N_FOLDS}" "labram-paper-cv")
  sets+=(
    clearml.task_name=finetune_tuab_paper
    cross_validation.enabled=true
    "cross_validation.n_folds=${N_FOLDS}"
    cross_validation.split_by=subject
  )
  submit "${CONFIG_DIR}/finetune_tuab_paper.json" "${sets[@]}"
}

run_paper_clip() {
  local sets
  mapfile -t sets < <(common_sets "paper_gradclip" "labram-paper-gradclip")
  sets+=(
    clearml.task_name=paper_gradclip
    "data.split_json=${SPLIT_JSON}"
    "optimizer.clip_grad=${CLIP_GRAD}"
  )
  submit "${CONFIG_DIR}/finetune_tuab_paper.json" "${sets[@]}"
}

run_codebook() {
  local sets
  mapfile -t sets < <(common_sets "codebook" "labram-codebook")
  sets+=(
    clearml.task_name=codebook
    "data.split_json=${SPLIT_JSON}"
    "model.codebook_reg.tokenizer_weight=${VQNSP}"
  )
  submit "${CONFIG_DIR}/finetune_tuab_codebook.json" "${sets[@]}"
}

run_labram_pp() {
  local sets
  mapfile -t sets < <(common_sets "labram_pp" "labram-plus-plus")
  sets+=(
    clearml.task_name=labram_plus_plus
    "data.split_json=${SPLIT_JSON}"
  )
  submit "${CONFIG_DIR}/finetune_tuab_labram_plus_plus.json" "${sets[@]}"
}

# ---- dispatch -----------------------------------------------------------
EXPERIMENTS=("$@")
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=(cv paper_clip codebook labram_pp)
fi

for exp in "${EXPERIMENTS[@]}"; do
  case "${exp}" in
    cv)         run_cv ;;
    paper_clip) run_paper_clip ;;
    codebook)   run_codebook ;;
    labram_pp)  run_labram_pp ;;
    *) echo "Unknown experiment '${exp}' (choose: cv paper_clip codebook labram_pp)" >&2; exit 2 ;;
  esac
done

echo "Done."
