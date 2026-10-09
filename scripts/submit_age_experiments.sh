#!/usr/bin/env bash
# --------------------------------------------------------
# Submit the brain-age (TUAB_AGE) fine-tuning ablations to AWS SageMaker:
#   A   epochs=15, lr=1e-4, warmup=2, evaluated on whole recordings
#   B   A + data.eval_minutes=5
#   C   B + data.window_sec=30            (batch 32 x update_freq 2 = 64)
#   C2  B + layer_decay 0.5, drop_path 0.2, drop 0.1, weight_decay 0.1, clip_grad 1.0
#   D   B + common average reference only (labram_plus, no per-patch z-score)
#   D2  B + L1 loss
#
# Scenario-D follow-ups (D is the default config; set EXP_TAG=age_ablation_D_2026-10):
#   DX1 D + amplitude reconstruction term on half the spectrum (codebook path)
#   DX2 D + Huber delta 2
#   DX3 D + layer decay 0.5, drop path 0.2, dropout 0.1, wd 0.1, clip 1.0 (run locally)
#   DX4 D + quantization (embedding) term (codebook path)
#
# finetune_tuab_age.json now defaults to scenario D (CAR on); the ablations pin
# labram_plus.enabled=false so A-D2 reproduce as originally run, and D turns it
# back on.
#
# All runs share one train/val/test split: every job builds it from the dataset's
# saved, seeded processed/age_split.json (subject-disjoint), and records the
# result as its data_split ClearML artifact. All trim 60 s at each end (the age
# config default).
#
# Usage:
#   DRY_RUN=1 scripts/submit_age_experiments.sh            # preview every plan
#   scripts/submit_age_experiments.sh                       # submit all, detached
#   scripts/submit_age_experiments.sh A B                   # a subset
#   SEQUENTIAL=1 scripts/submit_age_experiments.sh          # one job at a time
#
# SEQUENTIAL=1 waits for each job to finish before submitting the next -- for an
# on-demand quota of one concurrent instance. Keep the shell alive (nohup/tmux).
# --------------------------------------------------------
set -euo pipefail

ROLE="${ROLE:-arn:aws:iam::574441342949:role/SageMakerExecutionRole}"
# DATA_FORMAT=npy (default) reads the per-recording float32 files (processed_npy/,
# ~75 GB, 2,990 objects) and copies them to the instance before training (File
# mode): same samples, ~2.9x faster than pickle on scenario D. DATA_FORMAT=pickle
# streams the 409k window pickles with FastFile, as the first ablations did.
DATA_FORMAT="${DATA_FORMAT:-npy}"
if [[ "${DATA_FORMAT}" == "npy" ]]; then
  DATA="${DATA:-s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed_npy/}"
  INPUT_MODE="${INPUT_MODE:-File}"
else
  DATA="${DATA:-s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed/}"
  INPUT_MODE="${INPUT_MODE:-FastFile}"
fi
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g5.2xlarge}"
USE_SPOT="${USE_SPOT:-false}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-0}"                 # spot only: total window, >= max run
CLEARML_PROJECT="${CLEARML_PROJECT:-LaBraM/brain_age}"
EXP_TAG="${EXP_TAG:-age_ablation_2026-10}"
DRY_RUN="${DRY_RUN:-0}"
SEQUENTIAL="${SEQUENTIAL:-0}"
CONFIG="labram/configs/defaults/finetune_tuab_age.json"

MODE_FLAG=(--detach)
WAIT=false
if [[ "${DRY_RUN}" == "1" ]]; then
  MODE_FLAG=(--dry_run)
elif [[ "${SEQUENTIAL}" == "1" ]]; then
  MODE_FLAG=()
  WAIT=true
fi

common_sets() {  # common_sets <experiment>
  cat <<EOF
sagemaker.enabled=true
sagemaker.role=${ROLE}
sagemaker.instance_type=${INSTANCE_TYPE}
sagemaker.input_mode=${INPUT_MODE}
data.data_format=${DATA_FORMAT}
sagemaker.use_spot=${USE_SPOT}
sagemaker.max_wait_min=${MAX_WAIT_MIN}
sagemaker.job_name_prefix=labram-age-$(echo "$1" | tr '[:upper:]' '[:lower:]')
sagemaker.wait=${WAIT}
sagemaker.stream_logs=false
data.data_path=${DATA}
output.output_dir=
output.log_dir=
clearml.enabled=true
clearml.project_name=${CLEARML_PROJECT}
clearml.tags=["brain_age","${EXP_TAG}","exp_$1"]
trainer.epochs=15
optimizer.lr=1e-4
optimizer.warmup_epochs=2
data.eval_minutes=5
labram_plus.enabled=false
EOF
}

submit() {  # submit <experiment> <task_name> [extra set tokens...]
  local exp="$1" task="$2"; shift 2
  local sets
  mapfile -t sets < <(common_sets "${exp}")
  sets+=("clearml.task_name=${task}" "$@")
  echo "=============================================================="
  echo ">> ${exp}: ${task}"
  echo "=============================================================="
  python -m labram.runs.submit_sagemaker --config "${CONFIG}" "${MODE_FLAG[@]}" --set "${sets[@]}"
}

run_A()  { submit A  age_A_ep15_lr1e-4_fullrec data.eval_minutes=0; }
run_B()  { submit B  age_B_eval5min; }
run_C()  { submit C  age_C_eval5min_win30s data.window_sec=30 \
                     trainer.batch_size=32 trainer.update_freq=2; }
run_C2() { submit C2 age_C2_eval5min_regularized optimizer.layer_decay=0.5 \
                     model.drop_path=0.2 model.drop=0.1 optimizer.weight_decay=0.1 \
                     optimizer.clip_grad=1.0; }
run_D()  { submit D  age_D_eval5min_car labram_plus.enabled=true \
                     labram_plus.z_score_patches=false; }
run_D2() { submit D2 age_D2_eval5min_l1 loss.regression_loss=l1; }

# D-based runs re-enable CAR over the A-D2 default above. The codebook runs keep
# D's encoder training (all blocks, layer decay 0.65; lr_scale must stay < 1) and
# freeze the VQNSP quantizer/decoder, which act as fixed regularizers.
CODEBOOK=(model.codebook_reg.enabled=true model.codebook_reg.tokenizer_weight=./checkpoints/vqnsp.pth
          model.codebook_reg.encoder.n_last_trainable_layers=None
          model.codebook_reg.encoder.lr_scale=0.99
          model.codebook_reg.classifier_weight=1.0 model.codebook_reg.phase_weight=0.0)
run_DX1() { submit DX1 age_DX1_amp_freq05 labram_plus.enabled=true "${CODEBOOK[@]}" \
                     model.codebook_reg.amplitude_weight=1.0 \
                     model.codebook_reg.embedding_weight=0.0 loss.freq_fraction=0.5; }
run_DX2() { submit DX2 age_DX2_huber_delta2 labram_plus.enabled=true loss.huber_delta=2.0; }
run_DX3() { submit DX3 age_DX3_regularized labram_plus.enabled=true optimizer.layer_decay=0.5 \
                     model.drop_path=0.2 model.drop=0.1 optimizer.weight_decay=0.1 \
                     optimizer.clip_grad=1.0; }
run_DX4() { submit DX4 age_DX4_quant labram_plus.enabled=true "${CODEBOOK[@]}" \
                     model.codebook_reg.amplitude_weight=0.0 \
                     model.codebook_reg.embedding_weight=1.0; }

# Short scenario-D single-factor runs (4 epochs, 1 warmup epoch -- E2 showed the
# best epoch is reached by then). Each changes exactly one knob of D so the
# collapsed C2/DX3 bundle can be attributed (F), the head init bottleneck tested
# (G), and the layer-decay direction probed (H).
SHORT=(labram_plus.enabled=true trainer.epochs=4 optimizer.warmup_epochs=1)
run_F1() { submit F1 age_F1_short_ld05   "${SHORT[@]}" optimizer.layer_decay=0.5; }
run_F2() { submit F2 age_F2_short_drop01 "${SHORT[@]}" model.drop=0.1; }
run_F3() { submit F3 age_F3_short_dp02   "${SHORT[@]}" model.drop_path=0.2; }
run_F4() { submit F4 age_F4_short_wd01   "${SHORT[@]}" optimizer.weight_decay=0.1; }
run_G1() { submit G1 age_G1_short_init1  "${SHORT[@]}" model.init_scale=1.0; }
run_H1() { submit H1 age_H1_short_ld075  "${SHORT[@]}" optimizer.layer_decay=0.75; }
run_H2() { submit H2 age_H2_short_ld085  "${SHORT[@]}" optimizer.layer_decay=0.85; }

# Anti-memorization runs on the short schedule (M1 = mixup runs locally; see
# docs/age_regression.md "Anti-memorization options").
run_M2() { submit M2 age_M2_short_ema      "${SHORT[@]}" optimizer.model_ema=true \
                     optimizer.model_ema_decay=0.9995 evaluation.use_ema=true; }
run_M3() { submit M3 age_M3_short_lora16   "${SHORT[@]}" model.lora.enabled=true model.lora.rank=16 \
                     model.lora.alpha=32.0 optimizer.lr=1e-3 optimizer.layer_decay=1.0; }
run_M4() { submit M4 age_M4_short_softlabel "${SHORT[@]}" loss.regression_loss=soft_label \
                     loss.soft_label_sigma=2.5; }

ALL=(A B C C2 D D2 DX1 DX2 DX3 DX4 F1 F2 F3 F4 G1 H1 H2 M2 M3 M4)
EXPERIMENTS=("$@")
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=(A B C C2 D D2)
fi
for exp in "${EXPERIMENTS[@]}"; do
  if [[ " ${ALL[*]} " == *" ${exp} "* ]]; then
    "run_${exp}"
  else
    echo "Unknown experiment '${exp}' (expected ${ALL[*]})" >&2; exit 1
  fi
done
