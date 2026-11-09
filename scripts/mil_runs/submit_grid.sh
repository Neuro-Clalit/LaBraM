#!/bin/bash
# MIL (B1) grid on SageMaker: Stage 3 (warm start from E2L, 4 epochs) and
# Stage 4 (from labram-base, 12 epochs), each with its matched window-level
# control. Data, split, channels and eval protocol are E2L's. See
# docs/b1_attention_mil_plan.md and docs/mil_stage1_results.md.
#
#   scripts/mil_runs/submit_grid.sh --dry_run          # preview every job
#   scripts/mil_runs/submit_grid.sh                    # submit (detached)
# Streamed (FastFile) reads are latency-bound: W0 ran 2.2 s/step on ml.g5.2xlarge
# with 16 loader workers (1.85 s of it waiting for data), so the default is
# ml.g5.4xlarge with 32 workers (override with INSTANCE=..., WORKERS=...).
# Host memory: every loader worker slowly copies the dataset's ~17M-name file
# list, so ml.g5.2xlarge (32 GB) ran out of memory with 24 workers (16 held);
# use WORKERS=10 there. ml.g5.4xlarge (64 GB) with 32 workers is fine.
#   ONLY="W2 W3" scripts/mil_runs/submit_grid.sh       # a subset
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
EXTRA=("$@")
E2L_CKPT=s3://eeg-data-public/HEEDB/v4.1/derived/models/E2L/checkpoint-best.pth
BASE=(data.data_path=s3://eeg-data-public/HEEDB/v4.1/cohorts/tuh_heedb_v1_compact data.data_format=npy
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0
  channel_drop.consistency_weight=0.0 channel_drop.eval_consistency=false
  data.num_workers=${WORKERS:-32} mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false
  evaluation.agg_windows=mean evaluation.train_eval_every=1
  trainer.early_stopping=false output.output_dir= output.log_dir=
  clearml.enabled=true clearml.project_name=LaBraM/heedb_mil
  sagemaker.enabled=true sagemaker.role=arn:aws:iam::574441342949:role/SageMakerExecutionRole
  sagemaker.input_mode=FastFile sagemaker.instance_type=${INSTANCE:-ml.g5.4xlarge} sagemaker.volume_size_gb=200
  sagemaker.use_spot=${SPOT:-true} sagemaker.on_demand_fallback=true sagemaker.max_run_sec=86400
  sagemaker.wait=false sagemaker.stream_logs=false)
# Stage 3: warm start from E2L; 4 epochs, lr 1e-4, 1 warm-up epoch.
WARM=(finetune_checkpoint.finetune=$E2L_CKPT optimizer.lr=0.0001 optimizer.warmup_epochs=1 trainer.epochs=4)
# Stage 4: from labram-base like E2L; 12 epochs, lr 2.8e-4, 2 warm-up epochs.
SCRATCH=(optimizer.lr=0.00028 optimizer.warmup_epochs=2 trainer.epochs=12)
# Bag mode: 32 recordings x 8 windows per step (= E2L's 256 windows); eval bags = 30 windows.
BAG=(mil.enabled=true mil.bag_size=8 trainer.batch_size=32 evaluation.eval_batch_size=12
  mil.head_lr_scale=5.0 mil.entropy_weight=0.01 mil.dropout=0.1)
# Long warm start (all X runs): encoder frozen for the first 4 epochs (pool +
# head only), then trained for 16 (20 epochs; the cosine LR, 1 warm-up epoch,
# is at ~0.94 x peak when the encoder unfreezes). Up to ~40 h for X6, so the
# job cap is 72 h.
LONG=(finetune_checkpoint.finetune=$E2L_CKPT optimizer.lr=0.0001 optimizer.warmup_epochs=1 trainer.epochs=20
  mil.freeze_encoder_epochs=4 sagemaker.max_run_sec=259200)
# Random bag sizes: K drawn per recording, uniform over 8..32 (16 recordings per
# step, <= 512 windows) or 8..64 (8 recordings, <= 512 windows); an epoch reads
# ~2.5x / ~4x the windows of K = 8.
K32="mil.bag_size_choices=[$(seq -s, 8 32)] trainer.batch_size=16"
K64="mil.bag_size_choices=[$(seq -s, 8 64)] trainer.batch_size=8"
# Window-level control: 8 windows per recording per epoch, batch 256 windows.
WINDOW=(mil.enabled=false data.train_minutes_per_epoch=1.3334 trainer.batch_size=256)

declare -A JOBS=(
  [W0]="${WARM[*]} ${WINDOW[*]}"
  [W1]="${WARM[*]} ${BAG[*]} mil.pool=mean mil.freeze_encoder_epochs=1"
  [W2]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=q mil.freeze_encoder_epochs=1"
  [W3]="${WARM[*]} ${BAG[*]} mil.pool=sigmoid mil.score_input=q mil.attn_dim=64 mil.freeze_encoder_epochs=1"
  [W4]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=h mil.freeze_encoder_epochs=1"
  [W5]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.time_encoding=true mil.freeze_encoder_epochs=1"
  [W6]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.branches=4 mil.attn_dim=64 mil.topk_mask_prob=0.3 mil.freeze_encoder_epochs=1"
  [W7]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=q mil.freeze_encoder_epochs=1 mil.retention_weight=0.1"
  # X: the selected methods, longer (X1 = W4 + retention, X2 = L1), each with
  # descriptors added to the attention input (X3, X4), and random bag sizes (X5, X6).
  [X1]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.score_input=h mil.retention_weight=0.1"
  [X2]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.gated=false mil.score_input=h"
  [X3]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.retention_weight=0.1"
  [X4]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.gated=false mil.score_input=hq"
  [X5]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.score_input=h mil.retention_weight=0.1 $K32"
  [X6]="${LONG[*]} ${BAG[*]} mil.pool=gated mil.score_input=h mil.retention_weight=0.1 $K64"
  [F0]="${SCRATCH[*]} ${WINDOW[*]}"
  [F1]="${SCRATCH[*]} ${BAG[*]} mil.pool=mean"
  [F2]="${SCRATCH[*]} ${BAG[*]} mil.pool=gated mil.score_input=q"
  [F3]="${SCRATCH[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.time_encoding=true"
)
NAMES=${ONLY:-"W0 W1 W2 W3 W4 W5 W6 W7 F0 F1 F2 F3"}
for name in $NAMES; do
  read -r -a over <<< "${JOBS[$name]}"
  echo "=== $name: ${JOBS[$name]}"
  "$PY" -m labram.runs.submit_sagemaker --phase finetune --detach \
    --config labram/configs/defaults/finetune_tuab_age.json "${EXTRA[@]}" --set "${BASE[@]}" "${over[@]}" \
    clearml.task_name="heedb_mil_${name}" "clearml.tags=[\"heedb\",\"mil\",\"${name}\"]" \
    sagemaker.job_name_prefix="labram-mil-${name,,}"
done
