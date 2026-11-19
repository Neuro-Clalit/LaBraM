#!/bin/bash
# The MIL grid of scripts/mil_runs/submit_grid.sh on the local GPU, one job
# after another (local npy reads are ~10x faster than SageMaker's streamed S3).
#   ONLY="F0 F2" scripts/mil_runs/run_local_grid.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
OUT=${OUT:-/data/runs/heedb/mil_local}
E2L_CKPT=/data/runs/heedb/e2l/finetune_20261104_215141_883/checkpoint-best.pth
BASE=(data.data_path=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 data.data_format=npy data.stage_dir=off
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0
  channel_drop.consistency_weight=0.0 channel_drop.eval_consistency=false
  data.num_workers=20 mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false
  evaluation.agg_windows=mean evaluation.train_eval_every=1 trainer.early_stopping=false
  clearml.enabled=true clearml.project_name=LaBraM/heedb_mil)
WARM=(finetune_checkpoint.finetune=$E2L_CKPT optimizer.lr=0.0001 optimizer.warmup_epochs=1 trainer.epochs=4)
SCRATCH=(finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth
  optimizer.lr=0.00028 optimizer.warmup_epochs=2 trainer.epochs=12)
BAG=(mil.enabled=true mil.bag_size=8 trainer.batch_size=32 evaluation.eval_batch_size=12
  mil.head_lr_scale=5.0 mil.entropy_weight=0.01 mil.dropout=0.1)
WINDOW=(mil.enabled=false data.train_minutes_per_epoch=1.3334 trainer.batch_size=256)
declare -A JOBS=(
  [W2]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=q mil.freeze_encoder_epochs=1"
  [W3]="${WARM[*]} ${BAG[*]} mil.pool=sigmoid mil.score_input=q mil.attn_dim=64 mil.freeze_encoder_epochs=1"
  [W4]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=h mil.freeze_encoder_epochs=1"
  [W5]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.time_encoding=true mil.freeze_encoder_epochs=1"
  [W6]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.branches=4 mil.attn_dim=64 mil.topk_mask_prob=0.3 mil.freeze_encoder_epochs=1"
  [W7]="${WARM[*]} ${BAG[*]} mil.pool=gated mil.score_input=q mil.freeze_encoder_epochs=1 mil.retention_weight=0.1"
  [F0]="${SCRATCH[*]} ${WINDOW[*]}"
  [F1]="${SCRATCH[*]} ${BAG[*]} mil.pool=mean"
  [F2]="${SCRATCH[*]} ${BAG[*]} mil.pool=gated mil.score_input=q"
  [F3]="${SCRATCH[*]} ${BAG[*]} mil.pool=gated mil.score_input=hq mil.time_encoding=true"
)
for name in ${ONLY:-F0 F2 F1 F3}; do
  read -r -a over <<< "${JOBS[$name]}"
  echo "=== $(date -u +%FT%T) $name: ${JOBS[$name]}"
  "$PY" -m labram.runs.run_finetune --config labram/configs/defaults/finetune_tuab_age.json --set \
    "${BASE[@]}" "${over[@]}" output.output_dir=$OUT/$name output.log_dir=$OUT/$name/log \
    clearml.task_name="heedb_mil_${name}_local" "clearml.tags=[\"heedb\",\"mil\",\"${name}\",\"local\"]" \
    > "$OUT/$name.log" 2>&1
  echo "=== $(date -u +%FT%T) $name exit $?: $(grep -E 'Val EEG mae:|Test EEG mae:' "$OUT/$name.log" | tail -2 | tr '\n' ' ')"
done
