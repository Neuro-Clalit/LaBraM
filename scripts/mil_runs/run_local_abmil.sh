#!/bin/bash
# L1: the simplest ABMIL, end to end on the local GPU. Ilse et al.'s original
# (ungated) attention over the encoder's window features, warm start from E2L,
# 4 epochs (epoch 0: encoder frozen), E2L's data / split / channels / eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
NAME=${NAME:-L1_abmil}
/data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json --set \
  data.data_path=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 data.data_format=npy data.stage_dir=off \
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0 \
  channel_drop.consistency_weight=0.0 channel_drop.eval_consistency=false \
  data.num_workers=20 mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false \
  finetune_checkpoint.finetune=/data/runs/heedb/e2l/finetune_20261104_215141_883/checkpoint-best.pth \
  optimizer.lr=0.0001 optimizer.warmup_epochs=1 trainer.epochs=4 trainer.early_stopping=false \
  mil.enabled=true mil.pool=gated mil.gated=false mil.score_input=h mil.attn_dim=128 \
  mil.bag_size=8 trainer.batch_size=32 evaluation.eval_batch_size=12 \
  mil.head_lr_scale=5.0 mil.entropy_weight=0.01 mil.dropout=0.1 mil.freeze_encoder_epochs=1 \
  evaluation.agg_windows=mean evaluation.train_eval_every=1 \
  output.output_dir=/data/runs/heedb/mil_local/$NAME output.log_dir=/data/runs/heedb/mil_local/$NAME/log \
  clearml.enabled=true clearml.project_name=LaBraM/heedb_mil clearml.task_name=heedb_mil_${NAME}_local \
  "clearml.tags=[\"heedb\",\"mil\",\"abmil\",\"local\",\"${NAME}\"]" "$@"
