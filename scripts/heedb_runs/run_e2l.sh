#!/bin/bash
# E2L: E2 recipe (TUH + HEEDB, identical split, no condition inputs), longer and
# faster: 16 epochs / 4 warm-up, batch 512 (lr 4e-4 = 2e-4 x sqrt(512/128)),
# 20 loader workers (the 6-worker local runs were I/O-bound).
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export OMP_NUM_THREADS=1
/data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune --config labram/configs/defaults/finetune_tuab_age.json --set \
  data.data_path=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 data.data_format=npy \
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0 channel_drop.consistency_weight=0.0 \
  data.num_workers=20 data.train_minutes_per_epoch=5 mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false \
  optimizer.lr=0.0004 optimizer.warmup_epochs=4 trainer.batch_size=512 trainer.epochs=16 \
  trainer.early_stopping=true trainer.early_stopping_patience=5 \
  data.stage_dir= finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth \
  output.output_dir=/data/runs/heedb/e2l/finetune output.log_dir=/data/runs/heedb/e2l/finetune/log \
  clearml.enabled=true clearml.project_name=LaBraM/heedb clearml.task_name=heedb_E2L_tuh_heedb_bs512_ep16_local \
  'clearml.tags=["heedb","E2L","tuh","local","bs512","ep16"]' "$@"
