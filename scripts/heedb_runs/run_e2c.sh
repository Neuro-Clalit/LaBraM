#!/bin/bash
# E2c: E2 recipe (TUH + HEEDB, identical split) + clinical condition vectors
# (status normal/abnormal/NA + 14 pathology flags) into the regressor head.
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export OMP_NUM_THREADS=1
/data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune --config labram/configs/defaults/finetune_tuab_age.json --set \
  data.data_path=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 data.data_format=npy \
  data.session_features=conditions_v1.npz model.session_features=head \
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0 channel_drop.consistency_weight=0.0 \
  data.num_workers=6 data.train_minutes_per_epoch=5 mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false \
  optimizer.lr=0.0002 trainer.batch_size=128 trainer.early_stopping=true trainer.epochs=8 \
  data.stage_dir= finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth \
  output.output_dir=/data/runs/heedb/e2c/finetune output.log_dir=/data/runs/heedb/e2c/finetune/log \
  clearml.enabled=true clearml.project_name=LaBraM/heedb clearml.task_name=heedb_E2c_tuh_heedb_conditions_local \
  'clearml.tags=["heedb","E2c","tuh","conditions","local"]' "$@"
