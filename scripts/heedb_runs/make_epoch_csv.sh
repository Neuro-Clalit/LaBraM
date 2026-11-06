#!/bin/bash
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
/data/code/LaBraM/.venv/bin/python scripts/epoch_metrics_from_log.py \
  --run Exp4noearEEG=/tmp/e0/single/labram-tuh-exp4anymiss-noear-nolabel/finetune/log/run.log \
  --run H-E1=/data/runs/heedb/e1_r2/finetune/log/run.log \
  --run H-E2=/data/runs/heedb/e2_r2/finetune/log/run.log \
  --run H-E2c=/data/runs/heedb/e2c/train.log \
  --run H-E2L=/data/runs/heedb/e2l/train.log \
  --out docs/data/heedb_epoch_metrics.csv > /dev/null
