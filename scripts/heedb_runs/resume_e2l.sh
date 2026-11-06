#!/bin/bash
# Resume E2L (same run dir, split and validation history) after the last saved
# epoch, reading windows with the pread loader.
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export OMP_NUM_THREADS=1 LABRAM_NPY_READ=pread
exec /data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune --config /data/runs/heedb/e2l/finetune_20261104_215141_883/run_config.yaml \
  --set output.auto_resume=true clearml.task_name=heedb_E2L_tuh_heedb_bs512_ep16_local_resumed
