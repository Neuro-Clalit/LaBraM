#!/bin/bash
# Resume E2L after epoch 6 (same run dir, split, cosine position) with:
# batch 256 / lr 2.8e-4 (512 / 4e-4 gave no speed, 2x fewer steps and an unstable val),
# 12 workers (8 already saturate the GPU with the pread loader), train_eval every 4
# epochs + last, and the linear-time evaluation pooling.
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export OMP_NUM_THREADS=1 LABRAM_NPY_READ=pread
exec /data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune --config /data/runs/heedb/e2l/finetune_20261104_215141_883/run_config.yaml \
  --set output.auto_resume=true trainer.batch_size=256 optimizer.lr=0.00028 data.num_workers=12 \
        evaluation.train_eval_every=4 \
        clearml.task_name=heedb_E2L_tuh_heedb_ep16_local_resumed_bs256 \
        'clearml.tags=["heedb","E2L","tuh","local","ep16","bs256-from-ep7"]'
