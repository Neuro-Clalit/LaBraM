#!/bin/bash
# Resume E2L with the pread loader once epoch index 5 is fully evaluated.
R=$(ls -d /data/runs/heedb/e2l/finetune_*)
LOG=/data/runs/heedb/e2l/switch.log
echo "$(date -u +%T) waiting for epoch 5 in $R/train_state.json" >> $LOG
until python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/train_state.json'))['epoch']>=5 else 1)" 2>/dev/null; do sleep 20; done
sleep 30                                             # let the epoch's last writes finish
BASH_PID=$(pgrep -f "run_e2l.sh" | head -1)
MAIN=$(pgrep -P "$BASH_PID" -f labram.runs.run_finetune | head -1)
echo "$(date -u +%T) epoch 5 saved; stopping run_e2l.sh=$BASH_PID main=$MAIN" >> $LOG
kill -INT "$MAIN"; sleep 60
pgrep -P "$BASH_PID" >/dev/null && kill -TERM "$MAIN"
while pgrep -f "labram.runs.run_finetune" >/dev/null; do sleep 5; done
echo "$(date -u +%T) stopped; resuming with LABRAM_NPY_READ=pread" >> $LOG
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export OMP_NUM_THREADS=1 LABRAM_NPY_READ=pread
nohup /data/code/LaBraM/.venv/bin/python -m labram.runs.run_finetune --config $R/run_config.yaml \
  --set output.auto_resume=true clearml.task_name=heedb_E2L_tuh_heedb_bs512_ep16_local_resumed \
  >> /data/runs/heedb/e2l/train.log 2>&1 &
echo "$(date -u +%T) resumed pid $!" >> $LOG
