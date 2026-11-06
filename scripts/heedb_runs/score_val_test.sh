#!/bin/bash
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDx55CBxr45TmpannZj
export PYTHONPATH=.
PY=/data/code/LaBraM/.venv/bin/python
S=/data/runs/heedb/scores
B0=/tmp/e0/single/labram-tuh-exp4anymiss-noear-nolabel/finetune
for cohort in tuab:/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss heedb:/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1; do
  name=${cohort%%:*}; path=${cohort#*:}
  for run in E0=$B0 E1=/data/runs/heedb/e1_r2/finetune E2=/data/runs/heedb/e2_r2/finetune; do
    m=${run%%=*}
    out=$S/${m}_${name}_valtest.json
    [ -s $out ] && continue
    echo "$(date -u +%T) $m on $name"
    $PY scripts/eval_age_by_cohort.py --data-path $path --run $run --pred-stats-from-log --stage-dir none --splits val test --out $out 2>&1 | grep -E "^E[0-9] |Traceback|Error"
  done
done
echo "DONE $(date -u +%T)"
