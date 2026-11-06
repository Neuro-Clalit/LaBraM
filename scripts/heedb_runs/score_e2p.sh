#!/bin/bash
# Score E2P's best checkpoint per class: val/test on the joint cohort + a
# 2,000-recording training sample (reported as split "train").
set -e
J=${1:-labram-heedb-e2p-pathology-2026-11-05-20-52-14-760}
MODEL=s3://sagemaker-us-east-1-574441342949/$J/output/model.tar.gz
W=/data/runs/heedb/e2p_scoring; mkdir -p $W/work
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDyZr6ZFF9cF4BmbpuA
cat > $W/manifest.json <<JSON
[{"name": "E2P", "model": "$MODEL", "data_path": "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"},
 {"name": "E2P_trainsample", "model": "$MODEL", "data_path": "/data/runs/heedb/e2p_trainsample"}]
JSON
export AWS_PROFILE=mfa LABRAM_NPY_READ=pread PYTHONPATH=.
/data/code/LaBraM/.venv/bin/python scripts/eval_multitask_classification.py --manifest $W/manifest.json \
  --out $W/e2p_preds_raw.csv.gz --n-boot 0 --work-dir $W/work --num-workers 8
/data/code/LaBraM/.venv/bin/python - <<PY
import pandas as pd
p = pd.read_csv("$W/e2p_preds_raw.csv.gz")
t = p[p.run == "E2P_trainsample"]
t = t[t.split == "val"].assign(run="E2P", split="train")
pd.concat([p[p.run == "E2P"], t]).to_csv("$W/e2p_preds.csv.gz", index=False)
PY
/data/code/LaBraM/.venv/bin/python scripts/multitask_per_class_table.py --preds $W/e2p_preds.csv.gz \
  --out docs/data/e2p_per_class.csv
