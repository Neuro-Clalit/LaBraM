#!/bin/bash
# Fetch finished fold-0 SageMaker jobs of the feature candidates (submit_grid.sh P4F_*), score each on every
# test cohort (eval_p4_features.sh) and compare each with the baseline job P4F_L1 per cohort.
#
#   scripts/mil_runs/eval_candidates.sh <baseline job name> <name>=<job name> ...
#   e.g. scripts/mil_runs/eval_candidates.sh labram-mil-p4f-l1-f0-fold-0-2026-... \
#          P4F_GREEN_L1=labram-mil-p4f-green-l1-f0-fold-0-2026-... P4F_MS_L1=...
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
ROOT=${CAND_ROOT:-/data/runs/heedb/feature_candidates/runs}
E=/data/datasets/EEG-public; C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
fetch() {   # <name> <job>: the job's run folder (run_config.yaml, log.txt, checkpoint-best.pth) from its checkpoint S3 folder
  local name=$1 job=$2 dir=$ROOT/$1
  [[ -f $dir/checkpoint-best.pth ]] && grep -q "Age target normalization" "$dir/log/run.log" 2>/dev/null && return
  mkdir -p "$dir/log"
  local ck; ck=$(aws sagemaker describe-training-job --training-job-name "$job" --query CheckpointConfig.S3Uri --output text)
  for f in checkpoint-best.pth log.txt run_config.yaml log/run.log fold_metrics.json; do
    aws s3 cp --only-show-errors "${ck%/}/cv/fold_0/$f" "$dir/$f" || [[ $f == fold_metrics.json ]]
  done
  # the job's run.log in S3 can lack the start-up lines; the target normalisation (needed by
  # --pred-stats-from-log) is then taken from the job's CloudWatch log
  if ! grep -q "Age target normalization" "$dir/log/run.log"; then
    local stream; stream=$(aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs \
      --log-stream-name-prefix "$job" --query 'logStreams[0].logStreamName' --output text)
    aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name "$stream" \
      --start-from-head --limit 2000 --query 'events[].message' --output text | tr '\t' '\n' \
      | grep -m1 "Age target normalization" >> "$dir/log/run.log" || true   # grep -m1 closes the pipe early
  fi
}
BASE_JOB=$1; shift
fetch P4F_L1 "$BASE_JOB"
EVAL_OUT=$ROOT/P4F_L1/eval bash scripts/mil_runs/eval_p4_features.sh "$ROOT/P4F_L1" P4F_L1 P4_L1_f0 || true
for spec in "$@"; do
  name=${spec%%=*}; job=${spec#*=}
  fetch "$name" "$job"
  EVAL_OUT=$ROOT/$name/eval bash scripts/mil_runs/eval_p4_features.sh "$ROOT/$name" "$name" P4_L1_f0 || true
  B=$ROOT/P4F_L1/eval; N=$ROOT/$name/eval
  $PY scripts/mil_runs/compare_runs.py --name "$name" --ref P4F_L1 --out "$N/vs_baseline.csv" \
    --pair own:$N/own.json:$B/own.json:$C --pair lemon:$N/lemon.json:$B/lemon.json:$E/LEMON/cohorts/lemon_v1 \
    --pair chbp:$N/chbp.json:$B/chbp.json:$E/CHBP/cohorts/chbp_v1 --pair tdbrain:$N/tdbrain.json:$B/tdbrain.json:$E/TDBRAIN/cohorts/tdbrain_v1 \
    --pair dortmund:$N/dortmund.json:$B/dortmund.json:$E/DORTMUND/cohorts/dortmund_v1 --pair srm:$N/srm.json:$B/srm.json:$E/SRM/cohorts/srm_v1
done
