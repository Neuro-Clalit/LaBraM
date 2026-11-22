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
fetch() {   # <name> <job>: the job's output folder (run_config.yaml, log.txt, checkpoint-best.pth)
  local name=$1 job=$2 dir=$ROOT/$1
  [[ -f $dir/checkpoint-best.pth ]] && return
  mkdir -p "$dir"
  local art; art=$(aws sagemaker describe-training-job --training-job-name "$job" --query ModelArtifacts.S3ModelArtifacts --output text)
  aws s3 cp --only-show-errors "$art" "$dir/model.tar.gz"
  tar -xzf "$dir/model.tar.gz" -C "$dir" && rm "$dir/model.tar.gz"
  local best; best=$(find "$dir" -name checkpoint-best.pth | head -1)
  [[ -n $best ]] || { echo "$job: no checkpoint-best.pth in $art" >&2; return 1; }
  local src; src=$(dirname "$best")
  [[ $src == "$dir" ]] || cp -n "$src"/{checkpoint-best.pth,log.txt,run_config.yaml} "$dir"/
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
