#!/bin/bash
# Score one bag (MIL) run on every test cohort with the settings the report's fold models were scored
# with (scripts/external_runs/*.sh), then compare it per cohort with a reference model already scored
# there (default: the P4 fold model of the same fold).
#
#   scripts/mil_runs/eval_p4_features.sh <run dir> <name> [reference name]
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
RUN=$1; NAME=$2; REF=${3:-P4_L1_f${NAME##*_f}}
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
E=/data/datasets/EEG-public; C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1; X=/data/runs/heedb/external_eval
OUT=${EVAL_OUT:-$RUN/eval}; mkdir -p $OUT          # EVAL_OUT: another folder, e.g. for an interim checkpoint
score() {   # <cohort key> <cohort folder> [evaluation options]
  local key=$1 path=$2; shift 2
  [[ -f $OUT/$key.json ]] && return
  $PY scripts/eval_age_by_cohort.py --data-path $path --splits test --stage-dir none "$@" --run $NAME=$RUN \
    --pred-stats-from-log --save-windows $OUT/windows/$key --num-workers ${WORKERS:-8} --out $OUT/$key.json > $OUT/$key.log 2>&1
}
score own $C                                                                     # Temple + Harvard test: first 5 min after the 60 s trims
score lemon $E/LEMON/cohorts/lemon_v1 --eval-minutes 0                           # every window after the trims
score chbp $E/CHBP/cohorts/chbp_v1 --eval-minutes 0
score tdbrain $E/TDBRAIN/cohorts/tdbrain_v1 --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0    # short files: every window
score dortmund $E/DORTMUND/cohorts/dortmund_v1 --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0
score srm $E/SRM/cohorts/srm_v1 --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0
D=$E/DORTMUND/cohorts/dortmund_v1; T=$E/TDBRAIN/cohorts/tdbrain_v1
$PY scripts/mil_runs/compare_runs.py --name $NAME --ref $REF --out $OUT/comparison.csv \
  --pair own:$OUT/own.json:$X/own_5min_models.json:$C --pair lemon:$OUT/lemon.json:$X/lemon_all_models.json:$E/LEMON/cohorts/lemon_v1 \
  --pair chbp:$OUT/chbp.json:$X/chbp_all_models.json:$E/CHBP/cohorts/chbp_v1 --pair tdbrain:$OUT/tdbrain.json:$X/tdbrain_all_models.json:$T \
  --pair dortmund:$OUT/dortmund.json:$X/dortmund_all_models.json:$D --pair srm:$OUT/srm.json:$X/srm_all_models.json:$E/SRM/cohorts/srm_v1
