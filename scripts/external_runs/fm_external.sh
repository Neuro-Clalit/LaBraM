#!/usr/bin/env bash
# Stage B of docs/fm_encoder_onboarding.md: score a finished fine-tune with a foreign encoder on the
# external cohorts and compare it with P4_L1, paired (Tables 7.5, 7.13, 7.15, 7.16).
#
#   scripts/external_runs/fm_external.sh score <tag> <NAME>=<run dir>
#   scripts/external_runs/fm_external.sh table <tag> <NAME> [<test bag-weight parquet>]
#
# score  LEMON, Dortmund, TDBRAIN, SRM, CHBP with the run's best checkpoint, every window, no trims
#        for Dortmund, TDBRAIN and SRM (their recordings are short), de-normalised with the target
#        statistics the run logged -> $OUT/<cohort>_all_<tag>.json. A SageMaker run needs its
#        checkpoint-best.pth, run_config.yaml and log/run.log in <run dir>.
# table  paired comparison with P4 fold 0 and the ensembles -> $OUT/<tag>_vs_p4.csv
#
# Next to a training run on the same GPU: GPU_FRACTION=0.17 BAG_CHUNK=128 caps the scoring process
# (REVE-base fits in 4 GB that way); the results do not depend on either.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=1
PY=${PY:-/opt/pytorch/bin/python}
E=/data/datasets/EEG-public
O=${OUT:-/data/runs/heedb/fm_encoders/eval}
step=${1:?step: score | table}; tag=${2:?tag}
mkdir -p "$O"
case $step in
  score)
    run=${3:?NAME=run dir}
    for spec in "lemon LEMON" "dortmund DORTMUND NT" "tdbrain TDBRAIN NT" "srm SRM NT" "chbp CHBP"; do
      read -r c D nt <<< "$spec"
      A=(--data-path "$E/$D/cohorts/${c}_v1" --splits test --eval-minutes 0 --pred-stats-from-log
         --stage-dir none --num-workers "${WORKERS:-4}" --gpu-memory-fraction "${GPU_FRACTION:-0}"
         --bag-encode-chunk "${BAG_CHUNK:-0}")
      [ "${nt:-}" = NT ] && A+=(--trim-start-sec 0 --trim-end-sec 0)
      "$PY" scripts/eval_age_by_cohort.py "${A[@]}" --run "$run" --out "$O/${c}_all_${tag}.json" \
        > "$O/${c}_all_${tag}.log" 2>&1
      echo "$c exit=$? $(tail -1 "$O/${c}_all_${tag}.log" | cut -c1-120)"
    done ;;
  table)
    name=${3:?the run name used for score}
    "$PY" scripts/fm_external_compare.py --tag "$tag" --name "$name" --eval-dir "$O" \
      ${4:+--test-parquet "$4"} ;;
  *) echo "step: score | table" >&2; exit 1 ;;
esac
