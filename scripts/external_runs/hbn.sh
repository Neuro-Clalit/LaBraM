#!/usr/bin/env bash
# HBN-EEG (Healthy Brain Network, Child Mind Institute; Shirazi et al. 2024, OpenNeuro
# ds005505-ds005516, CC-BY-SA 4.0, no account): download the adults' resting-state EEG,
# convert to the npy cohort format and score the brain-age models zero-shot, as for
# LEMON / MIPDB / CHBP / TDBRAIN (docs/hbn_onboarding.md).
#
# Onboarded for reference only: HBN is a paediatric study and the models are trained on
# adults, so it is NOT part of the final external-evaluation reports (docs/hbn_onboarding.md).
#
# Only participants aged >= $MIN_AGE (default 18: 74 of 3,155, 63 with a usable resting
# file) are fetched, because the age models are trained on adults. MIN_AGE=0 fetches every
# participant's resting file (~300 GB).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
ROOT=${HBN_ROOT:-/data/datasets/EEG-public/HBN}
S3=s3://eeg-data-public/HBN/v1
COHORT=$ROOT/cohorts/${HBN_COHORT:-hbn_adults_v1}
PY=${PY:-/opt/pytorch/bin/python}            # training env (mne, torch)
OUT=/data/runs/heedb/external_eval
MIN_AGE=${MIN_AGE:-18}
RELEASES="ds005505 ds005506 ds005507 ds005508 ds005509 ds005510 ds005511 ds005512 ds005514 ds005515 ds005516"
step=${1:-all}

if [[ $step == download || $step == all ]]; then
  for ds in $RELEASES; do
    mkdir -p "$ROOT/raw/$ds"
    aws s3 sync --no-sign-request --only-show-errors "s3://openneuro.org/$ds" "$ROOT/raw/$ds" \
      --exclude '*' --include 'participants.*' --include 'dataset_description.json' --include 'README' \
      --include 'CHANGES' --include 'task-RestingState_*'
    # Release 11 (ds005516): the bucket's current objects lack the task columns of
    # participants.tsv and every .set (delete markers). The published snapshot is intact, so
    # the table comes from the dataset's git mirror and a missing .set from the snapshot's
    # file endpoint (which redirects to the versioned S3 object).
    if ! head -1 "$ROOT/raw/$ds/participants.tsv" | grep -q RestingState; then
      curl -fsSL --retry 4 "https://raw.githubusercontent.com/OpenNeuroDatasets/$ds/main/participants.tsv" \
        -o "$ROOT/raw/$ds/participants.tsv"
    fi
    ver=$(git ls-remote --tags --refs "https://github.com/OpenNeuroDatasets/$ds.git" | sed 's#.*refs/tags/##' | sort -V | tail -1)
    # column 4 of participants.tsv is the age in years
    awk -F'\t' -v a="$MIN_AGE" 'NR > 1 && $4 >= a {print $1}' "$ROOT/raw/$ds/participants.tsv" | while read -r sub; do
      aws s3 sync --no-sign-request --only-show-errors "s3://openneuro.org/$ds/$sub" "$ROOT/raw/$ds/$sub" \
        --exclude '*' --include '*RestingState*'
      f=${sub}_task-RestingState_eeg.set
      if [[ -f $ROOT/raw/$ds/$sub/eeg/${f%eeg.set}events.tsv && ! -f $ROOT/raw/$ds/$sub/eeg/$f ]]; then
        curl -fsSL --retry 4 -o "$ROOT/raw/$ds/$sub/eeg/$f.part" \
          "https://openneuro.org/crn/datasets/$ds/snapshots/$ver/files/$sub:eeg:$f" \
          && mv "$ROOT/raw/$ds/$sub/eeg/$f.part" "$ROOT/raw/$ds/$sub/eeg/$f" || echo "  no data file for $sub"
      fi
    done
    echo "$ds: $(find "$ROOT/raw/$ds" -name '*RestingState_eeg.set' | wc -l) resting files"
  done
fi

if [[ $step == mirror || $step == all ]]; then
  $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public'); print('bucket is private')"
  aws s3 sync "$ROOT/raw" "$S3/raw" --only-show-errors
fi

if [[ $step == convert || $step == all ]]; then
  $PY -m dataset_maker.make_external_npy --dataset hbn --raw "$ROOT/raw" --meta "$ROOT/raw" \
    --min-age "$MIN_AGE" --out "$COHORT" --jobs 6
  aws s3 sync "$COHORT" "$S3/cohorts/$(basename "$COHORT")" --only-show-errors
fi

# Every run trained on tuh_heedb_v1: training-age mean / std 53.52 / 19.29 y.
# MODELS="E2L" limits the scoring to those names.
RUNS=(
  --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
  --run E2Lrob=/data/runs/chbp_fix/e2l_band45_amp_cont_20261112_001004_075
  --run E2LrobF=/data/runs/chbp_fix/e2l_robust_full_20261112_175034_869
)
if [[ -n ${MODELS:-} ]]; then          # keep only the --run pairs named in $MODELS
  sel=(); for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
    [[ " $MODELS " == *" ${RUNS[i+1]%%=*} "* ]] && sel+=("${RUNS[i]}" "${RUNS[i+1]}")
  done; RUNS=("${sel[@]}")
fi
TAG=$(basename "$COHORT")
if [[ $step == score || $step == all ]]; then
  # The resting block is ~5.8 min: the training protocol (60 s trimmed at each end, first
  # 5 min) keeps ~3.8 min ("5min"); "all" scores every window; "inner" trims 10 s per end.
  $PY scripts/eval_age_by_cohort.py --data-path "$COHORT" --splits test --pred-stats 53.52 19.29 \
    --stage-dir none "${RUNS[@]}" --out $OUT/${TAG}_5min.json 2>&1 | tee $OUT/${TAG}_5min.log
  $PY scripts/eval_age_by_cohort.py --data-path "$COHORT" --splits test --pred-stats 53.52 19.29 \
    --stage-dir none --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0 "${RUNS[@]}" \
    --out $OUT/${TAG}_all.json 2>&1 | tee $OUT/${TAG}_all.log
  $PY scripts/eval_age_by_cohort.py --data-path "$COHORT" --splits test --pred-stats 53.52 19.29 \
    --stage-dir none --eval-minutes 0 --trim-start-sec 10 --trim-end-sec 10 "${RUNS[@]}" \
    --out $OUT/${TAG}_inner.json 2>&1 | tee $OUT/${TAG}_inner.log
fi

if [[ $step == cv || $step == all ]]; then
  # P3_L1 / P4_L1 (phased one-run MIL, docs MIL results section 13): the five CV fold models
  # of each, every window; their per-recording median is the fold ensemble
  # (scripts/external_fold_ensemble.py). Each fold de-normalizes with its own training-age
  # stats from its log; P4 fold 3 was resumed and its run.log is gone, so it takes the
  # stats of the same fold of P3 (same split).
  CV=/data/runs/heedb/mil_phased_cv
  FOLDS=(); for m in P3_L1 P4_L1; do for k in 0 1 2 3 4; do
    [[ $m == P4_L1 && $k == 3 ]] || FOLDS+=(--run ${m}_f$k=$CV/$m/fold_$k/cv/fold_$k)
  done; done
  for tag in all 5min; do
    extra=(); [[ $tag == all ]] && extra=(--eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0)
    $PY scripts/eval_age_by_cohort.py --data-path "$COHORT" --splits test --pred-stats-from-log \
      --stage-dir none "${extra[@]}" "${FOLDS[@]}" --out $OUT/${TAG}_${tag}_p34_cv.json 2>&1 | tee $OUT/${TAG}_${tag}_p34_cv.log
    $PY scripts/eval_age_by_cohort.py --data-path "$COHORT" --splits test --pred-stats 53.55 19.29 \
      --stage-dir none "${extra[@]}" --run P4_L1_f3=$CV/P4_L1/fold_3/cv/fold_3 \
      --out $OUT/${TAG}_${tag}_p34_cv_f3.json 2>&1 | tee $OUT/${TAG}_${tag}_p34_cv_f3.log
  done
fi

if [[ $step == report || $step == all ]]; then
  # metrics per group, predicted vs true age, signal QC and PSD against the TUH / HEEDB
  # references, next to MIPDB (same EGI net and lab) and LEMON
  mkdir -p docs/data/external_eval/hbn docs/figures/hbn
  E=/data/datasets/EEG-public
  $PY scripts/external_eval_report.py --plot-models E2L E2Lrob \
    --eval hbn=$OUT/${TAG}_all.json --cohort hbn="$COHORT" \
    --eval mipdb=$OUT/mipdb_all.json --cohort mipdb=$E/MIPDB/cohorts/mipdb_v1 --min-age mipdb=18 \
    --eval lemon=$OUT/lemon_all.json --cohort lemon=$E/LEMON/cohorts/lemon_v1 \
    --out-dir docs/data/external_eval/hbn --fig-dir docs/figures/hbn
fi
