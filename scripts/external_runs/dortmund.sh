#!/usr/bin/env bash
# Dortmund Vital Study resting-state EEG (Wascher et al., OpenNeuro ds005385, CC0):
# list, download, audit, private S3 mirror, convert to the npy cohort format and score the
# brain-age models zero-shot, as for LEMON / CHBP / TDBRAIN (docs/dortmund_onboarding.md).
# Test-only cohort: nothing here is used to fit, tune, select or calibrate a model.
#
#   step list     : file listing of the public OpenNeuro bucket -> $ROOT/meta/listing.txt
#   step fetch    : download (resumable; no AWS identity is sent: --no-sign-request)
#   step audit    : EDF calibration headers of every file (label-free)
#   step mirror   : copy raw + meta to the private bucket (refuses if it is public)
#   step convert  : npy cohort $COHORT, mirrored to S3
#   step score    : E2L family and the P3 / P4 fold models, every window, with window predictions
#   step ensemble : 5-fold ensembles
#   step tables   : per-run / per-session metrics and the cohort table
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
ROOT=${DORTMUND_ROOT:-/data/datasets/EEG-public/DORTMUND}
SRC=s3://openneuro.org/ds005385
S3=s3://eeg-data-public/DORTMUND/v1
BIDS=$ROOT/raw/ds005385
COHORT=$ROOT/cohorts/dortmund_v1
PY=${PY:-/opt/pytorch/bin/python}
OUT=/data/runs/heedb/external_eval
D=docs/data/external_eval/dortmund
step=${1:-all}

if [[ $step == list ]]; then
  mkdir -p "$ROOT/meta"
  aws s3 ls --no-sign-request --recursive $SRC/ > "$ROOT/meta/listing.txt"
  wc -l "$ROOT/meta/listing.txt"
fi

if [[ $step == fetch ]]; then
  mkdir -p "$BIDS"
  for i in 1 2 3 4 5; do
    aws s3 sync --no-sign-request $SRC/ "$BIDS/" --exclude ".datalad/*" --only-show-errors && break
    sleep 30
  done
  echo "EDF files: $(find "$BIDS" -name '*_eeg.edf' | wc -l) (listing: $(grep -c '_eeg.edf$' "$ROOT/meta/listing.txt"))"
fi

if [[ $step == audit || $step == all ]]; then
  mkdir -p $D
  $PY scripts/edf_header_audit.py --glob "$BIDS/sub-*/ses-*/eeg/*_eeg.edf" --out $D/edf_headers.csv
fi

if [[ $step == mirror || $step == all ]]; then
  $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public'); print('bucket is private')"
  aws s3 sync "$ROOT/raw" "$S3/raw" --only-show-errors
  aws s3 sync "$ROOT/meta" "$S3/meta" --only-show-errors
fi

if [[ $step == convert || $step == all ]]; then
  $PY -m dataset_maker.make_external_npy --dataset dortmund --raw "$BIDS" --meta "$BIDS/participants.tsv" \
    --out $COHORT --jobs ${JOBS:-7}
  $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public')"
  aws s3 sync $COHORT "$S3/cohorts/dortmund_v1" --only-show-errors
fi

# Each run is about 3 min (18 windows): the training trims (60 s at each end) would leave one
# minute, so every window is scored, as for TDBRAIN. Runs trained on tuh_heedb_v1: training-age
# mean / std 53.52 / 19.29 y; the CV fold models read theirs from their own log.
R=/data/runs/heedb/mil_phased_cv; W=$OUT/windows/dortmund_all
A=(--data-path $COHORT --splits test --stage-dir none --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0 --save-windows $W)
if [[ $step == score || $step == all ]]; then
  mkdir -p $W
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --pred-stats 53.52 19.29 \
    --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883 \
    --run E2LrobF=/data/runs/chbp_fix/e2l_robust_full_20261112_175034_869 \
    --out $OUT/dortmund_all.json 2>&1 | tee $OUT/dortmund_all.log
  CV=(); for p in P3 P4; do for k in 0 1 2 3 4; do
    [[ $p == P4 && $k == 3 ]] || CV+=(--run ${p}_L1_f$k=$R/${p}_L1/fold_$k/cv/fold_$k); done; done
  $PY scripts/eval_age_by_cohort.py "${A[@]}" "${CV[@]}" --pred-stats-from-log \
    --out $OUT/dortmund_all_p34.json > $OUT/dortmund_all_p34.log 2>&1
  # P4 fold 3 has no run.log and shares P3 fold 3's CV split: 53.55 / 19.29
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --run P4_L1_f3=$R/P4_L1/fold_3/cv/fold_3 --pred-stats 53.55 19.29 \
    --out $OUT/dortmund_all_p4f3.json > $OUT/dortmund_all_p4f3.log 2>&1
fi

if [[ $step == ensemble || $step == score || $step == all ]]; then
  $PY scripts/ensemble_predictions.py --eval $OUT/dortmund_all.json $OUT/dortmund_all_p34.json \
    $OUT/dortmund_all_p4f3.json --ensemble "P3ens=P3_L1_f*" --ensemble "P4ens=P4_L1_f*" \
    --out $OUT/dortmund_all_models.json
fi

if [[ $step == tables || $step == all ]]; then
  mkdir -p $D docs/figures/dortmund
  $PY scripts/dortmund_brain_age.py --eval $OUT/dortmund_all_models.json --cohort $COHORT \
    --windows $W --models E2L E2LrobF P3ens P4ens --out-dir $D --fig-dir docs/figures/dortmund
  # which sessions each label-free rejection score sets aside (the learned score is fitted on
  # validation folds of the training cohort: each P4 fold model's own fold at its best epoch)
  V4=$R/P4_L1/fold_1/cv/fold_1/analysis/val_bag_weights_epoch13.parquet,$R/P4_L1/fold_2/cv/fold_2/analysis/val_bag_weights_epoch15.parquet,$R/P4_L1/fold_4/cv/fold_4/analysis/val_bag_weights_epoch15.parquet
  $PY scripts/rejection_scatter.py --set dortmund:$COHORT:$OUT/dortmund_all_models.json:$W --model P4ens \
    --val-windows $V4 --out $D/rejection_overlap.csv --figure docs/figures/dortmund/dortmund_rejection_scatter.png
fi
