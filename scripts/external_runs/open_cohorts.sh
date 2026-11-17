#!/usr/bin/env bash
# Open-licence external test cohorts onboarded after Dortmund (docs/open_cohorts_onboarding.md):
#
#   ahepa      OpenNeuro ds004504  Alzheimer's / frontotemporal dementia / controls, 88, CC0
#   iowapd     OpenNeuro ds004584  Parkinson's / controls, 149, CC0
#   telavivpd  OpenNeuro ds007526  Parkinson's / controls, 144 (resting run), CC0
#   srm        OpenNeuro ds003775  healthy, 111 (+42 second sessions), CC0
#   nmt        Zenodo 10.5281/zenodo.21629135  routine clinical EEG, 2,417, normal / abnormal, CC BY 4.0
#
# Test-only: nothing here fits, tunes, selects or calibrates a model.
#
#   scripts/external_runs/open_cohorts.sh <cohort> <step>
#   steps: fetch | mirror | convert | score | tables | all (convert + score + tables)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
E=/data/datasets/EEG-public
OUT=/data/runs/heedb/external_eval
name=${1:?cohort: ahepa | iowapd | telavivpd | srm | nmt}
step=${2:-all}

# cohort -> folder, source, raw root, participants table, trim options of the scoring
TRIM=(--trim-start-sec 0 --trim-end-sec 0)      # short recordings (2-4 min): no trim, as TDBRAIN / Dortmund
case $name in
  ahepa)     DIR=AHEPA;     SRC=ds004504; TRIM=() ;;        # 5-21 min: the usual 60 s trims
  iowapd)    DIR=IOWAPD;    SRC=ds004584 ;;
  telavivpd) DIR=TELAVIVPD; SRC=ds007526 ;;
  srm)       DIR=SRM;       SRC=ds003775 ;;
  nmt)       DIR=NMT;       SRC=zenodo;   TRIM=() ;;        # about 15 min: the usual 60 s trims
  *) echo "unknown cohort $name"; exit 1 ;;
esac
ROOT=$E/$DIR
COHORT=$ROOT/cohorts/${name}_v1
S3=s3://eeg-data-public/$DIR/v1
if [[ $SRC == zenodo ]]; then RAW=$ROOT/raw/nmt_scalp_eeg_dataset; META=$RAW/Labels.csv
else RAW=$ROOT/raw/$SRC; META=$RAW/participants.tsv; fi
D=docs/data/external_eval/$name

if [[ $step == fetch ]]; then
  mkdir -p "$ROOT"/{raw,meta,cohorts}
  if [[ $SRC == zenodo ]]; then
    curl -sS -L -C - -A curl --retry 4 -o "$ROOT/raw/NMT-Scalp-EEG.zip" \
      "https://zenodo.org/api/records/21629135/files/NMT-Scalp-EEG.zip/content"
    echo "91fc1407ba2ee2018988a3d4ca87a4d0  $ROOT/raw/NMT-Scalp-EEG.zip" | md5sum -c -
    $PY -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ROOT/raw/NMT-Scalp-EEG.zip" "$ROOT/raw"
  else   # public OpenNeuro bucket, no AWS identity sent; preprocessed derivatives are not needed
    aws s3 ls --no-sign-request --recursive s3://openneuro.org/$SRC/ > "$ROOT/meta/listing.txt"
    aws s3 sync --no-sign-request s3://openneuro.org/$SRC/ "$RAW/" --exclude ".datalad/*" --exclude "derivatives/*" \
      --only-show-errors
  fi
fi

private() { $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public'); print('bucket is private')"; }

if [[ $step == mirror ]]; then
  private
  aws s3 sync "$ROOT/raw" "$S3/raw" --only-show-errors --exclude "*.zip"
  aws s3 sync "$ROOT/meta" "$S3/meta" --only-show-errors
fi

if [[ $step == convert || $step == all ]]; then
  $PY -m dataset_maker.make_external_npy --dataset $name --raw "$RAW" --meta "$META" --out $COHORT --jobs ${JOBS:-6}
  private
  aws s3 sync $COHORT "$S3/cohorts/${name}_v1" --only-show-errors
fi

R=/data/runs/heedb/mil_phased_cv; W=$OUT/windows/${name}_all
A=(--data-path $COHORT --splits test --stage-dir none --eval-minutes 0 "${TRIM[@]}" --save-windows $W)
if [[ $step == score || $step == all ]]; then
  mkdir -p $W
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --pred-stats 53.52 19.29 \
    --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883 \
    --run E2LrobF=/data/runs/chbp_fix/e2l_robust_full_20261112_175034_869 \
    --out $OUT/${name}_all.json > $OUT/${name}_all.log 2>&1
  CV=(); for p in P3 P4; do for k in 0 1 2 3 4; do
    [[ $p == P4 && $k == 3 ]] || CV+=(--run ${p}_L1_f$k=$R/${p}_L1/fold_$k/cv/fold_$k); done; done
  $PY scripts/eval_age_by_cohort.py "${A[@]}" "${CV[@]}" --pred-stats-from-log \
    --out $OUT/${name}_all_p34.json > $OUT/${name}_all_p34.log 2>&1
  # P4 fold 3 has no run.log and shares P3 fold 3's CV split: 53.55 / 19.29
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --run P4_L1_f3=$R/P4_L1/fold_3/cv/fold_3 --pred-stats 53.55 19.29 \
    --out $OUT/${name}_all_p4f3.json > $OUT/${name}_all_p4f3.log 2>&1
  $PY scripts/ensemble_predictions.py --eval $OUT/${name}_all.json $OUT/${name}_all_p34.json \
    $OUT/${name}_all_p4f3.json --ensemble "P3ens=P3_L1_f*" --ensemble "P4ens=P4_L1_f*" \
    --out $OUT/${name}_all_models.json
  grep -h "test all" $OUT/${name}_all.log | cut -c1-60
fi

if [[ $step == tables || $step == all ]]; then
  mkdir -p $D
  V3=$(for k in 0 1 2 3; do printf "%s," $R/P3_L1/fold_$k/cv/fold_$k/analysis/val_bag_weights_epoch16.parquet; done | sed 's/,$//')
  V4=$R/P4_L1/fold_1/cv/fold_1/analysis/val_bag_weights_epoch13.parquet,$R/P4_L1/fold_2/cv/fold_2/analysis/val_bag_weights_epoch15.parquet,$R/P4_L1/fold_4/cv/fold_4/analysis/val_bag_weights_epoch15.parquet
  SET=(--set $name:$COHORT:$OUT/${name}_all_models.json:$W)
  # all / healthy / pathological, without and with the three label-free rejection scores
  $PY scripts/external_cohort_table.py "${SET[@]}" --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --models E2L E2LrobF P3ens P4ens --out $D/external_cohort_table.csv
  # brain-age gap of each clinical group against the cohort's controls
  $PY scripts/clinical_cohort_gap.py --cohort $COHORT --eval $OUT/${name}_all_models.json \
    --models E2L E2LrobF P3ens P4ens --out $D/gap_by_group.csv
fi
