#!/usr/bin/env bash
# TDBRAIN (van Dijk et al. 2022, brainclinics.com/resources): fetch, unpack, private
# S3 mirror, convert to the npy cohort format with the clinical table, and score the
# brain-age models zero-shot, as for LEMON / MIPDB / CHBP (docs/tdbrain_onboarding.md).
#
# Access is per person: each user logs in at https://brainclinics.com/resources with
# ORCID and accepts the Data Use Agreement. The DUA forbids sharing the data with anyone
# (lab members included) who has not accepted it themselves, and forbids changing the
# raw files. The Synapse copy (syn25671079) is not downloadable (ACL: authors only).
#
#   step fetch   : download the zip parts listed in $ROOT/download_urls.txt (one URL per
#                  line, copied from the download page after login), or place the zips
#                  in $ROOT/zips by hand (browser download + scp)
#   step unpack  : extract into $ROOT/raw (the zips are kept unchanged)
#   step mirror  : copy zips + raw to the private bucket (refuses if it is public)
#   step convert / score / report / metrics : as scripts/external_runs/chbp.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
ROOT=${TDBRAIN_ROOT:-/data/datasets/EEG-public/TDBRAIN}
S3=s3://eeg-data-public/TDBRAIN/v1
COHORT=$ROOT/cohorts/tdbrain_v1
PY=${PY:-/opt/pytorch/bin/python}            # training env (mne, torch)
OUT=/data/runs/heedb/external_eval
step=${1:-all}

bids_root() {   # the folder holding the BIDS participants.tsv (not the derivatives copy)
  find "$ROOT/raw" -name participants.tsv -not -path '*derivatives*' -printf '%d %h\n' | sort -n | head -1 | cut -d' ' -f2
}

if [[ $step == fetch ]]; then
  mkdir -p "$ROOT/zips"
  [[ -s $ROOT/download_urls.txt ]] || { echo "put the download URLs in $ROOT/download_urls.txt"; exit 1; }
  while read -r url; do
    [[ -z $url || $url == \#* ]] && continue
    name=$(basename "${url%%\?*}")
    curl -fL --retry 4 -C - -o "$ROOT/zips/$name" "$url"
  done < "$ROOT/download_urls.txt"
  ls -la "$ROOT/zips"
fi

if [[ $step == unpack || $step == all ]]; then
  # The EEG zip is password protected (password e-mailed by Brainclinics after the DUA).
  # It is read from $TDBRAIN_ZIP_PASSWORD or $ROOT/.zip_password (mode 600, outside git
  # and outside the S3 mirror). 7z (apt-get install p7zip-full) handles ZipCrypto and
  # AES, Info-ZIP splits (.z01 ... .zip, opened through the .zip) and byte splits (.zip.001).
  command -v 7z >/dev/null || { echo "install p7zip-full (7z) first"; exit 1; }
  PW=${TDBRAIN_ZIP_PASSWORD:-$( [[ -r $ROOT/.zip_password ]] && head -n1 "$ROOT/.zip_password" )}
  mkdir -p "$ROOT/raw"
  shopt -s nullglob
  archives=("$ROOT"/zips/*.zip.001)
  [[ ${#archives[@]} -gt 0 ]] || archives=("$ROOT"/zips/*.zip)
  [[ ${#archives[@]} -gt 0 ]] || { echo "no zip in $ROOT/zips"; exit 1; }
  for z in "${archives[@]}"; do
    echo "extracting $(basename "$z")"
    7z x -y -aos -bd -p"$PW" -o"$ROOT/raw" "$z" | grep -E "^(Everything is Ok|Files|Folders|Size|ERROR|Wrong)" || true
  done
  # nested zips (e.g. a dataset zip inside a download bundle) are extracted in place
  find "$ROOT/raw" -name '*.zip' -not -path '*derivatives*' | while read -r inner; do
    echo "extracting nested $(basename "$inner")"
    7z x -y -aos -bd -p"$PW" -o"$(dirname "$inner")" "$inner" | grep -E "^(Everything is Ok|Files|ERROR|Wrong)" || true
  done
  echo "BIDS root: $(bids_root)"
  find "$ROOT/raw" -name '*_eeg.vhdr' -not -path '*derivatives*' | wc -l
fi

if [[ $step == mirror || $step == all ]]; then
  $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public'); print('bucket is private')"
  aws s3 sync "$ROOT/zips" "$S3/zips" --only-show-errors
  aws s3 sync "$ROOT/raw" "$S3/raw" --only-show-errors
  aws s3 sync "$ROOT/meta" "$S3/meta" --only-show-errors
fi

if [[ $step == convert || $step == all ]]; then
  BIDS=$(bids_root)
  $PY -m dataset_maker.make_external_npy --dataset tdbrain \
    --raw "$BIDS" --meta "$BIDS/participants.tsv" --out $COHORT --jobs 6
  aws s3 sync $COHORT "$S3/cohorts/tdbrain_v1" --only-show-errors
fi

# Every run trained on tuh_heedb_v1: training-age mean / std 53.52 / 19.29 y.
RUNS=(
  --run E2=/data/runs/heedb/e2_r2/finetune
  --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
  --run W4=/data/runs/heedb/mil_sagemaker/artifacts/w4/finetune
  --run L1=/data/runs/heedb/mil_local/L1_abmil_20261109_022309_458
  --run W7=/data/runs/heedb/mil_sagemaker/artifacts/w7/finetune
  --run X1=/data/runs/heedb/mil_sagemaker/artifacts/x1/finetune
  --run X3=/data/runs/heedb/mil_sagemaker/artifacts/x3/finetune
)
if [[ $step == score || $step == all ]]; then
  # Each file is 2 min (12 windows): the training trims (60 s at each end) would leave
  # nothing, so every window is scored ("all") and, as a check on edge effects, the
  # inner 100 s ("inner": 10 s trimmed at each end).
  $PY scripts/eval_age_by_cohort.py --data-path $COHORT --splits test --pred-stats 53.52 19.29 \
    --stage-dir none --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0 "${RUNS[@]}" \
    --out $OUT/tdbrain_all.json 2>&1 | tee $OUT/tdbrain_all.log
  $PY scripts/eval_age_by_cohort.py --data-path $COHORT --splits test --pred-stats 53.52 19.29 \
    --stage-dir none --eval-minutes 0 --trim-start-sec 10 --trim-end-sec 10 "${RUNS[@]}" \
    --out $OUT/tdbrain_inner.json 2>&1 | tee $OUT/tdbrain_inner.log
fi

if [[ $step == report || $step == all ]]; then
  mkdir -p docs/data/external_eval/tdbrain docs/figures/tdbrain
  $PY scripts/external_eval_report.py \
    --eval tdbrain=$OUT/tdbrain_all.json --cohort tdbrain=$COHORT \
    --out-dir docs/data/external_eval/tdbrain --fig-dir docs/figures/tdbrain
fi

if [[ $step == metrics || $step == all ]]; then
  VAL=(); for m in E2L L1 W4 W7; do VAL+=(--val $m=$OUT/own_all_val_$m.json); done
  for tag in all inner; do
    $PY scripts/external_age_metrics.py --eval $OUT/tdbrain_$tag.json \
      --meta $COHORT/age_metadata.json --models E2L L1 W4 W7 --train-mean 53.52 \
      "${VAL[@]}" --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \
      --out docs/data/external_eval/tdbrain/models_E2L_L1_W4_W7_$tag.csv
  done
fi
