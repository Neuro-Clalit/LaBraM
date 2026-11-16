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

bids_root() {   # the folder holding the BIDS dataset_description.json (not a derivatives copy)
  find "$ROOT/raw" -name dataset_description.json -not -path '*derivatives*' -printf '%d %h\n' | sort -n | head -1 | cut -d' ' -f2
}
# participants table: $TDBRAIN_META, else the BIDS participants.tsv, else a participants
# file under meta/ (release 3.1 ships the EEG zip without one)
participants() {
  if [[ -n ${TDBRAIN_META:-} ]]; then echo "$TDBRAIN_META"; return; fi
  { find "$(bids_root)" -maxdepth 1 -name participants.tsv; \
    find "$ROOT/meta" -maxdepth 1 -iname '*participant*' \( -name '*.tsv' -o -name '*.csv' -o -name '*.xlsx' \); } | head -1
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
    7z x -y -aos -bd -p"$PW" -o"$ROOT/raw" -x'r!.DS_Store' "$z" | grep -E "^(Everything is Ok|Files|Folders|Size|ERROR|Wrong)" || true
  done
  # nested zips (e.g. a dataset zip inside a download bundle) are extracted in place
  find "$ROOT/raw" -name '*.zip' -not -path '*derivatives*' | while read -r inner; do
    echo "extracting nested $(basename "$inner")"
    7z x -y -aos -bd -p"$PW" -o"$(dirname "$inner")" "$inner" | grep -E "^(Everything is Ok|Files|ERROR|Wrong)" || true
  done
  echo "BIDS root: $(bids_root)"
  find "$ROOT/raw" \( -name '*_eeg.vhdr' -o -name '*_eeg.bdf' \) -not -path '*derivatives*' | wc -l
fi

if [[ $step == mirror || $step == all ]]; then
  $PY -c "import boto3; from dataset_maker.download_heedb import check_private; \
check_private(boto3.client('s3'), 'eeg-data-public'); print('bucket is private')"
  aws s3 sync "$ROOT/zips" "$S3/zips" --only-show-errors
  aws s3 sync "$ROOT/raw" "$S3/raw" --only-show-errors
  aws s3 sync "$ROOT/meta" "$S3/meta" --only-show-errors
fi

if [[ $step == convert || $step == all ]]; then
  # without a participants table the signals are converted with a flagged placeholder age;
  # step "relabel" adds the real ages later without converting again
  BIDS=$(bids_root); META=$(participants)
  echo "BIDS root: $BIDS; participants: ${META:-none (placeholder ages)}"
  $PY -m dataset_maker.make_external_npy --dataset tdbrain \
    --raw "$BIDS" ${META:+--meta "$META"} --out $COHORT --jobs 7
  aws s3 sync $COHORT "$S3/cohorts/tdbrain_v1" --only-show-errors
fi

if [[ $step == relabel ]]; then
  BIDS=$(bids_root); META=$(participants)
  [[ -n $META ]] || { echo "no participants table: put it in $ROOT/meta or set TDBRAIN_META"; exit 1; }
  $PY -m dataset_maker.make_external_npy --dataset tdbrain --raw "$BIDS" --meta "$META" --out $COHORT --relabel
  aws s3 sync $COHORT "$S3/cohorts/tdbrain_v1" --only-show-errors --exclude 'recordings/*'
fi

# Every run trained on tuh_heedb_v1: training-age mean / std 53.52 / 19.29 y (checked in each run's log).
# MODELS="E2L E2Lrob" limits the scoring to those names.
RUNS=(
  --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
  --run E2=/data/runs/heedb/e2_r2/finetune
  --run W4=/data/runs/heedb/mil_sagemaker/artifacts/w4/finetune
  --run L1=/data/runs/heedb/mil_local/L1_abmil_20261109_022309_458
  --run W7=/data/runs/heedb/mil_sagemaker/artifacts/w7/finetune
  --run X1=/data/runs/heedb/mil_sagemaker/artifacts/x1/finetune
  --run X3=/data/runs/heedb/mil_sagemaker/artifacts/x3/finetune
  # device-robust E2L (band 1-45 Hz + per-window amplitude norm; docs/chbp_domain_shift.md §10):
  # fine-tuned from E2L (+6 epochs) and trained from scratch with the E2L recipe
  --run E2Lrob=/data/runs/chbp_fix/e2l_band45_amp_cont_20261112_001004_075
  --run E2LrobF=/data/runs/chbp_fix/e2l_robust_full_20261112_175034_869
)
if [[ -n ${MODELS:-} ]]; then          # keep only the --run pairs named in $MODELS
  sel=(); for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
    [[ " $MODELS " == *" ${RUNS[i+1]%%=*} "* ]] && sel+=("${RUNS[i]}" "${RUNS[i+1]}")
  done; RUNS=("${sel[@]}")
fi
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

if [[ $step == score_cv || $step == all ]]; then
  # phased one-run MIL models P3_L1 / P4_L1, 5-fold CV each (docs/mil_stage1_results.md §13), every
  # window of TDBRAIN / LEMON / CHBP / MIPDB, each fold de-normalised with its own training stats
  # (log/run.log); P4 fold 3 has no run.log and shares P3 fold 3's CV split: 53.55 / 19.29
  R=/data/runs/heedb/mil_phased_cv; E=/data/datasets/EEG-public
  CV=(); for p in P3 P4; do for k in 0 1 2 3 4; do
    [[ $p == P4 && $k == 3 ]] || CV+=(--run ${p}_L1_f$k=$R/${p}_L1/fold_$k/cv/fold_$k); done; done
  for spec in "tdbrain $COHORT --trim-start-sec 0 --trim-end-sec 0" "lemon $E/LEMON/cohorts/lemon_v1" \
              "chbp $E/CHBP/cohorts/chbp_v1" "mipdb $E/MIPDB/cohorts/mipdb_v1"; do
    set -- $spec; c=$1; path=$2; shift 2
    $PY scripts/eval_age_by_cohort.py --data-path $path --splits test --stage-dir none --eval-minutes 0 "$@" \
      "${CV[@]}" --pred-stats-from-log --out $OUT/${c}_all_p34.json > $OUT/${c}_all_p34.log 2>&1
    $PY scripts/eval_age_by_cohort.py --data-path $path --splits test --stage-dir none --eval-minutes 0 "$@" \
      --run P4_L1_f3=$R/P4_L1/fold_3/cv/fold_3 --pred-stats 53.55 19.29 --out $OUT/${c}_all_p4f3.json > $OUT/${c}_all_p4f3.log 2>&1
  done
fi

if [[ $step == windows || $step == score_cv || $step == all ]]; then
  # per-window predictions for the label-free 20 % rejection by spread (tdbrain_brain_age.py --windows)
  R=/data/runs/heedb/mil_phased_cv; W=$OUT/windows/tdbrain_all; mkdir -p $W
  A=(--data-path $COHORT --splits test --stage-dir none --eval-minutes 0 --trim-start-sec 0 --trim-end-sec 0 --save-windows $W)
  CV=(); for p in P3 P4; do for k in 0 1 2 3 4; do
    [[ $p == P4 && $k == 3 ]] || CV+=(--run ${p}_L1_f$k=$R/${p}_L1/fold_$k/cv/fold_$k); done; done
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883 \
    --pred-stats 53.52 19.29 --out $OUT/tdbrain_all_w_e2l.json
  $PY scripts/eval_age_by_cohort.py "${A[@]}" "${CV[@]}" --pred-stats-from-log --out $OUT/tdbrain_all_w_p34.json
  $PY scripts/eval_age_by_cohort.py "${A[@]}" --run P4_L1_f3=$R/P4_L1/fold_3/cv/fold_3 --pred-stats 53.55 19.29 \
    --out $OUT/tdbrain_all_w_p4f3.json
fi

if [[ $step == ensemble || $step == score_cv || $step == all ]]; then
  # 5-fold ensembles (mean of the fold models' recording predictions) next to the E2L family
  ENS=(--ensemble "P3ens=P3_L1_f*" --ensemble "P4ens=P4_L1_f*" --ensemble "P34ens=P[34]_L1_f*")
  $PY scripts/ensemble_predictions.py --eval $OUT/tdbrain_all.json $OUT/tdbrain_all_p34.json \
    $OUT/tdbrain_all_p4f3.json "${ENS[@]}" --out $OUT/tdbrain_all_models.json
  for c in lemon chbp mipdb; do
    $PY scripts/ensemble_predictions.py --eval $OUT/${c}_all.json $OUT/${c}_all_e2lrob.json \
      $OUT/${c}_all_p34.json $OUT/${c}_all_p4f3.json "${ENS[@]}" --out $OUT/${c}_all_models.json
  done
fi

if [[ $step == report || $step == all ]]; then
  # data analytics as for CHBP: metrics per group, predicted vs true age, signal QC and PSD
  # against the TUH / HEEDB references, TDBRAIN (adults) next to CHBP and LEMON (E2L family)
  mkdir -p docs/data/external_eval/tdbrain docs/figures/tdbrain
  for c in chbp lemon; do
    $PY -c "import json, sys; o = sys.argv[1]; c = sys.argv[2]; \
json.dump({'E2L': json.load(open(f'{o}/{c}_all.json'))['E2L'], **json.load(open(f'{o}/{c}_all_e2lrob.json'))}, \
open(f'{o}/{c}_all_e2l_family.json', 'w'))" $OUT $c
  done
  E=/data/datasets/EEG-public
  $PY scripts/external_eval_report.py --plot-models E2L E2Lrob \
    --eval tdbrain=$OUT/tdbrain_all.json --cohort tdbrain=$COHORT --min-age tdbrain=18 \
    --eval chbp=$OUT/chbp_all_e2l_family.json --cohort chbp=$E/CHBP/cohorts/chbp_v1 \
    --eval lemon=$OUT/lemon_all_e2l_family.json --cohort lemon=$E/LEMON/cohorts/lemon_v1 \
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

if [[ $step == compare || $step == all ]]; then
  # zero-shot metrics per cohort, every window, for the E2L family (Table 8.1 of the doc)
  E=/data/datasets/EEG-public
  $PY scripts/external_cohort_comparison.py \
    --set own_test:/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1:$OUT/own_all_w.json \
    --set lemon:$E/LEMON/cohorts/lemon_v1:$OUT/lemon_all_models.json \
    --set mipdb:$E/MIPDB/cohorts/mipdb_v1:$OUT/mipdb_all_models.json \
    --set chbp:$E/CHBP/cohorts/chbp_v1:$OUT/chbp_all_models.json \
    --set tdbrain:$COHORT:$OUT/tdbrain_all_models.json \
    --models E2L E2Lrob E2LrobF P3_L1_f0 P3_L1_f1 P3_L1_f2 P3_L1_f3 P3_L1_f4 \
             P4_L1_f0 P4_L1_f1 P4_L1_f2 P4_L1_f3 P4_L1_f4 P3ens P4ens P34ens \
    --min-age 18 --out docs/data/external_eval/tdbrain/cohort_comparison_all_windows.csv
  # TDBRAIN in detail, adults only (tdbrain_brain_age.py --min-age, default 18): scoring units,
  # age bands, EC / EO, within-cohort recalibration, brain-age gap per indication
  $PY scripts/tdbrain_brain_age.py --eval $OUT/tdbrain_all_models.json --cohort $COHORT \
    --models E2L E2Lrob E2LrobF P3ens P4ens P34ens --val E2L=$OUT/own_all_val_E2L.json \
    --windows $OUT/windows/tdbrain_all --reject 0.2 \
    --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \
    --out-dir docs/data/external_eval/tdbrain
fi

if [[ $step == tables || $step == all ]]; then
  # Tables of docs Sections 12-16: per cohort, all / healthy / pathological, without rejection and
  # with the three default rejection scores (spread, total, learned); score comparison; bar chart.
  # 'learned' is fitted on validation only: each fold model's own validation fold at its best epoch
  # (folds whose best-epoch val_bag_weights file exists).
  R=/data/runs/heedb/mil_phased_cv; E=/data/datasets/EEG-public; C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
  V3=$(for k in 0 1 2 3; do printf "%s," $R/P3_L1/fold_$k/cv/fold_$k/analysis/val_bag_weights_epoch16.parquet; done | sed 's/,$//')
  V4=$R/P4_L1/fold_1/cv/fold_1/analysis/val_bag_weights_epoch13.parquet,$R/P4_L1/fold_2/cv/fold_2/analysis/val_bag_weights_epoch15.parquet,$R/P4_L1/fold_4/cv/fold_4/analysis/val_bag_weights_epoch15.parquet
  SETS=(--set temple:$C:$OUT/own_5min_models.json:$OUT/windows/own_5min_models:temple
        --set harvard:$C:$OUT/own_5min_models.json:$OUT/windows/own_5min_models:harvard
        --set tdbrain:$COHORT:$OUT/tdbrain_all_models.json:$OUT/windows/tdbrain_all
        --set lemon:$E/LEMON/cohorts/lemon_v1:$OUT/lemon_all_models.json:$OUT/windows/lemon_all_models
        --set chbp:$E/CHBP/cohorts/chbp_v1:$OUT/chbp_all_models.json:$OUT/windows/chbp_all_models
        --set mipdb:$E/MIPDB/cohorts/mipdb_v1:$OUT/mipdb_all_models.json:$OUT/windows/mipdb_all_models)
  D=docs/data/external_eval/tdbrain
  $PY scripts/external_cohort_table.py "${SETS[@]}" --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --models E2L P3ens P4ens --out $D/external_cohort_table.csv
  $PY scripts/rejection_scores.py "${SETS[@]}" --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --models P4ens P3ens --fit-on temple harvard --out $D/rejection_scores.csv \
    --figure docs/figures/tdbrain/rejection_curves.png
  $PY scripts/external_cohort_bars.py --table $D/external_cohort_table.csv --out docs/figures/tdbrain/cohort_mae_bars.png
  # age-bias corrections fitted on normal-report validation predictions; predicted-vs-true figures
  $PY scripts/bias_correction_cohorts.py "${SETS[@]}" --models E2L P3ens P4ens \
    --val E2L=$OUT/own_all_val_E2L.json --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --val-meta $C/age_metadata.json --out $D/bias_correction.csv --fig-dir docs/figures/tdbrain
  # window-level confidence (attention weight, deviation, fold disagreement) against window error
  $PY scripts/window_confidence_figures.py "${SETS[@]}" --model P4ens --out $D/window_confidence.csv \
    --fig-dir docs/figures/tdbrain
  $PY scripts/window_confidence_figures.py "${SETS[@]}" --model P3ens --out $D/window_confidence_P3ens.csv
  # brain age per pathology group and report status (Temple + Harvard): test split and out-of-fold
  $PY scripts/pathology_brain_age.py --cohort $C --test-eval $OUT/own_5min_models.json --models E2L P3ens P4ens \
    --oof P3=$V3 --oof P4=$V4 --out $D/pathology_brain_age.csv --figure docs/figures/tdbrain/pathology_gap.png
  # technical report: one-panel bar chart, then LaTeX + Markdown + PDF from the one source
  $PY scripts/external_cohort_bars.py --table $D/external_cohort_table.csv --reject-score \
    --out docs/tech_report/fig_cohort_mae.png
  $PY scripts/build_tech_report.py
fi
