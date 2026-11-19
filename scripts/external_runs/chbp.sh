#!/usr/bin/env bash
# CHBP (Cuban Human Brain Mapping Project, Synapse syn22324937): download, convert
# to the npy cohort format and score the brain-age models zero-shot, as for
# LEMON / MIPDB (docs/external_eval_results.md). Token: ~/.synapseConfig or
# $SYNAPSE_AUTH_TOKEN (view + download scopes).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
ROOT=/data/datasets/EEG-public/CHBP
PY=${PY:-/opt/pytorch/bin/python}            # training env (mne, torch)
SYNPY=${SYNPY:-/data/venvs/synapse/bin/python}  # synapseclient lives in its own venv
OUT=/data/runs/heedb/external_eval
step=${1:-all}

if [[ $step == download || $step == all ]]; then
  $SYNPY -u -m dataset_maker.download_synapse chbp --out $ROOT/raw --jobs 8
fi

if [[ $step == convert || $step == all ]]; then
  # release 1.0 (the Engemann 2022 benchmark release); 1.1 has the same signals
  $PY -m dataset_maker.make_external_npy --dataset chbp \
    --raw $ROOT/raw/bids_v1.0 --meta $ROOT/raw/meta/cognitive_scales/Demographic_data.csv \
    --out $ROOT/cohorts/chbp_v1 --jobs 6
fi

# Every run trained on tuh_heedb_v1: training-age mean / std 53.52 / 19.29 y (logged by
# E2, E2L, L1; the SageMaker artifacts W4 / W7 / X1 / X3 carry no log/run.log).
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
  # training protocol (first 5 min after the 60 s trim = eyes closed) and every resting window
  $PY scripts/eval_age_by_cohort.py --data-path $ROOT/cohorts/chbp_v1 --splits test \
    --pred-stats 53.52 19.29 --stage-dir none "${RUNS[@]}" --out $OUT/chbp_5min.json 2>&1 | tee $OUT/chbp_5min.log
  $PY scripts/eval_age_by_cohort.py --data-path $ROOT/cohorts/chbp_v1 --splits test \
    --pred-stats 53.52 19.29 --stage-dir none --eval-minutes 0 "${RUNS[@]}" \
    --out $OUT/chbp_all.json 2>&1 | tee $OUT/chbp_all.log
fi

if [[ $step == report || $step == all ]]; then
  mkdir -p docs/data/external_eval/chbp docs/figures/chbp
  $PY scripts/external_eval_report.py \
    --eval chbp=$OUT/chbp_5min.json --cohort chbp=$ROOT/cohorts/chbp_v1 \
    --out-dir docs/data/external_eval/chbp --fig-dir docs/figures/chbp
fi

if [[ $step == metrics || $step == all ]]; then
  # the guide's external-test metrics for E2L / L1 / W4 / W7 + their median ensemble;
  # bias correction fitted on each model's own tuh_heedb_v1 validation predictions
  VAL=(); for m in E2L L1 W4 W7; do VAL+=(--val $m=$OUT/own_all_val_$m.json); done
  for tag in 5min all; do
    $PY scripts/external_age_metrics.py --eval $OUT/chbp_$tag.json \
      --meta $ROOT/cohorts/chbp_v1/age_metadata.json --models E2L L1 W4 W7 --train-mean 53.52 \
      "${VAL[@]}" --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \
      --out docs/data/external_eval/chbp/models_E2L_L1_W4_W7_$tag.csv
  done
fi
