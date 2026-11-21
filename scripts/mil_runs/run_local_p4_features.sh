#!/bin/bash
# P4 with handcrafted window features: the phased attention model (P4_L1, scripts/mil_runs/submit_grid.sh)
# on one CV fold, with every window's handcrafted features (labram/layers/window_features.py) joined to
# its class token, so the attention scores, the pool and the age head read both. Everything else is the
# fold's own saved recipe (run_config.yaml of the P4 fold model): same fold partition, same held-out
# test split, same schedule. Runs on the local GPU; then scores every test cohort (eval_p4_features.sh).
#
#   scripts/mil_runs/run_local_p4_features.sh            # fold 0
#   FOLD=2 NAME=P4F_L1 scripts/mil_runs/run_local_p4_features.sh mil.feature_dim=32
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
NAME=${NAME:-P4F_L1}; FOLD=${FOLD:-0}
# The P4 folds were trained on the compact copy of the cohort (s3://eeg-data-public/HEEDB/v4.1/cohorts/
# tuh_heedb_v1_compact), where recordings of the original val / test splits hold only the 5 minutes that
# are scored. DATA is a local view with the same window lists over the full local recordings:
#   mkdir $DATA && aws s3 cp s3://.../tuh_heedb_v1_compact/age_split.json $DATA/ &&
#   ln -s <local tuh_heedb_v1>/{manifest.json,age_metadata.json,cohort.parquet,recordings} $DATA/
DATA=${DATA:-/data/runs/heedb/mil_features/cohort_view_compact}
BASE=${BASE:-/data/runs/heedb/mil_phased_cv/P4_L1/fold_$FOLD/cv}          # the P4 fold model this run is compared with
OUT=${OUT:-/data/runs/heedb/mil_features/$NAME}
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
$PY -m labram.runs.finetune_cv --config $BASE/fold_$FOLD/run_config.yaml --set \
  data.data_path=$DATA data.stage_dir=off data.num_workers=${WORKERS:-10} \
  finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth \
  cross_validation.split_json=$BASE/cv_split.json cross_validation.base_dir=$OUT/cv cross_validation.fold=$FOLD \
  output.output_dir=$OUT/cv/fold_$FOLD output.log_dir=$OUT/cv/fold_$FOLD/log output.auto_resume=true \
  'evaluation.test_sets=["temple","lemon=/data/datasets/EEG-public/LEMON/cohorts/lemon_v1"]' \
  mil.window_features=handcrafted mil.feature_dim=64 mil.feature_dropout=0.1 \
  sagemaker.enabled=false distributed.nproc_per_node=1 \
  clearml.project_name=LaBraM/heedb_mil clearml.task_name=heedb_mil_${NAME}_fold${FOLD} \
  "clearml.tags=[\"heedb\",\"mil\",\"${NAME}\",\"window_features\",\"local\"]" "$@"
bash scripts/mil_runs/eval_p4_features.sh $OUT/cv/fold_$FOLD ${NAME}_f$FOLD
