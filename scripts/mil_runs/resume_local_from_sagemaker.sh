#!/bin/bash
# Continue a SageMaker fold job of scripts/mil_runs/submit_grid.sh on the local GPU, from its
# checkpoint folder on S3 and with the job's own saved configuration (the fold's run_config.yaml):
# only what differs between the training container and this machine is overridden (data, split
# and weight paths, test-set paths, loader workers, output folder, ClearML task name).
#
#   AWS_PROFILE=mfa scripts/mil_runs/resume_local_from_sagemaker.sh fetch RV5_L1 \
#       s3://sagemaker-us-east-1-574441342949/labram/checkpoints/labram-mil-rv5-l1-f0-fold-0-20261126T125136
#   systemd-run --user --unit=labram-rv5-l1-f0 --collect --working-directory=$PWD \
#       --setenv=WEIGHTS=./checkpoints/reve-base.safetensors /bin/bash -c \
#       'scripts/mil_runs/resume_local_from_sagemaker.sh run RV5_L1 >> /data/runs/heedb/fm_encoders/RV5_L1_fold0.log 2>&1'
#
# fetch: checkpoint.pth, checkpoint-best.pth, log.txt, train_state.json, the split and the
#        analysis files -> /data/runs/heedb/fm_encoders/heedb_mil_<JOB>_fold<FOLD>_local_cv5/fold_<FOLD>.
#        Fetch after an epoch has been scored (its row is in log.txt), so no epoch is lost.
# run:   labram.runs.finetune_cv with output.auto_resume=true: continues after the saved epoch,
#        with the optimiser, schedule and best-so-far state of the job. Extra overrides may follow.
#
# The two runs do not interfere (different folders and ClearML tasks): start the local run, check
# that it trains, then stop the SageMaker job. Its checkpoints stay on S3, so the move can be undone
# with RESUME=<checkpoint folder> of submit_grid.sh. As for every long local run, start it as its
# own systemd unit (header of run_local_fm.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
STEP=${1:?fetch | run}
JOB=${2:?job name of submit_grid.sh, e.g. RV5_L1}
FOLD=${FOLD:-0}
BASE=${OUT_ROOT:-/data/runs/heedb/fm_encoders}/heedb_mil_${JOB}_fold${FOLD}_local_cv5
DIR=$BASE/fold_$FOLD
E=/data/datasets/EEG-public
case $STEP in
  fetch)
    S3=${3:?the S3 checkpoint folder of the job}
    S3=${S3%/}/cv/fold_$FOLD
    mkdir -p "$DIR/analysis"
    for f in checkpoint.pth checkpoint-best.pth log.txt train_state.json data_split.json; do
      aws s3 cp --only-show-errors "$S3/$f" "$DIR/$f"
    done
    # Kept under other names: the local run writes its own run_config.yaml and cv_split.json.
    aws s3 cp --only-show-errors "$S3/run_config.yaml" "$DIR/run_config_sagemaker.yaml"
    aws s3 cp --only-show-errors "$S3/cv_split.json" "$DIR/cv_split_sagemaker.json"
    aws s3 sync --only-show-errors "$S3/analysis/" "$DIR/analysis/"
    "$PY" - "$DIR" <<'PYEOF'
import json
import sys

import torch

c = torch.load(sys.argv[1] + "/checkpoint.pth", map_location="cpu", weights_only=False)
rows = [json.loads(line) for line in open(sys.argv[1] + "/log.txt")]
print("checkpoint: after epoch", c["epoch"], "| log.txt: epochs scored up to", rows[-1]["epoch"])
if rows[-1]["epoch"] != c["epoch"]:
    print("NOTE: checkpoint and log.txt are of different epochs; fetch again once the epoch is scored")
PYEOF
    ;;
  run)
    "$PY" -m labram.runs.finetune_cv --config "$DIR/run_config_sagemaker.yaml" --set \
      cross_validation.base_dir="$BASE" cross_validation.split_json="$DIR/cv_split_sagemaker.json" \
      data.data_path="${DATA:-/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1}" data.stage_dir=off \
      data.num_workers="${WORKERS:-10}" \
      finetune_checkpoint.finetune="${WEIGHTS:?set WEIGHTS to the released weight file of the encoder}" \
      "evaluation.test_sets=[\"temple\",\"lemon=$E/LEMON/cohorts/lemon_v1\",\"tdbrain=$E/TDBRAIN/cohorts/tdbrain_v1?trim_start_sec=0&trim_end_sec=0&eval_minutes=0&age_min=18\"]" \
      output.auto_resume=true sagemaker.enabled=false sagemaker.resume_checkpoint_s3_uri= \
      clearml.project_name=LaBraM/heedb_mil clearml.task_name="heedb_mil_${JOB}_fold${FOLD}_local" \
      "clearml.tags=[\"heedb\",\"mil\",\"${JOB}\",\"local\"]" "${@:3}"
    ;;
  *) echo "usage: $0 fetch <JOB> <s3 checkpoint folder> | run <JOB> [overrides]" >&2; exit 1 ;;
esac
