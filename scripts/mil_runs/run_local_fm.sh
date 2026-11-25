#!/bin/bash
# P4 recipe with a foreign pretrained encoder on the local GPU, one fold of P4's CV split
# (docs/fm_encoder_onboarding.md, Section 7): the same overrides as the SageMaker jobs CB4_L1 /
# RV4_L1 of scripts/mil_runs/submit_grid.sh, with local data, split and weights.
#
#   scripts/mil_runs/run_local_fm.sh                      # CBraMod, fold 0
#   ENCODER=reve scripts/mil_runs/run_local_fm.sh         # REVE-base (about 3.5x the time)
#   JOB=RV4DP_L1 ENCODER=reve scripts/mil_runs/run_local_fm.sh model.encoder.drop_path=0.1
#                                                         # a variant: JOB names its folder and task
#   scripts/mil_runs/run_local_fm.sh output.auto_resume=true   # continue after an interruption
#
# The run writes to /data/runs/heedb/fm_encoders/heedb_mil_<JOB>_fold<FOLD>_local_cv5/fold_<FOLD>
# (the CV base folder is named after the ClearML task). About 16 h for CBraMod on one A10G.
# WORKERS stays at 10: every loader worker copies the cohort's file list, and the machine has
# 30 GB of RAM.
#
# Started from a Claude Code session, a run this long must be its own systemd unit: the nightly
# Claude Code update restarts the Remote Control service, and systemd then stops every process
# of that service, nohup / setsid or not (it ended the first CBraMod run in epoch 5):
#   systemd-run --user --unit=labram-cb4-l1-f0 --collect --working-directory=$PWD /bin/bash -c \
#     'scripts/mil_runs/run_local_fm.sh >> /data/runs/heedb/fm_encoders/CB4_L1_fold0.log 2>&1'
#   systemctl --user status labram-cb4-l1-f0        # journalctl --user -u labram-cb4-l1-f0
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
ENCODER=${ENCODER:-cbramod}
FOLD=${FOLD:-0}
SPLIT=${SPLIT:-/data/runs/heedb/mil_phased_cv/P4_L1/fold_0/cv/fold_0/cv_split.json}
case $ENCODER in
  cbramod) JOB=${JOB:-CB4_L1}
    ENC=(model.model=cbramod_base_patch200 labram_plus.band_hz=[0.3,99.0] model.input_clip_uv=100
         finetune_checkpoint.finetune=./checkpoints/cbramod-base.pth) ;;
  reve) JOB=${JOB:-RV4_L1}
    ENC=(model.model=reve_base_patch200 model.encoder.input_norm=zscore_window optimizer.layer_decay=0.8
         trainer.batch_size=16 trainer.update_freq=2 mil.window_batch_size=128
         finetune_checkpoint.finetune=./checkpoints/reve-base.safetensors) ;;
  *) echo "ENCODER: cbramod | reve" >&2; exit 1 ;;
esac
OUT=${OUT:-/data/runs/heedb/fm_encoders/$JOB}
# BASE, PHASED4, CV and BAG of submit_grid.sh, then the encoder
$PY -m labram.runs.finetune_cv --config labram/configs/defaults/finetune_tuab_age.json --set \
  data.data_path=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 data.data_format=npy data.stage_dir=off \
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0 \
  channel_drop.consistency_weight=0.0 channel_drop.eval_consistency=false \
  data.num_workers=${WORKERS:-10} mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false \
  evaluation.agg_windows=mean trainer.early_stopping=false \
  optimizer.lr=0.00028 optimizer.warmup_epochs=3 trainer.epochs=17 \
  mil.window_epochs=8 mil.window_minutes_per_epoch=1.3334 mil.window_bag_repeats=4 mil.window_batch_size=256 \
  mil.freeze_encoder_epochs=1 mil.freeze_head=true mil.window_loss_weight=0.5 mil.bag_repeats=2 \
  evaluation.train_eval_every=4 \
  cross_validation.enabled=true cross_validation.n_folds=5 cross_validation.fold=$FOLD \
  cross_validation.pool=train_val cross_validation.test_set=held_out cross_validation.split_by=subject \
  cross_validation.seed=0 cross_validation.split_json=$SPLIT \
  mil.enabled=true mil.bag_size=8 trainer.batch_size=32 evaluation.eval_batch_size=12 \
  mil.head_lr_scale=5.0 mil.entropy_weight=0.01 mil.dropout=0.1 \
  mil.pool=gated mil.gated=false mil.score_input=h \
  "${ENC[@]}" \
  output.output_dir=$OUT output.log_dir=$OUT \
  clearml.enabled=true clearml.project_name=LaBraM/heedb_mil clearml.task_name=heedb_mil_${JOB}_fold${FOLD}_local \
  "clearml.tags=[\"heedb\",\"mil\",\"${JOB}\",\"local\"]" "$@"
