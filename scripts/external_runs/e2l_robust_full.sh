#!/usr/bin/env bash
# E2L recipe from scratch (LaBraM base, tuh_heedb_v1, 16 epochs / 4 warm-up, 5 min per
# recording per epoch, mixup, batch 256 / lr 2.8e-4 = the settings E2L finished with)
# with the device-robust inputs: band 1-45 Hz + per-window amplitude normalization
# (labram_plus.band_hz / amplitude_norm; docs/chbp_domain_shift.md section 10.2).
# CHBP, LEMON and MIPDB are extra test sets: scored every 5 epochs and at the best
# epoch, never used for selection (selection = in-domain validation MAE).
#   scripts/external_runs/e2l_robust_full.sh            # ~19 h on one A10G; run detached
#   scripts/external_runs/e2l_robust_full.sh --set trainer.debug=true trainer.epochs=1 ...
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 LABRAM_NPY_READ=pread
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
OUT=${OUT:-/data/runs/chbp_fix/e2l_robust_full}
extra=()
[[ ${1:-} == --set ]] && { shift; extra=("$@"); }
$PY -m labram.runs.run_finetune --config $E2L/run_config.yaml --set \
  labram_plus.band_hz=[1.0,45.0] labram_plus.amplitude_norm=window labram_plus.amplitude_target=0.1 \
  finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth \
  trainer.batch_size=256 optimizer.lr=0.00028 optimizer.warmup_epochs=4 trainer.epochs=16 \
  trainer.early_stopping=true trainer.early_stopping_patience=5 \
  data.train_minutes_per_epoch=5 data.stage_dir=off data.num_workers=12 \
  evaluation.train_eval_every=4 evaluation.test_sets_every=5 evaluation.test_sets_at_best=true \
  'evaluation.test_sets=["temple","lemon=/data/datasets/EEG-public/LEMON/cohorts/lemon_v1","chbp=/data/datasets/EEG-public/CHBP/cohorts/chbp_v1","mipdb=/data/datasets/EEG-public/MIPDB/cohorts/mipdb_v1"]' \
  output.output_dir=$OUT output.log_dir=$OUT/log \
  output.auto_resume=false output.append_timestamp=true output.save_ckpt_freq=5 \
  clearml.project_name=LaBraM/heedb clearml.task_name=heedb_E2L_band1-45_ampnorm_ep16_local \
  clearml.upload_model_artifact=false \
  'clearml.tags=["heedb","E2L","chbp-fix","band1-45","ampnorm","ep16","local","full"]' \
  "${extra[@]}"
