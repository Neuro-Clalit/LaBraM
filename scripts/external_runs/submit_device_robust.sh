#!/usr/bin/env bash
# Device-robust brain-age variants of the E2L recipe on SageMaker (docs/chbp_domain_shift.md
# section 10). Same recipe as E2L (LaBraM base, tuh_heedb_v1, 16 epochs / 4 warm-up, 5 min per
# recording per epoch, mixup, global batch 256 / lr 2.8e-4, early stopping 5), 4-GPU DDP on one
# ml.g5.12xlarge; CHBP / LEMON / MIPDB / temple are extra test sets (scored, never selected on).
#   band      -- (2) band 1-45 Hz only, absolute amplitude kept; at test time one cohort-level gain
#   relamp    -- (3) band 1-45 Hz + per-window amplitude normalization + the recording's amplitude
#                relative to its acquisition group as a head input (rel_amplitude_v1.npz)
#   augment   -- (4) raw input, training-only random gain (0.55-1.8x) and band-edge augmentation
#   scripts/external_runs/submit_device_robust.sh band [--dry_run | --detach]
#   INSTANCE=ml.g6e.12xlarge scripts/external_runs/submit_device_robust.sh augment --detach   # other instance type
#   (one ml.g5.12xlarge per account for on-demand and one for spot: add --set sagemaker.use_spot=true)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/data/code/LaBraM/.venv/bin/python}
E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
variant=${1:?band | relamp | augment}; shift
TESTS='evaluation.test_sets=["temple","lemon=/data/datasets/EEG-public/LEMON/cohorts/lemon_v1","chbp=/data/datasets/EEG-public/CHBP/cohorts/chbp_v1","mipdb=/data/datasets/EEG-public/MIPDB/cohorts/mipdb_v1"]'
case $variant in
  band)    opts=(labram_plus.band_hz=[1.0,45.0]) ;;
  relamp)  opts=(labram_plus.band_hz=[1.0,45.0] labram_plus.amplitude_norm=window labram_plus.amplitude_target=0.1
                 data.session_features=rel_amplitude_v1.npz model.session_features=head) ;;
  augment) opts=(augment.enabled=true augment.hemisphere_flip_prob=0 augment.sign_flip_prob=0
                 augment.ft_surrogate_prob=0 augment.channel_dropout_prob=0 augment.time_mask_prob=0
                 augment.gain_prob=0.5 augment.gain_min=0.55 augment.gain_max=1.8
                 augment.lowpass_prob=0.5 augment.lowpass_min_hz=40 augment.lowpass_max_hz=75
                 augment.highpass_prob=0.3 augment.highpass_min_hz=0.1 augment.highpass_max_hz=1.0) ;;
  *) echo "unknown variant $variant"; exit 2 ;;
esac
$PY -m labram.runs.submit_sagemaker --phase finetune --config $E2L/run_config.yaml "$@" --set \
  data.data_path=s3://eeg-data-public/HEEDB/v4.1/cohorts/tuh_heedb_v1_compact data.data_format=npy \
  data.stage_dir=off data.num_workers=10 data.train_minutes_per_epoch=5 \
  finetune_checkpoint.finetune=/data/code/LaBraM/checkpoints/labram-base.pth \
  optimizer.lr=0.00028 optimizer.warmup_epochs=4 trainer.epochs=16 \
  trainer.batch_size=64 trainer.global_batch_size=256 distributed.nproc_per_node=-1 \
  trainer.early_stopping=true trainer.early_stopping_patience=5 \
  evaluation.train_eval_every=4 evaluation.test_sets_every=5 evaluation.test_sets_at_best=true "$TESTS" \
  "${opts[@]}" \
  output.output_dir= output.log_dir= output.auto_resume=false \
  clearml.enabled=true clearml.project_name=LaBraM/heedb clearml.task_name=heedb_E2L_robust_${variant}_ddp4 \
  clearml.upload_model_artifact=true \
  "clearml.tags=[\"heedb\",\"E2L\",\"chbp-fix\",\"device-robust\",\"${variant}\",\"ddp4\"]" \
  sagemaker.enabled=true sagemaker.role=arn:aws:iam::574441342949:role/SageMakerExecutionRole \
  sagemaker.input_mode=File sagemaker.instance_type=${INSTANCE:-ml.g5.12xlarge} sagemaker.volume_size_gb=1000 \
  sagemaker.job_name_prefix=labram-e2l-robust-${variant}
