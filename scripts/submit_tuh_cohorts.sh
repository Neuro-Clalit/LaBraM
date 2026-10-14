#!/usr/bin/env bash
# TUH cohort brain-age experiments on SageMaker, M1 baseline (scenario D +
# mixup 0.4, 4 epochs, 1 warmup). Cohorts are built by
# dataset_maker/make_cohort_dataset.py and copied to s3://.../TUH_EEG/cohorts/.
#   exp2: normal_tueg_v1 -- healthy TUEG patients outside TUAB, train + val, no test
#   exp3: all_both_v1    -- every recording, TUAB + TUEG, test = TUAB eval
#   exp3b: exp3 on a longer schedule (10 epochs, 3 warmup)
#   exp3q: exp3 + signal-quality window rejection in train/val/test (any sample
#          beyond +-800 uV or any channel std < 0.5 uV; needs the cohort's
#          window_stats.npz, python -m labram.data.window_qc). On-demand: its
#          first, spot, attempt (2026-10-13) failed when the instance was
#          reclaimed ("Artifact upload failed: Too many files are written").
#   exp1t: exp1 (normal_both_v1) + session-level classical features as extra
#          transformer tokens: the greedy top-10 families of the handcrafted
#          feature study (riemann, psd_bins, time, cepstral, mspe, multitaper,
#          cwt, dwt, spectral, wpd) -> 19 channel tokens of 272 + 6 Riemann
#          tokens of 171 (dataset_maker/make_session_features.py), each
#          Linear -> LayerNorm (gain 0.5, the EEG tokens' scale)
#   exp3cond: exp3 on all_both_v2 (shared TUEG patient split) + clinical
#          condition vectors concatenated to the pooled feature: status
#          (normal / abnormal / not available) and pathology (multi-hot:
#          healthy-or-n/a + 13 NEDC diagnosis groups),
#          dataset_maker/make_condition_features.py
#   exp3clscond: exp3cond with the head on the class token instead of the mean
#          of the patch tokens (model.use_mean_pooling=false)
#   exp4aug: exp3cond on the class token with all_both_v3 (adds the TUEG
#          recordings lacking only T1/T2) on 23 channels: recordings without
#          T1/T2 run without those tokens; per training batch, with p=0.2,
#          T1/T2 are dropped for every row, the loss is on that pass, plus
#          mean (f_full - f_dropped)^2 on the class token (channel_drop.*)
#   exp4ctrl: control for the local exp4-augT1T2 run -- identical (all_both_v3,
#          class token + conditions, missing channels left out per sample, 6
#          epochs / 2 warmup, fast settings: fused attention, 6 loader workers,
#          batch 128 at lr 2e-4, 5 min per recording per epoch) but no channel
#          augmentation (prob 0); val still reports the T1/T2-dropped MAE. Spot,
#          checkpointed: an interruption resumes from the last epoch
#   exp4base: the full-data baseline -- exp3's model (mean of the patch
#          tokens, no condition vectors, no channel augmentation) on all_both_v3
#          (every TUEG recording, missing channels left out per sample), 6
#          epochs / 2 warmup, fast settings (FAST_BATCH from the speed benchmark)
#   exp1f: the same features with no EEG encoder: an MLP head on the features
#          only (normal_both_v1_meta = the cohort's metadata + features, no EEG)
#   DRY_RUN=1 scripts/submit_tuh_cohorts.sh exp2 exp3
set -euo pipefail
ROLE="${ROLE:-arn:aws:iam::574441342949:role/SageMakerExecutionRole}"
PY="${PY:-python}"
S3=s3://eeg-data-public/TUH_EEG/cohorts
declare -A COHORT=([exp2]=normal_tueg_v1 [exp3]=all_both_v1 [exp3b]=all_both_v1 [exp3q]=all_both_v1
                   [exp1t]=normal_both_v1 [exp1f]=normal_both_v1_meta [exp3cond]=all_both_v2
                   [exp3clscond]=all_both_v2 [exp4aug]=all_both_v3
                   [exp4ctrl]=all_both_v3 [exp4base]=all_both_v3)
declare -A EPOCHS=([exp2]=4 [exp3]=4 [exp3b]=10 [exp3q]=4 [exp1t]=4 [exp1f]=4 [exp3cond]=4 [exp3clscond]=4 [exp4aug]=6 [exp4ctrl]=6 [exp4base]=6)
declare -A WARMUP=([exp2]=1 [exp3]=1 [exp3b]=3 [exp3q]=1 [exp1t]=1 [exp1f]=1 [exp3cond]=1 [exp3clscond]=1 [exp4aug]=2 [exp4ctrl]=2 [exp4base]=2)
declare -A VOLUME=([exp2]=120 [exp3]=590 [exp3b]=590 [exp3q]=590    # GB: File mode copies the cohort first
                   [exp1t]=200 [exp1f]=30 [exp3cond]=590 [exp3clscond]=590 [exp4aug]=590 [exp4ctrl]=590 [exp4base]=590)
# exp4ctrl runs on spot (its own ml.g5.4xlarge spot quota), leaving the one
# ml.g5.4xlarge on-demand slot to exp4base. exp3 (459 GB) exceeds the 450 GB local disk of ml.g5.2xlarge; ml.g5.4xlarge has the
# same A10G GPU and 600 GB of NVMe.
declare -A INSTANCE=([exp2]=ml.g5.2xlarge [exp3]=ml.g5.4xlarge [exp3b]=ml.g5.4xlarge [exp3q]=ml.g5.4xlarge
                     [exp1t]=ml.g5.2xlarge [exp1f]=ml.g5.2xlarge [exp3cond]=ml.g5.4xlarge
                     [exp3clscond]=ml.g5.4xlarge [exp4aug]=ml.g5.4xlarge
                     [exp4ctrl]=ml.g5.4xlarge [exp4base]=ml.g5.4xlarge)
# Extra --set overrides. SPOT (unused for now): exp3 ran 5.9 h, so a 10 h cap
# and up to 24 h of queueing; an interrupted job resumes from its S3 checkpoint.
SPOT="sagemaker.use_spot=true sagemaker.max_run_sec=36000 sagemaker.max_wait_min=2040"
# Speed settings (2026-10-14): fused attention, 6 loader workers, batch 128 at a
# linearly scaled lr, 5 min per recording per epoch, redrawn every epoch.
# FAST_BATCH overrides the batch (lr scales linearly from 1e-4 at 64).
FAST_BATCH=${FAST_BATCH:-128}
FAST_LR=$(python3 -c "print(1e-4 * ${FAST_BATCH} / 64)")
FAST="model.fused_attention=true data.num_workers=6 trainer.batch_size=${FAST_BATCH} optimizer.lr=${FAST_LR} data.train_minutes_per_epoch=5"
FEATS="data.session_features=session_features_v1.npz"
declare -A EXTRA=([exp2]= [exp3]= [exp3b]=
                  [exp3q]="data.window_max_abs_uv=800 data.window_min_std_uv=0.5"
                  [exp1t]="data.session_features=session_features_top10_v1.npz model.session_features=tokens model.session_feature_norm=post"
                  [exp3cond]="data.session_features=conditions_v1.npz model.session_features=head"
                  [exp3clscond]="data.session_features=conditions_v1.npz model.session_features=head model.use_mean_pooling=false"
                  [exp4aug]="data.session_features=conditions_v1.npz model.session_features=head model.use_mean_pooling=false channel_drop.enabled=true channel_drop.channels=[\"T1\",\"T2\"] channel_drop.min_drop=1 channel_drop.max_drop=2 channel_drop.prob=0.2 channel_drop.consistency_weight=1.0 ${FAST}"
                  [exp4base]="channel_drop.enabled=true channel_drop.channels=[\"T1\",\"T2\"] channel_drop.min_drop=1 channel_drop.max_drop=2 channel_drop.prob=0.0 channel_drop.consistency_weight=0.0 ${FAST}"
                  [exp4ctrl]="data.session_features=conditions_v1.npz model.session_features=head model.use_mean_pooling=false channel_drop.enabled=true channel_drop.channels=[\"T1\",\"T2\"] channel_drop.min_drop=1 channel_drop.max_drop=2 channel_drop.prob=0.0 channel_drop.consistency_weight=0.0 ${FAST} ${SPOT}"
                  [exp1f]="${FEATS} model.session_features=only model.head_type=mlp model.mlp_head.hidden_dims=[256] model.mlp_head.drop=0.3 model.session_feature_drop=0.2")
declare -A TAG=([exp2]= [exp3]= [exp3b]= [exp3q]=qc800-0.5 [exp1t]=feat-tokens [exp1f]=feat-only [exp3cond]=cond-status-pathology
                [exp3clscond]=cls-cond-status-pathology [exp4aug]=cls-cond-augT1T2-fast [exp4ctrl]=cls-cond-ctrl-fast
                [exp4base]=base-fast)
mode=(--detach); [[ "${DRY_RUN:-0}" == "1" ]] && mode=(--dry_run)
for exp in "$@"; do
  cohort=${COHORT[$exp]}
  short=${cohort%_meta}; short=${short%_v1}
  name="tuh_${exp}_${short}_M1_ep${EPOCHS[$exp]}${TAG[$exp]:+_${TAG[$exp]}}"
  read -r -a extra <<< "${EXTRA[$exp]}"
  "${PY}" -m labram.runs.submit_sagemaker --config labram/configs/defaults/finetune_tuab_age.json "${mode[@]}" --set \
    sagemaker.enabled=true sagemaker.role="${ROLE}" sagemaker.instance_type="${INSTANCE[$exp]}" \
    sagemaker.input_mode=File sagemaker.volume_size_gb="${VOLUME[$exp]}" \
    sagemaker.wait=false sagemaker.stream_logs=false \
    sagemaker.job_name_prefix="labram-tuh-${exp}-${cohort//_/-}-ep${EPOCHS[$exp]}" \
    data.data_format=npy data.data_path="${S3}/${cohort}/" \
    trainer.epochs="${EPOCHS[$exp]}" optimizer.warmup_epochs="${WARMUP[$exp]}" \
    mixup.enabled=true mixup.alpha=0.4 mixup.sigma=0.0 \
    output.output_dir= output.log_dir= \
    clearml.enabled=true clearml.project_name=LaBraM/brain_age clearml.task_name="${name}" \
    "clearml.tags=[\"brain_age\",\"tuh_cohorts\",\"${short}\",\"M1\",\"ep${EPOCHS[$exp]}\",\"sagemaker\"${TAG[$exp]:+,\"${TAG[$exp]}\"}]" \
    "${extra[@]}" \
    2>&1 | grep -v -i warn | tail -n "${TAIL:-2}"
done
