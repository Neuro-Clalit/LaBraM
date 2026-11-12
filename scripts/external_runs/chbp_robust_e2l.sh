#!/usr/bin/env bash
# Device-robust E2L (docs/chbp_domain_shift.md section 10.2): E2L fine-tuned on Temple + Harvard
# (tuh_heedb_v1 train split; CHBP is never used) with model-owned inputs band 1-45 Hz +
# per-window amplitude normalization, then scored on CHBP, LEMON and the forward-test samples.
#   scripts/external_runs/chbp_robust_e2l.sh train       # 4 epochs x 1 min per recording (~1.5 h, A10G)
#   scripts/external_runs/chbp_robust_e2l.sh continue    # +6 epochs x 2 min per recording (~5-6 h)
#   scripts/external_runs/chbp_robust_e2l.sh score       # scores of the latest run of each
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
OUT=/data/runs/chbp_fix
step=${1:-train}
common=(data.stage_dir=off data.num_workers=6 evaluation.train_eval_recordings=0
        output.auto_resume=false output.append_timestamp=true output.save_ckpt_freq=1
        clearml.upload_model_artifact=false)
if [[ $step == train ]]; then
  $PY -m labram.runs.run_finetune --config $E2L/run_config.yaml --set \
    labram_plus.band_hz=[1.0,45.0] labram_plus.amplitude_norm=window labram_plus.amplitude_target=0.1 \
    finetune_checkpoint.finetune=$E2L/checkpoint-best.pth \
    optimizer.lr=0.0001 optimizer.warmup_epochs=1 optimizer.min_lr=1e-06 trainer.epochs=4 \
    data.train_minutes_per_epoch=1 "${common[@]}" \
    output.output_dir=$OUT/e2l_band45_amp output.log_dir=$OUT/e2l_band45_amp/log \
    clearml.task_name=heedb_E2L_band1-45_ampnorm_ft4 \
    'clearml.tags=["heedb","E2L","chbp-fix","band1-45","ampnorm","local"]'
elif [[ $step == continue ]]; then
  RUN=$(ls -d $OUT/e2l_band45_amp_2* | tail -1)
  $PY -m labram.runs.run_finetune --config $RUN/run_config.yaml --set \
    finetune_checkpoint.finetune=$RUN/checkpoint-best.pth \
    optimizer.lr=0.0001 optimizer.warmup_epochs=0 optimizer.min_lr=1e-06 trainer.epochs=6 \
    data.train_minutes_per_epoch=2 "${common[@]}" \
    output.output_dir=$OUT/e2l_band45_amp_cont output.log_dir=$OUT/e2l_band45_amp_cont/log \
    clearml.task_name=heedb_E2L_band1-45_ampnorm_ft4+6 \
    'clearml.tags=["heedb","E2L","chbp-fix","band1-45","ampnorm","local","continuation"]'
elif [[ $step == score ]]; then
  D=/data/runs/heedb/domain_shift; C=$D/cohorts
  for pat in e2l_band45_amp_2 e2l_band45_amp_cont_2; do
    RUN=$(ls -d $OUT/${pat}* 2>/dev/null | tail -1) || continue
    [[ -f $RUN/checkpoint-best.pth ]] || continue
    S=$OUT/scores/$(basename $RUN); mkdir -p $S
    for set in chbp_raw:/data/datasets/EEG-public/CHBP/cohorts/chbp_v1 chbp64_psdmatchU:$C/chbp64_psdmatchU \
               chbp120_psdmatchU:$C/chbp120_psdmatchU lemon:/data/datasets/EEG-public/LEMON/cohorts/lemon_v1 \
               ref_raw:$C/ref_raw ref_x0.62_band:$C/ref_x0.62_band; do
      name=${set%%:*}; data=${set#*:}
      [[ -f $S/$name.json ]] || $PY scripts/eval_age_by_cohort.py --data-path $data --splits test \
        --pred-stats 53.52 19.29 --stage-dir none --run E2L-robust=$RUN --out $S/$name.json > $S/$name.log 2>&1
      echo "$(basename $RUN) $name: $(grep -E 'test all' $S/$name.log | awk '{print $5, $6}')"
    done
  done
fi
