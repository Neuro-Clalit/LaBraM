#!/usr/bin/env bash
# Stage A of docs/fm_encoder_onboarding.md: frozen pretrained encoders (LaBraM-base, CBraMod,
# REVE-base) as feature extractors for brain age, scored with a ridge probe on the fold of the
# best model's CV split (P4_L1 fold 0), on the held-out test split and on the external cohorts.
# Nothing here trains an encoder.
#
#   scripts/external_runs/fm_probe.sh <step>
#   steps: sample | screen | cache | external | fit | all (cache + external + fit)
#
# sample    recordings to cache: 2,000 + 1,000 (screening) and 20,000 + 5,000 (probe) training /
#           validation recordings of fold 0, seeded, proportional per source dataset
# screen    input variants of each encoder on the screening sample (validation only)
# cache     the three encoders with their selected input on the probe sample + the test split
# external  the same encoders on LEMON, Dortmund, TDBRAIN, SRM and CHBP (test-only cohorts)
# fit       ridge probes, metric tables, predictions and paired comparisons
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
O=${OUT:-/data/runs/fm_probe}
C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
E=/data/datasets/EEG-public
RUN=/data/runs/heedb/mil_phased_cv/P4_L1/fold_0/cv/fold_0
CFG=$RUN/run_config.yaml                       # data, window selection and channels of P4
READOUT=${READOUT:-channel_mean}
step=${1:?step: sample | screen | cache | external | fit | all}
mkdir -p "$O"

# encoder -> config overrides (weights + input options)
LB=(finetune_checkpoint.finetune=./checkpoints/labram-base.pth model.use_mean_pooling=true)
CB=(model.model=cbramod_base_patch200 finetune_checkpoint.finetune=./checkpoints/cbramod-base.pth)
RV=(model.model=reve_base_patch200 finetune_checkpoint.finetune=./checkpoints/reve-base.safetensors)
HP='labram_plus.band_hz=[0.3,99.0]'
A=()
add() { local n=$1; shift; A+=(--pretrained "$n=$CFG"); for o in "$@"; do A+=(--override "$n:$o"); done; }
# the input each encoder was selected with (Table 6.2); extra overrides ("$@") apply to all three
selected() {
  A=()
  add LaBraM "${LB[@]}" "$@"
  add CBraMod "${CB[@]}" "$HP" model.input_clip_uv=100 "$@"
  add REVE "${RV[@]}" model.encoder.input_norm=zscore_window "$@"
}

if [[ $step == sample ]]; then
  S=(--cohort $C --cv-split $RUN/cv_split.json --fold 0)
  $PY scripts/fm_probe.py sample "${S[@]}" --n-train 2000 --n-val 1000 --out $O/recordings_screen.json
  $PY scripts/fm_probe.py sample "${S[@]}" --n-train 20000 --n-val 5000 --out $O/recordings.json
fi

if [[ $step == screen ]]; then
  A=()
  add LB_mean "${LB[@]}"
  add LB_nocar "${LB[@]}" labram_plus.enabled=false
  add CB_car "${CB[@]}"
  add CB_nocar "${CB[@]}" labram_plus.enabled=false
  add CB_car_hp "${CB[@]}" "$HP"
  add CB_car_clip "${CB[@]}" model.input_clip_uv=100
  add CB_car_hp_clip "${CB[@]}" "$HP" model.input_clip_uv=100
  add RV_car "${RV[@]}"
  add RV_nocar "${RV[@]}" labram_plus.enabled=false
  add RV_zw_car "${RV[@]}" model.encoder.input_norm=zscore_window
  add RV_zw_nocar "${RV[@]}" model.encoder.input_norm=zscore_window labram_plus.enabled=false
  add RV_zr "${RV[@]}" labram_plus.enabled=false channel_drop.enabled=false
  for readout in pooled channel_mean; do
    D=$O/screen_$readout
    $PY scripts/mil_cache.py --data-path $C --out-dir $D/main --splits val train --n-train-windows 10 \
      --recordings $O/recordings_screen.json --no-descriptors --readout $readout --batch-size 96 \
      --recording-zscore RV_zr "${A[@]}" > $D.log 2>&1
    $PY scripts/fm_probe.py fit --root $D --recordings $O/recordings_screen.json --out $D/probe \
      --encoders LB_mean LB_nocar CB_car CB_nocar CB_car_hp CB_car_clip CB_car_hp_clip \
                 RV_car RV_nocar RV_zw_car RV_zw_nocar RV_zr | tee $D.txt
  done
fi

if [[ $step == cache || $step == all ]]; then
  selected
  $PY scripts/mil_cache.py --data-path $C --out-dir $O/probe/main --splits val test train \
    --recordings $O/recordings.json --no-descriptors --readout $READOUT --batch-size 128 "${A[@]}" \
    > $O/probe_cache.log 2>&1
fi

if [[ $step == external || $step == all ]]; then
  # window settings of the external scoring (scripts/external_runs/*.sh): every window, and no
  # trims for the cohorts of short recordings
  NT=(data.trim_start_sec=0 data.trim_end_sec=0)
  for spec in "lemon LEMON" "chbp CHBP" "dortmund DORTMUND NT" "tdbrain TDBRAIN NT" "srm SRM NT"; do
    read -r name dir trim <<< "$spec"
    if [[ ${trim:-} == NT ]]; then selected data.eval_minutes=0 "${NT[@]}"; else selected data.eval_minutes=0; fi
    $PY scripts/mil_cache.py --data-path $E/$dir/cohorts/${name}_v1 --out-dir $O/probe/$name --splits test \
      --no-descriptors --readout $READOUT --batch-size 128 "${A[@]}" > $O/probe_cache_$name.log 2>&1
  done
fi

if [[ $step == fit || $step == all ]]; then
  $PY scripts/fm_probe.py fit --root $O/probe --encoders LaBraM CBraMod REVE --recordings $O/recordings.json \
    --cohorts lemon dortmund tdbrain srm chbp --out $O/probe/probe | tee $O/probe_fit.txt
fi
