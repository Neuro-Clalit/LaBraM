#!/usr/bin/env bash
# CHBP domain-shift tests with frozen brain-age models (no CHBP labels are used
# by any transform; docs/chbp_domain_shift.md).
#   forward: age-matched Temple + Harvard normals made CHBP-like
#   inverse: CHBP normalized towards the training data
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
SRC=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
CH=/data/datasets/EEG-public/CHBP/cohorts/chbp_v1
D=/data/runs/heedb/domain_shift
C=$D/cohorts
mkdir -p $C $D/scores
step=${1:-all}

if [[ $step == features || $step == all ]]; then
  $PY scripts/domain_shift_analysis.py --cohort chbp=$CH --ref-cohort $SRC --out-dir $D --jobs 12 \
    --edf-root /data/datasets/EEG-public/CHBP/raw/bids_v1.0
fi

if [[ $step == prepare || $step == all ]]; then
  $PY scripts/make_transformed_cohort.py sample --src $SRC --match-ages $CH/age_metadata.json --n 150 \
    --out $D/ref_stems.txt
  $PY scripts/make_transformed_cohort.py psd --src $SRC --stems $D/ref_stems.txt --out $D/psd_ref.npz
  $PY scripts/make_transformed_cohort.py psd --src $CH --out $D/psd_chbp.npz
fi

# targets measured on the reference sample / CHBP (robust std 10.05 vs 6.22 uV,
# GSF -0.058 vs -0.185 log10 uV^2/Hz); 1/1.615 = 0.62
make() {  # name src transform [extra args]
  local name=$1 src=$2 tf=$3; shift 3
  [[ -f $C/$name/manifest.json ]] || $PY scripts/make_transformed_cohort.py make --src $src --out $C/$name \
    --transform "$tf" "$@"
}
if [[ $step == cohorts || $step == all ]]; then
  make ref_raw      $SRC "scale:1"                     --stems $D/ref_stems.txt
  make ref_x0.62    $SRC "scale:0.62"                  --stems $D/ref_stems.txt
  make ref_hp0.5    $SRC "bandlimit:0.5,99"            --stems $D/ref_stems.txt
  make ref_lp50     $SRC "bandlimit:0.01,50"           --stems $D/ref_stems.txt
  make ref_band     $SRC "bandlimit:0.5,50"            --stems $D/ref_stems.txt
  make ref_x0.62_band $SRC "scale:0.62+bandlimit:0.5,50" --stems $D/ref_stems.txt
  make ref_peramp   $SRC "peramp:10.05"                --stems $D/ref_stems.txt
  make ref_gsf      $SRC "gsf:-0.058"                  --stems $D/ref_stems.txt
  make chbp_x1.615  $CH  "scale:1.615"
  make chbp_peramp  $CH  "peramp:10.05"
  make chbp_gsf     $CH  "gsf:-0.058"
  make chbp_psdmatch $CH "psdmatch" --ref-psd $D/psd_ref.npz --src-psd $D/psd_chbp.npz
fi

RUNS=(
  --run E2L=/data/runs/heedb/e2l/finetune_20261104_215141_883
  --run L1=/data/runs/heedb/mil_local/L1_abmil_20261109_022309_458
  --run W4=/data/runs/heedb/mil_sagemaker/artifacts/w4/finetune
  --run W7=/data/runs/heedb/mil_sagemaker/artifacts/w7/finetune
)
if [[ $step == score || $step == all ]]; then
  for name in ref_raw ref_x0.62 ref_hp0.5 ref_lp50 ref_band ref_x0.62_band ref_peramp ref_gsf \
              chbp_x1.615 chbp_peramp chbp_gsf chbp_psdmatch; do
    [[ -f $D/scores/$name.json ]] && continue
    $PY scripts/eval_age_by_cohort.py --data-path $C/$name --splits test --pred-stats 53.52 19.29 \
      --stage-dir none "${RUNS[@]}" --out $D/scores/$name.json > $D/scores/$name.log 2>&1
    echo "$name: $(grep -E 'test all' $D/scores/$name.log | awk '{print $1, $5}' | tr '\n' ' ')"
  done
fi

# ---- section 10: closing the loop (docs/chbp_domain_shift.md) -----------------------------
if [[ $step == fix || $step == all ]]; then
  # label-free reference: 150 + 150 random adult validation recordings (not age-matched to CHBP)
  $PY scripts/make_transformed_cohort.py sample --src $SRC --n 150 --seed 1 --out $D/ref_stems_unmatched.txt
  $PY scripts/make_transformed_cohort.py psd --src $SRC --stems $D/ref_stems_unmatched.txt --out $D/psd_ref_unmatched_ch.npz
  cp $D/psd_ref_unmatched_ch.npz $D/psd_ref_unmatched.npz
  $PY - <<PYEOF
import pandas as pd
enc = pd.read_csv("$D/chbp_edf_encoding.csv").set_index("stem").enc
open("$D/stems_chbp64.txt", "w").write("\n".join(enc[enc == "12-bit"].index) + "\n")
open("$D/stems_chbp120.txt", "w").write("\n".join(enc[enc == "16-bit"].index) + "\n")
PYEOF
  for cap in 64 120; do
    $PY scripts/make_transformed_cohort.py psd --src $CH --stems $D/stems_chbp$cap.txt --out $D/psd_chbp${cap}_ch.npz
    cp $D/psd_chbp${cap}_ch.npz $D/psd_chbp$cap.npz
    make chbp${cap}_psdmatchU   $CH "psdmatch"   --stems $D/stems_chbp$cap.txt --ref-psd $D/psd_ref_unmatched.npz --src-psd $D/psd_chbp$cap.npz
    make chbp${cap}_psdmatchchU $CH "psdmatchch" --stems $D/stems_chbp$cap.txt --ref-psd $D/psd_ref_unmatched_ch.npz --src-psd $D/psd_chbp${cap}_ch.npz
  done
  make chbp_psdmatchU $CH "psdmatch" --ref-psd $D/psd_ref_unmatched.npz --src-psd $D/psd_chbp.npz
  # known truth: degrade, then repair with the same label-free method
  $PY scripts/make_transformed_cohort.py psd --src $C/ref_x0.62_band --out $D/psd_ref_x0.62_band.npz
  make ref_x0.62_band_psdmatchU $C/ref_x0.62_band "psdmatch" --ref-psd $D/psd_ref_unmatched.npz --src-psd $D/psd_ref_x0.62_band.npz
  make ref_lp50_fill $C/ref_lp50 "addnoise:48,75" --ref-psd $D/psd_ref.npz
  make ref_band_fill $C/ref_band "addnoise:48,75+addnoise:0.1,0.5" --ref-psd $D/psd_ref.npz
  for name in chbp64_psdmatchU chbp120_psdmatchU chbp_psdmatchU chbp64_psdmatchchU chbp120_psdmatchchU \
              ref_x0.62_band_psdmatchU ref_lp50_fill ref_band_fill; do
    [[ -f $D/scores/$name.json ]] && continue
    $PY scripts/eval_age_by_cohort.py --data-path $C/$name --splits test --pred-stats 53.52 19.29 \
      --stage-dir none "${RUNS[@]}" --out $D/scores/$name.json > $D/scores/$name.log 2>&1
  done
fi

if [[ $step == features || $step == all ]]; then
  # handcrafted features in the repo venv (kymatio needs scipy < 1.17)
  V=/data/code/LaBraM/.venv/bin/python; H=$D/hc; mkdir -p $H
  $V scripts/hc_domain_features.py sample --src $SRC --n 1200 --out $H/train_stems.txt
  $V scripts/hc_domain_features.py extract --cohort $SRC --stems $H/train_stems.txt --out $H/train.npz
  $V scripts/hc_domain_features.py extract --cohort $SRC --stems $D/ref_stems.txt --out $H/val_ref.npz
  $V scripts/hc_domain_features.py extract --cohort $CH --out $H/chbp.npz
  $V scripts/hc_domain_features.py extract --cohort $C/chbp_psdmatch --out $H/chbp_psdmatch.npz
  for cap in 64 120; do
    $V scripts/hc_domain_features.py extract --cohort $C/chbp${cap}_psdmatchU --out $H/chbp${cap}_psdmatchU.npz
  done
  $V - <<PYEOF
import numpy as np
a, b = np.load("$H/chbp64_psdmatchU.npz"), np.load("$H/chbp120_psdmatchU.npz")
np.savez_compressed("$H/chbp_psdmatchU.npz", **{k: np.concatenate([a[k], b[k]]) for k in a.files})
PYEOF
  $V scripts/hc_domain_features.py model --train $H/train.npz --test val=$H/val_ref.npz --test chbp=$H/chbp.npz \
    --test chbp_psdmatchU=$H/chbp_psdmatchU.npz --out $H/hc_models.csv
  $V scripts/hc_domain_features.py domain --ref $H/val_ref.npz --test chbp=$H/chbp.npz \
    --test chbp_psdmatch=$H/chbp_psdmatch.npz --out $H/domain_auc_partial.csv
fi

if [[ $step == report || $step == all ]]; then
  $PY scripts/domain_shift_figures.py --dir $D --out docs/figures/chbp \
    --tests docs/data/external_eval/chbp/domain_shift_tests.csv \
    --fixes docs/data/external_eval/chbp/chbp_fix_methods.csv
fi
