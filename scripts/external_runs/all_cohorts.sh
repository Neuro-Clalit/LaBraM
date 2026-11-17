#!/usr/bin/env bash
# One page with every test cohort and every model (docs/all_cohorts_summary.md): accuracy per model,
# and every age-bias correction per cohort and subset (normal / abnormal / no report, controls / patients).
# Needs the predictions written by tdbrain.sh, dortmund.sh and open_cohorts.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
OUT=/data/runs/heedb/external_eval; E=/data/datasets/EEG-public; C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
R=/data/runs/heedb/mil_phased_cv; D=docs/data/external_eval

# the device-robust model on the in-domain test split (first 5 minutes, as the other models)
[[ -f $OUT/own_5min_e2lrobf.json ]] || $PY scripts/eval_age_by_cohort.py --data-path $C --splits test --stage-dir none \
  --eval-minutes 5 --pred-stats 53.52 19.29 --run E2LrobF=/data/runs/chbp_fix/e2l_robust_full_20261112_175034_869 \
  --out $OUT/own_5min_e2lrobf.json > $OUT/own_5min_e2lrobf.log 2>&1

# corrections fitted on normal-report validation predictions (each fold model's own fold at its best epoch)
V3=$(for k in 0 1 2 3; do printf "%s," $R/P3_L1/fold_$k/cv/fold_$k/analysis/val_bag_weights_epoch16.parquet; done | sed 's/,$//')
V4=$R/P4_L1/fold_1/cv/fold_1/analysis/val_bag_weights_epoch13.parquet,$R/P4_L1/fold_2/cv/fold_2/analysis/val_bag_weights_epoch15.parquet,$R/P4_L1/fold_4/cv/fold_4/analysis/val_bag_weights_epoch15.parquet
SETS=(--set temple:$C:$OUT/own_5min_models.json:x:temple --set harvard:$C:$OUT/own_5min_models.json:x:harvard
      --set tdbrain:$E/TDBRAIN/cohorts/tdbrain_v1:$OUT/tdbrain_all_models.json:x
      --set lemon:$E/LEMON/cohorts/lemon_v1:$OUT/lemon_all_models.json:x
      --set dortmund:$E/DORTMUND/cohorts/dortmund_v1:$OUT/dortmund_all_models.json:x
      --set srm:$E/SRM/cohorts/srm_v1:$OUT/srm_all_models.json:x
      --set chbp:$E/CHBP/cohorts/chbp_v1:$OUT/chbp_all_models.json:x
      --set nmt:$E/NMT/cohorts/nmt_v1:$OUT/nmt_all_models.json:x
      --set ahepa:$E/AHEPA/cohorts/ahepa_v1:$OUT/ahepa_all_models.json:x
      --set iowapd:$E/IOWAPD/cohorts/iowapd_v1:$OUT/iowapd_all_models.json:x
      --set telavivpd:$E/TELAVIVPD/cohorts/telavivpd_v1:$OUT/telavivpd_all_models.json:x
      --set mipdb:$E/MIPDB/cohorts/mipdb_v1:$OUT/mipdb_all_models.json:x)
$PY scripts/bias_correction_cohorts.py "${SETS[@]}" --models E2L P3ens P4ens --val E2L=$OUT/own_all_val_E2L.json \
  --val-windows P4ens=$V4 --val-windows P3ens=$V3 --val-meta $C/age_metadata.json \
  --out $D/bias_correction_all_cohorts.csv --by-group $D/bias_correction_by_group.csv > /dev/null
$PY scripts/all_cohorts_summary.py --out $D/all_cohorts_summary.csv --doc docs/all_cohorts_summary.md \
  --bias-by-group $D/bias_correction_by_group.csv | tail -1
