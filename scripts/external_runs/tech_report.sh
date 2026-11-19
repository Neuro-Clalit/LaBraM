#!/usr/bin/env bash
# Technical report (docs/tech_report/): the result tables and figures over the seven report cohorts,
# then LaTeX + Markdown + PDF from the one source. Needs the predictions and saved windows written by
# tdbrain.sh, dortmund.sh and open_cohorts.sh.
#
# Report cohorts: Temple and Harvard (held-out test split of tuh_heedb_v1, first 5 minutes), and the
# zero-shot cohorts TDBRAIN, LEMON, Dortmund, SRM and CHBP (every window). Left out of the report
# (onboard-eeg-dataset skill, "Which cohorts may appear in a final report"): MIPDB adults (31, too
# few), NMT (band-limited release), AHEPA / Iowa / Tel Aviv PD (controls offset; contrasts only, in
# Section 6.6 from their gap_by_group.csv), HBN (paediatric study).
#
#   step tables   : cohort table, rejection scores, bias corrections, window confidence -> CSVs in
#                   docs/data/external_eval/tech_report/, figures next to the report
#   step protocol : fixed evaluation window and validation-fixed rejection thresholds (Section 10.4)
#   step build    : LaTeX + Markdown + PDF (scripts/build_tech_report.py)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-/opt/pytorch/bin/python}
OUT=/data/runs/heedb/external_eval; E=/data/datasets/EEG-public; C=/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
R=/data/runs/heedb/mil_phased_cv; T=docs/tech_report; D=docs/data/external_eval/tech_report
step=${1:-all}
mkdir -p $D

# validation windows of the fold models (each on its own validation fold, at its best epoch): the
# learned rejection score and the bias corrections are fitted on these only
V3=$(for k in 0 1 2 3; do printf "%s," $R/P3_L1/fold_$k/cv/fold_$k/analysis/val_bag_weights_epoch16.parquet; done | sed 's/,$//')
V4=$R/P4_L1/fold_1/cv/fold_1/analysis/val_bag_weights_epoch13.parquet,$R/P4_L1/fold_2/cv/fold_2/analysis/val_bag_weights_epoch15.parquet,$R/P4_L1/fold_4/cv/fold_4/analysis/val_bag_weights_epoch15.parquet
SETS=(--set temple:$C:$OUT/own_5min_models.json:$OUT/windows/own_5min_models:temple
      --set harvard:$C:$OUT/own_5min_models.json:$OUT/windows/own_5min_models:harvard
      --set tdbrain:$E/TDBRAIN/cohorts/tdbrain_v1:$OUT/tdbrain_all_models.json:$OUT/windows/tdbrain_all
      --set lemon:$E/LEMON/cohorts/lemon_v1:$OUT/lemon_all_models.json:$OUT/windows/lemon_all_models
      --set dortmund:$E/DORTMUND/cohorts/dortmund_v1:$OUT/dortmund_all_models.json:$OUT/windows/dortmund_all
      --set srm:$E/SRM/cohorts/srm_v1:$OUT/srm_all_models.json:$OUT/windows/srm_all
      --set chbp:$E/CHBP/cohorts/chbp_v1:$OUT/chbp_all_models.json:$OUT/windows/chbp_all_models)

if [[ $step == tables || $step == all ]]; then
  # Tables 5, 6, 18-24: all / healthy / pathological, without and with the three rejection scores
  $PY scripts/external_cohort_table.py "${SETS[@]}" --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --models E2L P3ens P4ens --out $D/external_cohort_table.csv
  $PY scripts/external_cohort_bars.py --table $D/external_cohort_table.csv --reject-score --out $T/fig_cohort_mae.png
  # Table 8, Figure 4
  $PY scripts/rejection_scores.py "${SETS[@]}" --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --models P4ens --fit-on temple harvard --out $D/rejection_scores.csv --figure $T/fig_rejection_curves.png
  # Tables 9, 14-16, Figures 2 and 7
  $PY scripts/bias_correction_cohorts.py "${SETS[@]}" --models E2L P3ens P4ens \
    --val E2L=$OUT/own_all_val_E2L.json --val-windows P4ens=$V4 --val-windows P3ens=$V3 \
    --val-meta $C/age_metadata.json --out $D/bias_correction.csv --by-group $D/bias_correction_by_group.csv --fig-dir $T
  # Figures 5 and 6
  $PY scripts/window_confidence_figures.py "${SETS[@]}" --model P4ens --out $D/window_confidence.csv --fig-dir $T
  rm -f $T/recording_error_vs_confidence.png                        # not used in the report
  $PY scripts/chbp_variant_bars.py --out $T/fig_chbp_fixes.png      # CHBP corrections (docs/chbp_domain_shift.md)
  # Figures 1 and 2: the pipeline, and attention pooling on one LEMON recording (open licence; scored after
  # the 60 s trims, so slot i is window i + 6 of the file)
  $PY scripts/tech_report_figures.py --out-dir $T --windows $OUT/windows/lemon_all_models \
    --cohort $E/LEMON/cohorts/lemon_v1 --recording sub-032490_lemon-rs --trim-windows 6
fi

if [[ $step == protocol || $step == all ]]; then
  # in-domain every-window predictions exist for E2L only (own_all_w); the ensembles were saved for 5 minutes
  EVERY=(--set temple_every:$C:$OUT/own_all_w.json:$OUT/windows/own_all_w:temple
         --set harvard_every:$C:$OUT/own_all_w.json:$OUT/windows/own_all_w:harvard)
  # Section 10.4: every cohort on the first 5 minutes and on every window (exact re-pooling of the saved
  # windows), and rejection with thresholds fixed on validation instead of the cohort's own quantile
  $PY scripts/report_protocol.py "${SETS[@]}" "${EVERY[@]}" --models E2L P3ens P4ens --val-windows P4ens=$V4 \
    --val-windows P3ens=$V3 --out-dir $D
fi

if [[ $step == build || $step == all ]]; then
  $PY scripts/build_tech_report.py
fi
