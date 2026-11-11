#!/bin/bash
# Phased one-run MIL from labram-base (P_L1 / P_W4 / P_W7, scripts/mil_runs/submit_grid.sh)
# on all 5 CV folds, plus L1 with 4-window bags on fold 0 (P_L1_K4): 16 jobs placed
# within the account's SageMaker quotas (on-demand: 1 g5.4xlarge, 1 g5.8xlarge,
# 4 g5.2xlarge; spot: 10 g5.2xlarge). Fold 0 of every model (the verification
# fold) runs on-demand. Every job shares one fold partition
# (cross_validation.split_json) and the held-out test split.
#
#   scripts/mil_runs/submit_phased_cv.sh            # submit (detached)
#   scripts/mil_runs/submit_phased_cv.sh --dry_run  # preview
set -euo pipefail
cd "$(dirname "$0")/../.."
G=scripts/mil_runs/submit_grid.sh
ARGS=("$@")
# run <job> <fold> <instance> <loader workers> <spot>
run() { ONLY="$1" FOLD="$2" INSTANCE="$3" WORKERS="$4" SPOT="$5" "$G" "${ARGS[@]}" 2>&1 \
    | grep -E "^=== |job name|Submitted|\[dry-run\] labram|ResourceLimit|An error occurred|Traceback" | cut -c1-160 || true; }
# on-demand
run P_L1    0 ml.g5.4xlarge 32 false
run P_L1_K4 0 ml.g5.8xlarge 32 false
run P_W4    0 ml.g5.2xlarge 10 false
run P_W7    0 ml.g5.2xlarge 10 false
run P_L1    1 ml.g5.2xlarge 10 false
run P_W4    1 ml.g5.2xlarge 10 false
# spot
run P_W7 1 ml.g5.2xlarge 10 true
for fold in 2 3 4; do
  for job in P_L1 P_W4 P_W7; do
    run "$job" "$fold" ml.g5.2xlarge 10 true
  done
done
