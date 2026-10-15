---
name: analyze-experiments
description: How to analyze and compare LaBraM fine-tune experiments (brain age / TUH cohorts) — not only the best score, but leakage-free shared evaluation sets, run-to-run noise, overfitting and memorization diagnostics, learning-curve headroom (train longer or not), val/test agreement, augmentation robustness and run-time cost. Load before reporting, comparing or deciding on next runs from experiment results.
---

# Analyzing fine-tune experiments

A result is only useful if it says *why* a run is better and *what to do next*.
For every run (or set of runs) report the five blocks below, in this order.

## 1. Compare on the same, leakage-free sets

* Score every model on **shared sets** with `scripts/eval_age_by_cohort.py`
  (`--train-data-path` = the model's own training cohort, so predictions are
  de-normalized with its own target stats), then break them down with
  `scripts/age_breakdown.py` (source TUAB/TUEG x cohort normal/abnormal; n,
  MAE, RMSE, R²). Append the rows to `docs/data/tuh_cohort_eval_v1.csv`.
* Shared sets: **TUAB test** (274 rec, the benchmark), **TUAB val** (556),
  and **TUEG val** = the TUEG part of `all_both_v2` val (2,325 rec). A model
  may only be scored on a set none of its training patients is in: cohorts v2+
  share one TUEG patient split (`cohorts/tueg_patient_split_v2.json`); v1
  all-cases models (Exp3, 3b, 3c, 3q) leak into TUEG val — never report them
  there (mark "leaky").
* A run's *own* val (in its log) is for its learning curve, not for comparing
  runs on different cohorts.
* Noise: one run's TUAB-test MAE moves ~0.2-0.5 y with the seed / split
  (Exp3 v1 7.26 vs Exp3 v2 7.42, same recipe); TUAB abnormal (125 rec) moves
  most. Call a difference real only if it is larger than that, or consistent
  across all shared sets; otherwise "tie". Resolve close calls with CV.

## 2. Overfitting and memorization

Run `scripts/overfit_report.py --run NAME=<log.txt>:<warmup> ...`
(`labram/eval/overfit.py`; `--per-epoch` for the curves) and read:

| column | meaning | warning sign |
|---|---|---|
| `gap`, `gap%` | val − train MAE at the last epoch, and as % of val | `gap% > 30`: **memorizing** the training recordings (Exp1t 76%, Exp1f 62%, Exp1b/3b 32-36%; healthy runs 9-20%) |
| `gap_slope` | OLS slope of the gap over the last k epochs | > 0 while val stalls: overfitting is starting |
| `overfit_ep`, `streak` | epochs where train improved and val got worse by > 0.02 y; trailing run of them | `streak >= 2`: stop earlier / regularize |
| `d_val`, `acc_val` | last val improvement and its change (acceleration) | `d_val ≈ 0` with `acc_val < 0`: converged |
| `d_train` | last train improvement | large `d_train` with `d_val ≈ 0`: capacity spent on memorizing |

Train metrics are running averages under mixup / augmentation, so early gaps
can be small or negative — read the **trend**, and compare runs with the same
augmentation. Per-recording inputs that are constant across a recording's
windows (session features, condition vectors) act as fingerprints: expect a
large gap and check val, not train.

## 3. Headroom: would more epochs help?

* `still improving` = the last epoch is the best by > 0.02 y.
* `headroom`: with >= 4 improving post-warm-up epochs, the remaining val gain
  if improvements keep shrinking at ratio r: `d_last * r / (1 - r)`. NaN = too
  few epochs to say ("train longer" if still improving).
* Verdicts: *still improving / not bending* → longer schedule (more epochs,
  longer warm-up; keep `trainer.early_stopping=true`); *slowing, small
  headroom* → at most marginal; *plateau* → converged, change the recipe
  instead; *overfitting / memorizing* → regularize (mixup, dropout, fewer
  epochs, drop fingerprint features) — never just train longer.
* Check the LR: a val gain in the last epochs of a cosine schedule is partly
  annealing; a longer schedule moves it, it does not add it.

## 4. Selection and val/test agreement

* `test@best`, `regret` = test MAE at the best-val epoch minus the best test
  MAE over epochs; `val~test rho` = Spearman of the two curves. rho >= 0.8
  and regret ~0: val is a good proxy, trust model selection. Low rho / large
  regret (Exp1 0.20 / 0.15): the val set does not represent the test set
  (e.g. normal-only val vs mixed test) — select on a more similar val.
* Never pick a model or hyperparameter on test.

## 5. Robustness, cost, and the decision

* Augmentation runs: compare full vs. perturbed metrics (e.g.
  `val_channel_drop_window_mae`, consistency) against a **control** that differs
  only by the augmentation. Without the control, report the effect as
  unidentified.
* Run time: epoch time, samples/s, total h, instance and $ (`epoch_time_sec`,
  `train_samples_per_sec` in log.txt; SageMaker `TrainingTimeInSeconds`,
  `BillableTimeInSeconds`). Compare only on the same GPU type; note CPU-count
  differences (ml.g5.16xlarge vs 4xlarge).
* End with a decision per run: keep / drop / rerun (what change, expected
  gain from the headroom, cost).
