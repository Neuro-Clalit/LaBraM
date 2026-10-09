# Brain-age model: improvement plan

Short plan for the next steps on TUAB_AGE fine-tuning (status as of 2026-10-04,
branch `fix-overfit`). Background: [`age_regression.md`](age_regression.md).

## Where we are

| Run | Val MAE | Test MAE | Note |
|---|---|---|---|
| Aug run, from `labram-base` (best epoch 2) | **8.31** | 9.04 | normal + abnormal recordings |
| Sep run `3700b756…` (auto-resumed an overfit checkpoint) | 9.32 | 10.92 | not a valid experiment |
| Published best, TUAB normal recordings (Gemein et al. 2024) | — | 6.60 | normal only, so not directly comparable |

**Main problem: memorization.** Train MAE reaches 0.36 years, while val stays
around 9 years from epoch 2 onwards. The model learns to recognize recordings
instead of age.

**New default:** scenario D (common average reference, 15 epochs, lr 1e-4) is
now `finetune_tuab_age.json`: val 7.77 / test 8.30 (case MAE). Technical
description: [`age_training_scenario_D.md`](age_training_scenario_D.md).

**Fixed on `fix-overfit`:**
- New runs start fresh: `output.auto_resume=false` by default, plus a
  per-run timestamp on `output_dir`/`log_dir`.
- Resuming continues after the saved epoch instead of restarting at 1.
- New data options: `data.case_filter`, `trim_start_sec`/`trim_end_sec`,
  `window_sec`, `eval_minutes`.

## Running now (SageMaker spot, shared split)

`scripts/submit_age_experiments.sh`; ClearML project `LaBraM/brain_age`, tag
`age_ablation_2026-10`. Common to all: 15 epochs, lr 1e-4, 2 warmup epochs,
first and last minute trimmed.

| Exp | Change | Question it answers |
|---|---|---|
| A | whole-recording eval | Baseline with the new schedule |
| B | A + `eval_minutes=5` | Cost of evaluating on only 5 minutes |
| C | B + `window_sec=30` | Does longer context help? |
| C2 | B + layer_decay 0.5, drop_path 0.2, drop 0.1, wd 0.1, clip 1.0 | Does stronger regularization reduce overfitting? |
| D | B + common average reference (no per-patch z-score) | Montage normalization |
| D2 | B + L1 loss | Loss choice (Gemein uses L1) |

**How to read the results:** compare on **val MAE (per recording)**; use
test only for the final pick. Carry forward every change that beats B by more
than about 0.3 years and keeps the train–val gap smaller.

## Next steps (ranked by expected impact)

1. **Random crops in training.** Overlapping crops at random offsets every
   epoch (Gemein's cropped decoding). Today every epoch sees identical windows,
   which makes recordings easy to memorize. *Needs code.*
2. **Channel-dropout augmentation and ±800 µV amplitude clipping.** Gemein's
   best augmentation; clipping also stops artifact windows from causing
   gradient spikes. *Needs code.*
3. **Train on normal recordings only** (`data.case_filter=normal`). Gemein
   found this better, and it makes results comparable with the 6.60 benchmark.
   Abnormal recordings then become a separate brain-age-gap analysis. *Config
   only (re-upload the labelled `age_metadata.json` to S3 first).*
4. **Evaluate EMA weights.** `optimizer.model_ema` is updated and saved but
   never evaluated, so turning it on does nothing for model selection today.
   *Needs code.*
5. **Train the head first, then fine-tune (LP-FT), or freeze the lower
   blocks.** Keeps the pretrained features from being distorted. *Needs code.*
6. **Reduce regression to the mean** (bias slope about −0.5): soft labels over
   age bins (SFCN-style KL loss) or label-distribution smoothing. Report the
   quadratic bias correction fitted on val. *Needs code.*
7. **Ensemble 5 seeds or 5 CV folds**, subject-disjoint
   (`labram.runs.finetune_cv`); report mean ± std. *Mostly config.*
8. **More data: TUH EEG corpus (TUEG) ages** (about 10× TUAB), excluding TUAB
   eval subjects. The most reliable fix for subject-identity memorization (see
   *The Identity Trap*). *Needs a preprocessing pass.*

## Infrastructure to-dos

- Spot capacity for `ml.g5.2xlarge` is scarce (placement score 1/10 for six
  instances). Fallback: one job at a time on spot `ml.g5.4xlarge` (score 9/10),
  or raise the on-demand quota `L-2D6DEB3C` (currently 1).
- Sync spot checkpoints (`checkpoint_s3_uri`) so an interrupted job resumes
  instead of restarting.
- Fix `scripts/check_sagemaker_capacity.py`, which crashes on the AWS rate
  limit when listing quotas.
- Apply the `data.*` selection options in `notebooks/finetune_evaluation.ipynb`.

## References

- Gemein et al. 2024, *Brain age revisited* (Imaging Neuroscience) — TCN, normal-only training, cropped decoding, channel dropout, L1 loss, 5-seed ensembles, quadratic bias correction; 6.60 MAE (normal) / 12.85 (abnormal).
- Engemann et al. 2022, *A reusable benchmark of brain-age prediction from M/EEG* (NeuroImage) — earlier best on TUAB, 7.75 MAE.
- *The Identity Trap in EEG Foundation Models* (arXiv 2606.06647) — LaBraM/CBraMod/REVE features are dominated by subject identity.
- Peng et al. 2021 (SFCN, soft-label brain age); Yang et al. 2021 (label-distribution smoothing); Kumar et al. 2022 (LP-FT).
