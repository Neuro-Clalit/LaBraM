# Brain-age anti-overfitting experiments: registry

This is the registry of every significant TUAB_AGE fine-tuning experiment run
between 2026-10-04 and 2026-10-10 on branch `fix-overfit` to fight
memorization. For each run it gives the ClearML experiment link, the results,
and the config keys that differ from its baseline. All runs are in ClearML
project `LaBraM/brain_age` (tags `age_ablation_2026-10`,
`age_ablation_D_2026-10`, `age_scratch_2026-10`). The task, the data and the
metric code are described in [`age_regression.md`](age_regression.md), and the
default config in [`age_training_scenario_D.md`](age_training_scenario_D.md).
Section 15 lists where each number comes from.

## 1. The problem: memorization

Every MAE below is a **recording-level** age error in years. The model
predicts each 10-s window, and the windows of recording $r$ are averaged
(`evaluation.agg_windows=mean`, Eq. (7.1) of `age_regression.md`) over its
evaluation windows $W_r$: the first `data.eval_minutes` = 5 minutes after the
60-s trims. $\hat y_i$ is the de-normalized window prediction:

$$
\hat y_r=\frac{1}{\lvert W_r\rvert}\sum_{i\in W_r}\hat y_i .
\tag{1.1}
$$

The recording MAE over the $R$ recordings of a split is Eq. (1.2), with the
recording prediction $\hat y_r$ of Eq. (1.1) and true age $y_r$ (years). This is `val_mae` / `test_mae`, Eq. (6.2) of
`age_regression.md`, computed by
`labram/utils/regression_metrics.py::regression_metrics_fn`:

$$
\operatorname{MAE}=\frac1R\sum_{r=1}^{R}\lvert \hat y_r-y_r\rvert .
\tag{1.2}
$$

The cohort MAE is Eq. (1.2) restricted to the $R_c$ recordings with TUAB label
$c\in\lbrace\text{normal},\text{abnormal}\rbrace$ (`test_normal_mae`,
`test_abnormal_mae`). The test split has $R_\text{normal}=149$ and
$R_\text{abnormal}=125$ recordings, and val has 291 and 265:

$$
\operatorname{MAE}_c=\frac1{R_c}\sum_{r\in\mathcal R_c}\lvert \hat y_r-y_r\rvert .
\tag{1.3}
$$

The bias slope (`test_age_bias_slope`, Eq. (6.7) of `age_regression.md`) is the
OLS slope of the residual $e_r=\hat y_r-y_r$ on the true age, using population
moments over the test recordings. $\beta=0$ means no age-dependent bias, and
$\beta=-1$ means a prediction that does not depend on age at all:

$$
\beta=\frac{\operatorname{Cov}(y,e)}{\operatorname{Var}(y)} .
\tag{1.4}
$$

Model selection keeps the epoch $e^{\star}$ (0-based, as in `log.txt`) with the
lowest val MAE, Eq. (1.5). With `evaluation.use_ema=true`, the EMA weights are
both scored and selected. Every val/test number in this registry is taken at
$e^{\star}$:

$$
e^{\star}=\operatorname*{arg\,min}_{e}\ \operatorname{MAE}_\text{val}(e) .
\tag{1.5}
$$

The train MAE of epoch $e$ is Eq. (1.2) over the training recordings. It uses
the predictions collected during that epoch in train mode, while the weights
are still changing (`train_mae`, `labram/train/train_finetune.py`). Under mixup
(marked †), no predictions are collected. `train_mae` is then the running
per-batch **window** MAE against the **mixed** target of Eq. (12.1) in
`age_regression.md`, so it cannot be compared with the unmarked values. The
generalization gap is Eq. (1.6):

$$
\Delta(e)=\operatorname{MAE}_\text{val}(e)-\operatorname{MAE}_\text{train}(e) .
\tag{1.6}
$$

Table 1.1 shows the problem on the baseline, scenario D. Val MAE is best at
epoch 1 (7.77), but training continues until the train MAE reaches 1.23 years.
The gap of Eq. (1.6) grows from 0.53 to 6.66 years, while val drifts up to 7.89
and test to 8.91. The model learns to recognize recordings instead of ages.
The pre-ablation run (lr 2e-4–5e-4, 25–50 epochs; Table 14.2) went further: its
train MAE reached 0.36 years while val stayed at 9.36 and test at 11.10. For
scale, a constant prediction scores about 12.8 on val and 14.9 on test
(`age_training_scenario_D.md`, section 1). The collapsed runs in sections 4
and 5 land exactly there, at 12.83 / 14.90 with $\beta=-1.00$.

### Table 1.1. Scenario D: train, val and test MAE per epoch

Run D ([`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log)). Columns: train MAE, Eq. (1.2) on the training recordings;
val and test MAE, Eq. (1.2); gap $\Delta$, Eq. (1.6). All in years.

| epoch $e$ | train MAE | val MAE | test MAE | $\Delta(e)$ |
|---|---|---|---|---|
| 0 | 10.68 | 8.49 | 8.93 | −2.19 |
| **1** ($e^{\star}$) | 7.24 | **7.77** | 8.30 | 0.53 |
| 3 | 4.14 | 7.77 | 8.31 | 3.63 |
| 7 | 1.95 | 8.01 | 8.93 | 6.06 |
| 14 (last) | 1.23 | 7.89 | 8.91 | 6.66 |

## 2. Baselines, shared settings and how to read the tables

All runs share the following:

* **Split.** The subject-disjoint split from `processed/age_split.json`. The
  npy-era SageMaker runs (D-npy variants, E\*, DX3) also pin
  `data.split_json` to D-npy's recorded `data_split.json`, which holds the same
  assignment.
* **Trims and eval budget.** 60-s trims at both ends, a 5-minute val/test
  budget per recording, and both normal and abnormal recordings.
* **Start.** `labram-base.pth`, except in section 11.

There are four baselines; each family section names the one it is compared to.
Table 2.1 gives D's key values and Table 2.2 the baseline results.

* **D** is scenario D, `labram/configs/defaults/finetune_tuab_age.json`. Its
  key values are in Table 2.1.
* **E2** is D on the short schedule: `trainer.epochs=4` and
  `optimizer.warmup_epochs=1`. It is the baseline for every run named F, G,
  H, M or B1/B2.
* **B** is D without the common average reference (`labram_plus.enabled=false`).
  It is the baseline for the first ablation wave (A, C, C2, D2).
* **C2** is the baseline for the partial fine-tunes in section 5.

Each family's config-diff table lists only the keys that differ from that
baseline; every other key is as in Table 2.1.

### Table 2.1. Baseline D: key configuration

The keys that the experiments change, with their scenario-D values
(`finetune_tuab_age.json`; the full file is described in
`age_training_scenario_D.md`).

| Key | D value | Key | D value |
|---|---|---|---|
| `trainer.epochs` | 15 | `model.drop` | 0.0 |
| `optimizer.warmup_epochs` | 2 | `model.drop_path` | 0.1 |
| `optimizer.lr` | 1e-4 | `model.init_scale` | 0.001 |
| `optimizer.layer_decay` | 0.65 | `model.head_type` | `linear` (mean patch token) |
| `optimizer.weight_decay` | 0.05 | `model.use_mean_pooling` / `concat_cls_token` | true / false |
| `optimizer.clip_grad` | none | `model.depth` | null (12 blocks) |
| `optimizer.opt_betas` | null (timm default) | `mixup.enabled` | false |
| `optimizer.model_ema` / `evaluation.use_ema` | false / false | `loss.regression_loss` / `huber_delta` | `huber` / 1.0 |
| `trainer.batch_size` × `update_freq` | 64 × 1 | `loss.balance` | `none` |
| `data.window_sec` | 10 | `model.codebook_reg.enabled` | false |
| `data.random_crop` | false | `model.lora.enabled` | false |
| `labram_plus.enabled` | true (CAR only, `z_score_patches=false`) | `finetune_checkpoint.finetune` | `./checkpoints/labram-base.pth` |

**Results-table columns** (Tables 2.2–11.2). Run name and ClearML link; val
and test MAE, Eq. (1.2); test normal / abnormal MAE, Eq. (1.3); test
$\beta$, Eq. (1.4); $e^{\star}$, Eq. (1.5), out of the epochs run; train
MAE at $e^{\star}$, Eq. (1.2) on train († = mixed-target window MAE, see
section 1). The `src` column codes are: C = ClearML (the run's `summary`
single values, or its per-epoch `err` scalars at $e^{\star}$); L = the local
`checkpoints/<run>/log.txt`; W = the SageMaker job's CloudWatch log;
C+L = ClearML for val/test plus the local log for train MAE or for epochs
ClearML did not receive. A "–" in the cohort column means the run predates the
normal/abnormal metrics (commit `62b159d`, 2026-10-07).

The verdicts compare a run with its baseline through Eq. (2.1), in years
(negative = better):

$$
\delta_\text{split}=\operatorname{MAE}_\text{split}(\text{run})-\operatorname{MAE}_\text{split}(\text{baseline}),
\qquad \text{split}\in\lbrace\text{val},\text{test}\rbrace .
\tag{2.1}
$$

**Noise floor.** Re-running D gives the same numbers: D-npy matches the
pickle run exactly, and the local D-cohorts run is within 0.01 years. Two seeds
of the from-scratch S2 config differ by 0.05 (val) and 0.14 (test), as Table
11.2 shows. A $\lvert\delta\rvert$ below about 0.15 years is therefore not
treated as a real effect.

### Table 2.2. Baselines and reproductions

D, its two reproductions, the short-schedule baseline E2, and the pre-D
evaluation-protocol runs A (whole-recording eval) and B (5-minute eval).
Columns as listed above.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| D-npy | [`5b9138de`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5b9138de74c542fe8c822fb954d87df6/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| D-cohorts | [`a39059ad`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a39059ad71d44fe1a4dbd0561c08ae4e/output/log) | 7.77 | 8.31 | 6.57 / 10.38 | −0.40 | 1 / 15 | 7.24 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| B | [`5261e2a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5261e2a18dfc45d1ae3fdd15991bdacd/output/log) | 8.40 | 8.91 | – | −0.43 | 2 / 6 | 5.42 | C |
| A | [`08833720`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/088337201a0b435d9e718e3e59dfa565/output/log) | 8.29 | 9.03 | – | −0.46 | 2 / 12 | 5.42 | C |

A and B were stopped by hand after 12 and 6 of 15 epochs; both selected
epoch 2. Adding CAR (B → D) gives $\delta$ = −0.63 (val) and −0.61 (test). It
is the only change in the whole study with a clear gain on both splits, but it
is a montage fix, not an anti-overfitting measure.

## 3. Schedule length and early stopping

**Hypothesis.** If the best epoch comes early, a shorter schedule (or a
longer one with averaging) should reach the same or a better optimum without
the memorization phase.

Table 3.1 lists the config diff and Table 3.2 the results.

### Table 3.1. Schedule runs: config diff

Keys that differ from D.

| Run | Key | Value | D value |
|---|---|---|---|
| E2 | `trainer.epochs`, `optimizer.warmup_epochs` | 4, 1 | 15, 2 |
| M12 | E2 + `mixup.enabled`, `mixup.alpha` | true, 0.4 | false |
| M12 | `optimizer.model_ema`, `model_ema_decay`, `evaluation.use_ema` | true, 0.9995, true | false, 0.9999, false |
| L1 | M12 with `trainer.epochs`, `optimizer.warmup_epochs` | 12, 2 | 15, 2 |

### Table 3.2. Schedule runs: results

Columns as in section 2.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| M12 | [`d2a9eb9c`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d2a9eb9c61474023b3dfae8d049a3b0d/output/log) | 7.48 | 8.45 | 6.57 / 10.69 | −0.42 | 3 / 4 | 6.38† | C |
| L1 | [`32c097a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/32c097a17ff84c98b3b9cfce5281a2da/output/log) | 7.51 | 8.47 | 6.65 / 10.63 | −0.42 | 3 / 12 | 6.54† | C |

**Verdict.** The 4-epoch schedule matches 15 epochs: E2 vs. D gives
$\delta$ = −0.09 / −0.05. Extending mixup + EMA from 4 to 12 epochs (L1 vs. M12)
gives +0.03 / +0.02, because L1 still selects epoch 3. Early stopping already
caps the damage, and the extra epochs only memorize. The short schedule is
about 4× cheaper.

## 4. Classic regularizers: weight decay, dropout, drop path, layer decay, head init

**Hypothesis.** Stronger weight-space regularization keeps the encoder near
its pretrained solution and slows the drop in train MAE. C2 and DX3 bundle
five regularizers; F1–F4 and G1 then test each one alone, and H1/H2 probe the
opposite direction (weaker layer decay, so more change in the lower blocks).
Table 4.1 lists the config diffs and Table 4.2 the results.

### Table 4.1. Classic-regularizer runs: config diff

Keys that differ from the baseline in the second column. `model.drop` is the
dropout after the token embedding (`pos_drop`) and inside every block's
attention projection and MLP (`labram/models/neural_transformer.py`).
`model.init_scale` scales the initial head weights.

| Run | Baseline | Key | Value | Baseline value |
|---|---|---|---|---|
| C2 | B | `optimizer.layer_decay` | 0.5 | 0.65 |
| C2 | B | `model.drop_path` | 0.2 | 0.1 |
| C2 | B | `model.drop` | 0.1 | 0.0 |
| C2 | B | `optimizer.weight_decay` | 0.1 | 0.05 |
| C2 | B | `optimizer.clip_grad` | 1.0 | none |
| DX3 | D | the same five keys as C2 | as C2 | as D |
| F1 | E2 | `optimizer.layer_decay` | 0.5 | 0.65 |
| F2 | E2 | `model.drop` | 0.1 | 0.0 |
| F3 | E2 | `model.drop_path` | 0.2 | 0.1 |
| F4 | E2 | `optimizer.weight_decay` | 0.1 | 0.05 |
| G1 | E2 | `model.init_scale` | 1.0 | 0.001 |
| H1 | E2 | `optimizer.layer_decay` | 0.75 | 0.65 |
| H2 | E2 | `optimizer.layer_decay` | 0.85 | 0.65 |

### Table 4.2. Classic-regularizer runs: results

Baselines B, D and E2 are repeated for reference. Columns as in section 2.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| B | [`5261e2a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5261e2a18dfc45d1ae3fdd15991bdacd/output/log) | 8.40 | 8.91 | – | −0.43 | 2 / 6 | 5.42 | C |
| C2 | [`c5f3e6bb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c5f3e6bbf29d4407b67ded877367d9b7/output/log) | 12.47 | 13.52 | – | −0.81 | 14 / 15 | 13.77 | C |
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| DX3 | [`739537da`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/739537daa93b4a3e9c9b0c3bfed1adf8/output/log) | 12.83 | 14.90 | – | −1.00 | 0 / 15 | 14.61 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| F1 | [`d57ef21e`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d57ef21e007f4cd883938f9e076d1792/output/log) | 7.74 | 8.36 | 6.42 / 10.68 | −0.36 | 3 / 4 | 5.88 | C |
| F2 | [`203499b4`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/203499b458ee41b4b98e0f4187e6ee4e/output/log) | 13.25 | 15.17 | 13.37 / 17.32 | −0.97 | 0 / 4 | 14.42 | C |
| F3 | [`d7d5c83a`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d7d5c83a99f74bdabe66b7724ed77a2e/output/log) | 7.70 | 8.41 | 6.31 / 10.91 | −0.35 | 3 / 4 | 5.67 | C |
| F4 | [`9ee26d9a`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/9ee26d9a82de454b9b942fd3057b1c7a/output/log) | 7.66 | 8.24 | 6.49 / 10.32 | −0.37 | 3 / 4 | 4.07 | C |
| G1 | [`971927e2`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/971927e2584d41a6ae6803278151516e/output/log) | 7.76 | 8.38 | 6.64 / 10.45 | −0.37 | 3 / 4 | 4.04 | C |
| H1 | [`fb510312`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/fb510312b11542e0aa2e10853cdd85b6/output/log) | 7.71 | 8.67 | 6.64 / 11.09 | −0.40 | 1 / 4 | 5.30 | C |
| H2 | [`06f8dc25`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/06f8dc2504cb4a4f85886fc93f9d25be/output/log) | 7.86 | 8.54 | 6.74 / 10.69 | −0.42 | 3 / 4 | 1.80 | C |

**Verdict.** None of these closes the gap. Dropout (`model.drop=0.1`) alone
collapses the model to a near-constant prediction: F2 gives $\delta$ =
+5.57 / +6.92 with $\beta=-0.97$, and its best epoch is 0. Dropout therefore
explains the C2 and DX3 collapses (DX3: $e^{\star}=0$, 12.83 / 14.90,
$\beta=-1.00$), not weight decay or drop path. Every other single knob stays
within the noise floor on val: F1 +0.06, F3 +0.02, F4 −0.02, G1 +0.08. They cost
between −0.01 and +0.16 on test. A weaker layer decay memorizes more: H2 reaches
a train MAE of 1.80 at $e^{\star}$ and gives +0.18 / +0.29, and H1 costs +0.42 on
test.

## 5. Restricting adaptation: head-only, last block, LoRA

**Hypothesis.** If memorization lives in the encoder, then training fewer
parameters, or only a low-rank update, should leave less capacity to store the
recording → age lookup.

Table 5.1 lists the config diff and Table 5.2 the results.

### Table 5.1. Restricted-adaptation runs: config diff

| Run | Baseline | Key | Value | Baseline value |
|---|---|---|---|---|
| C2-head | C2 | `model.trainable_prefixes` | `["head"]` | `[]` (all) |
| C2-last | C2 | `model.trainable_prefixes` | `["blocks.11", "head"]` | `[]` (all) |
| M3 | E2 | `model.lora.enabled`, `rank`, `alpha` | true, 16, 32.0 | false |
| M3 | E2 | `model.lora.targets` / `train_prefixes` | `qkv, proj, fc1, fc2` / `head, fc_norm, cls_norm` | – |
| M3 | E2 | `optimizer.lr` | 1e-3 | 1e-4 |
| M3 | E2 | `optimizer.layer_decay` | 1.0 | 0.65 |

### Table 5.2. Restricted-adaptation runs: results

Columns as in section 2. The ClearML task for C2-head failed after receiving
2 epochs. The numbers come from the local log, which holds 8 epochs before the
process stopped.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| C2 | [`c5f3e6bb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c5f3e6bbf29d4407b67ded877367d9b7/output/log) | 12.47 | 13.52 | – | −0.81 | 14 / 15 | 13.77 | C |
| C2-head | [`a2f82906`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a2f829069b064581abf55d3abc3c2200/output/log) | 12.77 | 14.78 | – | −0.99 | 5 / 8 | 14.59 | L |
| C2-last | [`bc668869`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/bc6688691fdb4ff1ab3e54d5e35de445/output/log) | 12.50 | 13.64 | – | −0.87 | 6 / 15 | 14.48 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| M3 | [`8b3df0b6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/8b3df0b6575740419da7e5b0bc06868d/output/log) | 7.68 | 9.24 | 7.41 / 11.42 | −0.44 | 3 / 4 | 1.82 | C |

**Verdict.** The head-only and last-block runs are inconclusive. They inherit
C2's `model.drop=0.1`, which section 4 showed collapses the model, so they
underfit (train MAE 14.4–14.6, $\beta\approx-0.9$) for a reason unrelated to
freezing. They should be rerun on E2 without dropout. LoRA does the opposite of
its intent. At rank 16 with lr 1e-3, it reaches a train MAE of 1.82 in
4 epochs; only H2 (1.80) memorizes as fast. Its val MAE is unchanged ($\delta$ =
0.00) and its test MAE is much worse (+0.99), with the largest cohort errors
in the study (7.41 / 11.42).

## 6. Mixup, C-Mixup and EMA evaluation

**Hypothesis.** Mixing inputs and targets across recordings, Eqs. (12.1)–(12.2)
of `age_regression.md`, means no window maps to a single recording's age.
Averaging weights, Eq. (12.3), smooths out the late recording-specific updates.

Table 6.1 lists the config diff and Table 6.2 the results.

### Table 6.1. Mixup and EMA runs: config diff

Keys that differ from E2.

| Run | Key | Value | E2 value |
|---|---|---|---|
| M1 | `mixup.enabled`, `alpha`, `sigma`, `prob` | true, 0.4, 0.0, 1.0 | false |
| M1c | as M1, but `mixup.sigma` (C-Mixup, years) | 5.0 | – |
| M2 | `optimizer.model_ema`, `model_ema_decay`, `evaluation.use_ema` | true, 0.9995, true | false, 0.9999, false |
| M12 | M1 + M2 | | |
| L1 | M12 + `trainer.epochs`, `optimizer.warmup_epochs` | 12, 2 | 4, 1 |

### Table 6.2. Mixup and EMA runs: results

Columns as in section 2. For the mixup runs, train MAE is the mixed-target
value (†). The EMA runs report the raw-weight train MAE.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| M1 | [`f7fb25cb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f7fb25cbdf0a427c8fa513f5abb48e1d/output/log) | 7.46 | 8.45 | 6.60 / 10.65 | −0.41 | 3 / 4 | 6.38† | C |
| M1c | [`a2698983`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a269898317d047d0ab9c2ada3bd8963f/output/log) | 7.54 | 8.23 | 6.50 / 10.29 | −0.39 | 3 / 4 | 6.30† | C |
| M2 | [`e86c89fe`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/e86c89fef38c4adbbe77e6942772f022/output/log) | 7.58 | 8.17 | 6.35 / 10.35 | −0.36 | 2 / 4 | 4.76 | C |
| M12 | [`d2a9eb9c`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d2a9eb9c61474023b3dfae8d049a3b0d/output/log) | 7.48 | 8.45 | 6.57 / 10.69 | −0.42 | 3 / 4 | 6.38† | C |
| L1 | [`32c097a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/32c097a17ff84c98b3b9cfce5281a2da/output/log) | 7.51 | 8.47 | 6.65 / 10.63 | −0.42 | 3 / 12 | 6.54† | C |

**Verdict.** Mixup gives the best val MAE in the study (M1 7.46, $\delta$ =
−0.22), but the gain does not reach test (+0.20). M12 and L1 behave the same
way. C-Mixup (M1c, −0.14 / −0.02) and EMA alone (M2, −0.10 / −0.08) are the most
balanced. M2 has the best test normal MAE (6.35), and M1c the best test
abnormal MAE (10.29), but both test deltas are below the noise floor. None of
them changes $e^{\star}$ (3, or 2 for M2) or $\beta$ by more than 0.05.

## 7. Loss shape: L1, Huber δ, soft labels, age-balanced loss

**Hypothesis.** A loss that is less sensitive to large residuals (L1, a wider
Huber), a distributional target (soft labels, Eqs. (12.7)–(12.9) of
`age_regression.md`), or reweighting the rare ages (Eqs. (13.5)–(13.8))
generalizes better, or at least reduces the bias slope.

Table 7.1 lists the config diff and Table 7.2 the results.

### Table 7.1. Loss runs: config diff

| Run | Baseline | Key | Value | Baseline value |
|---|---|---|---|---|
| D2 | B | `loss.regression_loss` | `l1` | `huber` |
| DX2 | D | `loss.huber_delta` | 2.0 | 1.0 |
| M4 | E2 | `loss.regression_loss`, `soft_label_sigma` | `soft_label`, 2.5 (bins 1–89 y, width 1 y) | `huber` |
| M4 | E2 | `model.nb_classes` (set by the loss) | 89 | 1 |
| B1 | E2 | `loss.balance` (`balance_bin_width` 1, `balance_lds_sigma` 2, `balance_max_weight` 10) | `sqrt_inverse` | `none` |
| B2 | E2 | `loss.balance` (same options) | `inverse` | `none` |

### Table 7.2. Loss runs: results

Columns as in section 2.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| B | [`5261e2a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5261e2a18dfc45d1ae3fdd15991bdacd/output/log) | 8.40 | 8.91 | – | −0.43 | 2 / 6 | 5.42 | C |
| D2 | [`6753cae7`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/6753cae73d0b40dc9f88b7b0e125a2c3/output/log) | 8.36 | 9.32 | – | −0.46 | 6 / 15 | 2.34 | C |
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| DX2 | [`a50cc416`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a50cc4161dd84bfc9c445ce17993f659/output/log) | 7.79 | 8.41 | – | −0.38 | 3 / 15 | 4.22 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| M4 | [`bfc213e1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/bfc213e1552b4e588d0d7e4f80b40d90/output/log) | 7.67 | 8.59 | 6.92 / 10.57 | −0.40 | 3 / 4 | 3.48 | C |
| B1 | [`9294d6c1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/9294d6c1ba2b43f6847e09c5571a2622/output/log) | 7.83 | 8.48 | 6.66 / 10.63 | −0.37 | 2 / 4 | 4.90 | C |
| B2 | [`ad5add69`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/ad5add69019a4caa9bb3dd338910a67d/output/log) | 7.96 | 8.57 | 6.70 / 10.80 | −0.33 | 1 / 4 | 6.52 | C |

**Verdict.** No loss beats Huber with δ = 1. L1 costs +0.41 on test (vs. B),
Huber δ 2 costs +0.02 / +0.11 (vs. D), and soft labels cost −0.01 / +0.34. The
age-balanced loss does what it is designed for, though only partly: test
$\beta$ stays at −0.37 with B1 and moves to −0.33 with B2, and val $\beta$ goes
from −0.33 to −0.32 and −0.29. It pays in MAE: +0.15 / +0.23 (B1) and +0.28 / +0.32 (B2). Use it only when
age-independent bias matters more than MAE.

## 8. Codebook regularization

**Hypothesis.** Requiring the fine-tuned encoder's patch tokens to stay
decodable by the frozen VQNSP quantizer and decoder
(`model.codebook_reg`, see `codebook_regularized_finetune_plan.md`) keeps them
close to the pretrained, spectrum-preserving representation. That should leave
less room for recording-specific features.

Table 8.1 lists the config diff and Table 8.2 the results.

### Table 8.1. Codebook-regularization runs: config diff

Keys that differ from D. All three runs use
`model.codebook_reg.enabled=true`, `tokenizer_weight=./checkpoints/vqnsp.pth`,
`encoder.n_last_trainable_layers=None` (all blocks train),
`encoder.lr_scale=0.99`, `classifier_weight=1.0` and `phase_weight=0.0`. The
codebook and the decoder are frozen.

| Run | `amplitude_weight` | `embedding_weight` (quantization) | `loss.freq_fraction` |
|---|---|---|---|
| DX1 | 1.0 | 0.0 | 0.5 |
| DX4 | 0.0 | 1.0 | 1.0 |
| DX4b | 0.0 | 50.0 | 1.0 |

### Table 8.2. Codebook-regularization runs: results

Columns as in section 2. DX4's ClearML task received no scalars, so all its
numbers come from the local log. DX4b's task received 5 of 15 epochs; $e^{\star}$
is 4 in both ClearML and the local log.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| DX1 | [`4188b773`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4188b77339234934bddd10aff47b56ec/output/log) | 7.72 | 8.54 | – | −0.39 | 4 / 15 | 4.10 | C |
| DX4 | [`8a16ee7d`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/8a16ee7df1bf4d9ca73ee290b9cc8b4e/output/log) | 7.67 | 8.44 | – | −0.39 | 3 / 15 | 4.21 | L |
| DX4b | [`ae93f2a6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/ae93f2a698464b73a0c700e3f8cf3645/output/log) | 7.78 | 8.66 | – | −0.37 | 4 / 15 | 4.04 | C+L |

**Verdict.** No gain. $e^{\star}$ moves from 1 to 3–4, so the regularizer slows
the fit (Table 8.2), but val is unchanged (−0.05, −0.10, +0.01) and test is worse
(+0.24, +0.14, +0.36). By epoch 14, train MAE still falls to 1.72 (DX1,
ClearML), 1.26 (DX4) and 1.69 (DX4b, local logs), against 1.23 for D. The
memorization is at best delayed, not prevented.

## 9. Input and augmentation: random crop, window length, gradient clipping

**Hypothesis.** If every epoch shows the same windows, recordings are easy to
memorize. Random crop offsets (`data.random_crop`) or longer context windows
should make that harder. Gradient clipping should stop artifact windows from
causing gradient spikes.

Table 9.1 lists the config diff and Table 9.2 the results.

### Table 9.1. Input and augmentation runs: config diff

| Run | Baseline | Key | Value | Baseline value |
|---|---|---|---|---|
| C | B | `data.window_sec` | 30 | 10 |
| C | B | `trainer.batch_size` × `update_freq` | 32 × 2 | 64 × 1 |
| D-clip1 | D | `optimizer.clip_grad` | 1.0 | none |
| E1 | D | `data.random_crop` | true | false |
| E1c | D | `data.random_crop`, `optimizer.clip_grad` | true, 1.0 | false, none |
| E2c | E2 | `optimizer.clip_grad` | 1.0 | none |

### Table 9.2. Input and augmentation runs: results

Columns as in section 2.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| B | [`5261e2a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5261e2a18dfc45d1ae3fdd15991bdacd/output/log) | 8.40 | 8.91 | – | −0.43 | 2 / 6 | 5.42 | C |
| C | [`b030fae6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/b030fae65bcb474bb84942f3abf04abf/output/log) | 8.24 | 9.35 | – | −0.42 | 8 / 15 | 3.33 | C |
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| D-clip1 | [`278af290`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/278af290dd6a455d8a6b5ce6869163e1/output/log) | 7.70 | 8.36 | – | −0.41 | 1 / 15 | 7.44 | C |
| E1 | [`f53cb248`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f53cb248f38443aa82c0f8306856af96/output/log) | 7.82 | 8.56 | – | −0.40 | 3 / 15 | 4.19 | C |
| E1c | [`23b5d1d9`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/23b5d1d9f8c0449ea413ac9ea0b83131/output/log) | 7.72 | 8.13 | – | −0.38 | 1 / 15 | 7.45 | C |
| E2 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | 4.07 | C |
| E2c | [`aa183ab4`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/aa183ab426074df3bcf305e5e320bf95/output/log) | 7.63 | 8.24 | – | −0.38 | 3 / 4 | 4.20 | C |

**Verdict.** Random crops alone do not stop memorization. E1's train MAE
reaches 1.28 by epoch 14, as D's does, and E1 gives +0.05 / +0.26. E1c (crop +
clip) has the lowest test MAE in the study (8.13, $\delta$ = −0.05 / −0.17).
But clipping alone is neutral (D-clip1 −0.07 / +0.06, E2c −0.05 / −0.01), and
the val gain is within the noise floor, so E1c's test gain is not a reliable
effect. 30-s windows cost +0.44 on test (C vs. B).

## 10. Prediction heads

**Hypothesis.** A head that reads the class token, or one that re-attends over
all channel × time patches (`model.head_type=attention`, Eqs. (15.1)–(15.4) of
`age_regression.md`), extracts more age signal from the same encoder than the
mean-pooled linear head. The MLP head is used only in the from-scratch S2 run
(section 11).

Table 10.1 lists the config diff and Table 10.2 the results.

### Table 10.1. Head runs: config diff

| Run | Baseline | Key | Value | Baseline value |
|---|---|---|---|---|
| D-cls | D | `model.use_mean_pooling` | false (class token) | true |
| D-clsmean | D | `model.concat_cls_token` | true ([mean patch; class token]) | false |
| N2L | D | `model.head_type` (`attention_head.num_heads` 4, `mlp_hidden_dim` 256) | `attention` | `linear` |
| N2L | D | `mixup.enabled`, `alpha`, `sigma` | true, 0.4, 0.0 | false |
| N2L | D | `optimizer.clip_grad` | 1.0 | none |
| N2L | D | `trainer.epochs`, `optimizer.warmup_epochs` | 12, 4 | 15, 2 |

N2L is `labram/configs/defaults/finetune_tuab_age_attn_head.json` with
`trainer.epochs=12 optimizer.warmup_epochs=4` (`.run_attn_head_local.sh`). The
planned controlled pair N1/N2 (4 epochs; `scripts/submit_age_experiments.sh`)
was never submitted: there is no ClearML task and no SageMaker job for it. The
closest controls for N2L are therefore M1 (mixup, 4 epochs) and L1 (mixup +
EMA, 12 epochs).

### Table 10.2. Head runs: results

Columns as in section 2.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| D-cls | [`c48bc257`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c48bc2571f694b119d4a2d94751c455a/output/log) | 7.90 | 8.24 | – | −0.38 | 2 / 15 | 5.49 | C |
| D-clsmean | [`5eb6e379`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5eb6e3799586403a851771cbf6f3f47d/output/log) | 7.80 | 8.35 | – | −0.41 | 1 / 15 | 7.31 | C |
| M1 | [`f7fb25cb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f7fb25cbdf0a427c8fa513f5abb48e1d/output/log) | 7.46 | 8.45 | 6.60 / 10.65 | −0.41 | 3 / 4 | 6.38† | C |
| L1 | [`32c097a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/32c097a17ff84c98b3b9cfce5281a2da/output/log) | 7.51 | 8.47 | 6.65 / 10.63 | −0.42 | 3 / 12 | 6.54† | C |
| N2L | [`74bf270f`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/74bf270fa9a44a7c9a4850b3f2be8e06/output/log) | 7.63 | 8.49 | 6.68 / 10.64 | −0.43 | 4 / 12 | 6.62† | C+L |

**Verdict.** No head helps. The class-token head gives +0.13 / −0.06 and the
concatenated head +0.03 / +0.05 (vs. D). The attention head gives
+0.12 / +0.02 (vs. L1) and +0.17 / +0.04 (vs. M1), and its best epoch moves to
4. The class-token head moves $e^{\star}$ from 1 to 2 without changing the result.

## 11. From scratch

**Hypothesis.** Measure what pretraining is worth. Then test whether a smaller
model, trained from random initialization with every anti-memorization option
switched on, can match the fine-tune.

Table 11.1 lists the config diff and Table 11.2 the results.

### Table 11.1. From-scratch runs: config diff

Keys that differ from D.

| Run | Key | Value | D value |
|---|---|---|---|
| S1, S2 | `finetune_checkpoint.finetune` | empty (random init) | `labram-base.pth` |
| S1, S2 | `optimizer.layer_decay` | 1.0 | 0.65 |
| S2 | `model.depth` | 6 | null (12) |
| S2 | `model.head_type`, `mlp_head.hidden_dims` | `mlp`, `[256]` | `linear` |
| S2 | `optimizer.lr`, `opt_betas` | 3e-4, [0.9, 0.98] | 1e-4, null |
| S2 | `optimizer.warmup_epochs`, `clip_grad` | 4, 1.0 | 2, none |
| S2 | `mixup.enabled`, `alpha` | true, 0.4 | false |
| S2 | `data.random_crop` | true | false |
| S2 | `optimizer.model_ema`, `model_ema_decay`, `evaluation.use_ema` | true, 0.9995, true | false, 0.9999, false |
| S2-seed1 | S2 + `distributed.seed` | 1 | 0 |

### Table 11.2. From-scratch runs: results

Columns as in section 2. Both S2 ClearML tasks stopped receiving data after
epoch 1 ("Forced stop (non-responsive)"), but both training processes finished
all 15 epochs. The S2 numbers come from the SageMaker job
`labram-age-s2-2026-10-10-09-11-41-932` (CloudWatch summary), and the S2-seed1
numbers from its local `log.txt`.

| Run | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | train MAE at $e^{\star}$ | src |
|---|---|---|---|---|---|---|---|---|
| D | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | 7.24 | C |
| S1 | [`f724ed15`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f724ed1592c34ccdbb1cf6edb539de93/output/log) | 8.79 | 9.42 | 7.75 / 11.42 | −0.49 | 14 / 15 | 3.22 | C+L |
| S2 | [`88ad5cff`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/88ad5cffb2a145baba5da39a0ce59b74/output/log) | 8.53 | 9.30 | 7.48 / 11.48 | −0.50 | 12 / 15 | 6.53† | W |
| S2-seed1 | [`c3804cfb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c3804cfb0bed4435a00a9115803fea16/output/log) | 8.58 | 9.44 | 7.84 / 11.35 | −0.50 | 14 / 15 | 6.45† | L |

**Verdict.** Pretraining is worth about one year. S1 gives $\delta$ =
+1.02 / +1.12 and the tuned S2 gives +0.76 / +1.00 (seed 1: +0.81 / +1.14).
The from-scratch runs select a late epoch (12–14), so they are not
memorization-limited within 15 epochs, but they stay well behind the fine-tune.
The two S2 seeds supply the noise-floor estimate of section 2.

## 12. All runs

Table 12.1 collects every run of Tables 2.2–11.2 in one place, each with its
ClearML link.

### Table 12.1. Master table

One row per significant run, grouped by family. The "Change" column is
relative to D unless it names another run. Metrics: val and test MAE, Eq.
(1.2); test cohort MAE, Eq. (1.3); test $\beta$, Eq. (1.4); $e^{\star}$,
Eq. (1.5). `src` codes as in section 2. Crashed, duplicate and superseded runs
are in Table 14.1.

| Run | Family | Change vs. D (or as stated) | Section | ClearML | val MAE | test MAE | test normal / abnormal | test $\beta$ | $e^{\star}$ / epochs | src |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Evaluation protocol | B with whole-recording val/test | §2 | [`08833720`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/088337201a0b435d9e718e3e59dfa565/output/log) | 8.29 | 9.03 | – | −0.46 | 2 / 12 | C |
| B | Evaluation protocol | CAR off, 15 ep, lr 1e-4, 5-min eval (pre-D baseline) | §2 | [`5261e2a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5261e2a18dfc45d1ae3fdd15991bdacd/output/log) | 8.40 | 8.91 | – | −0.43 | 2 / 6 | C |
| D | Baseline | B + CAR (`finetune_tuab_age.json`) | §2 | [`4af88820`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4af88820221b4fb482df489bcd5e4ff8/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | C |
| D-npy | Baseline (repro) | D on the npy format | §2 | [`5b9138de`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5b9138de74c542fe8c822fb954d87df6/output/log) | 7.77 | 8.30 | – | −0.40 | 1 / 15 | C |
| D-cohorts | Baseline (repro) | D, local, cohort metrics | §2 | [`a39059ad`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a39059ad71d44fe1a4dbd0561c08ae4e/output/log) | 7.77 | 8.31 | 6.57 / 10.38 | −0.40 | 1 / 15 | C |
| E2 | Schedule | D, 4 ep / 1 warmup | §3 | [`cd031cac`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/cd031cacc874464a9a416c5aec0dbce4/output/log) | 7.68 | 8.25 | – | −0.37 | 3 / 4 | C |
| L1 | Schedule | M12 on 12 ep / 2 warmup | §3, §6 | [`32c097a1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/32c097a17ff84c98b3b9cfce5281a2da/output/log) | 7.51 | 8.47 | 6.65 / 10.63 | −0.42 | 3 / 12 | C |
| C2 | Classic regularizers | B + ld 0.5, dp 0.2, drop 0.1, wd 0.1, clip 1 | §4 | [`c5f3e6bb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c5f3e6bbf29d4407b67ded877367d9b7/output/log) | 12.47 | 13.52 | – | −0.81 | 14 / 15 | C |
| DX3 | Classic regularizers | D + C2 bundle | §4 | [`739537da`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/739537daa93b4a3e9c9b0c3bfed1adf8/output/log) | 12.83 | 14.90 | – | −1.00 | 0 / 15 | C |
| F1 | Classic regularizers | E2 + layer decay 0.5 | §4 | [`d57ef21e`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d57ef21e007f4cd883938f9e076d1792/output/log) | 7.74 | 8.36 | 6.42 / 10.68 | −0.36 | 3 / 4 | C |
| F2 | Classic regularizers | E2 + dropout 0.1 | §4 | [`203499b4`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/203499b458ee41b4b98e0f4187e6ee4e/output/log) | 13.25 | 15.17 | 13.37 / 17.32 | −0.97 | 0 / 4 | C |
| F3 | Classic regularizers | E2 + drop path 0.2 | §4 | [`d7d5c83a`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d7d5c83a99f74bdabe66b7724ed77a2e/output/log) | 7.70 | 8.41 | 6.31 / 10.91 | −0.35 | 3 / 4 | C |
| F4 | Classic regularizers | E2 + weight decay 0.1 | §4 | [`9ee26d9a`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/9ee26d9a82de454b9b942fd3057b1c7a/output/log) | 7.66 | 8.24 | 6.49 / 10.32 | −0.37 | 3 / 4 | C |
| G1 | Head init | E2 + `init_scale` 1.0 | §4 | [`971927e2`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/971927e2584d41a6ae6803278151516e/output/log) | 7.76 | 8.38 | 6.64 / 10.45 | −0.37 | 3 / 4 | C |
| H1 | Layer decay | E2 + layer decay 0.75 | §4 | [`fb510312`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/fb510312b11542e0aa2e10853cdd85b6/output/log) | 7.71 | 8.67 | 6.64 / 11.09 | −0.40 | 1 / 4 | C |
| H2 | Layer decay | E2 + layer decay 0.85 | §4 | [`06f8dc25`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/06f8dc2504cb4a4f85886fc93f9d25be/output/log) | 7.86 | 8.54 | 6.74 / 10.69 | −0.42 | 3 / 4 | C |
| C2-head | Partial fine-tuning | C2, head only | §5 | [`a2f82906`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a2f829069b064581abf55d3abc3c2200/output/log) | 12.77 | 14.78 | – | −0.99 | 5 / 8 | L |
| C2-last | Partial fine-tuning | C2, last block + head | §5 | [`bc668869`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/bc6688691fdb4ff1ab3e54d5e35de445/output/log) | 12.50 | 13.64 | – | −0.87 | 6 / 15 | C |
| M3 | Partial fine-tuning | E2 + LoRA r16, lr 1e-3, ld 1.0 | §5 | [`8b3df0b6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/8b3df0b6575740419da7e5b0bc06868d/output/log) | 7.68 | 9.24 | 7.41 / 11.42 | −0.44 | 3 / 4 | C |
| M1 | Mixup / EMA | E2 + mixup 0.4 | §6 | [`f7fb25cb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f7fb25cbdf0a427c8fa513f5abb48e1d/output/log) | 7.46 | 8.45 | 6.60 / 10.65 | −0.41 | 3 / 4 | C |
| M1c | Mixup / EMA | E2 + C-Mixup 0.4, σ 5 y | §6 | [`a2698983`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a269898317d047d0ab9c2ada3bd8963f/output/log) | 7.54 | 8.23 | 6.50 / 10.29 | −0.39 | 3 / 4 | C |
| M2 | Mixup / EMA | E2 + EMA eval (0.9995) | §6 | [`e86c89fe`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/e86c89fef38c4adbbe77e6942772f022/output/log) | 7.58 | 8.17 | 6.35 / 10.35 | −0.36 | 2 / 4 | C |
| M12 | Mixup / EMA | E2 + mixup 0.4 + EMA eval | §6 | [`d2a9eb9c`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/d2a9eb9c61474023b3dfae8d049a3b0d/output/log) | 7.48 | 8.45 | 6.57 / 10.69 | −0.42 | 3 / 4 | C |
| D2 | Loss | B + L1 loss | §7 | [`6753cae7`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/6753cae73d0b40dc9f88b7b0e125a2c3/output/log) | 8.36 | 9.32 | – | −0.46 | 6 / 15 | C |
| DX2 | Loss | D + Huber δ 2 | §7 | [`a50cc416`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/a50cc4161dd84bfc9c445ce17993f659/output/log) | 7.79 | 8.41 | – | −0.38 | 3 / 15 | C |
| M4 | Loss | E2 + soft labels σ 2.5 y | §7 | [`bfc213e1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/bfc213e1552b4e588d0d7e4f80b40d90/output/log) | 7.67 | 8.59 | 6.92 / 10.57 | −0.40 | 3 / 4 | C |
| B1 | Loss | E2 + age-balanced, sqrt-inverse | §7 | [`9294d6c1`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/9294d6c1ba2b43f6847e09c5571a2622/output/log) | 7.83 | 8.48 | 6.66 / 10.63 | −0.37 | 2 / 4 | C |
| B2 | Loss | E2 + age-balanced, inverse | §7 | [`ad5add69`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/ad5add69019a4caa9bb3dd338910a67d/output/log) | 7.96 | 8.57 | 6.70 / 10.80 | −0.33 | 1 / 4 | C |
| DX1 | Codebook regularization | D + amplitude reconstruction (½ spectrum) | §8 | [`4188b773`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/4188b77339234934bddd10aff47b56ec/output/log) | 7.72 | 8.54 | – | −0.39 | 4 / 15 | C |
| DX4 | Codebook regularization | D + quantization term, weight 1 | §8 | [`8a16ee7d`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/8a16ee7df1bf4d9ca73ee290b9cc8b4e/output/log) | 7.67 | 8.44 | – | −0.39 | 3 / 15 | L |
| DX4b | Codebook regularization | D + quantization term, weight 50 | §8 | [`ae93f2a6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/ae93f2a698464b73a0c700e3f8cf3645/output/log) | 7.78 | 8.66 | – | −0.37 | 4 / 15 | C+L |
| C | Augmentation / input | B + 30-s windows | §9 | [`b030fae6`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/b030fae65bcb474bb84942f3abf04abf/output/log) | 8.24 | 9.35 | – | −0.42 | 8 / 15 | C |
| D-clip1 | Augmentation / input | D + clip 1.0 | §9 | [`278af290`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/278af290dd6a455d8a6b5ce6869163e1/output/log) | 7.70 | 8.36 | – | −0.41 | 1 / 15 | C |
| E1 | Augmentation / input | D + random crop | §9 | [`f53cb248`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f53cb248f38443aa82c0f8306856af96/output/log) | 7.82 | 8.56 | – | −0.40 | 3 / 15 | C |
| E1c | Augmentation / input | D + random crop + clip 1.0 | §9 | [`23b5d1d9`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/23b5d1d9f8c0449ea413ac9ea0b83131/output/log) | 7.72 | 8.13 | – | −0.38 | 1 / 15 | C |
| E2c | Augmentation / input | E2 + clip 1.0 | §9 | [`aa183ab4`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/aa183ab426074df3bcf305e5e320bf95/output/log) | 7.63 | 8.24 | – | −0.38 | 3 / 4 | C |
| D-cls | Head | D, CLS token instead of mean pooling | §10 | [`c48bc257`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c48bc2571f694b119d4a2d94751c455a/output/log) | 7.90 | 8.24 | – | −0.38 | 2 / 15 | C |
| D-clsmean | Head | D, [mean patch; CLS] concat | §10 | [`5eb6e379`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/5eb6e3799586403a851771cbf6f3f47d/output/log) | 7.80 | 8.35 | – | −0.41 | 1 / 15 | C |
| N2L | Head | D + mixup 0.4 + clip 1 + attention head, 12 ep / 4 warmup | §10 | [`74bf270f`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/74bf270fa9a44a7c9a4850b3f2be8e06/output/log) | 7.63 | 8.49 | 6.68 / 10.64 | −0.43 | 4 / 12 | C+L |
| S1 | From scratch | D, no pretrained weights, ld 1.0 | §11 | [`f724ed15`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/f724ed1592c34ccdbb1cf6edb539de93/output/log) | 8.79 | 9.42 | 7.75 / 11.42 | −0.49 | 14 / 15 | C+L |
| S2 | From scratch | scratch, 6 blocks + MLP head, tuned (seed 0) | §11 | [`88ad5cff`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/88ad5cffb2a145baba5da39a0ce59b74/output/log) | 8.53 | 9.30 | 7.48 / 11.48 | −0.50 | 12 / 15 | W |
| S2-seed1 | From scratch | S2, seed 1 | §11 | [`c3804cfb`](https://app.clear.ml/projects/c44d37fe41b1483481d5495d42b4ffdc/experiments/c3804cfb0bed4435a00a9115803fea16/output/log) | 8.58 | 9.44 | 7.84 / 11.35 | −0.50 | 14 / 15 | L |

## 13. Conclusions

1. **Nothing tried so far closes the train/val gap at the selected epoch while
   also improving test.** All 34 pretrained runs that did not collapse land in
   val 7.46–8.40 and test 8.13–9.35. Excluding the pre-CAR runs (A, B, C, C2,
   D2) and LoRA, the range is 7.46–7.96 / 8.13–8.67.
2. **Early stopping does most of the work.** Every pretrained run that did not
   collapse selects epoch 1–4; the exceptions are C (epoch 8) and D2 (epoch 6). A 4-epoch schedule
   (E2) is as good as 15 epochs (Table 3.2) and about 4× cheaper.
3. **Weight-space regularizers are neutral, and dropout is harmful.**
   `model.drop=0.1` alone collapses training (F2). That confounds every bundle
   containing it: C2, DX3, and the head-only and last-block runs (Tables 4.2
   and 5.2). Weight decay, drop path, layer decay 0.5 and head init are within
   the noise floor. A weaker layer decay (H1, H2) and LoRA memorize more and
   lose on test.
4. **Mixup improves val but not test.** M1 has the best val MAE (7.46) and test
   +0.20 (Table 6.2). EMA (M2) and C-Mixup (M1c) are the most balanced, but by
   less than the noise floor.
5. **Loss, codebook and head changes do not lower the MAE** (Tables 7.2, 8.2
   and 10.2). The age-balanced loss lowers the bias slope at an MAE cost of
   0.2–0.3 years.
6. **Pretraining is worth about one year** (Table 11.2).
7. **The abnormal cohort carries the error.** Test abnormal MAE is
   10.29–11.09 against 6.31–6.92 for normal recordings, across every run with
   cohort metrics except the collapsed, LoRA and from-scratch ones; no method
   narrows it. The "Next steps" section of `brain_age_improvement_plan.md` lists
   the remaining levers:
   more subjects (TUEG ages), normal-only training, seed or fold ensembles, and
   channel-dropout augmentation.
8. **Recommended configuration.** On val, which is the selection criterion,
   that is M1 (scenario D + mixup 0.4, 4 epochs). The attention-head config
   file adds a head that did not help (Table 10.2). Before any claim of a
   sub-0.15-year improvement, report a seed spread.

## 14. Appendix: excluded runs and discrepancies

Table 14.1 lists the runs left out of the registry, Table 14.2 the runs
before the October ablation, and Table 14.3 the numbers in other docs that
disagree with ClearML or the logs.

### Table 14.1. Crashed, duplicate and superseded runs

These runs are not in the tables above. The ClearML ID prefix is enough to
find each one in project `LaBraM/brain_age`.

| ClearML / location | Name | Status | Why excluded |
|---|---|---|---|
| `334aa154`, `73107f56`, `ce555e12`, `6cd56922`, `42537ea1`, `6eaac14c`, `e41800aa`, `41fcbf59` | first submissions of A, B (×2), C (×2), C2, D, D2 | stopped, 0 epochs | Stopped before the first epoch and resubmitted; the resubmissions are the rows in Tables 2.2–9.2 |
| `a2f82906` | `age_C2_eval5min_regularized_headonly` | failed after 2 epochs in ClearML | Reported as C2-head from the local log (8 epochs), Table 5.2 |
| `afcc8c7a` (local `checkpoints/age_DX3_regularized_*`) | DX3, local, pickle | stopped; 1 epoch in ClearML, 9 epochs locally | Superseded by the 15-epoch SageMaker DX3 (`739537da`); same collapse, 12.83 / 14.90 at epoch 0 |
| `8a16ee7d` | DX4 | stopped, no scalars | Reported from the local log, Table 8.2 |
| `0a377e6a` | `age_S2_scratch_tuned_ep15` | SageMaker job stopped after ~10 steps | First S2 attempt (linear head); superseded by `88ad5cff` |
| `35fbaf8e` | `age_S2_scratch_d6_mlp_ep15_local` | killed after epoch 0 | Local duplicate of the SageMaker S2 |
| `da4d4720` | `age_S2_..._seed1_local` (13:41) | killed after 3 epochs | ClearML overrode `distributed.seed` (its epoch 0 equals seed 0's exactly); fixed in `7b9a5fe` and rerun as `c3804cfb` |
| `checkpoints/finetune_tuab_age_20261007_2313*` (4 dirs) | – | only `run_config.yaml` | Aborted starts four minutes before M1; no ClearML task, no epochs |
| `9de492d0`, `fa9c8bd2` | `debug_logging_layout` | completed | Logging-layout debug runs, not experiments |
| – | N1, N2 (`submit_age_experiments.sh`) | never submitted | No ClearML task, no SageMaker job |

### Table 14.2. Pre-ablation runs (ClearML project `LaBraM`)

The memorization history before the October ablation. Val/test MAE, Eq. (1.2),
at the run's own best val epoch.

| ClearML | Name | Notes | val MAE | test MAE |
|---|---|---|---|---|
| – (local `checkpoints/finetune_tuab_age/log.txt`, first segment) | August run | 34 epochs; best epoch 2; train MAE 0.78 by epoch 33. Its exact config is not recorded: the directory's `run_config.yaml` belongs to a later restart (lr 5e-4, 50 epochs, CAR off) | 8.31 | 9.04 |
| [`fe92293b`](https://app.clear.ml/projects/914f0c38905e4ba090a6fd54fc2508f3/experiments/fe92293b0d4c4592bf6b5553fa77f18b/output/log) | `finetune_tuab_age_20260802_112833_994` | lr 2e-4, 25 epochs, warmup 5; logs 11 epochs that continue from an already-overfit state (train MAE 0.76 in its first logged epoch) | 9.25 | 10.97 |
| [`3700b756`](https://app.clear.ml/projects/914f0c38905e4ba090a6fd54fc2508f3/experiments/3700b75684684aba845480a50735018d/output/log) | `finetune_tuab_age_20260905_220055_746` | same config; auto-resumed an overfit checkpoint (not a valid experiment); train MAE ≈ 0.36 at the end, while val ends at 9.36 / test 11.10 | 9.32 | 10.92 |

`cb82684a` (2026-09-05, 1 epoch, stopped) is a crashed start of `3700b756`.

### Table 14.3. Discrepancies with the other docs

Differences between existing docs and ClearML or the run logs, as of
2026-10-10.

| Doc | Statement | ClearML / log |
|---|---|---|
| `age_training_scenario_D.md`, section 7 | B: val 8.49 / test 8.91 "(2 of 15 epochs at time of writing)" | Final: 8.40 / 8.91 at epoch 2, 6 of 15 epochs (`5261e2a1`, stopped) |
| `age_training_scenario_D.md`, section 7 | "All runs share … 15 epochs" | A ran 12 and B 6 of 15 epochs (both stopped) |
| `age_training_scenario_D.md`, section 7 | C2 head-only 12.77 / 14.78 | Matches the local log (8 epochs). The ClearML task failed with 2 epochs and shows 12.80 / 14.85 |
| `age_regression.md`, section 13, and `notebooks/age_error_analysis.ipynb`, section 9 | B1 test 8.47 | ClearML `test_mae` = 8.4751, which rounds to 8.48 (corrected in both) |
| `age_regression.md`, Table 15.2 | N1 / N2 listed as the attention-head experiments | Never run. The only attention-head run is N2L (12 epochs, local), Table 10.2 |
| `brain_age_improvement_plan.md`, "Running now" | A–D2 running | All finished or stopped by 2026-10-05 |
| ClearML vs. local logs | DX4 (no scalars), DX4b (5 of 15 epochs), DX3 local (1 of 9), S2 and S2-seed1 (2 of 15) | Local logs or CloudWatch hold the full runs; the numbers here come from them (`src` column) |

## 15. Sources and reproduction

* **ClearML.** Reached through the SDK in `.venv` (credentials from
  `~/clearml.conf`). For each task in `LaBraM/brain_age` (56 tasks, 54
  excluding debug runs), the registry reads:
  * `Task.get_reported_single_values()` (the `summary` values written by
    `runs/common.py::log_summary_metrics`);
  * `get_reported_scalars()` (`err/{train,val,test}` per epoch, used for runs
    that predate the summary);
  * `get_parameters_as_dict()["config"]`, diffed against the D-cohorts run
    `a39059ad` to get the config-diff tables.
* **Local runs.** `checkpoints/age_*/log.txt` (one JSON per epoch) and
  `run_config.yaml`. The launch scripts are `scripts/submit_age_experiments.sh`
  (SageMaker: A–D2, DX\*, F\*, G1, H\*, M2–M4, S1/S2) and the untracked
  `.run_*.sh` scripts in the repo root (local: D-cohorts, M1/M1c, M12, L1,
  B1/B2, N2L, S1, S2-seed1, plus `.run_short_ablations.sh`, the throttled
  submitter for M2–M4, F, G and H). The D-npy variants and E1/E1c/E2/E2c were
  submitted with `--set` overrides; their exact keys are in each ClearML
  task's `config` section.
* **SageMaker.** The S2 seed-0 results come from the CloudWatch log of job
  `labram-age-s2-2026-10-10-09-11-41-932`. Its `model.tar.gz` is KMS-encrypted
  and could not be read from this machine.
