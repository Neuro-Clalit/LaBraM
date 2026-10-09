# Age regression (EEG brain age) on TUH data

Predicting a patient's age from their EEG — "brain age" — as a downstream
regression task, using the same pre-trained LaBraM encoder as the abnormal/normal
classification task. Off by default; opt in with `data.dataset=TUAB_AGE`.

## Where the age comes from

TUH **no longer distributes the clinical reports** ("We no longer distribute
reports with our corpora"), so the EDF header is the only source of patient
demographics. It is a complete one: the standard EDF *local patient
identification* field holds the age directly.

```
bytes   8:88   local patient identification
               'aaaaantl F 01-JAN-0000 aaaaantl Age:42'
                subject   sex dob      subject   age
bytes  88:168  local recording identification
               'Startdate 01-JAN-2012 aaaaantl_s001 XXX X'
                          session year
```

**MNE cannot give you this.** `read_raw_edf(...).info['subject_info']` returns
only `{'his_id', 'sex', 'last_name'}`: the date of birth is anonymised to
`01-JAN-0000`, so there is nothing to subtract a birth year from. The literal
`Age:` token has to be read from the raw header bytes, which is what
`labram/data/tuh_metadata.py` does — reading only the first 256 bytes per file, so
a full-corpus sweep takes seconds and needs neither MNE nor pandas.

### Validation

On TUAB v3.0.0 the parser finds an `Age:` field in **all 2,993 recordings (100%,
0 missing)**, and the parsed values reproduce the corpus `AAREADME.txt`
DEMOGRAPHICS tables *exactly* — every age decade bucket per split × label, and
every gender count (F/M × normal/abnormal × train/eval). That agreement is the
evidence the byte offsets and token positions above are right.

Usable ages 1–89: **2,978 recordings**, mean 49.1, median 49, std 17.4.

### Sentinel values

Two `Age:` values are sentinels rather than ages; Table 1.1 lists them.

**Table 1.1.** Sentinel `Age:` values in the TUAB v3.0.0 EDF headers: count of
recordings and how the loader handles them.

| Value | Count (TUAB) | Meaning | Handling |
|---|---|---|---|
| `Age:999` | 12 | TUH's redaction for patients aged 90+ (HIPAA requires ages over 89 to be aggregated). Confirmed: the per-split/per-label counts match the README's "90-100" row exactly (1/1/6/4). | Excluded |
| `Age:0` | 3 | Ambiguous — a genuine neonate or a missing value. | Excluded by the default 1–89 range |

Sentinels are dropped, never trained on. `RecordingMetadata.age` is `None` for
them while `raw_age` keeps the as-parsed value for auditing, so a sentinel can
never silently become a target. Widen the range with `--max_age` if you want the
90+ group in (their true ages are unrecoverable, so this is rarely a good idea).

## Joining age onto the existing windows

The window pickles `dataset_maker/make_TUAB.py` already produced are named
`<subject>_s<NNN>_t<NNN>_<windowIdx>.pkl`, and 100% of processed recording stems
map back to an EDF stem. So the age joins onto the **existing** windows by
filename via a sidecar — **no re-preprocessing**, which would otherwise be a
multi-hour MNE pass over ~409k windows. Only ~0.6% of windows are lost to
sentinels.

```
<corpus>/edf/processed/
├── age_metadata.json     # stem -> {age, sex, subject, session, token, year}
├── age_split.json        # subject-disjoint train/val/test window lists
├── train/  val/  test/   # the pickles make_TUAB.py wrote, untouched
```

`TUABAgeLoader` resolves `age_metadata.json` by searching its data root and
parents, so it can recover its labels from the root alone. That matters because
`cross_validation._build_split_dataset` rebuilds loaders positionally as
`type(src)(root, files, sampling_rate)` — there is no opportunity to pass a lookup
through.

Windows whose recording has no usable age are **filtered at construction time**,
not at access time: `TUHLoader.__getitem__` catches `KeyError` and substitutes a
different window, so a missing age would otherwise corrupt the targets invisibly.

## Splits: by subject, aggregated by recording

The shipped `processed/` split has **16 subjects in both train and val**.
`make_TUAB.py` shuffles and splits subjects independently within `normal/` and
`abnormal/`, and 54 TUAB train subjects appear as both — so a subject can land in
train via its abnormal recording and val via its normal one. It is also unseeded,
so the split is not reproducible. For age this leaks badly, because a subject's
age is near-constant (measured within-subject spread: mean 1.4 years; 194 of 443
multi-file subjects have spread 0).

`labram/data/age_splits.py` rebuilds train/val from the pooled windows **by
subject**, with a fixed seed, and asserts zero overlap on both save and load —
a leaking split raises rather than passing silently. TUAB's official eval set
(`processed/test`) is left untouched. Table 3.1 gives the resulting split sizes.

**Table 3.1.** The subject-disjoint TUAB age split: subjects, recordings and
10 s windows per split, and the age (years) mean ± std.

| split | subjects | recordings | windows | age mean ± std |
|---|---|---|---|---|
| train | 1,650 | 2,145 | 294,253 | 48.9 ± 17.8 |
| val   |   412 |   556 |  75,717 | 48.1 ± 15.8 |
| test  |   251 |   274 |  36,728 | 49.9 ± 17.8 |

A pooled split stores files as `<subdir>/<name>.pkl` so each split stays a
*single* loader over one root rather than a `ConcatDataset` (which
`enable_window_ids` does not recurse into). Group-id helpers therefore derive ids
from the basename.

**Split by subject, aggregate by recording.** The corpus README warns that a
subject "might be represented more than once (with different ages)" — age is a
property of the *session*, not the subject. So `cross_validation.split_by='subject'`
prevents leakage while `evaluation.agg_case_by='recording'` keeps the prediction
unit correct. These are independent knobs; do not collapse them to the same key.

## The regression task

`nb_classes=1` already builds `nn.Linear(embed_dim, 1)` — a scalar head — and
`finetune_setup.load_finetune_checkpoint` already drops a shape-mismatched
`head.weight`/`head.bias`. **No new model code is needed.**

But `nb_classes == 1` also means "binary classification" across ~20 call sites, so
it cannot distinguish the two. `FinetuneModelConfig.task`
(`"classification"` | `"regression"`, set from `DatasetBundle.task`) is what
selects every row of Table 4.1:

**Table 4.1.** What `FinetuneModelConfig.task` switches between the
classification and the regression path.

| | classification | regression |
|---|---|---|
| criterion | BCE / cross-entropy | Huber (default), MSE or L1 — `loss.regression_loss` |
| output transform | `sigmoid` / `softmax` | none (a scalar, not a probability) |
| metrics | accuracy, ROC-AUC, PR-AUC, … | MAE, RMSE, R², Pearson/Spearman r |
| model selection | `accuracy`, higher is better | `mae`, **lower** is better |
| figures | confusion matrix, ROC/PR | predicted-vs-true scatter |
| window pooling | mean/median/max/vote/entropy | mean/median/max only |

Without that flag, `evaluate` would rebuild `BCEWithLogitsLoss` and score ages
with cross-entropy. `build_downstream_criterion` is the single dispatch point used
by both the train and eval paths.

**Target normalization.** The head is initialised with `init_scale=0.001`, so it
starts near zero and a raw target of ~49 would produce an enormous initial loss.
The loader z-scores the target with the **train split's** mean/std (carried on the
bundle as `target_stats`; Eq. (5.1)); `evaluate` de-normalizes once before computing
metrics (Eq. (6.1)), so every reported error is in **years**.

### Brain-age diagnostics

Age decoders regress toward the cohort mean: the old are predicted too young and
the young too old. Two extra metrics make that visible.

- `age_bias_slope` — least-squares slope of the residual `(pred - true)` against
  the true age, Eq. (6.7). Near 0 is good; strongly negative means regression to
  the mean dominates.
- `mae_corrected` — MAE after removing that linear bias, Eq. (6.8), i.e. the
  honest error once the effect is accounted for. Report this alongside raw MAE
  (Eq. (6.2)) when the slope is far from 0.

A model that just predicts the training mean scores MAE ≈ 14 on TUAB. Published
EEG brain-age benchmarks land around **MAE 7–8 years**, so that is the range to
aim for; anything near 14 means the target plumbing is broken.

## Loss functions

The downstream criterion is chosen by task, not by `nb_classes`:
`build_downstream_criterion(task, nb_classes, cfg)`
(`labram/losses/regression.py::build_downstream_criterion`) is the single dispatch point used by **both**
the training loop and `evaluate`, so a regression run can never silently fall back
to the classification criterion. When `task == "regression"` it returns
`build_regression_criterion` (`labram/losses/regression.py::build_regression_criterion`), selected by
`LossConfig.regression_loss` (`labram/configs/loss_config.py:48`).

**Everything is computed on the z-scored target.** The loader z-scores the age
with the train split's mean/std $(\mu,\sigma)$ before it ever reaches the model
(`labram/data/tuh_datasets.py::TUABAgeLoader._age_target`), Eq. (5.1):

$$
z_i \;=\; \frac{y_i - \mu}{\sigma},
\qquad (\mu,\sigma)=\texttt{target\_stats}\ \text{(train split; sample std)}
\tag{5.1}
$$

with $y_i$ the age in years. The statistics are taken over the $N_\text{tr}$
train windows (one entry per window, so a recording counts once per window), with
the $N_\text{tr}-1$ (sample) denominator and a floor on $\sigma$
(`tuh_datasets.py::prepare_TUAB_age_dataset`; recomputed over the selected train
samples by `window_selection.py::_train_target_stats` when a window selection is
active):

$$
\mu=\frac{1}{N_\text{tr}}\sum_{j=1}^{N_\text{tr}} y_j,
\qquad
\sigma=\max\Bigl(\sqrt{\tfrac{1}{N_\text{tr}-1}\textstyle\sum_{j=1}^{N_\text{tr}}(y_j-\mu)^{2}},\;10^{-6}\Bigr)
\quad\text{(years)}.
\tag{5.2}
$$

The scalar head predicts $\hat z_i$ in that same normalized space, so define the
per-window residual (z-score units), Eq. (5.3):

$$
r_i \;=\; \hat z_i - z_i .
\tag{5.3}
$$

For a batch of $N$ windows the three selectable losses are Eqs. (5.4)–(5.6):

* **Huber** (`nn.HuberLoss(delta=`$\delta$`)`, the default, `regression.py::build_regression_criterion`;
  $\delta=$ `loss.huber_delta` $=1.0$). Robust to the long tails of a clinical age
  distribution — quadratic near zero, linear in the tails, Eq. (5.4):

$$
\mathcal{L}_{\text{Huber}}
= \frac{1}{N}\sum_{i=1}^{N}\ell_\delta(r_i),
\qquad
\ell_\delta(r)=
\begin{cases}
\tfrac{1}{2}\,r^{2}, & |r|\le\delta,\\[4pt]
\delta\bigl(|r|-\tfrac{1}{2}\delta\bigr), & |r|>\delta.
\end{cases}
\tag{5.4}
$$

* **MSE** (`nn.MSELoss`, `regression.py::build_regression_criterion`), Eq. (5.5):

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{N}\sum_{i=1}^{N} r_i^{2}.
\tag{5.5}
$$

* **L1** (`nn.L1Loss`, `regression.py::build_regression_criterion`), Eq. (5.6):

$$
\mathcal{L}_{\text{L1}}=\frac{1}{N}\sum_{i=1}^{N} \lvert r_i\rvert .
\tag{5.6}
$$

Two opt-in options replace or reweight these: the soft-label KL loss,
Eq. (12.8), and the age-balanced weighting of Eqs. (5.4)–(5.6), Eq. (13.8).

Because $r_i$ is in **z-score units**, the Huber knee $\delta=1.0$ sits at one
standard deviation of age — with TUAB's $\sigma\approx 17.8$ years, the
quadratic→linear transition is at ≈ 17.8 years of error, not 1 year. Raise
`loss.huber_delta` to widen the quadratic region, lower it to make the loss more
L1-like. The loss value therefore stays $O(1)$ regardless of the age scale; the
**metrics** de-normalize (Eq. (6.1)) so their numbers read in years. For contrast,
the classification branch returns `BCEWithLogitsLoss` / `CrossEntropyLoss`
(`labram/losses/classification.py:14`) — never used for age.

## Evaluation metrics

Computed by `regression_metrics_fn` (`labram/utils/regression_metrics.py:81`) on
**de-normalized** arrays: `evaluate` undoes the z-scoring once on the gathered
predictions/targets (`denormalize`, `regression_metrics.py:154`; Eq. (6.1))
before any metric runs, so every number in Eqs. (6.2)–(6.9) is in **years**:

$$
\hat y_i=\hat z_i\,\sigma+\mu,
\qquad
y_i=z_i\,\sigma+\mu
\quad\text{(years; } \mu,\sigma \text{ from Eq. (5.2))}.
\tag{6.1}
$$

Let $\hat y_i,\,y_i$ be the de-normalized prediction/target of item $i$ (a window,
or a recording after the pooling of Eqs. (7.1)–(7.2)), $N$ the number of items,
$\bar y=\tfrac1N\sum_i y_i$, $\bar{\hat y}=\tfrac1N\sum_i \hat y_i$, the residual
$e_i=\hat y_i-y_i$ (years) and $\bar e=\tfrac1N\sum_i e_i$.

* **MAE** — the model-selection metric (lower is better), Eq. (6.2):

$$
\text{MAE}=\frac{1}{N}\sum_{i=1}^{N}\lvert e_i\rvert .
\tag{6.2}
$$

* **MSE / RMSE** (years², years), Eq. (6.3):

$$
\text{MSE}=\frac{1}{N}\sum_i e_i^{2},
\qquad
\text{RMSE}=\sqrt{\frac{1}{N}\sum_i e_i^{2}} .
\tag{6.3}
$$

* **R²** (coefficient of determination; `0` when $\text{SS}_\text{tot}=0$), Eq. (6.4):

$$
R^{2}=1-\frac{\sum_i (y_i-\hat y_i)^{2}}{\sum_i (y_i-\bar y)^{2}}
      =1-\frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} .
\tag{6.4}
$$

* **Pearson $r$** (`_correlation`, `regression_metrics.py:65`; `0` when either
  series is constant or $N<2$), Eq. (6.5):

$$
r=\frac{\sum_i (\hat y_i-\bar{\hat y})(y_i-\bar y)}
        {\sqrt{\sum_i (\hat y_i-\bar{\hat y})^{2}}\,\sqrt{\sum_i (y_i-\bar y)^{2}}} .
\tag{6.5}
$$

* **Spearman $r$** — Pearson $r$, Eq. (6.5), on the **average ranks** of $\hat y$
  and $y$ (`_rank`, `regression_metrics.py:51`; ties share the mean rank). With
  the 0-based ranks the code assigns, Eq. (6.6):

$$
\rho=r\bigl(R(\hat y),\,R(y)\bigr),
\qquad
R(a)_i=\bigl\lvert\lbrace j: a_j<a_i\rbrace\bigr\rvert+\tfrac12\Bigl(\bigl\lvert\lbrace j: a_j=a_i\rbrace\bigr\rvert-1\Bigr).
\tag{6.6}
$$

* **`age_bias_slope`** — OLS slope of the residual on the true age
  (`_ols_slope`, `regression_metrics.py:71`; population moments,
  `np.cov(..., bias=True) / np.var`; `0` when $N<2$ or $\operatorname{Var}(y)=0$),
  the brain-age regression-to-the-mean diagnostic, Eq. (6.7):

$$
\beta=\frac{\operatorname{Cov}(y,e)}{\operatorname{Var}(y)}
     =\frac{\tfrac1N\sum_i (y_i-\bar y)(e_i-\bar e)}{\tfrac1N\sum_i (y_i-\bar y)^{2}}
     =r\cdot\frac{\sigma_{\hat y}}{\sigma_{y}}-1 .
\tag{6.7}
$$

  with $\sigma_{\hat y},\sigma_y$ the population standard deviations of
  Eq. (6.9). A mean-collapsed decoder ($\hat y\equiv\bar y$) gives $\beta=-1$; an
  unbiased one gives $\beta=0$. So $\beta\in[-1,0]$ in practice (regression to the
  mean shrinks $\sigma_{\hat y}$ below $\sigma_y$).

* **`mae_corrected`** — MAE after removing that linear bias, with intercept
  $\alpha=\bar e-\beta\bar y$ (`regression_metrics.py:116`, `:128`), Eq. (6.8):

$$
\text{MAE}_\text{corr}
=\frac{1}{N}\sum_{i=1}^{N}\bigl\lvert e_i-(\beta y_i+\alpha)\bigr\rvert,
\qquad
\alpha=\bar e-\beta\,\bar y .
\tag{6.8}
$$

* **`pred_mean` / `pred_std` / `target_mean` / `target_std`** — $\tfrac1N\sum\hat y$,
  $\operatorname{std}(\hat y)$, and the same for $y$, Eq. (6.9) (`np.mean`,
  `np.std` with `ddof=0`, i.e. the population std). `pred_std`→0 flags
  mean-collapse; the `target_*` pair is a constant per-split reference.

$$
\bar{\hat y}=\frac1N\sum_i\hat y_i,
\quad
\sigma_{\hat y}=\sqrt{\frac1N\sum_i\bigl(\hat y_i-\bar{\hat y}\bigr)^{2}},
\qquad
\bar y=\frac1N\sum_i y_i,
\quad
\sigma_{y}=\sqrt{\frac1N\sum_i\bigl(y_i-\bar y\bigr)^{2}} .
\tag{6.9}
$$

`NaN`/`inf` from a degenerate batch are replaced with `0.0` by `_sanitize`
(`regression_metrics.py:43`) so logging never breaks. The bundle requests
`["mae","rmse","r2","pearson_r"]` (`labram/data/bundles.py::get_dataset_bundle`); with
`evaluation.detailed_metrics=true` (the default), `regression_report`
(`regression_metrics.py:137`) additionally computes **all** of
`REGRESSION_METRIC_NAMES` (`regression_metrics.py:15`), and those extra scalars
are what get logged too. Model selection uses `best_metric_for`
(`regression_metrics.py:162`): the first `LOWER_IS_BETTER` metric — MAE, Eq. (6.2) —
minimized, versus accuracy-maximized for classification.

## Logged plots

A regression run logs its epoch-level scalars **per metric**, with one series
per split (`train`, `val`, `test`) and the epoch on the x-axis
(`train_finetune.py::_log_regression_epoch`). The classification plots, one per
split (`val`, `val_err`, `val_window`, …), are not used for regression. Table 7.1
lists the regression plots.

A recording's windows are pooled into one prediction before the case-level
metrics run (`utils/eval_metrics.py::aggregate_windows`, called from
`train_finetune.py` for both modes). For a case $c$ (a recording, per
`evaluation.agg_case_by`) with window set $W_c$, Eq. (7.1) is the mean pooling and
Eq. (7.2) the median pooling (`np.median`: the mean of the two middle values for
an even $\lvert W_c\rvert$); the case target $y_c$ is the (constant) target of its
first window:

$$
\hat y_c^{\,\text{mean}}=\frac{1}{\lvert W_c\rvert}\sum_{i\in W_c}\hat y_i
\quad\text{(years)},
\tag{7.1}
$$

$$
\hat y_c^{\,\text{median}}=\operatorname{median}\bigl\lbrace\hat y_i : i\in W_c\bigr\rbrace
\quad\text{(years)}.
\tag{7.2}
$$

On the codebook-regularized path the optimized `total_loss` is the weighted sum of
the downstream term $\mathcal{L}_\text{reg}$ (one of Eqs. (5.4)–(5.6), z-score
units) and the unweighted spectral and quantization terms, Eq. (7.3):

$$
\mathcal{L}_\text{total}
=\lambda_\text{cls}\,\mathcal{L}_\text{reg}
+\lambda_\text{amp}\,\mathcal{L}_\text{mag}
+\lambda_\text{phase}\,\mathcal{L}_\text{phase}
+\lambda_\text{emb}\,\mathcal{L}_\text{quant}.
\tag{7.3}
$$

Column → equation key for Table 7.1: `mae_*` Eq. (6.2), `rmse_*` Eq. (6.3),
`r2_*` Eq. (6.4), computed on the pairs $(\hat y_c^{\,\text{mean}},y_c)$ of
Eq. (7.1) for `*_case_mean`, $(\hat y_c^{\,\text{median}},y_c)$ of Eq. (7.2) for
`*_case_median` and on the windows themselves for `*_window`; `pearson_r`
Eq. (6.5); `age_bias_slope` Eq. (6.7); `mae_corrected` Eq. (6.8);
`prediction_stats` Eq. (6.9); `train_step` `mae` Eq. (6.2) over the windows of one
batch, de-normalized by Eq. (6.1); `loss_terms` Eqs. (5.4)–(5.6) and, on the
codebook path, Eq. (7.3).

**Table 7.1.** Epoch-level plots a regression run logs: plot name, its series and
what it shows. Errors in years; one series per split (`train` / `val` / `test`).

| plot | series | content |
|---|---|---|
| `mae_case_mean`, `rmse_case_mean`, `r2_case_mean` | `train` / `val` / `test` | windows of a recording pooled by their **mean** prediction |
| `mae_case_median`, `rmse_case_median`, `r2_case_median` | `train` / `val` / `test` | windows pooled by their **median** prediction |
| `mae_window`, `rmse_window`, `r2_window` | `train` / `val` / `test` | one prediction per window, no pooling |
| `loss_epoch` | `train` / `val` / `test` | the split's total loss (the weighted total on the codebook path) |
| `loss_terms` | `train_*` per step; `val_*`, `test_*` per epoch | every loss term in absolute units (Notes; Eqs. (5.4)–(5.6), (7.3)) |
| `pearson_r`, `age_bias_slope`, `mae_corrected` | `train` / `val` / `test` | case-level diagnostics |
| `prediction_stats` | `{split}_pred_mean`, `_pred_std`, `_target_mean`, `_target_std` | spots collapse to the mean |
| `train_step` | `mae` | running per-batch window MAE, in years |

Notes:

- **RMSE, not MSE.** MSE (Eq. (6.3); years², about 10× the other errors) is computed but
  not plotted or reported as a summary value; RMSE gives the same information
  in years.
- **Mean and median pooling** are both computed every epoch
  (`case_mean_*` / `case_median_*` keys in the stats and `log.txt`).
  `evaluation.agg_windows` still decides the primary `mae` used for model
  selection.
- **Train metrics** pool the predictions made *during* the epoch (train mode,
  weights still moving), not a separate eval pass, so `train` is a running
  estimate. Under DDP each rank pools only its own shard of windows.
- **Loss terms use one name per term on both paths.** The downstream term is
  `regression_loss` (`classifier_loss` for classification) whether the
  criterion is the plain Huber/L1/MSE or the codebook-regularized one. Which
  criterion it is lives in the config (`loss.regression_loss`).
  The codebook path adds the unweighted `magnitude_loss`, `phase_loss`,
  `quantize_loss` and their weighted `total_loss`, Eq. (7.3)
  (`losses/codebook_regularized.py::CodebookRegularizedCriterion.forward`; the
  weights are `loss.classifier_weight`, `amplitude_weight`, `phase_weight`,
  `embedding_weight`). Val/test are scored with
  the training criterion (`evaluate(..., criterion=)`, decoder included), so
  their `total_loss` and `loss` match the train definition. The val/test
  points sit at the end of each epoch on the per-step axis.
- **`r2` is unbounded below.** Early or divergent epochs can show a large
  negative R², which stretches that plot's axis for a while.

At the end of training, `runs/common.py::log_summary_tables` reports one table per
split (`summary` / `train|val|test` under ClearML PLOTS, a markdown table in
TensorBoard's TEXT tab, and the console): a `best` row (the epoch selected on val
MAE) and a `last` row (the final epoch), with every metric formatted to two
decimals, including the median-pooled `case_median_*` columns.

## Recording and window selection

Five `data.*` options pick which windows each split uses
(`labram/data/window_selection.py`). They are applied once, in
`run_finetune.main`, after the bundle is built (or a recorded split / CV fold is
applied). The defaults keep everything; `finetune_tuab_age.json` trims a minute
at each end and evaluates on 5 minutes per recording. Table 8.1 lists the options;
Eqs. (8.1)–(8.2) define which samples they keep.

**Table 8.1.** The `data.*` window-selection options: class default, the value in
`finetune_tuab_age.json` (in parentheses where it differs) and their effect.

| Option | Default (age config) | Effect |
|---|---|---|
| `case_filter` | `all` | `normal` / `abnormal` / `all`: TUAB cases used for train **and** eval |
| `trim_start_sec` / `trim_end_sec` | `0` (`60` / `60`) | Drop the first/last seconds of every recording, rounded up to whole 10 s windows |
| `window_sec` | `10` | Model input length: `window_sec/10` consecutive pickles concatenated (multiple of 10, 10–60) |
| `eval_minutes` | `0` (`5`) | Val/test use only the first N minutes after the trimmed start (`0` = whole recording); train always uses everything |

**How samples are chosen.** Window `i` of a recording covers `[10i, 10i+10)` s.
Sample starts lie on a fixed grid per recording, every `window_sec` seconds from
the trimmed start. A sample is kept when all its windows exist on disk and end
before the trimmed end (and, for val/test, within the `eval_minutes` budget).
In window units (`window_selection.py::WindowSelection`), with $T_\text{start}$,
$T_\text{end}$ = `trim_start_sec`, `trim_end_sec` (s) and $M$ = `eval_minutes`,
Eq. (8.1) gives the trimmed start $s$, the trimmed-end count $e$, the windows per
sample $k$ and the evaluation budget $B$ (windows; $B=0$ means unlimited):

$$
s=\Bigl\lceil \frac{T_\text{start}}{10}\Bigr\rceil,
\qquad
e=\Bigl\lceil \frac{T_\text{end}}{10}\Bigr\rceil,
\qquad
k=\frac{\texttt{window\_sec}}{10},
\qquad
B=\Bigl\lfloor \frac{60\,M}{10}\Bigr\rfloor .
\tag{8.1}
$$

For a recording with $n$ windows on disk (one past the highest window index),
window $i$ starts a kept sample iff all of Eq. (8.2) hold
(`window_selection.py::select_files`; the last condition only for val/test with
$B>0$), and its recording passes `case_filter`:

$$
\begin{aligned}
&i\ge s,\qquad (i-s)\bmod k=0,\qquad i+k\le n-e,\\
&\lbrace i,\dots,i+k-1\rbrace\ \text{all on disk},\qquad i+k\le s+B .
\end{aligned}
\tag{8.2}
$$

Recording length is read from the window directories, not the split's file list,
so the selection is idempotent: re-applying it to a reused `data_split.json` or a
CV fold changes nothing. Recordings too short for one sample after trimming drop
out, and the run logs per-split sample/recording counts.

**Labels.** `case_filter` reads TUAB's normal/abnormal label from the metadata
sidecar, derived from the EDF folder (`.../train/abnormal/...`). Sidecars written
before the `label` field existed must be regenerated with `scan`; the loader raises
rather than silently dropping every recording. A filtered run recomputes the age
z-scoring stats from the selected train samples (normal-only TUAB train is
younger: mean 43.7 vs 48.9). `case_filter` is rejected for classification, where
it would leave a single class.

**Longer inputs.** The pretrained time embedding has 16 rows (1 per 1 s patch).
For `window_sec > 16` the model is built with `max_time_patches = window_sec`, and
`load_finetune_checkpoint` linearly interpolates the pretrained embedding to that
length. Attention cost grows with the square of the token count (23 channels ×
seconds). Table 8.2 gives the measured peak memory on an A10G (23 GB) for one AMP
train step.

**Table 8.2.** Peak GPU memory (GB) of one AMP train step on an A10G (23 GB) by
input length `window_sec` (rows) and per-GPU batch size (columns); OOM = out of
memory.

| `window_sec` | batch 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| 10 | 0.6 GB | 1.1 | 2.2 | 4.3 |
| 30 | 3.5 | 7.0 | 13.9 | OOM |
| 60 | 12.7 | OOM | OOM | OOM |

So keep the effective batch at 64 with `trainer.update_freq`: for example
`trainer.batch_size=8 trainer.update_freq=8` at 60 s. Longer inputs are not
supported with `model.codebook_reg` (the grafted VQNSP decoder has a fixed 16-row
time embedding).

## Per-recording npy format

The window pickles (409,083 files, float64, 150.6 GB) are slow to stream and
large to copy. `dataset_maker/make_TUAB_npy.py` repacks them, unchanged in
content, into one float32 file per recording next to `processed/`:

```
edf/processed_npy/
├── recordings/<stem>.npy   # float32 [T, 23], time-major, 200 Hz, µV; T = n_windows · 2000
├── manifest.json           # format, channels, provenance; per recording: source split,
│                           # n_windows, n_samples, sha256, age, sex, normal/abnormal label
├── age_metadata.json       # labelled sidecar (copied)
└── age_split.json          # the same subject-disjoint split (copied)
```

2,990 files, about 75 GB. Window `k` of a recording is the contiguous slice
`[2000k, 2000k + 2000)`, so any crop is one read of a memory-mapped file
(`np.load(path, mmap_mode="r")[s:s+L]`). The float64 → float32 cast is the one
the loader already applied to every pickle, so the model sees bit-identical
input (verified on 1,000 sampled windows).

`finetune_tuab_age.json` reads this format by default (`data.data_format=npy`;
the class default, and every other config, stays `pickle`). Pass
`data.data_format=pickle` to read the window pickles instead. `data.data_path`
may point at `edf/` or straight at `processed_npy/`. Items keep the pickle names, so
window selection, cross-validation and split reuse work unchanged.
`data.random_crop=true` (npy only, training only) moves each training sample to
a random start within half a sample length of its grid position, inside the
trimmed range, redrawn every epoch; evaluation always uses the fixed grid.
In samples (`tuh_datasets.py`, the npy loader's `_load`): a sample of
$L=2000k$ samples on grid start $a_0=2000\,i$ in a recording whose trimmed range
is $[l,h)=[2000\,s,\;2000\,(n-e))$ (Eqs. (8.1)–(8.2); $[0,T)$ without a window
selection) starts at $a$, drawn uniformly over the integers of Eq. (9.1); when
$a_\text{hi}\le a_\text{lo}$ it stays at $a_0$:

$$
a\sim\mathcal{U}\lbrace a_\text{lo},\dots,a_\text{hi}\rbrace,
\qquad
a_\text{lo}=\max\bigl(l,\;a_0-\lfloor L/2\rfloor\bigr),
\qquad
a_\text{hi}=\min\bigl(h-L,\;a_0+\lfloor L/2\rfloor\bigr).
\tag{9.1}
$$

Building it (on-demand CPU processing jobs reading the pickles from S3 and
writing the npy files back; re-runs skip finished recordings):

```bash
python scripts/submit_tuab_npy_conversion.py --instances 4          # convert
python -m dataset_maker.make_TUAB_npy merge \
  --dst s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed_npy \
  --sidecar-dir /path/to/edf/processed --expect 2990                # manifest
```

On SageMaker the age config uses File mode (`sagemaker.input_mode=File`,
`sagemaker.volume_size_gb=150`), and `scripts/submit_age_experiments.sh`
defaults to `DATA_FORMAT=npy`: the 75 GB copy happens once per job (about
4.5 extra minutes of staging), after which every epoch reads local disk instead
of streaming the pickles from S3. On scenario D (`ml.g5.2xlarge`) this cut the
15-epoch training loop from 9:47 to 3:20 (12.5 instead of ~35 minutes per
epoch) with identical metrics, per-epoch curves and data split.

## Usage

```bash
TUAB=/path/to/TUH_Abnormal/v3.0.0/edf
```

Extract the demographics (prints a summary to cross-check against the corpus
AAREADME):

```bash
python -m dataset_maker.make_TUAB_age scan --root "$TUAB"
```

Build the subject-disjoint split:

```bash
python -m dataset_maker.make_TUAB_age split --root "$TUAB"
```

Fine-tune:

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path="$TUAB" \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth
```

Normal-only, 60 s inputs, 5-minute evaluation:

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path="$TUAB" data.case_filter=normal data.window_sec=60 \
        trainer.batch_size=8 trainer.update_freq=8 \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth
```

Cross-validated (group-disjoint by subject; see `docs/cross_validation.md`):

```bash
python -m labram.runs.finetune_cv \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path="$TUAB" cross_validation.enabled=true
```

## Scaling to TUEG

The parser is corpus-agnostic — nothing in it is TUAB-specific. TUEG v2.0.2
(26,846 sessions, ~15,000 patients, ages 2 days–106 years) uses the same header
format, so the same `scan` works unchanged and yields roughly 10× more age labels
over a far wider range:

```bash
python dataset_maker/make_TUAB_age.py scan --root /path/to/tuh_eeg/v2.0.2/edf
```

Check the printed coverage and sentinel counts before trusting a new corpus —
TUEG's older recordings are less uniformly populated than TUAB's. You will also
need window pickles for it (`dataset_maker/make_h5dataset_for_pretrain.py` or a
TUAB-style preprocessing pass), since the join is by recording stem.

## Anti-memorization options

The fine-tune memorizes recordings (train MAE ≈ 1.3 years by epoch 15 while
validation is best at epoch 1–3), and the collapsed "strong regularization"
bundle (C2/DX3) showed that dropout / weight decay / drop-path attack the
wrong thing (see `brain_age_improvement_plan.md`). Four opt-in options target
the recording → age lookup directly; all are off by default. Table 12.1 lists
them; Eqs. (12.1)–(12.9) define what each computes.

**Table 12.1.** The anti-memorization options: config keys (defaults in
parentheses) and what each does.

| Option | Config | What it does |
|---|---|---|
| **Mixup / C-Mixup** | `mixup.enabled`, `alpha` (0.4), `prob`, `sigma` (years) | Each training batch is blended with a permuted copy (`lam ~ Beta(alpha, alpha)`), inputs and targets alike, so no window maps to one recording. `sigma > 0` draws the partner with probability `exp(-(y_i - y_j)^2 / 2 sigma^2)` (C-Mixup: similar ages). Regression only; mixed batches are excluded from the per-case train report. |
| **EMA evaluation** | `optimizer.model_ema=true`, `model_ema_decay`, `evaluation.use_ema=true` | Val/test are scored, and the best epoch selected, with the EMA weights (`model_ema` key in the checkpoints; `scripts/eval_age_by_cohort.py` picks it up). Use a decay matched to the run length: 0.9995 ≈ 2,000 steps ≈ half an epoch. |
| **LoRA** | `model.lora.enabled`, `rank` (16), `alpha` (32), `targets`, `train_prefixes` | Freezes the pretrained weights and trains a rank-`r` update on `qkv`/`proj`/`fc1`/`fc2` of every block plus the head and its norms (~10 % of the parameters at r = 16). Adaptation at every depth, but in a subspace too small to encode thousands of recordings. Use a higher LR (1e-3) and `optimizer.layer_decay=1.0`. Checkpoints keep the base keys and add `lora_A`/`lora_B`. |
| **Soft labels** | `loss.regression_loss=soft_label`, `soft_label_sigma` (2.5), `_min`, `_max`, `_bin_width` | The head predicts a distribution over 1-year age bins; the target is a Gaussian over the bins and the loss their KL divergence (SFCN, Peng et al. 2021). The scalar prediction is the expectation, so every metric is unchanged. Not supported with `model.codebook_reg`. |

**Mixup / C-Mixup** (`labram/train/mixup.py::mixup_batch`). With probability
`prob` a training batch is mixed; one $\lambda\sim\operatorname{Beta}(\alpha,\alpha)$
is drawn per batch and the same $\lambda$ and partner $\pi(i)$ mix the input and
the z-scored target, Eq. (12.1):

$$
\tilde x_i=\lambda\,x_i+(1-\lambda)\,x_{\pi(i)},
\qquad
\tilde z_i=\lambda\,z_i+(1-\lambda)\,z_{\pi(i)} .
\tag{12.1}
$$

With `sigma` $\le 0$, $\pi$ is a uniform random permutation (`torch.randperm`;
$\pi(i)=i$ is possible). With $\sigma_\text{C}=$ `sigma` $>0$ (years; divided by
$\sigma$ of Eq. (5.2) because the loop sees z-scored targets) the partner is drawn
from Eq. (12.2) (`mixup_partners`), the $10^{-12}$ keeping every off-diagonal
entry positive:

$$
P\bigl(\pi(i)=j\bigr)\;\propto\;
\begin{cases}
\exp\!\Bigl(-\dfrac{(y_i-y_j)^{2}}{2\,\sigma_\text{C}^{2}}\Bigr)+10^{-12}, & j\ne i,\\[6pt]
0, & j=i.
\end{cases}
\tag{12.2}
$$

**EMA evaluation** (timm `ModelEma.update`, called from
`optim_factory.py::optimizer_update` on every optimizer update, not every
micro-step). Every entry of the state dict follows Eq. (12.3), $d$ =
`model_ema_decay`; its time constant is $\tau=1/(1-d)$ updates, so $d=0.9995$
averages over $\tau=2{,}000$ steps:

$$
\bar\theta_t=d\,\bar\theta_{t-1}+(1-d)\,\theta_t,
\qquad
\tau=\frac{1}{1-d}\ \text{(optimizer updates)}.
\tag{12.3}
$$

**LoRA** (`labram/layers/lora.py::LoRALinear`). Each targeted linear layer keeps its
frozen weight $W_0$ and bias $b$ and adds a trainable rank-$r$ update scaled by
$\alpha/r$, Eq. (12.4), with $A\in\mathbb{R}^{r\times d_\text{in}}$
(Kaiming-uniform init), $B\in\mathbb{R}^{d_\text{out}\times r}$ (zero init, so
training starts at the pretrained model) and dropout on the LoRA branch only:

$$
h=W_0x+b+\frac{\alpha}{r}\,B\,A\,\operatorname{dropout}(x).
\tag{12.4}
$$

**Learning rate.** The LoRA row's `optimizer.layer_decay` enters the
per-parameter-group LR of Eq. (12.5), the warmup epoch of the short schedule
(4 epochs, 1 warmup epoch) the schedule of Eq. (12.6)
(`optim_factory.py::apply_lr_wd_schedule`, scales from
`run_finetune.py` + `optim_factory.py::get_num_layer_for_vit`): $L$ transformer
blocks, layer id $l=0$ for `cls_token`, `pos_embed` and `patch_embed.*`, $l=j+1$
for block $j$, $l=L+1$ for everything else (head, final norms, and also
`time_embed`); with `layer_decay` $\ge 1$ every scale is 1. The
`model.codebook_reg` path uses its own per-component scales instead.

$$
\eta_g(t)=\eta(t)\cdot \texttt{layer\_decay}^{\,L+1-l(g)} .
\tag{12.5}
$$

$\eta(t)$ is the default `sched=cosine` schedule of Eq. (12.6)
(`utils/training.py::cosine_scheduler`) over $T=E\cdot S$ optimizer steps ($E$
epochs, $S$ steps per epoch), with $W$ = `warmup_epochs`$\cdot S$ steps of linear
warmup from $0$ (`np.linspace(0, lr, W)`; `optimizer.warmup_lr` is not passed to
it), $\eta_0$ = `lr` and $\eta_\text{min}$ = `min_lr`:

$$
\eta(t)=
\begin{cases}
\eta_0\,\dfrac{t}{W-1}, & 0\le t<W,\\[8pt]
\eta_\text{min}+\tfrac12\,(\eta_0-\eta_\text{min})\Bigl(1+\cos\dfrac{\pi\,(t-W)}{T-W}\Bigr), & W\le t<T.
\end{cases}
\tag{12.6}
$$

**Soft labels** (`labram/losses/regression.py::SoftLabelRegressionLoss`). The
$K$ bin centers are $c_k=c_\text{min}+k\,\Delta$ (years), $k=0,\dots,K-1$,
$K=\operatorname{round}\bigl((c_\text{max}-c_\text{min})/\Delta\bigr)+1$ (89 bins
for the defaults 1–89 y, $\Delta=1$ y), z-scored like the target:
$\tilde c_k=(c_k-\mu)/\sigma$, $\tilde s=s/\sigma$ with $s$ =
`soft_label_sigma` (years). The soft label $q_{ik}$ and the head's distribution
$p_{ik}$ over the logits $o_{ik}$ are Eq. (12.7):

$$
q_{ik}=\operatorname{softmax}_k\Bigl(-\frac{(\tilde c_k-z_i)^{2}}{2\,\tilde s^{2}}\Bigr),
\qquad
p_{ik}=\operatorname{softmax}_k(o_{ik}).
\tag{12.7}
$$

The loss is their KL divergence, summed over bins and averaged over the $N$
windows of the batch (bins with $q_{ik}=0$ contribute 0), Eq. (12.8):

$$
\mathcal{L}_\text{soft}=\frac1N\sum_{i=1}^{N}\sum_{k=1}^{K} q_{ik}\bigl(\log q_{ik}-\log p_{ik}\bigr).
\tag{12.8}
$$

The scalar prediction the metrics score (`predict`) is the expectation over the
bins, Eq. (12.9); de-normalized by Eq. (6.1) it is $\sum_k p_{ik}c_k$ in years:

$$
\hat z_i=\sum_{k=1}^{K} p_{ik}\,\tilde c_k .
\tag{12.9}
$$

Short-schedule runs of each (4 epochs, 1 warmup epoch) are `M1`–`M4` in
`scripts/submit_age_experiments.sh` (M1 runs locally).

## Regression to the mean and the age-balanced loss

Predictions shrink toward the train mean (bias slope, Eq. (6.7), ≈ −0.4 on
val/test, −0.2 even on train). `notebooks/age_error_analysis.ipynb` (on
`labram/eval/age_analysis.py` + `labram/eval/age_plots.py`; open it with the
"LaBraM (.venv)" kernel) shows that the model is already calibrated: regressing age on prediction gives
slope ≈ 1 ($a'$ in Eq. (13.3)). No deployable post-hoc correction lowers the MAE. Cole's inversion
(Eq. (13.2)) flattens the bias but raises test MAE from 8.45 to 9.94, and the age-level
correction (Eq. (13.4)) needs the true age.

The corrections (`age_analysis.py::fit_bias_correction` /
`apply_bias_correction`) are fitted on val by ordinary least squares
(`np.polyfit(..., 1)`), Eq. (13.1): $\hat y$ on $y$ gives slope $a$ and intercept
$b$; $y$ on $\hat y$ gives $a'$ and $b'$ (all predictions and ages in years):

$$
\hat y\approx a\,y+b,
\qquad
y\approx a'\,\hat y+b' .
\tag{13.1}
$$

Cole's inversion (`"cole"`, Cole et al. 2018) uses only the prediction, Eq. (13.2):

$$
\hat y^{\,\text{cole}}=\frac{\hat y-b}{a} .
\tag{13.2}
$$

Recalibration (`"recalibrate"`) is the reverse regression, Eq. (13.3):

$$
\hat y^{\,\text{recal}}=a'\,\hat y+b' .
\tag{13.3}
$$

The age-level correction (`"age_level"`, de Lange & Cole 2020) subtracts the fitted
bias at the true age, Eq. (13.4), so it is not a deployable age estimate:

$$
\hat y^{\,\text{age}}=\hat y-(a-1)\,y-b .
\tag{13.4}
$$

Moving the tails has to happen in training, with the option in Table 13.1:

**Table 13.1.** The age-balanced loss option: config keys (defaults in
parentheses) and what it does.

| Option | Config | What it does |
|---|---|---|
| **Age-balanced loss** | `loss.balance` (`none` / `sqrt_inverse` / `inverse`), `balance_bin_width` (1 y), `balance_lds_sigma` (2 y), `balance_max_weight` (10) | Weights each window's mse/l1/huber term by the inverse (square root) of the train-window age density, LDS-smoothed (Yang et al. 2021), clipped at 10× the smallest weight and renormalized to mean 1. On TUAB `sqrt_inverse` gives children up to ~7.7×, 75–85 y ~1.2–1.6×, the 40–60 bulk ~0.8×. Train-only: val/test losses stay unweighted. Not with `soft_label` or `model.codebook_reg`. |

The weights (`labram/losses/regression.py::age_balance_weights`) are per age bin
of width $w$ = `balance_bin_width` (years), built from the raw ages $y_j$ of the
$N_\text{tr}$ train windows. Eq. (13.5) defines the bins and their window counts
$n_k$ ($\mathbf{1}\lbrace\cdot\rbrace$ the indicator); ages outside the range clamp to the
end bins:

$$
\ell=w\Bigl\lfloor\frac{\min_j y_j}{w}\Bigr\rfloor,
\quad
K=\Bigl\lfloor\frac{\max_j y_j-\ell}{w}\Bigr\rfloor+1,
\quad
b(y)=\operatorname{clip}\Bigl(\Bigl\lfloor\frac{y-\ell}{w}\Bigr\rfloor,0,K-1\Bigr),
\quad
n_k=\sum_{j=1}^{N_\text{tr}}\mathbf{1}\lbrace b(y_j)=k\rbrace.
\tag{13.5}
$$

LDS smoothing (Yang et al. 2021) convolves the counts with a normalized Gaussian
kernel truncated at $\pm H$ bins, $s=\sigma_\text{LDS}/w$ with $\sigma_\text{LDS}$ =
`balance_lds_sigma` (years), $H=\max(1,\operatorname{round}(3s))$, zero padding
($n_k=0$ outside $0\le k<K$), Eq. (13.6); with $\sigma_\text{LDS}=0$,
$\tilde n_k=n_k$:

$$
\tilde n_k=\sum_{m=-H}^{H} g_m\,n_{k+m},
\qquad
g_m=\frac{\exp\bigl(-m^{2}/(2s^{2})\bigr)}{\sum_{m'=-H}^{H}\exp\bigl(-m'^{2}/(2s^{2})\bigr)} .
\tag{13.6}
$$

Empty bins are floored at the smallest positive density, raised to the power
$-p$ ($p=1$ for `inverse`, $p=\tfrac12$ for `sqrt_inverse`), clipped at
$M$ = `balance_max_weight` times the smallest weight and renormalized to mean 1
over the train windows, Eq. (13.7):

$$
d_k=\max\bigl(\tilde n_k,\ \min_{k':\,\tilde n_{k'}>0}\tilde n_{k'}\bigr),
\quad
u_k=\min\bigl(d_k^{-p},\ M\min_{k'} d_{k'}^{-p}\bigr),
\quad
\omega_k=\frac{u_k}{\tfrac{1}{N_\text{tr}}\sum_{j=1}^{N_\text{tr}}u_{b(y_j)}} .
\tag{13.7}
$$

The training loss weights the element-wise term $\ell(r_i)$ of Eq. (5.4), (5.5) or
(5.6) ($\ell_\delta(r)$, $r^2$ or $\lvert r\rvert$) by the bin of the window's
target age in years, $y_i=z_i\sigma+\mu$ (the mixed age $\tilde z_i\sigma+\mu$ of
Eq. (12.1) under mixup), and averages over the batch (`BalancedRegressionLoss`),
Eq. (13.8). Val/test rebuild the criterion without train targets, so their loss is
the unweighted Eq. (5.4)–(5.6):

$$
\mathcal{L}_\text{bal}=\frac1N\sum_{i=1}^{N}\omega_{b(y_i)}\,\ell(r_i) .
\tag{13.8}
$$

Result on the short schedule (4 epochs, no mixup; E2 baseline val 7.68 / test
8.25): `sqrt_inverse` (B1) val 7.83 / test 8.47, `inverse` (B2) 7.96 / 8.57. The
bias slope shrinks (val −0.37 → −0.32 / −0.29) and the tails move toward the
diagonal (test 70–79 predicted 65.4 instead of 61.8 with B2; 10–19 predicted 26.4
instead of 29.6). Per-decade MAE (Eq. (6.2) over the items whose true age is in
$[10m,10m+10)$ years, `age_analysis.py::error_by_age_bin`) improves in the tails and worsens in the dense
40–59 range, so overall MAE goes up. Use it when age-independent bias matters
more than the lowest MAE.

## Artifact-aware window pooling

The worst under-predictions are recordings full of muscle (EMG) artifact (65 y
predicted 40, 78 y predicted 44): high-frequency power reads as "young".
`scripts/age_artifact_pooling.py` re-pools a trained run's cached window
predictions after dropping or down-weighting EMG-heavy windows. The model is not
re-run, so it takes minutes on CPU:

```bash
python scripts/age_artifact_pooling.py \
  --data-path /data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf \
  --run M1=checkpoints/age_M1_... --run M2=path/to/extracted/M2 \
  --out artifact_pooling.json
```

Table 14.1 summarizes the pooling procedure; Eqs. (14.1)–(14.5) define it
(`labram/eval/age_analysis.py`). For window $i$ with CAR-referenced signal on
channels $c=1,\dots,C$, let $P_{ic}(f)$ be its Welch PSD (`scipy.signal.welch`,
200 Hz, 400-sample = 2 s segments, 0.5 Hz bins; µV²/Hz). The relative EMG power of
a channel is the sum of the PSD bins in $[30,45)$ Hz over the bins in $[1,45)$ Hz
(`band_powers`), Eq. (14.1):

$$
\rho_{ic}=\frac{\sum_{f\in[30,\,45)}P_{ic}(f)}{\sum_{f\in[1,\,45)}P_{ic}(f)}
\quad\text{(dimensionless, }0\le\rho_{ic}\le 1\text{)}.
\tag{14.1}
$$

The window scores are its channel mean and channel maximum
(`window_artifact_features`), Eq. (14.2):

$$
\texttt{emg}_i=\frac1C\sum_{c=1}^{C}\rho_{ic},
\qquad
\texttt{emg\_max}_i=\max_{c}\,\rho_{ic} .
\tag{14.2}
$$

With score $q_i$ (`emg` or `emg_max`) and threshold $\vartheta$, each window gets a
pooling weight (`artifact_weights`), Eq. (14.3); the smooth weight is 0.5 at the
threshold:

$$
w_i^{\text{reject}}=\mathbf{1}\lbrace q_i\le\vartheta\rbrace,
\qquad
w_i^{\text{weight}}=\frac{1}{1+(q_i/\vartheta)^{4}} .
\tag{14.3}
$$

The $m$ = `--min-keep` cleanest windows of each recording (lowest $q_i$, ties by
order) are raised to weight $\ge 1$, $w_i'=\max(w_i,1)$, the others keep
$w_i'=w_i$, and the recording prediction is the weighted mean of its window
predictions in years (`artifact_pooled`), Eq. (14.4); with $\vartheta=\infty$ it is
the plain mean of Eq. (7.1):

$$
\hat y_c^{\,\text{art}}=\frac{\sum_{i\in W_c} w_i'\,\hat y_i}{\sum_{i\in W_c} w_i'} .
\tag{14.4}
$$

The threshold for grid point $\kappa$ is the $\kappa$-quantile (pandas
`Series.quantile`, linear interpolation) of the score over all **val** windows,
applied unchanged to test, with $\kappa=1$ meaning no rejection
(`artifact_pooling_sweep`), Eq. (14.5); the setting with the lowest val recording
MAE, Eq. (6.2) on Eq. (14.4), is selected (`select_pooling`):

$$
\vartheta_\kappa=
\begin{cases}
Q_\kappa\bigl(\lbrace q_i : i\in\text{val windows}\rbrace\bigr), & \kappa<1,\\
\infty, & \kappa=1,
\end{cases}
\qquad
\kappa\in\lbrace 0.5,0.7,0.8,0.9,0.95,0.97,0.99,1\rbrace.
\tag{14.5}
$$

**Table 14.1.** The pooling procedure.

| Piece | Choice |
|---|---|
| Artifact index | `emg` = relative 30–45 Hz power over 1–45 Hz per window (CAR applied), channel mean; `--score emg_max` uses the channel maximum (focal temporal/frontal EMG). The band stops below 60 Hz: TUAB is notched at 50 Hz but recorded on 60 Hz mains. |
| Pooling | `reject` drops windows above the threshold; `weight` scales them by `1 / (1 + (emg / thr)^4)`. Every recording keeps its `--min-keep` (1) cleanest windows, so none drops out. |
| Threshold | Quantiles 0.5–0.99 of `emg` over **val** windows, applied unchanged to test. The setting is selected on val MAE; its test MAE is the estimate to report. Quantile 1.0 is the plain-mean baseline. |
| Caches | `<run>/analysis/window_predictions.parquet` (from the notebook; `--predict` scores the model when missing) and `<run>/analysis/window_artifacts.parquet` (per-window features). |

The functions (`window_artifact_features`, `artifact_pooled`,
`artifact_pooling_sweep`, `select_pooling`) live in `labram/eval/age_analysis.py`.

**Result: no gain.** Table 14.2 covers the six local runs (val/test recording MAE, Eq. (6.2) on Eq. (14.4), in years,
`emg` score, `min_keep` 1). For five of the six, the validation set selects **no rejection**
(quantile 1.0, the plain mean). The one exception, M1c at q = 0.95, gains 0.01 y on val and loses
0.04 y on test. The best test MAE over the whole grid, which is an optimistic,
test-selected bound, is only 0.02–0.16 y below the baseline. Aggressive rejection (keep the
cleanest 50% of windows) raises val MAE by about 1.1–1.5 y. EMG-heavy recordings are
artifact-laden throughout, so dropping their windows leaves nothing cleaner to average. Robust
inputs or artifact-aware training are the remaining options; pooling is not.

**Table 14.2.** Artifact-aware pooling per run: baseline (mean pooling) vs. the val-selected
setting, and the best test MAE anywhere on the grid.

| run | baseline val / test | val-selected setting | selected val / test | best test on grid |
|---|---|---|---|---|
| M1 mixup | 7.46 / 8.45 | reject, q = 1.00 | 7.46 / 8.45 | 8.37 |
| M1c C-Mixup | 7.54 / 8.23 | reject, q = 0.95 | 7.53 / 8.27 | 8.21 |
| M12 mixup + EMA | 7.48 / 8.45 | reject, q = 1.00 | 7.48 / 8.45 | 8.38 |
| L1 mixup + EMA, 12 ep | 7.51 / 8.47 | reject, q = 1.00 | 7.51 / 8.47 | 8.41 |
| B1 balance sqrt-inv | 7.83 / 8.47 | reject, q = 1.00 | 7.83 / 8.47 | 8.38 |
| B2 balance inverse | 7.96 / 8.57 | reject, q = 1.00 | 7.96 / 8.57 | 8.41 |

## Next steps

The default config (`finetune_tuab_age.json`) is scenario D of the October 2026
ablation: common average reference on the input, 15 epochs at lr 1e-4. Its full
technical description (loss, feature building, optimization, evaluation) is in
[`age_training_scenario_D.md`](age_training_scenario_D.md) (PDF:
`age_training_scenario_D.pdf`, rebuilt with `scripts/md_to_pdf.py`). The ranked
improvement plan is in [`brain_age_improvement_plan.md`](brain_age_improvement_plan.md).

## Files

Table 16.1 maps each file to its role.

**Table 16.1.** The files that implement age regression and its analysis.

| Path | Role |
|---|---|
| `labram/data/tuh_metadata.py` | EDF-header parser, sidecar I/O, age lookup |
| `labram/data/age_splits.py` | subject-disjoint split + leakage assertions |
| `labram/data/window_selection.py` | case filter, trimming, multi-window samples, eval budget |
| `labram/data/tuh_datasets.py` | `TUABAgeLoader`, `prepare_TUAB_age_dataset` |
| `labram/data/bundles.py` | `TUAB_AGE` bundle, `task` / `target_stats` |
| `labram/losses/regression.py` | criterion selection + `build_downstream_criterion`, age-balanced loss |
| `labram/eval/age_analysis.py` | per-window/recording predictions, error by age, bias correction, spectral features, artifact-aware pooling |
| `scripts/age_artifact_pooling.py` | re-score runs with EMG-aware window pooling (threshold chosen on val) |
| `labram/eval/age_plots.py` | age-scale plots, real-EEG comparisons, `AgeExplorer` (interactive browser) |
| `notebooks/age_error_analysis.ipynb` | error analysis of the best run, with real EEG of accurate / too-young / too-old recordings |
| `labram/utils/regression_metrics.py` | MAE/RMSE/R²/r + brain-age diagnostics |
| `dataset_maker/make_TUAB_age.py` | `scan` / `split` CLI |
| `labram/configs/defaults/finetune_tuab_age.json` | ready-to-run config |
