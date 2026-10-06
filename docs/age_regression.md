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
(`processed/test`) is left untouched.

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
selects:

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
bundle as `target_stats`); `evaluate` de-normalizes once before computing metrics,
so every reported error is in **years**.

### Brain-age diagnostics

Age decoders regress toward the cohort mean: the old are predicted too young and
the young too old. Two extra metrics make that visible.

- `age_bias_slope` — least-squares slope of the residual `(pred - true)` against
  the true age. Near 0 is good; strongly negative means regression to the mean
  dominates.
- `mae_corrected` — MAE after removing that linear bias, i.e. the honest error
  once the effect is accounted for. Report this alongside raw MAE when the slope
  is far from 0.

A model that just predicts the training mean scores MAE ≈ 14 on TUAB. Published
EEG brain-age benchmarks land around **MAE 7–8 years**, so that is the range to
aim for; anything near 14 means the target plumbing is broken.

## Loss functions

The downstream criterion is chosen by task, not by `nb_classes`:
`build_downstream_criterion(task, nb_classes, cfg)`
(`labram/losses/regression.py:36`) is the single dispatch point used by **both**
the training loop and `evaluate`, so a regression run can never silently fall back
to the classification criterion. When `task == "regression"` it returns
`build_regression_criterion` (`labram/losses/regression.py:16`), selected by
`LossConfig.regression_loss` (`labram/configs/loss_config.py:48`).

**Everything is computed on the z-scored target.** The loader z-scores the age
with the train split's mean/std $(\mu,\sigma)$ before it ever reaches the model
(`labram/data/tuh_datasets.py:205`):

$$
z_i \;=\; \frac{y_i - \mu}{\sigma},
\qquad (\mu,\sigma)=\texttt{target\_stats}\ \text{(train split; sample std)}
$$

with $y_i$ the age in years. The scalar head predicts $\hat z_i$ in that same
normalized space, so define the per-window residual

$$
r_i \;=\; \hat z_i - z_i .
$$

For a batch of $N$ windows the three selectable losses are:

* **Huber** (`nn.HuberLoss(delta=`$\delta$`)`, the default, `regression.py:31`;
  $\delta=$ `loss.huber_delta` $=1.0$). Robust to the long tails of a clinical age
  distribution — quadratic near zero, linear in the tails:

$$
\mathcal{L}_{\text{Huber}}
= \frac{1}{N}\sum_{i=1}^{N}\ell_\delta(r_i),
\qquad
\ell_\delta(r)=
\begin{cases}
\tfrac{1}{2}\,r^{2}, & |r|\le\delta,\\[4pt]
\delta\bigl(|r|-\tfrac{1}{2}\delta\bigr), & |r|>\delta.
\end{cases}
$$

* **MSE** (`nn.MSELoss`, `regression.py:27`):

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{N}\sum_{i=1}^{N} r_i^{2}.
$$

* **L1** (`nn.L1Loss`, `regression.py:29`):

$$
\mathcal{L}_{\text{L1}}=\frac{1}{N}\sum_{i=1}^{N} \lvert r_i\rvert .
$$

Because $r_i$ is in **z-score units**, the Huber knee $\delta=1.0$ sits at one
standard deviation of age — with TUAB's $\sigma\approx 17.8$ years, the
quadratic→linear transition is at ≈ 17.8 years of error, not 1 year. Raise
`loss.huber_delta` to widen the quadratic region, lower it to make the loss more
L1-like. The loss value therefore stays $O(1)$ regardless of the age scale; the
**metrics** de-normalize (below) so their numbers read in years. For contrast,
the classification branch returns `BCEWithLogitsLoss` / `CrossEntropyLoss`
(`labram/losses/classification.py:14`) — never used for age.

## Evaluation metrics

Computed by `regression_metrics_fn` (`labram/utils/regression_metrics.py:81`) on
**de-normalized** arrays: `evaluate` undoes the z-scoring once on the gathered
predictions/targets (`denormalize`, `regression_metrics.py:154`;
$\text{value}\cdot\sigma+\mu$) before any metric runs, so every number below is in
**years**. Let $\hat y_i,\,y_i$ be the de-normalized prediction/target,
$\bar y=\tfrac1N\sum_i y_i$, and the residual $e_i=\hat y_i-y_i$.

* **MAE** — the model-selection metric (lower is better):

$$
\text{MAE}=\frac{1}{N}\sum_{i=1}^{N}\lvert e_i\rvert .
$$

* **MSE / RMSE**:

$$
\text{MSE}=\frac{1}{N}\sum_i e_i^{2},
\qquad
\text{RMSE}=\sqrt{\frac{1}{N}\sum_i e_i^{2}} .
$$

* **R²** (coefficient of determination; `0` when $\text{SS}_\text{tot}=0$):

$$
R^{2}=1-\frac{\sum_i (y_i-\hat y_i)^{2}}{\sum_i (y_i-\bar y)^{2}}
      =1-\frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} .
$$

* **Pearson $r$** (`_correlation`, `regression_metrics.py:65`; `0` when either
  series is constant or $N<2$):

$$
r=\frac{\sum_i (\hat y_i-\bar{\hat y})(y_i-\bar y)}
        {\sqrt{\sum_i (\hat y_i-\bar{\hat y})^{2}}\,\sqrt{\sum_i (y_i-\bar y)^{2}}} .
$$

* **Spearman $r$** — Pearson $r$ on the **average ranks** of $\hat y$ and $y$
  (`_rank`, `regression_metrics.py:51`; ties share the mean rank).

* **`age_bias_slope`** — OLS slope of the residual on the true age
  (`_ols_slope`, `regression_metrics.py:71`), the brain-age regression-to-the-mean
  diagnostic:

$$
\beta=\frac{\operatorname{Cov}(y,e)}{\operatorname{Var}(y)}
     =r\cdot\frac{\sigma_{\hat y}}{\sigma_{y}}-1 .
$$

  A mean-collapsed decoder ($\hat y\equiv\bar y$) gives $\beta=-1$; an unbiased one
  gives $\beta=0$. So $\beta\in[-1,0]$ in practice (regression to the mean shrinks
  $\sigma_{\hat y}$ below $\sigma_y$).

* **`mae_corrected`** — MAE after removing that linear bias, with intercept
  $\alpha=\bar e-\beta\bar y$ (`regression_metrics.py:116`, `:128`):

$$
\text{MAE}_\text{corr}
=\frac{1}{N}\sum_{i=1}^{N}\bigl\lvert e_i-(\beta y_i+\alpha)\bigr\rvert .
$$

* **`pred_mean` / `pred_std` / `target_mean` / `target_std`** — $\tfrac1N\sum\hat y$,
  $\operatorname{std}(\hat y)$, and the same for $y$. `pred_std`→0 flags
  mean-collapse; the `target_*` pair is a constant per-split reference.

`NaN`/`inf` from a degenerate batch are replaced with `0.0` by `_sanitize`
(`regression_metrics.py:43`) so logging never breaks. The bundle requests
`["mae","rmse","r2","pearson_r"]` (`labram/data/bundles.py:61`); with
`evaluation.detailed_metrics=true` (the default), `regression_report`
(`regression_metrics.py:137`) additionally computes **all** of
`REGRESSION_METRIC_NAMES` (`regression_metrics.py:15`), and those extra scalars
are what get logged too. Model selection uses `best_metric_for`
(`regression_metrics.py:162`): the first `LOWER_IS_BETTER` metric — MAE —
minimized, versus accuracy-maximized for classification.

## Validity of the metrics' active range in the logging

To keep a large-magnitude series from flattening a small one on a shared axis,
`_log_eval_stats` (`labram/train/train_finetune.py:517`) routes each scalar to one
of three plots by its expected range:

| plot (`head` suffix) | key set (`train_finetune.py`) | intended range | regression members |
|---|---|---|---|
| `{head}` | `_LOGGED_EVAL_RATE_KEYS` (`:485`) | $\sim[-1,1]$ + `loss` ($O(1)$) | `r2`, `pearson_r`, `spearman_r`, `age_bias_slope` |
| `{head}_err` | `_LOGGED_EVAL_ERROR_KEYS` (`:493`) | target units (years, $O(10)$) | `mae`, `rmse`, `mse`, `mae_corrected`, `pred_mean`, `pred_std`, `target_mean`, `target_std` |
| `{head}_cm` | `_LOGGED_EVAL_COUNT_KEYS` (`:500`) | integer counts | classification only |

Every one of the 12 `REGRESSION_METRIC_NAMES` lands in exactly one group, so
nothing is dropped. Checking each against its **actual** range:

| metric | theoretical range | typical (TUAB age) | plot | on-scale? |
|---|---|---|---|---|
| `pearson_r`, `spearman_r` | $[-1,1]$ | 0.4–0.8 | rate | ✓ |
| `age_bias_slope` | $[-1,0]$ typ. ($-1$=mean-collapse) | $-0.7\ldots-0.2$ | rate | ✓ |
| `r2` | $(-\infty,\,1]$ | 0–0.6 | rate | ⚠ unbounded below |
| `mae`, `rmse`, `mae_corrected` | $[0,\infty)$, years | 7–18 | err | ✓ |
| `pred_mean`, `target_mean` | $\sim[1,89]$ years | ≈ 49 | err | ✓ |
| `pred_std`, `target_std` | $[0,\infty)$, years | ≈ 15–18 | err | ✓ |
| `mse` | $[0,\infty)$, years² | 80–350 | err | ⚠ one order above the others |
| `loss` (z-score Huber) | $[0,\infty)$, $O(1)$ | 0.1–0.5 | rate | ✓ |

The grouping is sound for a trained model, with **two ranges worth flagging**:

1. **`mse` shares `{head}_err` with the year-scale series.** MSE is in years²
   ($\approx\text{RMSE}^2$, so $O(10^2)$), an order of magnitude above `mae` /
   `rmse` / `mae_corrected` and the year-scale location/scale stats ($O(10)$). On a
   shared linear axis it visually dominates and squashes them. MSE is not in the
   default/bundle metric list, but `detailed_metrics=true` computes and logs it,
   and it is monotone-redundant with RMSE anyway — so either drop it from the error
   plot or give it its own `head`. This is a plot-readability issue, not a
   correctness bug.

2. **`r2` on the rate plot is unbounded below.** `pearson_r`, `spearman_r` and
   `age_bias_slope` are all bounded to $\sim[-1,1]$, but $R^2\to$ large-negative
   when the model does worse than predicting the mean (early or divergent epochs).
   A single very-negative $R^2$ auto-scales the shared axis and flattens its
   bounded companions for that plot. For a converged model $R^2\in[0,1]$ and there
   is no problem — the risk is transient, and `_sanitize` only guards `NaN`/`inf`,
   not large finite magnitudes.

Everything else sits correctly in its band: the four rate metrics are genuinely in
$[-1,1]$ for a trained model, and the eight error-plot series are all in year (or
year-derived) units. The per-step training curve is consistent — it logs the
running per-batch window MAE under `head="train_step"`, computed in years via
`denormalize` before the meter update.

### The `err` plot and per-case train metrics

`err` holds one series per split — `train`, `val`, `test` — each the epoch's
**per-case** MAE: the window predictions of a recording are pooled by
`evaluation.agg_windows` (`mean` by default) before scoring, so the three curves
compare like with like. Train gets case ids exactly like val/test
(`enable_window_ids` runs on all three splits), and `train_one_epoch` pools the
predictions it made *during* the epoch — train mode, weights still moving —
rather than running a separate eval pass. So `train` is a running estimate, and
under DDP each rank pools only its own shard of windows. The per-window train
metrics are mirrored under `train_window` / `train_window_err`, as for val/test.

Each loss term is logged in absolute units on `loss_terms` every step: the
criterion's single term (e.g. `huber_loss`) on the plain path, or the unweighted
components plus `total_loss` with the codebook-regularized criterion.

At the end of training, `runs/common.py::log_summary_tables` reports one table per
split (`summary` / `train|val|test` under ClearML PLOTS, a markdown table in
TensorBoard's TEXT tab, and the console): a `best` row (the epoch selected on val
MAE) and a `last` row (the final epoch), with every metric formatted to two
decimals.

## Recording and window selection

Five `data.*` options pick which windows each split uses
(`labram/data/window_selection.py`). They are applied once, in
`run_finetune.main`, after the bundle is built (or a recorded split / CV fold is
applied). The defaults keep everything; `finetune_tuab_age.json` trims a minute
at each end and evaluates on 5 minutes per recording.

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
seconds). Measured peak memory on an A10G (23 GB) for one AMP train step:

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

Select it with `data.data_format=npy` (default `pickle`); `data.data_path` may
point at `edf/` or straight at `processed_npy/`. Items keep the pickle names, so
window selection, cross-validation and split reuse work unchanged.
`data.random_crop=true` (npy only, training only) moves each training sample to
a random start within half a sample length of its grid position, inside the
trimmed range, redrawn every epoch; evaluation always uses the fixed grid.

Building it (on-demand CPU processing jobs reading the pickles from S3 and
writing the npy files back; re-runs skip finished recordings):

```bash
python scripts/submit_tuab_npy_conversion.py --instances 4          # convert
python -m dataset_maker.make_TUAB_npy merge \
  --dst s3://eeg-data-public/TUH_Abnormal/v3.0.0/edf/processed_npy \
  --sidecar-dir /path/to/edf/processed --expect 2990                # manifest
```

On SageMaker use File mode with it (`scripts/submit_age_experiments.sh` does so
for `DATA_FORMAT=npy`): the 75 GB copy happens once per job, after which every
epoch reads local disk instead of streaming ~108 GB from S3.

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

## Next steps

The default config (`finetune_tuab_age.json`) is scenario D of the October 2026
ablation: common average reference on the input, 15 epochs at lr 1e-4. Its full
technical description (loss, feature building, optimization, evaluation) is in
[`age_training_scenario_D.md`](age_training_scenario_D.md) (PDF:
`age_training_scenario_D.pdf`, rebuilt with `scripts/md_to_pdf.py`). The ranked
improvement plan is in [`brain_age_improvement_plan.md`](brain_age_improvement_plan.md).

## Files

| Path | Role |
|---|---|
| `labram/data/tuh_metadata.py` | EDF-header parser, sidecar I/O, age lookup |
| `labram/data/age_splits.py` | subject-disjoint split + leakage assertions |
| `labram/data/window_selection.py` | case filter, trimming, multi-window samples, eval budget |
| `labram/data/tuh_datasets.py` | `TUABAgeLoader`, `prepare_TUAB_age_dataset` |
| `labram/data/bundles.py` | `TUAB_AGE` bundle, `task` / `target_stats` |
| `labram/losses/regression.py` | criterion selection + `build_downstream_criterion` |
| `labram/utils/regression_metrics.py` | MAE/RMSE/R²/r + brain-age diagnostics |
| `dataset_maker/make_TUAB_age.py` | `scan` / `split` CLI |
| `labram/configs/defaults/finetune_tuab_age.json` | ready-to-run config |
