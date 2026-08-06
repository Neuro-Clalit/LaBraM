# Logging & ClearML experiment tracking

LaBraM emits two kinds of log output:

1. **Human-readable status logs** — model/config summaries, per-iteration progress,
   epoch averages, warnings/errors — via Python's `logging` module.
2. **Metric streams** — scalars (loss, accuracy, LR, …) — to TensorBoard and,
   optionally, to [ClearML](https://clear.ml).

Both are opt-in on top of the existing TensorBoard behavior: nothing changes for
an existing run unless `clearml.enabled` is set.

## Python logging

All framework messages go through the shared `labram` logger. Modules obtain it
with `labram.utils.get_logger(__name__)` and log through it instead of calling
`print`.

`runs/common.py::setup_environment` configures the logger once, after the
distributed rank is known, via `labram.utils.configure_logging`:

- **Rank-aware.** Rank 0 logs at `INFO`; other ranks are raised to `WARNING`, so
  a distributed job does not multiply per-step chatter across processes.
- **Console + file.** A stdout handler is always attached. On rank 0, when the
  run has a `log_dir` (or `output_dir`), a `run.log` file handler is added there.
- **Idempotent.** Re-running `configure_logging` replaces its own handlers rather
  than stacking them, and leaves foreign handlers untouched.

```python
from labram.utils import get_logger
logger = get_logger(__name__)
logger.info("Creating model: %s", model_name)
```

## Metric writers

The per-phase training loops write metrics through a single `log_writer` object.
`runs/common.py::create_log_writer` builds it on rank 0:

| Configured sinks            | `log_writer` returned         |
|-----------------------------|-------------------------------|
| `log_dir` only              | `TensorboardLogger`           |
| ClearML only                | `ClearMLLogger`               |
| both                        | `MultiWriter([tb, clearml])`  |
| none / non-rank-0           | `None`                        |

`MultiWriter` fans every `set_step` / `update` / `update_image` / `flush` call
out to each underlying writer, so the training loops stay unchanged whether one
or both backends are active. `ClearMLLogger` maps Tensorboard's `head/name`
scalar convention onto ClearML's `(title, series)` pairs.

### Scalar plot grouping (scale hygiene)

Scalars are grouped by `head` (the ClearML plot *title* / TensorBoard tag group).
Series that share a plot share a single (normalized) y-axis, so a series whose
values are `>> 1` would flatten small-valued series plotted next to it. To keep
each plot on a comparable scale, series are grouped by magnitude:

| Plot (`head`)   | Series                                             | Scale        |
|-----------------|----------------------------------------------------|--------------|
| `loss`          | `loss`, `class_acc`, per-component loss shares      | O(1)         |
| `opt`           | `lr`, `min_lr`, `weight_decay`                      | ≤ ~1         |
| `grad`          | `grad_norm` (+ per-component grad-norm shares)      | can be `>> 1`|
| `scale`         | `loss_scale` (AMP dynamic scale, up to ~65536)     | `>> 1`       |
| `val` / `test` / `train`      | `accuracy`, `f1`, `roc_auc`, … , `loss` | [0, 1] / O(1)|
| `val_cm` / `test_cm` / `train_cm` | `cm_tn`, `cm_fp`, `cm_fn`, `cm_tp` (counts) | `>> 1`  |
| `val_window` / `test_window`  | per-window (per-crop) copy of the above | as above     |

Confusion-matrix cell **counts** and the AMP **loss scale** / **gradient norm**
therefore live on their own plots rather than squashing the normalized metrics
on the `val`/`test`/`opt` plots.

### Relative (scale-free) metrics

Metrics are reported in **relative** terms by default, so plots from different
runs can be read — and overlaid — without first normalizing them by hand. Two
`LoggingConfig` options control this; both are on by default and both *replace*
the absolute form rather than adding a second series next to it.

| Option (`logging.…`)         | Default | Effect                                                            |
|------------------------------|---------|-------------------------------------------------------------------|
| `relative_loss_components`   | `true`  | Per-component losses / gradient norms are logged as their **share of the component total** |
| `relative_step_axis`         | `true`  | Every metric is plotted against **normalized training progress**   |
| `relative_step_scale`        | `1000`  | x-axis units the full run spans (progress in per-mille)            |

**Relative loss components.** When a run has a composite loss (the
codebook-regularized fine-tune's `classifier` / `magnitude` / `phase` /
`quantize` terms, or VQNSP's `rec` / `rec_angle` / `quant` terms), the writer
receives `‹name›_loss_rel = |‹name›| / Σ|components|` — a `[0, 1]` series where
all components sum to 1 — instead of the raw magnitude. The same applies to the
per-component gradient norms (`grad_norm_‹name›_rel`) logged when
`evaluation.log_grad_components` is on. This shows how the terms *trade off*,
which is comparable across loss weights, datasets and runs, where the raw
magnitudes are not.

The **aggregate** loss (`loss`, VQNSP's `total_loss`) keeps its absolute value —
it is the quantity being minimized — as do non-loss counters such as VQNSP's
`unused_code`. Pre-training reports a single total loss and so has no component
breakdown to relativize. The console `MetricLogger`, the per-epoch `log.txt`
lines and the checkpoint/summary stats all keep the **raw magnitudes**; only the
metric writers switch to shares.

**Relative x-axis.** Instead of the raw global iteration, metrics are reported
at `round(progress × relative_step_scale)`, where `progress ∈ [0, 1]` is the
fraction of the run completed (`total_steps = epochs × steps_per_epoch ×
update_freq`). Two runs with different dataset sizes, batch sizes or epoch
counts then land on the same 0…1000 axis and can be compared directly in the
ClearML/TensorBoard overlay. Per-epoch metrics (`val`/`test`) follow the same
axis: epoch *e* of *E* reports at `(e + 1) / E`, exactly where the per-iteration
series stands at that moment, so the train and eval curves stay aligned.

The wiring is `runs/common.py::configure_relative_step_axis` (called by each
phase's `train_loop`, which knows the run length) plus the writer-side mapping in
`utils/logging.py`. Writers that are never configured — e.g. the offline
evaluation toolkit — stay on the absolute axis.

To get the old absolute logging back:

```bash
python -m labram.runs.finetune --config labram/configs/defaults/finetune_tuab.json \
  --set logging.relative_loss_components=false logging.relative_step_axis=false
```

### Per-window vs. per-case metrics

Fine-tuning evaluates on ~10 s crop windows, but a clinical decision is per EEG
*case* (recording or subject). When `evaluation.agg_windows` is set (the shipped
`finetune_*` configs default to `mean`), the eval loop reports **both**: the
per-case metrics (primary, on the `val`/`test` plots, pooling every window of a
case via `agg_windows`) *and* the per-window metrics (mirrored under `window_*`
keys, on the `val_window`/`test_window` plots). `evaluation.agg_case_by` selects
whether a "case" is a `recording` or a `subject`.

### Final metrics for cross-experiment comparison

Per-epoch metrics are logged as scalar *series* (good for curves, but in
ClearML's **Compare** view they show as plots, not a clean side-by-side table).
So at the end of training the run's **best-epoch** eval metrics are additionally
recorded as ClearML **single values** via `report_single_value` — `best_epoch`,
`val_<metric>`, `test_<metric>` (accuracy, balanced_accuracy, f1, roc_auc,
pr_auc, …). ClearML collects single values into the experiment's SCALARS
"Summary" and renders them as a **side-by-side table when experiments are
compared**, which is exactly what you want to compare runs (or CV folds) at a
glance. The same flat dict is also `connect`-ed to the task as a `final_metrics`
config section, so the values appear as **sortable columns in the experiments
table** and in the hyperparameter comparison.

This is handled by `runs/common.py::log_summary_metrics` (called from
`run_finetune.main`), so every fine-tune — including each cross-validation fold —
gets a comparable final-metrics table. (A CV study additionally logs a
`cv_summary` task with the across-fold mean ± std table; see
[`cross_validation.md`](cross_validation.md).)

## Enabling ClearML

ClearML is an **optional dependency**. If the `clearml` package is not installed,
a run with `clearml.enabled=true` logs a warning and continues with
TensorBoard-only logging — training is never blocked.

Install it and configure credentials once:

```bash
pip install clearml
clearml-init   # paste your server credentials
```

Then enable tracking through the config. Every `*RunConfig` carries a `clearml`
section (`labram.configs.train_config.ClearMLConfig`):

| Field                     | Default   | Meaning                                                        |
|---------------------------|-----------|----------------------------------------------------------------|
| `enabled`                 | `false`   | Master switch for ClearML tracking.                            |
| `project_name`            | `LaBraM`  | ClearML project the task is filed under.                       |
| `task_name`               | `""`      | Task name; empty ⇒ derived from `output_dir` (or model name).  |
| `append_timestamp`        | `true`    | Append a millisecond timestamp (`YYYYmmdd_HHMMSS_fff`) to the task name so each run is uniquely identifiable. |
| `tags`                    | `[]`      | Tags added to the task. Two more are added automatically: `debug` for debug runs, and `sagemaker` when `sagemaker.enabled` is also set. |
| `output_uri`              | `""`      | Artifact upload target; empty ⇒ ClearML default.               |
| `offline`                 | `false`   | Run without a server, storing results locally.                 |
| `continue_last_task`      | `false`   | Resume/append to the previous task instead of creating a new.  |
| `auto_connect_frameworks` | `true`    | Let ClearML auto-hook frameworks (PyTorch, TensorBoard, …).    |

The full run config is connected to the task (under `run_config`) for
reproducibility. Only rank 0 initializes the task.

When both `clearml.enabled` and `sagemaker.enabled` are true, the task also gets
a `sagemaker` tag (`labram.runs.common.SAGEMAKER_TAG`), so managed-training runs
filter apart from local ones in the experiments table. The flag travels with the
config into the container, so the tag is applied by the run itself, not by the
submitting machine; a CV study's `cv_summary` task is tagged the same way.

### Examples

```bash
# Pre-train with ClearML tracking on, tagged.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.pretrain \
  --config labram/configs/defaults/pretrain.json \
  --set clearml.enabled=true \
        clearml.project_name="LaBraM/pretrain" \
        clearml.tags='["base","8k-vocab"]'

# Fine-tune, offline mode (no ClearML server required).
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.finetune \
  --config labram/configs/defaults/finetune_tuab.json \
  --set clearml.enabled=true clearml.offline=true
```
