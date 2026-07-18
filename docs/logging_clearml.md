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
| `loss`          | `loss`, `class_acc`, per-component losses          | O(1)         |
| `opt`           | `lr`, `min_lr`, `weight_decay`                      | ≤ ~1         |
| `grad`          | `grad_norm` (+ per-component grad norms)            | can be `>> 1`|
| `scale`         | `loss_scale` (AMP dynamic scale, up to ~65536)     | `>> 1`       |
| `val` / `test` / `train`      | `accuracy`, `f1`, `roc_auc`, … , `loss` | [0, 1] / O(1)|
| `val_cm` / `test_cm` / `train_cm` | `cm_tn`, `cm_fp`, `cm_fn`, `cm_tp` (counts) | `>> 1`  |
| `val_window` / `test_window`  | per-window (per-crop) copy of the above | as above     |

Confusion-matrix cell **counts** and the AMP **loss scale** / **gradient norm**
therefore live on their own plots rather than squashing the normalized metrics
on the `val`/`test`/`opt` plots.

### Per-window vs. per-case metrics

Fine-tuning evaluates on ~10 s crop windows, but a clinical decision is per EEG
*case* (recording or subject). When `evaluation.agg_windows` is set (the shipped
`finetune_*` configs default to `mean`), the eval loop reports **both**: the
per-case metrics (primary, on the `val`/`test` plots, pooling every window of a
case via `agg_windows`) *and* the per-window metrics (mirrored under `window_*`
keys, on the `val_window`/`test_window` plots). `evaluation.agg_case_by` selects
whether a "case" is a `recording` or a `subject`.

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
| `tags`                    | `[]`      | Tags added to the task.                                        |
| `output_uri`              | `""`      | Artifact upload target; empty ⇒ ClearML default.               |
| `offline`                 | `false`   | Run without a server, storing results locally.                 |
| `continue_last_task`      | `false`   | Resume/append to the previous task instead of creating a new.  |
| `auto_connect_frameworks` | `true`    | Let ClearML auto-hook frameworks (PyTorch, TensorBoard, …).    |

The full run config is connected to the task (under `run_config`) for
reproducibility. Only rank 0 initializes the task.

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
