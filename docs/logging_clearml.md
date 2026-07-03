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
