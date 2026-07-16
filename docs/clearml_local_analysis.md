# Local ClearML experiment analysis

Pull a finished (or running) **ClearML experiment** down to a local, plain-data
snapshot and run heuristic analysis over it to surface **concrete, actionable
insights** — overfitting, under-training, divergence, learning-rate-schedule
problems, gradient instability, AMP loss-scale collapse, class imbalance — so a
run can be diagnosed offline (in a notebook, a script, or a Claude session)
without opening the ClearML web UI.

This complements the existing model loader (`labram.eval.load_clearml_assets`,
see [`docs/finetune_evaluation.md`](finetune_evaluation.md)): that fetches the
**checkpoint / config / data-split** to *rebuild and evaluate the model*; this
fetches the **metrics / hyperparameters / console** to *diagnose the training
run*.

All logic lives in **`labram.eval.clearml_analysis`** (unit-tested in
`tests/test_clearml_analysis.py`). The ClearML fetch is best-effort and isolated
behind the optional `clearml` dependency; the analysis and reporting operate on
the pure `ExperimentSnapshot`, so they are fully testable and reusable offline.

## Quick start (CLI)

```bash
# Write snapshot.json + report.md for a task into ./analysis/
python -m labram.eval.clearml_report --task-id <TASK_ID> --output-dir ./analysis/

# Or resolve by project + task name, and print the report to stdout
python -m labram.eval.clearml_report \
  --project-name "LaBraM/finetune" --task-name "tuab-base" --print
```

The written **`report.md`** leads with the insights (each with a
recommendation and supporting evidence), followed by a metric summary table,
the hyperparameters, the artifact/model list, and the console tail. Point Claude
at that file (or the raw `snapshot.json`) to reason about what to fix next.

## Quick start (Python / notebook)

```python
from labram.eval import (
    load_clearml_experiment, analyze_experiment,
    render_report, save_experiment_report,
)

snapshot = load_clearml_experiment(task_id="…")     # or project_name=…, task_name=…
insights = analyze_experiment(snapshot)
for ins in insights:
    print(f"[{ins.severity}] {ins.category}: {ins.message}")
    print("   →", ins.recommendation)

# Or in one shot, also writing the report to disk:
from labram.eval import load_and_analyze
snapshot, insights = load_and_analyze(task_id="…", output_dir="./analysis/")
print(render_report(snapshot, insights))
```

## What is fetched

`load_clearml_experiment` builds an `ExperimentSnapshot` with:

| Field | Source | Notes |
| ----- | ------ | ----- |
| `task_id` / `task_name` / `project_name` / `status` / `tags` / `comment` | task metadata | |
| `created` / `started` / `completed` | `task.data` timestamps | |
| `hyperparameters` | `task.get_parameters_as_dict()` | flattened to `section/key` |
| `scalars` | `task.get_reported_scalars()` | `{title/series: ScalarSeries}` with full iteration/value history |
| `console_tail` | `task.get_reported_console_output()` | last N lines |
| `artifacts` / `models` | registered artifact + output-model names | |

Every section is fetched independently and best-effort: anything ClearML cannot
provide (or that a given server/version does not expose) is logged and left
empty, so a partial experiment still yields a usable snapshot. The snapshot is
JSON round-trippable via `ExperimentSnapshot.save_json` / `.load_json`.

## Insights produced

`analyze_experiment` returns a list of `Insight(severity, category, message,
recommendation, evidence)`, most-severe first. The analysers key off the
metric-series names the trainer logs (`train/loss`, `val/loss`,
`val/accuracy`, `val/balanced_accuracy`, `opt/lr`, `opt/grad_norm`,
`opt/loss_scale`, …) and degrade to no-ops when a needed series is absent.

| Category | Trigger | Severity |
| -------- | ------- | -------- |
| `run-status` | task ended `failed` / `aborted` | critical |
| `divergence` | NaN/Inf in any series (critical if in a loss) | critical / warning |
| `overfitting` | validation loss climbs > 10 % after its minimum | warning |
| `generalization-gap` | train-vs-val headline metric gap > 0.15 | warning |
| `checkpoint-selection` | best val metric well before the final epoch | info |
| `undertraining` | train loss still falling at > 30 % of its initial rate | info |
| `plateau` | val metric flat over the final third of epochs | info |
| `lr-schedule` | LR barely decayed, or collapsed to ~0 early | info / warning |
| `grad-instability` | grad-norm peak > 20× the median | warning |
| `amp-loss-scale` | AMP loss scale dropped > 1000× | warning |
| `class-imbalance` | accuracy exceeds balanced accuracy by > 0.10 | warning |

These are deliberately simple, explainable heuristics meant to *flag where to
look*, not to replace judgement — each insight ships with the concrete evidence
(epochs, values, ratios) behind it.

## Requirements

ClearML is an optional dependency (`pip install clearml`, then `clearml-init`
once). Only the fetch step (`load_clearml_experiment` / the CLI) needs it; the
analysis, reporting, and snapshot (de)serialisation work on any
`ExperimentSnapshot` with no ClearML installed. See
[`docs/logging_clearml.md`](logging_clearml.md) for how runs publish to ClearML
in the first place.
