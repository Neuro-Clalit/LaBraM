# Fine-tune evaluation toolkit & notebook

Apply a **trained fine-tune model** to a validation / test split and analyse its
predictions offline: aggregate per-window predictions into per-EEG-case
predictions, compare aggregation methods, sweep the decision threshold, study
the entropy↔accuracy relationship, and inspect individual cases.

All logic lives in the reusable **`labram.eval`** package (unit-tested in
`tests/test_eval_toolkit.py`); the notebook
[`notebooks/finetune_evaluation.ipynb`](../notebooks/finetune_evaluation.ipynb)
only orchestrates and visualizes. The notebook ships with a **synthetic demo**
(`USE_SYNTHETIC = True`) so it runs end-to-end with no real data or checkpoint.

## `labram.eval` at a glance

| Module | Responsibility |
| ------ | -------------- |
| `loading.py` | Resolve a run's **checkpoint + run config + `data_split.json`** from a local directory (`find_experiment_assets`) or a **ClearML experiment** (`load_clearml_assets`); rebuild the architecture (`build_finetune_model` / `build_model_by_name`), load weights (`load_checkpoint_weights`), and build a sequential, case-id-aware eval loader (`build_case_loader`). |
| `inference.py` | `collect_predictions(model, loader, …)` → a `PredictionResult` with per-window `probs`, `logits`, `targets`, case `groups`, `files`, plus a `.entropy` property. |
| `aggregation.py` | Per-case pooling (`aggregate_result`), method comparison (`compare_aggregations`, `comparison_table`), `entropy_accuracy_curve`, `selective_accuracy`, `rank_cases`, `case_window_predictions`. |
| `plots.py` | Matplotlib figure builders (metric-vs-threshold, aggregation comparison, ROC/PR overlay, entropy/accuracy, selective prediction, per-threshold confusion matrices, EEG trace, per-window predictions). Each returns `None` if matplotlib is absent. |

### 0 · Loading a trained model

```python
from labram import eval as leval

# (a) local run directory
assets = leval.find_experiment_assets("./checkpoints/finetune_tuab_base")
# (b) or a ClearML experiment
assets = leval.load_clearml_assets(task_id="…")            # or project_name/task_name

model, config, data_split = leval.load_model_from_assets(
    assets, device="cpu", model_name="labram_base_patch200_200", nb_classes=1)
```

`load_model_from_assets` restores the exact architecture from the saved
`run_config.yaml` when present, otherwise falls back to `model_name` +
`nb_classes`. The ClearML loader fetches the registered `OutputModel`
checkpoint, the `run_config` configuration, and the `data_split` artifact
uploaded by the trainer (see [`docs/logging_clearml.md`](logging_clearml.md) and
[`docs/training_enhancements.md`](training_enhancements.md)).

### 0b · Per-window vs per-case metrics

Window aggregation reuses the exact same `labram.utils.aggregate_windows` used
inside the training/eval loop (`evaluation.agg_windows`), so notebook numbers
match a `trainer.eval=true` run. `compare_aggregations` reports the window-level
baseline alongside each case-level method:

```python
loader, files = leval.build_case_loader(dataset, agg_case_by="recording")
result = leval.collect_predictions(model, loader, device, is_binary=True, nb_classes=1, files=files)
reports = leval.compare_aggregations(result, modes=("mean", "median", "entropy"))
```

### 1 · Aggregation methods

`aggregate_windows` (and thus `evaluation.agg_windows`) now supports:

| mode | pooling |
| ---- | ------- |
| `mean` | average probabilities (softmax-averaged logits for multiclass) |
| `median` | per-case median probability — robust to a few outlier windows |
| `vote` | majority vote over hard per-window predictions |
| `max` | max positive probability / any-window-positive |
| `entropy` | **confidence-weighted mean**: weight each window by `1 − normalized_entropy`, so confident windows dominate and uncertain ones are discounted |

### 2 · Metric curves

`labram.utils.threshold_sweep(y_score, y_true)` returns accuracy / balanced
accuracy / F1 / precision / recall / specificity as a function of the decision
threshold; `plots.metric_vs_threshold_figure`, `plots.confusion_matrices_grid`
and `plots.roc_pr_overlay_figure` render them. ROC/PR points and AUCs come from
the existing `classification_report`.

### 3 · Entropy vs accuracy

`labram.utils.prediction_entropy` gives the normalized (`[0, 1]`) Shannon
entropy per prediction (binary and multiclass). `entropy_accuracy_curve` bins
windows by entropy and measures accuracy per bin — a calibrated model is more
accurate on low-entropy windows.

### 4 · Effectiveness of the entropy label

`selective_accuracy` treats entropy as an abstention signal: sort windows most-
to-least confident and, at each coverage level, keep only the most-confident
fraction. Rising accuracy as coverage drops means entropy usefully flags
unreliable predictions.

### 5 · Case inspection

`rank_cases` orders cases from most confidently-correct to most confidently-
wrong; `case_window_predictions` returns a case's per-window probs / entropy /
files. `plots.eeg_case_figure` draws the raw multi-channel EEG and
`plots.window_prediction_figure` the per-window prediction trace.

## Running the notebook on a real experiment

1. Open `notebooks/finetune_evaluation.ipynb`.
2. Set `USE_SYNTHETIC = False`.
3. Point at a run — `LOCAL_DIR` **or** `CLEARML_TASK_ID` (or `CLEARML_PROJECT` +
   `CLEARML_TASK_NAME`).
4. Set `DATASET` / `DATA_PATH` / `SPLIT` (`val` or `test`) and run all cells.

The analyses are dataset-agnostic (binary TUAB and multiclass TUEV).

## Diagnosing the training run (not just the model)

The loaders above fetch a run's **checkpoint / config / data-split** to rebuild
and evaluate the *model*. To instead diagnose the *training run* — its metric
curves, hyperparameters, and console — pull the ClearML experiment into a local
snapshot and get concrete heuristic insights (overfitting, divergence,
LR-schedule issues, …):

```bash
python -m labram.eval.clearml_report --task-id <TASK_ID> --output-dir ./analysis/
```

See [`docs/clearml_local_analysis.md`](clearml_local_analysis.md).
