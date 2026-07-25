# Cross-validation fine-tuning

K-fold cross-validation is an opt-in fine-tuning mode. It partitions the data
into `n_folds` **group-disjoint** folds (grouped by subject / recording so the
same case never straddles train/val/test), trains each fold as its own
sub-experiment whose name and output folder embed the fold number, records the
fold partition as a reproducible `cv_split.json` artifact, and aggregates
evaluation metrics across all folds (mean ± std) with optional ClearML logging.

It is **off by default** (`cross_validation.enabled = false`): existing
fine-tune runs behave exactly as before.

## Quick start

```bash
# Run all 5 folds in one process (local study):
python -m labram.runs.finetune_cv \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set data.data_path=/data/TUAB \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth

# Run a single fold (one process / job per fold — the pattern used for SageMaker):
python -m labram.runs.finetune_cv \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set cross_validation.fold=2 data.data_path=/data/TUAB

# Aggregate the fold results afterwards and (optionally) log to ClearML:
python -m labram.eval.cv_report --base_dir ./checkpoints/finetune_tuab_cv5
```

## Configuration (`cross_validation`)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `n_folds` | `5` | Number of folds K. |
| `fold` | `-1` | `-1` → iterate every fold in-process; `>=0` → train only that fold (one job per fold). |
| `split_by` | `subject` | Group unit that never straddles splits: `subject` / `recording` / `window`. |
| `shuffle` | `true` | Shuffle groups before partitioning. |
| `seed` | `42` | Deterministic fold assignment. |
| `pool` | `train_val` | Which data is re-partitioned: `train_val` (keep the original test set out of CV) or `all`. |
| `split_json` | `''` | Reuse a saved `cv_split.json` so separately-dispatched fold jobs share an identical partition. |
| `base_dir` | `''` | Base folder holding the fold sub-runs; `''` → derived from `output.output_dir`. Each fold lives in `<base_dir>/fold_<k>`. |

## How folds are built

The pool (train+val, or all splits) is grouped by `split_by`; groups are shuffled
(seeded) and split into K contiguous chunks. For fold *k*:

- **test** = fold *k*
- **val** = fold *(k+1) mod K*
- **train** = the remaining folds

So every case is used for testing exactly once across the K folds, and no subject
appears in more than one of a fold's train/val/test sets (verified by a leakage
self-check). With `pool = train_val` the dataset's original test split is left
untouched for a separate final evaluation.

## Experiment naming ("base folder + fold number")

All folds of a study share a base experiment name (`<name>_cv<K>`, e.g.
`finetune_tuab_cv5`). Each fold sub-run gets:

- **Output dir** `‹base_dir›/fold_<k>/` (with `log/` inside).
- **ClearML** project sub-folder `‹project›/‹experiment›` and task name
  `fold_<k>`, tagged `cross-validation`, the experiment name, and `fold_<k>` — so
  the folds group together in the ClearML UI under one folder and are trivially
  sortable by fold.

## Artifacts

Written to the CV base folder and each fold dir, and uploaded to ClearML when
tracking is on:

- **`cv_split.json`** (base folder + copied into each fold dir) — the fold
  partition: per-fold group ids, counts, the held-out test summary, and the
  train/val/test convention. This is the reproducibility record; pass it back via
  `cross_validation.split_json` to reproduce the exact partition.
- **`data_split.json`** (per fold) — that fold's concrete train/val/test case
  assignment (the existing per-run artifact).
- **`fold_metrics.json`** (per fold) — the fold's best-epoch val/test metrics.
- **`cv_summary.json`** (base folder) — metrics aggregated over all folds.

## Aggregating results across folds

`labram.eval.cv_aggregation` collects the per-fold `fold_metrics.json` files (or
pulls the folds' scalars from ClearML) and computes, for every metric,
`mean / std / min / max` and the per-fold values across folds. An in-process
all-folds run aggregates automatically at the end; otherwise run the report CLI:

```bash
# From local fold outputs:
python -m labram.eval.cv_report --base_dir ./checkpoints/finetune_tuab_cv5

# From a ClearML project sub-folder, logging the summary as a cv_summary task:
python -m labram.eval.cv_report \
  --clearml_project LaBraM/finetune_tuab_cv5 --log_clearml
```

The summary is saved to `cv_summary.json` and, when ClearML is enabled, logged to
a parent `cv_summary` task in the study's project folder as `cv_<split>_mean` /
`cv_<split>_std` scalars plus a metrics table.

## Running on SageMaker

Each fold is naturally its own training job — see [`sagemaker.md`](sagemaker.md).
`labram.runs.submit_sagemaker` submits one job per fold (job name
`‹prefix›-fold-‹k›`), all sharing one uploaded `cv_split.json` so the partition
is identical across jobs.
