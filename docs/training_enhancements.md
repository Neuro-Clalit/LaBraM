# Training & evaluation enhancements

This document describes the configurable training/evaluation features added on
top of the base LaBraM pipeline: gradient clipping, selectable LR schedulers,
detailed classification metrics, per-loss gradient logging, inference-time
window aggregation, post-training machine shutdown, and explicit ClearML model
artifact upload.

All settings are plain config fields, so they work via a JSON/YAML `--config`
file or `--set key=value` overrides on any runner.

## 1. Gradient clipping (`optimizer.clip_grad`)

Gradient clipping is an optimizer parameter (`OptimizerConfig.clip_grad`,
`float | null`). `null` disables clipping (the AMP scaler still *measures* the
grad norm for logging); a float clips the global grad norm to that value.

```
--set optimizer.clip_grad=1.0
```

Previously the pre-training and VQNSP loops used `clip_grad or 0`, which passed
`0` to `clip_grad_norm_` when unset — clipping the norm to zero and thereby
**zeroing every gradient**. All three phases now treat `null` as "no clipping",
matching the fine-tune loop.

## 2. Learning-rate scheduler (`optimizer.sched`)

The LR schedule after warmup is selectable via `optimizer.sched`:

| `sched`     | behaviour                                              | extra params |
| ----------- | ------------------------------------------------------ | ------------ |
| `cosine`    | cosine annealing `lr → min_lr` (default; historical)   | —            |
| `constant`  | hold `lr`                                              | —            |
| `linear`    | linear decay `lr → min_lr`                             | —            |
| `step`      | multiply by `decay_rate` every `decay_epochs` epochs   | `decay_epochs`, `decay_rate` |
| `multistep` | multiply by `decay_rate` at each `decay_milestones` epoch | `decay_milestones`, `decay_rate` |

All policies share the same linear warmup (`warmup_epochs` / `warmup_steps`,
`warmup_lr`) and are floored at `min_lr` for `step`/`multistep`.

```
--set optimizer.sched=multistep optimizer.decay_milestones='[20,40]' optimizer.decay_rate=0.1
```

## 3. Detailed classification metrics (`evaluation.*`, fine-tuning)

`EvaluationConfig` (on `FinetuneRunConfig` as `evaluation`) enables a richer
report for **train, val and test**:

- Scalars: `accuracy`, `balanced_accuracy`, `precision`, `recall`/`sensitivity`,
  `specificity`, `f1`, `f1_weighted`, `roc_auc`, `pr_auc`, `cohen_kappa`, the
  binary confusion-matrix cells as **absolute counts** (`cm_tn/cm_fp/cm_fn/cm_tp`)
  **and as relative rates** (`cm_tn_rate/cm_fp_rate/cm_fn_rate/cm_tp_rate`,
  row-normalized per true class — `cm_tp_rate` is the TPR/sensitivity,
  `cm_tn_rate` the TNR/specificity, `cm_fp_rate` the FPR, `cm_fn_rate` the FNR).
  The rates are comparable across class-imbalanced splits where the raw counts
  are not.
- Confusion matrix: native in ClearML, markdown table in the TensorBoard *Text*
  tab (`evaluation.log_confusion_matrix`).
- ROC and precision-recall curves (binary): matplotlib figures pushed to both
  backends (`evaluation.log_curves`; degrades to scalars only if matplotlib is
  absent).
- Plot history: by default each epoch's confusion-matrix / ROC / PR figure is
  logged under its own series (`confusion_matrix/epoch_003`, …) so **every epoch
  is retained and viewable** instead of the last one overwriting the rest
  (ClearML's *Plots* tab and a single-series figure keep only the latest
  iteration). Set `evaluation.plot_per_epoch=false` to keep a single rolling
  series (fewer entries; only the final epoch shown). The per-epoch scalars
  (including the confusion-matrix cells above) are unaffected — they always plot
  as one curve over epochs.

```
--set evaluation.detailed_metrics=true evaluation.log_confusion_matrix=true \
      evaluation.log_curves=true evaluation.plot_per_epoch=true
```

Train metrics are accumulated across the epoch (no extra forward pass) and
reported under the `train` head; val/test under `val`/`test`.

## 4. Per-loss-component gradient norms (`evaluation.log_grad_components`)

For the codebook-regularized criterion (classification + amplitude + phase +
quantization), the gradient norm each loss component contributes can be logged
under the `grad` head. It costs one extra backward per component, so it is
opt-in and periodic:

```
--set evaluation.log_grad_components=true evaluation.log_grad_freq=50
```

The total grad norm continues to be logged every step regardless.

## 5. Inference-time window aggregation (`evaluation.agg_windows`)

In eval-only runs (`trainer.eval=true`), per-window predictions can be pooled
into one prediction per EEG **case** before metrics are computed:

- `evaluation.agg_windows`: `none` (window-level, default), `mean` (average
  probabilities), `vote` (majority), `max` (max probability / any-positive).
- `evaluation.agg_case_by`: `recording` (default) or `subject`.

The case id is parsed from the pickle filename (recording = filename minus the
trailing window index; subject = the leading token) and threaded through the
TUH dataset loaders only when aggregation is requested.

```
--set trainer.eval=true evaluation.agg_windows=mean evaluation.agg_case_by=recording
```

## 6. Stop the machine when training finishes (`shutdown.*`)

`ShutdownConfig` (on every run config as `shutdown`) can stop the machine after
training, on rank 0, after a delay:

- `shutdown.stop_instance_on_finish`: enable (default `false`).
- `shutdown.stop_delay_minutes`: delay before stopping (default `5`).
- `shutdown.stop_method`: `ec2` (boto3 `stop_instances` for this instance, id and
  region read from EC2 IMDS) or `os` (`shutdown -h +<minutes>`).

```
--set shutdown.stop_instance_on_finish=true shutdown.stop_delay_minutes=5 shutdown.stop_method=ec2
```

Best-effort and never fatal: a failure to read metadata or issue the stop is
logged, not raised. Set the `LABRAM_DISABLE_SHUTDOWN` environment variable to
veto the stop at runtime.

## 7. ClearML model artifact in S3 (`clearml.upload_model_artifact`)

When ClearML tracking is enabled, the final/best checkpoint is explicitly
registered as a ClearML `OutputModel` at the end of training, uploading it to
`clearml.output_uri` (e.g. an S3 bucket) rather than relying solely on
framework auto-capture.

- `clearml.upload_model_artifact`: enable (default `true`).
- `clearml.artifact_name`: optional model name (defaults to `trained_model`).
- `clearml.output_uri`: the S3 (or other) storage target, e.g. `s3://my-bucket`.

```
--set clearml.enabled=true clearml.output_uri=s3://my-bucket clearml.upload_model_artifact=true
```

The best checkpoint (`checkpoint-best.pth`) is preferred, then `checkpoint.pth`,
then the highest-epoch checkpoint in `output_dir`.

## Layer freezing & trainability logging

Codebook-regularized fine-tuning can freeze part of the encoder via
`codebook_reg.encoder`:

- `codebook_reg.encoder.trainable=false` freezes the whole encoder.
- `codebook_reg.encoder.n_last_trainable_layers=N` keeps only the last `N`
  transformer blocks (plus the final `norm`/`fc_norm`) trainable and freezes the
  earlier blocks and the patch embedding / positional embeddings.

Freezing sets `requires_grad=False`, and `get_parameter_groups` skips any
parameter with `requires_grad=False` — so **frozen layers are excluded from the
optimizer's parameter groups** (they are neither stepped nor given an LR/WD).
This is applied at model-build time, before the optimizer is created, and is
verified by tests in `tests/test_codebook_classifier.py`
(`TestOptimizerExcludesFrozen`).

At startup each runner now logs, on rank 0:

- the full **model structure** (`str(model)`), and
- a **trainable/frozen summary** plus the list of non-trainable (frozen) layers,
  via `optim_factory.log_trainable_parameters`, e.g.:

  ```
  Trainable parameters: 486482 / 1716410 (28.3%); frozen: 1229928 params across 33 layers
  Non-trainable (frozen, excluded from optimizer) layers:
    [frozen] encoder.blocks.0.attn.qkv
    [frozen] encoder.blocks.0.mlp.fc1
    ...
  ```

This makes it easy to confirm that `n_last_trainable_layers` froze exactly the
layers you expect and that they were dropped from the optimizer.

## Model-architecture visualization (`logging.log_model_graph`)

Each runner renders the model as a graph coloured by trainability:

- **green** = fully trainable, **red** = fully frozen, **orange** = mixed,
  **grey** = no parameters.

Rendering uses Graphviz to a **vector SVG** (`logging.model_graph_format` =
`svg` or `png`); the DOT source is also written. If the Graphviz `dot` binary is
unavailable it falls back to a matplotlib figure, so a graph is always produced.
The graph is logged to TensorBoard (as a figure) and to ClearML (the SVG as
media, plus the SVG and DOT source as artifacts). Files are written under
`output_dir` (or `log_dir`) as `model_graph.svg` / `model_graph.dot`.

```
--set logging.log_model_graph=true logging.model_graph_format=svg
```

Because the colouring is driven by `requires_grad`, the graph doubles as a
visual confirmation of which layers `n_last_trainable_layers` froze.

## Run config as ClearML hyperparameters

The full run config is connected to ClearML two ways: as a JSON blob under
**Configuration** (`run_config`) and — flattened to dotted keys
(`optimizer/lr`, `trainer/epochs`, …) — as **tabular hyperparameters**
(`config` section) that ClearML can display, search and compare across
experiments. This is automatic whenever `clearml.enabled` is set.

## Logging config & save-only-final model

- `LoggingConfig` (`logging`) groups the diagnostic-artifact toggles
  (`log_model_graph`, `model_graph_format`, `log_data_split`), independent of
  the metric backend (`clearml` / `output.log_dir`).
- `output.save_only_final_model=true` skips the periodic/rolling per-epoch
  checkpoints (and the best checkpoint) and writes a single
  `checkpoint-final.pth` at the end of training — useful to save disk on long
  runs. The final model is always written at the end regardless; this flag only
  suppresses the intermediate ones.

```
--set output.save_only_final_model=true
```

## Paper fine-tuning config (TUAB abnormal detection)

`labram/configs/defaults/finetune_tuab_paper.json` reproduces the paper's TUAB
normal/abnormal fine-tuning recipe: `labram_base_patch200_200`, `lr=5e-4`,
`layer_decay=0.65`, `warmup_epochs=5`, `epochs=50`, `weight_decay=0.05`,
absolute position embeddings on, relative-position-bias and qkv-bias off,
batch size 64, fine-tuning from `./checkpoints/labram-base.pth`.

```
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_paper.json \
  --set finetune_checkpoint.finetune=./checkpoints/labram-base.pth data.data_path=./datasets/TUAB
```

## Data-split record (`logging.log_data_split`)

At fine-tuning start the train/val/test assignment is recorded to
`output_dir/data_split.json` and uploaded as a ClearML artifact (`data_split`).
It lists, per split, the window/recording/subject counts and their identifiers
(parsed from the pickle filenames), plus a **subject-overlap check** that warns
if the same subject appears in more than one split (a common data-leakage bug).

```
--set logging.log_data_split=true
```
