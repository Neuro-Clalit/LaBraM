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
  `specificity`, `f1`, `f1_weighted`, `roc_auc`, `pr_auc`, `cohen_kappa`, and the
  binary confusion-matrix cells (`cm_tn/cm_fp/cm_fn/cm_tp`).
- Confusion matrix: native in ClearML, markdown table in the TensorBoard *Text*
  tab (`evaluation.log_confusion_matrix`).
- ROC and precision-recall curves (binary): matplotlib figures pushed to both
  backends (`evaluation.log_curves`; degrades to scalars only if matplotlib is
  absent).

```
--set evaluation.detailed_metrics=true evaluation.log_confusion_matrix=true evaluation.log_curves=true
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
