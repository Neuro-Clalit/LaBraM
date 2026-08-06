# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LaBraM (Large Brain Model) is a foundation model for EEG signal processing implementing the ICLR 2024 paper "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI". Training has three phases: (1) VQNSP tokenizer, (2) masked EEG pre-training, (3) downstream fine-tuning.

## Commands

```bash
# Run all tests (all synthetic data, no external datasets required)
pytest tests/ -v

# Run a single test file
pytest tests/test_runner_common.py -v

# Train VQNSP tokenizer (8-GPU)
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.vqnsp \
  --output_dir ./checkpoints/vqnsp/ --log_dir ./log/vqnsp/ \
  --model vqnsp_encoder_base_decoder_3x200x12 \
  --codebook_n_emd 8192 --codebook_emd_dim 64 --quantize_kmeans_init \
  --batch_size 128 --opt adamw --epochs 100

# Pre-train LaBraM (8-GPU)
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.pretrain \
  --output_dir ./checkpoints/labram_base --log_dir ./log/labram_base \
  --model labram_base_patch200_1600_8k_vocab \
  --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
  --tokenizer_weight ./checkpoints/vqnsp.pth \
  --batch_size 64 --lr 5e-4 --epochs 50

# Fine-tune on TUAB (8-GPU)
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.finetune \
  --output_dir ./checkpoints/finetune_tuab_base/ --log_dir ./log/finetune_tuab_base \
  --model labram_base_patch200_200 --finetune ./checkpoints/labram-base.pth \
  --dataset TUAB --batch_size 64 --lr 5e-4 --epochs 50 \
  --layer_decay 0.65 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias

# Codebook-regularized fine-tune (config-driven): adds spectral + quantization
# losses from a frozen-codebook VQNSP decoder as regularization.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.finetune \
  --config labram/configs/defaults/finetune_tuab_codebook.json \
  --set codebook_reg.tokenizer_weight=./checkpoints/vqnsp.pth \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth

# LaBraM++ mode (opt-in): sin/cos phase loss + per-patch CAR & z-scoring
# (arXiv:2505.16724). Off by default; the *_labram_plus_plus.json configs turn
# it on, or add --set labram_plus.enabled=true to any config. See
# docs/labram_plus_plus.md.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.vqnsp \
  --config labram/configs/defaults/vqnsp_labram_plus_plus.json

# Age regression / EEG brain age (opt-in): patient age comes from the EDF
# headers (MNE cannot see it -- the DOB is anonymized), joined onto the existing
# window pickles by filename. Two read-only prep steps, then a normal fine-tune.
# See docs/age_regression.md.
python dataset_maker/make_TUAB_age.py scan  --root /data/TUAB/edf   # age_metadata.json
python dataset_maker/make_TUAB_age.py split --root /data/TUAB/edf   # age_split.json
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path=/data/TUAB/edf \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth

# K-fold cross-validation fine-tune (opt-in): group-disjoint folds (by
# subject/recording), a reproducible cv_split.json artifact, per-fold experiments
# named with the fold number, and metrics aggregated across folds. See
# docs/cross_validation.md.
python -m labram.runs.finetune_cv \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set data.data_path=/data/TUAB
# ...one fold per process/job: --set cross_validation.fold=2
# Aggregate + report the fold results (optionally to ClearML):
python -m labram.eval.cv_report --base_dir ./checkpoints/finetune_tuab_cv5

# Submit any trainer to AWS SageMaker (--phase vqnsp|pretrain|finetune; a
# fine-tune with CV enabled dispatches one job per fold). Needs the SDK from
# requirements-sagemaker.txt (submit side only). See docs/sagemaker.md.
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_cv.json --dry_run   # preview
python -m labram.runs.submit_sagemaker \
  --config labram/configs/defaults/finetune_tuab_cv.json \
  --set sagemaker.enabled=true sagemaker.role=arn:aws:iam::ACCT:role/SM
python -m labram.runs.submit_sagemaker --phase vqnsp \
  --config labram/configs/defaults/vqnsp.json --dry_run
# Stream a big S3 corpus instead of downloading it to the EBS volume first:
#   --set sagemaker.input_mode=FastFile
```

### Installation

The code requires **torch>=2.3** (it uses the `torch.amp.GradScaler` /
`torch.amp.autocast` string-device API). Install the CUDA build of PyTorch that
matches your machine from the PyTorch index, then the remaining requirements:

```bash
conda create -n labram python=3.11
conda activate labram
# GPU (CUDA 11.8) — validated combination:
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Only if you submit training to SageMaker from this machine (the SDK is kept
# out of requirements.txt because that file is also installed inside the
# training container, where its dependency tree conflicts with pyhealth):
pip install -r requirements-sagemaker.txt
```

DeepSpeed (`--enable_deepspeed`) is optional and not installed by
`requirements.txt`; the old `deepspeed==0.4.0` pin is incompatible with
torch>=2.3. Install a compatible DeepSpeed manually only if you need it.

## Architecture

### Package layout

- `labram/layers/` — Neural-network primitives, one responsibility per module: `drop_path.py`, `mlp.py`, `attention.py`, `patch_embed.py` (PatchEmbed, TemporalConv), `transformer_block.py` (Block); all re-exported from `labram.layers`
- `labram/models/` — Config-driven model definitions: `neural_transformer.py` (shared backbone), `masked_eeg.py` (pre-training heads), `vqnsp.py` (tokenizer), `quantizer.py` (VQ with EMA), `components.py` (shared quantize/decode helpers reused by VQNSP and the regularized classifier), `codebook_classifier.py` (`CodebookRegularizedClassifier` for regularized fine-tuning), `outputs.py` (`PredictorOutput`), `registry.py` (all timm `@register_model` factories)
- `labram/data/` — All data concerns: `eeg_constants.py` (standard 10-20 layout, channel-index helpers), `hdf5_datasets.py` (`SingleShockDataset`/`ShockDataset`), `tuh_datasets.py` (TUAB/TUEV loaders), `bundles.py` (per-task `DatasetBundle`/`get_dataset_bundle`), `preprocess.py`, `pretraining.py` (`build_pretraining_dataset`); public API on `labram.data`
- `labram/losses/` — Configurable training losses: `config.py` (`LossConfig`), `spectral.py` (`SpectralReconstructionLoss`), `vqnsp.py` (`get_vqnsp_losses`), `classification.py` (`build_classification_criterion`), `codebook_regularized.py` (`CodebookRegularizedCriterion` combining classification + spectral + quantization losses), `outputs.py` (`LossBreakdown`)
- `labram/configs/` — Dataclass config tree on `ConfigBase` (JSON/YAML round-trip): model/data/optim/train/runner configs + `defaults/*.json`; constructors are config-driven (`VQNSP(VQNSPArchConfig)`, `NeuralTransformer(TransformerArchConfig)`)
- `labram/train/` — Per-phase training/eval loops (`train_vqnsp.py`, `train_pretrain.py`, `train_finetune.py`); shared optimizer/LR/metric helpers (`optimizer_update`, `apply_lr_wd_schedule`, `log_lr_wd_grad_metrics`) live in `labram/optim_factory.py`
- `labram/runs/` — Entry-point scripts (`run_vqnsp.py`, `run_pretrain.py`, `run_finetune.py`) invoked via `python -m`; `common.py` holds shared DDP setup, LR schedules, and dataloader construction; `finetune_setup.py`/`finetune_args.py` for fine-tuning setup; `finetune_cv.py` (K-fold cross-validation orchestration: fold naming, `cv_split.json`, per-fold runs, cross-fold aggregation); `submit_sagemaker.py` (submit fine-tune/CV as SageMaker jobs) + `sagemaker_entry.py` (in-container dispatcher)
- `labram/aws/` — `sagemaker.py`: generic SageMaker SDK wrapper (`SageMakerJobSpec`, pure `estimator_kwargs`, `SageMakerLauncher` with `resolve_role`/`resolve_image_uri`), vendored from the shared `common` repo (mirrors `labram/file_system`). `SageMakerConfig` lives on the base `RunConfig`, so any phase (vqnsp/pretrain/finetune) can be submitted via `submit_sagemaker --phase`. Submission bridges the gap between the submitting machine and the container: `s3://` data/weight paths become input channels rewritten to their `/opt/ml/input/data/...` mounts, a *local* weight file is uploaded to become one — unless it has an S3 mirror in `sagemaker.weight_s3_uris` (the shipped `./checkpoints/{labram-base,vqnsp}.pth` do, so they are served from `s3://eeg-data-public/models/labram/...` instead of being re-uploaded on every submission), and a local `data.data_path` is rejected — too big. Weight staging covers `finetune_checkpoint.finetune` + `model.codebook_reg.tokenizer_weight` (finetune) and `model.tokenizer.tokenizer_weight` (pretrain's frozen VQNSP). ClearML credentials are forwarded from the environment or `clearml.conf`, and `source_dir` is packaged from the git-tracked tree minus weights rather than the multi-GB repo root. Because the SDK's own code upload inherits `output_kms_key` (which an account default can set to a CMK the submitter is denied), the CLI tars/uploads the source itself and hands the estimator an `s3://…/sourcedir.tar.gz`, which makes the SDK skip that upload. Channels whose S3 uri addresses a *single object* (`config`, `pretrained`, `tokenizer`) get a per-channel `File` `InputMode`, because `FastFile`/`Pipe` expose only the keys *beneath* a prefix and would mount them empty; the bulk `dataset` channel still streams. The submitter also ships the checkout's git metadata inside the tarball so the in-container ClearML task can record branch/commit/uncommitted diff (no `.git` exists in the container), prints a banner stating the job outlives the local process, supports `--detach` (submit and exit, no log streaming), and caps `max_run_sec` at 24h by default. Because `CreateTrainingJob` rejects a `RoleArn` from another account ("Cross-account pass role is not allowed") only *after* the code/config/weights have been uploaded to the caller's bucket, submission preflights `sts:GetCallerIdentity` against the account parsed out of `sagemaker.role` and aborts first; `sagemaker.profile` pins which credential profile (hence account) submits, instead of depending on `AWS_PROFILE` being set in the shell. See `docs/sagemaker.md`
- `labram/eval/` — Offline evaluation toolkit for trained fine-tune models: `loading.py` (resolve checkpoint/config/`data_split.json` from a local dir or ClearML; rebuild the model; build a case-id-aware eval loader), `inference.py` (`collect_predictions` → `PredictionResult`), `aggregation.py` (per-case window pooling across methods, entropy/accuracy + selective-prediction analysis, case ranking), `cv_aggregation.py` + `cv_report.py` (collect a CV study's per-fold results from local dirs or ClearML, compute metrics over all folds (mean ± std), log a `cv_summary` to ClearML), `plots.py` (evaluation figure builders); `clearml_analysis.py` + `clearml_report.py` (pull a ClearML experiment's metrics/hyperparameters/console into a local `ExperimentSnapshot` and derive concrete heuristic insights — overfitting, divergence, LR-schedule issues — via `python -m labram.eval.clearml_report`; see `docs/clearml_local_analysis.md`). Driven by `notebooks/finetune_evaluation.ipynb`; see `docs/finetune_evaluation.md`
- `labram/utils/` — `checkpoint.py`, `distributed.py`, `training.py` (cosine LR, layer decay), `logging.py` (`MetricLogger`, `TensorboardLogger`, `ClearMLLogger`, `MultiWriter` fan-out writer, plus `configure_logging`/`get_logger` for the rank-aware Python `logging` setup), `metrics.py`, `cli.py`, `git_info.py` (capture a checkout's branch/commit/uncommitted diff on the submitting machine and replay it onto a ClearML task via `Task.set_script`, since a SageMaker container has no `.git` for ClearML to detect); `__init__.py` also re-exports the `labram.data` public API for backward compatibility
- `dataset_maker/` — Preprocessing scripts that convert raw EEG files (`.cnt`/`.edf`/`.bdf`) to HDF5 (`make_h5dataset_for_pretrain.py`) and TUH datasets to pickle (`make_TUAB.py`, `make_TUEV.py`)

### Data flow

**VQNSP**: Raw EEG `[B, channels, T]` → `NeuralTransformer` encoder → `NormEMAVectorQuantizer` → discrete codebook indices → `NeuralTransformer` decoder → reconstructed amplitude & phase spectrum (loss on spectrum reconstruction).

**Pre-training**: Raw EEG → frozen VQNSP → discrete codes → 50% random masking → trainable `NeuralTransformerForMaskedEEGModeling` → cross-entropy loss predicting masked tokens.

**Fine-tuning**: Raw EEG → pre-trained `NeuralTransformer` (from `--finetune` checkpoint, head discarded) → mean-pool over time → classification head → cross-entropy. Set `data.split_json` (local or `s3://`) to reuse a previous run's recorded `data_split.json` and pin the exact train/val/test case assignment across models (`labram/data/data_split_reuse.py`).

**Codebook-regularized fine-tuning** (opt-in, `codebook_reg.enabled`): Raw EEG → trainable encoder (from pre-trained checkpoint) → classification head over configurable feature sources (`encoder_mean` / `quantize_mean` / `bag_of_codes`); the same patch tokens also go through the grafted VQNSP quantizer (codebook frozen) + trainable decoder to reconstruct the spectrum. Loss = classification + spectral (amplitude/phase) + quantization, combined by `CodebookRegularizedCriterion`. Encoder/decoder/codebook use LR scales below the head LR. See `docs/codebook_regularized_finetune_plan.md`.

**Cross-validation fine-tuning** (opt-in, `cross_validation.enabled`): the data pool (train+val, or all splits) is partitioned into K **group-disjoint** folds (grouped by subject/recording so a case never straddles train/val/test). For fold *k*: test = fold *k*, val = fold *(k+1) mod K*, train = the rest. Each fold trains via the normal fine-tune path (`run_finetune.main(config, bundle=fold_bundle)`) as its own sub-experiment — output dir `‹base›/fold_<k>/` and ClearML `‹project›/‹experiment›` + task `fold_<k>`. Artifacts: `cv_split.json` (the reproducible fold partition), per-fold `fold_metrics.json`, and `cv_summary.json` (metrics aggregated across folds). Run all folds in-process (`cross_validation.fold=-1`) or one fold per job (`fold=k`). See `docs/cross_validation.md`.

**Age-regression fine-tuning** (opt-in, `data.dataset=TUAB_AGE`): a scalar-target
downstream task (EEG brain age). The label is **not** in the window pickles — TUH no longer
distributes clinical reports, so the age is read from the EDF header's patient field
(bytes 8:88, `... Age:42`). MNE cannot supply it: the DOB is anonymized to `01-JAN-0000`, so
`raw.info['subject_info']` has no age and raw-byte parsing is mandatory
(`data/tuh_metadata.py`). Verified on TUAB v3.0.0: 100% coverage over 2,993 recordings,
reproducing the corpus AAREADME demographics tables exactly; `Age:999` is TUH's HIPAA
redaction for 90+ and is excluded. Because processed window filenames map 1:1 onto EDF stems,
ages join onto the **existing** pickles via an `age_metadata.json` sidecar — no
re-preprocessing. `data/age_splits.py` also rebuilds a **subject-disjoint, seeded** train/val
split (the shipped `make_TUAB.py` split leaks 16 subjects across train/val, which matters far
more for age than for a class label) and asserts zero overlap on save *and* load. Task type
lives on `DatasetBundle.task` / `FinetuneModelConfig.task` rather than being inferred from
`nb_classes`, since a scalar head and a binary classifier are both `nb_classes == 1`; it
selects the criterion (`losses/regression.py`, Huber by default),
the metrics (`utils/regression_metrics.py` — MAE/RMSE/R²/r plus the `age_bias_slope` and
`mae_corrected` brain-age diagnostics), and lower-is-better model selection on MAE. Targets are
z-scored by the loader using train-split stats and de-normalized before metrics, so errors read
in years. Split by subject, aggregate by recording — age is a property of the session, not the
subject. See `docs/age_regression.md`.

**LaBraM++ mode** (opt-in, `labram_plus.enabled`, arXiv:2505.16724): a bundle of three signal-processing/loss improvements over the original LaBraM, off by default so existing checkpoints/configs are unaffected. (1) Per-patch **Common Average Reference** and (2) per-patch **z-scoring** are applied to model inputs; (3) the VQNSP tokenizer's phase reconstruction uses a **sin/cos circular loss** (`‖sin φ̂ − sin φ‖² + ‖cos φ̂ − cos φ‖²`) instead of the raw-angle MSE, removing the ±π wrap-around discontinuity. The single switch is `LaBraMPlusConfig` (`labram/configs/labram_plus_config.py`) on every `RunConfig` as `config.labram_plus`; preprocessing is model-owned (applied in `NeuralTransformer._embed_inputs`, `NeuralTransformerForMaskedEEGModeling.forward_features`, and `VQNSP._preprocess`) so it stays consistent across training and evaluation and never double-applies. Config files: `configs/defaults/{vqnsp,pretrain_,finetune_tuab_}labram_plus_plus.json`. See `docs/labram_plus_plus.md`.

### Key cross-cutting concerns

**Channel handling**: 62-channel standard 10-20 layout is defined in `data/eeg_constants.py::standard_1020`. `get_channel_indices()` maps each dataset's channel names to this standard order. Fine-tuning setup (`runs/finetune_setup.py`) reorders loaded checkpoint weights to match the target dataset channel order — this is critical for transfer learning.

**Distributed training**: DDP initialized in `runs/common.py::setup_environment()`. All metric logging is gated on `utils.is_main_process()`. `--auto_resume` resumes from the latest checkpoint automatically.

**LR scheduling**: Cosine annealing with warmup (`utils/training.py`). Fine-tuning uses layer-wise LR decay via `optim_factory.py::LayerDecayValueAssigner` (timm-style per-parameter group scaling, controlled by `--layer_decay`).

**Model registry**: Models are registered with timm (`timm.models.register_model`) and instantiated via `timm.models.create_model(name, ...)`. Model names like `labram_base_patch200_200` encode architecture hyperparameters.

**Logging & experiment tracking**: Status messages go through the rank-aware `labram` Python logger (`utils.get_logger`), configured once in `runs/common.py::setup_environment` (rank 0 at INFO, other ranks WARNING; optional `run.log` file). Metrics flow through a single `log_writer` built by `runs/common.py::create_log_writer` — TensorBoard, ClearML, or both (combined via `MultiWriter`). At the end of a fine-tune, `runs/common.py::log_summary_metrics` records the best-epoch eval metrics as ClearML **single values** (`report_single_value`, rendered as a side-by-side table in ClearML compare mode) plus a `final_metrics` config section (sortable experiment-table columns) — so runs and CV folds are directly comparable. ClearML tracking is opt-in per-run (`clearml.enabled`, `ClearMLConfig` on every `*RunConfig`) and is an optional dependency: a missing `clearml` package downgrades to TensorBoard-only with a warning. Metrics reach the writers in **relative (scale-free) form** by default (`LoggingConfig`): per-component losses and gradient norms are logged as their share of the component total (`‹name›_loss_rel`, summing to 1) instead of raw magnitudes, and every series is plotted against normalized training progress (`0…logging.relative_step_scale`) instead of the raw global iteration — so runs of different length/batch size overlay directly. Aggregate losses and the console/`log.txt` stats stay absolute. Turn either half off with `logging.relative_loss_components=false` / `logging.relative_step_axis=false`. See `docs/logging_clearml.md`.
