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
- `labram/runs/` — Entry-point scripts (`run_vqnsp.py`, `run_pretrain.py`, `run_finetune.py`) invoked via `python -m`; `common.py` holds shared DDP setup, LR schedules, and dataloader construction; `finetune_setup.py`/`finetune_args.py` for fine-tuning setup
- `labram/utils/` — `checkpoint.py`, `distributed.py`, `training.py` (cosine LR, layer decay), `logging.py` (`MetricLogger`, `TensorboardLogger`, `ClearMLLogger`, `MultiWriter` fan-out writer, plus `configure_logging`/`get_logger` for the rank-aware Python `logging` setup), `metrics.py`, `cli.py`; `__init__.py` also re-exports the `labram.data` public API for backward compatibility
- `dataset_maker/` — Preprocessing scripts that convert raw EEG files (`.cnt`/`.edf`/`.bdf`) to HDF5 (`make_h5dataset_for_pretrain.py`) and TUH datasets to pickle (`make_TUAB.py`, `make_TUEV.py`)

### Data flow

**VQNSP**: Raw EEG `[B, channels, T]` → `NeuralTransformer` encoder → `NormEMAVectorQuantizer` → discrete codebook indices → `NeuralTransformer` decoder → reconstructed amplitude & phase spectrum (loss on spectrum reconstruction).

**Pre-training**: Raw EEG → frozen VQNSP → discrete codes → 50% random masking → trainable `NeuralTransformerForMaskedEEGModeling` → cross-entropy loss predicting masked tokens.

**Fine-tuning**: Raw EEG → pre-trained `NeuralTransformer` (from `--finetune` checkpoint, head discarded) → mean-pool over time → classification head → cross-entropy.

**Codebook-regularized fine-tuning** (opt-in, `codebook_reg.enabled`): Raw EEG → trainable encoder (from pre-trained checkpoint) → classification head over configurable feature sources (`encoder_mean` / `quantize_mean` / `bag_of_codes`); the same patch tokens also go through the grafted VQNSP quantizer (codebook frozen) + trainable decoder to reconstruct the spectrum. Loss = classification + spectral (amplitude/phase) + quantization, combined by `CodebookRegularizedCriterion`. Encoder/decoder/codebook use LR scales below the head LR. See `docs/codebook_regularized_finetune_plan.md`.

### Key cross-cutting concerns

**Channel handling**: 62-channel standard 10-20 layout is defined in `data/eeg_constants.py::standard_1020`. `get_channel_indices()` maps each dataset's channel names to this standard order. Fine-tuning setup (`runs/finetune_setup.py`) reorders loaded checkpoint weights to match the target dataset channel order — this is critical for transfer learning.

**Distributed training**: DDP initialized in `runs/common.py::setup_environment()`. All metric logging is gated on `utils.is_main_process()`. `--auto_resume` resumes from the latest checkpoint automatically.

**LR scheduling**: Cosine annealing with warmup (`utils/training.py`). Fine-tuning uses layer-wise LR decay via `optim_factory.py::LayerDecayValueAssigner` (timm-style per-parameter group scaling, controlled by `--layer_decay`).

**Model registry**: Models are registered with timm (`timm.models.register_model`) and instantiated via `timm.models.create_model(name, ...)`. Model names like `labram_base_patch200_200` encode architecture hyperparameters.

**Logging & experiment tracking**: Status messages go through the rank-aware `labram` Python logger (`utils.get_logger`), configured once in `runs/common.py::setup_environment` (rank 0 at INFO, other ranks WARNING; optional `run.log` file). Metrics flow through a single `log_writer` built by `runs/common.py::create_log_writer` — TensorBoard, ClearML, or both (combined via `MultiWriter`). ClearML tracking is opt-in per-run (`clearml.enabled`, `ClearMLConfig` on every `*RunConfig`) and is an optional dependency: a missing `clearml` package downgrades to TensorBoard-only with a warning. See `docs/logging_clearml.md`.
