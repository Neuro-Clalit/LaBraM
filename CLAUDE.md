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
```

### Installation

```bash
conda create -n labram python=3.11
conda activate labram
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tensorboardX
pip install -r requirements.txt
```

## Architecture

### Package layout

- `labram/models/` — Model definitions split by abstraction level: `layers.py` (DropPath, Mlp, Attention, PatchEmbed, TemporalConv), `blocks.py` (transformer Block), `neural_transformer.py` (shared backbone), `masked_eeg.py` (pre-training heads), `vqnsp.py` (tokenizer), `quantizer.py` (VQ with EMA), `registry.py` (all timm `@register_model` factories)
- `labram/trainers/` — Per-phase training/eval loops (`train_vqnsp.py`, `train_pretrain.py`, `train_finetune.py`); shared LR/metric helpers in `base.py`
- `labram/runners/` — Entry-point scripts invoked via `python -m`; `common.py` holds shared DDP setup, LR schedules, and dataloader construction used by all three runners
- `labram/utils/` — `channels.py` (standard 10-20 layout), `checkpoint.py`, `distributed.py`, `training.py` (cosine LR, layer decay), `logging.py` (MetricLogger, TensorboardLogger)
- `labram/data_processor/` — `SingleShockDataset` / `ShockDataset`: HDF5-backed datasets for pre-training
- `dataset_maker/` — Preprocessing scripts that convert raw EEG files (`.cnt`/`.edf`/`.bdf`) to HDF5 (`make_h5dataset_for_pretrain.py`) and TUH datasets to pickle (`make_TUAB.py`, `make_TUEV.py`)

### Data flow

**VQNSP**: Raw EEG `[B, channels, T]` → `NeuralTransformer` encoder → `NormEMAVectorQuantizer` → discrete codebook indices → `NeuralTransformer` decoder → reconstructed amplitude & phase spectrum (loss on spectrum reconstruction).

**Pre-training**: Raw EEG → frozen VQNSP → discrete codes → 50% random masking → trainable `NeuralTransformerForMaskedEEGModeling` → cross-entropy loss predicting masked tokens.

**Fine-tuning**: Raw EEG → pre-trained `NeuralTransformer` (from `--finetune` checkpoint, head discarded) → mean-pool over time → classification head → cross-entropy.

### Key cross-cutting concerns

**Channel handling**: 62-channel standard 10-20 layout is defined in `utils/channels.py::standard_1020`. `get_channel_indices()` maps each dataset's channel names to this standard order. Fine-tuning setup (`runners/finetune_setup.py`) reorders loaded checkpoint weights to match the target dataset channel order — this is critical for transfer learning.

**Distributed training**: DDP initialized in `runners/common.py::setup_environment()`. All metric logging is gated on `utils.is_main_process()`. `--auto_resume` resumes from the latest checkpoint automatically.

**LR scheduling**: Cosine annealing with warmup (`utils/training.py`). Fine-tuning uses layer-wise LR decay via `optim_factory.py::LayerDecayValueAssigner` (timm-style per-parameter group scaling, controlled by `--layer_decay`).

**Model registry**: Models are registered with timm (`timm.models.register_model`) and instantiated via `timm.models.create_model(name, ...)`. Model names like `labram_base_patch200_200` encode architecture hyperparameters.
