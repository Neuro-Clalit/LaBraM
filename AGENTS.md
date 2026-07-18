# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Overview

LaBraM (Large Brain Model) is a Python package for EEG foundation-model research. It implements the ICLR 2024 paper "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI".

The main training phases are:

1. VQNSP tokenizer training
2. Masked EEG pre-training
3. Downstream fine-tuning

## Commands

```bash
# Run all tests. Tests use synthetic data and should not require external EEG datasets.
pytest tests/ -v

# Run a focused test file.
pytest tests/test_runner_common.py -v
```

Installation follows the README. Runtime dependencies are intentionally not pinned in `pyproject.toml` because torch builds are environment-specific.

```bash
conda create -n labram python=3.11
conda activate labram
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tensorboardX
pip install -r requirements.txt
```

Use module entry points under `labram.runners` for maintained training commands:

```bash
# Train VQNSP tokenizer.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.vqnsp \
  --output_dir ./checkpoints/vqnsp/ --log_dir ./log/vqnsp/ \
  --model vqnsp_encoder_base_decoder_3x200x12 \
  --codebook_n_emd 8192 --codebook_emd_dim 64 --quantize_kmeans_init \
  --batch_size 128 --opt adamw --epochs 100

# Pre-train LaBraM.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.pretrain \
  --output_dir ./checkpoints/labram_base --log_dir ./log/labram_base \
  --model labram_base_patch200_1600_8k_vocab \
  --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
  --tokenizer_weight ./checkpoints/vqnsp.pth \
  --batch_size 64 --lr 5e-4 --epochs 50

# Fine-tune on TUAB.
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.finetune \
  --output_dir ./checkpoints/finetune_tuab_base/ --log_dir ./log/finetune_tuab_base \
  --model labram_base_patch200_200 --finetune ./checkpoints/labram-base.pth \
  --dataset TUAB --batch_size 64 --lr 5e-4 --epochs 50 \
  --layer_decay 0.65 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias
```

## Repository Layout

- `labram/layers/`: neural-network primitives, one per module (`drop_path`, `mlp`, `attention`, `patch_embed`, `transformer_block`), re-exported from `labram.layers`.
- `labram/models/`: config-driven model definitions for tokenizer, pre-training, fine-tuning, and vector quantization.
- `labram/data/`: all data concerns — channel layouts/index helpers (`eeg_constants`), HDF5 datasets (`hdf5_datasets`), TUH loaders (`tuh_datasets`), per-task bundles (`bundles`), preprocessing (incl. LaBraM++ CAR + per-patch z-scoring in `preprocess.py`), and pre-training assembly; public API on `labram.data`.
- `labram/losses/`: configurable training losses — `LossConfig` (incl. `phase_loss` for the LaBraM++ sin/cos objective), `SpectralReconstructionLoss`, `get_vqnsp_losses`, `build_classification_criterion`.
- `labram/configs/`: dataclass config tree on `ConfigBase` (JSON/YAML round-trip); model/data/optim/train/runner configs + `defaults/*.json`. `LaBraMPlusConfig` (`labram_plus_config.py`) is the opt-in LaBraM++ switch on every `RunConfig`.
- `labram/train/`: training and evaluation loops for each phase (`train_finetune.py`, `train_pretrain.py`, `train_vqnsp.py`); shared helpers in `base.py`.
- `labram/runs/`: command-line entry points (`run_vqnsp.py`, `run_pretrain.py`, `run_finetune.py`) and setup code. `common.py` owns shared DDP setup, dataloaders, and scheduling helpers.
- `labram/utils/`: checkpointing, distributed utilities, metrics, logging, and training schedules; `__init__` re-exports the `labram.data` public API for backward compatibility.
- `dataset_maker/`: scripts that convert raw EEG datasets into HDF5 or pickle artifacts.
- `tests/`: pytest coverage using synthetic data.

## Engineering Notes

- Preserve the existing package style and module boundaries. Avoid unrelated refactors while changing training, runner, or model code.
- Prefer `python -m labram.runners.<phase>` entry points over legacy top-level script names when documenting or testing training flows.
- Keep test additions synthetic and lightweight unless the task explicitly requires external EEG data.
- Be careful with channel order. The standard 10-20 layout is defined in `labram/data/eeg_constants.py`, and fine-tuning setup reorders checkpoint weights to match target dataset channel order.
- Distributed training uses DDP setup in `labram/runs/common.py`. Metric logging should stay gated on main-process checks.
- Models are registered through `timm.models.register_model` and instantiated with `timm.models.create_model`.
- Fine-tuning checkpoint loading discards or adapts classification heads; avoid breaking transfer-learning behavior.

## Data Assumptions

Preprocessing should match the paper and README unless a task says otherwise:

- Remove irrelevant channels.
- Bandpass filter: 0.1-75 Hz.
- Notch filter: 50 Hz.
- Resample to 200 Hz.
- Convert signal units to microvolts.

## Verification

For code changes, run the most focused relevant pytest file first, then broaden to `pytest tests/ -v` when the change touches shared code, model contracts, runner setup, or data handling.
