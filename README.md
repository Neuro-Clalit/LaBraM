# LaBraM

Official implementation of our ICLR 2024 paper:
[**Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI**](https://openreview.net/forum?id=QzTpTRVtrP)

![labram](labram.png)

## Abstract

The current electroencephalogram (EEG) based deep learning models are typically designed for specific datasets and applications in brain-computer interaction (BCI), limiting the scale of the models and thus diminishing their perceptual capabilities and generalizability. Recently, Large Language Models (LLMs) have achieved unprecedented success in text processing, prompting us to explore the capabilities of Large EEG Models (LEMs). We hope that LEMs can break through the limitations of different task types of EEG datasets, and obtain universal perceptual capabilities of EEG signals through unsupervised pre-training. Then the models can be fine-tuned for different downstream tasks. However, compared to text data, the volume of EEG datasets is generally small and the format varies widely. For example, there can be mismatched numbers of electrodes, unequal length data samples, varied task designs, and low signal-to-noise ratio. To overcome these challenges, we propose a unified foundation model for EEG called Large Brain Model (LaBraM). LaBraM enables cross-dataset learning by segmenting the EEG signals into EEG channel patches. Vector-quantized neural spectrum prediction is used to train a semantically rich neural tokenizer that encodes continuous raw EEG channel patches into compact neural codes. We then pre-train neural Transformers by predicting the original neural codes for the masked EEG channel patches. The LaBraMs were pre-trained on about 2,500 hours of various types of EEG signals from around 20 datasets and validated on multiple different types of downstream tasks. Experiments on abnormal detection, event type classification, emotion recognition, and gait prediction show that our LaBraM outperforms all compared SOTA methods in their respective fields.

---

## Environment Setup

Create the environment and install dependencies:

```bash
conda create -n labram python=3.11
conda activate labram
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tensorboardX
pip install -r requirements.txt
```

---

## [IMPORTANT] Fine-tune on Your Own Dataset

You can adapt LaBraM to your own datasets by following the fine-tuning scripts provided for TUAB and TUEV. Simply replace the dataset-specific parts of the code with your own data. When doing so, make sure to:

1. Load a pre-trained LaBraM checkpoint.
2. Provide the input channel order list to specify the channel configuration.

**The above two points are significant to obtain normal performance of LaBraM.**

---

## Running Experiments

Training is **config-driven**. Each of the three phases has an entry point invoked
with `python -m`, and a default config under [`labram/configs/defaults/`](labram/configs/defaults):

| Phase | Entry point | Default config |
| --- | --- | --- |
| VQ-NSP tokenizer | `labram.runs.run_vqnsp` | `vqnsp.json` |
| Pre-training | `labram.runs.run_pretrain` | `pretrain.json` |
| Fine-tune (TUAB) | `labram.runs.run_finetune` | `finetune_tuab.json` |
| Fine-tune (TUEV) | `labram.runs.run_finetune` | `finetune_tuev.json` |

Every runner accepts:

* `--config PATH` — a JSON or YAML run-config file (omit to use built-in dataclass defaults).
* `--set KEY=VALUE [KEY=VALUE ...]` — dotted-path overrides applied on top of the config,
  e.g. `--set optimizer.lr=1e-4 trainer.epochs=20 output.output_dir=./out`.

On start, the resolved config is written to `<output_dir>/run_config.yaml` for reproducibility.
Copy a default config and edit it, or keep the default and override fields with `--set`.

### 1. Prepare Pre-training Data

Convert raw EEG files (e.g., `.cnt`, `.edf`, `.bdf`) into HDF5 format using:

```
dataset_maker/make_h5dataset_for_pretrain.py
```

You may also implement your own preprocessing pipeline, but please ensure it matches the setup in our paper:

* Remove irrelevant channels
* Bandpass filter: **0.1–75 Hz**
* Notch filter: **50 Hz**
* Resample to **200 Hz**
* Set unit to **µV**

---

### 2. Train the Neural Tokenizer

The tokenizer is trained via **vector-quantized neural spectrum prediction (VQ-NSP)**. We recommend training on **8 × NVIDIA RTX 3090 (or better)** GPUs.

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_vqnsp \
    --config labram/configs/defaults/vqnsp.json
```

The default config sets `vqnsp_encoder_base_decoder_3x200x12`, an 8192-entry codebook of
dim 64, k-means codebook init, `batch_size=128`, `adamw` with `opt_betas=[0.9, 0.99]`,
`weight_decay=1e-4`, `warmup_epochs=10`, and `epochs=100`. Override anything inline:

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_vqnsp \
    --config labram/configs/defaults/vqnsp.json \
    --set trainer.epochs=50 output.output_dir=./checkpoints/vqnsp_short/
```

---

### 3. Pre-train LaBraM

Pre-train LaBraM by reconstructing masked neural codes from EEG channel patches:

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_pretrain \
    --config labram/configs/defaults/pretrain.json \
    --set tokenizer.tokenizer_weight=./checkpoints/vqnsp.pth
```

The default config uses `labram_base_patch200_1600_8k_vocab` with the
`vqnsp_encoder_base_decoder_3x200x12` tokenizer, `lr=5e-4`, `warmup_epochs=5`,
`clip_grad=3.0`, `layer_scale_init_value=0.1`, `opt_betas=[0.9, 0.98]`, and `epochs=50`.
Point `tokenizer.tokenizer_weight` at the checkpoint produced in step 2.

---

### 4. Fine-tune on Downstream Tasks

Preprocess datasets (e.g., TUAB, TUEV) using:

```
dataset_maker/make_TUAB.py
dataset_maker/make_TUEV.py
```

This includes preprocessing and splitting into train/val/test sets. Hyperparameters such as **learning rate** and **warmup epochs** strongly affect results—tune them for best performance.

**TUAB** (binary abnormal detection):

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
    --config labram/configs/defaults/finetune_tuab.json \
    --set finetune_checkpoint.finetune=./checkpoints/labram-base.pth \
          data_path=./datasets/TUAB
```

**TUEV** (6-class event classification):

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
    --config labram/configs/defaults/finetune_tuev.json \
    --set finetune_checkpoint.finetune=./checkpoints/labram-base.pth \
          data_path=./datasets/TUEV
```

Both default configs use `labram_base_patch200_200` with `layer_decay=0.65`, `lr=5e-4`,
`warmup_epochs=5`, `epochs=50`, absolute position embeddings, and relative-position-bias /
qkv-bias disabled — matching the paper's fine-tuning recipe. To fine-tune on your **own
dataset**, copy one of these configs and point `data_path` at your data (`nb_classes` is
inferred from the dataset bundle). Tune `optimizer.lr`, `optimizer.warmup_epochs`, and
`layer_decay` via `--set` for best results.

---

## Citation

If you find our paper/code useful, please consider citing our work:

```bibtex
@inproceedings{
jiang2024large,
title={Large Brain Model for Learning Generic Representations with Tremendous {EEG} Data in {BCI}},
author={Wei-Bang Jiang and Li-Ming Zhao and Bao-Liang Lu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=QzTpTRVtrP}
}
```
