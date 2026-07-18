# LaBraM++ training mode

LaBraM++ is an opt-in training mode that adds the signal-processing and loss
improvements from *"Advancing Brainwave Modeling with a Codebook-Based
Foundation Model"* ([arXiv:2505.16724](https://arxiv.org/abs/2505.16724)) on top
of the original LaBraM. It is **off by default**: with `labram_plus.enabled =
false` the models, losses, and checkpoints are byte-for-byte the original
LaBraM, so every existing config, checkpoint, and workflow is unchanged.

## What LaBraM++ changes

Three improvements, each grounded in EEG signal processing and each individually
ablatable:

| Feature | Config flag | Where it acts | Effect |
|---|---|---|---|
| **Common Average Reference (CAR)** | `common_average_reference` | model input, per patch | Subtracts the across-channel mean to suppress noise shared by every electrode. |
| **Per-patch z-scoring** | `z_score_patches` | model input, per patch | Standardises each `(channel, patch)` window over time so amplitude differences across channels do not dominate. |
| **Sin/cos phase loss** | `phase_loss` | VQNSP tokenizer loss | Reconstructs phase as `(sin φ, cos φ)` instead of the raw angle. |

### The phase-loss fix

The original tokenizer minimises `‖φ̂ − φ‖²` on the raw Fourier phase. Because
phase is circular, that loss is discontinuous at the ±π boundary: a prediction
`φ̂ = −π + ε` for a target `φ = π − ε` describes almost the same angle yet incurs
a loss approaching `(2π)² ≈ 39`. LaBraM++ replaces it with

```
L_phase = ‖sin(φ̂) − sin(φ)‖² + ‖cos(φ̂) − cos(φ)‖²
        = 2 − 2·cos(φ̂ − φ)  = 4·sin²((φ̂ − φ)/2)
```

which is smooth and periodic — the same ±π boundary case now costs `≈ 4ε² → 0`.
In `phase_loss="sincos"` mode the phase target is the **raw angle** in radians
(not std-normalised), and the `sin`/`cos` transform is applied inside the loss.

## How it is wired

The single switch is `LaBraMPlusConfig` (`labram/configs/labram_plus_config.py`),
attached to every `RunConfig` as `config.labram_plus`:

```python
@dataclass
class LaBraMPlusConfig(ConfigBase):
    enabled: bool = False                 # master switch
    common_average_reference: bool = True # applied only when enabled
    z_score_patches: bool = True          # applied only when enabled
    z_score_eps: float = 1e-5
    phase_loss: str = "sincos"            # "sincos" (LaBraM++) | "angle"
```

`enabled=False` forces `use_car`, `use_z_score`, and `resolved_phase_loss ==
"angle"` regardless of the sub-flags, so the default path is exactly the
original LaBraM. When `enabled=True` the sub-flags take effect and any one can
be turned off to ablate that feature.

**Preprocessing is model-owned**, so it applies identically during training,
evaluation, and offline inference — there is one preprocessing boundary per
model and it never double-applies:

- **Fine-tune backbone** (`NeuralTransformer`) preprocesses in
  `maybe_preprocess_input` at the top of `_embed_inputs` (raw-EEG path only).
- **Pre-training** (`NeuralTransformerForMaskedEEGModeling`) preprocesses at the
  top of `forward_features`.
- **VQNSP tokenizer** owns preprocessing in `_preprocess` (called by both
  `forward` and `encode`), so the reconstruction target and the encoder input
  share one preprocessed signal. Its internal encoder/decoder are built with a
  *disabled* `labram_plus`, guaranteeing no double application.
- **Codebook-regularized classifier** preprocesses at the top of `forward` and
  sets `x_patched` to the preprocessed signal for the spectral target.

The config threads into the timm registry factories via a `labram_plus` kwarg
(`create_model(..., labram_plus=config.labram_plus)`), and the phase-loss mode
flows into the `LossConfig` that `SpectralReconstructionLoss` consumes.

## Running LaBraM++

Ready-made configs live in `labram/configs/defaults/`:

```bash
# 1) Train the LaBraM++ VQNSP tokenizer (sin/cos phase loss + CAR + z-scoring)
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.vqnsp \
  --config labram/configs/defaults/vqnsp_labram_plus_plus.json

# 2) Pre-train LaBraM++ on top of that tokenizer
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.pretrain \
  --config labram/configs/defaults/pretrain_labram_plus_plus.json \
  --set model.tokenizer.tokenizer_weight=./checkpoints/vqnsp_labram_plus_plus/vqnsp.pth

# 3) Fine-tune LaBraM++ on TUAB
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.finetune \
  --config labram/configs/defaults/finetune_tuab_labram_plus_plus.json \
  --set finetune_checkpoint.finetune=./checkpoints/labram_plus_plus_base/checkpoint.pth
```

You can also enable LaBraM++ on any existing config from the CLI without a
dedicated file:

```bash
-m labram.runs.finetune --config labram/configs/defaults/finetune_tuab.json \
  --set labram_plus.enabled=true
```

Ablate a single feature, e.g. keep the sin/cos loss but drop CAR:

```bash
--set labram_plus.enabled=true labram_plus.common_average_reference=false
```

> **Consistency note.** The tokenizer, pre-training, and fine-tuning phases
> should all use the same `labram_plus` settings: codes produced by a LaBraM++
> tokenizer assume CAR + z-scored inputs, so a downstream model must preprocess
> its inputs the same way. The shipped `*_labram_plus_plus.json` configs keep
> the three phases aligned.

## Tests

`tests/test_labram_plus.py` covers the config semantics, the preprocessing math
(CAR cancels shared signal; z-scoring standardises each patch), the circular
phase loss (parity + the ±π continuity fix), the VQNSP / backbone / masked-EEG
wiring (enabled vs. disabled, no double preprocessing), and that the shipped
LaBraM++ configs enable the mode while the original defaults keep it disabled.
