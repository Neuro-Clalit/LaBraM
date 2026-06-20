# Dev Plan — Codebook-Regularized Fine-Tuning

## 1. Goal

Add an **opt-in** fine-tuning mode in which the downstream classifier is regularized by
the pre-trained VQNSP tokenizer. During fine-tuning the model additionally runs the EEG
through the **quantizer + decoder**, reconstructs the amplitude/phase spectrum, and adds
**spectral reconstruction losses + a quantization (commitment) loss** to the
classification loss. The intent is to prevent the encoder from overfitting to the
downstream labels and to improve domain adaptation by keeping representations anchored to
the pre-trained spectral manifold.

Re-implementation of `NeurolCodebookClassifier` (`LaBraM-OLD`), with a cleaner,
config-driven design that fits the current package layout, **a strict model/loss split**,
and **configurable classification features**.

## 2. Confirmed design decisions

| Topic | Choice |
|---|---|
| **Weight source** | Two checkpoints: encoder from the pre-trained LaBraM checkpoint (`--finetune`, as today); decoder + quantizer + codebook + projection MLPs from a **separate VQNSP checkpoint** (`vqnsp.pth`). |
| **Default behavior** | Fully opt-in. Disabled ⇒ fine-tuning is byte-for-byte unchanged. |

### 2.1 Trainability (all config-driven)

| Component | Default | Config control |
|---|---|---|
| **Encoder** | **trainable (all layers)** | `encoder.trainable: bool=True`; optional `encoder.n_last_trainable_layers: int|None=None` → when set, only the last *N* transformer blocks (+ norm/head) train, earlier blocks frozen. `None` = all layers. |
| **Decoder** | **frozen** | `decoder.trainable: bool=False` → set `True` to train. |
| **Codebook** | **frozen** | `codebook.trainable: bool=False` → set `True` to re-enable EMA updates (`quantize.embedding.update`). |

### 2.2 Learning rates (per group)

Per-group LRs are config-defined and **all of encoder / decoder / codebook must use a
smaller LR than the classification (prediction) head**. Concretely:

- `head_lr` = base fine-tune LR (largest).
- `encoder_lr`, `decoder_lr`, `codebook_lr` = expressed as multipliers of `head_lr`
  (e.g. `encoder_lr_scale: float=0.1`), validated at startup so each effective LR
  `< head_lr` (warn/raise otherwise).
- Encoder may *also* keep layer-wise decay (`--layer_decay`) **within** its group, on top
  of `encoder_lr`. Decoder/codebook get flat group LRs (no block decay).
- Frozen groups contribute **no** optimizer param group.

## 3. Strict model / loss split

A hard separation (per your remark):

- **Model only predicts** — `forward` returns a structured prediction object, never a
  scalar loss. The quantizer's commitment term is a byproduct of its forward, so it is
  *carried* in the prediction object as data, not as "the loss".
- **Losses are independent `nn.Module`s** — they consume the prediction object (+ target)
  and return a structured **breakdown** object with `total` and each component.
- **Training loop** wires them and steps the optimizer via an extracted grad-update
  helper (§7).

### 3.1 Prediction container
`labram/models/outputs.py`

```python
@dataclass
class PredictorOutput:
    logits: torch.Tensor                 # [B, num_classes]
    recon_magnitude: torch.Tensor | None # [B, N*A, decoder_out_dim]
    recon_phase: torch.Tensor | None
    quantize_loss: torch.Tensor | None   # scalar (carried, not weighted here)
    # raw inputs needed by the loss module to build spectral targets:
    x_patched: torch.Tensor              # [B, N, A, T] (for FFT targets)
```

### 3.2 Loss breakdown container
`labram/losses/outputs.py`

```python
@dataclass
class LossBreakdown:
    total: torch.Tensor
    components: dict[str, torch.Tensor]  # {'classifier','magnitude','phase','quantize'}
    def to_log_dict(self, split: str) -> dict[str, float]: ...
```

## 4. Model: `CodebookRegularizedClassifier`
`labram/models/codebook_classifier.py`

Composes (does **not** subclass/freeze a whole `VQNSP`):

- `encoder: NeuralTransformer` (trainable per §2.1), loaded from `--finetune`.
- `quantize: NormEMAVectorQuantizer` (codebook frozen by default), from VQNSP ckpt.
- `decoder: NeuralTransformer` (frozen by default), from VQNSP ckpt.
- `encode_task_layer`, `decode_task_layer_magnitude`, `decode_task_layer_phase` — VQNSP
  projection MLPs (see rename §6).
- `feature_embedders: nn.ModuleDict` — one per enabled feature source (§5).
- `classifier_head: Linear | MlpClassifier` over the concatenated feature dim.

**Single encoder forward** drives everything (no double compute):

```
x [B, N, A, T]
  └─ encoder.forward_features(..., return_patch_tokens=True)  → patch_tokens [B, N*A, D]
        ├── feature sources (enabled subset, see §5) → embed → concat → norm → head → logits
        └── recon path (mirrors VQNSP.encode/decode, via shared helper §4.1):
              encode_task_layer → rearrange → quantize(frozen) → (quantize_loss, codes, quantized)
              decoder(quantized, ...) → decode_task_layer_magnitude/phase → recon_magnitude/phase
```

`forward` returns a `PredictorOutput`. An `inference`/eval path can skip the recon branch
(classification only) for speed.

### 4.1 Shared encode/decode helper
`VQNSP.encode`/`decode` and this model share the "patch-tokens → quantize → decode →
magnitude/phase" body. Extract it into reusable functions/mixin in `labram/models/` so
there is a single source of truth (better than OLD, which duplicated/wrapped).

### 4.2 Calibration caveat
`encode_task_layer`/codebook were trained against the *VQNSP* encoder, not the masked-EEG
pre-trained encoder. The pre-trained encoder is index-prediction-aligned to this codebook,
so grafting is sound; the small init mismatch is absorbed by the trainable encoder + small
aux-loss weights. Document it.

## 5. Configurable classification features (concat of sources)
`labram/layers/feature_embedders.py` (port from OLD `layers/embedding_blocks.py`)

`feature_sources: list[str]` — concatenated; **default `["encoder_mean"]` only** (==
today's behavior). Supported:

| Source key | Meaning | From | Embedder |
|---|---|---|---|
| `encoder_mean` *(default)* | encoder patch tokens, **mean over chunks** | `patch_tokens [B,N*A,D]` | `FeatureEmbedder(reduce_dim=1)` |
| `quantize_mean` | quantized codes, **mean over chunks** | `quantized [B,N*A,Dq]` | `FeatureEmbedder(reduce_dim=1)` |
| `bag_of_codes` | **normalized bag-of-codes statistics** | `codebook_ind [B,N*A]` | `CodeBookBagEmbedder` (`nn.EmbeddingBag`) |

Each source → its embedder (out `features_emb_dim`) → `torch.cat(dim=-1)` → optional
`LayerNorm(features_emb_dim * n_sources)` → head. With only `encoder_mean` enabled and a
linear identity embedder, this reduces to the current mean-pool + head path (regression
guard in tests).

## 6. Rename (touches VQNSP + checkpoints)

- `decode_task_layer`        → `decode_task_layer_magnitude`
- `decode_task_layer_angle`  → `decode_task_layer_phase`

Apply in `labram/models/vqnsp.py` and everywhere referenced. **Checkpoint compat:** add a
key-remap in the VQNSP/predictor checkpoint loader (`decode_task_layer.` →
`decode_task_layer_magnitude.`, `decode_task_layer_angle.` → `decode_task_layer_phase.`)
so existing `vqnsp.pth` files still load. Update any tests referencing the old names.

## 7. Loss module + training-loop split
`labram/losses/codebook_regularized.py`, `labram/train/`

### 7.1 Loss module (independent `nn.Module`)
```python
class CodebookRegularizedCriterion(nn.Module):
    def __init__(self, classification_criterion, loss_cfg): ...
    def forward(self, out: PredictorOutput, target) -> LossBreakdown:
        cls = self.classification_criterion(out.logits, target)
        amp, ph = self.spectral(out.recon_magnitude, out.recon_phase,
                                *self.spectral.spectrum_targets(out.x_patched))
        comps = {'classifier': cls, 'magnitude': amp, 'phase': ph,
                 'quantize': out.quantize_loss}
        total = (w.classifier*cls + w.amplitude*amp + w.phase*ph + w.embedding*quantize)
        return LossBreakdown(total, comps)
```
Reuses `SpectralReconstructionLoss` and `build_classification_criterion`. Weights from
`LossConfig` (§8). The module is the *only* place losses are summed.

### 7.2 Extracted grad-update step
`labram/train/base.py` — pull the optimizer/scaler/clip/zero-grad block out of
`train_one_epoch` into one reusable function (used by plain and regularized paths):
```python
def optimizer_update(loss, *, optimizer, loss_scaler, parameters,
                     clip_grad, update_grad, create_graph=False) -> float | None:
    "scale→backward→(clip)→step→zero_grad on update boundaries; returns grad_norm"
```
`train_one_epoch` then: `out = model(...)`; `breakdown = criterion(out, target)`;
`grad_norm = optimizer_update(breakdown.total, ...)`; `log.update(breakdown.to_log_dict(split))`.
Evaluate path: classification only, metrics unchanged.

## 8. Config & CLI

1. **`loss_config.py`** — add `classifier_weight: float = 1.0`. Reuse `amplitude_weight`,
   `phase_weight`, `embedding_weight` (quantize), `use_smooth_l1`, `freq_fraction`,
   `vq_commitment_beta`, `classification_label_smoothing`. **Existing defaults unchanged**
   (VQNSP parity). Aux weights set small in fine-tune config; all-zero ⇒ plain fine-tune.
2. **`model_config.py`** — new nested config:
   ```python
   @dataclass
   class ComponentTrainConfig(ConfigBase):
       trainable: bool = True
       lr_scale: float = 1.0            # × head_lr; must be < 1 for enc/dec/codebook
       n_last_trainable_layers: int | None = None  # encoder only

   @dataclass
   class CodebookRegConfig(ConfigBase):
       enabled: bool = False
       tokenizer_model: str = 'vqnsp_encoder_base_decoder_3x200x12'
       tokenizer_weight: str = ''
       feature_sources: list[str] = field(default_factory=lambda: ['encoder_mean'])
       features_emb_dim: int = 128
       linear_embedding: bool = True
       norm_embedding: bool = True
       classifier_type: str = 'linear'     # 'linear' | 'mlp'
       encoder: ComponentTrainConfig = field(default_factory=ComponentTrainConfig)
       decoder: ComponentTrainConfig = field(default_factory=lambda: ComponentTrainConfig(trainable=False, lr_scale=0.1))
       codebook: ComponentTrainConfig = field(default_factory=lambda: ComponentTrainConfig(trainable=False, lr_scale=0.1))
   ```
   Add `codebook_reg: CodebookRegConfig` + aux loss weights to `FinetuneRunConfig`.
3. **Defaults JSON** — leave existing finetune defaults `enabled:false`; add example
   `defaults/finetune_tuab_codebook.json`.
4. CLI via `--config` / `--set codebook_reg.enabled=true ...` (existing `parse_overrides`).

## 9. Assembly & checkpoint loading
`labram/runs/finetune_setup.py`

- Build encoder as today; load `--finetune` into it.
- If `codebook_reg.enabled`: build tokenizer via `create_model(tokenizer_model)`, load
  `tokenizer_weight` (with rename remap §6), extract `quantize` / `decoder` /
  `encode_task_layer` / `decode_task_layer_magnitude` / `decode_task_layer_phase`, assemble
  `CodebookRegularizedClassifier`.
- Apply trainability: freeze per §2.1 (`requires_grad`, encoder last-N, codebook
  `embedding.update=False`); validate `embed_dim` compatibility (fail loudly).
- **Optimizer groups:** head (base LR) + encoder (LR×scale, optional layer decay) +
  decoder (LR×scale) + codebook (if trainable). Exclude frozen params. Validate each
  enc/dec/codebook effective LR `< head_lr`.
- **DDP:** ensure frozen params don't trip unused-grad detection (exclude from optimizer;
  set `find_unused_parameters` if needed).

## 10. Tests
`tests/test_codebook_classifier.py`, extend `tests/test_losses.py`

- `LossConfig.classifier_weight` default 1.0; existing defaults intact.
- `CodebookRegularizedCriterion` → `LossBreakdown`: weighting/sum correct; component keys
  correct; `to_log_dict` keys correct.
- `PredictorOutput` forward on synthetic `[B,N,400]`: `logits` shape; recon tensors finite;
  train & eval (eval skips recon).
- **Feature sources:** each source builds & concats to expected dim; default `encoder_mean`
  + linear-identity == current mean-pool path (regression guard).
- **Trainability matrix:** decoder frozen by default (no grad) / trainable when on;
  codebook `embedding.update` follows config + weight unchanged when frozen; encoder
  `n_last_trainable_layers` freezes the right blocks.
- **LR validation:** enc/dec/codebook LR < head LR enforced; frozen groups absent.
- **Rename/back-compat:** old `vqnsp.pth` keys remap & load; VQNSP still trains.
- `optimizer_update` helper parity with previous inline stepping (grad accumulation,
  clip, scaler).
- **Disabled == baseline** end-to-end regression guard.

## 11. Milestones

1. Rename + checkpoint remap; keep VQNSP/tests green. *(small)*
2. Containers (`PredictorOutput`, `LossBreakdown`) + `classifier_weight`; config dataclasses. *(small)*
3. Extract shared encode/decode helper from `VQNSP`. *(small/med)*
4. Feature embedders port + tests. *(small/med)*
5. `CodebookRegularizedClassifier` forward + tests. *(med)*
6. `CodebookRegularizedCriterion` + `optimizer_update` extraction + tests. *(med)*
7. Assembly, checkpoint grafting, optimizer groups, trainability/LR validation. *(med)*
8. Training-loop integration + logging; eval classify-only. *(med)*
9. Example config + CPU/synthetic smoke run; update `CLAUDE.md`. *(small)*

## 12. Risks / caveats

- Calibration gap (§4.2) — mitigated by trainable encoder + small aux weights.
- Compute/memory: decoder + FFT each step (~VQNSP-forward overhead).
- Layer-decay × per-group LR interaction — make encoder/decoder LRs explicit & validated.
- DDP frozen-param handling.
- `embed_dim` match between pre-trained encoder and VQNSP encoder (200 base) — validate.
- Rename ripples through any external scripts referencing the old task-layer names.
