# LaBraM Refactor Work Plan

Goal: finish the package refactor of `LaBraM`. `LaBraM-OLD` is used **only as a hint** for
how concerns can be separated — we do **not** copy its structure. We propose the optimal
layout for *this* repo. All work happens in `LaBraM`; nothing in `LaBraM-OLD` is touched.

The four requested workstreams map to phases:

| # | Request | Phase | Status |
|---|---------|-------|--------|
| 1 | Move all data-relevant methods into a `data/` folder | **Phase B** | ✅ done |
| 2 | Split/move NN-module methods into a `layers/` folder | **Phase A** | ✅ done |
| 3 | Add a losses config | **Phase C** | ✅ done |
| 4 | Another refactor recommendation based on `LaBraM-OLD` | **Phase D** | pending |

Sequencing rationale: Phase A (layers) first — it is the most self-contained (3 importers + 1
test). Then B (data), then C (losses), then D (config system, the biggest change) last so the
earlier, lower-risk moves land first. Each phase is a standalone PR that keeps `pytest tests/ -v`
green.

---

## Current state (what exists today)

```
labram/
  data_processor/      dataset.py            SingleShockDataset, ShockDataset (HDF5)
                       data_preprocess.py    mask_channels, normalization, collate_mask_time
  models/              layers.py             DropPath, Mlp, Attention, PatchEmbed, TemporalConv
                       att_blocks.py         Block
                       neural_transformer.py NeuralTransformerBase, NeuralTransformer
                       masked_eeg.py         masked-pretrain heads
                       quantizer.py          EmbeddingEMA, NormEMAVectorQuantizer (+ VQ loss)
                       vqnsp.py              VQNSP (+ FFT amplitude/phase recon loss, inline)
                       registry.py           all @register_model factories (3 phases in 1 file)
  optim_factory.py     (top-level)           create_optimizer, LayerDecayValueAssigner, get_parameter_groups
  runners/             common.py, vqnsp.py, pretrain.py, finetune.py,
                       finetune_args.py, finetune_setup.py
                       finetune_datasets.py  DatasetBundle, get_dataset_bundle, _TUH_EEG_CH_NAMES
  utils/               channels.py           standard_1020, get_channel_indices, build_pretraining_dataset
                       datasets_tuh.py       TUHLoader, TUABLoader, TUEVLoader, prepare_TUAB/TUEV_dataset
                       checkpoint.py, cli.py, distributed.py, logging.py, metrics.py, training.py
dataset_maker/         make_TUAB.py, make_TUEV.py, make_h5dataset_for_pretrain.py, shock/...
```

Two problems this plan fixes:
- **Data code is scattered** across `data_processor/`, `utils/channels.py`, `utils/datasets_tuh.py`,
  and `runners/finetune_datasets.py`.
- **Layer primitives and losses are tangled** into model files (`Block` in `att_blocks.py`,
  reconstruction loss inline in `vqnsp.py`, VQ loss inline in `quantizer.py`, classification
  criterion inline in `runners/finetune.py`).

Import blast-radius (verified) is small, which keeps every phase low-risk:
- `models.layers` ← imported by `att_blocks.py`, `neural_transformer.py`, `tests/test_modeling_finetune.py`
- `models.att_blocks` ← `neural_transformer.py`, `tests/test_modeling_finetune.py`
- `data_processor.dataset` ← `utils/channels.py`, `tests/test_datasets.py`
- `utils.channels` / `datasets_tuh` ← re-exported via `utils/__init__.py`; `tests/test_channels.py`
- `runners.finetune_datasets` ← `runners/finetune.py`
- `optim_factory` ← `runners/{pretrain,vqnsp,finetune}.py`

---

## Target package layout (proposed — optimal for this repo)

```
labram/
  layers/              # Phase A — NN primitives, one responsibility per module
    __init__.py        #   re-exports DropPath, Mlp, Attention, Block, PatchEmbed, TemporalConv
    drop_path.py       #   DropPath
    mlp.py             #   Mlp
    attention.py       #   Attention
    transformer_block.py  #   Block (composes Attention + Mlp + DropPath)
    patch_embed.py     #   PatchEmbed, TemporalConv
  data/                # Phase B — all data concerns
    __init__.py        #   public re-exports
    eeg_constants.py   #   standard_1020, TUH_EEG_CH_NAMES, sampling-rate consts,
                       #   get_channel_indices, normalize_ch_names
    hdf5_datasets.py   #   SingleShockDataset, ShockDataset
    tuh_datasets.py    #   TUHLoader, TUABLoader, TUEVLoader, prepare_TUAB/TUEV_dataset
    bundles.py         #   DatasetBundle, get_dataset_bundle
    preprocess.py      #   mask_channels, normalization, collate_mask_time
    pretraining.py     #   build_pretraining_dataset
  losses/              # Phase C — losses + losses config
    __init__.py
    config.py          #   @dataclass LossConfig  ← the "losses config"
    spectral.py        #   SpectralReconstructionLoss (FFT amplitude + phase)
    vqnsp.py           #   get_vqnsp_losses(...) -> dict of weighted components
    classification.py  #   build_classification_criterion(nb_classes, label_smoothing)
  config/              # Phase D (recommendation) — dataclass config system
    __init__.py
    base.py            #   ConfigBase (as_dict / load / save / update)
    serialization.py   #   YAML+JSON load/save
    data.py, model.py, optim.py, train.py, run.py
  models/              # slimmed: neural_transformer, masked_eeg, quantizer, vqnsp, registry/
  optim/               # Phase D — optim_factory.py moves here -> labram/optim/factory.py
  runners/             # slimmed: common, vqnsp, pretrain, finetune, finetune_args, finetune_setup
  utils/               # slimmed: checkpoint, cli, distributed, logging, metrics, training
dataset_maker/         # unchanged (standalone CLI scripts, not library code)
```

---

## Phase A — `labram/layers/` (Request #2)

Split the monolithic `models/layers.py` + `models/att_blocks.py` into one module per
responsibility. (`LaBraM-OLD` proves the granularity works: it split into
`attention_blocks / mlp_blocks / patch_conv_blocks / norm_layers / embedding_blocks`. We
keep only the modules this repo actually needs.)

**Moves**
1. Create `labram/layers/` package.
2. `models/layers.py::DropPath` → `layers/drop_path.py`
3. `models/layers.py::Mlp` → `layers/mlp.py`
4. `models/layers.py::Attention` → `layers/attention.py`
5. `models/layers.py::{PatchEmbed, TemporalConv}` → `layers/patch_embed.py`
6. `models/att_blocks.py::Block` → `layers/transformer_block.py` (imports Attention, Mlp, DropPath from siblings)
7. `layers/__init__.py` re-exports all six symbols so `from labram.layers import Block, Attention, ...` works.

**Import updates**
- `models/neural_transformer.py`: `from labram.models.att_blocks import Block` →
  `from labram.layers import Block`; `from labram.models.layers import PatchEmbed, TemporalConv` →
  `from labram.layers import PatchEmbed, TemporalConv`.
- `tests/test_modeling_finetune.py`: update the two import lines to `from labram.layers import ...`.

**Cleanup**
- Delete `models/layers.py` and `models/att_blocks.py`.
- *Optional safety shim:* leave `models/layers.py` and `models/att_blocks.py` as one-line
  re-exports from `labram.layers` for one release, then delete in a follow-up. Recommended only
  if external code imports them; internal code does not.

**Verify**: `pytest tests/test_modeling_finetune.py -v`, then `pytest tests/ -v`.

---

## Phase B — `labram/data/` (Request #1)

Consolidate every data-relevant symbol scattered across `data_processor/`, `utils/`, and
`runners/` into one `labram/data/` package. (`LaBraM-OLD` grouped the same concerns under
`data/eeg_consts.py + hdf5_datasets.py + patch_datasets.py`.)

**Moves**
| From | To |
|------|----|
| `data_processor/dataset.py` (`SingleShockDataset`, `ShockDataset`) | `data/hdf5_datasets.py` |
| `data_processor/data_preprocess.py` (`mask_channels`, `normalization`, `collate_mask_time`) | `data/preprocess.py` |
| `utils/channels.py` (`standard_1020`, `get_channel_indices`) | `data/eeg_constants.py` |
| `utils/channels.py` (`build_pretraining_dataset`) | `data/pretraining.py` |
| `utils/datasets_tuh.py` (`TUHLoader`, `TUABLoader`, `TUEVLoader`, `prepare_TUAB/TUEV_dataset`) | `data/tuh_datasets.py` |
| `runners/finetune_datasets.py` (`DatasetBundle`, `get_dataset_bundle`, `_TUH_EEG_CH_NAMES`, `_normalize_ch_names`) | `data/bundles.py` + consts to `data/eeg_constants.py` |

Consolidate the two channel-name constants (`standard_1020` and `_TUH_EEG_CH_NAMES`) and the
two normalization helpers (`get_channel_indices`, `_normalize_ch_names`) into
`data/eeg_constants.py` — today they live in two different files.

**Import updates**
- `data/pretraining.py`: import `ShockDataset` from `labram.data.hdf5_datasets` (was `data_processor.dataset`).
- `runners/finetune.py`: `from labram.runners.finetune_datasets import get_dataset_bundle` →
  `from labram.data import get_dataset_bundle`.
- `runners/finetune_setup.py` and any channel-reorder code: import `get_channel_indices` /
  `standard_1020` from `labram.data` (they currently come via `labram.utils`).
- `utils/__init__.py`: **keep the public surface** by re-exporting the moved data names from
  `labram.data` (transitional back-compat), so existing `from labram.utils import standard_1020,
  get_channel_indices, build_pretraining_dataset, TUABLoader, ...` keep working. Mark as
  deprecated in a comment; remove in a later cleanup.
- Tests: `tests/test_datasets.py` → `from labram.data.hdf5_datasets import ...`;
  `tests/test_channels.py` → `from labram.data.eeg_constants import ...`.

**Cleanup**
- Delete `labram/data_processor/`, `labram/utils/channels.py`, `labram/utils/datasets_tuh.py`,
  `labram/runners/finetune_datasets.py`.
- `data/__init__.py` re-exports the public API (datasets, loaders, constants, bundle factory).

**Decision point**: leave `dataset_maker/` where it is. It is a set of standalone preprocessing
CLI scripts (run directly, not imported), exactly like `LaBraM-OLD/preproc_data/`. Moving it into
the importable `labram.data` package would conflate library code with scripts. (Flag if you want
it under `labram/data/scripts/` instead.)

**Verify**: `pytest tests/test_datasets.py tests/test_channels.py tests/test_finetune_setup.py -v`,
then `pytest tests/ -v`.

---

## Phase C — `labram/losses/` + losses config (Request #3)

Today losses are inline and un-configurable:
- `vqnsp.py`: `loss = embedding_loss + amplitude_loss + angle_loss` (implicit weight 1.0 each;
  `smooth_l1` vs `mse` chosen by a constructor flag).
- `quantizer.py`: `loss = self.beta * F.mse_loss(z_q.detach(), z)`.
- `runners/finetune.py`: `BCEWithLogitsLoss` / `LabelSmoothingCrossEntropy` / `CrossEntropyLoss`.
- `train_pretrain.py`: `nn.CrossEntropyLoss()` (MLM, `loss_rec + loss_rec_sym`).

(`LaBraM-OLD` solved this with `train/losses.py::{SpectralPatchedLoss, get_vqnsp_losses}` plus a
`LossesWeightsConfig` dataclass holding `phase_recon / magnitude_recon / quantize_err /
classifier / frequency_cutoff` weights.)

**New module `labram/losses/`**
1. `config.py` — **the losses config**:
   ```python
   @dataclass
   class LossConfig:
       amplitude_weight: float = 1.0
       phase_weight: float = 1.0
       vq_commitment_beta: float = 1.0      # was quantizer beta
       use_smooth_l1: bool = False          # was the smooth_l1_loss flag
       classification_label_smoothing: float = 0.0
   ```
2. `spectral.py` — `SpectralReconstructionLoss(nn.Module)`: FFT → amplitude/phase →
   `mse` or `smooth_l1` (from `LossConfig.use_smooth_l1`). Extracted verbatim from
   `vqnsp.py` so numerics are unchanged.
3. `vqnsp.py` — `get_vqnsp_losses(x, decoder_out, embedding_loss, cfg) -> dict`: returns
   `{"amplitude": ..., "phase": ..., "embedding": ..., "total": ...}` with weights applied
   from `LossConfig`. Mirrors `LaBraM-OLD`'s dict-returning aggregator.
4. `classification.py` — `build_classification_criterion(nb_classes, cfg)`: the
   BCE/CE/LabelSmoothing selection currently inline in `runners/finetune.py`.

**Wiring**
- `models/vqnsp.py`: replace inline FFT+recon loss with `SpectralReconstructionLoss`; accept a
  `LossConfig` (default `LossConfig()` preserves current behavior).
- `models/quantizer.py`: read `beta` from `LossConfig.vq_commitment_beta` (keep default 1.0).
- `runners/finetune.py`: replace the inline criterion block with
  `build_classification_criterion(...)`.
- Thread `LossConfig` through the runner/model construction with defaults equal to today's
  hard-coded values — **no behavior change** in this phase; it just makes weights configurable.

**Verify**: `pytest tests/test_engine_vqnsp.py tests/test_engine_pretraining.py tests/test_finetune.py -v`,
then `pytest tests/ -v`. Numeric parity is the acceptance criterion — defaults must reproduce
current losses exactly.

---

## Phase D — config system + smaller cleanups (Request #4)

The single biggest structural difference between this repo and `LaBraM-OLD` is configuration.
This repo drives everything through scattered `argparse` (`finetune_args.py`, per-runner
`add_argument` calls); `LaBraM-OLD` uses a typed, serializable, hierarchical dataclass system
(`configs/` with `ConfigBase`, `ConfigProcEEGDataset`, `ConfigNeuralTransformer`,
`ConfigClassifierOptimizer`, `ConfigClassifierTrainer`, `LossesWeightsConfig`, `LoggerParams`,
all composed under `ConfigRunClassifierModel`, with YAML/JSON serialization).

**Primary recommendation — `labram/config/` package** (do this as its own PR, after A–C):
- `base.py` — `ConfigBase` dataclass mixin: `as_dict()`, `load_from(path)`, `save_to(path)`,
  `update(**dotted_kwargs)`.
- `serialization.py` — YAML + JSON load/save (the `LaBraM-OLD` pattern, without `jsonpickle`).
- `data.py / model.py / optim.py / train.py` — one config dataclass per concern; `train.py`
  embeds the `LossConfig` from Phase C (so the losses config slots straight in).
- `run.py` — top-level `RunConfig` composing the above via `field(default_factory=...)`.
- Migration strategy: build configs **alongside** argparse first (argparse populates a config,
  config drives the code). This lets runners accept either `--config run.yaml` or the existing
  flags, so nothing breaks for current users. Retire redundant flags later.
- Use enums for choice-typed options (optimizer, norm layer, dataset name) instead of bare strings.

**Smaller cleanups (each a tiny independent PR, all hinted by `LaBraM-OLD`)**
1. **`optim_factory.py` → `labram/optim/factory.py`.** It is the only top-level module left in the
   package; move it into an `optim/` package (mirrors `LaBraM-OLD/train/optimizers.py`). Update
   the 3 runner imports.
2. **Split `models/registry.py` by phase** into `models/registry/{vqnsp,pretrain,finetune}.py`
   (re-exported from `models/registry/__init__.py`) — `LaBraM-OLD` keeps three registry files. Do
   this only if the single file keeps growing; it is the lowest-priority item.
3. **`models/io.py` for checkpoint/weight loading.** Consolidate `vqnsp.py::load_vqnsp_weights`,
   the channel-reorder logic in `runners/finetune_setup.py`, and `utils/checkpoint.py`'s
   model-weight helpers into one `models/io.py` (mirrors `LaBraM-OLD/models/models_io.py`). Keeps
   the transfer-learning weight surgery in one reviewable place.

---

## Cross-cutting conventions

- **One PR per phase**, each green on `pytest tests/ -v`. Matches the repo's existing refactor
  cadence (recent history: "split models by abstraction, rename engines/ to trainers/").
- **Pure moves, not rewrites.** Preserve symbol names and numerics; only relocate + re-wire
  imports. Verify with `git diff --stat` that moved files are mostly unchanged.
- **Back-compat shims** at old import sites (`utils/__init__.py` re-exports; optional
  `models/layers.py` re-export) during transition, removed in a final cleanup PR.
- **Update `AGENTS.md` and `CLAUDE.md`** "Repository Layout" sections at the end of each phase so
  the docs never drift from the tree.
- Branch off `main` for each PR (current branch is a worktree branch).

## Suggested commit/PR sequence
1. `refactor: extract NN primitives into labram/layers/` (Phase A)
2. `refactor: consolidate data code into labram/data/` (Phase B)
3. `feat: add labram/losses/ with configurable LossConfig` (Phase C)
4. `refactor: relocate optim_factory into labram/optim/` (Phase D.1)
5. `feat: dataclass config system in labram/config/` (Phase D, primary)
6. `refactor: models/io.py + registry split + remove back-compat shims` (Phase D.2/3, cleanup)

## Open decisions (recommended defaults in **bold**)
- `dataset_maker/`: **leave as top-level scripts** vs. move under `labram/data/scripts/`.
- Back-compat shims: **keep for one release** vs. hard cutover now.
- Phase D config system: **separate follow-up PR after A–C** vs. in-scope now.
