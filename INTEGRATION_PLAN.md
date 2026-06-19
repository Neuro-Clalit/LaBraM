# Integration Plan: `feature/add_configs` ↔ refactor A+B+C

## TL;DR

- **Nothing was deleted.** `feature/add_configs` is an *unmerged* branch off the same base
  (`37de8af`) as my refactor. `main` never had configs; my PRs branch from `main`, so they
  don't either. The config work is intact on its branch.
- The two efforts are **orthogonal in intent** (configs vs. layers/data/losses extraction) but
  **collide structurally** because both reorganized `models/`, `runners/`, `trainers/`.
- **Good news:** `feature/add_configs` left *all loss logic at the original baseline* (inline
  VQNSP loss, inline criterion selection). So Phase C re-applies almost verbatim. Phases A/B are
  mechanical renames + import updates.
- **Recommended:** land configs first, then re-apply A+B+C on top → one reconciled trunk
  (Option 1 below). A blind `git merge` is the wrong tool here.

## The two branches (both off base `37de8af` = merge of PR #26)

| | `feature/add_configs` (6 commits) | refactor A+B+C — PRs #27/#28 (2 commits) |
|---|---|---|
| **Adds** | `labram/configs/` (ConfigBase + JSON/YAML, model/data/optim/train/runner configs, `defaults/*.json`, cloud FS/S3) | `labram/layers/`, `labram/data/`, `labram/losses/` |
| **Renames dirs** | `runners/` → `runs/`, `trainers/` → `train/` | (keeps `runners/`, `trainers/`) |
| **Restructures** | runner entrypoints → `runs/run_{vqnsp,pretrain,finetune}.py`; **config-driven model/trainer constructors** (`VQNSP(config)`, `NeuralTransformer(cfg)`, `NormEMAVectorQuantizer(q_cfg)`) | splits `models/layers.py`+`att_blocks.py` → `layers/`; consolidates data; extracts losses |
| **Loss logic** | **unchanged from baseline** (inline FFT loss; `config.smooth_l1_loss`, `QuantizerConfig.beta`; inline BCE/CE/LabelSmoothing reading `config.smoothing`) | extracted to `labram/losses/` (`LossConfig`, `SpectralReconstructionLoss`, `get_vqnsp_losses`, `build_classification_criterion`) |

## Conflict inventory (from a real 3-way merge simulation)

`git merge-tree --merge-base=37de8af claude/losses-config origin/feature/add_configs` →
**6 conflicts**:

| # | Path | Type | Resolution |
|---|------|------|------------|
| 1 | `models/neural_transformer.py` | content | Take theirs' config-driven ctor; swap layer imports to `from labram.layers import Block, PatchEmbed, TemporalConv` |
| 2 | `models/vqnsp.py` | content | Take theirs' config-driven ctor; **re-apply Phase C** (use `SpectralReconstructionLoss` + `get_vqnsp_losses`; source weights/`use_smooth_l1`/`beta` from configs) |
| 3 | `runners/finetune.py` | modify/delete | Theirs deleted it (→ `runs/run_finetune.py`). Discard my edit here; re-apply the criterion change inside `runs/run_finetune.py` |
| 4 | `runners/finetune_datasets.py` | rename/rename | Mine → `data/bundles.py`; theirs → `runs/finetune_datasets.py`. **Keep `data/bundles.py`** (data concern); update `runs/run_finetune.py` import |
| 5 | `train/train_finetune.py` | content | Take theirs (renamed `trainers/`→`train/` + edits); re-apply `build_classification_criterion` in `evaluate()` |
| 6 | `tests/test_modeling_finetune.py` | content | Combine: theirs' edits + my `from labram.layers import …` import line |

`AGENTS.md` and `CLAUDE.md` auto-merge textually but should be **manually reconciled** (both rewrote the layout section).

## Stale import paths theirs still references (my refactor moved these)

`feature/add_configs` predates Phases A/B, so it imports paths that no longer exist post-refactor:

| Old path (theirs imports) | New home (mine) |
|---|---|
| `labram.models.layers` | `labram.layers` |
| `labram.models.att_blocks` | `labram.layers` (`transformer_block`) |
| `labram.data_processor.dataset` | `labram.data.hdf5_datasets` |
| `labram.utils.channels` | `labram.data.eeg_constants` |
| `labram.utils.datasets_tuh` | `labram.data.tuh_datasets` |

Most occurrences are in the 6 conflicted files (resolved there) or in files only *I* changed
(merge auto-takes my fixed version). **But** files only *theirs* changed (`registry.py`,
`masked_eeg.py`, etc.) and `runs/`/`train/` modules need a post-merge **import sweep** + test run
to catch any lingering old paths.

## Structural decisions for you to make

1. **Directory names:** theirs uses `runs/` + `train/`; mine keeps `runners/` + `trainers/`.
   Pick one convention for the trunk. *(Recommend adopting theirs' `runs/`+`train/` since it's the
   larger restructure, unless you prefer the existing names — note PRs #27/#28 use `runners/`+`trainers/`.)*
2. **Dataset bundles location:** `data/bundles.py` (mine, data concern) vs `runs/finetune_datasets.py`
   (theirs). *Recommend `data/bundles.py`.*
3. **`LossConfig` placement:** consolidate theirs' scattered loss options
   (`VQNSPArchConfig.smooth_l1_loss`, `QuantizerConfig.beta`, finetune `config.smoothing`) into a
   single `configs/loss_config.py::LossConfig` (or fold into `train_config`), and have the model/
   quantizer read from it. This is the natural home for Phase C's `LossConfig` and the new
   amplitude/phase/embedding weights.

## Recommended execution — Option 1 (configs-first trunk)

1. **Merge `feature/add_configs` → `main`** first (self-contained, working branch). *Decision: does
   it pass CI on its own? Verify before merging.*
2. **Create `integration/refactor-on-configs`** off the new `main`.
3. **Re-apply Phase A** (`labram/layers/`): split `models/layers.py`+`att_blocks.py`, update theirs'
   `neural_transformer.py` imports. Mechanical.
4. **Re-apply Phase B** (`labram/data/`): consolidate `data_processor`, `utils/channels`,
   `utils/datasets_tuh`, and theirs' `runs/finetune_datasets.py` → `labram/data/`. Update `utils/__init__`
   re-exports and `runs/run_finetune.py` import.
5. **Re-apply Phase C** (`labram/losses/`) on theirs' config-driven `vqnsp.py`: swap the inline loss
   for `SpectralReconstructionLoss` + `get_vqnsp_losses`; add `configs/loss_config.py::LossConfig`;
   route `runs/run_finetune.py` + `train/train_finetune.py::evaluate` through
   `build_classification_criterion`.
6. **Import sweep + full test run** (`pytest tests/ -v`). Merge both test suites
   (`test_configs_base.py`, `test_runner_configs.py` from theirs; `test_losses.py`,
   `test_checkpoint_loading.py` from mine) and fix imports in theirs' tests.
7. Open one PR superseding #27/#28; close those (or retarget).

Why Option 1: theirs' loss code == baseline, so Phase C re-applies trivially; my layers/data moves
are mechanical renames that drop onto theirs' content with only import edits. Re-applying small
moves onto a big restructure is far easier than the reverse.

## Alternatives

- **Option 2 — A/B/C first, rebase configs on top.** My #27/#28 merge to `main`, then rebase
  `feature/add_configs`. *Harder:* each of theirs' 6 config-driven commits replays against the moved
  files (`models.layers`→`layers`, `runners/`, `trainers/`), conflicting repeatedly.
- **Option 3 — fresh reconciled branch superseding all three.** One integration branch built by
  hand (configs + layers + data + losses, mutually consistent), then close
  `feature/add_configs` + #27 + #28. *Cleanest end state, most up-front work; effectively Option 1's
  steps 2–7 without merging configs to main first.*

## Effort / risk

- **Mechanical (low risk):** layers split, data consolidation, import sweep — covered by existing tests.
- **Moderate:** reconciling config-driven `vqnsp.py`/`neural_transformer.py` ctors with Phase C losses;
  placing `LossConfig` in the config tree; the `runs/`-vs-`runners/` naming decision.
- **Safety nets in place:** `test_checkpoint_loading.py` + `test_engine_*` pin model loading and the
  training loops; `test_losses.py` pins loss parity. Run them after each re-apply step.

## Impact on open PRs

- **#27 (A+B)** and **#28 (C, stacked on #27)** target `main` with `runners/`+`trainers/`. If we adopt
  theirs' `runs/`+`train/`, these PRs are **superseded** by the integration branch — close them with a
  pointer, don't merge both. If we instead keep `runners/`+`trainers/`, #27/#28 stay and configs
  rebases onto them (Option 2).
