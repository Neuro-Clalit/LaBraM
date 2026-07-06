# Design-review audit (`labram/` package)

Snapshot produced by the vendored `design_reviewer` skill
(`.claude/skills/design_reviewer/`) run over the whole package:

```bash
python .claude/skills/design_reviewer/scripts/design_review_metrics.py --format json labram/
```

The findings are notes for follow-up work. **This PR does not fix any of them** —
it installs the skill + the tooling and captures the first audit so the move
stays reviewable, exactly as `docs/duplicate-logic-audit.md` did for the
restructure. Each item below is scoped as an independent follow-up PR.

## Verdict & headline metrics

| Metric | Value | Gate | Status |
| --- | --- | --- | --- |
| **Overall verdict** | `reject` | — | hard gate breached |
| Max McCabe | **34** — `optim_factory.py:109 create_optimizer` | warn > 7, **reject > 12** | reject |
| Min Maintainability Index | **36.2** — `utils/logging.py` | warn < 65 | warn |
| Lint errors (ruff `E`+`F`) | **9** | 0 | warn |
| Functions over McCabe 12 | **12** | — | reject band |
| Functions McCabe 8–12 | **17** | — | warn band |
| Modules with MI < 65 | **28** | — | warn band |

The `reject` verdict is driven entirely by cyclomatic complexity: twelve
functions exceed the McCabe-12 hard gate. MI and lint are advisory (`warn`).
Numbers reproduce from the JSON output above; every location below is a real
`file:line` in the current tree.

## 1. Structural complexity — the `reject` drivers (McCabe > 12)

The offenders cluster in three areas: the **optimizer factory**, the **per-phase
train loops**, and the **run/setup entry points**. These are the highest-ROI
targets.

| McCabe | Location | Function |
| --- | --- | --- |
| 34 | `labram/optim_factory.py:109` | `create_optimizer` |
| 27 | `labram/runs/run_finetune.py:59` | `main` |
| 23 | `labram/train/train_finetune.py:55` | `train_one_epoch` |
| 23 | `labram/utils/checkpoint.py:147` | `auto_load_model` |
| 19 | `labram/configs/common/types.py:85` | `correct_type` |
| 17 | `labram/train/train_finetune.py:268` | `train_loop` |
| 16 | `labram/train/train_vqnsp.py:31` | `train_one_epoch` |
| 15 | `labram/train/train_pretrain.py:39` | `train_one_epoch` |
| 14 | `labram/train/train_vqnsp.py:186` | `train_loop` |
| 14 | `labram/runs/finetune_setup.py:134` | `load_finetune_checkpoint` |
| 13 | `labram/optim_factory.py:59` | `get_parameter_groups` |
| 13 | `labram/utils/training.py:28` | `get_grad_norm_` |

**Suggested fixes:**

- **`create_optimizer` (34) + `get_parameter_groups` (13)** — the single worst
  hotspot. Split parameter-group construction (weight-decay skip-list, layer
  decay via `LayerDecayValueAssigner`, per-group LR scaling) from optimizer
  instantiation (the `opt` name → optimizer-class dispatch). The dispatch is a
  long `if/elif` chain that maps cleanly to a small lookup table.
- **`train_one_epoch` across the three engines (23 / 16 / 15)** — this is the
  same cross-engine duplication already catalogued as item #6 in
  `docs/duplicate-logic-audit.md`. Each re-implements the lr/wd schedule update,
  the loss-scaler call, the min/max-lr accumulator, and metric-logger
  boilerplate. Carving a shared step helper (or a `Trainer`-style base) removes
  three copies at once and drops all three under the gate.
- **`run_finetune.py:main` (27)** — an entry point doing argument wiring, model
  build, checkpoint load, and dispatch inline. Extract the setup phases
  (already partly factored into `runs/finetune_setup.py`) so `main` becomes
  orchestration only.
- **`auto_load_model` (23)** — flatten the resume-path discovery / DeepSpeed vs
  plain branching into small named helpers.
- **`correct_type` (19)** in `configs/common/types.py` — the config coercion
  ladder; a type→coercer dispatch table replaces the branch cascade and makes
  each rule unit-testable in isolation.

## 2. Maintainability Index (MI < 65, advisory)

28 modules fall below MI 65; the tail correlates with the complexity hotspots
above (fixing §1 will lift most of these). Worst offenders:

| MI | Module |
| --- | --- |
| 36.2 | `labram/utils/logging.py` |
| 37.2 | `labram/optim_factory.py` |
| 37.4 | `labram/train/train_vqnsp.py` |
| 40.1 | `labram/utils/checkpoint.py` |
| 40.6 | `labram/train/train_finetune.py` |
| 43.0 | `labram/train/train_pretrain.py` |
| 45.0 | `labram/configs/base_configs.py` |
| 45.7 | `labram/utils/distributed.py` |
| 46.7 | `labram/models/neural_transformer.py` |

`utils/logging.py` (the `MetricLogger` / `TensorboardLogger` / `ClearMLLogger` /
`MultiWriter` stack) is the lowest — a candidate for splitting the writer
back-ends into separate modules. No standalone action is required beyond the §1
work plus these targeted splits.

## 3. Static hygiene (9 ruff `E`+`F` findings, mechanical)

All nine are trivially auto-fixable (`ruff check --fix`) except the two
ambiguous-name `E741`s and the `E731` lambda, which want a one-line manual edit:

| Location | Rule | Issue |
| --- | --- | --- |
| `labram/configs/base_configs.py:10` | F401 | `OPT_STR` imported but unused |
| `labram/configs/run_configs.py:22` | F401 | `typing.List` imported but unused |
| `labram/utils/training.py:8` | F401 | `typing.Any` imported but unused |
| `labram/models/quantizer.py:37` | F841 | local `device` assigned but never used |
| `labram/runs/run_finetune.py:114` | F841 | local `window_size` assigned but never used |
| `labram/train/train_pretrain.py:32` | F841 | local `x_masked` assigned but never used |
| `labram/configs/common/types.py:132` | E731 | lambda assignment — use `def` |
| `labram/models/neural_transformer.py:267` | E741 | ambiguous variable name `l` |
| `labram/models/neural_transformer.py:277` | E741 | ambiguous variable name `l` |

The three `F841` unused-locals are worth a human look before deletion — a
dropped `x_masked` / `window_size` can indicate an intended-but-missing use
rather than pure dead code.

## 4. AI-readiness & testability (qualitative)

- **Semantic duplication** — the three `train_one_epoch`/`train_loop` pairs are
  the dominant duplication (see §1 and audit item #6). Consolidating them both
  cuts complexity and removes the drift risk.
- **Blast radius** — `optim_factory.py` and `utils/checkpoint.py` are imported
  across every run entry point, so their high complexity has wide reach;
  prioritise them.
- **Testability seams** — the monolithic `train_one_epoch` functions mix the
  step math with logging/DDP side effects, so the numeric core isn't assertable
  without a full loop. Extracting a pure step helper (§1) also creates the seam
  a focused unit test needs.

## Prioritised remediation plan (recommended order)

Each is an independent follow-up PR; ordered by ROI (impact on the `reject`
verdict per unit of change):

1. **Hygiene sweep** (§3) — `ruff check --fix` + 3 manual edits. Near-zero risk,
   clears all 9 lint findings, makes future diffs quieter. *(fastest win)*
2. **`optim_factory.create_optimizer` / `get_parameter_groups`** (§1) — split
   param-group build from optimizer dispatch (table). Removes the McCabe-34 and
   -13 peaks and lifts the module's MI. *(biggest single complexity drop)*
3. **Shared train-step helper** (§1, audit item #6) — factor the common
   epoch-loop body out of the three engines. Drops three `reject`-band functions
   at once, kills the largest duplication, and adds a unit-test seam.
4. **Entry-point / setup extraction** — `run_finetune.main`, `auto_load_model`,
   `load_finetune_checkpoint`: push logic into helpers so entry points
   orchestrate only.
5. **`configs/common/types.correct_type`** — dispatch table for type coercion;
   makes each rule independently testable.
6. **Module splits for the lowest-MI files** (§2) — e.g. break `utils/logging.py`
   writer back-ends apart. Do last; mostly falls out of 1–5.

## Out of scope for this PR

This PR only **installs the `design_reviewer` skill + its tooling** (radon/ruff
dev deps, `[tool.ruff]` config, permission allowlist) and records this audit
snapshot. No production code under `labram/` is modified. Re-run the command at
the top after any remediation PR to watch the verdict move from `reject` →
`revise` → `approve`.
