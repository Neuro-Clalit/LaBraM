# Duplicate logic audit (post-restructure)

Snapshot at PR #25 (the `labram/` package restructure). Findings are notes
for follow-up work; this PR does **not** fix any of them so the move stays
mechanical and reviewable.

## 1. `forward_features` in `NeuralTransformerForMaskedEEGModeling` (`labram/models/pretrain.py`)

Re-implements ~95% of `NeuralTransformer._embed_inputs` (`labram/models/finetune.py`)
with one addition: mask-token splicing between `patch_embed` and the cls
concat.

```python
# labram/models/pretrain.py  (NeuralTransformerForMaskedEEGModeling.forward_features)
batch_size, c, time_window, _ = x.size()
x = self.patch_embed(x)
batch_size, seq_len, _ = x.size()
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
mask_token = self.mask_token.expand(batch_size, seq_len, -1)
mask_weight = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
x = x * (1 - mask_weight) + mask_token * mask_weight    # <-- unique
x = torch.cat((cls_tokens, x), dim=1)
pos_embed_used = self.pos_embed[:, channel_indices] if channel_indices is not None else self.pos_embed
if self.pos_embed is not None:
    pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
    pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
    x = x + pos_embed
if self.time_embed is not None:
    time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
    x[:, 1:, :] += time_embed
x = self.pos_drop(x)
```

**Suggested fix:** add an optional `mask_token` / `bool_masked_pos`
parameter to `NeuralTransformer._embed_inputs`, or pull both classes onto a
shared base class that holds the embedding setup. Bonus: the pretrain
version has a latent bug at the pos_embed concat line (uses `pos_embed[:,0:1,:]`
instead of `pos_embed_used[:,0:1,:]`, which would surface only if the
caller passed a sliced `channel_indices` — and a single fix point would
make the bug obviously wrong-only-in-one-place.

## 2. `forward_return_qkv` and `get_last_selfattention` in the same class

Both methods (lines 143–202 of pre-restructure `modeling_pretrain.py`) are
**dead code** — zero callers in the codebase (`grep -rn` confirmed). Each
re-implements yet another copy of the patch + cls + pos_embed setup, with
no `channel_indices` support, so they would not work for any model trained
on fewer than 128 channels.

**Suggested fix:** delete in a follow-up cleanup, or rewire through the
shared `_embed_inputs` helper. Removing them shrinks `labram/models/pretrain.py`
by ~60 lines with zero behavioral impact.

## 3. `NeuralTransformer` ↔ `NeuralTransformerForMaskedEEGModeling` shared init scaffolding

The two classes (now in `labram/models/finetune.py` and
`labram/models/pretrain.py`) share substantial setup code:

- `cls_token` / `mask_token` / `pos_embed` / `time_embed` parameter
  allocation
- `blocks = nn.ModuleList([Block(...) for i in range(depth)])`
- `_init_weights`, `fix_init_weight`, `no_weight_decay`, `get_num_layers`,
  `trunc_normal_` on the same set of leaves

**Suggested fix:** carve a `NeuralTransformerBase` (in
`labram/models/_base.py` or similar) holding the shared init + weight-init
+ no_weight_decay surface. `NeuralTransformer` adds the classification head
+ `_embed_inputs`. `NeuralTransformerForMaskedEEGModeling` adds the LM
head + mask-token splicing. Saves ~80 LOC and pins the duplication so it
cannot drift.

## 4. `EmbeddingEMA` and `NormEMAVectorQuantizer` (`labram/models/norm_ema_quantizer.py`)

No duplicates elsewhere in the codebase — domain-specific to the VQ-NSP
tokenizer. Self-contained; no action needed.

## 5. `TemporalConv`, `Block`, `Mlp`, `Attention`, `DropPath`, `PatchEmbed`, `_cfg`

Already consolidated. `labram/models/pretrain.py` imports `Block`,
`TemporalConv`, and `_cfg` from `labram.models.finetune`. No duplicates.

## 6. Cross-engine duplication (`labram/engines/{finetune,pretrain,vqnsp}.py`)

Each `train_one_epoch` runs the same metric-logger boilerplate
(`metric_logger.add_meter('lr', ...)`, `metric_logger.add_meter('min_lr', ...)`,
the lr/wd update loop, the loss-scaler call, the min_lr/max_lr accumulator
loop). They diverge on:

- whether they receive a `vqnsp` model (pretrain only)
- whether they iterate over a list-of-dataloaders (pretrain/vqnsp) or a
  single dataloader (finetune)
- whether they iterate gradient-accumulation micro-steps (pretrain only)

**Suggested fix:** a `_step_progress` context (mutates the metric logger
post-step) or a `Trainer`-style base class. Lower ROI than #1–#3; the
boilerplate is small per engine and the divergence points are real.

## Out of scope for this PR

This PR only restructures files into `labram/`. The dedup work above is
catalogued so the next PR can target whichever item gives the best ROI.
Recommended order: **#2** (delete dead code, free win) → **#1** (extend
`_embed_inputs`, removes a latent bug at the same time) → **#3** (shared
base class).
