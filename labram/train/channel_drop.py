# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Channel-dropout augmentation for a whole batch with a consistency loss to a
# frozen all-channels pass, and per-sample handling of missing channels.
# ---------------------------------------------------------
"""``channel_drop``: one model for recordings with and without some channels.

LaBraM has one token per channel and patch, so a channel is *dropped* by
leaving its tokens out (not by feeding zeros). Two mechanisms, both on when
``channel_drop.enabled``:

* **Missing channels** (train and eval). A channel that is all zero in a
  sample (zero-filled in the npy: e.g. TUEG recordings without T1/T2) is
  missing for that sample. Each batch is split by its present-channel
  patterns and every group runs on its own channel subset; outputs are put
  back in batch order. The encoder has no batch-coupled layers, so this
  equals a per-sample forward.
* **Augmentation** (train only). With probability ``prob`` per batch, draw
  ``n ~ U{min_drop..max_drop}`` channels among ``channels`` (empty = every
  channel) and drop them from every sample of the batch. The training loss is
  computed on this dropped pass and, for the samples that had at least one of
  the dropped channels,

      L = L_task(dropped) + lambda * mean_samples mean_d (f_full - f_drop)^2

  with ``f`` the feature the head reads (the class token with
  ``model.use_mean_pooling=false``) and ``lambda = consistency_weight``. With
  ``detach_full`` (default) the all-channels pass ``f_full`` is a frozen
  target: run without gradient and in eval mode (no dropout / drop-path), so
  only the dropped pass learns to match it.

``eval_consistency`` also scores val/test with a drop drawn by the same rule
(seeded per batch), reporting ``channel_drop_consistency`` and the dropped-
channel MAE next to the usual metrics. Mixup blends a channel only where both
partners have it; elsewhere the mixed channel is zeroed (missing).
"""
from typing import List, Optional, Sequence

import torch

from labram.data.eeg_constants import get_channel_indices


class ChannelDropper:
    def __init__(self, cfg, ch_names: Sequence[str]):
        self.cfg = cfg
        self.ch_names = list(ch_names)
        upper = [c.upper() for c in ch_names]
        names = [c.upper() for c in cfg.channels]
        missing = sorted(set(names) - set(upper))
        if missing:
            raise ValueError(f"channel_drop.channels {missing} not among {list(ch_names)}")
        # Candidate channels for the augmentation (empty config = all).
        self.candidates: List[int] = ([upper.index(n) for n in names] if names
                                      else list(range(len(upper))))
        lo, hi = int(cfg.min_drop), int(cfg.max_drop)
        if not 1 <= lo <= hi:
            raise ValueError(f"channel_drop needs 1 <= min_drop <= max_drop, got [{lo}, {hi}]")
        if hi > len(self.candidates) or hi >= len(upper):
            raise ValueError(f"channel_drop.max_drop={hi} exceeds the {len(self.candidates)} "
                             f"candidate channels (and must leave one)")
        self.min_drop, self.max_drop = lo, hi
        self.ci_all = get_channel_indices(self.ch_names)

    @classmethod
    def from_config(cls, cfg, ch_names) -> Optional["ChannelDropper"]:
        if cfg is None or not getattr(cfg, "enabled", False) or ch_names is None:
            return None
        return cls(cfg, ch_names)

    # ------------------------------------------------------------- channels
    @staticmethod
    def present(samples: torch.Tensor) -> torch.Tensor:
        """(B, C) True where the channel carries signal (not all zero);
        ``samples`` is (B, C, A, T) or (B, C, L)."""
        return samples.flatten(2).abs().amax(-1) > 0

    def draw(self, generator: Optional[torch.Generator] = None) -> List[int]:
        """Channels to drop for one batch: n ~ U{min..max} of the candidates."""
        n = int(torch.randint(self.min_drop, self.max_drop + 1, (1,), generator=generator))
        pick = torch.randperm(len(self.candidates), generator=generator)[:n]
        return sorted(self.candidates[int(i)] for i in pick)

    def drop_mask(self, present: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
        mask = torch.zeros_like(present)
        if channels:
            mask[:, list(channels)] = True
        return mask

    def zero_missing(self, samples: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Zero the channels where ``keep`` (B, C) is False."""
        if bool(keep.all()):
            return samples
        shape = (*keep.shape,) + (1,) * (samples.dim() - 2)
        return samples * keep.reshape(shape).to(samples.dtype)

    # -------------------------------------------------------------- forward
    def forward(self, model, samples, present, condition=None, return_features=False, **kwargs):
        """Run every sample on its ``present`` (B, C) channels; outputs (and
        head-input features) in batch order."""
        patterns, inverse = torch.unique(present, dim=0, return_inverse=True)
        outs, feats, rows = [], [], []
        for p in range(patterns.shape[0]):
            idx = (inverse == p).nonzero().squeeze(1)
            chans = patterns[p].nonzero().squeeze(1)
            if chans.numel() == 0:
                raise ValueError("a sample has no channel left after dropping")
            x = samples[idx][:, chans]
            ci = [self.ci_all[0]] + [self.ci_all[int(c) + 1] for c in chans]
            extra = dict(kwargs)
            if condition is not None:
                extra["condition"] = condition[idx]
            out, feat = model(x, ci, return_features=True, **extra)
            outs.append(out)
            feats.append(feat)
            rows.append(idx)
        order = torch.argsort(torch.cat(rows))
        out = torch.cat([o.to(outs[0].dtype) for o in outs])[order]
        if not return_features:
            return out
        if any(f is None for f in feats):
            return out, None
        return out, torch.cat([f.to(feats[0].dtype) for f in feats])[order]

    def consistency(self, model, samples, present, dropped, feat_dropped, condition=None,
                    **kwargs):
        """mean (f_full - f_dropped)^2 over the samples that lost a channel."""
        affected = (present & dropped).any(1)
        if not bool(affected.any()) or feat_dropped is None:
            return samples.new_zeros(())
        idx = affected.nonzero().squeeze(1)
        cond = condition[idx] if condition is not None else None
        if self.cfg.detach_full:
            # Frozen all-channels target: no gradient and deterministic (eval
            # mode: no dropout / drop-path), then back to the caller's mode.
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    _, f_full = self.forward(model, samples[idx], present[idx], condition=cond,
                                             return_features=True, **kwargs)
            finally:
                model.train(was_training)
            f_full = f_full.detach()
        else:
            _, f_full = self.forward(model, samples[idx], present[idx], condition=cond,
                                     return_features=True, **kwargs)
        return (f_full.float() - feat_dropped[idx].float()).pow(2).mean()
