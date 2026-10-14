# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training-time dropout of named channels (e.g. T1/T2) for a whole batch, a
# consistency loss between the full and the dropped representation, and
# per-sample handling of recordings that lack those channels.
# ---------------------------------------------------------
"""``channel_drop``: one model for recordings with and without some channels.

A cohort can mix recordings that have every channel with recordings that lack
some (zero-filled in the npy, e.g. TUEG's linked-ear recordings without
T1/T2). LaBraM has one token per channel and patch, so a recording without
T1/T2 is run with those tokens left out rather than as zeros:

* every forward pass splits the batch per sample: rows whose dropped channels
  are all zero run on the kept channels only, the others on all channels;
  outputs are reassembled in batch order (the encoder has no batch-coupled
  layers, so this equals a per-sample forward);
* training, with probability ``prob`` per batch: every row runs without the
  channels -- the loss is computed on these outputs -- and the rows that do
  have them are also run on all channels, adding

      L_cons = mean_{rows with the channels} mean_d (f_full - f_drop)^2

  times ``consistency_weight``, where ``f`` is the feature the head reads
  (the class token with ``model.use_mean_pooling=false``, else the mean of
  the patch tokens), so the representation does not depend on whether the
  channels are present. ``detach_full`` stops the gradient through ``f_full``.

Mixup partners are blended only where both have the channels; elsewhere the
mixed channels are zeroed, so a mixed sample is never half present.
"""
from typing import List, Optional, Sequence

import torch

from labram.data.eeg_constants import get_channel_indices


class ChannelDropper:
    def __init__(self, cfg, ch_names: Sequence[str]):
        self.cfg = cfg
        names = {c.upper() for c in cfg.channels}
        upper = [c.upper() for c in ch_names]
        missing = sorted(names - set(upper))
        if missing:
            raise ValueError(f"channel_drop.channels {missing} not among {list(ch_names)}")
        self.drop: List[int] = [i for i, c in enumerate(upper) if c in names]
        self.keep: List[int] = [i for i, c in enumerate(upper) if c not in names]
        full = get_channel_indices(list(ch_names))
        self.ci_full = full
        self.ci_kept = [full[0]] + [full[i + 1] for i in self.keep]

    @classmethod
    def from_config(cls, cfg, ch_names) -> Optional["ChannelDropper"]:
        if cfg is None or not cfg.channels or ch_names is None:
            return None
        return cls(cfg, ch_names)

    # ------------------------------------------------------------------ rows
    def has_channels(self, samples: torch.Tensor) -> torch.Tensor:
        """(B,) True where any dropped channel carries signal; ``samples`` is
        (B, N, A, T) or (B, N, L)."""
        return samples[:, self.drop].flatten(1).abs().amax(1) > 0

    def zero_rows(self, samples: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Zero the dropped channels of ``rows`` (bool (B,))."""
        if rows.any():
            idx = rows.nonzero().squeeze(1)
            samples = samples.clone()
            samples[idx[:, None], torch.as_tensor(self.drop, device=samples.device)] = 0
        return samples

    # --------------------------------------------------------------- forward
    def forward(self, model, samples, full_rows, condition=None, return_features=False,
                **kwargs):
        """Rows in ``full_rows`` on all channels, the rest on the kept ones.
        Returns the output (and the head-input feature) in batch order."""
        parts = []
        for rows, channels, ci in ((full_rows, None, self.ci_full),
                                   (~full_rows, self.keep, self.ci_kept)):
            if not rows.any():
                continue
            idx = rows.nonzero().squeeze(1)
            x = samples[idx] if channels is None else samples[idx][:, channels]
            extra = dict(kwargs)
            if condition is not None:
                extra['condition'] = condition[idx]
            res = model(x, ci, return_features=True, **extra)
            parts.append((idx, res))
        out = _scatter([(i, r[0]) for i, r in parts], samples.shape[0])
        if not return_features:
            return out
        feats = [(i, r[1]) for i, r in parts]
        return out, (_scatter(feats, samples.shape[0]) if all(f is not None for _, f in feats)
                     else None)

    def consistency(self, model, samples, has, feat_dropped, condition=None, **kwargs):
        """L_cons over the rows that have the channels (0 if none do)."""
        if not has.any() or feat_dropped is None:
            return samples.new_zeros(())
        idx = has.nonzero().squeeze(1)
        extra = dict(kwargs)
        if condition is not None:
            extra['condition'] = condition[idx]
        _, f_full = model(samples[idx], self.ci_full, return_features=True, **extra)
        if self.cfg.detach_full:
            f_full = f_full.detach()
        return (f_full.float() - feat_dropped[idx].float()).pow(2).mean()


def _scatter(parts, n):
    """Reassemble ``[(row_idx, tensor)]`` into one (n, ...) tensor in row order."""
    first = parts[0][1]
    if len(parts) == 1 and parts[0][0].numel() == n:
        order = parts[0][0]
        return first if torch.equal(order, torch.arange(n, device=order.device)) else \
            first[torch.argsort(order)]
    idx = torch.cat([i for i, _ in parts])
    val = torch.cat([v.to(first.dtype) for _, v in parts])
    return val[torch.argsort(idx)]
