# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training-time EEG augmentations that keep each channel's power spectrum.
# ---------------------------------------------------------
"""Label-preserving EEG augmentations for fine-tuning (``augment.*``).

Brain age is carried by the spectrum (the alpha peak slows, total power falls
with age), so every augmentation here leaves each channel's power spectrum and
amplitude scale intact; frequency/time stretching, global amplitude scaling and
band-stop filters are deliberately absent. Each one is drawn per sample with its
own probability, on the training batch ``x`` of shape ``(B, N, T)`` (channels
``N`` in ``ch_names`` order, time ``T`` at ``sampling_rate``) before mixup.
"""

import math
import re
from typing import List, Optional, Sequence

import torch

from labram.configs.train_config import AugmentConfig

_DIGIT_SUFFIX = re.compile(r'^(.*?)(\d+)$')


def hemisphere_permutation(ch_names: Sequence[str]) -> List[int]:
    """Channel order that swaps each left electrode with its right homologue.

    10-20 naming puts odd numbers on the left and the next even number on the
    right (FP1/FP2, F7/F8, T3/T4, A1/A2, ...); midline (``*Z``) and unpaired
    channels map to themselves. The result is an involution.
    """
    index = {name.upper(): i for i, name in enumerate(ch_names)}
    perm = list(range(len(ch_names)))
    for i, name in enumerate(ch_names):
        m = _DIGIT_SUFFIX.match(name.upper())
        if not m:
            continue
        stem, num = m.group(1), int(m.group(2))
        partner = f"{stem}{num + 1 if num % 2 else num - 1}"
        if partner in index:
            perm[i] = index[partner]
    return perm


def _pick(batch: int, prob: float, device) -> torch.Tensor:
    return torch.rand(batch, device=device) < prob


def sign_flip(x: torch.Tensor, prob: float) -> torch.Tensor:
    sel = _pick(x.shape[0], prob, x.device)
    return torch.where(sel[:, None, None], -x, x)


def hemisphere_flip(x: torch.Tensor, perm: Sequence[int], prob: float) -> torch.Tensor:
    sel = _pick(x.shape[0], prob, x.device)
    flipped = x[:, torch.as_tensor(perm, device=x.device)]
    return torch.where(sel[:, None, None], flipped, x)


def ft_surrogate(x: torch.Tensor, prob: float, phase_noise: float = 1.0) -> torch.Tensor:
    """Add one random phase per frequency, shared by all channels of a sample.

    ``|X_f|`` is untouched (so is each channel's power spectrum) and so are the
    phase differences between channels; only the waveform changes. DC (and
    Nyquist for even ``T``) keep phase 0 so the result stays real.
    """
    sel = _pick(x.shape[0], prob, x.device)
    if not bool(sel.any()):
        return x
    t = x.shape[-1]
    spec = torch.fft.rfft(x.float(), dim=-1)
    phase = torch.rand(x.shape[0], 1, spec.shape[-1], device=x.device) * (2 * math.pi * phase_noise)
    phase[..., 0] = 0
    if t % 2 == 0:
        phase[..., -1] = 0
    surrogate = torch.fft.irfft(spec * torch.polar(torch.ones_like(phase), phase), n=t, dim=-1)
    return torch.where(sel[:, None, None], surrogate.to(x.dtype), x)


def channel_dropout(x: torch.Tensor, prob: float, p_channel: float) -> torch.Tensor:
    """Zero each channel with probability ``p_channel`` in the selected samples,
    always keeping at least one channel."""
    b, n, _ = x.shape
    sel = _pick(b, prob, x.device)
    drop = (torch.rand(b, n, device=x.device) < p_channel) & sel[:, None]
    all_dropped = drop.all(dim=1)
    if bool(all_dropped.any()):
        keep = torch.randint(n, (b,), device=x.device)
        drop[all_dropped, keep[all_dropped]] = False
    return x.masked_fill(drop[:, :, None], 0.0)


def time_mask(x: torch.Tensor, prob: float, max_count: int, min_len: int, max_len: int) -> torch.Tensor:
    """Zero 1..``max_count`` random segments of ``min_len``..``max_len`` samples
    (all channels) in the selected samples."""
    b, _, t = x.shape
    sel = _pick(b, prob, x.device)
    if not bool(sel.any()) or max_len <= 0:
        return x
    max_len = min(max_len, t)
    min_len = max(1, min(min_len, max_len))
    pos = torch.arange(t, device=x.device)
    mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
    counts = torch.randint(1, max(1, max_count) + 1, (b,), device=x.device)
    for k in range(max(1, max_count)):
        length = torch.randint(min_len, max_len + 1, (b,), device=x.device)
        start = (torch.rand(b, device=x.device) * (t - length + 1)).long()
        seg = (pos[None] >= start[:, None]) & (pos[None] < (start + length)[:, None])
        mask |= seg & (sel & (counts > k))[:, None]
    return x.masked_fill(mask[:, None, :], 0.0)


def augment_batch(x: torch.Tensor, cfg: Optional[AugmentConfig],
                  ch_names: Optional[Sequence[str]] = None,
                  sampling_rate: int = 200) -> torch.Tensor:
    """Apply the enabled augmentations to a training batch ``(B, N, T)``.

    Order: hemisphere flip, sign flip, FT surrogate (which reshape the signal),
    then channel dropout and time masking (which blank parts of it), so the
    surrogate never smears a blanked segment.
    """
    if cfg is None or not cfg.enabled:
        return x
    if cfg.hemisphere_flip_prob > 0 and ch_names is not None:
        x = hemisphere_flip(x, hemisphere_permutation(ch_names), cfg.hemisphere_flip_prob)
    if cfg.sign_flip_prob > 0:
        x = sign_flip(x, cfg.sign_flip_prob)
    if cfg.ft_surrogate_prob > 0:
        x = ft_surrogate(x, cfg.ft_surrogate_prob, cfg.ft_phase_noise)
    if cfg.channel_dropout_prob > 0 and cfg.channel_dropout_p > 0:
        x = channel_dropout(x, cfg.channel_dropout_prob, cfg.channel_dropout_p)
    if cfg.time_mask_prob > 0:
        x = time_mask(x, cfg.time_mask_prob, cfg.time_mask_max_count,
                      int(round(cfg.time_mask_min_sec * sampling_rate)),
                      int(round(cfg.time_mask_max_sec * sampling_rate)))
    return x
