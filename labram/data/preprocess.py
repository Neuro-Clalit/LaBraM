# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Lightweight EEG preprocessing / masking helpers.
# ---------------------------------------------------------

import random
import numpy as np
import torch


def common_average_reference(x: torch.Tensor) -> torch.Tensor:
    """Common Average Reference (CAR): subtract the per-patch across-channel mean.

    ``x``: ``[B, N, A, T]`` (batch, electrodes, patches, patch samples). The mean
    over the electrode axis is removed independently for every (patch, sample)
    position, suppressing activity shared by all channels. Returns a tensor of
    the same shape.
    """
    return x - x.mean(dim=1, keepdim=True)


def z_score_per_patch(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standardise each (channel, patch) window over time.

    ``x``: ``[B, N, A, T]``. Each patch is centred and scaled by its own mean and
    standard deviation across the ``T`` sample axis, so amplitude differences
    across channels do not dominate the representation. ``eps`` guards against
    division by zero on flat patches.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def apply_labram_plus_preprocess(x: torch.Tensor, cfg=None) -> torch.Tensor:
    """Apply the LaBraM++ input preprocessing (CAR then z-scoring) to ``x``.

    ``x``: ``[B, N, A, T]``. ``cfg`` is a
    :class:`labram.configs.labram_plus_config.LaBraMPlusConfig` (or ``None``).
    Each step is gated on the config's effective ``use_car`` / ``use_z_score``
    flags, so a ``None`` or disabled config returns ``x`` unchanged -- preserving
    the original LaBraM behaviour. CAR is applied before z-scoring so the shared
    reference is removed prior to per-patch standardisation.
    """
    if cfg is None:
        return x
    if cfg.use_car:
        x = common_average_reference(x)
    if cfg.use_z_score:
        x = z_score_per_patch(x, cfg.z_score_eps)
    return x


def mask_channels(data, channels=[1, 2, 3]):
    # mask = torch.from_numpy(np.zeros((data.shape[0], len(channels), data.shape[2])))
    data[:, channels, :] = 0
    return data

def normalization(data):
    return data / 100

def collate_mask_time(data, mask_percentage):
    data = torch.from_numpy(np.array(data)) / 100
    data_len = data.shape[-1]
    mask_start_idx = random.randint(0, int(data_len * (1-mask_percentage)))
    mask_end_idx = mask_start_idx + int(data_len*mask_percentage)
    masked_data = data.clone()
    masked_data[:, :, mask_start_idx:mask_end_idx] = 0
    return masked_data, data, [mask_start_idx, mask_end_idx]
