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


def band_limit(x: torch.Tensor, lo: float, hi: float, fs: float = 200.0, order: int = 8) -> torch.Tensor:
    """Zero-phase band-limit of every channel of every window.

    ``x``: ``[B, N, A, T]``; the ``A`` patches of ``T`` samples are one
    continuous window per channel. The spectrum is multiplied by the squared
    magnitude of an order-``order`` Butterworth high-pass ``lo`` and low-pass
    ``hi`` (a forward-backward filter; ``lo = 0`` skips the high-pass), after
    reflect padding by up to 1 s on both ends against wrap-around. Computed in
    float32 and returned in ``x``'s dtype.
    """
    b, n, a, t = x.shape
    length = a * t
    pad = min(int(fs), length - 1)
    z = x.reshape(b * n, 1, length).float()
    z = torch.nn.functional.pad(z, (pad, pad), mode="reflect")[:, 0]
    f = torch.fft.rfftfreq(z.shape[-1], d=1.0 / fs, device=z.device)
    gain = 1.0 / (1.0 + (f / hi) ** (2 * order))
    if lo > 0:
        r = (f / lo) ** (2 * order)
        gain = gain * r / (1.0 + r)
    z = torch.fft.irfft(torch.fft.rfft(z, dim=-1) * gain, n=z.shape[-1], dim=-1)
    return z[:, pad:pad + length].reshape(b, n, a, t).to(x.dtype)


def amplitude_normalize(x: torch.Tensor, target: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """Rescale every window so the median over channels of the channel std is ``target``.

    ``x``: ``[B, N, A, T]``. The std of each channel is taken over the whole
    window (``A * T`` samples); all-zero (missing) channels are left out of the
    median. A global gain is removed; topography and spectral shape are kept.
    """
    b, n, a, t = x.shape
    z = x.float()
    std = z.reshape(b, n, a * t).std(dim=-1)                       # [B, N]
    std_masked = torch.where(std > eps, std, torch.full_like(std, float("nan")))
    scale = torch.nanmedian(std_masked, dim=1).values               # [B]
    scale = torch.nan_to_num(scale, nan=target).clamp_min(eps)
    return (z * (target / scale)[:, None, None, None]).to(x.dtype)


def apply_labram_plus_preprocess(x: torch.Tensor, cfg=None) -> torch.Tensor:
    """Apply the LaBraM++ input preprocessing (CAR, band-limit, amplitude, z-scoring) to ``x``.

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
    if getattr(cfg, "use_band", False):
        x = band_limit(x, float(cfg.band_hz[0]), float(cfg.band_hz[1]))
    if getattr(cfg, "use_amplitude_norm", False):
        x = amplitude_normalize(x, float(cfg.amplitude_target))
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
