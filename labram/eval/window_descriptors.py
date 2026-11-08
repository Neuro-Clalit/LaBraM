# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-window signal descriptors for confidence-weighted pooling.
# ---------------------------------------------------------
"""Window reading and per-window descriptors (``docs/window_confidence_results.md``).

:class:`TTAWindows` serves every window of a window-selected npy dataset, once,
optionally with shifted crops for test-time augmentation. The descriptor
functions take a batch of raw windows ``(B, C, L)`` in uV at 200 Hz (all-zero
channels = missing), common-average reference the channels that carry signal,
and return ``{name: (B,) tensor}``:

* :func:`quality_descriptors` -- EMG ratio, amplitude, flat channels;
* :func:`stft_descriptors` -- classical spectrogram (2 s Hann frames, 0.5 s
  hop): relative band powers, their variability over frames, spectral
  entropy, aperiodic slope, alpha peak, high-frequency share, channel spread
  and the largest frame-power excursion;
* :func:`cwt_descriptors` -- complex-Morlet wavelet spectrogram (log-spaced
  1-45 Hz, 6 cycles): relative band powers, burstiness (coefficient of
  variation over time) and peak-to-median ratio per band, wavelet entropy.
"""

import bisect
import math
import os

import numpy as np
import torch

from labram.data.tuh_datasets import WINDOW_SAMPLES
from labram.data.window_selection import _leaves

FS = 200.0
EMG_BAND, TOTAL_BAND = (30.0, 45.0), (1.0, 45.0)
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


class TTAWindows(torch.utils.data.Dataset):
    """Every window of a (window-selected npy) dataset as ``n_crops + 1``
    copies: the window itself and ``n_crops`` crops whose start moves by at most
    ``pad`` samples, kept inside the recording's trimmed range. One read per
    window (the window plus ``pad`` on each side). Crop offsets are seeded per
    window, so every run and every re-run sees the same copies."""

    def __init__(self, dataset, n_crops: int, pad: int, seed: int = 0):
        self.leaves = _leaves(dataset)
        self.starts = np.cumsum([0] + [len(leaf.files) for leaf in self.leaves])
        self.n_crops, self.pad, self.seed = n_crops, pad, seed

    def __len__(self):
        return int(self.starts[-1])

    def _locate(self, i):
        li = bisect.bisect_right(self.starts, i) - 1
        leaf = self.leaves[li]
        return leaf, leaf.files[i - self.starts[li]]

    def stem(self, i):
        leaf, filename = self._locate(i)
        return os.path.basename(filename)[:-4].rsplit(leaf._recording_sep, 1)[0]

    def __getitem__(self, i):
        leaf, filename = self._locate(i)
        stem, k = os.path.basename(filename)[:-4].rsplit(leaf._recording_sep, 1)
        length = WINDOW_SAMPLES * leaf.windows_per_item
        start = int(k) * WINDOW_SAMPLES
        lo, hi = leaf.sample_bounds.get(stem) or (0, leaf._n_samples(stem))
        a, b = max(lo, start - self.pad), min(hi, start + length + self.pad)
        a, b = min(a, start), max(b, start + length)
        seg = leaf._read_rows(stem, a, b - a)
        if leaf.keep_channels is not None:
            seg = seg[:, leaf.keep_channels]
        x = np.ascontiguousarray(seg.T, dtype=np.float32)
        rng = np.random.default_rng((self.seed, i))
        offsets = [start] + [int(o) for o in rng.integers(a, b - length + 1, self.n_crops)]
        copies = np.stack([x[:, o - a:o - a + length] for o in offsets])
        y = leaf._label_fn(None, filename)
        shift = (np.asarray(offsets[1:]) - start) / FS
        return torch.from_numpy(copies), torch.as_tensor(y).float(), stem, int(k), torch.from_numpy(shift)


def quality_descriptors(x: torch.Tensor) -> dict:
    """Quality descriptors of a batch of raw windows ``(B, C, L)`` in uV, over
    the channels that carry signal, after common-average reference."""
    present = (x.abs().amax(-1) > 0).float()                     # (B, C)
    n = present.sum(1).clamp_min(1)
    car = (x - (x * present[..., None]).sum(1, keepdim=True) / n[:, None, None]) * present[..., None]
    spec = torch.fft.rfft(car, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / FS).to(x.device)
    band = lambda lo, hi: spec[..., (freqs >= lo) & (freqs < hi)].sum(-1)   # noqa: E731
    rel = band(*EMG_BAND) / band(*TOTAL_BAND).clamp_min(1e-12)   # (B, C)
    std = car.std(-1)
    big = torch.finfo(std.dtype).max
    return {
        "emg": ((rel * present).sum(1) / n),
        "emg_max": (rel * present).amax(1),
        "max_abs": (car.abs().amax(-1) * present).amax(1),
        "amp_uv": (std * present).sum(1) / n,
        "min_std": torch.where(present > 0, std, torch.full_like(std, big)).amin(1),
    }


def _car(x: torch.Tensor):
    """(CAR signal with missing channels zeroed, present mask (B, C), n present (B,))."""
    present = (x.abs().amax(-1) > 0).float()
    n = present.sum(1).clamp_min(1)
    car = (x - (x * present[..., None]).sum(1, keepdim=True) / n[:, None, None]) * present[..., None]
    return car, present, n


def _chan_mean(v: torch.Tensor, present: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Mean over present channels of ``v`` (B, C, ...)."""
    shape = present.shape + (1,) * (v.dim() - 2)
    return (v * present.reshape(shape)).sum(1) / n.reshape((-1,) + (1,) * (v.dim() - 2))


def _band_mask(freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (freqs >= lo) & (freqs < hi)


def stft_descriptors(x: torch.Tensor, n_fft: int = 400, hop: int = 100) -> dict:
    """Classical spectrogram descriptors of raw windows ``(B, C, L)``."""
    car, present, n = _car(x.float())
    b, c, l = car.shape
    spec = torch.stft(car.reshape(b * c, l), n_fft, hop, window=torch.hann_window(n_fft, device=x.device),
                      center=False, return_complex=True).abs().pow(2).reshape(b, c, n_fft // 2 + 1, -1)
    freqs = torch.fft.rfftfreq(n_fft, 1 / FS).to(x.device)
    eps = 1e-12
    total_m = _band_mask(freqs, *TOTAL_BAND)
    p_ft = _chan_mean(spec, present, n)                               # (B, F, T), channel mean
    total_t = p_ft[:, total_m].sum(1).clamp_min(eps)                  # (B, T)
    out = {"stft_logpow": total_t.mean(1).log()}
    for name, (lo, hi) in BANDS.items():
        bt = p_ft[:, _band_mask(freqs, lo, hi)].sum(1)                # (B, T)
        out[f"stft_rel_{name}"] = (bt.mean(1) / total_t.mean(1)).clamp_min(eps).log()
        out[f"stft_cv_{name}"] = bt.std(1) / bt.mean(1).clamp_min(eps)
    hf = p_ft[:, _band_mask(freqs, 45.0, 70.0)].sum(1).mean(1)
    out["stft_rel_hf"] = (hf / total_t.mean(1)).clamp_min(eps).log()
    spec_mean = p_ft[:, total_m].mean(-1)                             # (B, F) time-mean spectrum
    prob = spec_mean / spec_mean.sum(1, keepdim=True).clamp_min(eps)
    out["stft_entropy"] = -(prob * prob.clamp_min(eps).log()).sum(1) / math.log(prob.shape[1])
    fit = _band_mask(freqs, 2.0, 40.0) & ~_band_mask(freqs, 7.0, 14.0)
    lf, lp = freqs[fit].log(), p_ft[:, fit].mean(-1).clamp_min(eps).log()   # log-log, alpha excluded
    lf_c = lf - lf.mean()
    out["stft_slope"] = ((lp - lp.mean(1, keepdim=True)) * lf_c).sum(1) / (lf_c ** 2).sum()
    am = _band_mask(freqs, 7.0, 14.0)
    alpha = p_ft[:, am].mean(-1)
    out["stft_paf"] = freqs[am][alpha.argmax(1)]
    out["stft_alpha_prom"] = (alpha.amax(1) / alpha.median(1).values.clamp_min(eps)).log()
    ch_pow = spec[:, :, total_m].sum(2).mean(-1).clamp_min(eps).log()   # (B, C)
    mu = (ch_pow * present).sum(1) / n
    out["stft_ch_sd"] = (((ch_pow - mu[:, None]) ** 2 * present).sum(1) / n).sqrt()
    out["stft_peak_ratio"] = (total_t.amax(1) / total_t.median(1).values).log()
    return out


def morlet_bank(n: int, freqs, cycles: float = 6.0, device=None) -> torch.Tensor:
    """Frequency responses ``(F, n)`` of unit-energy complex Morlet wavelets,
    analytic (zero at negative frequencies), for an ``n``-sample FFT."""
    f = torch.fft.fftfreq(n, 1 / FS, device=device)
    rows = []
    for fc in freqs:
        sigma_f = fc / cycles
        rows.append(torch.exp(-0.5 * ((f - fc) / sigma_f) ** 2) * (f > 0))
    bank = torch.stack(rows)
    return bank / bank.pow(2).sum(1, keepdim=True).sqrt()


CWT_FREQS = tuple(float(v) for v in np.geomspace(1.0, 45.0, 16))


def cwt_descriptors(x: torch.Tensor, freqs=CWT_FREQS, cycles: float = 6.0, step: int = 10) -> dict:
    """Wavelet-spectrogram descriptors of raw windows ``(B, C, L)``; power is
    taken every ``step`` samples, 1 s at each edge is dropped (edge effects)."""
    car, present, n = _car(x.float())
    b, c, l = car.shape
    bank = morlet_bank(l, freqs, cycles, x.device)                    # (F, L)
    coef = torch.fft.ifft(torch.fft.fft(car)[:, :, None, :] * bank, dim=-1)
    power = coef[..., int(FS):l - int(FS):step].abs().pow(2)          # (B, C, F, T)
    p_ft = _chan_mean(power, present, n)                              # (B, F, T)
    fr = torch.as_tensor(freqs, device=x.device)
    eps = 1e-12
    total_t = p_ft.sum(1).clamp_min(eps)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = _band_mask(fr, lo, hi)
        bt = p_ft[:, m].sum(1)
        out[f"cwt_rel_{name}"] = (bt.mean(1) / total_t.mean(1)).clamp_min(eps).log()
        out[f"cwt_cv_{name}"] = bt.std(1) / bt.mean(1).clamp_min(eps)
        out[f"cwt_peak_{name}"] = (bt.amax(1) / bt.median(1).values.clamp_min(eps)).log()
    prob = p_ft.mean(-1)
    prob = prob / prob.sum(1, keepdim=True).clamp_min(eps)
    out["cwt_entropy"] = -(prob * prob.clamp_min(eps).log()).sum(1) / math.log(len(freqs))
    return out
