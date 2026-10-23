# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Handcrafted EEG feature extractors (time, spectral, wavelet, entropy, covariance).
# ---------------------------------------------------------
"""Per-window EEG feature extractors.

Every extractor takes ``x`` shaped ``[..., C, T]`` (any leading axes, usually
``[W, C, T]`` windows x channels x samples, in µV) and returns
``(features [..., C, F], names [F])`` so families can be concatenated, aggregated
over windows, and labelled uniformly. Power-like features are returned in
``log10`` so they are roughly Gaussian across recordings (the standard choice for
linear models on EEG power).
"""

from __future__ import annotations

import math

import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, stats

FS = 200.0
EPS = 1e-12

# Canonical clinical bands (Hz). Gamma is capped at 45 Hz so it stays below
# line noise (50/60 Hz) and the 75 Hz anti-alias edge of the stored TUAB data.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
# Filter bank used for covariance (Riemannian) features: beta split in two as in
# Engemann et al. 2022 / Sabbagh et al. 2020.
FILTERBANK = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ----------------------------------------------------------------- time domain
def time_features(x: np.ndarray):
    """Classic time-domain descriptors (Hjorth, line length, moments, ZCR)."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = x.var(axis=-1) + EPS
    var1 = dx.var(axis=-1) + EPS
    var2 = ddx.var(axis=-1) + EPS
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    centred = x - x.mean(axis=-1, keepdims=True)
    zcr = (np.diff(np.signbit(centred), axis=-1)).mean(axis=-1)
    feats = [
        np.log10(var0),
        np.log10(np.abs(dx).mean(axis=-1) + EPS),  # line length
        stats.skew(x, axis=-1),
        stats.kurtosis(x, axis=-1),
        mobility,
        complexity,
        zcr,
    ]
    names = ["log_var", "log_line_length", "skew", "kurtosis",
             "hjorth_mobility", "hjorth_complexity", "zero_cross_rate"]
    return np.stack(feats, axis=-1), names


# ------------------------------------------------------------- spectral (FFT)
def welch_psd(x: np.ndarray, fs: float = FS, seg_sec: float = 2.0):
    """Welch PSD with 2 s Hann segments, 50% overlap (0.5 Hz resolution)."""
    nper = int(seg_sec * fs)
    return signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)


def _band_power(freqs, psd, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    return psd[..., sel].sum(axis=-1) * (freqs[1] - freqs[0])


def aperiodic_exponent(freqs, psd, fit=(2.0, 40.0), exclude=(7.0, 14.0)):
    """1/f exponent from a log-log linear fit, alpha peak excluded.

    A lightweight stand-in for FOOOF/specparam's fixed-mode aperiodic fit
    (Donoghue et al. 2020): the PSD is ``b - chi * log10(f)`` outside the alpha
    range, so the negated slope is the exponent ``chi``. Returns
    ``(exponent, offset)``, each ``[...]``.
    """
    sel = (freqs >= fit[0]) & (freqs <= fit[1]) & ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    lf = np.log10(freqs[sel])
    lp = np.log10(psd[..., sel] + EPS)
    lf_c = lf - lf.mean()
    slope = (lp - lp.mean(axis=-1, keepdims=True)) @ lf_c / (lf_c @ lf_c)
    offset = lp.mean(axis=-1) - slope * lf.mean()
    return -slope, offset


def spectral_features(freqs, psd):
    """Band powers (abs + relative), ratios, edge frequency, entropy, PAF, 1/f."""
    total_lo, total_hi = 1.0, 45.0
    abs_bp = np.stack([_band_power(freqs, psd, lo, hi) for lo, hi in BANDS.values()], axis=-1)
    total = _band_power(freqs, psd, total_lo, total_hi)[..., None] + EPS
    rel_bp = abs_bp / total
    d, t, a, b = (abs_bp[..., i] + EPS for i in range(4))

    sel = (freqs >= total_lo) & (freqs < total_hi)
    p = psd[..., sel] / (psd[..., sel].sum(axis=-1, keepdims=True) + EPS)
    spec_entropy = -(p * np.log(p + EPS)).sum(axis=-1) / np.log(sel.sum())
    cum = np.cumsum(p, axis=-1)
    sef95 = freqs[sel][np.argmax(cum >= 0.95, axis=-1)]
    # Peak (posterior dominant rhythm) frequency in 6-14 Hz. A real PDR exists
    # mostly on posterior channels; elsewhere this is a weak descriptor.
    asel = (freqs >= 6.0) & (freqs <= 14.0)
    paf = freqs[asel][np.argmax(psd[..., asel], axis=-1)]
    chi, offset = aperiodic_exponent(freqs, psd)

    feats = [np.log10(abs_bp + EPS), rel_bp,
             np.stack([np.log10(t / a), np.log10((d + t) / (a + b)), sef95,
                       spec_entropy, paf, chi, offset], axis=-1)]
    names = ([f"log_abs_{k}" for k in BANDS] + [f"rel_{k}" for k in BANDS]
             + ["log_theta_alpha", "log_slowing_ratio", "sef95", "spectral_entropy",
                "peak_alpha_freq", "aperiodic_exponent", "aperiodic_offset"])
    return np.concatenate(feats, axis=-1), names


def psd_bin_features(freqs, psd, lo=1.0, hi=45.0, width=1.0):
    """Log power in contiguous 1 Hz bins: the full spectrum as a feature vector."""
    edges = np.arange(lo, hi + 1e-9, width)
    feats = np.stack([np.log10(_band_power(freqs, psd, e, e + width) + EPS) for e in edges[:-1]],
                     axis=-1)
    return feats, [f"log_psd_{e:g}Hz" for e in edges[:-1]]


# ------------------------------------------------------------ spectrograms
def stft_spectrogram(x: np.ndarray, fs: float = FS, seg_sec: float = 1.0, lo: float = 1.0,
                     hi: float = 45.0, pad: int = 1):
    """Short-time Fourier power ``[..., F, frames]`` (Hann, 50 % overlap), ``lo <= f <= hi``.

    A 1 s segment gives 1 Hz resolution and 19 frames per 10 s window: a
    fixed time-frequency tiling, unlike the constant-Q wavelet tilings.
    ``pad`` zero-pads each segment (``nfft = pad * nperseg``) to interpolate the
    frequency grid without changing the resolution.
    """
    nper = int(seg_sec * fs)
    f, _, s = signal.spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                                 nfft=pad * nper, scaling="density", mode="psd", axis=-1)
    sel = (f >= lo) & (f <= hi)
    return f[sel], s[..., sel, :]


def multitaper_spectrogram(x: np.ndarray, fs: float = FS, seg_sec: float = 2.0, nw: float = 2.0,
                           n_tapers: int = 3, lo: float = 1.0, hi: float = 45.0):
    """Multitaper (DPSS) spectrogram ``[..., F, frames]`` (Thomson 1982; Prerau et al. 2017).

    Averaging ``K = 2NW - 1`` orthogonal Slepian tapers reduces the variance of
    each time-frequency estimate ~K-fold at a fixed spectral resolution of
    ``2 NW / seg_sec`` Hz (here 2 Hz on 2 s segments with 1 s step).
    """
    nper = int(seg_sec * fs)
    tapers = signal.windows.dpss(nper, nw, Kmax=n_tapers)
    acc = None
    for taper in tapers:
        f, _, s = signal.spectrogram(x, fs=fs, window=taper, nperseg=nper, noverlap=nper // 2,
                                     scaling="density", mode="psd", axis=-1)
        acc = s if acc is None else acc + s
    sel = (f >= lo) & (f <= hi)
    return f[sel], acc[..., sel, :] / len(tapers)


def spectrogram_stats(freqs, spec, lo: float = 1.0, hi: float = 45.0, width: float = 1.0,
                      prefix: str = "stft"):
    """Mean and std over time of log power in contiguous ``width``-Hz bins.

    The mean is a (Welch-like) average spectrum; the std over frames measures
    how much each frequency's power fluctuates inside the window.
    """
    edges = np.arange(lo, hi + 1e-9, width)[:-1]
    logs = []
    for e in edges:
        sel = (freqs >= e) & (freqs < e + width)
        if not sel.any():  # coarse grids (multitaper 0.5 Hz bins fit; guard anyway)
            sel = np.argmin(np.abs(freqs - (e + width / 2)))
            logs.append(np.log10(spec[..., sel, :] + EPS))
        else:
            logs.append(np.log10(spec[..., sel, :].mean(axis=-2) + EPS))
    logs = np.stack(logs, axis=-2)  # [..., bins, frames]
    feats = np.concatenate([logs.mean(axis=-1), logs.std(axis=-1)], axis=-1)
    names = ([f"{prefix}_mean_{e:g}Hz" for e in edges] + [f"{prefix}_std_{e:g}Hz" for e in edges])
    return feats, names


def spectrogram_dynamics(freqs, spec):
    """Temporal dynamics of the spectrogram: per-band variability, persistence and flux.

    For each clinical band ``b`` with log band power ``l_b(tau)`` over frames:
    ``std_tau l_b`` (variability), lag-1 autocorrelation of ``l_b`` (persistence:
    sustained vs intermittent rhythm), plus the spectral flux
    ``mean_tau ||log S(., tau+1) - log S(., tau)||_2 / sqrt(F)``.
    """
    df = freqs[1] - freqs[0]
    lb = np.stack([np.log10(spec[..., (freqs >= lo) & (freqs < hi), :].sum(axis=-2) * df + EPS)
                   for lo, hi in BANDS.values()], axis=-2)  # [..., 5, frames]
    c = lb - lb.mean(axis=-1, keepdims=True)
    ac1 = (c[..., 1:] * c[..., :-1]).mean(axis=-1) / (c.var(axis=-1) + EPS)
    logs = np.log10(spec + EPS)
    flux = np.sqrt((np.diff(logs, axis=-1) ** 2).mean(axis=-2)).mean(axis=-1, keepdims=True)
    feats = np.concatenate([lb.std(axis=-1), ac1, flux], axis=-1)
    names = ([f"std_logpow_{b}" for b in BANDS] + [f"ac1_logpow_{b}" for b in BANDS]
             + ["spectral_flux"])
    return feats, names


def cepstral_features(freqs, spec, n_filters: int = 16, n_coef: int = 13, lo: float = 1.0,
                      hi: float = 45.0, f_break: float = 4.0):
    """MFCC-style cepstral coefficients of the spectrogram (mean and std over time).

    Triangular filters equally spaced on the warped axis
    ``u(f) = log(1 + f / f_break)`` -- the EEG analogue of the mel scale: linear
    below ~``f_break`` Hz, logarithmic above -- pool the power spectrum into
    ``M = n_filters`` energies ``E_m(tau)``; the DCT-II
    ``c_k(tau) = sqrt(2/M) sum_m log10 E_m(tau) cos(pi k (m + 1/2) / M)`` (orthonormal)
    decorrelates them. ``c_0`` is the overall log level; low ``k`` encode the
    spectral envelope (tilt, broad slowing), higher ``k`` finer shape (peaks).
    A gain change only moves ``c_0``. Use a finely sampled spectrum (e.g.
    ``stft_spectrogram(..., pad=4)``) so every filter covers several bins.
    """
    u = np.log1p(np.asarray(freqs) / f_break)
    centres = np.linspace(np.log1p(lo / f_break), np.log1p(hi / f_break), n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for m in range(n_filters):
        a, b, c = centres[m], centres[m + 1], centres[m + 2]
        fb[m] = np.clip(np.minimum((u - a) / (b - a), (c - u) / (c - b)), 0.0, None)
    if (fb > 0).sum(axis=1).min() < 2:
        raise ValueError("frequency grid too coarse for the cepstral filter bank")
    fb /= fb.sum(axis=1, keepdims=True)
    energies = np.log10(np.einsum("mf,...ft->...mt", fb, spec) + EPS)  # [..., M, frames]
    from scipy.fft import dct

    cep = dct(energies, type=2, axis=-2, norm="ortho")[..., :n_coef, :]
    feats = np.concatenate([cep.mean(axis=-1), cep.std(axis=-1)], axis=-1)
    names = [f"cep_mean_c{k}" for k in range(n_coef)] + [f"cep_std_c{k}" for k in range(n_coef)]
    return feats, names


# --------------------------------------------------------------- wavelets: DWT
def dwt_band_edges(level: int, fs: float = FS):
    """Nominal (ideal-filter) frequency band of each DWT output, A_L first."""
    bands = [(0.0, fs / 2 ** (level + 1))]
    bands += [(fs / 2 ** (j + 1), fs / 2 ** j) for j in range(level, 0, -1)]
    return bands


def dwt_features(x: np.ndarray, wavelet: str = "db4", level: int = 6):
    """Multilevel DWT sub-band log energy, relative energy and wavelet entropy.

    ``pywt.wavedec`` returns ``[A_L, D_L, ..., D_1]``; at 200 Hz and 6 levels the
    nominal bands are A6 0-1.6, D6 1.6-3.1, D5 3.1-6.3, D4 6.3-12.5,
    D3 12.5-25, D2 25-50, D1 50-100 Hz -- the dyadic tiling roughly matches
    delta/theta/alpha/beta/gamma, which is why 5-6 levels is the EEG default.
    Energies are per-coefficient means so levels of different length compare.
    Wavelet entropy is Rosso et al. (2001)'s Shannon entropy of the relative
    energy distribution.
    """
    coeffs = pywt.wavedec(x, wavelet, level=level, axis=-1, mode="symmetric")
    energy = np.stack([(c ** 2).mean(axis=-1) for c in coeffs], axis=-1)
    total = np.stack([(c ** 2).sum(axis=-1) for c in coeffs], axis=-1)
    rel = total / (total.sum(axis=-1, keepdims=True) + EPS)
    went = -(rel * np.log(rel + EPS)).sum(axis=-1, keepdims=True)
    labels = [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]
    feats = np.concatenate([np.log10(energy + EPS), rel, went], axis=-1)
    names = [f"log_E_{lab}" for lab in labels] + [f"rel_E_{lab}" for lab in labels] + ["wavelet_entropy"]
    return feats, names


def wpd_features(x: np.ndarray, wavelet: str = "db4", level: int = 5, max_freq: float = 50.0,
                 fs: float = FS):
    """Wavelet-packet log energy in a uniform frequency tiling.

    Unlike the DWT (which only re-splits the low-pass branch), the wavelet
    packet splits both branches, giving ``2**level`` equal-width sub-bands
    (3.125 Hz at level 5, 200 Hz) -- finer resolution in beta/gamma. Nodes are
    taken in frequency order (``order="freq"``); those above ``max_freq`` are
    dropped.
    """
    wp = pywt.WaveletPacket(x, wavelet, mode="symmetric", maxlevel=level, axis=-1)
    nodes = wp.get_level(level, order="freq")
    width = fs / 2 / 2 ** level
    keep = [i for i in range(len(nodes)) if (i + 1) * width <= max_freq + 1e-9]
    feats = np.stack([np.log10((nodes[i].data ** 2).mean(axis=-1) + EPS) for i in keep], axis=-1)
    names = [f"log_E_wp_{i * width:g}-{(i + 1) * width:g}Hz" for i in keep]
    return feats, names


# --------------------------------------------------------------- wavelets: CWT
def cwt_frequencies(n: int = 25, lo: float = 1.0, hi: float = 45.0):
    """Log-spaced CWT analysis frequencies (constant-Q tiling)."""
    return np.geomspace(lo, hi, n)


def morlet_params(wavelet: str):
    """Parse pywt's ``cmorB-C`` name into ``(B, C)``."""
    b, c = wavelet[len("cmor"):].split("-")
    return float(b), float(c)


def cwt_log_power(x: np.ndarray, freqs, wavelet: str = "cmor1.5-1.0", fs: float = FS):
    """Complex-Morlet scalogram ``log10 |W(f, t)|^2`` -> ``[F, ..., C, T]``.

    Uses pywt's ``cmorB-C`` parameterisation, ``psi(t) = exp(-t^2/B) exp(i 2 pi C t)
    / sqrt(pi B)``: the bandwidth ``B`` sets the time/frequency trade-off. In
    MNE's terms the wavelet has ``n_cycles = pi * C * sqrt(2B)`` (~5.4 for
    ``cmor1.5-1.0``) and a spectral width ``sigma_f = f / n_cycles``: larger
    ``B`` = sharper in frequency, blurrier in time.

    The transform is evaluated in the frequency domain (one FFT of the signal,
    then a Gaussian analytic filter ``exp(-pi^2 B C^2 (f/f0 - 1)^2)`` per
    analysis frequency ``f0``) -- mathematically the same convolution as
    ``pywt.cwt`` but much faster on long segments. Each filter has unit peak
    gain on the analytic signal, so a sinusoid of amplitude ``A`` at ``f0`` has
    ``|W| = A`` (scalogram in µV^2, comparable across frequencies).
    """
    bw, cf = morlet_params(wavelet)
    n = x.shape[-1]
    spec = np.fft.fft(x, axis=-1)
    f = np.fft.fftfreq(n, d=1.0 / fs)
    out = np.empty((len(freqs),) + x.shape, dtype=np.float32)
    for i, f0 in enumerate(freqs):
        h = 2.0 * np.exp(-(np.pi ** 2) * bw * cf ** 2 * (f / f0 - 1.0) ** 2) * (f > 0)
        coef = np.fft.ifft(spec * h, axis=-1)
        out[i] = np.log10(coef.real ** 2 + coef.imag ** 2 + EPS)
    return out


def _decimation(f0: float, n_cycles: float, win: int, fs: float, support: float = 6.0,
                min_rate: float = 5.0):
    """Largest divisor ``D`` of ``win`` whose rate ``fs/D`` still holds the filter band.

    ``min_rate`` keeps >= 50 envelope samples per 10 s window so the per-window
    mean/std of the log envelope are not dominated by sampling noise.
    """
    need = max(2.0 * support * f0 / n_cycles, min_rate)  # two-sided band of +-support*sigma_f
    divisors = [d for d in range(1, win + 1) if win % d == 0 and fs / d >= need]
    return max(divisors) if divisors else 1


def cwt_window_features(x: np.ndarray, freqs, wavelet: str = "cmor1.5-1.0", win: int = 2000,
                        fs: float = FS):
    """Per-window scalogram features from a continuous ``[C, N]`` segment.

    Returns ``[N // win, C, 2F]``: per window, the mean over time of
    ``log10 |W(f, t)|^2`` (a smoothed log spectrum, close to Welch) and its
    standard deviation over time -- what the CWT adds: how bursty /
    non-stationary each rhythm is inside the window (e.g. intermittent slowing).

    Same filter as :func:`cwt_log_power`, but each band is evaluated by complex
    demodulation: the Gaussian band around ``f0`` is shifted to baseband and
    inverse-transformed at a reduced rate ``fs/D`` that still contains it, which
    yields ``|W|`` exactly on every ``D``-th sample at a fraction of the cost.
    The transform runs on the whole segment, so windows have no edge effects
    except at the segment ends.
    """
    bw, cf = morlet_params(wavelet)
    n_cycles = np.pi * cf * np.sqrt(2.0 * bw)
    c, n = x.shape
    n_win = n // win
    spec = np.fft.fft(x[:, : n_win * win], axis=-1)
    n = n_win * win
    mean = np.empty((n_win, c, len(freqs)), dtype=np.float32)
    std = np.empty_like(mean)
    for i, f0 in enumerate(freqs):
        d = _decimation(f0, n_cycles, win, fs)
        m = n // d
        k0 = int(round(f0 * n / fs))
        offs = np.arange(-(m // 2), m - m // 2)
        fk = (k0 + offs) * fs / n
        h = 2.0 * np.exp(-(np.pi ** 2) * bw * cf ** 2 * (fk / f0 - 1.0) ** 2) * (fk > 0)
        base = np.zeros((c, m), dtype=complex)
        base[:, offs % m] = spec[:, (k0 + offs) % n] * h
        env = np.fft.ifft(base, axis=-1) * (m / n)
        logp = np.log10(env.real ** 2 + env.imag ** 2 + EPS).reshape(c, n_win, win // d)
        mean[:, :, i] = logp.mean(axis=-1).T
        std[:, :, i] = logp.std(axis=-1).T
    names = [f"cwt_mean_{f:.1f}Hz" for f in freqs] + [f"cwt_std_{f:.1f}Hz" for f in freqs]
    return np.concatenate([mean, std], axis=-1), names


# ------------------------------------------------------ wavelet scattering
_SCATTERING = {}


def _scattering_op(J: int, Q: tuple, n: int):
    """Cached kymatio ``Scattering1D`` (torch backend) plus coefficient metadata."""
    key = (J, tuple(Q), n)
    if key not in _SCATTERING:
        from kymatio.torch import Scattering1D

        op = Scattering1D(J=J, shape=n, Q=tuple(Q), max_order=2)
        _SCATTERING[key] = (op, op.meta())
    return _SCATTERING[key]


def scattering_features(x: np.ndarray, J: int = 7, Q: tuple = (8, 1), fs: float = 100.0,
                        eps: float = 1e-6):
    """Time-averaged wavelet scattering coefficients (Mallat 2012; Andén & Mallat 2014).

    The scattering transform is a fixed (not learned) two-layer convolutional
    network: Morlet wavelet filter bank -> modulus -> filter bank -> modulus,
    each layer averaged by a low-pass of width ``2**J`` samples. First-order
    coefficients ``S1(l1) = |x * psi_l1| * phi`` are a constant-Q band energy
    (close to the CWT mean); second-order coefficients
    ``S2(l1, l2) = ||x * psi_l1| * psi_l2| * phi`` measure how the envelope of
    rhythm ``l1`` fluctuates at the slower modulation frequency ``l2`` --
    information a power spectrum discards. ``Q = (Q1, Q2)`` is the number of
    wavelets per octave in each layer.

    ``x``: ``[..., T]`` at ``fs`` Hz. Coefficients are averaged over the window
    and returned as ``log10 S1`` and the renormalised ``log10(S2 / S1(l1))``
    (Andén & Mallat 2014, which removes the trivial dependence of S2 on the
    band's power). Uses kymatio's reference implementation (torch backend, CPU).
    """
    import torch

    n = x.shape[-1]
    op, meta = _scattering_op(J, Q, n)
    flat = torch.from_numpy(np.ascontiguousarray(x.reshape(-1, n), dtype=np.float32))
    with torch.no_grad():
        s = op(flat).mean(dim=-1).double().numpy()  # [M, K]
    order, xi, idx = meta["order"], meta["xi"], meta["n"]
    o1, o2 = np.where(order == 1)[0], np.where(order == 2)[0]
    first_of = {int(idx[k, 0]): k for k in o1}
    parent = np.array([first_of[int(idx[k, 0])] for k in o2])
    log_s1 = np.log10(s[:, o1] + eps)
    log_s2 = np.log10(s[:, o2] + eps) - np.log10(s[:, parent] + eps)
    feats = np.concatenate([log_s1, log_s2], axis=-1).reshape(x.shape[:-1] + (-1,))
    names = ([f"S1_{xi[k, 0] * fs:.2f}Hz" for k in o1]
             + [f"S2_{xi[k, 0] * fs:.2f}Hz_mod{xi[k, 1] * fs:.3f}Hz" for k in o2])
    return feats, names


# ------------------------------------------------------- multiscale complexity
def coarse_grain(x: np.ndarray, scale: int):
    """Costa et al. (2002) coarse-graining: non-overlapping means of ``scale``."""
    n = x.shape[-1] // scale
    return x[..., : n * scale].reshape(x.shape[:-1] + (n, scale)).mean(axis=-1)


def permutation_entropy(x: np.ndarray, order: int = 4, delay: int = 1):
    """Normalised permutation entropy (Bandt & Pompe 2002), vectorised."""
    n = x.shape[-1] - (order - 1) * delay
    emb = sliding_window_view(x, (order - 1) * delay + 1, axis=-1)[..., ::delay]
    ranks = np.argsort(emb, axis=-1, kind="stable")
    codes = (ranks * (order ** np.arange(order))).sum(axis=-1)  # unique per permutation
    flat = codes.reshape(-1, n)
    k = order ** order
    offsets = (np.arange(flat.shape[0]) * k)[:, None]
    counts = np.bincount((flat + offsets).ravel(), minlength=flat.shape[0] * k).reshape(-1, k)
    p = counts / n
    h = -(p * np.log(p + EPS)).sum(axis=-1) / math.log(math.factorial(order))
    return h.reshape(x.shape[:-1])


def multiscale_pe_features(x: np.ndarray, scales=(1, 2, 3, 4, 6, 8, 10, 15, 20), order: int = 4):
    """Multiscale permutation entropy: PE of the coarse-grained signal per scale.

    At 200 Hz, scale ``s`` low-passes to ~``100/s`` Hz, so the curve moves from
    fast (fine) to slow (coarse) dynamics -- the complexity counterpart of a
    spectrum. Ageing raises fine-scale and lowers coarse-scale entropy
    (McIntosh et al. 2014).
    """
    feats = np.stack([permutation_entropy(coarse_grain(x, s), order=order) for s in scales], axis=-1)
    return feats, [f"mspe_s{s}" for s in scales]


# -------------------------------------------------------- covariance (spatial)
def bandpass(x: np.ndarray, lo: float, hi: float, fs: float = FS, order: int = 4):
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def window_covariances(x: np.ndarray):
    """Spatial covariance per window: ``[..., C, T]`` -> ``[..., C, C]``."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return np.einsum("...ct,...dt->...cd", xc, xc) / x.shape[-1]
