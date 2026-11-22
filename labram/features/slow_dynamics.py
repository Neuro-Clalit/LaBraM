"""Slow amplitude dynamics of a recording: how band amplitude waxes and wanes over seconds to minutes.

Per channel and band (theta 4-8, alpha 8-13, beta 13-30 Hz) of a few contiguous minutes:

* ``dfa``: long-range temporal correlations of the amplitude envelope, the detrended-fluctuation
  exponent over 1-20 s (Linkenkaer-Hansen et al. 2001; Nikulin et al. 2012);
* ``burst_rate`` (per minute) and ``burst_dur`` (s): episodes where the envelope stays above its 75th
  percentile for at least two cycles of the band's centre frequency;
* ``env_cv``: coefficient of variation of the envelope.

These need minutes of contiguous signal, so they are one vector per recording (19 x 3 x 4 = 228),
looked up by the model per recording (``mil.recording_features``, ``labram/layers/window_features.py``).
Absent channels (all zero) give NaN.
"""
import numpy as np

FS = 200.0
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
KINDS = ("dfa", "burst_rate", "burst_dur", "env_cv")
ENV_DECIM = 8                       # envelope at 25 Hz for the DFA
DFA_SEC = (1.0, 20.0)
DFA_SCALES = 8


def feature_names(channels):
    return [f"{ch}:{b}:{k}" for ch in channels for b in BANDS for k in KINDS]


def _envelope(x, lo, hi, fs=FS):
    """Amplitude envelope ``[C, T]`` of the ``lo``-``hi`` Hz band (analytic signal by FFT)."""
    n = x.shape[-1]
    spec = np.fft.fft(x, axis=-1)
    f = np.fft.fftfreq(n, 1.0 / fs)
    h = ((f >= lo) & (f <= hi)).astype(float) * 2.0          # positive band only, doubled
    return np.abs(np.fft.ifft(spec * h, axis=-1))


def dfa_exponent(env, fs_env, smin=DFA_SEC[0], smax=DFA_SEC[1], n_scales=DFA_SCALES):
    """DFA exponent ``[C]`` of signals ``[C, T]``: slope of log F(s) on log s, windows with 50 % overlap."""
    c, t = env.shape
    y = np.cumsum(env - env.mean(axis=1, keepdims=True), axis=1)
    scales = np.unique(np.round(np.logspace(np.log10(smin * fs_env), np.log10(smax * fs_env), n_scales)).astype(int))
    scales = scales[scales * 4 <= t]                             # at least ~7 overlapping windows
    if len(scales) < 3:
        return np.full(c, np.nan)
    logf = []
    for s in scales:
        starts = np.arange(0, t - s + 1, max(1, s // 2))
        seg = np.stack([y[:, a:a + s] for a in starts], axis=1)  # [C, W, s]
        u = np.arange(s) - (s - 1) / 2.0
        slope = (seg * u).sum(-1, keepdims=True) / (u ** 2).sum()
        resid = seg - seg.mean(-1, keepdims=True) - slope * u
        logf.append(0.5 * np.log10((resid ** 2).mean(axis=(1, 2)) + 1e-30))
    logs = np.log10(scales)
    logf = np.stack(logf, 1)                                      # [C, S]
    ls = logs - logs.mean()
    return ((logf - logf.mean(1, keepdims=True)) * ls).sum(1) / (ls ** 2).sum()


def burst_stats(env, min_len, fs=FS):
    """``(rate per minute [C], mean duration in s [C])`` of supra-75th-percentile episodes >= ``min_len`` samples."""
    c, t = env.shape
    above = env > np.percentile(env, 75, axis=1, keepdims=True)
    rate, dur = np.zeros(c), np.full(c, np.nan)
    for i in range(c):
        a = np.concatenate([[False], above[i], [False]]).astype(np.int8)
        d = np.diff(a)
        lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        lengths = lengths[lengths >= min_len]
        rate[i] = len(lengths) / (t / fs / 60.0)
        if len(lengths):
            dur[i] = lengths.mean() / fs
    return rate, dur


def slow_dynamics(x, present=None, fs=FS):
    """Features ``[19 * 3 * 4]`` of a referenced segment ``x [19, T]`` (µV); NaN for absent channels."""
    c = x.shape[0]
    present = np.ones(c, bool) if present is None else present
    out = np.full((c, len(BANDS), len(KINDS)), np.nan)
    for j, (lo, hi) in enumerate(BANDS.values()):
        env = _envelope(x, lo, hi, fs)
        n = env.shape[1] // ENV_DECIM * ENV_DECIM
        env_ds = env[:, :n].reshape(c, -1, ENV_DECIM).mean(-1)
        out[:, j, 0] = dfa_exponent(env_ds, fs / ENV_DECIM)
        rate, dur = burst_stats(env, int(round(2 * fs / ((lo + hi) / 2))), fs)
        out[:, j, 1], out[:, j, 2] = rate, dur
        out[:, j, 3] = env.std(1) / (env.mean(1) + 1e-12)
    out[~present] = np.nan
    return out.reshape(-1).astype(np.float32)


def referenced_segment(arr, stored_channels, channels, start_window, n_windows, win=2000):
    """``(x [19, T] µV with a common average reference over the present channels, present [19])`` of
    windows ``start_window`` .. ``start_window + n_windows`` of a stored ``[T, C]`` recording."""
    idx = [stored_channels.index(ch) for ch in channels]
    a = min(start_window, arr.shape[0] // win) * win
    b = min((start_window + n_windows) * win, arr.shape[0] // win * win)
    x = np.asarray(arr[a:b], dtype=np.float64).T[idx]
    present = np.abs(x).max(axis=1) > 0 if x.shape[1] else np.zeros(len(idx), bool)
    if present.any():
        x = (x - x[present].mean(axis=0, keepdims=True)) * present[:, None]
    return x, present
