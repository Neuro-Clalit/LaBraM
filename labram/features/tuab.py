# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Recording-level handcrafted-feature extraction for the TUAB per-recording .npy export.
# ---------------------------------------------------------
"""Extract handcrafted features from the TUAB ``processed_npy`` export.

The export (``dataset_maker/make_TUAB_npy.py``) stores one ``float32
[n_samples, 23]`` array per recording, already band-passed 0.1-75 Hz, notched at
50 Hz and resampled to 200 Hz by ``make_TUAB.py``. :func:`extract_recording`
loads the first ``n_skip + n_use`` 10 s windows of one recording and returns:

* ``quality``: per-window artifact statistics on the stored signal;
* ``mean_psd``: channel-averaged Welch PSD 0-100 Hz on the stored signal
  (to audit line noise / filter edges);
* ``core/<variant>/<family>``: per-window ``[W, C, F]`` features of a cheap
  family set under several signal-preparation variants (for the
  preparation ablations, which are then done on per-window data);
* ``rec/<family>``: recording-level features (median over clean windows after
  the skip) of every family under the reference preparation, including the
  wavelet-choice sweeps;
* ``fbcov``: mean filter-bank spatial covariance over clean windows
  (``[bands, C, C]``) for Riemannian tangent-space features.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from labram.features import extractors as fx

STORED_CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                   "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2"]
# 19 scalp 10-20 electrodes: ear references (A1/A2) and the anterior-temporal
# T1/T2 (not part of the 10-20 system, absent on many montages) removed.
CHANNELS_1020 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                 "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
WIN = 2000  # 10 s at 200 Hz, the LaBraM / make_TUAB.py window


@dataclass(frozen=True)
class Prep:
    """A signal-preparation variant applied to the stored continuous segment."""

    name: str
    channels: tuple = tuple(STORED_CHANNELS)
    car: bool = False
    band: tuple | None = None  # (lo, hi) Hz zero-phase Butterworth band-pass


PREP_STORED = Prep("stored_23ch")
PREP_CAR = Prep("1020_car", channels=tuple(CHANNELS_1020), car=True)
PREP_REF = Prep("1020_car_1-45Hz", channels=tuple(CHANNELS_1020), car=True, band=(1.0, 45.0))
CORE_PREPS = (PREP_STORED, PREP_CAR, PREP_REF)

DWT_WAVELETS = ("haar", "db2", "db4", "db8", "sym4", "sym9", "coif3", "bior3.5", "dmey")
DWT_LEVELS = (4, 5, 6, 7)
CWT_WAVELETS = ("cmor0.5-1.0", "cmor1.5-1.0", "cmor5.0-1.0")
CWT_FREQS = fx.cwt_frequencies(25, 1.0, 45.0)

# Wavelet scattering is computed at 100 Hz (the reference signal is band-limited
# to 45 Hz, so the 50-100 Hz octave at 200 Hz would be empty); J is in 100 Hz
# samples: J=7 -> 1.28 s averaging / lowest band ~0.4 Hz, J=9 -> 5.12 s.
SCAT_FS = 100
SCAT_CONFIGS = {"scat_J7_Q8": (7, (8, 1)), "scat_J9_Q8": (9, (8, 1))}

# Window rejection on the *stored* signal: Gemein et al. (2020) clip at
# +-800 µV; a channel with < 0.5 µV std over 10 s is disconnected / flat.
REJECT_ABS_UV = 800.0
FLAT_STD_UV = 0.5


def load_segment(path: str, n_windows: int) -> np.ndarray:
    """First ``n_windows`` windows of a recording as ``[C, n_windows * WIN]`` float64."""
    arr = np.load(path, mmap_mode="r")
    n = min(n_windows, arr.shape[0] // WIN) * WIN
    return np.asarray(arr[:n], dtype=np.float64).T


def prepare(seg: np.ndarray, prep: Prep) -> np.ndarray:
    """Apply a preparation variant to a continuous ``[23, N]`` stored segment.

    Filtering is done on the continuous segment *before* windowing so edge
    transients do not repeat in every window.
    """
    idx = [STORED_CHANNELS.index(c) for c in prep.channels]
    x = seg[idx]
    if prep.car:
        x = x - x.mean(axis=0, keepdims=True)
    if prep.band is not None:
        x = fx.bandpass(x, *prep.band)
    return x


def to_windows(x: np.ndarray) -> np.ndarray:
    """``[C, W*WIN]`` -> ``[W, C, WIN]``."""
    c, n = x.shape
    return x.reshape(c, n // WIN, WIN).transpose(1, 0, 2)


def window_quality(xw: np.ndarray) -> np.ndarray:
    """``[W, 4]``: max |x|, max peak-to-peak, min channel std, median channel std (µV).

    Channels that are zero in every window are left out: a channel the montage lacks is stored
    zero-filled (``channel_mask``), and counting it would mark every window as flat, which
    switches the rejection off for the whole recording.
    """
    present = np.abs(xw).max(axis=(0, 2)) > 0
    if present.any() and not present.all():
        xw = xw[:, present]
    std = xw.std(axis=-1)
    return np.stack([np.abs(xw).max(axis=(1, 2)), np.ptp(xw, axis=-1).max(axis=1),
                     std.min(axis=1), np.median(std, axis=1)], axis=-1)


def clean_mask(quality: np.ndarray) -> np.ndarray:
    return (quality[:, 0] <= REJECT_ABS_UV) & (quality[:, 2] >= FLAT_STD_UV)


def core_families(xw: np.ndarray) -> dict:
    """The cheap families computed under every preparation variant."""
    freqs, psd = fx.welch_psd(xw)
    return {"time": fx.time_features(xw)[0],
            "spectral": fx.spectral_features(freqs, psd)[0],
            "dwt_db4": fx.dwt_features(xw, "db4", 6)[0]}


def _median_over(feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median over the selected windows; falls back to all windows if none are clean."""
    sel = feats[mask] if mask.any() else feats
    return np.median(sel, axis=0).astype(np.float32)


def extract_recording(path: str, n_skip: int = 6, n_use: int = 30) -> dict:
    """All features for one recording (see module docstring for the layout)."""
    seg = load_segment(path, n_skip + n_use)
    stored_w = to_windows(seg)
    quality = window_quality(stored_w)
    out = {"quality": quality.astype(np.float32), "n_windows": stored_w.shape[0]}

    f, p = fx.welch_psd(stored_w)
    out["mean_psd"] = np.log10(p.mean(axis=(0, 1)) + fx.EPS).astype(np.float32)
    out["mean_psd_freqs"] = f.astype(np.float32)

    for prep in CORE_PREPS:
        xw = to_windows(prepare(seg, prep))
        for fam, val in core_families(xw).items():
            out[f"core/{prep.name}/{fam}"] = val.astype(np.float32)

    # Reference preparation: skip the first minute, keep clean windows.
    x = prepare(seg, PREP_REF)
    xw = to_windows(x)
    use = np.zeros(xw.shape[0], dtype=bool)
    use[n_skip:] = True
    mask = use & clean_mask(quality)
    if not mask.any():
        mask = use
    out["n_clean"] = int(mask.sum())
    rec = {}
    freqs, psd = fx.welch_psd(xw)
    rec["time"] = fx.time_features(xw)[0]
    rec["spectral"] = fx.spectral_features(freqs, psd)[0]
    rec["psd_bins"] = fx.psd_bin_features(freqs, psd)[0]
    for w in DWT_WAVELETS:
        rec[f"dwt_{w}_L6"] = fx.dwt_features(xw, w, 6)[0]
    for lev in DWT_LEVELS:
        rec[f"dwt_db4_L{lev}"] = fx.dwt_features(xw, "db4", lev)[0]
    rec["wpd_db4_L5"] = fx.wpd_features(xw, "db4", 5)[0]
    rec["mspe"] = fx.multiscale_pe_features(xw)[0]
    for w in CWT_WAVELETS:
        # Scalogram on the continuous segment (no per-window edge effects),
        # then cut into the same 10 s windows.
        rec[f"cwt_{w}"] = fx.cwt_window_features(x, CWT_FREQS, w, WIN)[0]
    for fam, val in rec.items():
        out[f"rec/{fam}"] = _median_over(val, mask)

    covs = []
    for lo, hi in fx.FILTERBANK.values():
        xb = to_windows(fx.bandpass(x, lo, hi))
        covs.append(fx.window_covariances(xb[mask]).mean(axis=0))
    out["fbcov"] = np.stack(covs).astype(np.float32)
    return out


def extract_recording_scattering(path: str, n_skip: int = 6, n_use: int = 30) -> dict:
    """Wavelet-scattering features (``rec/scat_*``) under the reference preparation.

    Same windows and clean-window median as :func:`extract_recording`, but only
    the windows that enter the median are transformed (scattering is the most
    expensive family).
    """
    from scipy.signal import resample_poly
    import torch

    torch.set_num_threads(1)
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    mask = use & clean_mask(quality)
    if not mask.any():
        mask = use
    x = resample_poly(prepare(seg, PREP_REF), SCAT_FS, int(fx.FS), axis=-1)
    win = WIN * SCAT_FS // int(fx.FS)
    xw = x.reshape(x.shape[0], -1, win).transpose(1, 0, 2)[mask]
    out = {}
    for name, (J, Q) in SCAT_CONFIGS.items():
        out[f"rec/{name}"] = np.median(fx.scattering_features(xw, J, Q, SCAT_FS)[0],
                                       axis=0).astype(np.float32)
    return out


def scattering_names() -> dict:
    win = WIN * SCAT_FS // int(fx.FS)
    z = np.zeros((1, win))
    return {name: fx.scattering_features(z, J, Q, SCAT_FS)[1]
            for name, (J, Q) in SCAT_CONFIGS.items()}


def feature_names() -> dict:
    """Per-channel feature names of every family (computed on a dummy input)."""
    xw = np.random.default_rng(0).standard_normal((1, 2, WIN))
    freqs, psd = fx.welch_psd(xw)
    names = {"time": fx.time_features(xw)[1],
             "spectral": fx.spectral_features(freqs, psd)[1],
             "psd_bins": fx.psd_bin_features(freqs, psd)[1],
             "wpd_db4_L5": fx.wpd_features(xw, "db4", 5)[1],
             "mspe": fx.multiscale_pe_features(xw)[1]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dmey level-6 boundary warning
        for w in DWT_WAVELETS:
            names[f"dwt_{w}_L6"] = fx.dwt_features(xw, w, 6)[1]
    for lev in DWT_LEVELS:
        names[f"dwt_db4_L{lev}"] = fx.dwt_features(xw, "db4", lev)[1]
    names["dwt_db4"] = names["dwt_db4_L6"]
    for w in CWT_WAVELETS:
        names[f"cwt_{w}"] = fx.cwt_window_features(xw[0], CWT_FREQS, w, WIN)[1]
    return names


def load_manifest(npy_root: str) -> list:
    with open(os.path.join(npy_root, "manifest.json")) as fh:
        return json.load(fh)["recordings"]


def spectrogram_families(xw: np.ndarray) -> dict:
    """Spectrogram-based families for windows ``[W, C, WIN]`` at 200 Hz."""
    f, s = fx.stft_spectrogram(xw)
    fp, sp = fx.stft_spectrogram(xw, pad=4)  # 0.25 Hz grid for the cepstral filter bank
    fm, sm = fx.multitaper_spectrogram(xw)
    return {"stft": fx.spectrogram_stats(f, s, prefix="stft")[0],
            "stft_dyn": fx.spectrogram_dynamics(f, s)[0],
            "cepstral": fx.cepstral_features(fp, sp)[0],
            "multitaper": fx.spectrogram_stats(fm, sm, prefix="mt")[0]}


def extract_recording_spectrogram(path: str, n_skip: int = 6, n_use: int = 30) -> dict:
    """Spectrogram features (``rec/<family>``) under the reference preparation.

    Same clean windows and median aggregation as :func:`extract_recording`.
    """
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    mask = use & clean_mask(quality)
    if not mask.any():
        mask = use
    xw = to_windows(prepare(seg, PREP_REF))[mask]
    return {f"rec/{k}": np.median(v, axis=0).astype(np.float32)
            for k, v in spectrogram_families(xw).items()}


def spectrogram_names() -> dict:
    xw = np.random.default_rng(0).standard_normal((1, 1, WIN))
    f, s = fx.stft_spectrogram(xw)
    fp, sp = fx.stft_spectrogram(xw, pad=4)
    fm, sm = fx.multitaper_spectrogram(xw)
    return {"stft": fx.spectrogram_stats(f, s, prefix="stft")[1],
            "stft_dyn": fx.spectrogram_dynamics(f, s)[1],
            "cepstral": fx.cepstral_features(fp, sp)[1],
            "multitaper": fx.spectrogram_stats(fm, sm, prefix="mt")[1]}


COHORT_FAMILIES = ("time", "spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "cwt_cmor1.5-1.0",
                   "mspe", "stft", "stft_dyn", "cepstral")


def extract_recording_cohort(path: str, n_skip: int = 6, n_use: int = 30) -> dict:
    """Lean single-pass extractor for large cohorts (e.g. TUEG): reference preparation only.

    Computes exactly the families of the combined age model without scattering
    (:data:`COHORT_FAMILIES`) plus the filter-bank covariances, with the same
    windows, rejection and median aggregation as :func:`extract_recording` and
    :func:`extract_recording_spectrogram`, so the outputs are interchangeable
    with the TUAB caches (same ``rec/<family>`` keys). Skips the wavelet sweeps,
    the extra preparation variants and the per-window core arrays.
    """
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    mask = use & clean_mask(quality)
    if not mask.any():
        mask = use
    x = prepare(seg, PREP_REF)
    xw = to_windows(x)[mask]
    freqs, psd = fx.welch_psd(xw)
    rec = {"time": fx.time_features(xw)[0], "spectral": fx.spectral_features(freqs, psd)[0],
           "psd_bins": fx.psd_bin_features(freqs, psd)[0], "dwt_db4_L6": fx.dwt_features(xw, "db4", 6)[0],
           "wpd_db4_L5": fx.wpd_features(xw, "db4", 5)[0], "mspe": fx.multiscale_pe_features(xw)[0]}
    rec["cwt_cmor1.5-1.0"] = fx.cwt_window_features(x, CWT_FREQS, "cmor1.5-1.0", WIN)[0][mask]
    spec = spectrogram_families(xw)
    rec.update({k: spec[k] for k in ("stft", "stft_dyn", "cepstral")})
    out = {f"rec/{k}": np.median(v, axis=0).astype(np.float32) for k, v in rec.items()}
    covs = [fx.window_covariances(to_windows(fx.bandpass(x, lo, hi))[mask]).mean(axis=0)
            for lo, hi in fx.FILTERBANK.values()]
    out["fbcov"] = np.stack(covs).astype(np.float32)
    out["n_clean"] = int(mask.sum())
    out["quality"] = quality.astype(np.float32)
    return out


# ---------------------------------------------------------------- state-resolved spectra
# Periodic / aperiodic spectra per vigilance state, after Azami et al. 2025 (periodic and
# aperiodic power spectral density of eyes-closed and eyes-open rest). Clinical recordings carry
# no eye-state labels, so the two states are the recording's own high-alpha and low-alpha windows.
STATE_POSTERIOR = ("O1", "O2", "P3", "P4", "PZ", "T5", "T6")
STATE_ALPHA_HZ = (7.0, 13.0)
STATE_FRACTION = 1.0 / 3.0          # each state = this share of the clean windows (top / bottom by alpha)
STATE_MIN_WINDOWS = 3
STATES = ("high", "low")            # high alpha = eyes-closed-like, low alpha = eyes-open-like


def _aperiodic_residual(freqs, psd):
    """``(log10 PSD - aperiodic fit, exponent, offset)`` with the log-log line of
    :func:`extractors.aperiodic_exponent`; frequencies below 1 Hz are left out."""
    sel = freqs >= 1.0
    chi, offset = fx.aperiodic_exponent(freqs, psd)
    fit = offset[..., None] - chi[..., None] * np.log10(freqs[sel])
    return np.log10(psd[..., sel] + fx.EPS) - fit, chi, offset, freqs[sel]


def window_alpha_index(freqs, psd) -> np.ndarray:
    """``[W]``: height of the alpha peak above the aperiodic fit (log10 units), the maximum in
    7-13 Hz averaged over the posterior channels; ``psd`` is ``[W, 19, F]`` in ``CHANNELS_1020`` order."""
    post = [CHANNELS_1020.index(c) for c in STATE_POSTERIOR]
    resid, _, _, f = _aperiodic_residual(freqs, psd[:, post])
    a = (f >= STATE_ALPHA_HZ[0]) & (f <= STATE_ALPHA_HZ[1])
    return resid[..., a].max(axis=-1).mean(axis=-1)


def _papsd(freqs, psd, lo=1.0, hi=45.0):
    """Periodic spectrum in 1 Hz bins ``[..., 44]`` (mean of log10 PSD minus the aperiodic fit
    inside each bin), aperiodic ``[..., 2]`` (exponent, offset) and log power per bin ``[..., 44]``."""
    resid, chi, offset, f = _aperiodic_residual(freqs, psd)
    edges = np.arange(lo, hi)
    periodic = np.stack([resid[..., (f >= e) & (f < e + 1.0)].mean(axis=-1) for e in edges], axis=-1)
    return periodic, np.stack([chi, offset], axis=-1), fx.psd_bin_features(freqs, psd, lo, hi)[0]


def split_states(alpha: np.ndarray, clean: np.ndarray):
    """Boolean masks ``(high, low)`` over the windows: the top and the bottom third of the clean
    windows by alpha index (at least :data:`STATE_MIN_WINDOWS` each); with fewer than twice that
    many clean windows both states are every clean window."""
    idx = np.flatnonzero(clean)
    high, low = np.zeros(len(alpha), bool), np.zeros(len(alpha), bool)
    if len(idx) < 2 * STATE_MIN_WINDOWS:
        high[idx] = low[idx] = True
        return high, low
    n = max(STATE_MIN_WINDOWS, int(round(len(idx) * STATE_FRACTION)))
    order = idx[np.argsort(alpha[idx], kind="stable")]
    low[order[:n]] = True
    high[order[-n:]] = True
    return high, low


def extract_recording_state_spectra(path: str, n_skip: int = 6, n_use: int = 30) -> dict:
    """Periodic / aperiodic spectra of the recording's high-alpha and low-alpha windows.

    Same windows, reference preparation and rejection as :func:`extract_recording_cohort`. Each
    clean window gets an alpha index (:func:`window_alpha_index`); the Welch spectra of the top
    third (``high``: eyes-closed-like) and of the bottom third (``low``: eyes-open-like) are
    combined per state and channel (median of the log spectrum over the state's windows) and split
    into a periodic spectrum (1 Hz bins), the aperiodic exponent / offset and the log power per
    bin. ``state/all/...`` is the same over every clean window (no state split).
    """
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    clean = use & clean_mask(quality)
    if not clean.any():
        clean = use
    freqs, psd = fx.welch_psd(to_windows(prepare(seg, PREP_REF)))          # [W, 19, F]
    alpha = window_alpha_index(freqs, psd)
    high, low = split_states(alpha, clean)
    out = {}
    for name, mask in (("high", high), ("low", low), ("all", clean)):
        # median over the state's windows of the log spectrum: one artifact window (EMG, movement)
        # would dominate a plain average of the power
        state_psd = 10.0 ** np.median(np.log10(psd[mask] + fx.EPS), axis=0)
        periodic, aperiodic, logpsd = _papsd(freqs, state_psd)
        out[f"state/{name}/periodic"] = periodic.astype(np.float32)
        out[f"state/{name}/aperiodic"] = aperiodic.astype(np.float32)
        if name != "all":
            out[f"state/{name}/logpsd"] = logpsd.astype(np.float32)
    out["alpha_index"] = np.array([alpha[high].mean(), alpha[low].mean(), alpha[clean].mean()], np.float32)
    out["n_state"] = int(high.sum())
    return out


def state_spectra_names() -> dict:
    """Column names of :func:`extract_recording_state_spectra`'s families (per channel)."""
    bins = [f"{e:g}Hz" for e in np.arange(1.0, 45.0)]
    out = {}
    for name in ("high", "low", "all"):
        out[f"state/{name}/periodic"] = [f"periodic_{name}_{b}" for b in bins]
        out[f"state/{name}/aperiodic"] = [f"aperiodic_exponent_{name}", f"aperiodic_offset_{name}"]
        if name != "all":
            out[f"state/{name}/logpsd"] = [f"log_psd_{name}_{b}" for b in bins]
    return out


# ---------------------------------------------------------------- method study
# One pass that computes the same cheap features under different analysis choices, so each choice
# can be tested on its own (scripts/feature_stack.py, kind "study"): window length, eye state,
# how the windows of a recording are summarised, and connectivity estimated on short or long data.
STUDY_CORE = ("len10", "len30", "len60", "len30_seg4", "state_high", "state_low",
              "agg_mean", "agg_q10", "agg_q90", "agg_sd", "agg_attn")
STUDY_CONN = ("conn_all", "conn_len10", "conn_len30", "conn_high", "conn_low")
STUDY_COV = ("cov_all", "cov_high", "cov_low")


def _core(xw: np.ndarray, seg_sec: float = 2.0) -> np.ndarray:
    """``[W, C, 68]``: spectral (17), 1 Hz log-PSD bins (44) and time-domain (7) features per window."""
    freqs, psd = fx.welch_psd(xw, seg_sec=seg_sec)
    return np.concatenate([fx.spectral_features(freqs, psd)[0], fx.psd_bin_features(freqs, psd)[0],
                           fx.time_features(xw)[0]], axis=-1)


def _long_windows(x: np.ndarray, clean: np.ndarray, first: int, k: int):
    """Windows of ``k`` consecutive 10 s windows from window ``first`` on: ``([Wl, C, k*WIN], clean)``;
    a long window is clean when all its 10 s windows are."""
    n = (len(clean) - first) // k
    xl = x[:, first * WIN:(first + n * k) * WIN].reshape(x.shape[0], n, k * WIN).transpose(1, 0, 2)
    return xl, clean[first:first + n * k].reshape(n, k).all(axis=1)


def _robust(feats: np.ndarray, mask: np.ndarray, fn=np.median) -> np.ndarray:
    return fn(feats[mask] if mask.any() else feats, axis=0).astype(np.float32)


def extract_recording_study(path: str, attention=None, n_skip: int = 6, n_use: int = 30) -> dict:
    """Features of one recording under each analysis choice of the method study.

    Same data as :func:`extract_recording_cohort` (reference preparation, the 5 minutes after the
    first, the same 10 s rejection). ``attention`` is the model's weight per scored window
    (window ``n_skip + i`` has weight ``attention[i]``), for the attention-weighted summary.

    * window length (median over clean windows of the core features): ``len10``, ``len30``,
      ``len60``; ``len30_seg4`` = 30 s windows with 4 s Welch segments (0.25 Hz resolution);
    * eye state: ``state_high`` / ``state_low`` = the core features of the high-alpha
      (eyes-closed-like) and low-alpha (eyes-open-like) thirds of the 10 s windows;
    * summary over the 10 s windows: ``agg_mean``, ``agg_q10``, ``agg_q90``, ``agg_sd`` (spread),
      ``agg_attn`` (weighted by the model's attention);
    * connectivity (imaginary coherence and coherence per band and channel pair): ``conn_all``
      from all clean data at once, ``conn_len10`` / ``conn_len30`` as the median over 10 s / 30 s
      estimates, ``conn_high`` / ``conn_low`` per eye state;
    * covariances per band from the cross-spectra: ``cov_all``, ``cov_high``, ``cov_low``.
    """
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    clean = use & clean_mask(quality)
    if not clean.any():
        clean = use
    x = prepare(seg, PREP_REF)
    xw = to_windows(x)
    core = _core(xw)                                                    # [W, 19, 68]
    out = {"len10": _robust(core, clean)}
    for k, name in ((3, "len30"), (6, "len60")):
        xl, cl = _long_windows(x, clean, n_skip, k)
        out[name] = _robust(_core(xl), cl) if len(xl) else np.full(core.shape[1:], np.nan, np.float32)
        if k == 3:
            out["len30_seg4"] = (_robust(_core(xl, seg_sec=4.0), cl) if len(xl)
                                 else np.full(core.shape[1:], np.nan, np.float32))
    freqs, psd = fx.welch_psd(xw)
    high, low = split_states(window_alpha_index(freqs, psd), clean)
    out["state_high"], out["state_low"] = _robust(core, high), _robust(core, low)
    cc = core[clean]
    out["agg_mean"], out["agg_sd"] = cc.mean(axis=0).astype(np.float32), cc.std(axis=0).astype(np.float32)
    out["agg_q10"], out["agg_q90"] = (np.quantile(cc, q, axis=0).astype(np.float32) for q in (0.1, 0.9))
    out["agg_attn"] = np.full(core.shape[1:], np.nan, np.float32)
    if attention is not None and len(attention):
        w = np.asarray(attention, float)[: core.shape[0] - n_skip]
        out["agg_attn"] = (np.tensordot(w / w.sum(), core[n_skip:n_skip + len(w)], axes=1)).astype(np.float32)
    f, csd = fx.cross_spectra(xw)                                       # [W, 19, 19, F]
    conn = lambda c: fx.band_connectivity(f, c)[0]                      # noqa: E731
    cov = lambda c: fx.band_covariance(f, c).astype(np.float32)         # noqa: E731
    out["conn_all"] = conn(csd[clean].mean(axis=0)).astype(np.float32)
    out["conn_len10"] = _robust(conn(csd), clean)
    n3 = (csd.shape[0] - n_skip) // 3
    if n3:
        csd3 = csd[n_skip:n_skip + 3 * n3].reshape(n3, 3, *csd.shape[1:]).mean(axis=1)
        out["conn_len30"] = _robust(conn(csd3), clean[n_skip:n_skip + 3 * n3].reshape(n3, 3).all(axis=1))
    else:
        out["conn_len30"] = np.full_like(out["conn_all"], np.nan)
    out["conn_high"], out["conn_low"] = (conn(csd[m].mean(axis=0)).astype(np.float32) for m in (high, low))
    out["cov_all"], out["cov_high"], out["cov_low"] = (cov(csd[m].mean(axis=0)) for m in (clean, high, low))
    out["n_clean"] = int(clean.sum())
    return out


STUDY2_CONN = ("conn_calm", "conn_attn")
STUDY2_COV = ("cov_calm", "cov_rand", "cov_tn", "cov_attn")


def extract_recording_study2(path: str, attention=None, n_skip: int = 6, n_use: int = 30) -> dict:
    """Controls for the eye-state result of :func:`extract_recording_study`: is a covariance from the
    high-alpha third better because of the eye state, or because those windows are free of artifacts?

    Band covariances (and connectivity) from other selections of the same clean 10 s windows:
    ``calm`` = the third with the lowest amplitude (median channel SD), whatever its alpha;
    ``rand`` = a random third (as many windows as a state, no selection); ``tn`` = every clean
    window, each scaled to the same total power before averaging (no window dominates);
    ``attn`` = every scored window weighted by the model's attention.
    """
    import zlib
    seg = load_segment(path, n_skip + n_use)
    quality = window_quality(to_windows(seg))
    use = np.zeros(quality.shape[0], dtype=bool)
    use[n_skip:] = True
    clean = use & clean_mask(quality)
    if not clean.any():
        clean = use
    f, csd = fx.cross_spectra(to_windows(prepare(seg, PREP_REF)))       # [W, 19, 19, F]
    idx = np.flatnonzero(clean)
    n = len(idx) if len(idx) < 2 * STATE_MIN_WINDOWS else max(STATE_MIN_WINDOWS, int(round(len(idx) * STATE_FRACTION)))
    calm = idx[np.argsort(quality[idx, 3], kind="stable")[:n]]
    rand = np.random.default_rng(zlib.crc32(os.path.basename(path).encode())).choice(idx, size=n, replace=False)
    power = np.real(np.einsum("wiif->w", csd))                          # total power of each window
    tn = (csd[idx] / power[idx, None, None, None]).mean(axis=0) * np.median(power[idx])
    cov = lambda c: fx.band_covariance(f, c).astype(np.float32)         # noqa: E731
    conn = lambda c: fx.band_connectivity(f, c)[0].astype(np.float32)   # noqa: E731
    out = {"cov_calm": cov(csd[calm].mean(axis=0)), "cov_rand": cov(csd[rand].mean(axis=0)), "cov_tn": cov(tn),
           "conn_calm": conn(csd[calm].mean(axis=0))}
    out["cov_attn"] = np.full_like(out["cov_calm"], np.nan)
    out["conn_attn"] = np.full_like(out["conn_calm"], np.nan)
    if attention is not None and len(attention):
        w = np.asarray(attention, float)[: csd.shape[0] - n_skip]
        weighted = np.tensordot(w / w.sum(), csd[n_skip:n_skip + len(w)], axes=1)
        out["cov_attn"], out["conn_attn"] = cov(weighted), conn(weighted)
    alpha = window_alpha_index(*fx.welch_psd(to_windows(prepare(seg, PREP_REF))))
    high, _ = split_states(alpha, clean)
    out["calm_high_overlap"] = float(np.isin(calm, np.flatnonzero(high)).mean())   # share of calm windows that are high-alpha
    return out


def study_names() -> dict:
    """Column names of :func:`extract_recording_study`'s families (core: per channel)."""
    xw = np.random.default_rng(0).standard_normal((1, 2, WIN))
    freqs, psd = fx.welch_psd(xw)
    core = (list(fx.spectral_features(freqs, psd)[1]) + list(fx.psd_bin_features(freqs, psd)[1])
            + list(fx.time_features(xw)[1]))
    f, csd = fx.cross_spectra(np.random.default_rng(0).standard_normal((1, len(CHANNELS_1020), WIN)))
    conn = fx.band_connectivity(f, csd)[1]
    iu = np.triu_indices(len(CHANNELS_1020))
    cov = [f"cov_{band}_{i}-{j}" for band in fx.FILTERBANK for i in range(len(CHANNELS_1020))
           for j in range(len(CHANNELS_1020))]
    return {**{k: core for k in STUDY_CORE}, **{k: conn for k in STUDY_CONN + STUDY2_CONN},
            **{k: cov for k in STUDY_COV + STUDY2_COV}}


EXTRACTORS = {"main": extract_recording, "scattering": extract_recording_scattering,
              "spectrogram": extract_recording_spectrogram, "cohort": extract_recording_cohort}


def default_cache_path(npy_root: str, kind: str = "main") -> str:
    """``<edf>/features_handcrafted/tuab_<kind>_v1.npz`` next to ``processed_npy``."""
    fname = "tuab_features_v1.npz" if kind == "main" else f"tuab_{kind}_v1.npz"
    return os.path.join(os.path.dirname(os.path.abspath(npy_root)), "features_handcrafted", fname)


def _worker(args):
    path, n_skip, n_use, kind = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with warnings.catch_warnings():
        # dmey at level 6 on 2000 samples warns about boundary effects (expected).
        warnings.simplefilter("ignore")
        return EXTRACTORS[kind](path, n_skip, n_use)


def extract_corpus(npy_root: str, out_path: str, n_jobs: int = 6, n_skip: int = 6,
                   n_use: int = 30, limit: int | None = None, kind: str = "main",
                   stems=None) -> str:
    """Run an extractor (``kind``: ``main`` = :func:`extract_recording`,
    ``scattering`` = :func:`extract_recording_scattering`) over every recording
    and save one ``.npz``.

    Per-recording outputs are stacked along a leading recording axis (in
    manifest order); recording metadata (stem, subject, split, label, age, sex)
    is stored alongside as ``meta/<field>`` arrays. Windows beyond a recording's
    length (never the case on TUAB: >= 96 windows each) would be NaN-padded.
    """
    recs = load_manifest(npy_root)
    if stems is not None:  # e.g. a curated cohort of a larger export (TUEG)
        keep = set(stems)
        recs = [r for r in recs if r["stem"] in keep]
    recs = recs[:limit]
    paths = [os.path.join(npy_root, "recordings", r["stem"] + ".npy") for r in recs]
    n_win = n_skip + n_use
    stacked: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, out in enumerate(pool.map(_worker, [(p, n_skip, n_use, kind) for p in paths], chunksize=4)):
            for key, val in out.items():
                val = np.asarray(val)
                if key not in stacked:
                    shape = list(val.shape)
                    if key == "quality" or key.startswith("core/"):
                        shape[0] = n_win
                    is_float = val.dtype.kind == "f"
                    stacked[key] = np.full([len(paths)] + shape, np.nan if is_float else 0,
                                           dtype=np.float32 if is_float else val.dtype)
                if key == "quality" or key.startswith("core/"):
                    stacked[key][i, : val.shape[0]] = val
                else:
                    stacked[key][i] = val
            if (i + 1) % 100 == 0:
                rate = (time.time() - t0) / (i + 1)
                print(f"{i + 1}/{len(paths)} recordings, {rate:.2f} s/rec, "
                      f"eta {rate * (len(paths) - i - 1) / 60:.1f} min", flush=True)
    for field in ("stem", "subject", "source_split", "label", "sex"):
        stacked[f"meta/{field}"] = np.array([str(r.get(field)) for r in recs])
    stacked["meta/age"] = np.array([np.nan if r.get("age") is None else float(r["age"]) for r in recs])
    stacked["meta/n_skip"] = np.array(n_skip)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **stacked)
    os.replace(tmp, out_path)
    print(f"saved {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npy_root", required=True, help="TUAB processed_npy directory")
    ap.add_argument("--out", default=None, help="output .npz (default: next to processed_npy)")
    ap.add_argument("--n_jobs", type=int, default=6)
    ap.add_argument("--n_skip", type=int, default=6, help="10 s windows skipped at the start")
    ap.add_argument("--n_use", type=int, default=30, help="10 s windows used after the skip")
    ap.add_argument("--limit", type=int, default=None, help="only the first N recordings (debug)")
    ap.add_argument("--kind", choices=sorted(EXTRACTORS), default="main",
                    help="main feature set, wavelet scattering (needs kymatio), or spectrograms")
    ap.add_argument("--stems_file", default=None,
                    help="only these recordings: a text file with one stem per line, or a parquet "
                         "table with a 'stem' column (e.g. TUEG healthy_subset_v1.parquet)")
    a = ap.parse_args()
    stems = None
    if a.stems_file:
        if a.stems_file.endswith(".parquet"):
            import pandas as pd

            stems = pd.read_parquet(a.stems_file)["stem"].tolist()
        else:
            stems = [ln.strip() for ln in open(a.stems_file) if ln.strip()]
    extract_corpus(a.npy_root, a.out or default_cache_path(a.npy_root, a.kind), a.n_jobs,
                   a.n_skip, a.n_use, a.limit, a.kind, stems)


if __name__ == "__main__":
    main()
