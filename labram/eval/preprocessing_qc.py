# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Signal-quality checks of the LaBraM preprocessing protocol on TUH recordings.
# ---------------------------------------------------------
"""Does the LaBraM/BIOT TUAB protocol (0.1-75 Hz band-pass, 50 Hz notch,
resample to 200 Hz, microvolts) suit TUH data? These helpers measure, per
recording, what the protocol leaves in the signal: mains interference (TUH is
US data, 60 Hz), harmonics folded back by acquisition or resampling, clipping,
flat channels and amplitude scale. They drive
``notebooks/tuh_preprocessing_analysis.ipynb``.

Line-noise prominence (dB) of a frequency ``f0`` in a channel-median PSD ``P``:

    LNP(f0) = 10 log10( mean P[f0-0.5, f0+0.5] / median P([f0-5, f0-2] u [f0+2, f0+5]) )

i.e. how far the narrow band around ``f0`` stands above its own neighbourhood;
~0 dB means no line, 10 dB a line ten times its surroundings.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.signal import welch

from labram.data.tuh_metadata import TUH_CHANNELS, channel_map, read_edf_signal_header

L_FREQ, H_FREQ, TARGET_SFREQ = 0.1, 75.0, 200.0
# Frequencies where mains energy can appear: the fundamentals, the 2nd/3rd US
# harmonics, and where the 180 Hz harmonic folds to when sampled at 250/256 Hz
# (acquisition aliasing) or at 200 Hz (resampling aliasing).
PROBE_FREQS_RAW = (50.0, 60.0, 70.0, 76.0, 120.0, 180.0)
PROBE_FREQS_OUT = (20.0, 50.0, 60.0, 70.0, 76.0, 80.0)


@dataclass
class Protocol:
    name: str
    notch: Sequence[float] = (50.0,)
    l_freq: float = L_FREQ
    h_freq: float = H_FREQ
    sfreq: float = TARGET_SFREQ


LABRAM = Protocol("labram (50 Hz notch)", notch=(50.0,))
NOTCH60 = Protocol("60 Hz notch", notch=(60.0,))
NOTCH_BOTH = Protocol("50 + 60 Hz notch", notch=(50.0, 60.0))
LOWPASS45 = Protocol("60 Hz notch, 45 Hz low-pass", notch=(60.0,), h_freq=45.0)
PROTOCOLS = (LABRAM, NOTCH60, NOTCH_BOTH, LOWPASS45)


def load_segment(edf_path: str, start_sec: float = 60.0, dur_sec: float = 120.0):
    """The 23 TUH channels (those present) of ``[start, start + dur)``, raw, in
    their native sampling rate; µV via ``get_data(units='uV')`` downstream."""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    present = channel_map(raw.ch_names)
    raw.pick([present[c] for c in TUH_CHANNELS if c in present])
    tmax = min(start_sec + dur_sec, raw.times[-1])
    tmin = max(0.0, min(start_sec, tmax - dur_sec))
    raw.crop(tmin, tmax, include_tmax=False)
    raw.load_data(verbose="error")
    return raw


def apply_protocol(raw, protocol: Protocol):
    out = raw.copy()
    out.filter(l_freq=protocol.l_freq, h_freq=protocol.h_freq, verbose="error")
    for f0 in protocol.notch:
        if f0 < out.info["sfreq"] / 2:
            out.notch_filter(f0, verbose="error")
    out.resample(protocol.sfreq, verbose="error")
    return out


def channel_median_psd(x_uv: np.ndarray, fs: float):
    """Welch PSD per channel (4 s Hann segments, 50% overlap), median over
    channels: µV²/Hz on a 0.25 Hz grid."""
    f, p = welch(x_uv, fs=fs, nperseg=int(4 * fs), axis=-1)
    return f, np.median(p, axis=0)


def line_prominence_db(f: np.ndarray, p: np.ndarray, f0: float) -> float:
    """LNP(f0) of the module docstring; NaN when f0 or its flanks are not in f."""
    band = (f >= f0 - 0.5) & (f <= f0 + 0.5)
    flank = ((f >= f0 - 5) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 5))
    if not band.any() or flank.sum() < 4 or f0 + 5 > f[-1]:
        return float("nan")
    return float(10 * np.log10(p[band].mean() / max(np.median(p[flank]), 1e-12)))


def band_fraction(f: np.ndarray, p: np.ndarray, lo: float, hi: float,
                  total=(0.5, 75.0)) -> float:
    """Share of the [total] band power that falls in [lo, hi] (dimensionless)."""
    tot = (f >= total[0]) & (f <= total[1])
    sel = (f >= lo) & (f <= hi)
    return float(p[sel].sum() / max(p[tot].sum(), 1e-12))


def amplitude_stats(x_uv: np.ndarray, phys_max: float, fs: float) -> Dict[str, float]:
    """Scale and integrity of a ``[C, T]`` µV segment.

    * ``median_abs_uv`` / ``p99_abs_uv``: median and 99th percentile of |x| over
      channels and samples (µV);
    * ``clip_frac``: share of samples at >= 99.5% of the EDF physical maximum;
    * ``flat_channel_frac``: share of channels whose std over any 10 s window
      is below 0.5 µV (disconnected or saturated);
    * ``dc_offset_uv``: median over channels of |mean| (µV).
    """
    a = np.abs(x_uv)
    n = int(10 * fs)
    stds = np.stack([x_uv[:, i:i + n].std(axis=1) for i in range(0, x_uv.shape[1] - n + 1, n)])
    return {
        "median_abs_uv": float(np.median(a)),
        "p99_abs_uv": float(np.percentile(a, 99)),
        "clip_frac": float((a >= 0.995 * phys_max).mean()) if np.isfinite(phys_max) else float("nan"),
        "flat_channel_frac": float((stds.min(axis=0) < 0.5).mean()),
        "dc_offset_uv": float(np.median(np.abs(x_uv.mean(axis=1)))),
    }


def analyze_recording(edf_path: str, *, start_sec: float = 60.0, dur_sec: float = 120.0,
                      protocols: Sequence[Protocol] = PROTOCOLS) -> Dict[str, object]:
    """Every measurement for one recording: raw line prominences, header filter
    and unit fields, amplitude stats, and per protocol the prominences left at
    the output plus the 55-65 Hz power share."""
    hdr = read_edf_signal_header(edf_path)
    present = channel_map(hdr.labels)
    idx = [hdr.labels.index(present[c]) for c in TUH_CHANNELS if c in present]
    out: Dict[str, object] = {
        "units": ",".join(sorted({hdr.units[i] for i in idx})),
        "prefilter": max((hdr.prefilters[i] for i in idx), key=len, default=""),
        "phys_max": float(np.nanmin([hdr.physical_max[i] for i in idx])),
    }
    raw = load_segment(edf_path, start_sec, dur_sec)
    fs = raw.info["sfreq"]
    x = raw.get_data(units="uV")
    out["native_sfreq"] = fs
    out.update(amplitude_stats(x, out["phys_max"], fs))
    f, p = channel_median_psd(x, fs)
    for f0 in PROBE_FREQS_RAW:
        out[f"raw_lnp_{int(f0)}"] = line_prominence_db(f, p, f0)
    out["raw_psd"] = (f, p)
    for proto in protocols:
        y = apply_protocol(raw, proto)
        fy, py = channel_median_psd(y.get_data(units="uV"), y.info["sfreq"])
        key = proto.name
        for f0 in PROBE_FREQS_OUT:
            out[f"{key}|lnp_{int(f0)}"] = line_prominence_db(fy, py, f0)
        out[f"{key}|frac_55_65"] = band_fraction(fy, py, 55, 65)
        out[f"{key}|psd"] = (fy, py)
    return out


def _analyze_one(job):
    stem, path = job
    try:
        return stem, analyze_recording(path)
    except Exception as exc:              # one bad file must not stop the sample
        return stem, {"error": f"{type(exc).__name__}: {exc}"}


def analyze_sample(rows, edf_path_of, cache: Optional[str] = None, workers: int = 6):
    """``analyze_recording`` over index rows (dicts with ``stem``); returns
    (table of scalars, {stem: spectra}). ``edf_path_of(row)`` -> local EDF
    path. With ``cache`` (a .pkl path) results are reused across runs."""
    import concurrent.futures as cf
    import pickle

    import pandas as pd
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    jobs = [(row["stem"], edf_path_of(row)) for row in rows]
    with cf.ProcessPoolExecutor(workers) as pool:
        results = dict(pool.map(_analyze_one, jobs, chunksize=2))
    spectra = {s: {k: v for k, v in r.items() if k.endswith("psd")} for s, r in results.items()}
    table = pd.DataFrame([{"stem": s, **{k: v for k, v in r.items() if not k.endswith("psd")}}
                          for s, r in results.items()])
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump((table, spectra), fh)
    return table, spectra
