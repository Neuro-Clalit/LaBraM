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
    """``[W, 4]``: max |x|, max peak-to-peak, min channel std, median channel std (µV)."""
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


EXTRACTORS = {"main": extract_recording, "scattering": extract_recording_scattering}


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
                   n_use: int = 30, limit: int | None = None, kind: str = "main") -> str:
    """Run an extractor (``kind``: ``main`` = :func:`extract_recording`,
    ``scattering`` = :func:`extract_recording_scattering`) over every recording
    and save one ``.npz``.

    Per-recording outputs are stacked along a leading recording axis (in
    manifest order); recording metadata (stem, subject, split, label, age, sex)
    is stored alongside as ``meta/<field>`` arrays. Windows beyond a recording's
    length (never the case on TUAB: >= 96 windows each) would be NaN-padded.
    """
    recs = load_manifest(npy_root)[:limit]
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
        stacked[f"meta/{field}"] = np.array([str(r[field]) for r in recs])
    stacked["meta/age"] = np.array([np.nan if r["age"] is None else float(r["age"]) for r in recs])
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
                    help="main feature set, or wavelet scattering (needs kymatio)")
    a = ap.parse_args()
    extract_corpus(a.npy_root, a.out or default_cache_path(a.npy_root, a.kind), a.n_jobs,
                   a.n_skip, a.n_use, a.limit, a.kind)


if __name__ == "__main__":
    main()
