# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-recording meta features for the TUH EEG corpora: everything we can learn
# about a recording *besides* the EEG windows the model trains on -- extended
# EDF-header fields, the acquisition setup, signal quality, the auxiliary
# channels (EKG, photic stimulator, device burst-suppression trends) and a few
# clinically motivated qEEG summaries. Feeds notebooks/tuab_meta_features.ipynb.
# See docs/meta_features.md.
# ---------------------------------------------------------

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis

from labram.data.tuh_metadata import EDF_HEADER_BYTES, parse_edf_header_metadata

logger = logging.getLogger(__name__)

# The 19 scalp electrodes of the 10-20 system. A1/A2 (ear) and T1/T2 are
# recorded too but are reference/auxiliary sites, so the quality and qEEG
# summaries are computed over these only.
SCALP_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)
HOMOLOGOUS_PAIRS = (
    ("FP1", "FP2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
    ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6"),
)
POSTERIOR_CHANNELS = ("O1", "O2", "P3", "P4")
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}

EPOCH_SEC = 10.0
FLAT_STD_UV = 0.5           # a 10-s epoch flatter than this is a dead electrode
HIGH_AMP_P2P_UV = 500.0     # peak-to-peak beyond this (0.5-45 Hz) is artifact, not brain
CLIP_FRACTION_OF_RANGE = 0.99


@dataclass(frozen=True)
class EDFSignalHeader:
    """The per-signal part of an EDF header plus the global record layout."""

    labels: List[str]
    physical_min: List[float]
    physical_max: List[float]
    samples_per_record: List[int]
    n_records: int
    record_duration: float
    recording_field: str

    def sfreq(self, index: int) -> float:
        return self.samples_per_record[index] / self.record_duration


def read_edf_signal_header(path: str) -> EDFSignalHeader:
    """Parse the full EDF header (global block + one 256-byte block per signal)."""
    with open(path, "rb") as fh:
        head = fh.read(EDF_HEADER_BYTES)
        n_signals = int(head[252:256])
        block = fh.read(n_signals * 256)
    if len(block) < n_signals * 256:
        raise ValueError(f"{path}: truncated EDF signal header")

    def field(offset: int, width: int) -> List[str]:
        return [block[offset + i * width: offset + (i + 1) * width].decode("latin-1").strip()
                for i in range(n_signals)]

    # Per-signal fields are stored field-major: all labels, then all transducers, ...
    widths = [("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
              ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8)]
    fields, offset = {}, 0
    for name, width in widths:
        fields[name] = field(offset, width)
        offset += n_signals * width

    return EDFSignalHeader(
        labels=fields["label"],
        physical_min=[float(v) for v in fields["pmin"]],
        physical_max=[float(v) for v in fields["pmax"]],
        samples_per_record=[int(v) for v in fields["nsamp"]],
        n_records=int(head[236:244]),
        record_duration=float(head[244:252]),
        recording_field=head[88:168].decode("latin-1").strip(),
    )


def canonical_channel(label: str) -> str:
    """'EEG FP1-REF' -> 'FP1'; non-EEG labels ('PHOTIC-REF', 'IBI') pass through."""
    name = label.upper().strip()
    if name.startswith("EEG "):
        name = name[4:]
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def header_features(path: str) -> Dict[str, object]:
    """Demographics + acquisition setup, from the header alone (no signal read)."""
    meta = parse_edf_header_metadata(path)
    hdr = read_edf_signal_header(path)
    names = [canonical_channel(label) for label in hdr.labels]
    eeg_index = names.index("FP1") if "FP1" in names else 0
    # The recording field is 'Startdate <date> <subject>_<session> <technician> <equipment>';
    # TUH anonymises the technician to 'XXX' but some files keep an equipment tag
    # ('Compact_32ch', '#####'), which identifies the amplifier family.
    tail = hdr.recording_field.split()[3:]
    equipment = " ".join(tail[1:]) if len(tail) > 1 else ""

    def session_number(tag: Optional[str]) -> Optional[int]:
        return int(tag[1:]) if tag and tag[1:].isdigit() else None

    return {
        "stem": meta.stem,
        "subject": meta.subject,
        "session": meta.session,
        "session_number": session_number(meta.session),
        "token_number": session_number(meta.token),
        "age": meta.age,
        "raw_age": meta.raw_age,
        "age_redacted_90plus": meta.is_redacted,
        "sex": meta.sex,
        "year": meta.year,
        "duration_min": hdr.n_records * hdr.record_duration / 60.0,
        "sfreq": hdr.sfreq(eeg_index),
        "physical_max_uv": hdr.physical_max[eeg_index],
        "equipment_tag": equipment,
        "n_signals": len(names),
        "has_ekg": any("EKG" in n or "ECG" in n for n in names),
        "has_emg": "EMG" in names,
        "has_eog": "LOC" in names and "ROC" in names,
        "has_photic_channel": "PHOTIC" in names,
        # Sphenoidal electrodes are placed for temporal-lobe epilepsy work-ups.
        "has_sphenoidal": "SP1" in names or "SP2" in names,
        "has_numbered_extra": any(n.isdigit() for n in names),
        "n_scalp_channels": sum(n in names for n in SCALP_CHANNELS),
    }


# --------------------------------------------------------------------------- #
# Signal-derived features. Each takes plain arrays so it is testable without EDFs.
# --------------------------------------------------------------------------- #

def _epochs(data: np.ndarray, sfreq: float, epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """[channels, samples] -> [epochs, channels, epoch_samples], dropping the tail."""
    size = int(round(epoch_sec * sfreq))
    n_epochs = data.shape[1] // size
    if n_epochs == 0:
        raise ValueError("recording shorter than one epoch")
    return data[:, : n_epochs * size].reshape(data.shape[0], n_epochs, size).transpose(1, 0, 2)


def quality_features(raw_uv: np.ndarray, sfreq: float, physical_max_uv: float) -> Dict[str, float]:
    """Artifact / dead-electrode / mains-noise summary over scalp channels.

    ``raw_uv`` is the unfiltered [channels, samples] signal in microvolts.
    """
    clipped = np.abs(raw_uv) >= CLIP_FRACTION_OF_RANGE * physical_max_uv
    sos = sps.butter(4, [0.5, 45.0], btype="bandpass", fs=sfreq, output="sos")
    filtered = sps.sosfiltfilt(sos, raw_uv, axis=-1)
    epochs = _epochs(filtered, sfreq)
    epoch_std = epochs.std(axis=-1)                               # [epochs, channels]
    epoch_p2p = np.ptp(epochs, axis=-1)
    flat = epoch_std < FLAT_STD_UV
    high_amp = epoch_p2p > HIGH_AMP_P2P_UV
    flat_channel = flat.mean(axis=0) > 0.5                        # dead for most of the record
    # A channel whose overall amplitude is an outlier among its neighbours is
    # usually a bad contact even when it is not flat.
    log_std = np.log(filtered.std(axis=-1) + 1e-6)
    med = np.median(log_std)
    mad = np.median(np.abs(log_std - med)) + 1e-6
    noisy_channel = np.abs(log_std - med) / (1.4826 * mad) > 5.0

    freqs, psd = sps.welch(raw_uv, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)
    psd = psd.mean(axis=0)
    broadband = psd[(freqs >= 1) & (freqs <= 45)].sum()

    def line_ratio(f0: float) -> float:
        band = psd[(freqs >= f0 - 1) & (freqs <= f0 + 1)].sum()
        return float(np.log10(band / broadband + 1e-12)) if f0 + 1 < sfreq / 2 else float("nan")

    return {
        "q_clip_frac": float(clipped.mean()),
        "q_flat_epoch_frac": float(flat.any(axis=1).mean()),
        "q_high_amp_epoch_frac": float(high_amp.any(axis=1).mean()),
        "q_bad_epoch_frac": float((flat.any(axis=1) | high_amp.any(axis=1)).mean()),
        "q_n_flat_channels": int(flat_channel.sum()),
        "q_n_noisy_channels": int((noisy_channel & ~flat_channel).sum()),
        "q_median_amp_uv": float(np.median(epoch_std)),
        "q_line60_log_ratio": line_ratio(60.0),
        "q_line50_log_ratio": line_ratio(50.0),
    }


def qeeg_features(raw_uv: np.ndarray, sfreq: float, channels: Sequence[str]) -> Dict[str, float]:
    """Clinically motivated spectral summaries of the scalp EEG.

    Abnormal clinical EEGs are mostly read on: background slowing (more
    delta/theta, less alpha), a slow or absent posterior dominant rhythm (PDR,
    normally 8-13 Hz in adults), focal/hemispheric asymmetry, and sharp
    transients. Spectra are the *median* over 10-s epochs, so short artifacts
    do not dominate. Signals are re-referenced to the common average first.
    """
    channels = list(channels)
    data = raw_uv - raw_uv.mean(axis=0, keepdims=True)
    sos = sps.butter(4, [0.5, 45.0], btype="bandpass", fs=sfreq, output="sos")
    data = sps.sosfiltfilt(sos, data, axis=-1)
    epochs = _epochs(data, sfreq)
    freqs, psd = sps.welch(epochs, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)
    psd = np.median(psd, axis=0)                                   # [channels, freqs]

    def band_power(lo: float, hi: float) -> np.ndarray:
        return psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=-1)

    total = band_power(1.0, 30.0) + 1e-12
    rel = {name: band_power(lo, hi) / total for name, (lo, hi) in BANDS.items()}
    out = {f"qeeg_rel_{name}": float(values.mean()) for name, values in rel.items()}
    out["qeeg_log_total_power"] = float(np.log10(total.mean()))
    out["qeeg_theta_alpha_ratio"] = float(np.log10(rel["theta"].mean() / (rel["alpha"].mean() + 1e-12)))
    out["qeeg_slowing_ratio"] = float(np.log10(
        (rel["delta"] + rel["theta"]).mean() / ((rel["alpha"] + rel["beta"]).mean() + 1e-12)))

    posterior = [channels.index(c) for c in POSTERIOR_CHANNELS if c in channels]
    if posterior:
        post = psd[posterior].mean(axis=0)
        win = (freqs >= 5.0) & (freqs <= 14.0)
        # Peak height over a 1/f fit in log-log space = how *distinct* the PDR is.
        fit = (freqs >= 2.0) & (freqs <= 30.0)
        coef = np.polyfit(np.log10(freqs[fit]), np.log10(post[fit] + 1e-12), 1)
        residual = np.log10(post + 1e-12) - np.polyval(coef, np.log10(np.maximum(freqs, 1e-3)))
        peak = np.argmax(np.where(win, residual, -np.inf))
        out["qeeg_pdr_freq"] = float(freqs[peak])
        out["qeeg_pdr_prominence"] = float(residual[peak])
        out["qeeg_aperiodic_slope"] = float(coef[0])

    asym_total, asym_delta = [], []
    for left, right in HOMOLOGOUS_PAIRS:
        if left in channels and right in channels:
            li, ri = channels.index(left), channels.index(right)
            asym_total.append(np.log10(total[li] / total[ri]))
            delta = band_power(1.0, 4.0)
            asym_delta.append(np.log10((delta[li] + 1e-12) / (delta[ri] + 1e-12)))
    if asym_total:
        out["qeeg_max_asymmetry"] = float(np.max(np.abs(asym_total)))
        out["qeeg_max_delta_asymmetry"] = float(np.max(np.abs(asym_delta)))
    # Excess kurtosis of the filtered trace: sharp transients (spikes, sharp
    # waves -- and some artifacts) make the amplitude distribution heavy-tailed.
    out["qeeg_median_kurtosis"] = float(np.median(kurtosis(data, axis=-1)))
    return out


def ekg_features(ekg_uv: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Heart rate / variability from the EKG lead via R-peak detection."""
    nan = {"ekg_hr_bpm": float("nan"), "ekg_rr_std_ms": float("nan"), "ekg_valid_frac": 0.0}
    if ekg_uv.std() < 1.0:
        return nan
    sos = sps.butter(3, [5.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    x = sps.sosfiltfilt(sos, ekg_uv)
    # The lead polarity varies across montages; R-peaks are the heavier tail.
    if abs(np.percentile(x, 0.5)) > abs(np.percentile(x, 99.5)):
        x = -x
    height = np.percentile(x, 99) * 0.4
    peaks, _ = sps.find_peaks(x, height=height, distance=int(0.33 * sfreq))
    if len(peaks) < 10:
        return nan
    rr = np.diff(peaks) / sfreq
    plausible = (rr > 0.33) & (rr < 2.0)                         # 30-180 bpm
    valid_frac = float(plausible.mean())
    if plausible.sum() < 10:
        return {**nan, "ekg_valid_frac": valid_frac}
    rr = rr[plausible]
    return {
        "ekg_hr_bpm": float(60.0 / np.median(rr)),
        "ekg_rr_std_ms": float(1000.0 * np.std(rr)),
        "ekg_valid_frac": valid_frac,
    }


def auxiliary_features(aux: Dict[str, np.ndarray], sfreq: float) -> Dict[str, float]:
    """Photic stimulator and the device-computed burst-suppression trends.

    ``SUPPR`` (% suppression), ``BURSTS`` (bursts/min) and ``IBI`` (inter-burst
    interval, s) are 1-Hz trend channels written by the acquisition software.
    An all-zero trend means the device did not compute it, so it is NaN rather
    than "no suppression".
    """
    out: Dict[str, float] = {}
    photic = aux.get("PHOTIC")
    out["photic_performed"] = bool(photic is not None and np.ptp(photic) > 0)
    if photic is not None and out["photic_performed"]:
        # The stimulator channel carries one pulse per flash; a second containing
        # any pulse counts as a second of stimulation.
        seconds = int(len(photic) // sfreq)
        per_second = photic[: int(seconds * sfreq)].reshape(seconds, -1)
        out["photic_stim_sec"] = float(np.any(per_second != 0, axis=1).sum())
    else:
        out["photic_stim_sec"] = 0.0 if photic is not None else float("nan")
    for name in ("SUPPR", "BURSTS", "IBI"):
        trace = aux.get(name)
        if trace is None or not np.any(trace):
            out[f"trend_{name.lower()}_mean"] = float("nan")
            continue
        out[f"trend_{name.lower()}_mean"] = float(np.mean(trace))
    suppr = aux.get("SUPPR")
    out["trend_suppr_gt50_frac"] = (
        float((suppr > 50).mean()) if suppr is not None and np.any(suppr) else float("nan"))
    return out


def extract_recording_features(path: str) -> Dict[str, object]:
    """Every meta feature for one EDF recording. Never raises: errors are recorded."""
    import mne  # heavy import, deferred so header-only callers do not pay for it

    row: Dict[str, object] = {"path": path}
    parts = path.replace("\\", "/").split("/")
    row["tuh_split"] = next((p for p in parts if p in ("train", "eval")), None)
    row["label"] = next((p for p in parts if p in ("normal", "abnormal")), None)
    try:
        row.update(header_features(path))
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        names = [canonical_channel(c) for c in raw.ch_names]
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data() * 1e6                              # V -> uV

        scalp = [c for c in SCALP_CHANNELS if c in names]
        scalp_data = data[[names.index(c) for c in scalp]]
        row.update(quality_features(scalp_data, sfreq, float(row["physical_max_uv"])))
        row.update(qeeg_features(scalp_data, sfreq, scalp))

        ekg = next((i for i, n in enumerate(names) if "EKG" in n or "ECG" in n), None)
        row.update(ekg_features(data[ekg], sfreq) if ekg is not None
                   else ekg_features(np.zeros(10), sfreq))
        # Trend channels are stored in their own units, not volts.
        aux = {n: data[i] / 1e6 if n in ("SUPPR", "BURSTS", "IBI") else data[i]
               for i, n in enumerate(names) if n in ("PHOTIC", "SUPPR", "BURSTS", "IBI")}
        row.update(auxiliary_features(aux, sfreq))
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001 -- one bad file must not sink a corpus sweep
        logger.warning("Feature extraction failed for %s: %s", path, exc)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def extract_corpus_features(
    root: str,
    *,
    pattern: str = "**/*.edf",
    n_jobs: int = 1,
    limit: Optional[int] = None,
):
    """Sweep every EDF under *root* into a pandas DataFrame (one row per recording)."""
    import pandas as pd

    paths = sorted(p for p in glob(os.path.join(root, pattern), recursive=True)
                   if "/processed" not in p)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No EDF files matched {pattern!r} under {root}")
    rows = []
    if n_jobs <= 1:
        rows = [extract_recording_features(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(extract_recording_features, p) for p in paths]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 200 == 0:
                    logger.info("%d / %d recordings", i, len(paths))
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
