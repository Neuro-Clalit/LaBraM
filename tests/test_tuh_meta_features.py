"""Tests for data.tuh_meta_features: per-recording meta features for TUH corpora.

Signals are synthetic (sinusoids, noise, spike trains) with a known answer, and
the end-to-end test writes a minimal TUH-style EDF file, so no dataset is needed.
"""
import numpy as np
import pytest

from labram.data.tuh_meta_features import (
    SCALP_CHANNELS,
    auxiliary_features,
    canonical_channel,
    ekg_features,
    extract_recording_features,
    header_features,
    npy_signal_features,
    parse_header_dump,
    qeeg_features,
    quality_features,
    read_edf_signal_header,
)

SFREQ = 250.0
RNG = np.random.default_rng(0)


def _background(n_channels=len(SCALP_CHANNELS), seconds=120, amp=10.0):
    return amp * RNG.standard_normal((n_channels, int(seconds * SFREQ)))


def _write_tuh_edf(path, signals, labels, *, sfreq=250, age=42, sex="M", physical_max=4999.847,
                   session="s003", recording_tail="XXX X"):
    """A minimal EDF in the layout TUH uses (1-s records, int16, uV)."""
    n_sig, n_samp = signals.shape
    n_rec = n_samp // sfreq
    dmax = 32767
    header = (
        b"0".ljust(8)
        + f"aaaaabcd {sex} 01-JAN-0000 aaaaabcd Age:{age}".encode().ljust(80)
        + f"Startdate 01-JAN-2012 aaaaabcd_{session} {recording_tail}".encode().ljust(80)
        + b"01.01.12" + b"00.00.00"
        + str(256 * (n_sig + 1)).encode().ljust(8) + b"".ljust(44)
        + str(n_rec).encode().ljust(8) + b"1.000000" + str(n_sig).encode().ljust(4)
    )
    fields = [
        [lab.ljust(16) for lab in labels],
        ["Unknown".ljust(80)] * n_sig,
        ["uV".ljust(8)] * n_sig,
        [f"{-physical_max:.2f}".ljust(8)] * n_sig,
        [f"{physical_max:.3f}"[:8].ljust(8)] * n_sig,
        [f"{-dmax}".ljust(8)] * n_sig,
        [f"{dmax}".ljust(8)] * n_sig,
        ["HP:0.000 Hz LP:0.0 Hz N:0.0".ljust(80)] * n_sig,
        [str(sfreq).ljust(8)] * n_sig,
        ["".ljust(32)] * n_sig,
    ]
    header += "".join("".join(f) for f in fields).encode()
    digital = np.clip(np.round(signals / physical_max * dmax), -dmax, dmax).astype("<i2")
    records = digital[:, : n_rec * sfreq].reshape(n_sig, n_rec, sfreq).transpose(1, 0, 2)
    path.write_bytes(header + records.tobytes())
    return path


def test_canonical_channel_strips_tuh_prefix_and_reference():
    assert canonical_channel("EEG FP1-REF") == "FP1"
    assert canonical_channel("EEG T3-LE") == "T3"
    assert canonical_channel("PHOTIC-REF") == "PHOTIC"
    assert canonical_channel("IBI") == "IBI"


def test_quality_flags_flat_clipped_and_mains_noise():
    data = _background()
    data[0] = 0.01 * RNG.standard_normal(data.shape[1])           # dead electrode
    data[1, :1000] = 4999.847                                      # saturated amplifier
    t = np.arange(data.shape[1]) / SFREQ
    data[2:] += 50.0 * np.sin(2 * np.pi * 60.0 * t)                # US mains hum

    q = quality_features(data, SFREQ, physical_max_uv=4999.847)
    assert q["q_n_flat_channels"] == 1
    assert q["q_clip_frac"] > 0
    assert q["q_line60_log_ratio"] > q["q_line50_log_ratio"] + 1.0


def test_quality_of_clean_recording_is_clean():
    q = quality_features(_background(), SFREQ, physical_max_uv=4999.847)
    assert q["q_clip_frac"] == 0
    assert q["q_n_flat_channels"] == 0
    assert q["q_bad_epoch_frac"] == 0


def test_qeeg_finds_posterior_rhythm_and_slowing():
    channels = list(SCALP_CHANNELS)
    t = np.arange(int(120 * SFREQ)) / SFREQ
    alpha = _background(amp=2.0)
    for c in ("O1", "O2", "P3", "P4"):
        alpha[channels.index(c)] += 30.0 * np.sin(2 * np.pi * 9.5 * t)
    slow = _background(amp=2.0) + 30.0 * np.sin(2 * np.pi * 2.0 * t) * RNG.uniform(
        0.5, 1.5, (len(channels), 1))

    healthy = qeeg_features(alpha, SFREQ, channels)
    slowed = qeeg_features(slow, SFREQ, channels)
    assert healthy["qeeg_pdr_freq"] == pytest.approx(9.5, abs=0.5)
    assert slowed["qeeg_rel_delta"] > healthy["qeeg_rel_delta"]
    assert slowed["qeeg_slowing_ratio"] > healthy["qeeg_slowing_ratio"]


def test_qeeg_asymmetry_detects_one_sided_slowing():
    channels = list(SCALP_CHANNELS)
    t = np.arange(int(120 * SFREQ)) / SFREQ
    sym = _background()
    focal = sym.copy()
    focal[channels.index("T3")] += 60.0 * np.sin(2 * np.pi * 2.0 * t)
    assert (qeeg_features(focal, SFREQ, channels)["qeeg_max_delta_asymmetry"]
            > qeeg_features(sym, SFREQ, channels)["qeeg_max_delta_asymmetry"] + 0.5)


def test_ekg_heart_rate_from_r_peaks():
    n = int(120 * SFREQ)
    ekg = 5.0 * RNG.standard_normal(n)
    beats = np.arange(0, n, int(SFREQ * 60 / 72))                  # 72 bpm
    for b in beats:
        ekg[b: b + 5] -= 400.0                                     # inverted lead
    out = ekg_features(ekg, SFREQ)
    assert out["ekg_hr_bpm"] == pytest.approx(72, abs=2)
    assert out["ekg_valid_frac"] > 0.9


def test_ekg_absent_is_nan_not_zero():
    out = ekg_features(np.zeros(5000), SFREQ)
    assert np.isnan(out["ekg_hr_bpm"])


def test_auxiliary_trends_all_zero_means_not_computed():
    n = int(60 * SFREQ)
    photic = np.zeros(n)
    photic[: int(10 * SFREQ): 25] = 1024.0                         # 10 s of flashes
    out = auxiliary_features({"PHOTIC": photic, "SUPPR": np.zeros(n),
                              "BURSTS": np.full(n, 2.0)}, SFREQ)
    assert out["photic_performed"] is True
    assert out["photic_stim_sec"] == 10
    assert np.isnan(out["trend_suppr_mean"])
    assert out["trend_bursts_mean"] == pytest.approx(2.0)


def test_header_and_end_to_end_on_synthetic_edf(tmp_path):
    labels = [f"EEG {c}-REF" for c in SCALP_CHANNELS] + ["EEG EKG1-REF", "EEG SP1-REF", "PHOTIC-REF"]
    signals = np.vstack([_background(), _background(n_channels=3)])
    (tmp_path / "abnormal").mkdir()
    path = _write_tuh_edf(tmp_path / "abnormal" / "aaaaabcd_s003_t001.edf",
                          signals, labels, recording_tail="XXX Compact_32ch")

    hdr = read_edf_signal_header(str(path))
    assert hdr.n_records == 120 and hdr.sfreq(0) == 250

    h = header_features(str(path))
    assert h["age"] == 42 and h["sex"] == "M" and h["session_number"] == 3
    assert h["token_number"] == 1
    assert h["equipment_tag"] == "Compact_32ch"
    assert h["has_sphenoidal"] and h["has_ekg"] and not h["has_emg"]
    assert h["duration_min"] == pytest.approx(2.0)

    row = extract_recording_features(str(path))
    assert row["error"] == ""
    assert row["label"] == "abnormal"
    assert row["q_n_flat_channels"] == 0
    assert "qeeg_pdr_freq" in row


def test_extraction_records_errors_instead_of_raising(tmp_path):
    bad = tmp_path / "broken.edf"
    bad.write_bytes(b"0" * 100)
    row = extract_recording_features(str(bad))
    assert row["error"]


def _dump_block(i, path, labels, *, machine="Compact_32ch", n_recs=600, rec_size=256):
    n = len(labels)
    return [
        f"       {i}: {path}",
        "\tBlock 3: Local Recording Information",
        f"\t lrci_machine = [{machine}]",
        f"\t ghdi_num_recs = [{n_recs}]",
        "\t ghdi_dur_rec = [1.000000]",
        f"\t chan_labels ({n}) = " + ", ".join(f"[{lab}]" for lab in labels),
        f"\t chan_phys_max ({n}) = " + ", ".join("[  5482.288]" for _ in labels),
        f"\t chan_rec_size ({n}) = " + ", ".join(f"[       {rec_size}]" for _ in labels),
        "\t duration of recording (secs) =      600.0",
        "\t  channel[   0]:      256.0 Hz (EEG FP1-REF)",
    ]


def test_parse_header_dump_gives_acquisition_columns():
    scalp = [f"EEG {c}-REF" for c in SCALP_CHANNELS]
    lines = (["beginning argument processing..."]
             + _dump_block(1, "../edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf",
                           scalp + ["EEG EKG1-REF", "EEG SP1-REF", "PHOTIC-REF", "SUPPR"])
             + _dump_block(2, "../edf/000/aaaaaaab/s002_2004/02_tcp_le/aaaaaaab_s002_t001.edf",
                           scalp + ["EMG-REF"], machine="#####", n_recs=1200, rec_size=250)
             + ["processed 2 out of 2 files successfully"])
    rows = list(parse_header_dump(lines))
    assert [r["stem"] for r in rows] == ["aaaaaaaa_s001_t000", "aaaaaaab_s002_t001"]
    first, second = rows
    assert first["path"].startswith("edf/000/aaaaaaaa")
    assert first["equipment_family"] == "Compact_32ch" and second["equipment_family"] == "unknown"
    assert first["has_ekg"] and first["has_sphenoidal"] and first["has_photic_channel"]
    assert first["has_trend_channels"] and not first["has_emg"]
    assert second["has_emg"] and not second["has_ekg"]
    assert first["n_scalp_channels"] == 19 and first["n_signals"] == 23
    assert first["sfreq"] == 256.0 and second["sfreq"] == 250.0
    assert first["duration_min"] == pytest.approx(10.0) and second["duration_min"] == pytest.approx(20.0)
    assert first["physical_max_uv"] == pytest.approx(5482.288)


def test_npy_features_skip_zero_filled_channels_and_unmeasurable_columns():
    sfreq, seconds = 200.0, 120
    channels = list(SCALP_CHANNELS) + ["A1", "A2"]
    t = np.arange(int(seconds * sfreq)) / sfreq
    data = 10.0 * RNG.standard_normal((len(t), len(channels)))
    data[:, channels.index("O1")] += 30.0 * np.sin(2 * np.pi * 10.0 * t)
    data[:, channels.index("O2")] += 30.0 * np.sin(2 * np.pi * 10.0 * t)
    mask = [True] * len(channels)
    data[:, channels.index("T3")] = 0.0                  # missing, zero-filled by the converter
    mask[channels.index("T3")] = False
    out = npy_signal_features(data.astype(np.float16), channels, mask, sfreq=sfreq, max_minutes=1.5)
    assert out["n_scalp_recorded"] == 18
    assert out["q_n_flat_channels"] == 0                 # the zero-filled T3 is not "flat"
    assert out["analysed_min"] == pytest.approx(1.5)
    assert 9.0 <= out["qeeg_pdr_freq"] <= 11.0
    assert not {"q_clip_frac", "q_line60_log_ratio", "q_line50_log_ratio"} & set(out)
