"""Line-noise and amplitude measures of labram.eval.preprocessing_qc."""
import numpy as np

from labram.eval import preprocessing_qc as qc


def _signal(fs=250.0, seconds=60, line_uv=0.0, f0=60.0, seed=0):
    t = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 10, size=(4, t.size)) + line_uv * np.sin(2 * np.pi * f0 * t)
    return x, fs


def test_line_prominence_detects_a_line_only_where_it_is():
    x, fs = _signal(line_uv=5.0)
    f, p = qc.channel_median_psd(x, fs)
    assert qc.line_prominence_db(f, p, 60.0) > 10
    assert abs(qc.line_prominence_db(f, p, 50.0)) < 2
    clean_f, clean_p = qc.channel_median_psd(_signal()[0], fs)
    assert abs(qc.line_prominence_db(clean_f, clean_p, 60.0)) < 2
    assert np.isnan(qc.line_prominence_db(f, p, 124.0))          # flank beyond Nyquist


def test_band_fraction_and_amplitude_stats():
    x, fs = _signal(line_uv=20.0)
    f, p = qc.channel_median_psd(x, fs)
    assert 0.3 < qc.band_fraction(f, p, 55, 65) < 1.0
    x[1] = 0.0                                                       # a flat channel
    stats = qc.amplitude_stats(x, phys_max=5000.0, fs=fs)
    assert stats["flat_channel_frac"] == 0.25 and stats["clip_frac"] == 0.0


def test_spectrum_interpolation_removes_line_without_hole():
    clean, fs = _signal(seconds=60)
    # No bin selected (f0 between grid points): the STFT round trip is exact.
    assert np.allclose(qc.spectrum_interpolate(clean, fs, f0=60.05, half=0.01), clean, atol=1e-6)
    lined, _ = _signal(seconds=60, line_uv=5.0)
    out = qc.spectrum_interpolate(lined, fs)
    f, p = qc.channel_median_psd(out, fs)
    assert abs(qc.line_prominence_db(f, p, 60.0)) < 2                 # line gone ...
    f, p = qc.channel_median_psd(qc.spectrum_interpolate(clean, fs), fs)
    assert qc.line_prominence_db(f, p, 60.0) > -2                       # ... and no hole
    assert out.shape == lined.shape
