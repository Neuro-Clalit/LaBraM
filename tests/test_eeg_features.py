"""Handcrafted EEG feature extractors (labram.features) on synthetic signals."""

import json
import warnings

import numpy as np
import pytest

pywt = pytest.importorskip("pywt")

from labram.features import extractors as fx  # noqa: E402
from labram.features import tuab  # noqa: E402

FS = 200.0
T = np.arange(2000) / FS


def _sine(freq, amp=10.0):
    return amp * np.sin(2 * np.pi * freq * T)


def _pink(n, exponent, seed=0):
    """Noise whose PSD falls as 1/f**exponent."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    return np.fft.irfft(spec / f ** (exponent / 2), n)


def test_spectral_features_alpha_sine_dominates_alpha_band():
    x = (_sine(10.0) + 0.1 * np.random.default_rng(0).standard_normal(2000))[None, None]
    freqs, psd = fx.welch_psd(x)
    feats, names = fx.spectral_features(freqs, psd)
    f = dict(zip(names, feats[0, 0]))
    assert f["rel_alpha"] > 0.9
    assert f["peak_alpha_freq"] == pytest.approx(10.0, abs=0.5)
    assert len(names) == feats.shape[-1]


@pytest.mark.parametrize("exponent", [1.0, 2.0])
def test_aperiodic_exponent_recovers_power_law(exponent):
    x = np.stack([_pink(20000, exponent, seed=s) for s in range(4)])
    freqs, psd = fx.welch_psd(x)
    chi, _ = fx.aperiodic_exponent(freqs, psd)
    assert np.allclose(chi, exponent, atol=0.15)


def test_dwt_sine_lands_in_matching_level():
    # 10 Hz is inside D4 (6.25-12.5 Hz) at 200 Hz.
    feats, names = fx.dwt_features(_sine(10.0)[None], "db4", 6)
    rel = {n[len("rel_E_"):]: v for n, v in zip(names, feats[0]) if n.startswith("rel_E_")}
    assert max(rel, key=rel.get) == "D4"
    assert fx.dwt_band_edges(6)[3] == (6.25, 12.5)


def test_wpd_uniform_tiling_and_sine_node():
    feats, names = fx.wpd_features(_sine(20.0)[None], "db4", 5)
    assert len(names) == 16  # 3.125 Hz nodes up to 50 Hz
    assert names[int(np.argmax(feats[0]))] == "log_E_wp_18.75-21.875Hz"


def test_cwt_sine_amplitude_and_peak_frequency():
    t = np.arange(8000) / FS
    x = (3.0 * np.sin(2 * np.pi * 10.0 * t))[None]
    freqs = fx.cwt_frequencies(25)
    logp = fx.cwt_log_power(x, freqs)
    mid = logp[:, 0, 2000:6000].mean(axis=-1)
    assert freqs[np.argmax(mid)] == pytest.approx(10.0, rel=0.1)
    # Unit peak gain: a 3 µV sinusoid analysed exactly at its frequency -> 9 µV^2.
    at10 = fx.cwt_log_power(x, [10.0])
    assert 10 ** at10[0, 0, 4000] == pytest.approx(9.0, rel=0.01)


@pytest.mark.parametrize("wavelet", ["cmor0.5-1.0", "cmor1.5-1.0", "cmor5.0-1.0"])
def test_cwt_window_features_match_full_resolution(wavelet):
    rng = np.random.default_rng(1)
    x = np.stack([_pink(8000, 1.5, seed=s) for s in range(2)]) + 0.1 * rng.standard_normal((2, 8000))
    freqs = fx.cwt_frequencies(10)
    feats, names = fx.cwt_window_features(x, freqs, wavelet, win=2000)
    full = fx.cwt_log_power(x, freqs, wavelet).reshape(len(freqs), 2, 4, 2000)
    ref_mean = full.mean(axis=-1).transpose(2, 1, 0)
    assert feats.shape == (4, 2, 2 * len(freqs)) and len(names) == 2 * len(freqs)
    assert np.abs(feats[..., : len(freqs)] - ref_mean).mean() < 0.05


def test_permutation_entropy_extremes():
    noise = np.random.default_rng(0).standard_normal((3, 4000))
    assert np.all(fx.permutation_entropy(noise) > 0.98)
    assert fx.permutation_entropy(np.arange(1000.0)[None])[0] == pytest.approx(0.0, abs=1e-9)


def test_multiscale_pe_and_time_features_shapes():
    x = np.random.default_rng(0).standard_normal((2, 3, 2000))
    mspe, mnames = fx.multiscale_pe_features(x)
    tf, tnames = fx.time_features(x)
    assert mspe.shape == (2, 3, len(mnames)) and tf.shape == (2, 3, len(tnames))
    assert fx.coarse_grain(x, 3).shape == (2, 3, 666)


def test_prepare_car_and_channel_selection():
    seg = np.random.default_rng(0).standard_normal((23, 4000))
    x = tuab.prepare(seg, tuab.PREP_CAR)
    assert x.shape == (19, 4000)
    assert np.allclose(x.mean(axis=0), 0.0)
    assert tuab.to_windows(x).shape == (2, 19, 2000)


def test_clean_mask_rejects_large_and_flat_windows():
    xw = np.random.default_rng(0).standard_normal((3, 4, 2000)) * 10
    xw[1, 2, 100] = 1000.0  # > 800 µV
    xw[2, 0] = 0.0  # flat channel
    assert tuab.clean_mask(tuab.window_quality(xw)).tolist() == [True, False, False]


def test_extract_corpus_on_synthetic_npy(tmp_path):
    rng = np.random.default_rng(0)
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    recs = []
    for i in range(2):
        stem = f"aaaaaaa{i}_s001_t000"
        np.save(rec_dir / f"{stem}.npy", (rng.standard_normal((8 * 2000, 23)) * 10).astype(np.float32))
        recs.append({"stem": stem, "subject": f"aaaaaaa{i}", "source_split": "train",
                     "label": "normal", "sex": "F", "age": 40 + i})
    (tmp_path / "manifest.json").write_text(json.dumps({"recordings": recs}))
    out = tmp_path / "feats.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuab.extract_corpus(str(tmp_path), str(out), n_jobs=1, n_skip=2, n_use=4)
    d = np.load(out)
    names = tuab.feature_names()
    assert d["quality"].shape == (2, 6, 4)
    assert d["core/1020_car_1-45Hz/dwt_db4"].shape == (2, 6, 19, len(names["dwt_db4"]))
    for fam in ("time", "spectral", "psd_bins", "mspe", "wpd_db4_L5", "cwt_cmor1.5-1.0"):
        assert d[f"rec/{fam}"].shape == (2, 19, len(names[fam]))
    assert d["fbcov"].shape == (2, len(fx.FILTERBANK), 19, 19)
    assert d["meta/age"].tolist() == [40.0, 41.0]
    assert np.isfinite(d["rec/spectral"]).all()


def _random_spd(n_mats, n, seed=0):
    a = np.random.default_rng(seed).standard_normal((n_mats, n, 3 * n))
    return a @ np.swapaxes(a, -1, -2) / (3 * n)


def test_riemann_mean_is_affine_invariant_and_tangent_norm_is_distance():
    from labram.features.riemann import mean_riemann, tangent_space

    covs = _random_spd(20, 5)
    ref = mean_riemann(covs)
    # Tangent vectors at the Karcher mean average to zero.
    assert np.allclose(tangent_space(covs, ref).mean(axis=0), 0.0, atol=1e-6)
    # Congruence by any invertible A maps the mean to A ref A^T.
    a = np.random.default_rng(1).standard_normal((5, 5)) + 5 * np.eye(5)
    assert np.allclose(mean_riemann(a @ covs @ a.T), a @ ref @ a.T, rtol=1e-5, atol=1e-8)
    # ||tangent vector|| equals the affine-invariant distance sqrt(sum log^2 eig(ref^-1 C)).
    eig = np.linalg.eigvals(np.linalg.solve(ref, covs[0])).real
    assert np.linalg.norm(tangent_space(covs[:1], ref)) == pytest.approx(np.sqrt((np.log(eig) ** 2).sum()))


def test_scattering_first_order_peak_and_second_order_modulation():
    pytest.importorskip("kymatio")
    t = np.arange(1000) / 100.0
    carrier = np.sin(2 * np.pi * 10.0 * t)
    am = (1.0 + 0.9 * np.sin(2 * np.pi * 0.5 * t)) * carrier
    feats, names = fx.scattering_features(np.stack([carrier, am]), J=7, Q=(8, 1), fs=100.0)
    assert feats.shape == (2, len(names))
    s1 = [i for i, n in enumerate(names) if n.startswith("S1_")]
    f1 = np.array([float(names[i][3:-2]) for i in s1])
    assert f1[np.argmax(feats[0, s1])] == pytest.approx(10.0, rel=0.1)
    # Envelope fluctuation at 0.5 Hz shows up in the second order of the ~10 Hz band only for AM.
    s2 = [i for i, n in enumerate(names) if n.startswith("S2_")]
    near = [i for i in s2 if abs(float(names[i].split("_")[1][:-2]) - 10.0) < 0.6
            and 0.4 < float(names[i].split("mod")[1][:-2]) < 0.8]
    assert near and np.all(feats[1, near] > feats[0, near] + 0.5)


def test_extract_recording_scattering_shapes(tmp_path):
    pytest.importorskip("kymatio")
    path = tmp_path / "rec.npy"
    np.save(path, (np.random.default_rng(0).standard_normal((8 * 2000, 23)) * 10).astype(np.float32))
    out = tuab.extract_recording_scattering(str(path), n_skip=2, n_use=3)
    names = tuab.scattering_names()
    for key in tuab.SCAT_CONFIGS:
        assert out[f"rec/{key}"].shape == (19, len(names[key]))
        assert np.isfinite(out[f"rec/{key}"]).all()


def test_stft_and_multitaper_spectrograms_peak_and_variance():
    x = (_sine(10.0) + np.random.default_rng(0).standard_normal(2000))[None]
    f, s = fx.stft_spectrogram(x)
    feats, names = fx.spectrogram_stats(f, s)
    means = feats[0, : len(names) // 2]
    assert names[int(np.argmax(means))] == "stft_mean_10Hz"
    # Multitaper averaging lowers the frame-to-frame variance of log power on noise.
    noise = np.random.default_rng(1).standard_normal((4, 2000))
    _, s1 = fx.stft_spectrogram(noise, seg_sec=2.0)
    _, sm = fx.multitaper_spectrogram(noise)
    assert np.log10(sm).std(axis=-1).mean() < np.log10(s1).std(axis=-1).mean()


def test_spectrogram_dynamics_separates_bursty_from_sustained_alpha():
    t = np.arange(4000) / FS
    sustained = np.sin(2 * np.pi * 10.0 * t)
    bursty = sustained * (np.sin(2 * np.pi * 0.25 * t) > 0)
    x = np.stack([sustained, bursty]) + 0.05 * np.random.default_rng(0).standard_normal((2, 4000))
    f, s = fx.stft_spectrogram(x)
    feats, names = fx.spectrogram_dynamics(f, s)
    i = names.index("std_logpow_alpha")
    assert feats[1, i] > feats[0, i] + 0.3
    assert len(names) == feats.shape[-1] == 11


def test_cepstral_features_shape_and_gain_invariance_of_higher_coefficients():
    x = np.random.default_rng(0).standard_normal((2, 2000))
    f, s = fx.stft_spectrogram(x, pad=4)
    c1, names = fx.cepstral_features(f, s)
    c2, _ = fx.cepstral_features(f, s * 100.0)  # +20 dB gain shifts only c0
    assert c1.shape == (2, len(names)) == (2, 26)
    assert np.allclose(c1[:, 1:13], c2[:, 1:13], atol=1e-6)
    assert np.all(c2[:, 0] > c1[:, 0])
    with pytest.raises(ValueError):  # 1 Hz grid is too coarse for the narrow low filters
        fx.cepstral_features(*fx.stft_spectrogram(x, pad=1), n_filters=40)
