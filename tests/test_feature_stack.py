"""scripts/feature_stack.py: the stacking test of handcrafted features on a frozen fold model."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import feature_stack as fs  # noqa: E402


def _direct_ridge(z, x, r, alpha):
    """Normal equations with the penalty on the feature block only."""
    a = np.c_[z, x]
    pen = np.diag(np.r_[np.zeros(z.shape[1]), np.full(x.shape[1], alpha)])
    return np.linalg.solve(a.T @ a + pen, a.T @ r)


def test_ridge_path_matches_normal_equations_tall_and_wide():
    rng = np.random.default_rng(0)
    for n, p in ((120, 15), (30, 80)):          # Gram eigenvectors, and the thin SVD for p > n
        z = np.c_[np.ones(n), rng.normal(size=n)]
        x = rng.normal(size=(n, p))
        r = x[:, 0] - 2 * z[:, 1] + rng.normal(size=n)
        alphas = np.array([0.5, 20.0, 3000.0])
        zt, xt = np.c_[np.ones(7), rng.normal(size=7)], rng.normal(size=(7, p))
        got = fs.ridge_path(z, x, r, alphas)(zt, xt)
        for a, g in zip(alphas, got):
            coef = _direct_ridge(z, x, r, a)
            np.testing.assert_allclose(g, np.c_[zt, xt] @ coef, rtol=1e-6, atol=1e-8)


def test_ridge_path_without_features_is_the_unpenalised_fit():
    rng = np.random.default_rng(1)
    z = np.c_[np.ones(50), rng.normal(size=50)]
    r = 1.5 + 0.3 * z[:, 1] + rng.normal(size=50)
    got = fs.ridge_path(z, np.zeros((50, 0)), r, np.array([10.0]))(z, np.zeros((50, 0)))[0]
    np.testing.assert_allclose(got, z @ np.linalg.lstsq(z, r, rcond=None)[0], atol=1e-10)


def test_grouped_folds_keep_a_patient_together():
    groups = np.repeat(np.arange(40).astype(str), 3)
    folds = fs.grouped_folds(groups, n_folds=5)
    assert set(folds) == set(range(5))
    assert all(len(set(folds[groups == g])) == 1 for g in np.unique(groups))


def test_paired_bootstrap_resamples_patients():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(200), 2).astype(str)
    lo, hi = fs.paired_bootstrap(rng.normal(-0.5, 1.0, size=400), groups, n_boot=500)
    assert lo < -0.5 < hi and hi < 0
    # one patient with every recording: no spread between resamples
    lo, hi = fs.paired_bootstrap(np.array([1.0, 3.0]), np.array(["a", "a"]), n_boot=50)
    assert lo == hi == 2.0


def test_standardize_clips_and_fills():
    x = np.array([[0.0, 1.0], [2.0, np.nan], [4.0, 3.0], [np.inf, 5.0]])
    std = fs.Standardize().fit(x)
    z = std(np.array([[2.0, np.nan], [1e6, 3.0]]))
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0          # the fit mean; a missing value becomes the mean
    assert z[1, 0] == 6.0                             # clipped


def test_fold_stems_split_by_the_leading_stem_token(tmp_path):
    cohort, fold = tmp_path / "cohort", tmp_path / "run" / "cv" / "fold_1"
    cohort.mkdir()
    fold.mkdir(parents=True)
    stems = ["a_s1_t0", "a_s2_t0", "b_s1_t0", "c_s1_t0", "d_s1_t0"]
    pd.DataFrame({"stem": stems, "split": ["train", "val", "train", "val", "test"]}).to_parquet(cohort / "cohort.parquet")
    json.dump({"split_by": "subject", "test_set": "held_out",
               "folds": [{"groups": ["a"]}, {"groups": ["b", "c"]}]}, open(fold / "cv_split.json", "w"))
    train, val, test = fs.fold_stems(str(fold), str(cohort))
    assert (train, val, test) == (["a_s1_t0", "a_s2_t0"], ["b_s1_t0", "c_s1_t0"], ["d_s1_t0"])


def test_cross_fitted_age_is_out_of_fold_on_the_fit_rows():
    rng = np.random.default_rng(3)
    n, p = 400, 12
    x = rng.normal(size=(n, p))
    age = 50 + 8 * x[:, 0] - 5 * x[:, 1] + rng.normal(scale=2.0, size=n)
    fit = np.ones(n, bool)
    fit[:20] = False                                  # e.g. patients shared with the training set
    folds = fs.grouped_folds(np.arange(fit.sum()).astype(str))
    xt = rng.normal(size=(50, p))
    fv, ft, alpha, mae = fs.cross_fitted_age(x, xt, age, fit, folds, np.array([1e-3, 1.0, 1e3]))
    assert fv.shape == (n,) and ft.shape == (50,) and mae < 3.0
    assert np.corrcoef(fv, age)[0, 1] > 0.95
    # the fit rows are predicted by models that did not see them: not the all-rows model's prediction
    full = fs.ridge_path(np.ones((fit.sum(), 1)), x[fit], age[fit], np.array([alpha]))(np.ones((n, 1)), x)[0]
    assert not np.allclose(fv[fit], full[fit]) and np.allclose(fv[~fit], full[~fit])


def _state_recording(tmp_path, n_windows=36, alpha_windows=range(6, 21)):
    """23-channel recording: pink-ish noise everywhere, a 10 Hz rhythm on the posterior channels in some windows."""
    from labram.features import tuab
    rng = np.random.default_rng(4)
    w = tuab.WIN
    x = np.cumsum(rng.normal(size=(n_windows * w, 23)), axis=0) * 0.5 + rng.normal(size=(n_windows * w, 23)) * 3
    x -= x.mean(axis=0)
    t = np.arange(w) / 200.0
    for k in alpha_windows:
        for ch in tuab.STATE_POSTERIOR:
            x[k * w:(k + 1) * w, tuab.STORED_CHANNELS.index(ch)] += 25 * np.sin(2 * np.pi * 10 * t)
    path = tmp_path / "rec.npy"
    np.save(path, x.astype(np.float32))
    return str(path)


def test_state_spectra_separate_the_alpha_windows(tmp_path):
    from labram.features import tuab
    out = tuab.extract_recording_state_spectra(_state_recording(tmp_path))
    names = tuab.state_spectra_names()
    assert all(out[k].shape == (19, len(v)) for k, v in names.items())
    o1, b10 = tuab.CHANNELS_1020.index("O1"), names["state/high/periodic"].index("periodic_high_10Hz")
    assert out["n_state"] == 10                               # a third of the 30 clean windows
    assert out["state/high/periodic"][o1, b10] > out["state/low/periodic"][o1, b10] + 0.5   # log10 units
    assert out["alpha_index"][0] > out["alpha_index"][2] > out["alpha_index"][1]
    assert np.isfinite(out["state/all/aperiodic"]).all() and (out["state/all/aperiodic"][:, 0] > 0).all()


def test_split_states_with_few_clean_windows():
    from labram.features import tuab
    alpha = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
    clean = np.array([1, 1, 0, 1, 1, 0, 0, 0], bool)          # 4 clean: fewer than twice the minimum
    high, low = tuab.split_states(alpha, clean)
    assert (high == clean).all() and (low == clean).all()
    high, low = tuab.split_states(alpha, np.ones(8, bool))    # 8 clean: three per state
    assert set(np.flatnonzero(high)) == {1, 6, 4} and set(np.flatnonzero(low)) == {0, 5, 3}


def test_cross_spectra_diagonal_is_the_welch_spectrum():
    from scipy import signal
    from labram.features import extractors as fx
    x = np.random.default_rng(5).normal(size=(3, 4, 2000))
    freqs, csd = fx.cross_spectra(x)
    f, psd = signal.welch(x, fs=200.0, nperseg=400, noverlap=200, axis=-1)
    sel = (f >= 1.0) & (f < 45.0)
    np.testing.assert_allclose(freqs, f[sel])
    np.testing.assert_allclose(np.real(np.einsum("wiif->wif", csd)), psd[..., sel], rtol=1e-9)
    np.testing.assert_allclose(csd, np.conj(np.swapaxes(csd, 1, 2)))            # Hermitian


def test_band_connectivity_separates_lagged_from_zero_lag_coupling():
    from scipy import signal
    from labram.features import extractors as fx
    rng = np.random.default_rng(6)
    n = 30 * 2000
    src = 4.0 * fx.bandpass(rng.normal(size=(1, n)), 8.0, 13.0)[0]             # an alpha-band rhythm
    lagged = np.imag(signal.hilbert(src))                                       # the same, a quarter cycle later
    noise = rng.normal(size=(4, n))
    x = np.stack([src + noise[0], src + noise[1], lagged + noise[2], noise[3]]).reshape(4, 30, 2000).transpose(1, 0, 2)
    freqs, csd = fx.cross_spectra(x)
    feats, names = fx.band_connectivity(freqs, csd.mean(axis=0))
    v = dict(zip(names, feats))
    assert v["coh_alpha_0-1"] > 0.5 and v["icoh_alpha_0-1"] < 0.1               # same phase: coherent, no lag
    assert v["icoh_alpha_0-2"] > 0.5                                             # quarter-cycle lag
    assert v["coh_alpha_0-3"] < 0.1 and v["icoh_alpha_0-3"] < 0.1               # unrelated noise
    assert v["coh_beta_0-1"] < 0.1                                               # no coupling outside the band
    assert len(names) == feats.shape[-1] == 2 * 5 * 6


def test_band_covariance_adds_up_to_the_band_limited_variance():
    from labram.features import extractors as fx
    x = np.random.default_rng(7).normal(size=(40, 3, 2000))
    freqs, csd = fx.cross_spectra(x)
    cov = fx.band_covariance(freqs, csd.mean(axis=0))
    assert cov.shape == (6, 3, 3)
    # white noise of unit variance: power 1/100 per Hz, so a band of width B holds about B / 100
    widths = np.array([hi - lo for lo, hi in fx.FILTERBANK.values()])
    np.testing.assert_allclose(cov[:, 0, 0], widths / 100.0, rtol=0.15)
    assert np.abs(cov[:, 0, 1]).max() < 0.02


def test_study_extractor_variants(tmp_path):
    from labram.features import extractors as fx
    from labram.features import tuab
    path = _state_recording(tmp_path)
    w = np.linspace(1.0, 2.0, 30)
    out = tuab.extract_recording_study(path, attention=w / w.sum())
    names = tuab.study_names()
    for fam in tuab.STUDY_CORE:
        assert out[fam].shape == (19, len(names[fam]))
    assert all(out[f].shape == (len(names[f]),) for f in tuab.STUDY_CONN)
    assert all(out[f].shape == (6, 19, 19) for f in tuab.STUDY_COV)
    ref = tuab.extract_recording_cohort(path)                                    # the standard setting
    np.testing.assert_allclose(out["len10"], np.concatenate([ref["rec/spectral"], ref["rec/psd_bins"], ref["rec/time"]],
                                                            axis=1), rtol=1e-5, atol=1e-5)
    # attention-weighted summary = the weighted mean of the per-window features
    xw = tuab.to_windows(tuab.prepare(tuab.load_segment(path, 36), tuab.PREP_REF))[6:]
    np.testing.assert_allclose(out["agg_attn"], np.tensordot(w / w.sum(), tuab._core(xw), axes=1), rtol=1e-4, atol=1e-4)
    assert np.isnan(tuab.extract_recording_study(path)["agg_attn"]).all()        # no weights given
    o1, alpha = tuab.CHANNELS_1020.index("O1"), names["len10"].index("rel_alpha")
    assert out["state_high"][o1, alpha] > out["state_low"][o1, alpha]
    # fewer segments per estimate inflate coherence: 10 s estimates sit above the all-data estimate
    coh = np.array([n.startswith("coh_") for n in names["conn_all"]])
    assert out["conn_len10"][coh].mean() > out["conn_len30"][coh].mean() > out["conn_all"][coh].mean()


def test_study2_controls(tmp_path):
    from labram.features import tuab
    path = _state_recording(tmp_path)                 # windows 6-20 carry the posterior alpha rhythm
    w = np.r_[np.ones(15), np.zeros(15)]              # all the weight on the alpha windows
    out = tuab.extract_recording_study2(path, attention=w / w.sum())
    assert all(out[f].shape == (6, 19, 19) for f in tuab.STUDY2_COV)
    assert all(out[f].shape == (len(tuab.study_names()[f]),) for f in tuab.STUDY2_CONN)
    alpha, o1 = list(tuab.fx.FILTERBANK).index("alpha"), tuab.CHANNELS_1020.index("O1")
    # the calm third avoids the high-amplitude alpha windows; the attention-weighted estimate is made of them
    assert out["cov_attn"][alpha, o1, o1] > 3 * out["cov_calm"][alpha, o1, o1]
    assert out["calm_high_overlap"] == 0.0
    assert out["cov_calm"][alpha, o1, o1] < out["cov_rand"][alpha, o1, o1] < out["cov_attn"][alpha, o1, o1]
    assert np.isnan(tuab.extract_recording_study2(path)["cov_attn"]).all()
