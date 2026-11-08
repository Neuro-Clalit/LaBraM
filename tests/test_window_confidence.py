"""Confidence-weighted window pooling (labram/eval/window_confidence.py), synthetic data."""
import numpy as np
import pandas as pd
import pytest

from labram.eval import window_confidence as wc


def _windows(n_rec=60, n_win=20, seed=0, noisy_frac=0.2):
    """Recordings whose noisy windows have a large TTA spread *and* a large
    error, so every confidence should find them."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rec):
        age = rng.uniform(20, 80)
        bias = rng.normal(0, 3)
        for k in range(n_win):
            noisy = rng.random() < noisy_frac
            sd = 15.0 if noisy else 2.0
            pred = age + bias + rng.normal(0, sd)
            rows.append({"split": "val" if r % 2 else "test", "recording": f"r{r}", "window": k,
                         "age": age, "cohort": "normal", "dataset": "temple", "pred": pred,
                         **{f"tta_{j}": pred + rng.normal(0, sd) for j in (1, 2, 3)},
                         "emg": (0.3 if noisy else 0.05) * rng.uniform(0.8, 1.2),
                         "emg_max": 0.5, "max_abs": 100.0, "amp_uv": 20.0, "min_std": 5.0})
    return wc.add_tta_stats(pd.DataFrame(rows))


def test_huber_weights_reduce_to_mean_and_downweight_outliers():
    df = pd.DataFrame({"split": "val", "recording": "a",
                       "pred": [58.0, 61.0, 59.0, 60.0, 44.0, 35.0]})
    w, mu = wc.huber_weights(df, delta=5.0)
    assert mu.iloc[0] == pytest.approx(57.0, abs=0.1)        # worked example of the plan
    assert np.all(w[:4] == 1.0) and w[5] < w[4] < 1.0
    w_big, mu_big = wc.huber_weights(df, delta=1e6)
    assert np.all(w_big == 1.0) and mu_big.iloc[0] == pytest.approx(df.pred.mean())


def test_inverse_variance_pool_beats_mean_when_spread_tracks_error():
    df = _windows()
    mean = wc.pool(df)["abs_err"].mean()
    a1 = wc.pool(df, wc.inverse_variance_weights(df.tta_var, 1.0), "tta_mean")["abs_err"].mean()
    assert a1 < mean


def test_weight_error_correlation_is_negative_within_recordings():
    df = _windows()
    df["w"] = wc.inverse_variance_weights(df.tta_var)
    df["e"] = (df.tta_mean - df.age).abs()
    r = wc.weight_error_correlation(df, "w", "e")
    assert r["rho_within"] < -0.1 and r["frac_neg"] > 0.6
    assert r["err_low_w"] > 0 > r["err_high_w"]
    split = wc.error_variance_split(df)
    assert 0 < split["share_within"] < 1


def test_a3_learns_to_downweight_emg():
    df = _windows(n_rec=80)
    w, theta = wc.a3_weights(df)
    assert np.allclose(df.assign(w=w).groupby(["split", "recording"]).w.sum(), 1.0)
    assert theta["log_emg"] < 0 or theta["log_tta_var"] < 0
    test = df.split == "test"
    a3 = wc.pool(df[test], w[test.to_numpy()])["abs_err"].mean()
    assert a3 < wc.pool(df[test])["abs_err"].mean()


def test_recording_confidence_and_risk_coverage():
    df = _windows(noisy_frac=0.0)
    # a third of the recordings are noisy throughout: large spread and error
    noisy = df.recording.isin([f"r{i}" for i in range(0, 60, 3)])
    df.loc[noisy, "pred"] += np.random.default_rng(1).normal(0, 20, noisy.sum())
    df = wc.add_tta_stats(df)
    w = pd.DataFrame({"w_A2": wc.huber_weights(df)[0], "w_A3": wc.a3_weights(df)[0]})
    conf = wc.recording_confidence(df, w)
    assert set(wc.UNCERTAINTIES) <= set(conf)
    t = wc.confidence_error_table(wc.pool(df), conf)
    sd = t.set_index("measure").loc["u_sd"]
    assert sd["rho"] > 0.15 and sd["mae@50"] < sd["mae@100"]
    assert sd["aurc_oracle"] <= sd["aurc"]
