"""Pure helpers of labram.eval.age_analysis (no model, no dataset)."""

import numpy as np
import pandas as pd
import pytest

from labram.eval.age_analysis import (
    artifact_pooled, artifact_pooling_sweep, artifact_weights, select_pooling,
    window_artifact_features,
    apply_bias_correction, assign_age_bin, band_powers, error_by_age_bin, fit_bias_correction,
    peak_alpha_frequency, per_recording, pick_examples, regression_summary, summary_table,
    window_number)


def _windows():
    return pd.DataFrame({
        "split": ["val"] * 5, "idx": range(5),
        "recording": ["a", "a", "a", "b", "b"], "window": [3, 4, 5, 0, 1],
        "cohort": ["normal"] * 3 + ["abnormal"] * 2,
        "age": [30.0] * 3 + [70.0] * 2, "pred": [32.0, 34.0, 36.0, 60.0, 64.0]})


def test_window_number():
    assert window_number("train/aaaaaaaq_s004_t000_10.pkl") == 10
    assert window_number("weird") == -1


def test_per_recording_pools_mean_and_median():
    rec = per_recording(_windows()).set_index("recording")
    assert rec.loc["a", "pred"] == pytest.approx(34.0)
    assert rec.loc["a", "err"] == pytest.approx(4.0)
    assert rec.loc["b", "abs_err"] == pytest.approx(8.0)
    assert rec.loc["a", "n_windows"] == 3 and rec.loc["b", "cohort"] == "abnormal"


def test_regression_summary_and_table():
    s = regression_summary(per_recording(_windows()))
    assert s["mae"] == pytest.approx(6.0) and s["median_ae"] == pytest.approx(6.0)
    assert s["mean_err"] == pytest.approx(-2.0)
    t = summary_table({"recording": per_recording(_windows())})
    assert set(t.index.get_level_values("cohort")) == {"all", "normal", "abnormal"}


def test_error_by_age_bin():
    rec = per_recording(_windows())
    t = error_by_age_bin(rec)
    assert list(t.index.astype(str)) == ["30-39", "70-79"]
    assert t.loc["70-79", "mean_err"] == pytest.approx(-8.0)
    assert t.loc["70-79", "age_mean"] == pytest.approx(70.0)
    assert t.loc["70-79", "pred_mean"] == pytest.approx(62.0)
    assert str(assign_age_bin(pd.Series([0.0]))[0]) == "0-9"


def test_bias_corrections_undo_a_linear_shrinkage():
    age = np.linspace(5, 85, 50)
    pred = 0.6 * age + 20.0                       # regression to the mean
    p = fit_bias_correction(pred, age)
    assert p["slope"] == pytest.approx(0.6) and p["intercept"] == pytest.approx(20.0)
    np.testing.assert_allclose(apply_bias_correction(pred, p, "cole"), age, atol=1e-8)
    np.testing.assert_allclose(apply_bias_correction(pred, p, "age_level", age), age, atol=1e-8)
    np.testing.assert_allclose(apply_bias_correction(pred, p, "recalibrate"), age, atol=1e-8)
    with pytest.raises(ValueError):
        apply_bias_correction(pred, p, "age_level")


def test_band_powers_find_alpha():
    fs, t = 200.0, np.arange(2000) / 200.0
    x = np.sin(2 * np.pi * 10.0 * t)[None] + 0.01 * np.random.default_rng(0).standard_normal((1, 2000))
    rel, freqs, psd = band_powers(x, fs)
    assert rel["alpha"][0] > 0.9
    assert peak_alpha_frequency(freqs, psd[0]) == pytest.approx(10.0)


def test_pick_examples():
    rec = per_recording(_windows())
    ex = pick_examples(rec, n=1)
    assert list(ex["kind"]) == ["best", "over-predicted", "under-predicted"]
    assert ex.iloc[1]["recording"] == "a" and ex.iloc[2]["recording"] == "b"


# ------------------------------------------------- artifact-aware pooling
def _emg_windows():
    """Recording ``a`` (age 70) has two EMG windows predicted young; ``b`` is clean;
    ``c`` is artifact throughout."""
    return pd.DataFrame({
        "split": ["val"] * 7 + ["test"] * 2, "idx": [0, 1, 2, 3, 4, 5, 6, 0, 1],
        "recording": ["a", "a", "a", "a", "b", "b", "c", "d", "d"],
        "cohort": ["normal"] * 9, "age": [70.0] * 4 + [30.0] * 2 + [50.0] + [60.0] * 2,
        "pred": [68.0, 70.0, 40.0, 42.0, 31.0, 29.0, 45.0, 58.0, 30.0],
        "emg": [0.01, 0.02, 0.30, 0.40, 0.02, 0.03, 0.50, 0.01, 0.35]})


def test_artifact_weights():
    s = np.array([0.1, 0.2, 0.4])
    np.testing.assert_array_equal(artifact_weights(s, 0.2, "reject"), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(artifact_weights(s, 0.2, "weight"), [16 / 17, 0.5, 1 / 17])
    np.testing.assert_array_equal(artifact_weights(s, np.inf, "weight"), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        artifact_weights(s, 0.2, "vote")


def test_artifact_pooled_rejects_emg_windows():
    win = _emg_windows()
    plain = artifact_pooled(win).set_index("recording")
    assert plain.loc["a", "pred"] == pytest.approx(55.0)            # plain mean
    assert plain.loc["a", "pred"] == pytest.approx(
        per_recording(win).set_index("recording").loc["a", "pred"])
    rec = artifact_pooled(win, threshold=0.1).set_index("recording")
    assert rec.loc["a", "pred"] == pytest.approx(69.0)
    assert rec.loc["a", "frac_kept"] == pytest.approx(0.5)
    assert rec.loc["c", "pred"] == pytest.approx(45.0)              # min_keep keeps it scored
    assert rec.loc["c", "n_kept"] == pytest.approx(1.0)
    assert rec.loc["a", "abs_err"] == pytest.approx(1.0)
    none = artifact_pooled(win, threshold=0.1, min_keep=0).set_index("recording")
    assert np.isnan(none.loc["c", "pred"])


def test_artifact_pooling_sweep_thresholds_from_val():
    sw = artifact_pooling_sweep(_emg_windows(), quantiles=(1.0, 0.5), methods=("reject",))
    allc = sw[sw.cohort == "all"].set_index(["quantile", "split"])
    assert np.isinf(allc.loc[(1.0, "val"), "threshold"])
    thr = allc.loc[(0.5, "val"), "threshold"]
    assert thr == pytest.approx(0.03)                                # median of val emg
    assert allc.loc[(0.5, "test"), "threshold"] == pytest.approx(thr)
    assert allc.loc[(0.5, "val"), "mae"] < allc.loc[(1.0, "val"), "mae"]
    assert allc.loc[(0.5, "test"), "mae"] == pytest.approx(2.0)      # d: 30-y window dropped
    best = select_pooling(sw)
    assert best["method"] == "reject" and best["quantile"] == 0.5
    assert best["mae_test"] == pytest.approx(2.0)


def test_window_artifact_features_flags_high_frequency_power():
    fs, t = 200.0, np.arange(2000) / 200.0
    rng = np.random.default_rng(0)
    clean = np.stack([np.sin(2 * np.pi * 10.0 * t + k) for k in range(4)])
    noisy = clean.copy()
    noisy[0] += 2.0 * np.sin(2 * np.pi * 38.0 * t)                  # focal EMG-like burst
    noisy += 0.01 * rng.standard_normal(noisy.shape)

    class _DS:
        def __getitem__(self, i):
            return ([clean, noisy][i], 0.0)

    win = pd.DataFrame({"split": ["val", "val"], "idx": [0, 1]})
    f = window_artifact_features(_DS(), win, fs).set_index("idx")
    assert f.loc[0, "emg"] < 0.01 and f.loc[1, "emg"] > 0.1
    assert f.loc[1, "emg_max"] > f.loc[1, "emg"]


# ------------------------------------------------------------- age_plots
class _StubRun:
    ch_names = ["FP1", "O1", "O2", "PZ"]

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self._x = rng.standard_normal((n, 4, 2000)) * 10

    def split(self, _name):
        return [(torch_tensor(x), 0.0) for x in self._x]


def torch_tensor(x):
    import torch
    return torch.as_tensor(x, dtype=torch.float32)


def _explorer():
    from labram.eval.age_plots import AgeExplorer
    ages = {"r1": 60.0, "r2": 62.0, "r3": 61.0, "r4": 30.0}
    preds = {"r1": 61.0, "r2": 45.0, "r3": 75.0, "r4": 33.0}
    rows = [{"split": "test", "idx": 2 * k + j, "recording": r, "window": j, "cohort": "normal",
             "age": ages[r], "pred": preds[r] + (j - 0.5)} for k, r in enumerate(ages) for j in (0, 1)]
    from labram.eval.age_analysis import add_errors
    win = add_errors(pd.DataFrame(rows))
    return AgeExplorer(_StubRun(len(rows)), win, per_recording(win))


def test_explorer_find_and_pick_at_age():
    ex = _explorer()
    assert list(ex.find(age=61, kind="under").recording)[:1] == ["r2"]
    assert list(ex.find(age=61, kind="over").recording)[:1] == ["r3"]
    picks = ex.pick_at_age(61)
    assert {k: v.recording for k, v in picks.items()} == {
        "accurate": "r1", "under-predicted": "r2", "over-predicted": "r3"}


def test_explorer_figures_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns
    ex = _explorer()
    fig = ex.compare_at_age(61, channels=None, number="8.1")
    assert_max_columns(fig)
    titles = sorted(a.get_title() for a in fig.axes)
    assert [t[:3] for t in titles] == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    assert fig._suptitle.get_text().startswith("Figure 8.1.")
    plt.close(fig)
    fig = ex.show("r3", reference=False, channels=["O1", "O2"], seconds=(0, 5), number="8.5")
    assert_max_columns(fig)
    assert sorted(a.get_title()[:3] for a in fig.axes) == ["(a)", "(b)", "(c)", "(d)"]
    plt.close(fig)


def test_numbering_helpers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import (
        assert_max_columns, figure_caption, numbered_table, panel_labels)
    fig, ax = plt.subplots(2, 2)
    for a in ax.ravel():
        a.set_title("t")
    ordered = panel_labels(ax)
    assert [a.get_title() for a in ordered] == ["(a) t", "(b) t", "(c) t", "(d) t"]
    assert ordered[1] is ax[0, 1] and ordered[2] is ax[1, 0]
    assert_max_columns(fig)
    figure_caption(fig, "3.1", "x")
    assert fig._suptitle.get_text() == "Figure 3.1. x"
    plt.close(fig)
    fig, _ = plt.subplots(1, 3)
    with pytest.raises(AssertionError):
        assert_max_columns(fig)
    plt.close(fig)
    html = numbered_table(pd.DataFrame({"mae": [1.234]}), "2.1", "summary").to_html()
    assert "Table 2.1. summary" in html and "1.23" in html


def test_plot_eeg_clips_to_spacing():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import channel_indices, plot_eeg
    x = np.zeros((2, 2000))
    x[0, 100] = 1e4                                  # one huge artifact sample
    fig, ax = plt.subplots()
    spacing = plot_eeg(ax, x, ["A", "B"], spacing=50.0, clip=1.0)
    assert spacing == 50.0 and max(line.get_ydata().max() for line in ax.lines[:2]) <= 50.0
    assert channel_indices(["Fp1", "O1"], ["O1", "XX"]) == [1]
    plt.close(fig)
