"""Tests for the scalar-target (brain-age) metrics and criterion.

Values are checked against closed-form results on tiny arrays so the assertions
document what each metric means rather than pinning whatever the code returns.
"""
import numpy as np
import pytest
import torch

from labram.configs.loss_config import LossConfig
from labram.losses import build_downstream_criterion, build_regression_criterion
from labram.utils import aggregate_windows, get_metrics
from labram.utils.regression_metrics import (
    best_metric_for,
    denormalize,
    regression_metrics_fn,
    regression_report,
)


# ------------------------------------------------------------------ metrics


def test_error_metrics_match_closed_form():
    true = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 33.0, 37.0])          # residuals +2 -2 +3 -3
    out = regression_metrics_fn(pred, true, ["mae", "mse", "rmse"])

    assert out["mae"] == pytest.approx(2.5)
    assert out["mse"] == pytest.approx((4 + 4 + 9 + 9) / 4)
    assert out["rmse"] == pytest.approx(np.sqrt(6.5))


def test_perfect_prediction_scores_perfectly():
    true = np.array([20.0, 40.0, 60.0])
    out = regression_metrics_fn(true.copy(), true, ["mae", "rmse", "r2", "pearson_r"])

    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["pearson_r"] == pytest.approx(1.0)


def test_predicting_the_mean_gives_zero_r2():
    true = np.array([10.0, 20.0, 30.0, 40.0])
    out = regression_metrics_fn(np.full_like(true, true.mean()), true, ["r2", "mae"])

    assert out["r2"] == pytest.approx(0.0)
    # The MAE floor a trivial model achieves; a real model must beat it.
    assert out["mae"] == pytest.approx(10.0)


def test_r2_is_negative_when_worse_than_the_mean():
    true = np.array([10.0, 20.0, 30.0])
    assert regression_metrics_fn(np.array([30.0, 20.0, 10.0]), true, ["r2"])["r2"] < 0


def test_spearman_captures_monotone_but_nonlinear_agreement():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 4.0, 9.0, 16.0])             # perfectly rank-ordered
    out = regression_metrics_fn(pred, true, ["pearson_r", "spearman_r"])

    assert out["spearman_r"] == pytest.approx(1.0)
    assert out["pearson_r"] < 1.0


def test_spearman_handles_ties():
    out = regression_metrics_fn(
        np.array([5.0, 5.0, 9.0]), np.array([1.0, 1.0, 2.0]), ["spearman_r"])
    assert out["spearman_r"] == pytest.approx(1.0)


def test_age_bias_slope_detects_regression_to_the_mean():
    """An age decoder pulled toward the cohort mean predicts the old too young and
    the young too old, which shows up as a negative residual-vs-true slope."""
    true = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    center = true.mean()
    pred = center + 0.5 * (true - center)              # shrink toward the mean
    out = regression_metrics_fn(pred, true, ["age_bias_slope", "mae", "mae_corrected"])

    assert out["age_bias_slope"] == pytest.approx(-0.5)
    # Removing that purely linear bias explains the error away entirely.
    assert out["mae_corrected"] == pytest.approx(0.0, abs=1e-9)
    assert out["mae"] > out["mae_corrected"]


def test_age_bias_slope_is_zero_for_an_unbiased_model():
    true = np.array([20.0, 40.0, 60.0, 80.0])
    # Residuals chosen orthogonal to the target: cov(true, residual) == 0, so the
    # error is real but carries no systematic age-dependent component.
    pred = true + np.array([2.0, -2.0, -2.0, 2.0])
    out = regression_metrics_fn(pred, true, ["age_bias_slope", "mae", "mae_corrected"])

    assert out["age_bias_slope"] == pytest.approx(0.0, abs=1e-9)
    # With no bias to remove, correcting for it changes nothing.
    assert out["mae_corrected"] == pytest.approx(out["mae"])


def test_metrics_accept_column_vectors_from_the_model_head():
    """The head emits (N, 1); metrics must not care about the trailing axis."""
    true = np.array([[10.0], [20.0], [30.0]])
    pred = np.array([[11.0], [19.0], [31.0]])
    assert regression_metrics_fn(pred, true, ["mae"])["mae"] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics_fn(np.zeros(3), np.zeros(4), ["mae"])


def test_unknown_metric_name_raises():
    with pytest.raises(ValueError, match="Unknown regression metric"):
        regression_metrics_fn(np.zeros(3), np.zeros(3), ["roc_auc"])


def test_degenerate_inputs_stay_finite():
    """A constant target makes R2 and Pearson undefined; logging must not see NaN."""
    out = regression_metrics_fn(np.array([5.0, 5.0]), np.array([5.0, 5.0]), None)
    assert all(np.isfinite(v) for v in out.values())

    empty = regression_metrics_fn(np.array([]), np.array([]), ["mae", "r2"])
    assert empty == {"mae": 0.0, "r2": 0.0}


def test_report_exposes_scalars_and_scatter_payload():
    true = np.array([20.0, 40.0, 60.0])
    report = regression_report(np.array([25.0, 38.0, 55.0]), true, ["mae"])

    assert "mae" in report.scalars
    # The diagnostics come along even when only 'mae' was requested.
    assert "age_bias_slope" in report.scalars
    assert report.predictions is not None and report.targets is not None
    assert np.allclose(report.targets, true)


# ------------------------------------------------------------------ dispatch


def test_get_metrics_routes_regression_by_task():
    true = np.array([10.0, 20.0, 30.0])
    pred = np.array([11.0, 21.0, 31.0])
    out = get_metrics(pred, true, ["mae"], is_binary=True, task="regression")

    assert out["mae"] == pytest.approx(1.0)


def test_get_metrics_still_defaults_to_classification():
    """The default must preserve every pre-existing call site's behaviour."""
    out = get_metrics(
        np.array([0.9, 0.1, 0.8]), np.array([1, 0, 1]), ["accuracy"], is_binary=True)
    assert out["accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize("task, metrics, expected", [
    ("classification", ["accuracy"], ("accuracy", "max")),
    ("regression", ["mae", "r2"], ("mae", "min")),
    ("regression", ["rmse"], ("rmse", "min")),
    # No lower-is-better metric requested -> fall back to MAE.
    ("regression", ["r2"], ("mae", "min")),
])
def test_best_metric_direction(task, metrics, expected):
    assert best_metric_for(task, metrics) == expected


def test_denormalize_inverts_the_loaders_z_scoring():
    stats = (50.0, 10.0)
    ages = np.array([30.0, 50.0, 70.0])
    scaled = (ages - stats[0]) / stats[1]

    assert np.allclose(denormalize(scaled, stats), ages)
    # No stats -> a pass-through, so an unnormalized dataset needs no special case.
    assert denormalize(scaled, None) is scaled


# ------------------------------------------------------------------ aggregation


def test_window_mean_pooling_for_a_constant_per_case_target():
    """Each case's windows share one age; pooling averages the predictions and
    keeps the single true value."""
    pred = np.array([[28.0], [32.0], [61.0], [59.0]])
    true = np.array([[30.0], [30.0], [60.0], [60.0]])
    groups = ["rec_a", "rec_a", "rec_b", "rec_b"]

    case_pred, case_true = aggregate_windows(
        pred, true, groups, "mean", is_binary=False, is_regression=True)

    assert case_pred.ravel().tolist() == [30.0, 60.0]
    assert case_true.ravel().tolist() == [30.0, 60.0]


def test_window_median_pooling_ignores_an_outlier_window():
    pred = np.array([[30.0], [31.0], [900.0]])
    true = np.array([[30.0], [30.0], [30.0]])

    case_pred, _ = aggregate_windows(
        pred, true, ["r", "r", "r"], "median", is_binary=False, is_regression=True)
    assert case_pred.ravel().tolist() == [31.0]


@pytest.mark.parametrize("mode", ["vote", "entropy"])
def test_classification_only_aggregation_modes_are_rejected(mode):
    """'vote' and 'entropy' need class probabilities, which an age does not have."""
    with pytest.raises(ValueError, match="classification-only"):
        aggregate_windows(
            np.array([[30.0]]), np.array([[30.0]]), ["r"], mode,
            is_binary=False, is_regression=True)


# ------------------------------------------------------------------ criterion


@pytest.mark.parametrize("name, expected", [
    ("mse", torch.nn.MSELoss),
    ("l1", torch.nn.L1Loss),
    ("huber", torch.nn.HuberLoss),
])
def test_regression_criterion_selection(name, expected):
    assert isinstance(
        build_regression_criterion(LossConfig(regression_loss=name)), expected)


def test_regression_criterion_defaults_to_huber():
    assert isinstance(build_regression_criterion(), torch.nn.HuberLoss)


def test_unknown_regression_loss_raises():
    with pytest.raises(ValueError, match="Unknown regression_loss"):
        build_regression_criterion(LossConfig(regression_loss="nope"))


def test_downstream_criterion_never_scores_a_regression_with_cross_entropy():
    """The bug this guards: nb_classes == 1 means 'binary' everywhere, so without
    the task the eval path would compute BCE on ages."""
    reg = build_downstream_criterion("regression", 1, LossConfig())
    cls = build_downstream_criterion("classification", 1, LossConfig())

    assert isinstance(reg, torch.nn.HuberLoss)
    assert isinstance(cls, torch.nn.BCEWithLogitsLoss)


def test_huber_delta_is_configurable():
    crit = build_regression_criterion(LossConfig(regression_loss="huber", huber_delta=5.0))
    assert crit.delta == 5.0
