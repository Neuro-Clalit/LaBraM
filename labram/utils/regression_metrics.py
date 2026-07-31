# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Metrics for scalar-target downstream tasks (brain-age regression).
# pyhealth only covers classification, so these are computed here.
# ---------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reported by default when a caller does not narrow the list.
DEFAULT_REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson_r")

REGRESSION_METRIC_NAMES = (
    "mae", "rmse", "mse", "r2", "pearson_r", "spearman_r",
    "age_bias_slope", "mae_corrected", "pred_mean", "pred_std",
    "target_mean", "target_std",
)

# Metrics where a smaller value is better, for model selection.
LOWER_IS_BETTER = frozenset({"mae", "rmse", "mse", "mae_corrected"})


@dataclass
class RegressionReport:
    """Evaluation result for a scalar-target split.

    ``scalars`` are the loggable numbers; ``predictions``/``targets`` are kept so
    a caller can draw a predicted-vs-true scatter or a residual plot, which are
    the regression counterparts of a confusion matrix and ROC curve.
    """

    scalars: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None


def _flat(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _sanitize(scalars: Dict[str, float]) -> Dict[str, float]:
    """Replace NaN/inf with 0.0 so a degenerate batch cannot break logging."""
    return {
        k: (0.0 if v is None or not np.isfinite(v) else float(v))
        for k, v in scalars.items()
    }


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so Spearman handles ties correctly."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # Average the ranks within each group of equal values.
    unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x."""
    if len(x) < 2:
        return 0.0
    var = np.var(x)
    if var == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / var)


def regression_metrics_fn(
    y_pred,
    y_true,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Scalar-target metrics, in the target's own units (years, for age).

    Beyond the usual error/correlation measures this reports the brain-age
    diagnostics:

    * ``age_bias_slope`` — least-squares slope of the residual ``(pred - true)``
      against the true age. Age decoders regress to the cohort mean, which shows
      up as a systematically negative slope: old subjects predicted too young and
      young subjects too old. A slope near 0 means the model is not just
      predicting the mean.
    * ``mae_corrected`` — MAE after removing that linear bias, i.e. the honest
      error once regression-to-the-mean is accounted for.
    """
    pred = _flat(y_pred)
    true = _flat(y_true)
    if pred.shape != true.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {pred.shape} vs {true.shape}")

    requested = list(metrics) if metrics else list(DEFAULT_REGRESSION_METRICS)
    unknown = [m for m in requested if m not in REGRESSION_METRIC_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown regression metric(s) {unknown}; expected from {REGRESSION_METRIC_NAMES}")

    if pred.size == 0:
        return {m: 0.0 for m in requested}

    residual = pred - true
    slope = _ols_slope(true, residual)
    intercept = float(np.mean(residual) - slope * np.mean(true))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))

    available = {
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "pearson_r": _correlation(pred, true),
        "spearman_r": _correlation(_rank(pred), _rank(true)),
        "age_bias_slope": slope,
        "mae_corrected": float(np.mean(np.abs(residual - (slope * true + intercept)))),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "target_mean": float(np.mean(true)),
        "target_std": float(np.std(true)),
    }
    return _sanitize({m: available[m] for m in requested})


def regression_report(
    output,
    target,
    metrics: Optional[Sequence[str]] = None,
) -> RegressionReport:
    """Build a :class:`RegressionReport` from predictions and targets."""
    pred = _flat(output)
    true = _flat(target)
    # The report is the "detailed" view, so compute everything available.
    scalars = regression_metrics_fn(pred, true, REGRESSION_METRIC_NAMES)
    if metrics:
        # Keep the requested metrics first but retain the diagnostics.
        ordered = list(metrics) + [m for m in scalars if m not in metrics]
        scalars = {m: scalars[m] for m in ordered if m in scalars}
    return RegressionReport(scalars=scalars, predictions=pred, targets=true)


def denormalize(values, target_stats: Optional[Tuple[float, float]]):
    """Undo the loader's target z-scoring so metrics read in original units."""
    if target_stats is None:
        return values
    mean, std = target_stats
    return np.asarray(values, dtype=float) * std + mean


def best_metric_for(task: str, metrics: Optional[List[str]] = None) -> Tuple[str, str]:
    """``(metric_name, direction)`` used to pick the best epoch.

    Classification selects on accuracy (higher is better); regression selects on
    the first lower-is-better metric it reports, which is MAE in practice.
    """
    if task != "regression":
        return "accuracy", "max"
    for name in (metrics or DEFAULT_REGRESSION_METRICS):
        if name in LOWER_IS_BETTER:
            return name, "min"
    return "mae", "min"
