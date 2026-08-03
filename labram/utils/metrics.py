# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Downstream metric helpers: classification (binary + multi-class) via pyhealth,
# regression computed locally since pyhealth has no regression metrics.
# ---------------------------------------------------------

import warnings
from typing import Any, Dict, List

import numpy as np
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

from labram.utils.regression_metrics import regression_metrics_fn


def _is_single_class(target: Any) -> bool:
    """True when ``target`` contains fewer than 2 distinct values."""
    return np.unique(np.asarray(target).ravel()).size < 2


def get_metrics(
    output: Any,
    target: Any,
    metrics: List[str],
    is_binary: bool,
    threshold: float = 0.5,
    task: str = "classification",
) -> Dict[str, float]:
    """Metrics for one split. ``task='regression'`` scores a scalar target.

    The default ``task`` keeps every existing classification call unchanged.
    """
    if task == "regression":
        return regression_metrics_fn(output, target, metrics)

    single_class = _is_single_class(target)

    if is_binary:
        if single_class and 'roc_auc' in metrics:
            return {m: 0.0 for m in metrics}
        with warnings.catch_warnings():
            if single_class:
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message=".*single label.*")
                warnings.filterwarnings("ignore", message=".*invalid value.*")
                warnings.filterwarnings("ignore", message=".*classes not in y_true.*")
            return binary_metrics_fn(target, output, metrics=metrics, threshold=threshold)

    with warnings.catch_warnings():
        if single_class:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*single label.*")
            warnings.filterwarnings("ignore", message=".*invalid value.*")
            warnings.filterwarnings("ignore", message=".*classes not in y_true.*")
        return multiclass_metrics_fn(target, output, metrics=metrics)
