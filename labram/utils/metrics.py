# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Downstream metric helpers: classification (binary + multi-class) via pyhealth,
# regression computed locally since pyhealth has no regression metrics.
# ---------------------------------------------------------

from typing import Any, Dict, List

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

from labram.utils.regression_metrics import regression_metrics_fn


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
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:
            return binary_metrics_fn(target, output, metrics=metrics, threshold=threshold)
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }
    return multiclass_metrics_fn(target, output, metrics=metrics)
