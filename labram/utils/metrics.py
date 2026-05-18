# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Classification metric helpers (binary + multi-class) via pyhealth.
# ---------------------------------------------------------

from typing import Any, Dict, List

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn


def get_metrics(
    output: Any,
    target: Any,
    metrics: List[str],
    is_binary: bool,
    threshold: float = 0.5,
) -> Dict[str, float]:
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
