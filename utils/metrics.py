# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Classification metric helpers (binary + multi-class) via pyhealth.
# ---------------------------------------------------------

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
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
