# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Matplotlib figure builders for evaluation curves (best-effort).
#
# matplotlib is an optional plotting dependency: every builder returns ``None``
# when it is unavailable, so callers degrade to scalar-only logging rather than
# failing a training run.
# ---------------------------------------------------------

from typing import Any, Optional, Sequence

import numpy as np

from labram.utils.logging import get_logger

logger = get_logger(__name__)

_MPL_WARNED = False


def _figure_module() -> Optional[Any]:
    """Import matplotlib with a non-interactive backend, or return ``None``."""
    global _MPL_WARNED
    try:
        import matplotlib
        matplotlib.use('Agg', force=False)
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover - matplotlib absent/broken
        if not _MPL_WARNED:
            logger.warning(
                "matplotlib unavailable; skipping ROC/PR curve figures "
                "(scalar metrics are still logged).")
            _MPL_WARNED = True
        return None


def roc_curve_figure(fpr: np.ndarray, tpr: np.ndarray, auc: Optional[float] = None,
                     title: str = 'ROC curve') -> Optional[Any]:
    plt = _figure_module()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    label = 'ROC' if auc is None else f'ROC (AUC={auc:.3f})'
    ax.plot(fpr, tpr, color='C0', label=label)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def prediction_scatter_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[dict] = None,
    title: str = 'Predicted vs true',
    max_points: int = 5000,
) -> Optional[Any]:
    """Predicted-vs-true scatter for a scalar target, with the identity line.

    The regression counterpart of the confusion matrix / ROC curve. Points below
    the diagonal at high true values and above it at low ones are the visual
    signature of regression to the mean (see ``age_bias_slope``). Large splits are
    subsampled to ``max_points`` to keep the figure light.
    """
    plt = _figure_module()
    if plt is None:
        return None
    pred = np.asarray(predictions, dtype=float).ravel()
    true = np.asarray(targets, dtype=float).ravel()
    if pred.size == 0:
        return None
    if pred.size > max_points:
        idx = np.linspace(0, pred.size - 1, max_points).astype(int)
        pred, true = pred[idx], true[idx]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true, pred, s=6, alpha=0.25, color='C0', edgecolors='none')
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1,
            label='ideal')
    if metrics:
        summary = ', '.join(f'{k}={v:.3f}' for k, v in metrics.items() if v is not None)
        if summary:
            ax.set_title(f'{title}\n{summary}', fontsize=9)
        else:
            ax.set_title(title)
    else:
        ax.set_title(title)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    return fig


def pr_curve_figure(recall: np.ndarray, precision: np.ndarray, ap: Optional[float] = None,
                    title: str = 'Precision-Recall curve') -> Optional[Any]:
    plt = _figure_module()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    label = 'PR' if ap is None else f'PR (AP={ap:.3f})'
    ax.plot(recall, precision, color='C1', label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig


def confusion_matrix_figure(matrix: np.ndarray, labels: Optional[Sequence] = None,
                            title: str = 'Confusion matrix') -> Optional[Any]:
    plt = _figure_module()
    if plt is None:
        return None
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    ticks = list(labels) if labels is not None else list(range(n))
    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(int(matrix[i, j]), 'd'),
                    ha='center', va='center',
                    color='white' if matrix[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig
