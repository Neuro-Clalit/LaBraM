# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Matplotlib figure builders for offline fine-tune evaluation: metric-vs-
# threshold curves, aggregation-method comparison, entropy/accuracy and
# selective-prediction plots, per-threshold confusion matrices, and per-case
# EEG + window-prediction views. Each returns None when matplotlib is absent.
# ---------------------------------------------------------

from typing import Any, Dict, Optional, Sequence

import numpy as np

from labram.utils.eval_metrics import ClassificationReport
from labram.utils.plots import _figure_module, confusion_matrix_figure  # noqa: F401


def metric_vs_threshold_figure(
    sweep: Dict[str, np.ndarray],
    metrics: Sequence[str] = ('accuracy', 'f1', 'precision', 'recall', 'specificity'),
    title: str = 'Metrics vs decision threshold',
) -> Optional[Any]:
    """Plot selected metrics against the decision threshold (from
    :func:`labram.utils.threshold_sweep`)."""
    plt = _figure_module()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    t = sweep['thresholds']
    for m in metrics:
        if m in sweep:
            ax.plot(t, sweep[m], label=m)
    # Mark the threshold that maximises F1.
    if 'f1' in sweep and len(sweep['f1']):
        best = int(np.nanargmax(sweep['f1']))
        ax.axvline(t[best], color='gray', linestyle='--', linewidth=1)
        ax.annotate(f'best F1 @ {t[best]:.2f}', xy=(t[best], sweep['f1'][best]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def aggregation_comparison_figure(
    reports: Dict[str, ClassificationReport],
    metrics: Sequence[str] = ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc'),
    title: str = 'Aggregation methods',
) -> Optional[Any]:
    """Grouped bar chart comparing metrics across aggregation methods."""
    plt = _figure_module()
    if plt is None:
        return None
    methods = list(reports)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2, 4))
    for i, method in enumerate(methods):
        vals = [reports[method].scalars.get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha='right')
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def roc_pr_overlay_figure(
    reports: Dict[str, ClassificationReport],
    title: str = 'ROC / PR by method',
) -> Optional[Any]:
    """Overlay ROC and PR curves for every method that produced them (binary)."""
    plt = _figure_module()
    if plt is None:
        return None
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
    for label, report in reports.items():
        if report.roc is not None:
            fpr, tpr = report.roc
            auc = report.scalars.get('roc_auc')
            ax_roc.plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})' if auc else label)
        if report.pr is not None:
            rec, prec = report.pr
            ap = report.scalars.get('pr_auc')
            ax_pr.plot(rec, prec, label=f'{label} (AP={ap:.3f})' if ap else label)
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax_roc.set_xlabel('False positive rate'); ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('ROC'); ax_roc.legend(fontsize=8, loc='lower right')
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall'); ax_pr.legend(fontsize=8, loc='lower left')
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def entropy_accuracy_figure(curve, title: str = 'Accuracy vs prediction entropy') -> Optional[Any]:
    """Accuracy per entropy bin (line) with the window count per bin (bars)."""
    plt = _figure_module()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax2 = ax.twinx()
    ax2.bar(curve.bin_centers, curve.counts, width=(
        (curve.bin_edges[1] - curve.bin_edges[0]) * 0.8 if len(curve.bin_edges) > 1 else 0.05),
        color='lightgray', alpha=0.6, label='count')
    ax.plot(curve.bin_centers, curve.accuracy, 'o-', color='C0', label='accuracy')
    ax.set_xlabel('Normalized prediction entropy')
    ax.set_ylabel('Accuracy', color='C0')
    ax2.set_ylabel('Window count', color='gray')
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def selective_accuracy_figure(curve, title: str = 'Selective prediction (reject high-entropy)') -> Optional[Any]:
    """Accuracy over the most-confident kept fraction (coverage)."""
    plt = _figure_module()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.coverage, curve.accuracy, 'o-', color='C2')
    ax.set_xlabel('Coverage (fraction of most-confident windows kept)')
    ax.set_ylabel('Accuracy over kept windows')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def confusion_matrices_grid(
    y_score: np.ndarray,
    y_true: np.ndarray,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    labels: Sequence = (0, 1),
    title: str = 'Confusion matrix by threshold',
) -> Optional[Any]:
    """A row of binary confusion matrices, one per decision threshold (Task 2)."""
    plt = _figure_module()
    if plt is None:
        return None
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, thresholds):
        cm = confusion_matrix(y_true, (y_score >= t).astype(int), labels=list(labels))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'thr={t:.2f}')
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def eeg_case_figure(
    signal: np.ndarray,
    ch_names: Optional[Sequence[str]] = None,
    sampling_rate: int = 200,
    title: str = 'EEG',
    max_channels: int = 16,
) -> Optional[Any]:
    """Stacked multi-channel EEG trace for one window/recording (Task 5).

    ``signal`` is ``(channels, time)``. Channels are vertically offset for a
    classic EEG montage look; at most ``max_channels`` are drawn.
    """
    plt = _figure_module()
    if plt is None:
        return None
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[None, :]
    n_ch = min(signal.shape[0], max_channels)
    t = np.arange(signal.shape[1]) / float(sampling_rate)
    # Offset each channel by a multiple of the overall signal scale.
    scale = np.nanstd(signal[:n_ch]) * 4 + 1e-6
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * n_ch)))
    for c in range(n_ch):
        ax.plot(t, signal[c] + c * scale, linewidth=0.6, color='C0')
    ax.set_yticks([c * scale for c in range(n_ch)])
    ax.set_yticklabels([ch_names[c] if ch_names is not None and c < len(ch_names)
                        else f'ch{c}' for c in range(n_ch)], fontsize=7)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def window_prediction_figure(
    probs: np.ndarray,
    target: int,
    threshold: float = 0.5,
    entropy: Optional[np.ndarray] = None,
    title: str = 'Per-window prediction',
) -> Optional[Any]:
    """Per-window positive-class probability across a case, with the target and
    decision threshold marked (Task 5). Bars are coloured by correctness; the
    optional entropy is overlaid as a secondary line."""
    plt = _figure_module()
    if plt is None:
        return None
    probs = np.asarray(probs, dtype=float).ravel()
    x = np.arange(len(probs))
    pred = (probs >= threshold).astype(int)
    correct = (pred == int(target))
    colors = ['C2' if c else 'C3' for c in correct]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(probs) + 2), 4))
    ax.bar(x, probs, color=colors)
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, label=f'threshold={threshold}')
    ax.axhline(float(target), color='black', linestyle=':', linewidth=1, label=f'target={int(target)}')
    if entropy is not None:
        ax2 = ax.twinx()
        ax2.plot(x, np.asarray(entropy).ravel(), 'o-', color='C1', markersize=3, label='entropy')
        ax2.set_ylabel('Entropy', color='C1')
        ax2.set_ylim(0, 1.02)
    ax.set_xlabel('Window index')
    ax.set_ylabel('P(positive)')
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig
