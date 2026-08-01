# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Detailed classification metrics (confusion matrix, ROC/PR curves, F1,
# sensitivity/specificity) and per-case window aggregation for inference.
# ---------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationReport:
    """Rich evaluation result for one split.

    ``scalars`` are the loggable numeric metrics; ``matrix`` is the confusion
    matrix (rows = true, cols = predicted); ``roc``/``pr`` hold curve points
    ``(x, y)`` for binary problems (``None`` for multiclass or degenerate
    single-class batches). ``labels`` are the class ids in matrix order.
    """

    scalars: Dict[str, float] = field(default_factory=dict)
    matrix: Optional[np.ndarray] = None
    labels: Optional[List[int]] = None
    roc: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (fpr, tpr)
    pr: Optional[Tuple[np.ndarray, np.ndarray]] = None    # (recall, precision)


def _is_single_class(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size < 2


def _sanitize(scalars: Dict[str, float]) -> Dict[str, float]:
    """Replace non-finite metric values (e.g. kappa on a single-class batch)
    with 0.0 so downstream JSON/TensorBoard/ClearML logging stays well-formed."""
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in scalars.items()}


def _binary_report(y_score: np.ndarray, y_true: np.ndarray, threshold: float) -> ClassificationReport:
    y_true = y_true.astype(int).ravel()
    y_score = y_score.astype(float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    scalars = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(sensitivity),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'cm_tn': float(tn), 'cm_fp': float(fp), 'cm_fn': float(fn), 'cm_tp': float(tp),
    }

    roc = pr = None
    if not _is_single_class(y_true):
        scalars['roc_auc'] = float(roc_auc_score(y_true, y_score))
        scalars['pr_auc'] = float(average_precision_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        roc = (fpr, tpr)
        pr = (rec, prec)
    else:
        # AUROC/AUPRC undefined with a single class present; report 0.0 to keep
        # the key set stable for logging (matches the historical get_metrics).
        scalars['roc_auc'] = 0.0
        scalars['pr_auc'] = 0.0

    return ClassificationReport(scalars=_sanitize(scalars), matrix=cm, labels=[0, 1], roc=roc, pr=pr)


def _multiclass_report(logits: np.ndarray, y_true: np.ndarray, nb_classes: int) -> ClassificationReport:
    y_true = y_true.astype(int).ravel()
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits[:, None]
    probs = softmax(logits, axis=1)
    y_pred = probs.argmax(axis=1)
    labels = list(range(nb_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Per-class sensitivity (recall) and specificity from the confusion matrix.
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_sens = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        per_spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)

    scalars = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'sensitivity': float(per_sens.mean()),
        'specificity': float(per_spec.mean()),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
    # Multiclass AUROC (one-vs-rest, macro) requires every class present.
    if np.unique(y_true).size == nb_classes and nb_classes > 1:
        try:
            scalars['roc_auc'] = float(
                roc_auc_score(y_true, probs, multi_class='ovr', average='macro'))
        except ValueError:
            scalars['roc_auc'] = 0.0
    else:
        scalars['roc_auc'] = 0.0

    return ClassificationReport(scalars=_sanitize(scalars), matrix=cm, labels=labels)


def classification_report(
    output: np.ndarray,
    target: np.ndarray,
    is_binary: bool,
    nb_classes: int = 2,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Build a :class:`ClassificationReport` from predictions and targets.

    For binary problems ``output`` holds positive-class probabilities; for
    multiclass it holds raw logits (softmax is applied internally). ``target``
    holds integer class labels (shape ``(N,)`` or ``(N, 1)``).
    """
    if is_binary:
        return _binary_report(np.asarray(output), np.asarray(target), threshold)
    return _multiclass_report(np.asarray(output), np.asarray(target), nb_classes)


# ---------------------------------------------------------------------------
# Per-case window aggregation (inference)
# ---------------------------------------------------------------------------

_AGG_MODES = ('none', 'mean', 'median', 'vote', 'max', 'entropy')
# 'vote' and 'entropy' need class probabilities, which a scalar target has not.
_REGRESSION_AGG_MODES = ('none', 'mean', 'median', 'max')


def prediction_entropy(
    scores: np.ndarray,
    is_binary: bool,
    normalize: bool = True,
) -> np.ndarray:
    """Per-sample Shannon entropy of a probabilistic prediction (in bits).

    ``scores`` are positive-class probabilities ``(N,)``/``(N, 1)`` for binary,
    or per-class probabilities ``(N, C)`` for multiclass (rows are renormalized
    defensively). With ``normalize`` the entropy is divided by ``log2(n_classes)``
    so it lands in ``[0, 1]`` (0 = fully confident, 1 = uniform) and is directly
    comparable across binary and multiclass. Returns a ``(N,)`` array.
    """
    scores = np.asarray(scores, dtype=float)
    if is_binary:
        p = np.clip(scores.ravel(), 1e-12, 1 - 1e-12)
        probs = np.stack([1.0 - p, p], axis=1)
    else:
        if scores.ndim == 1:
            scores = scores[:, None]
        probs = np.clip(scores, 1e-12, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
    h = -(probs * np.log2(probs)).sum(axis=1)
    if normalize:
        n_classes = probs.shape[1]
        if n_classes > 1:
            h = h / np.log2(n_classes)
    return h


def aggregate_windows(
    output: np.ndarray,
    target: np.ndarray,
    groups: Sequence,
    mode: str,
    is_binary: bool,
    threshold: float = 0.5,
    is_regression: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool per-window predictions into one prediction per case (``group``).

    ``groups`` gives each window's case id (e.g. a recording or subject). The
    target of a case is taken from its windows (assumed constant within a case;
    the first is used). Returns aggregated ``(output, target)`` arrays with one
    row per unique case, ordered by first appearance.

    Modes:

    * ``mean`` — average probabilities (binary) / softmax-averaged logits
      (multiclass). The natural choice for probabilistic pooling.
    * ``median`` — per-case median probability, robust to a few outlier windows.
    * ``vote`` — majority vote over per-window hard predictions.
    * ``max``  — max positive probability (binary: case positive if any window
      crosses ``threshold``) / max over per-class scores (multiclass).
    * ``entropy`` — confidence-weighted mean: each window is weighted by
      ``1 - normalized_entropy`` so confident (low-entropy) windows dominate and
      uncertain windows are down-weighted. Falls back to a plain mean when every
      window in the case is maximally uncertain.
    * ``none`` — return inputs unchanged (window-level).

    With ``is_regression`` the prediction is a scalar in the target's units, so
    only the order-statistic modes (``mean``, ``median``, ``max``) apply; ``vote``
    and ``entropy`` are classification-only and are rejected.
    """
    mode = (mode or 'none').lower()
    if mode == 'none':
        return output, target
    if mode not in _AGG_MODES:
        raise ValueError(f"Unknown aggregation mode {mode!r}; expected one of {_AGG_MODES}")
    if is_regression and mode not in _REGRESSION_AGG_MODES:
        raise ValueError(
            f"Aggregation mode {mode!r} is classification-only; for a regression "
            f"target use one of {_REGRESSION_AGG_MODES}")

    output = np.asarray(output)
    target = np.asarray(target)
    groups = list(groups)

    # Preserve first-appearance order of case ids.
    order: List = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            order.append(g)
    idx_by_group = {g: [i for i, gg in enumerate(groups) if gg == g] for g in order}

    # A regression prediction pools exactly like a binary score: one scalar per
    # window, averaged (or median/max) per case, with the case's constant target.
    if is_binary or is_regression:
        scores = output.astype(float).ravel()
        ent = prediction_entropy(scores, is_binary=True) if mode == 'entropy' else None
        out_rows = []
        tgt_rows = []
        for g in order:
            idx = idx_by_group[g]
            s = scores[idx]
            if mode == 'mean':
                agg = float(s.mean())
            elif mode == 'median':
                agg = float(np.median(s))
            elif mode == 'max':
                agg = float(s.max())
            elif mode == 'entropy':
                w = 1.0 - ent[idx]
                agg = float(np.average(s, weights=w) if w.sum() > 0 else s.mean())
            else:  # vote
                agg = float(((s >= threshold).mean() >= 0.5))
            out_rows.append(agg)
            tgt_rows.append(float(np.asarray(target).ravel()[idx][0]))
        return np.array(out_rows).reshape(-1, 1), np.array(tgt_rows).reshape(-1, 1)

    # multiclass: output are logits
    logits = np.asarray(output, dtype=float)
    if logits.ndim == 1:
        logits = logits[:, None]
    probs = softmax(logits, axis=1)
    nb_classes = probs.shape[1]
    ent = prediction_entropy(probs, is_binary=False) if mode == 'entropy' else None
    out_rows = []
    tgt_rows = []
    tgt_flat = np.asarray(target).ravel()
    for g in order:
        idx = idx_by_group[g]
        if mode == 'mean':
            agg = probs[idx].mean(axis=0)
        elif mode == 'median':
            agg = np.median(probs[idx], axis=0)
        elif mode == 'max':
            agg = probs[idx].max(axis=0)
        elif mode == 'entropy':
            w = 1.0 - ent[idx]
            agg = (np.average(probs[idx], axis=0, weights=w)
                   if w.sum() > 0 else probs[idx].mean(axis=0))
        else:  # vote
            votes = probs[idx].argmax(axis=1)
            counts = np.bincount(votes, minlength=nb_classes)
            agg = counts.astype(float) / max(1, counts.sum())
        out_rows.append(agg)
        tgt_rows.append(float(tgt_flat[idx][0]))
    # Return as (N, C) "logit-like" (log of the aggregated probabilities) so the
    # downstream softmax in the report/metrics recovers the aggregated probs.
    agg_probs = np.clip(np.array(out_rows), 1e-12, None)
    return np.log(agg_probs), np.array(tgt_rows).reshape(-1, 1)


def threshold_sweep(
    y_score: np.ndarray,
    y_true: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Binary metrics as a function of decision threshold.

    Returns a dict of equal-length arrays keyed by ``thresholds`` plus
    ``accuracy``, ``balanced_accuracy``, ``f1``, ``precision``, ``recall``
    (= sensitivity) and ``specificity`` — one value per threshold. Handy for
    picking an operating point and for plotting metric-vs-threshold curves.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    thresholds = np.asarray(thresholds, dtype=float)

    keys = ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'specificity')
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0
        out['accuracy'].append((tp + tn) / max(1, len(y_true)))
        out['balanced_accuracy'].append((sens + spec) / 2.0)
        out['f1'].append(f1)
        out['precision'].append(prec)
        out['recall'].append(sens)
        out['specificity'].append(spec)
    result: Dict[str, np.ndarray] = {'thresholds': thresholds}
    result.update({k: np.asarray(v, dtype=float) for k, v in out.items()})
    return result
