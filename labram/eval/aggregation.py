# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Offline analysis of collected predictions: per-case window aggregation across
# methods, entropy-vs-accuracy relationship, and selective-prediction (reject
# uncertain windows) risk/coverage — built on labram.utils.eval_metrics.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import labram.utils as utils
from labram.eval.inference import PredictionResult

logger = utils.get_logger(__name__)

# Aggregation methods surfaced in the evaluation notebook. ``vote`` and ``max``
# remain available via aggregate_windows but the three below are the ones the
# comparison focuses on: probabilistic mean, robust median, and the
# entropy-(confidence-)weighted pooling that discounts uncertain windows.
DEFAULT_AGG_MODES: Tuple[str, ...] = ('mean', 'median', 'entropy')


def aggregate_result(
    result: PredictionResult,
    mode: str,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool ``result``'s per-window predictions into one score per case.

    Returns ``(scores, targets)`` where ``scores`` is the per-case positive-class
    probability (binary) or logit-like array (multiclass), matching the
    convention of :func:`labram.utils.aggregate_windows`. Requires per-window
    case ids (``result.groups``).
    """
    if result.groups is None:
        raise ValueError(
            "aggregate_result needs per-window case ids; enable them with "
            "enable_window_ids(dataset) before collecting predictions.")
    return utils.aggregate_windows(
        result.scores, result.targets, list(result.groups), mode,
        result.is_binary, threshold)


def report_for_mode(
    result: PredictionResult,
    mode: str,
    threshold: float = 0.5,
) -> utils.ClassificationReport:
    """Classification report for one aggregation ``mode`` (``'none'`` =
    window-level)."""
    if mode == 'none':
        scores, targets = result.scores, result.targets
    else:
        scores, targets = aggregate_result(result, mode, threshold)
    return utils.classification_report(
        scores, targets, result.is_binary, result.nb_classes, threshold)


def compare_aggregations(
    result: PredictionResult,
    modes: Sequence[str] = DEFAULT_AGG_MODES,
    include_window: bool = True,
    threshold: float = 0.5,
) -> Dict[str, utils.ClassificationReport]:
    """Build a ``{label: ClassificationReport}`` map comparing aggregation modes.

    When ``include_window`` (and case ids are present) a ``'window'`` entry gives
    the per-window (un-aggregated) baseline for reference.
    """
    reports: Dict[str, utils.ClassificationReport] = {}
    if include_window:
        reports['window'] = report_for_mode(result, 'none', threshold)
    if result.groups is None:
        logger.warning("No case ids on result; only the window-level report is available.")
        return reports
    for mode in modes:
        reports[mode] = report_for_mode(result, mode, threshold)
    return reports


def comparison_table(reports: Dict[str, utils.ClassificationReport],
                     keys: Sequence[str] = ('accuracy', 'balanced_accuracy',
                                            'f1', 'roc_auc', 'pr_auc',
                                            'sensitivity', 'specificity')):
    """Flatten a comparison map into a list of ``{'method', <metric>...}`` rows
    (a pandas-friendly record list; no hard pandas dependency)."""
    rows = []
    for label, report in reports.items():
        row = {'method': label}
        row.update({k: report.scalars.get(k) for k in keys})
        rows.append(row)
    return rows


@dataclass
class EntropyAccuracyCurve:
    """Accuracy as a function of prediction entropy, binned into quantiles."""

    bin_centers: np.ndarray     # mean entropy within each bin
    accuracy: np.ndarray        # accuracy within each bin
    counts: np.ndarray          # windows per bin
    bin_edges: np.ndarray


def entropy_accuracy_curve(
    result: PredictionResult,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> EntropyAccuracyCurve:
    """Bin windows by prediction entropy and measure accuracy per bin.

    A well-calibrated model is more accurate on low-entropy (confident) windows;
    this curve makes that relationship explicit (Task 3). Bins are equal-width
    over the observed entropy range.
    """
    ent = result.entropy
    correct = _correct_mask(result, threshold)
    if ent.size == 0:
        empty = np.array([])
        return EntropyAccuracyCurve(empty, empty, empty, empty)

    lo, hi = float(ent.min()), float(ent.max())
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(ent, edges[1:-1]), 0, n_bins - 1)

    centers, acc, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        if m.any():
            centers.append(float(ent[m].mean()))
            acc.append(float(correct[m].mean()))
        else:
            centers.append(float((edges[b] + edges[b + 1]) / 2))
            acc.append(np.nan)
    return EntropyAccuracyCurve(
        np.asarray(centers), np.asarray(acc), np.asarray(counts, dtype=int), edges)


@dataclass
class SelectiveCurve:
    """Risk/coverage for entropy-based rejection of uncertain predictions."""

    coverage: np.ndarray        # fraction of windows kept
    accuracy: np.ndarray        # accuracy over the kept (most-confident) windows
    entropy_threshold: np.ndarray  # entropy cut-off at each coverage level


def selective_accuracy(
    result: PredictionResult,
    threshold: float = 0.5,
    n_points: int = 20,
) -> SelectiveCurve:
    """Selective-prediction curve: keep only the most-confident windows.

    Windows are sorted by ascending entropy and, at each coverage level, we keep
    that most-confident fraction and measure accuracy over them. If entropy is a
    useful confidence signal, accuracy rises as coverage drops — quantifying how
    effective the entropy label is at flagging unreliable predictions (Task 4).
    """
    ent = result.entropy
    correct = _correct_mask(result, threshold)
    n = ent.size
    if n == 0:
        empty = np.array([])
        return SelectiveCurve(empty, empty, empty)

    order = np.argsort(ent)
    correct_sorted = correct[order]
    ent_sorted = ent[order]

    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    cov, acc, thr = [], [], []
    for c in coverages:
        k = max(1, int(round(c * n)))
        cov.append(k / n)
        acc.append(float(correct_sorted[:k].mean()))
        thr.append(float(ent_sorted[k - 1]))
    return SelectiveCurve(np.asarray(cov), np.asarray(acc), np.asarray(thr))


def _correct_mask(result: PredictionResult, threshold: float) -> np.ndarray:
    """Boolean per-window correctness at the given decision threshold."""
    y_true = result.targets.astype(int).ravel()
    if result.is_binary:
        y_pred = (result.probs.ravel() >= threshold).astype(int)
    else:
        y_pred = result.probs.argmax(axis=1)
    return (y_pred == y_true)


def case_window_predictions(
    result: PredictionResult,
    case_id,
) -> Dict[str, np.ndarray]:
    """Per-window predictions for a single case, for inspection/visualization.

    Returns ``{'probs', 'entropy', 'target', 'files', 'index'}`` (Task 5).
    """
    if result.groups is None:
        raise ValueError("case_window_predictions needs per-window case ids.")
    idx = np.where(result.groups == case_id)[0]
    ent = result.entropy
    return {
        'index': idx,
        'probs': result.probs[idx],
        'entropy': ent[idx],
        'target': result.targets[idx],
        'files': None if result.files is None else result.files[idx],
    }


def rank_cases(
    result: PredictionResult,
    mode: str = 'mean',
    threshold: float = 0.5,
) -> List[Tuple[object, float, int, float]]:
    """Rank cases by aggregated-prediction correctness/confidence.

    Returns a list of ``(case_id, score, target, margin)`` sorted from most
    confidently-correct to most confidently-wrong, where ``margin`` is the
    signed distance of the aggregated score from the decision boundary in favour
    of the true class. Useful to pick a clearly-correct and a clearly-wrong case
    for the qualitative EEG walkthrough (Task 5).
    """
    scores, targets = aggregate_result(result, mode, threshold)
    order = _case_order(result.groups)
    ranked: List[Tuple[object, float, int, float]] = []
    if result.is_binary:
        s = scores.ravel()
        t = targets.ravel().astype(int)
        for i, cid in enumerate(order):
            # margin > 0 when the score points to the true class.
            margin = (s[i] - threshold) if t[i] == 1 else (threshold - s[i])
            ranked.append((cid, float(s[i]), int(t[i]), float(margin)))
    else:
        from scipy.special import softmax
        probs = softmax(scores, axis=1)
        t = targets.ravel().astype(int)
        for i, cid in enumerate(order):
            p_true = probs[i, t[i]]
            margin = float(p_true - probs[i][np.arange(probs.shape[1]) != t[i]].max())
            ranked.append((cid, float(p_true), int(t[i]), margin))
    ranked.sort(key=lambda r: r[3], reverse=True)
    return ranked


def _case_order(groups: Optional[np.ndarray]) -> List:
    order: List = []
    seen = set()
    for g in (groups if groups is not None else []):
        if g not in seen:
            seen.add(g)
            order.append(g)
    return order
