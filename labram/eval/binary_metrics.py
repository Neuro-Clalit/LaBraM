"""Threshold-level metrics for binary scores with rare positives.

Used to read the multi-task diagnosis heads (``scripts/eval_multitask_classification.py``,
Table 2.11 of ``notebooks/brain_age_research_summary.ipynb``): beyond the
threshold-free ROC-AUC / PR-AUC, the best F1 over all thresholds, the errors
(confusion counts) at that threshold, the same threshold carried from
validation to test, and bootstrap confidence intervals (with 7-50 positives a
point estimate alone is misleading).
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The score threshold ``t`` (predict positive when ``p >= t``) that
    maximizes F1 over every distinct score; ties go to the higher threshold."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12), 0.0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thr[best])


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    """Confusion counts and the rates derived from them at threshold ``t``."""
    y = np.asarray(y).astype(bool)
    hat = np.asarray(p) >= t
    tp, fp = int((hat & y).sum()), int((hat & ~y).sum())
    fn, tn = int((~hat & y).sum()), int((~hat & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "specificity": spec,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "error_rate": (fp + fn) / len(y), "balanced_error": 1 - (recall + spec) / 2,
            "fdr": fp / (tp + fp) if tp + fp else 0.0}


def score_summary(y: np.ndarray, p: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    """ROC-AUC, PR-AUC and the metrics at the best-F1 threshold of ``(y, p)``
    (``best_*``); with ``threshold`` also at that fixed threshold (``fixed_*``,
    e.g. the validation best-F1 threshold applied to test)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    out = {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
           "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
    out.update({f"best_{k}": v for k, v in threshold_metrics(y, p, best_f1_threshold(y, p)).items()})
    if threshold is not None:
        out.update({f"fixed_{k}": v for k, v in threshold_metrics(y, p, threshold).items()})
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, keys: Sequence[str] = ("roc_auc", "pr_auc", "best_f1"),
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Percentile ``1 - alpha`` intervals of ``score_summary`` metrics over
    stratified bootstrap resamples (positives and negatives resampled
    separately, so every resample keeps both classes). The best-F1 threshold
    is re-chosen in every resample, as it is on the full sample."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        s = score_summary(y[i], p[i])
        for k in keys:
            draws[k].append(s[k])
    out = {}
    for k, v in draws.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(q) for q in np.quantile(v, [alpha / 2, 1 - alpha / 2]))
    return out


def classification_table(preds: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per run x target x split metrics of a per-recording prediction table
    with columns ``run, split, target, y, p`` (``y`` NaN = unknown, dropped).
    Test also gets ``fixed_*``: the same run's validation best-F1 threshold
    applied to test (the honest operating point; ``best_*`` on test picks its
    threshold on the test labels). Targets whose labels hold one class only
    are skipped."""
    rows = []
    known = preds.dropna(subset=["y"])
    for (run, target), g in known.groupby(["run", "target"], sort=False):
        val_t = None
        for split in ("val", "test"):
            s = g[g.split == split]
            y, p = s.y.to_numpy().astype(int), s.p.to_numpy()
            if len(np.unique(y)) < 2:
                continue
            row = {"run": run, "target": target, "split": split,
                   **score_summary(y, p, val_t if split == "test" else None)}
            if n_boot:
                row.update(bootstrap_ci(y, p, n_boot=n_boot, seed=seed))
            if split == "val":
                val_t = row["best_threshold"]
            rows.append(row)
    return pd.DataFrame(rows)
