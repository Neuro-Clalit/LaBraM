"""Figures for the multi-task status / diagnosis-group heads
(``notebooks/pathology_prediction.ipynb``): ROC and precision-recall curves per
target with the best-F1 operating point, and best-F1 with its bootstrap
interval per target. Per-recording inputs: ``run, split, target, y, p`` as
written by ``scripts/eval_multitask_classification.py``.

Every figure keeps the house rule of at most two panels side by side
(``age_plots.assert_max_columns``).
"""
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from labram.eval.binary_metrics import best_f1_threshold, threshold_metrics

# Fixed categorical order: a run keeps its colour in every figure.
RUN_COLORS: Dict[str, str] = {"Exp4mt": "#2a78d6", "Exp4mtbal": "#eb6834",
                              "Exp4cls": "#1baf7a", "Exp4clsage": "#eda100"}


def _color(run: str, i: int) -> str:
    fallback = ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    return RUN_COLORS.get(run, fallback[i % len(fallback)])


def roc_pr_figure(preds: pd.DataFrame, target: str, runs: Sequence[str], split: str = "val"):
    """(a) ROC and (b) precision-recall curve of ``target`` on ``split`` for each
    run, with each run's best-F1 point (Eq. (2.4)) marked; chance is the
    diagonal in (a) and the prevalence line in (b), and (b) carries iso-F1
    contours. Returns ``(fig, axes)``; the caller labels the panels."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    roc, pr = axes
    prevalence = None
    for i, run in enumerate(runs):
        s = preds[(preds.run == run) & (preds.split == split) & (preds.target == target)].dropna(subset=["y"])
        if s.empty or s.y.nunique() < 2:
            continue
        y, p = s.y.to_numpy().astype(int), s.p.to_numpy()
        c = _color(run, i)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        best = threshold_metrics(y, p, best_f1_threshold(y, p))
        roc.plot(fpr, tpr, color=c, lw=2, label=f"{run}  AUC {roc_auc_score(y, p):.3f}")
        pr.plot(rec, prec, color=c, lw=2, drawstyle="steps-post",
                label=f"{run}  AP {average_precision_score(y, p):.3f}, F1 {best['f1']:.3f}")
        for ax, xy in ((roc, (1 - best["specificity"], best["recall"])), (pr, (best["recall"], best["precision"]))):
            ax.plot(*xy, "o", ms=12, color=c, mec="black", mew=1.2, zorder=5)
        prevalence = y.mean()
        n_pos, n = int(y.sum()), len(y)
    roc.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="chance")
    roc.set(xlabel="false-positive rate (1 − specificity)", ylabel="true-positive rate (recall)",
            xlim=(0, 1), ylim=(0, 1.01), title="ROC curve")
    if prevalence is not None:
        pr.axhline(prevalence, color="0.6", lw=1, ls="--", label=f"chance (prevalence {prevalence:.3f})")
        r = np.linspace(0.01, 1, 200)
        for f in (0.2, 0.4, 0.6, 0.8):             # iso-F1: P = F R / (2R - F)
            q = f * r / (2 * r - f)
            ok = (q > 0) & (q <= 1) & (2 * r > f)
            pr.plot(r[ok], q[ok], color="0.85", lw=0.8, zorder=0)
            pr.text(1.0, f / (2 - f), f"F1={f}", color="0.55", fontsize=7, ha="right", va="bottom")
        pr.set_title(f"precision-recall curve ({n_pos} positives of {n})")
    pr.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
    for ax in axes:
        ax.legend(loc="lower right" if ax is roc else "lower left", fontsize=8, framealpha=0.9)
    return fig, axes


def f1_interval_figure(table: pd.DataFrame, runs: Sequence[str], split: str = "val", order=None):
    """Best F1 (Eq. (2.4)) per target with its 95 % bootstrap interval (Eq. (2.7)),
    one marker per run: (a) the normal/abnormal status and the frequent groups
    (>= 40 positives), (b) the rare groups. ``table`` is
    ``binary_metrics.classification_table``. Returns ``(fig, axes)``."""
    t = table[(table.split == split) & table.run.isin(runs)]
    order = list(order) if order is not None else list(dict.fromkeys(t.target))
    pos = t.groupby("target").positives.first()
    groups = ([g for g in order if pos.get(g, 0) >= 40], [g for g in order if 0 < pos.get(g, 0) < 40])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    width = 0.8 / max(len(runs), 1)
    for ax, names, title in zip(axes, groups, ("status and frequent groups (≥ 40 positives)",
                                               "rare groups (< 40 positives)")):
        for i, run in enumerate(runs):
            r = t[t.run == run].set_index("target").reindex(names)
            x = np.arange(len(names)) + (i - (len(runs) - 1) / 2) * width
            ok = r.best_f1.notna().to_numpy()
            ax.errorbar(x[ok], r.best_f1[ok], yerr=[(r.best_f1 - r.best_f1_lo)[ok], (r.best_f1_hi - r.best_f1)[ok]],
                        fmt="o", ms=8, color=_color(run, i), mec="white", mew=1.5, capsize=3, lw=1.5, label=run)
        prev = pos.reindex(names) / t.groupby("target").n.first().reindex(names)
        ax.scatter(np.arange(len(names)), 2 * prev / (1 + prev), marker="_", s=300, color="0.5",
                   zorder=4, label="F1 of flagging every recording")
        ax.set_xticks(np.arange(len(names)), [f"{g}\n({int(pos[g])})" for g in names],
                      rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="best F1 (95 % bootstrap interval)", ylim=(0, 1), title=title)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    return fig, axes
