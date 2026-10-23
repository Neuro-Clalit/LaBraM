# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Overfitting and headroom diagnostics from a fine-tune's per-epoch log.txt.
# ---------------------------------------------------------
"""Learning-curve diagnostics beyond the best score.

From the per-epoch metric ``m`` (MAE: lower is better) of train ``t_e``,
validation ``v_e`` and test ``s_e``:

* gap            ``g_e = v_e - t_e``; its slope over the last ``k`` epochs (OLS).
  A widening gap while ``v`` stalls is overfitting. Training metrics are
  running averages under the run's augmentation/mixup, so ``g`` can be
  negative early on; its *trend* is what matters.
* improvement    ``d_e = v_{e-1} - v_e`` (> 0: still improving) and the train
  counterpart ``d^t_e``; acceleration ``a_e = d_e - d_{e-1}`` (< 0: slowing).
* overfit epochs  epochs with ``d^t_e > 0`` (train improving) and
  ``d_e < -min_delta`` (val worse by more than noise, 0.02 y); the run ends in an
  *overfit streak* when the last epochs are.
* val/test agreement  Spearman correlation of ``v`` and ``s`` over epochs, the
  best-val and best-test epochs, and the selection regret
  ``s_{argmin v} - min s`` (test given away by selecting on val).
* headroom       if the post-warm-up improvements shrink geometrically with
  ratio ``r = median(d_e / d_{e-1})`` over the last ``k`` positive steps,
  ``0 < r < 1``, the remaining val gain is ``d_last * r / (1 - r)``; a ratio
  >= 1 means the curve is not yet bending (still linear: more epochs needed).
  Needs at least 4 improving post-warm-up epochs (3 ratios), else not estimated.
* memorization   relative gap ``g_last / v_last > 0.3``: the model fits its
  training recordings far better than new ones (e.g. a per-recording feature
  acting as a fingerprint), even while val still improves.
* still improving  the last epoch is the best one (by > ``min_delta``).
"""
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def load_log(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path) if line.startswith("{")]


def _slope(y) -> float:
    y = np.asarray(y, float)
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class OverfitReport:
    epochs: int
    best_val_epoch: int
    best_val: float
    last_val: float
    test_at_best_val: float
    best_test: float
    best_test_epoch: int
    selection_regret: float
    val_test_spearman: float
    final_gap: float
    relative_gap: float
    gap_slope_last_k: float
    last_val_improvement: float
    last_val_acceleration: float
    last_train_improvement: float
    overfit_epochs: List[int]
    overfit_streak: int
    still_improving: bool
    headroom_ratio: float
    headroom_estimate: float
    verdict: str
    per_epoch: List[Dict] = field(default_factory=list)


def analyze(log: List[Dict], metric: str = "mae", k: int = 3, warmup_epochs: int = 0,
            min_delta: float = 0.02) -> OverfitReport:
    t = [e.get(f"train_{metric}", float("nan")) for e in log]
    v = [e.get(f"val_{metric}", float("nan")) for e in log]
    s = [e.get(f"test_{metric}", float("nan")) for e in log]
    n = len(log)
    g = [vi - ti for vi, ti in zip(v, t)]
    d = [float("nan")] + [v[i - 1] - v[i] for i in range(1, n)]
    dt = [float("nan")] + [t[i - 1] - t[i] for i in range(1, n)]
    acc = [float("nan"), float("nan")] + [d[i] - d[i - 1] for i in range(2, n)]
    ibv = int(np.nanargmin(v))
    ibs = int(np.nanargmin(s)) if not all(math.isnan(x) for x in s) else -1
    overfit = [i for i in range(1, n) if dt[i] > 0 and d[i] < -min_delta]
    streak = 0
    for i in range(n - 1, 0, -1):
        if i in overfit:
            streak += 1
        else:
            break
    post = [i for i in range(max(1, warmup_epochs), n)]
    pos = [d[i] for i in post if d[i] > 0]
    ratios = [pos[j] / pos[j - 1] for j in range(1, len(pos))][-k:]
    r = float(np.median(ratios)) if ratios else float("nan")
    last_pos = pos[-1] if pos else float("nan")
    if len(pos) < 4:               # >= 3 ratios, else too noisy to extrapolate
        r = float("nan")
    headroom = (last_pos * r / (1 - r)) if 0 < r < 1 else float("nan")
    still = ibv == n - 1 and (n < 2 or d[-1] > min_delta)
    gap_slope = _slope(g[-k:])
    rel_gap = g[-1] / v[-1] if v[-1] else float("nan")
    if streak >= 2 or (gap_slope > 0.05 and not still and ibv < n - k):
        verdict = "overfitting: val got worse while train kept improving"
    elif rel_gap > 0.3:
        verdict = (f"memorizing: train {t[-1]:.2f} vs val {v[-1]:.2f} "
                   f"(gap {100 * rel_gap:.0f}% of val)")
    elif still and len(pos) < 4:
        verdict = "still improving; too few epochs to extrapolate: train longer"
    elif still and (math.isnan(r) or r >= 1):
        verdict = "still improving, curve not bending: train longer"
    elif still:
        verdict = (f"still improving but slowing; ~{headroom:.2f} more val gain expected"
                   if math.isfinite(headroom) else "still improving but slowing")
    elif ibv >= n - k:
        verdict = "plateau near the end: converged for this schedule"
    else:
        verdict = f"peaked at epoch {ibv} and did not recover: stop earlier / regularize"
    per_epoch = [{"epoch": i, "train": t[i], "val": v[i], "test": s[i], "gap": g[i],
                  "val_improvement": d[i], "val_acceleration": acc[i],
                  "train_improvement": dt[i], "lr": log[i].get("train_lr")} for i in range(n)]
    return OverfitReport(
        epochs=n, best_val_epoch=ibv, best_val=v[ibv], last_val=v[-1],
        test_at_best_val=s[ibv], best_test=s[ibs] if ibs >= 0 else float("nan"),
        best_test_epoch=ibs, selection_regret=s[ibv] - s[ibs] if ibs >= 0 else float("nan"),
        val_test_spearman=_spearman(v, s), final_gap=g[-1], relative_gap=rel_gap,
        gap_slope_last_k=gap_slope,
        last_val_improvement=d[-1], last_val_acceleration=acc[-1],
        last_train_improvement=dt[-1], overfit_epochs=overfit, overfit_streak=streak,
        still_improving=still, headroom_ratio=r, headroom_estimate=headroom,
        verdict=verdict, per_epoch=per_epoch)
