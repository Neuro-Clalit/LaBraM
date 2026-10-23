# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Pull a finished (or running) ClearML experiment down to a local, plain-data
# snapshot -- hyperparameters, full scalar histories, console tail, artifact /
# model list -- and run heuristic analysis over it to surface *concrete*,
# actionable insights (overfitting, under-training, divergence, LR-schedule
# problems, gradient instability, class imbalance, ...).
#
# The point is offline, Claude-friendly analysis: the fetch step is best-effort
# and isolated behind the optional ``clearml`` dependency, while the analysis and
# reporting operate on the pure ``ExperimentSnapshot`` and are fully testable
# without a ClearML server. See docs/clearml_local_analysis.md.
# ---------------------------------------------------------

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import labram.utils as utils

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Plain-data snapshot
# ---------------------------------------------------------------------------

@dataclass
class ScalarSeries:
    """One reported metric curve: ``title/series`` with aligned iters + values."""

    title: str
    series: str
    iterations: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.title}/{self.series}"

    def finite(self) -> List[Tuple[float, float]]:
        """(iteration, value) pairs with a finite value, in order."""
        return [(it, v) for it, v in zip(self.iterations, self.values)
                if v is not None and math.isfinite(v)]

    @property
    def last(self) -> Optional[float]:
        finite = self.finite()
        return finite[-1][1] if finite else None

    @property
    def first(self) -> Optional[float]:
        finite = self.finite()
        return finite[0][1] if finite else None

    def best(self, mode: str = 'min') -> Optional[Tuple[float, float]]:
        """Return ``(iteration, value)`` of the min (``mode='min'``) or max
        (``mode='max'``) finite value, or ``None`` when the series is empty."""
        finite = self.finite()
        if not finite:
            return None
        pick = min if mode == 'min' else max
        return pick(finite, key=lambda pair: pair[1])

    def has_nonfinite(self) -> bool:
        return any(v is None or not math.isfinite(v) for v in self.values)


@dataclass
class ExperimentSnapshot:
    """Everything worth analysing about one ClearML experiment, as plain data."""

    task_id: Optional[str] = None
    task_name: Optional[str] = None
    project_name: Optional[str] = None
    status: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    created: Optional[str] = None
    started: Optional[str] = None
    completed: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    scalars: Dict[str, ScalarSeries] = field(default_factory=dict)
    console_tail: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    source: str = ''

    # -- lookup helpers ----------------------------------------------------
    def find(self, prefix: Optional[str] = None,
             suffix: Optional[str] = None) -> List[ScalarSeries]:
        """Series whose ``title`` starts with ``prefix`` and/or ``series`` (or
        key) ends with ``suffix``. Matching is case-insensitive."""
        out = []
        for s in self.scalars.values():
            if prefix is not None and not s.title.lower().startswith(prefix.lower()):
                continue
            if suffix is not None and not (
                    s.series.lower().endswith(suffix.lower())
                    or s.key.lower().endswith(suffix.lower())):
                continue
            out.append(s)
        return out

    def get(self, *keys: str) -> Optional[ScalarSeries]:
        """First series matching any of ``keys`` exactly (case-insensitive)."""
        lower = {k.lower(): v for k, v in self.scalars.items()}
        for key in keys:
            if key.lower() in lower:
                return lower[key.lower()]
        return None

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict turns ScalarSeries into nested dicts already; keep the mapping.
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentSnapshot':
        scalars = {
            key: ScalarSeries(**val) if not isinstance(val, ScalarSeries) else val
            for key, val in (data.get('scalars') or {}).items()
        }
        kwargs = {k: v for k, v in data.items() if k != 'scalars'}
        return cls(scalars=scalars, **kwargs)

    def save_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load_json(cls, path: str) -> 'ExperimentSnapshot':
        with open(path, 'r', encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class Insight:
    """One concrete, actionable observation about the experiment."""

    severity: str          # 'info' | 'warning' | 'critical'
    category: str          # short slug, e.g. 'overfitting'
    message: str           # human-readable finding
    recommendation: str = ''
    evidence: Dict[str, Any] = field(default_factory=dict)


_SEVERITY_ORDER = {'critical': 0, 'warning': 1, 'info': 2}


# ---------------------------------------------------------------------------
# ClearML fetch (best-effort; requires the optional `clearml` dependency)
# ---------------------------------------------------------------------------

def load_clearml_experiment(
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
    project_name: Optional[str] = None,
    max_console_lines: int = 200,
) -> ExperimentSnapshot:
    """Fetch a ClearML experiment into an :class:`ExperimentSnapshot`.

    Resolve the task by ``task_id`` or by ``(project_name, task_name)``. Every
    field is fetched best-effort: a section that ClearML cannot provide (or that
    a given server/version does not expose) is logged and left empty rather than
    raising, so a partial experiment still yields a usable snapshot.
    """
    from clearml import Task  # optional dependency

    if task_id:
        task = Task.get_task(task_id=task_id)
    else:
        task = Task.get_task(project_name=project_name, task_name=task_name)
    if task is None:
        raise ValueError(
            f"No ClearML task for id={task_id!r} project={project_name!r} "
            f"name={task_name!r}")

    snap = ExperimentSnapshot(source=f'clearml:{getattr(task, "id", "?")}')
    snap.task_id = getattr(task, 'id', None)
    snap.task_name = getattr(task, 'name', None)

    _try(lambda: setattr(snap, 'project_name', task.get_project_name()),
         "project name")
    _try(lambda: setattr(snap, 'status', str(task.get_status())), "status")
    _try(lambda: setattr(snap, 'tags', list(task.get_tags() or [])), "tags")
    _try(lambda: setattr(snap, 'comment', getattr(task, 'comment', None)), "comment")

    def _fetch_times():
        data = getattr(task, 'data', None)
        if data is not None:
            snap.created = _str_or_none(getattr(data, 'created', None))
            snap.started = _str_or_none(getattr(data, 'started', None))
            snap.completed = _str_or_none(getattr(data, 'completed', None))
    _try(_fetch_times, "timestamps")

    def _fetch_params():
        params = task.get_parameters_as_dict()
        if params:
            snap.hyperparameters = _flatten(params)
    _try(_fetch_params, "hyperparameters")

    _try(lambda: snap.scalars.update(_parse_reported_scalars(
        task.get_reported_scalars())), "scalars")

    def _fetch_console():
        lines = task.get_reported_console_output(number_of_reports=max_console_lines)
        if lines:
            snap.console_tail = [str(x) for x in lines][-max_console_lines:]
    _try(_fetch_console, "console output")

    _try(lambda: snap.artifacts.extend(sorted((getattr(task, 'artifacts', {}) or {}).keys())),
         "artifact list")

    def _fetch_models():
        models = getattr(task, 'models', None)
        if models:
            out = models.get('output', []) if hasattr(models, 'get') else []
            snap.models = [getattr(m, 'name', str(m)) for m in out]
    _try(_fetch_models, "model list")

    logger.info("Loaded ClearML snapshot %s: %d scalar series, %d hyperparameters",
                snap.task_id, len(snap.scalars), len(snap.hyperparameters))
    return snap


def _parse_reported_scalars(reported: Any) -> Dict[str, ScalarSeries]:
    """Map ClearML's ``{title: {series: {'x': [...], 'y': [...]}}}`` into
    ``{title/series: ScalarSeries}``."""
    out: Dict[str, ScalarSeries] = {}
    if not reported:
        return out
    for title, series_map in reported.items():
        if not isinstance(series_map, dict):
            continue
        for series, xy in series_map.items():
            xs = list(xy.get('x', []) or []) if isinstance(xy, dict) else []
            ys = list(xy.get('y', []) or []) if isinstance(xy, dict) else []
            ss = ScalarSeries(title=str(title), series=str(series),
                              iterations=[_as_float(v) for v in xs],
                              values=[_as_float(v) for v in ys])
            out[ss.key] = ss
    return out


def _try(fn, what: str) -> None:
    try:
        fn()
    except Exception as exc:  # pragma: no cover - depends on clearml/server
        logger.warning("Could not fetch ClearML %s: %s", what, exc)


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _str_or_none(v: Any) -> Optional[str]:
    return None if v is None else str(v)


def _flatten(d: Any, parent: str = '') -> Dict[str, Any]:
    """Flatten nested hyperparameter sections into ``section/key`` scalars."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{parent}/{k}" if parent else str(k)
            out.update(_flatten(v, key))
    else:
        out[parent] = d
    return out


# ---------------------------------------------------------------------------
# Heuristic analysis (pure; operates on an ExperimentSnapshot)
# ---------------------------------------------------------------------------

def analyze_experiment(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Derive concrete, actionable insights from a snapshot's curves + metadata.

    Ordered most-severe first. Each analyser is defensive: it simply produces no
    insight when the series it needs are absent, so partial snapshots are fine.
    """
    insights: List[Insight] = []
    insights += _check_run_status(snapshot)
    insights += _check_nonfinite(snapshot)
    insights += _check_overfitting(snapshot)
    insights += _check_best_vs_final(snapshot)
    insights += _check_undertraining(snapshot)
    insights += _check_plateau(snapshot)
    insights += _check_lr_schedule(snapshot)
    insights += _check_grad_instability(snapshot)
    insights += _check_loss_scale(snapshot)
    insights += _check_class_imbalance(snapshot)
    insights += _check_val_test_gap(snapshot)
    if _is_regression(snapshot):
        insights += _check_mean_collapse(snapshot)
        insights += _check_calibration(snapshot)
        insights += _check_age_bias(snapshot)
        insights += _check_age_benchmark(snapshot)
        insights += _check_regression_loss(snapshot)
    insights.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))
    return insights


def _epoch_loss(snapshot: ExperimentSnapshot, split: str) -> Optional[ScalarSeries]:
    return snapshot.get(f'{split}/loss')


def _epoch_metric(snapshot: ExperimentSnapshot, split: str) -> Optional[ScalarSeries]:
    """Preferred headline classification metric for a split, if present."""
    for name in ('balanced_accuracy', 'accuracy', 'roc_auc', 'f1'):
        s = snapshot.get(f'{split}/{name}')
        if s is not None and s.finite():
            return s
    return None


# -- task / hyperparameter helpers -------------------------------------------

def _hparam(snapshot: ExperimentSnapshot, name: str) -> Any:
    """Hyperparameter whose flattened key is ``name`` or ends in ``/name``.

    The trainer connects its config under a ``config/`` section
    (``config/loss/huber_delta``), so match on the key's tail."""
    target = name.lower()
    for key, value in snapshot.hyperparameters.items():
        k = key.lower()
        if k == target or k.endswith('/' + target):
            return value
    return None


def _hparam_float(snapshot: ExperimentSnapshot, name: str) -> Optional[float]:
    try:
        value = float(_hparam(snapshot, name))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _hparam_bool(snapshot: ExperimentSnapshot, name: str) -> Optional[bool]:
    # ClearML hands hyperparameters back as strings.
    value = _hparam(snapshot, name)
    if isinstance(value, bool) or value is None:
        return value
    return {'true': True, '1': True, 'false': False, '0': False}.get(
        str(value).strip().lower())


def _is_regression(snapshot: ExperimentSnapshot) -> bool:
    """Scalar-target run (e.g. EEG brain age)? Trust the recorded
    ``model.task``; fall back to the ``{split}_err`` plot regression runs log."""
    task = _hparam(snapshot, 'model/task')
    if task is not None and str(task).strip():
        return str(task).strip().lower() == 'regression'
    return snapshot.get('val_err/mae', 'test_err/mae', 'train_err/mae') is not None


def _is_age_task(snapshot: ExperimentSnapshot) -> bool:
    dataset = str(_hparam(snapshot, 'data/dataset') or '').lower()
    return 'age' in dataset or any('age' in t.lower() for t in snapshot.tags)


def _headline(snapshot: ExperimentSnapshot,
              split: str) -> Optional[Tuple[ScalarSeries, str]]:
    """The metric model selection ranks ``split`` by, with its direction:
    MAE (``'min'``) for regression, a classification rate (``'max'``) otherwise."""
    if _is_regression(snapshot):
        s = snapshot.get(f'{split}_err/mae')
        return (s, 'min') if s is not None and s.finite() else None
    s = _epoch_metric(snapshot, split)
    return (s, 'max') if s is not None else None


def _best_val_point(snapshot: ExperimentSnapshot) -> Optional[Tuple[float, float]]:
    """``(x, value)`` of the best validation headline metric -- the epoch
    ``checkpoint-best.pth`` was saved at."""
    head = _headline(snapshot, 'val')
    return head[0].best(head[1]) if head is not None else None


def _value_at(series: Optional[ScalarSeries], x: float) -> Optional[float]:
    """Value a series logged at x-coordinate ``x`` (val/test/window metrics of
    one epoch share the same x), or ``None``."""
    if series is None:
        return None
    for it, v in series.finite():
        if abs(it - x) < 1e-9:
            return v
    return None


def _at(snapshot: ExperimentSnapshot, key: str, x: float) -> Optional[float]:
    return _value_at(snapshot.get(key), x)


def _epoch_of(snapshot: ExperimentSnapshot, x: float) -> float:
    """Epoch index of an epoch-level point's x-coordinate.

    On the relative step axis (``logging.relative_step_axis``, the default)
    epoch ``e`` of ``E`` is logged at ``round((e + 1) / E * scale)``, so invert
    that; on the absolute axis ``x`` already is the epoch."""
    if not _hparam_bool(snapshot, 'logging/relative_step_axis'):
        return x
    epochs = _hparam_float(snapshot, 'trainer/epochs')
    scale = _hparam_float(snapshot, 'logging/relative_step_scale') or 1000.0
    if not epochs or scale <= 0:
        return x
    return float(round(x * epochs / scale) - 1)


def _mean_predictor_mae(target_std: float) -> float:
    """MAE of always predicting the mean of a Gaussian target: sqrt(2/pi)*sigma.
    The floor any regressor has to beat to have learned anything."""
    return math.sqrt(2.0 / math.pi) * target_std


def _check_run_status(snapshot: ExperimentSnapshot) -> List[Insight]:
    status = (snapshot.status or '').lower()
    if status in ('failed', 'aborted'):
        return [Insight(
            severity='critical', category='run-status',
            message=f"Experiment did not finish cleanly (status={snapshot.status!r}).",
            recommendation="Inspect the console tail for the terminating error "
                           "before trusting any metric.",
            evidence={'status': snapshot.status})]
    return []


def _check_nonfinite(snapshot: ExperimentSnapshot) -> List[Insight]:
    bad = [s.key for s in snapshot.scalars.values() if s.has_nonfinite()]
    if not bad:
        return []
    loss_bad = [k for k in bad if k.lower().endswith('loss')]
    sev = 'critical' if loss_bad else 'warning'
    return [Insight(
        severity=sev, category='divergence',
        message=f"NaN/Inf detected in {len(bad)} metric series"
                + (f", including loss ({', '.join(loss_bad)})" if loss_bad else ''),
        recommendation="Training diverged. Lower the learning rate, add/upgrade "
                       "gradient clipping, or check AMP loss-scaling and input "
                       "normalisation.",
        evidence={'series': bad[:10]})]


def _check_overfitting(snapshot: ExperimentSnapshot) -> List[Insight]:
    train_loss = _epoch_loss(snapshot, 'train')
    val_loss = _epoch_loss(snapshot, 'val')
    out: List[Insight] = []
    if val_loss is not None and val_loss.finite():
        best = val_loss.best('min')
        finite = val_loss.finite()
        if best is not None and len(finite) >= 4:
            min_val = best[1]
            final_val = finite[-1][1]
            # Validation loss climbed meaningfully after its minimum.
            if min_val > 0 and (final_val - min_val) / abs(min_val) > 0.10:
                best_epoch = _epoch_of(snapshot, best[0])
                out.append(Insight(
                    severity='warning', category='overfitting',
                    message=(f"Validation loss rose {100*(final_val-min_val)/abs(min_val):.0f}% "
                             f"from its minimum {min_val:.4f} (epoch {best_epoch:.0f}) "
                             f"to {final_val:.4f} at the end."),
                    recommendation="Overfitting after the best epoch. Add "
                                   "regularisation (drop_path/weight_decay), stop "
                                   "earlier, or use checkpoint-best rather than the "
                                   "final weights.",
                    evidence={'best_epoch': best_epoch, 'min_val_loss': min_val,
                              'final_val_loss': final_val}))
    if _is_regression(snapshot):
        return out + _regression_gap(snapshot)
    # Train/val accuracy gap.
    train_acc = _epoch_metric(snapshot, 'train')
    val_acc = _epoch_metric(snapshot, 'val')
    if (train_acc is not None and val_acc is not None
            and train_acc.last is not None and val_acc.last is not None):
        gap = train_acc.last - val_acc.last
        if gap > 0.15:
            out.append(Insight(
                severity='warning', category='generalization-gap',
                message=(f"Large train/val gap on {train_acc.series}: "
                         f"train={train_acc.last:.3f} vs val={val_acc.last:.3f} "
                         f"(gap {gap:.3f})."),
                recommendation="The model fits train far better than val. Increase "
                               "regularisation or training data, or reduce capacity.",
                evidence={'train': train_acc.last, 'val': val_acc.last, 'gap': gap}))
    return out


def _regression_gap(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Train-vs-val error gap for a scalar target. Compared per window: the
    train MAE is per-window, while ``val_err`` is pooled per recording."""
    train = snapshot.get('train_err/mae')
    val = snapshot.get('val_window_err/mae', 'val_err/mae')
    if train is None or val is None or train.last is None or val.last is None:
        return []
    if train.last <= 0 or val.last / train.last <= 1.3:
        return []
    ratio = val.last / train.last
    return [Insight(
        severity='warning', category='generalization-gap',
        message=(f"Validation MAE {val.last:.2f} is {ratio:.1f}x the train MAE "
                 f"{train.last:.2f} ({val.key} vs {train.key})."),
        recommendation="The model fits the training subjects far better than "
                       "unseen ones. Raise drop_path / weight_decay, lower "
                       "layer_decay so early layers stay closer to pre-training, "
                       "or stop earlier; more subjects (e.g. TUEG ages) help more "
                       "than more epochs.",
        evidence={'train_mae': train.last, 'val_mae': val.last, 'ratio': ratio})]


def _check_best_vs_final(snapshot: ExperimentSnapshot) -> List[Insight]:
    head = _headline(snapshot, 'val')
    if head is None:
        return []
    val_metric, mode = head
    finite = val_metric.finite()
    if len(finite) < 3:
        return []
    best = val_metric.best(mode)
    last_x, last_value = finite[-1]
    if best is None:
        return []
    # Best is well before the end and final is clearly worse than best: an
    # absolute 0.01 on a rate, 2% relative on an error in target units.
    worse_by = last_value - best[1] if mode == 'min' else best[1] - last_value
    tolerance = 0.02 * abs(best[1]) if mode == 'min' else 0.01
    if best[0] < last_x and worse_by > tolerance:
        best_epoch = _epoch_of(snapshot, best[0])
        last_epoch = _epoch_of(snapshot, last_x)
        return [Insight(
            severity='info', category='checkpoint-selection',
            message=(f"Best val {val_metric.series} {best[1]:.3f} was at epoch "
                     f"{best_epoch:.0f}, not the final epoch {last_epoch:.0f} "
                     f"({last_value:.3f})."),
            recommendation="Evaluate/deploy checkpoint-best.pth, and consider "
                           "shortening training to around the best epoch.",
            evidence={'best_epoch': best_epoch, 'best_value': best[1],
                      'final_value': last_value})]
    return []


def _check_undertraining(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Train loss still descending steeply at the end -> train longer."""
    train_loss = _epoch_loss(snapshot, 'train')
    if train_loss is None:
        # Fall back to the iteration-level training loss (head "loss/loss").
        train_loss = snapshot.get('loss/loss')
    if train_loss is None:
        return []
    finite = train_loss.finite()
    if len(finite) < 6:
        return []
    values = [v for _, v in finite]
    if values[0] <= 0 or (values[0] - values[-1]) / abs(values[0]) <= 0.05:
        return []  # negligible overall progress -> not an under-training signal
    # Compare the mean per-step decrease late in training to that early on: if
    # the loss is still dropping at a good fraction of its initial rate, the run
    # very likely stopped before convergence.
    drops = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    third = max(1, len(drops) // 3)
    early_rate = sum(drops[:third]) / third
    late_rate = sum(drops[-third:]) / third
    if early_rate > 0 and late_rate > 0.30 * early_rate:
        return [Insight(
            severity='info', category='undertraining',
            message=("Training loss is still falling near the end at "
                     f"{100*late_rate/early_rate:.0f}% of its initial rate."),
            recommendation="The model likely has not converged -- train for more "
                           "epochs or raise the learning rate.",
            evidence={'early_rate': early_rate, 'late_rate': late_rate})]
    return []


def _check_plateau(snapshot: ExperimentSnapshot) -> List[Insight]:
    head = _headline(snapshot, 'val')
    if head is None:
        return []
    val_metric = head[0]
    finite = val_metric.finite()
    if len(finite) < 8:
        return []
    values = [v for _, v in finite]
    tail = max(3, len(values) // 3)
    late = values[-tail:]
    spread = max(late) - min(late)
    ref = abs(sum(late) / len(late)) or 1.0
    if spread / ref < 0.005:
        return [Insight(
            severity='info', category='plateau',
            message=(f"Validation {val_metric.series} plateaued over the last "
                     f"{tail} epochs (spread {spread:.4f})."),
            recommendation="Extra epochs are unlikely to help. Change the recipe "
                           "(LR schedule, augmentation, capacity) rather than "
                           "training longer.",
            evidence={'tail_epochs': tail, 'spread': spread})]
    return []


def _check_lr_schedule(snapshot: ExperimentSnapshot) -> List[Insight]:
    lr = snapshot.get('opt/lr', 'lr')
    if lr is None:
        return []
    finite = lr.finite()
    if len(finite) < 3:
        return []
    values = [v for _, v in finite]
    peak = max(values)
    if peak <= 0:
        return []
    final = values[-1]
    out: List[Insight] = []
    # Cosine schedule is expected to decay near zero by the end.
    if final > 0.5 * peak:
        out.append(Insight(
            severity='info', category='lr-schedule',
            message=(f"Learning rate ended at {final:.2e}, still "
                     f"{100*final/peak:.0f}% of its peak {peak:.2e}."),
            recommendation="The schedule barely decayed -- verify warmup/total "
                           "epochs so cosine annealing can anneal, or train longer.",
            evidence={'peak_lr': peak, 'final_lr': final}))
    # Peak reached only at the very start with no warmup ramp is fine; flag an
    # LR that collapses to ~0 within the first 20% of steps.
    early_zero_idx = next((i for i, v in enumerate(values) if v <= 1e-8), None)
    if early_zero_idx is not None and early_zero_idx < 0.2 * len(values):
        out.append(Insight(
            severity='warning', category='lr-schedule',
            message="Learning rate collapsed to ~0 within the first 20% of "
                    "training.",
            recommendation="Most of the run trained at ~0 LR. Check warmup steps "
                           "and the schedule length.",
            evidence={'zero_at_index': early_zero_idx, 'n': len(values)}))
    return out


def _check_grad_instability(snapshot: ExperimentSnapshot) -> List[Insight]:
    # grad_norm moved from the "opt" plot to its own "grad" plot; accept both so
    # snapshots from before and after the split are analysed.
    grad = snapshot.get('grad/grad_norm', 'opt/grad_norm', 'grad_norm')
    if grad is None:
        return []
    finite = [v for _, v in grad.finite()]
    if len(finite) < 10:
        return []
    ordered = sorted(finite)
    median = ordered[len(ordered) // 2]
    peak = max(finite)
    if median > 0 and peak > 20 * median:
        return [Insight(
            severity='warning', category='grad-instability',
            message=(f"Gradient-norm spikes: peak {peak:.2f} is "
                     f"{peak/median:.0f}x the median {median:.2f}."),
            recommendation="Add or tighten gradient clipping (clip_grad) and/or "
                           "lower the learning rate to stabilise training.",
            evidence={'median_grad_norm': median, 'peak_grad_norm': peak})]
    return []


def _check_loss_scale(snapshot: ExperimentSnapshot) -> List[Insight]:
    # loss_scale moved from the "opt" plot to its own "scale" plot; accept both.
    ls = snapshot.get('scale/loss_scale', 'opt/loss_scale', 'loss_scale')
    if ls is None:
        return []
    finite = [v for _, v in ls.finite()]
    if len(finite) < 5:
        return []
    if min(finite) < max(finite) / 1000 and max(finite) > 0:
        return [Insight(
            severity='warning', category='amp-loss-scale',
            message=(f"AMP loss scale dropped sharply (from {max(finite):.0f} to "
                     f"{min(finite):.0f})."),
            recommendation="Frequent gradient overflow -- inspect for exploding "
                           "activations, lower the LR, or disable AMP for the "
                           "unstable layers.",
            evidence={'max_loss_scale': max(finite), 'min_loss_scale': min(finite)})]
    return []


def _check_class_imbalance(snapshot: ExperimentSnapshot) -> List[Insight]:
    for split in ('val', 'test'):
        acc = snapshot.get(f'{split}/accuracy')
        bal = snapshot.get(f'{split}/balanced_accuracy')
        if acc is None or bal is None or acc.last is None or bal.last is None:
            continue
        if acc.last - bal.last > 0.10:
            return [Insight(
                severity='warning', category='class-imbalance',
                message=(f"On {split}, accuracy {acc.last:.3f} is well above "
                         f"balanced accuracy {bal.last:.3f}."),
                recommendation="The model likely favours the majority class. Use "
                               "class weights / resampling and track balanced "
                               "accuracy or F1 as the headline metric.",
                evidence={'accuracy': acc.last, 'balanced_accuracy': bal.last,
                          'split': split})]
    return []


def _check_val_test_gap(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Test headline metric at the selected (best-val) epoch vs its val value."""
    head = _headline(snapshot, 'val')
    best = _best_val_point(snapshot)
    if head is None or best is None:
        return []
    series, mode = head
    x, val_value = best
    test_value = _at(snapshot, 'test' + series.key[len('val'):], x)
    if test_value is None:
        return []
    epoch = _epoch_of(snapshot, x)
    evidence = {'epoch': epoch, 'val': val_value, 'test': test_value}
    select_elsewhere = ("Select with cross-validation (docs/cross_validation.md) "
                        "or a val split drawn like test, and compare the val/test "
                        "cohorts before trusting val-based model selection.")
    if mode == 'max':
        if val_value - test_value <= 0.05:
            return []
        return [Insight(
            severity='warning', category='val-test-gap',
            message=(f"At the selected epoch {epoch:.0f}, test {series.series} "
                     f"{test_value:.3f} is {val_value - test_value:.3f} below val "
                     f"{val_value:.3f}."),
            recommendation="Val does not represent test. " + select_elsewhere,
            evidence=evidence)]
    if val_value <= 0 or (test_value - val_value) / val_value <= 0.15:
        return []
    # Raw MAE is not comparable across splits whose ages spread differently:
    # compare each against its own mean-predictor floor ("skill").
    val_std = _at(snapshot, 'val_err/target_std', x)
    test_std = _at(snapshot, 'test_err/target_std', x)
    rel = 100 * (test_value - val_value) / val_value
    if val_std and test_std:
        val_skill = 1 - val_value / _mean_predictor_mae(val_std)
        test_skill = 1 - test_value / _mean_predictor_mae(test_std)
        evidence.update(val_target_std=val_std, test_target_std=test_std,
                        val_skill=val_skill, test_skill=test_skill)
        if val_skill - test_skill < 0.05:
            return [Insight(
                severity='info', category='val-test-gap',
                message=(f"Test MAE {test_value:.2f} is {rel:.0f}% above val "
                         f"{val_value:.2f} at the selected epoch {epoch:.0f}, but "
                         f"test ages are more spread (std {test_std:.1f} vs "
                         f"{val_std:.1f}); skill over the mean predictor is "
                         f"similar (val {val_skill:.2f}, test {test_skill:.2f})."),
                recommendation="Test is a harder cohort, not a generalisation "
                               "failure. Compare splits on skill "
                               "(1 - MAE / mean-predictor MAE), R² or r, not raw MAE.",
                evidence=evidence)]
    return [Insight(
        severity='warning', category='val-test-gap',
        message=(f"Test MAE {test_value:.2f} is {rel:.0f}% above val "
                 f"{val_value:.2f} at the selected epoch {epoch:.0f}."),
        recommendation="The model generalises worse to the test cohort than val "
                       "suggests (TUAB's eval set is a separate cohort). "
                       + select_elsewhere,
        evidence=evidence)]


# -- regression (EEG brain age) ------------------------------------------------
# All read the case-level (recording-pooled) val metrics at the best-val epoch,
# i.e. the model checkpoint-best.pth holds.

# Upper end of published EEG brain-age MAE (7-8 years; docs/age_regression.md).
_AGE_MAE_BENCHMARK = 8.0


def _check_mean_collapse(snapshot: ExperimentSnapshot) -> List[Insight]:
    best = _best_val_point(snapshot)
    if best is None:
        return []
    x, mae = best
    target_std = _at(snapshot, 'val_err/target_std', x)
    if not target_std or target_std <= 0:
        return []
    floor = _mean_predictor_mae(target_std)
    if mae < 0.9 * floor:
        return []
    pred_std = _at(snapshot, 'val_err/pred_std', x)
    spread = (f"; predictions spread only {pred_std:.1f} vs the true "
              f"{target_std:.1f}" if pred_std is not None else '')
    return [Insight(
        severity='critical', category='mean-collapse',
        message=(f"Best val MAE {mae:.2f} is within 10% of the {floor:.2f} MAE of "
                 f"always predicting the mean (target std {target_std:.1f})"
                 f"{spread}."),
        recommendation="The model has learned little beyond the cohort mean. "
                       "Check the target plumbing first (labels joined, "
                       "target_stats z-scoring and de-normalisation), that the "
                       "pre-trained encoder weights loaded, and that the LR is "
                       "not ~0.",
        evidence={'best_val_mae': mae, 'mean_predictor_mae': floor,
                  'pred_std': pred_std, 'target_std': target_std})]


def _check_calibration(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Error a linear recalibration ``y' = a + b * y_hat`` would remove.

    The best linear map of the predictions reaches RMSE ``std_y * sqrt(1 - r²)``
    (population stds, like the logged RMSE), so the gap to the observed RMSE is
    free error. Its slope ``b = r * std_y / std_pred`` says whether predictions
    are compressed (b > 1) or over-dispersed (b < 1); the mean offset is the
    intercept's share."""
    best = _best_val_point(snapshot)
    if best is None:
        return []
    x = best[0]
    r = _at(snapshot, 'val/pearson_r', x)
    rmse = _at(snapshot, 'val_err/rmse', x)
    pred_std = _at(snapshot, 'val_err/pred_std', x)
    target_std = _at(snapshot, 'val_err/target_std', x)
    pred_mean = _at(snapshot, 'val_err/pred_mean', x)
    target_mean = _at(snapshot, 'val_err/target_mean', x)
    if None in (r, rmse, pred_std, target_std, pred_mean, target_mean):
        return []
    if rmse <= 0 or pred_std <= 0 or target_std <= 0:
        return []
    r = max(-1.0, min(1.0, r))
    rmse_linear = target_std * math.sqrt(1 - r * r)
    headroom = 1 - rmse_linear / rmse
    if headroom < 0.05:
        return []
    slope = r * target_std / pred_std
    offset = pred_mean - target_mean
    defects = []
    if slope > 1.1:
        defects.append(f"compressed toward the mean (true-on-predicted slope "
                       f"{slope:.2f} > 1)")
    elif slope < 0.9:
        defects.append(f"over-dispersed (true-on-predicted slope {slope:.2f} < 1)")
    if abs(offset) > 0.1 * target_std:
        defects.append(f"offset by {offset:+.1f} on average")
    what = ' and '.join(defects) or 'miscalibrated'
    return [Insight(
        severity='warning' if headroom >= 0.10 else 'info',
        category='calibration',
        message=(f"Val predictions at the best epoch are {what}. A linear "
                 f"recalibration would cut val RMSE from {rmse:.2f} to "
                 f"≈{rmse_linear:.2f} ({100 * headroom:.0f}%)."),
        recommendation="Fit y' = a + b·ŷ on the val predictions and apply it "
                       "unchanged to test (the ≈ figure is in-sample on val; never "
                       "fit on test). Compression usually means under-fitting -- "
                       "the head starts at init_scale=0.001 and grows its output "
                       "scale slowly -- so also try more epochs or a higher LR.",
        evidence={'pearson_r': r, 'rmse': rmse, 'rmse_recalibrated': rmse_linear,
                  'slope_true_on_pred': slope, 'mean_offset': offset})]


def _check_age_bias(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Brain-age regression to the mean: slope of residual on true age."""
    best = _best_val_point(snapshot)
    if best is None:
        return []
    x, mae = best
    beta = _at(snapshot, 'val/age_bias_slope', x)
    if beta is None or beta >= -0.2:
        return []
    r = _at(snapshot, 'val/pearson_r', x)
    target_std = _at(snapshot, 'val_err/target_std', x)
    mae_corrected = _at(snapshot, 'val_err/mae_corrected', x)
    message = (f"Residuals regress to the mean: age-bias slope β={beta:.2f} "
               f"(0 unbiased, -1 predicting the mean)")
    if target_std:
        message += (f", so a subject {target_std:.0f} y above the mean age is "
                    f"predicted ≈{abs(beta) * target_std:.0f} y too young (and "
                    f"one below, too old)")
    message += '.'
    if r is not None:
        # For predictions calibrated in the least-squares sense beta = r² - 1.
        expected = r * r - 1
        if beta < expected - 0.05:
            message += (f" That is below the r²-1={expected:.2f} a calibrated "
                        f"model with r={r:.2f} shows: the predictions are "
                        f"compressed beyond what the correlation explains "
                        f"(see calibration).")
        else:
            message += (f" That matches r²-1={expected:.2f} for r={r:.2f}: it is "
                        f"the expected consequence of the correlation, and only a "
                        f"higher r removes it.")
    corrected = (f" (val {mae_corrected:.2f} vs raw {mae:.2f})"
                 if mae_corrected is not None else '')
    return [Insight(
        severity='warning' if beta < -0.5 else 'info', category='age-bias',
        message=message,
        recommendation=(f"Report mae_corrected{corrected} only as a diagnostic: it "
                        "uses the true age, so no model reaches it at inference. "
                        "Before using brain-age deltas as a biomarker, apply the "
                        "β-correction fitted on val. To shrink β itself, raise r "
                        "(more training subjects, a stronger encoder)."),
        evidence={'age_bias_slope': beta, 'pearson_r': r,
                  'mae': mae, 'mae_corrected': mae_corrected})]


def _check_age_benchmark(snapshot: ExperimentSnapshot) -> List[Insight]:
    if not _is_age_task(snapshot):
        return []
    best = _best_val_point(snapshot)
    if best is None:
        return []
    x, mae = best
    epoch = _epoch_of(snapshot, x)
    test_mae = _at(snapshot, 'test_err/mae', x)
    window_mae = _at(snapshot, 'val_window_err/mae', x)
    where = f"epoch {epoch:.0f}" + (f"; test {test_mae:.2f}" if test_mae is not None else '')
    pooling = (f" Pooling windows per recording takes val MAE from "
               f"{window_mae:.2f} to {mae:.2f}." if window_mae is not None else '')
    evidence = {'best_val_mae': mae, 'test_mae': test_mae,
                'val_window_mae': window_mae, 'benchmark_mae': _AGE_MAE_BENCHMARK}
    if mae <= _AGE_MAE_BENCHMARK:
        return [Insight(
            severity='info', category='age-benchmark',
            message=(f"Best val MAE {mae:.2f} y ({where}) is within the ~7-8 y of "
                     f"published EEG brain-age models.{pooling}"),
            recommendation="Confirm on test and across CV folds before claiming it.",
            evidence=evidence)]
    return [Insight(
        severity='warning' if mae > 1.25 * _AGE_MAE_BENCHMARK else 'info',
        category='age-benchmark',
        message=(f"Best val MAE {mae:.2f} y ({where}) is "
                 f"{mae - _AGE_MAE_BENCHMARK:.1f} y above the ~7-8 y of published "
                 f"EEG brain-age models.{pooling}"),
        recommendation="Work the calibration / age-bias / loss insights first, "
                       "then add subjects: TUEG carries ~10x TUAB's age labels "
                       "(docs/age_regression.md). Compare recipes by CV, not a "
                       "single val split.",
        evidence=evidence)]


def _check_regression_loss(snapshot: ExperimentSnapshot) -> List[Insight]:
    """Huber on a z-scored target with delta >= 1 is MSE in practice."""
    loss = str(_hparam(snapshot, 'loss/regression_loss') or '').strip().lower()
    delta = _hparam_float(snapshot, 'loss/huber_delta')
    if loss != 'huber' or delta is None or delta < 1.0:
        return []
    best = _best_val_point(snapshot)
    target_std = _at(snapshot, 'val_err/target_std', best[0]) if best else None
    years = f" (≈{delta * target_std:.0f} y of error)" if target_std else ''
    return [Insight(
        severity='info', category='loss-config',
        message=(f"Huber delta={delta:g} is in z-score units{years}, so the loss is "
                 f"quadratic for almost every error: its robustness never engages, "
                 f"and the run fits the conditional mean while model selection "
                 f"ranks by MAE (a median criterion)."),
        recommendation="Try loss.huber_delta≈0.25-0.5 or loss.regression_loss=l1 so "
                       "the objective matches MAE selection; compare on val MAE "
                       "and age_bias_slope.",
        evidence={'regression_loss': loss, 'huber_delta': delta})]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


_SELECTED_EPOCH_COLUMNS = (
    ('val', ('val', 'val_err')),
    ('test', ('test', 'test_err')),
    ('val window', ('val_window', 'val_window_err')),
    ('test window', ('test_window', 'test_window_err')),
)


def _selected_epoch_lines(snapshot: ExperimentSnapshot) -> List[str]:
    """Every val/test metric at the best-val epoch (the one checkpoint-best.pth
    holds), side by side -- the numbers the run should be judged on."""
    head = _headline(snapshot, 'val')
    best = _best_val_point(snapshot)
    if head is None or best is None:
        return []
    series, mode = head
    x = best[0]
    rows: Dict[str, Dict[str, float]] = {}
    for column, titles in _SELECTED_EPOCH_COLUMNS:
        for s in snapshot.scalars.values():
            value = _value_at(s, x) if s.title in titles else None
            if value is not None:
                rows.setdefault(s.series, {})[column] = value
    columns = [c for c, _ in _SELECTED_EPOCH_COLUMNS
               if any(c in row for row in rows.values())]
    direction = 'lower' if mode == 'min' else 'higher'
    lines = ['## Selected epoch', '',
             f"Best val `{series.series}` ({direction} is better): "
             f"{best[1]:.4g} at epoch {_epoch_of(snapshot, x):.0f}.", '',
             '| Metric | ' + ' | '.join(columns) + ' |',
             '| ------ | ' + ' | '.join('-' * len(c) for c in columns) + ' |']
    for name in sorted(rows, key=lambda n: (n != series.series, n)):
        cells = [_fmt(rows[name].get(c)) for c in columns]
        lines.append(f"| {name} | " + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def render_report(snapshot: ExperimentSnapshot,
                  insights: Optional[List[Insight]] = None) -> str:
    """Render a Markdown report of the snapshot + insights for local reading."""
    if insights is None:
        insights = analyze_experiment(snapshot)
    lines: List[str] = []
    title = snapshot.task_name or snapshot.task_id or 'ClearML experiment'
    lines.append(f"# Experiment analysis: {title}")
    lines.append('')
    meta = [
        ('Task id', snapshot.task_id), ('Project', snapshot.project_name),
        ('Status', snapshot.status), ('Tags', ', '.join(snapshot.tags) or None),
        ('Started', snapshot.started), ('Completed', snapshot.completed),
        ('Source', snapshot.source),
    ]
    for label, value in meta:
        if value:
            lines.append(f"- **{label}:** {value}")
    lines.append('')

    # Insights first -- the point of the report.
    lines.append('## Insights')
    lines.append('')
    if not insights:
        lines.append('_No heuristic issues detected._')
    else:
        icon = {'critical': '🔴', 'warning': '🟠', 'info': '🔵'}
        for ins in insights:
            lines.append(f"### {icon.get(ins.severity, '•')} "
                         f"[{ins.severity}] {ins.category}")
            lines.append('')
            lines.append(ins.message)
            if ins.recommendation:
                lines.append('')
                lines.append(f"**Recommendation:** {ins.recommendation}")
            if ins.evidence:
                ev = ', '.join(f"{k}={_fmt(v) if isinstance(v, (int, float)) else v}"
                               for k, v in ins.evidence.items())
                lines.append('')
                lines.append(f"_Evidence: {ev}_")
            lines.append('')

    lines.extend(_selected_epoch_lines(snapshot))

    # Scalar summary.
    if snapshot.scalars:
        lines.append('## Metric summary')
        lines.append('')
        lines.append('| Series | n | first | last | min | max |')
        lines.append('| ------ | - | ----- | ---- | --- | --- |')
        for key in sorted(snapshot.scalars):
            s = snapshot.scalars[key]
            bmin = s.best('min')
            bmax = s.best('max')
            lines.append(
                f"| {key} | {len(s.finite())} | {_fmt(s.first)} | {_fmt(s.last)} "
                f"| {_fmt(bmin[1] if bmin else None)} "
                f"| {_fmt(bmax[1] if bmax else None)} |")
        lines.append('')

    # Hyperparameters.
    if snapshot.hyperparameters:
        lines.append('## Hyperparameters')
        lines.append('')
        for key in sorted(snapshot.hyperparameters):
            lines.append(f"- `{key}` = {snapshot.hyperparameters[key]}")
        lines.append('')

    if snapshot.artifacts or snapshot.models:
        lines.append('## Artifacts & models')
        lines.append('')
        if snapshot.models:
            lines.append(f"- Models: {', '.join(snapshot.models)}")
        if snapshot.artifacts:
            lines.append(f"- Artifacts: {', '.join(snapshot.artifacts)}")
        lines.append('')

    if snapshot.console_tail:
        lines.append('## Console tail')
        lines.append('')
        lines.append('```')
        lines.extend(snapshot.console_tail[-40:])
        lines.append('```')
        lines.append('')

    return '\n'.join(lines)


def save_experiment_report(
    snapshot: ExperimentSnapshot,
    output_dir: str,
    insights: Optional[List[Insight]] = None,
) -> Dict[str, str]:
    """Write ``snapshot.json`` and ``report.md`` under ``output_dir``.

    Returns the map of written paths. The JSON is the full, reload-able snapshot;
    the Markdown is the human/Claude-facing analysis.
    """
    if insights is None:
        insights = analyze_experiment(snapshot)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'snapshot.json')
    report_path = os.path.join(output_dir, 'report.md')
    snapshot.save_json(json_path)
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(render_report(snapshot, insights))
    logger.info("Wrote experiment report to %s", output_dir)
    return {'snapshot': json_path, 'report': report_path}


def load_and_analyze(
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
    project_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[ExperimentSnapshot, List[Insight]]:
    """Fetch a ClearML experiment, analyse it, and (optionally) write a report.

    Convenience one-shot for notebooks / scripts / the CLI.
    """
    snapshot = load_clearml_experiment(
        task_id=task_id, task_name=task_name, project_name=project_name)
    insights = analyze_experiment(snapshot)
    if output_dir:
        save_experiment_report(snapshot, output_dir, insights)
    return snapshot, insights
