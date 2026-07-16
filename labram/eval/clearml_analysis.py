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
                out.append(Insight(
                    severity='warning', category='overfitting',
                    message=(f"Validation loss rose {100*(final_val-min_val)/abs(min_val):.0f}% "
                             f"from its minimum {min_val:.4f} (epoch {best[0]:.0f}) "
                             f"to {final_val:.4f} at the end."),
                    recommendation="Overfitting after the best epoch. Add "
                                   "regularisation (drop_path/weight_decay), stop "
                                   "earlier, or use checkpoint-best rather than the "
                                   "final weights.",
                    evidence={'best_epoch': best[0], 'min_val_loss': min_val,
                              'final_val_loss': final_val}))
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


def _check_best_vs_final(snapshot: ExperimentSnapshot) -> List[Insight]:
    val_metric = _epoch_metric(snapshot, 'val')
    if val_metric is None:
        return []
    finite = val_metric.finite()
    if len(finite) < 3:
        return []
    best = val_metric.best('max')
    last_epoch = finite[-1][0]
    if best is None:
        return []
    # Best is well before the end and final is clearly worse than best.
    if best[0] < last_epoch and (best[1] - finite[-1][1]) > 0.01:
        return [Insight(
            severity='info', category='checkpoint-selection',
            message=(f"Best val {val_metric.series} {best[1]:.3f} was at epoch "
                     f"{best[0]:.0f}, not the final epoch {last_epoch:.0f} "
                     f"({finite[-1][1]:.3f})."),
            recommendation="Evaluate/deploy checkpoint-best.pth, and consider "
                           "shortening training to around the best epoch.",
            evidence={'best_epoch': best[0], 'best_value': best[1],
                      'final_value': finite[-1][1]})]
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
    val_metric = _epoch_metric(snapshot, 'val')
    if val_metric is None:
        return []
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
    grad = snapshot.get('opt/grad_norm', 'grad_norm')
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
    ls = snapshot.get('opt/loss_scale', 'loss_scale')
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


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
