# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Collect the per-fold results of a cross-validation study and compute evaluation
# metrics (mean ± std, min/max, per-fold) across all folds, with optional logging
# to ClearML. Fold results are read from local ``fold_metrics.json`` files (the
# CV runner writes one per fold) and/or pulled from the folds' ClearML tasks.
# See docs/cross_validation.md.
# ---------------------------------------------------------

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

import labram.utils as utils

logger = utils.get_logger(__name__)


# ------------------------------------------------------------------ collect


def collect_fold_metrics_from_dir(base_dir: str) -> List[Dict[str, Any]]:
    """Load every ``<base_dir>/fold_*/fold_metrics.json`` (sorted by fold)."""
    records: List[Dict[str, Any]] = []
    pattern = os.path.join(base_dir, 'fold_*', 'fold_metrics.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                records.append(json.load(f))
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            logger.warning("Could not read %s: %s", path, exc)
    records.sort(key=lambda r: r.get('fold', 0))
    logger.info("Collected %d fold metric file(s) from %s", len(records), base_dir)
    return records


def collect_fold_metrics_from_clearml(project_name: str,
                                      task_name_prefix: str = 'fold_') -> List[Dict[str, Any]]:
    """Pull per-fold best metrics from the folds' ClearML tasks.

    Reads each fold task's reported scalars (last value of the ``val``/``test``
    series). Best-effort: returns ``[]`` if ClearML is unavailable or no matching
    tasks are found, so callers can fall back to the local files.
    """
    try:
        from clearml import Task
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("clearml not installed; cannot collect fold metrics from ClearML.")
        return []
    try:
        tasks = Task.get_tasks(project_name=project_name,
                               task_name=f'{task_name_prefix}*') or []
    except Exception as exc:  # pragma: no cover - depends on a live server
        logger.warning("ClearML task query failed for %r: %s", project_name, exc)
        return []

    records: List[Dict[str, Any]] = []
    for task in tasks:
        # Task names are ``fold_<k>`` optionally suffixed with a run timestamp
        # (clearml.append_timestamp), e.g. ``fold_2_1699999999999`` — pull the
        # fold number from the ``fold_<k>`` token, not the trailing token.
        m = re.search(r'fold[_-](\d+)', task.name)
        if not m:
            continue
        fold = int(m.group(1))
        scalars = task.get_reported_scalars() or {}
        val = _last_series_values(scalars.get('val', {}))
        test = _last_series_values(scalars.get('test', {}))
        records.append({
            'fold': fold, 'val': val, 'test': test,
            'max_accuracy_test': test.get('accuracy'),
            'max_accuracy_val': val.get('accuracy'),
        })
    records.sort(key=lambda r: r.get('fold', 0))
    logger.info("Collected %d fold task(s) from ClearML project %r", len(records), project_name)
    return records


def _last_series_values(head: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for series, data in (head or {}).items():
        ys = data.get('y') if isinstance(data, dict) else None
        if ys:
            out[series] = float(ys[-1])
    return out


# ------------------------------------------------------------------ aggregate


def _numeric_keys(records: List[Dict[str, Any]], split: str) -> List[str]:
    keys: List[str] = []
    for rec in records:
        for k, v in (rec.get(split) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in keys:
                keys.append(k)
    return keys


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'),
                'max': float('nan'), 'n': 0}
    return {'mean': float(arr.mean()), 'std': float(arr.std()),
            'min': float(arr.min()), 'max': float(arr.max()), 'n': int(arr.size)}


def aggregate_fold_metrics(records: List[Dict[str, Any]],
                           experiment: Optional[str] = None) -> Dict[str, Any]:
    """Compute mean ± std (and min/max, per-fold values) over the fold records
    for every numeric metric found in the ``val`` and ``test`` sub-dicts."""
    summary: Dict[str, Any] = {
        'experiment': experiment,
        'n_folds_collected': len(records),
        'folds': sorted(r.get('fold') for r in records),
    }
    for split in ('val', 'test'):
        keys = _numeric_keys(records, split)
        agg: Dict[str, Any] = {}
        for key in keys:
            per_fold = [(rec.get(split) or {}).get(key) for rec in records]
            agg[key] = {**_stats(per_fold), 'per_fold': per_fold}
        summary[split] = agg

    # Headline numbers pulled from the runner's explicit best-metric fields.
    for field in ('max_accuracy_test', 'max_accuracy_val'):
        vals = [rec.get(field) for rec in records if rec.get(field) is not None]
        if vals:
            summary[field] = _stats(vals)
    return summary


def aggregate_cv_dir(base_dir: str, experiment: Optional[str] = None) -> Dict[str, Any]:
    """Collect the fold metric files under ``base_dir`` and aggregate them."""
    records = collect_fold_metrics_from_dir(base_dir)
    summary = aggregate_fold_metrics(records, experiment or os.path.basename(os.path.normpath(base_dir)))
    save_cv_summary(summary, base_dir)
    return summary


def save_cv_summary(summary: Dict[str, Any], output_dir: str,
                    filename: str = 'cv_summary.json') -> Optional[str]:
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved CV summary -> %s", path)
    return path


def format_summary_table(summary: Dict[str, Any], split: str = 'test') -> List[List[str]]:
    """A ``[[metric, mean, std, min, max, n], ...]`` table (header first)."""
    rows = [['metric', 'mean', 'std', 'min', 'max', 'n']]
    for key, st in (summary.get(split) or {}).items():
        rows.append([key, f"{st['mean']:.4f}", f"{st['std']:.4f}",
                     f"{st['min']:.4f}", f"{st['max']:.4f}", str(st['n'])])
    return rows


# ------------------------------------------------------------------ clearml


def log_cv_summary(summary: Dict[str, Any], config: Any, base_dir: str) -> Optional[Any]:
    """Log the aggregated CV metrics to ClearML as a parent ``cv_summary`` task.

    Reports each split's ``mean``/``std`` scalars, a per-split metrics table, and
    uploads ``cv_summary.json``. Best-effort and gated on ``clearml.enabled``;
    returns the Task or ``None``.
    """
    clearml_cfg = getattr(config, 'clearml', None)
    if clearml_cfg is None or not clearml_cfg.enabled:
        return None
    try:
        from clearml import Task
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("clearml.enabled but clearml not installed; skipping CV summary logging.")
        return None

    from labram.runs.finetune_cv import cv_experiment_name
    experiment = cv_experiment_name(config)
    project = f"{clearml_cfg.project_name or 'LaBraM'}/{experiment}"
    try:
        task = Task.init(project_name=project, task_name='cv_summary',
                         output_uri=clearml_cfg.output_uri or None,
                         auto_connect_frameworks=False, reuse_last_task_id=False)
    except Exception as exc:  # pragma: no cover - depends on a live server
        logger.warning("Could not init ClearML cv_summary task: %s", exc)
        return None

    task.add_tags(['cross-validation', 'cv-summary', experiment])
    clearml_logger = task.get_logger()
    for split in ('val', 'test'):
        for key, st in (summary.get(split) or {}).items():
            clearml_logger.report_scalar(title=f'cv_{split}_mean', series=key,
                                         value=st['mean'], iteration=0)
            clearml_logger.report_scalar(title=f'cv_{split}_std', series=key,
                                         value=st['std'], iteration=0)
        table = format_summary_table(summary, split)
        if len(table) > 1:
            try:
                clearml_logger.report_table(title='cv_metrics', series=split,
                                            iteration=0, table_plot=table)
            except Exception as exc:  # pragma: no cover
                logger.warning("ClearML report_table failed: %s", exc)

    path = os.path.join(base_dir, 'cv_summary.json')
    if os.path.exists(path):
        try:
            task.upload_artifact(name='cv_summary', artifact_object=path)
        except Exception as exc:  # pragma: no cover
            logger.warning("ClearML cv_summary artifact upload failed: %s", exc)
    logger.info("Logged CV summary to ClearML project %r", project)
    return task
