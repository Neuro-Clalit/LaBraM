# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Cross-validation fine-tuning entry point.
#
#   python -m labram.runs.finetune_cv --config <finetune_cv.json>
#       [--set cross_validation.n_folds=5 cross_validation.fold=2 ...]
#
# Partitions the data pool into K group-disjoint folds (saved as a cv_split.json
# artifact), then trains each fold as its own sub-experiment whose ClearML
# project folder and output directory embed the CV experiment name and the fold
# number. Run every fold in-process (cross_validation.fold=-1) for a quick local
# study, or a single fold (cross_validation.fold=k) when each fold is dispatched
# as its own process / SageMaker job. After an in-process multi-fold run the
# metrics are aggregated across folds and logged. See docs/cross_validation.md.
# ---------------------------------------------------------

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import labram.models.registry  # noqa: F401
import labram.utils as utils
from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.utils_conf import parse_overrides
from labram.data import (
    apply_cv_split,
    build_grouped_folds,
    get_dataset_bundle,
    load_cv_split_dict,
    materialize_fold,
    save_cv_split,
    subject_overlap,
)
from labram.data.cross_validation import GroupedFolds

logger = utils.get_logger(__name__)


# ------------------------------------------------------------------ naming


def cv_experiment_name(config: FinetuneRunConfig) -> str:
    """Base experiment name shared by every fold, e.g. ``finetune_tuab_cv5``.

    Prefers an explicit ``clearml.task_name``; otherwise derives from the
    fine-tune ``output.output_dir`` basename (or the model name), and appends the
    ``cv<K>`` suffix so the CV study is self-describing.
    """
    base = config.clearml.task_name
    if not base and config.output.output_dir:
        base = os.path.basename(os.path.normpath(config.output.output_dir))
    if not base:
        base = getattr(config.model, 'model', None) or 'finetune'
    suffix = f"cv{config.cross_validation.n_folds}"
    return base if base.endswith(suffix) else f"{base}_{suffix}"


def cv_base_dir(config: FinetuneRunConfig) -> str:
    """Base output folder that holds every fold sub-run (``<base>/fold_<k>``)."""
    if config.cross_validation.base_dir:
        return config.cross_validation.base_dir
    if config.output.output_dir:
        parent = os.path.dirname(os.path.normpath(config.output.output_dir))
        return os.path.join(parent or '.', cv_experiment_name(config))
    return cv_experiment_name(config)


def fold_dir_name(k: int) -> str:
    return f"fold_{k}"


def derive_fold_config(config: FinetuneRunConfig, k: int) -> FinetuneRunConfig:
    """Deep-copy ``config`` and specialize it for fold ``k``.

    The fold gets its own output/log dirs under the CV base folder, its
    ``cross_validation.fold`` pinned to ``k``, and — so all folds group together
    in ClearML — a ``<project>/<experiment>`` project folder with a
    ``fold_<k>`` task name and CV tags.
    """
    fold_config = copy.deepcopy(config)
    fold_config.cross_validation.fold = k

    experiment = cv_experiment_name(config)
    base = cv_base_dir(config)
    fold_out = os.path.join(base, fold_dir_name(k))
    fold_config.output.output_dir = fold_out
    fold_config.output.log_dir = os.path.join(fold_out, 'log')

    # Group the fold experiments under a common ClearML project sub-folder named
    # after the CV study, with the fold number as the task name.
    project = config.clearml.project_name or 'LaBraM'
    fold_config.clearml.project_name = f"{project}/{experiment}"
    fold_config.clearml.task_name = fold_dir_name(k)
    tags = list(fold_config.clearml.tags)
    for t in ('cross-validation', experiment, fold_dir_name(k)):
        if t not in tags:
            tags.append(t)
    fold_config.clearml.tags = tags
    return fold_config


# ------------------------------------------------------------------ folds


def prepare_folds(config: FinetuneRunConfig, bundle) -> GroupedFolds:
    """Build the fold partition, or reuse a saved ``cv_split.json`` so that
    separately-dispatched fold jobs all see an identical partition."""
    cv = config.cross_validation
    if cv.split_json:
        logger.info("Reusing CV split from %s", cv.split_json)
        split = load_cv_split_dict(cv.split_json)
        return apply_cv_split(bundle, split, config.data.dataset)
    return build_grouped_folds(bundle, cv, config.data.dataset)


def write_fold_metrics(output_dir: str, k: int, summary: Dict,
                       folds: GroupedFolds) -> Optional[str]:
    """Persist a fold's best metrics to ``<output_dir>/fold_metrics.json`` so the
    aggregation step can collect results without a running ClearML server."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    record = {
        'fold': k,
        'n_folds': folds.n_folds,
        'split_by': folds.split_by,
        'max_accuracy_val': summary.get('max_accuracy'),
        'max_accuracy_test': summary.get('max_accuracy_test'),
        'best_epoch': summary.get('best_epoch'),
        'val': summary.get('best_val_stats', {}),
        'test': summary.get('best_test_stats', {}),
    }
    path = os.path.join(output_dir, 'fold_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    logger.info("Saved fold %d metrics -> %s", k, path)
    return path


# ------------------------------------------------------------------ run


def run_fold(config: FinetuneRunConfig, folds: GroupedFolds, base_bundle,
             k: int) -> Dict:
    """Train a single fold and return its best-metric summary."""
    from labram.runs.run_finetune import main as finetune_main

    fold_config = derive_fold_config(config, k)
    fold_bundle = materialize_fold(folds, k, base_bundle)

    out_dir = fold_config.output.output_dir
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Copy the fold partition into the fold dir so the run (local or remote)
        # can attach it as an artifact, and save the fold's resolved config.
        save_cv_split(folds, out_dir)
        fold_config.save_to(os.path.join(out_dir, 'run_config.yaml'))

    logger.info("===== Cross-validation fold %d/%d =====", k, folds.n_folds - 1)
    summary = finetune_main(fold_config, bundle=fold_bundle) or {}
    write_fold_metrics(out_dir, k, summary, folds)
    return summary


def run_cross_validation(config: FinetuneRunConfig) -> Dict:
    """Orchestrate a CV study: build+save the split, run the requested fold(s),
    and (for an in-process multi-fold run) aggregate metrics across folds."""
    config.cross_validation.enabled = True
    config.cross_validation.validate()

    base_bundle = get_dataset_bundle(config.data.dataset, config.data.data_path)
    folds = prepare_folds(config, base_bundle)

    overlap = subject_overlap(folds)
    if overlap:  # pragma: no cover - impossible by construction, guards regressions
        logger.warning("CV fold group overlap detected (leakage!): %s",
                       {k: len(v) for k, v in overlap.items()})

    base = cv_base_dir(config)
    if utils.is_main_process():
        save_cv_split(folds, base)

    single = config.cross_validation.fold
    fold_indices: List[int] = [single] if single is not None and single >= 0 else list(range(folds.n_folds))
    logger.info("Cross-validation experiment %r: running fold(s) %s of %d",
                cv_experiment_name(config), fold_indices, folds.n_folds)

    results: Dict[int, Dict] = {}
    for k in fold_indices:
        results[k] = run_fold(config, folds, base_bundle, k)

    # Aggregate only when this process trained the whole study; a single-fold
    # job leaves aggregation to `labram.eval.cv_report` after all jobs finish.
    if len(fold_indices) > 1:
        from labram.eval.cv_aggregation import aggregate_cv_dir, log_cv_summary
        summary = aggregate_cv_dir(base)
        if utils.is_main_process():
            log_cv_summary(summary, config, base)
        return {'folds': results, 'summary': summary}
    return {'folds': results}


# ------------------------------------------------------------------ cli


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('LaBraM cross-validation fine-tuning', add_help=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML FinetuneRunConfig file.')
    parser.add_argument('--set', dest='overrides', nargs='*', default=[],
                        metavar='KEY=VALUE')
    return parser.parse_args()


def build_config(cli: argparse.Namespace) -> FinetuneRunConfig:
    overrides = parse_overrides(cli.overrides)
    return FinetuneRunConfig.load_config(cli.config, **overrides)


if __name__ == '__main__':
    cli = parse_cli()
    config = build_config(cli)
    run_cross_validation(config)
