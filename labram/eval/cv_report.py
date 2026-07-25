# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# CLI to collect a cross-validation study's per-fold results and report the
# metrics aggregated over all folds (mean ± std), optionally logging the summary
# to ClearML.
#
#   # Aggregate fold results written under a local CV base folder:
#   python -m labram.eval.cv_report --base_dir ./checkpoints/finetune_tuab_cv5
#
#   # Pull fold results from a ClearML project sub-folder and log the summary:
#   python -m labram.eval.cv_report \
#       --clearml_project LaBraM/finetune_tuab_cv5 --log_clearml
# ---------------------------------------------------------

import argparse
import json

import labram.utils as utils
from labram.eval.cv_aggregation import (
    aggregate_fold_metrics,
    collect_fold_metrics_from_clearml,
    collect_fold_metrics_from_dir,
    format_summary_table,
    log_cv_summary,
    save_cv_summary,
)

logger = utils.get_logger(__name__)


def _print_table(summary: dict, split: str) -> None:
    table = format_summary_table(summary, split)
    if len(table) <= 1:
        return
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    print(f"\n[{split}] metrics over {summary.get('n_folds_collected')} folds:")
    for row in table:
        print("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser('LaBraM cross-validation report')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='CV base folder containing fold_*/fold_metrics.json')
    parser.add_argument('--clearml_project', type=str, default=None,
                        help='ClearML project sub-folder (e.g. LaBraM/finetune_tuab_cv5)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name recorded in the summary.')
    parser.add_argument('--log_clearml', action='store_true',
                        help='Log the aggregated summary to a ClearML cv_summary task.')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to write cv_summary.json (defaults to base_dir).')
    args = parser.parse_args()

    if not args.base_dir and not args.clearml_project:
        parser.error('Provide --base_dir and/or --clearml_project.')

    records = []
    if args.base_dir:
        records = collect_fold_metrics_from_dir(args.base_dir)
    if not records and args.clearml_project:
        records = collect_fold_metrics_from_clearml(args.clearml_project)
    if not records:
        logger.error("No fold results found; nothing to aggregate.")
        return

    summary = aggregate_fold_metrics(records, args.experiment or args.clearml_project or args.base_dir)
    out_dir = args.output or args.base_dir
    if out_dir:
        save_cv_summary(summary, out_dir)

    _print_table(summary, 'val')
    _print_table(summary, 'test')
    print("\n" + json.dumps({k: summary[k] for k in ('experiment', 'folds', 'n_folds_collected')
                             if k in summary}, indent=2))

    if args.log_clearml:
        if args.clearml_project:
            from types import SimpleNamespace
            project, _, experiment = args.clearml_project.rpartition('/')

            class _Cfg:  # minimal shim carrying what log_cv_summary needs
                clearml = SimpleNamespace(enabled=True, project_name=project or 'LaBraM',
                                          output_uri='')
                cross_validation = SimpleNamespace(n_folds=len(records))
                output = SimpleNamespace(output_dir=out_dir)
                model = SimpleNamespace(model=experiment)

            # experiment name is derived from clearml.task_name in cv_experiment_name;
            # set it so the summary lands in the same project sub-folder.
            _Cfg.clearml.task_name = experiment
            log_cv_summary(summary, _Cfg, out_dir or '.')
        else:
            logger.warning("--log_clearml needs --clearml_project to place the summary task.")


if __name__ == '__main__':
    main()
