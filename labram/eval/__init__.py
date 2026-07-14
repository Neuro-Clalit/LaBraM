# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Offline evaluation toolkit for trained fine-tune models: load a model + config
# + data split (from a local run directory or a ClearML experiment), run
# inference over a split, aggregate window predictions per EEG case with several
# methods, and analyse/visualize the results. See docs/finetune_evaluation.md
# and notebooks/finetune_evaluation.ipynb.
# ---------------------------------------------------------

from labram.eval.aggregation import (
    DEFAULT_AGG_MODES,
    EntropyAccuracyCurve,
    SelectiveCurve,
    aggregate_result,
    case_window_predictions,
    compare_aggregations,
    comparison_table,
    entropy_accuracy_curve,
    rank_cases,
    report_for_mode,
    selective_accuracy,
)
from labram.eval.inference import (
    PredictionResult,
    collect_predictions,
    ordered_files,
)
from labram.eval.loading import (
    ExperimentAssets,
    build_case_loader,
    build_finetune_model,
    build_model_by_name,
    find_experiment_assets,
    load_checkpoint_weights,
    load_clearml_assets,
    load_data_split,
    load_model_from_assets,
    load_run_config,
)

__all__ = [
    'DEFAULT_AGG_MODES',
    'EntropyAccuracyCurve',
    'ExperimentAssets',
    'PredictionResult',
    'SelectiveCurve',
    'aggregate_result',
    'build_case_loader',
    'build_finetune_model',
    'build_model_by_name',
    'case_window_predictions',
    'collect_predictions',
    'compare_aggregations',
    'comparison_table',
    'entropy_accuracy_curve',
    'find_experiment_assets',
    'load_checkpoint_weights',
    'load_clearml_assets',
    'load_data_split',
    'load_model_from_assets',
    'load_run_config',
    'ordered_files',
    'rank_cases',
    'report_for_mode',
    'selective_accuracy',
]
