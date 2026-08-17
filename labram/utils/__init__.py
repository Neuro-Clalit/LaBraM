# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Re-exports for backward compatibility with the pre-split flat utils.py.
# Prefer importing from the focused submodules in new code.
# ---------------------------------------------------------

from labram.data.eeg_constants import get_channel_indices, standard_1020
from labram.data.pretraining import build_pretraining_dataset
from labram.utils.checkpoint import (
    auto_load_model,
    create_ds_config,
    load_pretrained_weights,
    load_state_dict,
    save_model,
    save_nan_model,
)
from labram.data.tuh_datasets import (
    TUABLoader,
    TUEVLoader,
    TUHLoader,
    prepare_TUAB_dataset,
    prepare_TUEV_dataset,
)
from labram.utils.cli import bool_flag, get_model
from labram.utils.git_info import (
    GIT_INFO_FILENAME,
    apply_git_info_to_task,
    collect_git_info,
    format_git_summary,
    git_info_bytes,
    load_git_info,
)
from labram.utils.distributed import (
    GatherLayer,
    all_gather_batch,
    all_gather_batch_with_grad,
    all_reduce,
    gather_sharded_eval,
    interleave_shards,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    save_on_master,
    setup_for_distributed,
)
from labram.utils.logging import (
    ClearMLLogger,
    MetricLogger,
    MultiWriter,
    SmoothedValue,
    TensorboardLogger,
    configure_logging,
    get_logger,
    relative_components,
    relative_components_if_enabled,
)
from labram.utils.metrics import get_metrics
from labram.utils.eval_metrics import (
    ClassificationReport,
    aggregate_windows,
    classification_report,
    prediction_entropy,
    threshold_sweep,
)
from labram.utils.regression_metrics import (
    RegressionReport,
    best_metric_for,
    denormalize,
    regression_metrics_fn,
    regression_report,
)
from labram.utils.training import (
    NativeScalerWithGradNormCount,
    build_lr_schedule,
    cosine_scheduler,
    get_grad_norm,
    get_grad_norm_,
)
from labram.utils.timing import PhaseTimer, StepTimer, log_timing_stats, timing_stats


__all__ = [
    'NativeScalerWithGradNormCount',
    'ClearMLLogger',
    'GatherLayer',
    'MetricLogger',
    'MultiWriter',
    'SmoothedValue',
    'TUABLoader',
    'TUEVLoader',
    'TUHLoader',
    'TensorboardLogger',
    'all_gather_batch',
    'all_gather_batch_with_grad',
    'all_reduce',
    'auto_load_model',
    'bool_flag',
    'build_pretraining_dataset',
    'configure_logging',
    'build_lr_schedule',
    'cosine_scheduler',
    'create_ds_config',
    'gather_sharded_eval',
    'interleave_shards',
    'get_channel_indices',
    'get_logger',
    'get_grad_norm',
    'get_grad_norm_',
    'get_metrics',
    'ClassificationReport',
    'RegressionReport',
    'aggregate_windows',
    'best_metric_for',
    'classification_report',
    'denormalize',
    'prediction_entropy',
    'regression_metrics_fn',
    'regression_report',
    'threshold_sweep',
    'get_model',
    'get_rank',
    'get_world_size',
    'init_distributed_mode',
    'is_dist_avail_and_initialized',
    'is_main_process',
    'load_pretrained_weights',
    'load_state_dict',
    'relative_components',
    'relative_components_if_enabled',
    'prepare_TUAB_dataset',
    'prepare_TUEV_dataset',
    'save_model',
    'save_nan_model',
    'save_on_master',
    'setup_for_distributed',
    'standard_1020',
    'PhaseTimer',
    'StepTimer',
    'log_timing_stats',
    'timing_stats',
]
