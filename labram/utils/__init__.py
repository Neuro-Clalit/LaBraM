# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Re-exports for backward compatibility with the pre-split flat utils.py.
# Prefer importing from the focused submodules in new code.
# ---------------------------------------------------------

from labram.utils.channels import (
    build_pretraining_dataset,
    get_channel_indices,
    standard_1020,
)
from labram.utils.checkpoint import (
    auto_load_model,
    create_ds_config,
    load_state_dict,
    save_model,
    save_nan_model,
)
from labram.utils.cli import bool_flag, get_model
from labram.utils.datasets_tuh import (
    TUABLoader,
    TUEVLoader,
    prepare_TUAB_dataset,
    prepare_TUEV_dataset,
)
from labram.utils.distributed import (
    GatherLayer,
    all_gather_batch,
    all_gather_batch_with_grad,
    all_reduce,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    save_on_master,
    setup_for_distributed,
)
from labram.utils.logging import MetricLogger, SmoothedValue, TensorboardLogger
from labram.utils.metrics import get_metrics
from labram.utils.training import (
    NativeScalerWithGradNormCount,
    cosine_scheduler,
    get_grad_norm,
    get_grad_norm_,
)


__all__ = [
    'NativeScalerWithGradNormCount',
    'GatherLayer',
    'MetricLogger',
    'SmoothedValue',
    'TUABLoader',
    'TUEVLoader',
    'TensorboardLogger',
    'all_gather_batch',
    'all_gather_batch_with_grad',
    'all_reduce',
    'auto_load_model',
    'bool_flag',
    'build_pretraining_dataset',
    'cosine_scheduler',
    'create_ds_config',
    'get_channel_indices',
    'get_grad_norm',
    'get_grad_norm_',
    'get_metrics',
    'get_model',
    'get_rank',
    'get_world_size',
    'init_distributed_mode',
    'is_dist_avail_and_initialized',
    'is_main_process',
    'load_state_dict',
    'prepare_TUAB_dataset',
    'prepare_TUEV_dataset',
    'save_model',
    'save_nan_model',
    'save_on_master',
    'setup_for_distributed',
    'standard_1020',
]
