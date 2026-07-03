# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Helpers shared across the three runner scripts
# (run_class_finetuning, run_labram_pretraining, run_vqnsp_training).
#
# Each runner still owns its own argparse, model factory call, dataset
# preparation, and per-epoch logging shape. What lives here is the
# bookkeeping every runner duplicated: distributed init, device/seed,
# tensorboard wiring, list-of-dataloader construction, DDP wrap,
# auto-resume hook, train-log line, and the cosine schedules.
# --------------------------------------------------------

import datetime
import json
import os
import time
from argparse import Namespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import labram.utils as utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import ClearMLConfig, DistributedConfig, OutputConfig, TrainerConfig

logger = utils.get_logger(__name__)


def setup_environment(config, init_cudnn_benchmark: bool = True) -> Tuple[torch.device, int, int]:
    """Initialize distributed, resolve device, seed, and cudnn flags.

    Accepts either a full RunConfig or any object with a ``.distributed``
    attribute of type DistributedConfig.  The runtime-computed ``distributed``
    and ``gpu`` fields are written back onto ``config.distributed`` so callers
    can access them without a separate return value.

    Returns (device, num_tasks, global_rank).
    """
    dist_cfg = config.distributed
    # Bridge to the legacy init_distributed_mode which mutates a Namespace.
    _ns = Namespace(
        dist_on_itp=dist_cfg.dist_on_itp,
        dist_url=dist_cfg.dist_url,
        world_size=dist_cfg.world_size,
        local_rank=dist_cfg.local_rank,
    )
    utils.init_distributed_mode(_ns)
    dist_cfg.distributed = getattr(_ns, 'distributed', False)
    dist_cfg.gpu = getattr(_ns, 'gpu', 0)

    device_str = getattr(_ns, 'device', dist_cfg.device)
    if device_str == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    device = torch.device(device_str)
    dist_cfg.device = str(device)

    seed = dist_cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if init_cudnn_benchmark and torch.cuda.is_available():
        cudnn.benchmark = True

    _configure_run_logging(config)

    return device, utils.get_world_size(), utils.get_rank()


def _configure_run_logging(config) -> None:
    """Attach the rank-aware ``labram`` logger. On rank 0 also tee to a file
    (``run.log``) under the run's log_dir/output_dir when either is set."""
    rank = utils.get_rank()
    log_file = None
    output_cfg = getattr(config, 'output', None)
    if rank == 0 and output_cfg is not None:
        base_dir = output_cfg.log_dir or output_cfg.output_dir
        if base_dir:
            log_file = os.path.join(base_dir, 'run.log')
    utils.configure_logging(rank=rank, log_file=log_file)


def _derive_clearml_task_name(run_config: Any) -> str:
    """Pick a stable, human-readable ClearML task name from the run config."""
    output_cfg = getattr(run_config, 'output', None)
    if output_cfg is not None and output_cfg.output_dir:
        name = os.path.basename(os.path.normpath(output_cfg.output_dir))
        if name:
            return name
    model_cfg = getattr(run_config, 'model', None)
    if model_cfg is not None and getattr(model_cfg, 'model', None):
        return str(model_cfg.model)
    return 'labram-run'


def init_clearml_task(
    clearml_cfg: ClearMLConfig,
    run_config: Any,
    global_rank: int,
) -> Optional[Any]:
    """Initialize a ClearML ``Task`` on rank 0 when tracking is enabled.

    Returns the ``Task`` (or ``None`` when disabled, off-rank, or ClearML is not
    installed). ClearML is an optional dependency: a missing package downgrades
    to a warning and TensorBoard-only logging rather than failing the run.
    """
    if not clearml_cfg.enabled or global_rank != 0:
        return None
    try:
        from clearml import Task
    except ImportError:
        logger.warning(
            "clearml.enabled is set but the `clearml` package is not installed; "
            "continuing with TensorBoard-only logging. Run `pip install clearml` "
            "to enable ClearML experiment tracking.")
        return None

    if clearml_cfg.offline:
        Task.set_offline(offline_mode=True)

    task_name = clearml_cfg.task_name or _derive_clearml_task_name(run_config)
    task = Task.init(
        project_name=clearml_cfg.project_name or 'LaBraM',
        task_name=task_name,
        output_uri=clearml_cfg.output_uri or None,
        continue_last_task=clearml_cfg.continue_last_task,
        auto_connect_frameworks=clearml_cfg.auto_connect_frameworks,
    )
    if clearml_cfg.tags:
        task.add_tags(list(clearml_cfg.tags))
    if run_config is not None and hasattr(run_config, 'as_dict'):
        try:
            task.connect_configuration(run_config.as_dict(), name='run_config')
        except Exception as exc:  # pragma: no cover - defensive: never fail a run on tracking
            logger.warning("ClearML connect_configuration failed: %s", exc)
    logger.info("ClearML tracking enabled: project=%r task=%r", clearml_cfg.project_name, task_name)
    return task


def create_log_writer(
    output_cfg: OutputConfig,
    global_rank: int,
    clearml_cfg: Optional[ClearMLConfig] = None,
    run_config: Any = None,
) -> Optional[Any]:
    """Build the rank-0 metric writer: TensorBoard, ClearML, or both.

    Returns ``None`` on non-rank-0 processes or when no sink is configured. A
    single sink is returned bare; multiple sinks are combined in a
    :class:`~labram.utils.MultiWriter` so callers keep one ``log_writer``.
    """
    if global_rank != 0:
        return None

    writers: List[Any] = []
    if output_cfg.log_dir:
        os.makedirs(output_cfg.log_dir, exist_ok=True)
        writers.append(utils.TensorboardLogger(log_dir=output_cfg.log_dir))

    if clearml_cfg is not None and clearml_cfg.enabled:
        task = init_clearml_task(clearml_cfg, run_config, global_rank)
        if task is not None:
            writers.append(utils.ClearMLLogger(task=task))

    if not writers:
        return None
    if len(writers) == 1:
        return writers[0]
    return utils.MultiWriter(writers)


def build_distributed_train_sampler_list(
    datasets: Sequence[torch.utils.data.Dataset],
    num_tasks: int,
    rank: int,
) -> List[torch.utils.data.DistributedSampler]:
    """One shuffled DistributedSampler per training dataset."""
    return [
        torch.utils.data.DistributedSampler(
            d, num_replicas=num_tasks, rank=rank, shuffle=True,
        )
        for d in datasets
    ]


def build_distributed_eval_sampler_list(
    datasets: Sequence[torch.utils.data.Dataset],
    num_tasks: int,
    rank: int,
    dist_eval: bool,
) -> List[torch.utils.data.Sampler]:
    """DistributedSampler (shuffle=False) when dist_eval else SequentialSampler."""
    if dist_eval:
        return [
            torch.utils.data.DistributedSampler(
                d, num_replicas=num_tasks, rank=rank, shuffle=False,
            )
            for d in datasets
        ]
    return [torch.utils.data.SequentialSampler(d) for d in datasets]


def build_dataloader_list(
    datasets: Sequence[torch.utils.data.Dataset],
    samplers: Sequence[torch.utils.data.Sampler],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> List[torch.utils.data.DataLoader]:
    """Pair each (dataset, sampler) into a DataLoader with shared per-loader settings."""
    return [
        torch.utils.data.DataLoader(
            d,
            sampler=s,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        for d, s in zip(datasets, samplers)
    ]


def wrap_distributed(dist_cfg: DistributedConfig, model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """DDP-wrap model when dist_cfg.distributed is true. Returns (model, model_without_ddp)."""
    if dist_cfg.distributed:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_cfg.gpu], find_unused_parameters=True,
        )
        return wrapped, wrapped.module
    return model, model


def make_lr_schedule(optim_cfg: OptimizerConfig, trainer_cfg: TrainerConfig, num_training_steps_per_epoch: int) -> np.ndarray:
    """Cosine LR schedule with optional warmup. Common to all runs."""
    return utils.cosine_scheduler(
        optim_cfg.lr, optim_cfg.min_lr, trainer_cfg.epochs, num_training_steps_per_epoch,
        warmup_epochs=optim_cfg.warmup_epochs, warmup_steps=optim_cfg.warmup_steps,
    )


def make_wd_schedule(optim_cfg: OptimizerConfig, trainer_cfg: TrainerConfig, num_training_steps_per_epoch: int) -> np.ndarray:
    """Cosine WD schedule. If weight_decay_end is None, decay stays flat at weight_decay."""
    wd_end = optim_cfg.weight_decay_end if optim_cfg.weight_decay_end is not None else optim_cfg.weight_decay
    return utils.cosine_scheduler(
        optim_cfg.weight_decay, wd_end, trainer_cfg.epochs, num_training_steps_per_epoch,
    )


def append_log_line(output_cfg: OutputConfig, log_stats: dict) -> None:
    """Append a single JSON line to output_cfg.output_dir/log.txt (main process only)."""
    if output_cfg.output_dir and utils.is_main_process():
        with open(os.path.join(output_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


def print_training_time(start_time: float) -> None:
    """Log elapsed wall time as HH:MM:SS."""
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')
