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
from labram.configs.train_config import DistributedConfig, OutputConfig, TrainerConfig


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

    return device, utils.get_world_size(), utils.get_rank()


def create_log_writer(output_cfg: OutputConfig, global_rank: int) -> Optional[Any]:
    """Construct a TensorboardLogger if and only if rank 0 has output_cfg.log_dir."""
    if global_rank == 0 and output_cfg.log_dir:
        os.makedirs(output_cfg.log_dir, exist_ok=True)
        return utils.TensorboardLogger(log_dir=output_cfg.log_dir)
    return None


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
    """Print elapsed wall time as HH:MM:SS."""
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
