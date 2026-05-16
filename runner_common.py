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
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import utils


def setup_environment(args, init_cudnn_benchmark: bool = True) -> Tuple[torch.device, int, int]:
    """Initialize distributed, resolve device, seed, and cudnn flags.

    Returns (device, num_tasks, global_rank). The args object is mutated by
    utils.init_distributed_mode to add .distributed / .gpu / .rank / etc.
    """
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if init_cudnn_benchmark and torch.cuda.is_available():
        cudnn.benchmark = True

    return device, utils.get_world_size(), utils.get_rank()


def create_log_writer(args, global_rank: int) -> Optional[Any]:
    """Construct a TensorboardLogger if and only if rank 0 has args.log_dir."""
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        return utils.TensorboardLogger(log_dir=args.log_dir)
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


def wrap_distributed(args, model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """DDP-wrap model when args.distributed is true. Returns (model, model_without_ddp)."""
    if args.distributed:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True,
        )
        return wrapped, wrapped.module
    return model, model


def make_lr_schedule(args, num_training_steps_per_epoch: int) -> np.ndarray:
    """Cosine LR schedule with optional warmup. Common to all runners."""
    return utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )


def make_wd_schedule(args, num_training_steps_per_epoch: int) -> np.ndarray:
    """Cosine WD schedule. If args.weight_decay_end is None, decay stays flat at args.weight_decay."""
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    return utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch,
    )


def append_log_line(args, log_stats: dict) -> None:
    """Append a single JSON line to args.output_dir/log.txt (main process only)."""
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


def print_training_time(start_time: float) -> None:
    """Print elapsed wall time as HH:MM:SS."""
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
