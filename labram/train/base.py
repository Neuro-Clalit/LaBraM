from typing import Any, Optional, Sequence

import torch


def apply_lr_wd_schedule(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
) -> None:
    """Apply cosine LR and weight-decay schedules to each param group for one step."""
    if lr_schedule_values is None and wd_schedule_values is None:
        return
    for param_group in optimizer.param_groups:
        if lr_schedule_values is not None:
            param_group["lr"] = lr_schedule_values[global_step] * param_group.get("lr_scale", 1.0)
        if wd_schedule_values is not None and param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_schedule_values[global_step]


def log_lr_wd_grad_metrics(
    metric_logger: Any,
    optimizer: torch.optim.Optimizer,
    grad_norm: Optional[float],
    log_writer: Optional[Any] = None,
) -> None:
    """Update metric_logger (and optionally log_writer) with lr/min_lr/weight_decay/grad_norm."""
    min_lr = 10.0
    max_lr = 0.0
    for group in optimizer.param_groups:
        min_lr = min(min_lr, group["lr"])
        max_lr = max(max_lr, group["lr"])
    weight_decay_value = None
    for group in optimizer.param_groups:
        if group["weight_decay"] > 0:
            weight_decay_value = group["weight_decay"]
    metric_logger.update(lr=max_lr)
    metric_logger.update(min_lr=min_lr)
    metric_logger.update(weight_decay=weight_decay_value)
    metric_logger.update(grad_norm=grad_norm)
    if log_writer is not None:
        log_writer.update(lr=max_lr, head="opt")
        log_writer.update(min_lr=min_lr, head="opt")
        log_writer.update(weight_decay=weight_decay_value, head="opt")
        log_writer.update(grad_norm=grad_norm, head="opt")
