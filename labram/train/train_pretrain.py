# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import math
import sys
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

import labram.utils as utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import TrainerConfig
from labram.optim_factory import apply_lr_wd_schedule, log_lr_wd_grad_metrics

logger = utils.get_logger(__name__)


def random_masking(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Per-sample random masking via per-sample shuffling."""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.to(torch.bool)


def train_one_epoch(
    model: torch.nn.Module,
    vqnsp: torch.nn.Module,
    data_loader_list: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    trainer_cfg: TrainerConfig,
    optim_cfg: OptimizerConfig,
    distributed: bool = False,
    log_writer: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    start_steps: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
    ch_names_list: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()
    grad_accum = trainer_cfg.gradient_accumulation_steps
    # None => no clipping (the scaler still measures the grad norm). NOTE the
    # previous `clip_grad or 0` passed 0 to clip_grad_norm_ when unset, which
    # clips the grad norm to zero — i.e. zeroes all gradients every step.
    clip_grad = optim_cfg.clip_grad

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        channel_indices = utils.get_channel_indices(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * grad_accum, header)):
            global_step = start_steps + step + step_loader
            apply_lr_wd_schedule(optimizer, global_step, lr_schedule_values, wd_schedule_values)

            samples = batch
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                    input_ids = vqnsp.get_codebook_indices(samples, channel_indices)
                labels = input_ids[bool_masked_pos]
                labels_sym = input_ids[~bool_masked_pos]

            my_context = (model.no_sync if distributed and (step + 1) % grad_accum != 0
                          else nullcontext)
            with my_context():
                with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                    outputs = model(samples, channel_indices, bool_masked_pos=bool_masked_pos)
                    x_rec, x_rec_sym = outputs
                    loss_rec = loss_fn(x_rec, labels)
                    loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                    loss = loss_rec + loss_rec_sym

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.error("Loss is %s, stopping training at rank %s", loss_value, utils.get_rank())
                sys.exit(1)

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= grad_accum
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % grad_accum == 0)
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
            if (step + 1) % grad_accum == 0:
                optimizer.zero_grad()

            if device.type == 'cuda':
                torch.cuda.synchronize()

            mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
            mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
            metric_logger.update(mlm_acc=mlm_acc, mlm_acc_sym=mlm_acc_sym,
                                 loss_rec=loss_rec.item() / 2, loss=loss_value,
                                 loss_scale=loss_scale_value)

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, mlm_acc_sym=mlm_acc_sym,
                                  loss_rec=loss_rec.item() / 2, head="loss")

            log_lr_wd_grad_metrics(metric_logger, optimizer, grad_norm, log_writer)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                # loss_scale (AMP) can reach ~65536 — its own plot keeps it off
                # the small-valued "opt" (lr/wd) axis.
                log_writer.update(loss_scale=loss_scale_value, head="scale")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_loop(
    config,  # PretrainRunConfig — avoid circular import by not typing it
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    vqnsp: torch.nn.Module,
    data_loader_list,
    train_ch_names_list,
    optimizer,
    device: torch.device,
    loss_scaler,
    log_writer,
    n_parameters: int,
    num_training_steps_per_epoch: int,
) -> None:
    """Epoch loop extracted from the runner so runner.main() only owns setup."""
    import time
    import labram.utils as utils
    import labram.runs.common as runner_common

    lr_schedule_values = runner_common.make_lr_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)

    # Pre-training reports a single total loss (no component breakdown), so the
    # relative treatment here is the normalized-progress x-axis.
    runner_common.configure_relative_step_axis(
        log_writer, config, num_training_steps_per_epoch)

    logger.info(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if config.distributed.distributed:
            for dl in data_loader_list:
                dl.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_list, optimizer, device, epoch, loss_scaler,
            trainer_cfg=config.trainer,
            optim_cfg=config.optimizer,
            distributed=config.distributed.distributed,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names_list=train_ch_names_list,
        )

        if config.output.output_dir and not config.output.save_only_final_model:
            utils.save_model(
                output_cfg=config.output, trainer_cfg=config.trainer,
                model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if log_writer is not None and config.output.output_dir and utils.is_main_process():
            log_writer.flush()
        runner_common.append_log_line(config.output, log_stats)

    if config.output.output_dir:
        utils.save_model(
            output_cfg=config.output, trainer_cfg=config.trainer,
            model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, epoch="final",
        )

    runner_common.print_training_time(start_time)
