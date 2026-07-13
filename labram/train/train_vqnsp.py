# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import math
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

import labram.utils as utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import OutputConfig
from labram.optim_factory import apply_lr_wd_schedule, log_lr_wd_grad_metrics

logger = utils.get_logger(__name__)


def _get_codebook_zero_count(inner_model: torch.nn.Module) -> Optional[int]:
    if not hasattr(inner_model, 'quantizer'):
        return None
    try:
        cluster_size = inner_model.quantizer._codebook.cluster_size
    except AttributeError:
        cluster_size = inner_model.quantizer.cluster_size
    return int((cluster_size == 0).sum().item())


def train_one_epoch(
    model: torch.nn.Module,
    data_loader_list: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    optim_cfg: OptimizerConfig,
    output_cfg: OutputConfig,
    log_writer: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    start_steps: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    ch_names_list: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # None => no clipping (the scaler still measures the grad norm). The
    # previous `clip_grad or 0` zeroed all gradients every step when clip_grad
    # was unset, because clip_grad_norm_(params, 0) clips the norm to zero.
    clip_grad = optim_cfg.clip_grad
    output_dir = output_cfg.output_dir

    inner_model = utils.get_model(model)
    if hasattr(inner_model, 'quantizer'):
        try:
            inner_model.quantizer.reset_cluster_size(device)
            logger.info("Reset the codebook statistic info in quantizer before each epoch")
        except AttributeError:
            pass

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        channel_indices = utils.get_channel_indices(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            global_step = start_steps + step + step_loader
            apply_lr_wd_schedule(optimizer, global_step, lr_schedule_values)

            eeg_batch = batch.float().to(device, non_blocking=True) / 100

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                loss, loss_dict = model(eeg_batch, channel_indices=channel_indices)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.error("Loss is %s, stopping training", loss_value)
                utils.save_nan_model(output_dir, model)
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            filtered_loss_dict = {k.split('/')[-1]: v for k, v in loss_dict.items() if k not in ['total_loss']}
            metric_logger.update(**filtered_loss_dict)

            log_lr_wd_grad_metrics(metric_logger, optimizer, grad_norm, log_writer)

            if log_writer is not None:
                log_writer.update(**filtered_loss_dict, head="train/loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['unused_code'] = zero_cnt
        logger.info("Unused code in codebook: %s", zero_cnt)
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader_list: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    log_writer: Optional[Any] = None,
    epoch: Optional[int] = None,
    ch_names_list: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    model.eval()
    inner_model = utils.get_model(model)

    if hasattr(inner_model, 'quantizer'):
        try:
            inner_model.quantizer.reset_cluster_size(device)
            logger.info("Reset the codebook statistic info in quantizer before testing")
        except AttributeError:
            pass

    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        channel_indices = utils.get_channel_indices(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            eeg_batch = batch.float().to(device, non_blocking=True) / 100
            loss, loss_dict = model(eeg_batch, channel_indices=channel_indices)
            metric_logger.update(loss=loss.item())
            filtered_loss_dict = {k.split('/')[-1]: v for k, v in loss_dict.items() if k not in ['total_loss']}
            metric_logger.update(**filtered_loss_dict)

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        logger.info("Unused code in codebook: %s", zero_cnt)
        return test_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def calculate_codebook_usage(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    codebook_size: int,
    log_writer: Optional[Any] = None,
    epoch: Optional[int] = None,
) -> None:
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Calculating codebook usage:'
    model.eval()
    codebook_cnt = torch.zeros(codebook_size, dtype=torch.float64).to(device)

    for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg_batch = batch.float().to(device, non_blocking=True) / 100
        outputs = utils.get_model(model).get_tokens(eeg_batch)['token'].view(-1)
        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1)
        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_size)

    zero_cnt = (codebook_cnt == 0).sum()
    logger.info("STAT:  %s tokens (%s%%) never used.", zero_cnt, (zero_cnt / codebook_size) * 100)


def train_loop(
    config,  # VQNSPRunConfig
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    data_loader_train_list,
    data_loader_val_list,
    train_ch_names_list,
    val_ch_names_list,
    optimizer,
    device: torch.device,
    loss_scaler,
    log_writer,
    n_learnable_parameters: int,
    num_training_steps_per_epoch: int,
) -> None:
    """Epoch loop extracted from the runner."""
    import time
    import labram.runs.common as runner_common

    lr_schedule_values = runner_common.make_lr_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)

    logger.info(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if config.distributed.distributed:
            for dl in data_loader_train_list:
                dl.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train_list, optimizer, device, epoch, loss_scaler,
            optim_cfg=config.optimizer,
            output_cfg=config.output,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
        )

        if config.output.output_dir and not config.output.save_only_final_model:
            utils.save_model(
                output_cfg=config.output, trainer_cfg=config.trainer,
                model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        if data_loader_val_list is not None:
            test_stats = evaluate(
                data_loader_val_list, model, device, log_writer, epoch,
                ch_names_list=val_ch_names_list,
            )
            logger.info(f"Validation loss: {test_stats['loss']:.4f}")
            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}

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
