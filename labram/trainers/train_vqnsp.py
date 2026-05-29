# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

import labram.utils as utils
from labram.trainers.base import apply_lr_wd_schedule, log_lr_wd_grad_metrics


def _get_codebook_zero_count(inner_model: torch.nn.Module) -> Optional[int]:
    """Return number of unused codebook entries, or None if model has no quantizer."""
    if not hasattr(inner_model, 'quantize'):
        return None
    try:
        cluster_size = inner_model.quantize._codebook.cluster_size
    except AttributeError:
        cluster_size = inner_model.quantize.cluster_size
    return int((cluster_size == 0).sum().item())


def train_one_epoch(
    model: torch.nn.Module,
    data_loader_list: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: float = 0,
    log_writer: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    start_steps: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    ch_names_list: Optional[List[List[str]]] = None,
    output_dir: str = '',
) -> Dict[str, float]:
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    inner_model = utils.get_model(model)
    if hasattr(inner_model, 'quantize'):
        try:
            inner_model.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
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
                print("Loss is {}, stopping training".format(loss_value), force=True)
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
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
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

    if hasattr(inner_model, 'quantize'):
        try:
            inner_model.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
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
    print("Averaged stats:", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
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

    codebook_num = codebook_size
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg_batch = batch.float().to(device, non_blocking=True) / 100

        outputs = utils.get_model(model).get_tokens(eeg_batch)['token'].view(-1)

        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1)

        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

    zero_cnt = (codebook_cnt == 0).sum()
    print(f"STAT:  {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never are used in this codebook.")
