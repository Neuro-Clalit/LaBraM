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
from labram.configs.train_config import LoggingConfig, OutputConfig
from labram.optim_factory import apply_lr_wd_schedule, log_lr_wd_grad_metrics

logger = utils.get_logger(__name__)


# Aggregate losses (and non-loss counters such as ``unused_code``) are reported
# as-is; only the per-component losses become shares of the component total.
_TOTAL_LOSS_KEYS = frozenset({'loss', 'total_loss'})


def _writer_loss_values(values: Dict[str, Any],
                        logging_cfg: Optional[LoggingConfig]) -> Dict[str, Any]:
    """Prepare a VQNSP loss dict for the metric writer.

    Totals and counters pass through unchanged; the per-component losses
    (``rec_loss`` / ``rec_angle_loss`` / ``quant_loss``) are reported as their
    share of the component total when relative logging is on, so the plot shows
    how the reconstruction/quantization terms trade off rather than their raw
    (weight- and dataset-dependent) magnitudes.
    """
    components = {k: v for k, v in values.items()
                  if k.endswith('_loss') and k not in _TOTAL_LOSS_KEYS}
    passthrough = {k: v for k, v in values.items() if k not in components}
    return {**passthrough,
            **(utils.relative_components_if_enabled(components, logging_cfg) or {})}


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
    logging_cfg: Optional[LoggingConfig] = None,
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
    step_timer = utils.StepTimer(
        device, precise_cuda=bool(getattr(logging_cfg, 'precise_cuda_timing', False)))
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        channel_indices = utils.get_channel_indices(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            data_time = step_timer.start_step()
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

            metric_logger.update(loss=loss_value)
            filtered_loss_dict = {k.split('/')[-1]: v for k, v in loss_dict.items() if k not in ['total_loss']}
            metric_logger.update(**filtered_loss_dict)

            log_lr_wd_grad_metrics(metric_logger, optimizer, grad_norm, log_writer)
            step_timing = step_timer.end_step(data_time, batch.shape[0])
            step_timing.update(step_timer.collect_ready_gpu_times())
            metric_logger.update(**step_timing)

            if log_writer is not None:
                log_writer.update(**step_timing, head="timing")
                log_writer.update(**_writer_loss_values(filtered_loss_dict, logging_cfg),
                                  head="train/loss")
                # loss_scale (AMP) can reach ~65536 — its own plot keeps it off
                # the small-valued "opt" (lr/wd) axis.
                log_writer.update(loss_scale=loss_scale_value, head="scale")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step

    metric_logger.update(**step_timer.finish())
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['unused_code'] = zero_cnt
        logger.info("Unused code in codebook: %s", zero_cnt)
        train_stat['samples_processed'] = step_timer.samples_processed * utils.get_world_size()
        return train_stat
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    train_stat['samples_processed'] = step_timer.samples_processed * utils.get_world_size()
    return train_stat


@torch.no_grad()
def evaluate(
    data_loader_list: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    log_writer: Optional[Any] = None,
    epoch: Optional[int] = None,
    ch_names_list: Optional[List[List[str]]] = None,
    logging_cfg: Optional[LoggingConfig] = None,
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

    step_timer = utils.StepTimer(
        device, precise_cuda=bool(getattr(logging_cfg, 'precise_cuda_timing', False)))
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        channel_indices = utils.get_channel_indices(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            data_time = step_timer.start_step()
            eeg_batch = batch.float().to(device, non_blocking=True) / 100
            loss, loss_dict = model(eeg_batch, channel_indices=channel_indices)
            metric_logger.update(loss=loss.item())
            filtered_loss_dict = {k.split('/')[-1]: v for k, v in loss_dict.items() if k not in ['total_loss']}
            metric_logger.update(**filtered_loss_dict)
            metric_logger.update(**step_timer.end_step(data_time, batch.shape[0]))
            metric_logger.update(**step_timer.collect_ready_gpu_times())

    metric_logger.update(**step_timer.finish())
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    zero_cnt = _get_codebook_zero_count(inner_model)
    if zero_cnt is not None:
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        logger.info("Unused code in codebook: %s", zero_cnt)
        test_stat['samples_processed'] = step_timer.samples_processed * utils.get_world_size()
        return test_stat
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stat['samples_processed'] = step_timer.samples_processed * utils.get_world_size()
    return test_stat


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

    runner_common.configure_relative_step_axis(
        log_writer, config, num_training_steps_per_epoch)

    logger.info(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        epoch_timer = utils.PhaseTimer(device)
        if config.distributed.distributed:
            for dl in data_loader_train_list:
                dl.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_timer = utils.PhaseTimer(device)
        train_stats = train_one_epoch(
            model, data_loader_train_list, optimizer, device, epoch, loss_scaler,
            optim_cfg=config.optimizer,
            output_cfg=config.output,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
            logging_cfg=config.logging,
        )
        train_timing = utils.timing_stats(
            'train', train_timer.elapsed(), int(train_stats.get('samples_processed', 0)))

        checkpoint_timer = utils.PhaseTimer(device)
        if config.output.output_dir and not config.output.save_only_final_model:
            utils.save_model(
                output_cfg=config.output, trainer_cfg=config.trainer,
                model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )
        checkpoint_timing = utils.timing_stats('checkpoint', checkpoint_timer.elapsed())

        if data_loader_val_list is not None:
            validation_timer = utils.PhaseTimer(device)
            test_stats = evaluate(
                data_loader_val_list, model, device, log_writer, epoch,
                ch_names_list=val_ch_names_list,
                logging_cfg=config.logging,
            )
            validation_timing = utils.timing_stats(
                'validation', validation_timer.elapsed(),
                int(test_stats.get('samples_processed', 0)))
            logger.info(f"Validation loss: {test_stats['loss']:.4f}")
            if log_writer is not None:
                log_writer.update(**_writer_loss_values(test_stats, config.logging),
                                  head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters,
                         **train_timing, **checkpoint_timing, **validation_timing}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters,
                         **train_timing, **checkpoint_timing}

        epoch_timing = {
            **utils.timing_stats('epoch', epoch_timer.elapsed()),
            **utils.timing_stats('run_elapsed', time.time() - start_time),
        }
        log_stats.update(epoch_timing)
        utils.log_timing_stats(log_writer, {k: v for k, v in log_stats.items()
                                             if k.endswith('_time_sec') or k.endswith('_samples_per_sec')}, epoch)

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
