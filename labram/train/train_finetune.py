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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from timm.utils import ModelEma
from einops import rearrange

import labram.utils as utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import TrainerConfig
from labram.train.base import apply_lr_wd_schedule, log_lr_wd_grad_metrics


def train_class_batch(
    model: torch.nn.Module,
    samples: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    channel_indices: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(samples, channel_indices)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model: torch.nn.Module) -> float:
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    trainer_cfg: TrainerConfig,
    optim_cfg: OptimizerConfig,
    num_training_steps_per_epoch: Optional[int] = None,
    model_ema: Optional[ModelEma] = None,
    log_writer: Optional[Any] = None,
    start_steps: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
    ch_names: Optional[List[str]] = None,
    is_binary: bool = True,
) -> Dict[str, float]:
    update_freq = trainer_cfg.update_freq
    max_norm    = optim_cfg.clip_grad or 0

    channel_indices = None
    if ch_names is not None:
        channel_indices = utils.get_channel_indices(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        global_step = start_steps + step
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            apply_lr_wd_schedule(optimizer, global_step, lr_schedule_values, wd_schedule_values)

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, channel_indices)
        else:
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                loss, output = train_class_batch(
                    model, samples, targets, criterion, channel_indices)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)

        log_lr_wd_grad_metrics(metric_logger, optimizer, grad_norm, log_writer)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    header: str = 'Test:',
    ch_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    is_binary: bool = True,
) -> Dict[str, float]:
    if metrics is None:
        metrics = ['acc']
    channel_indices = None
    if ch_names is not None:
        channel_indices = utils.get_channel_indices(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg_batch = batch[0]
        target = batch[-1]
        eeg_batch = eeg_batch.float().to(device, non_blocking=True) / 100
        eeg_batch = rearrange(eeg_batch, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            output = model(eeg_batch, channel_indices=channel_indices)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = eeg_batch.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret


_LOGGED_EVAL_KEYS = (
    'accuracy', 'balanced_accuracy', 'f1_weighted', 'pr_auc', 'roc_auc', 'cohen_kappa', 'loss',
)


def _log_eval_stats(log_writer, stats, head, epoch):
    if log_writer is None:
        return
    for key, value in stats.items():
        if key in _LOGGED_EVAL_KEYS:
            log_writer.update(**{key: value}, head=head, step=epoch)


def train_loop(
    config,  # FinetuneRunConfig
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    criterion: torch.nn.Module,
    loaders,
    optimizer,
    device: torch.device,
    loss_scaler,
    log_writer,
    ch_names,
    metrics,
    n_parameters: int,
    num_training_steps_per_epoch: int,
    model_ema=None,
    enable_deepspeed: bool = False,
) -> None:
    """Epoch loop extracted from the runner."""
    import time
    import labram.runs.common as runner_common

    lr_schedule_values = runner_common.make_lr_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)

    nb_classes = config.model.nb_classes
    is_binary = nb_classes == 1

    print(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()
    max_accuracy = max_accuracy_test = 0.0

    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if config.distributed.distributed:
            loaders.train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * config.trainer.update_freq)

        train_stats = train_one_epoch(
            model, criterion, loaders.train, optimizer,
            device, epoch, loss_scaler,
            trainer_cfg=config.trainer,
            optim_cfg=config.optimizer,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            model_ema=model_ema,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names=ch_names,
            is_binary=is_binary,
        )

        if config.output.output_dir and config.output.save_ckpt:
            utils.save_model(
                output_cfg=config.output, trainer_cfg=config.trainer,
                model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch=epoch, model_ema=model_ema,
                enable_deepspeed=enable_deepspeed)

        if loaders.val is not None:
            val_stats = evaluate(loaders.val, model, device, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=is_binary)
            print(f"Val EEG accuracy: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(loaders.test, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=is_binary)
            print(f"Test EEG accuracy: {test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if config.output.output_dir and config.output.save_ckpt:
                    utils.save_model(
                        output_cfg=config.output, trainer_cfg=config.trainer,
                        model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best", model_ema=model_ema,
                        enable_deepspeed=enable_deepspeed)
                max_accuracy_test = test_stats["accuracy"]
            print(f'Max accuracy val: {max_accuracy:.2f}%, test: {max_accuracy_test:.2f}%')

            _log_eval_stats(log_writer, val_stats, head="val", epoch=epoch)
            _log_eval_stats(log_writer, test_stats, head="test", epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}

        if log_writer is not None and config.output.output_dir and utils.is_main_process():
            log_writer.flush()
        runner_common.append_log_line(config.output, log_stats)

    runner_common.print_training_time(start_time)
