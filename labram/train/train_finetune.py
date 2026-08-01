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
import torch.utils.data
from timm.utils import ModelEma
from einops import rearrange

import labram.utils as utils
from labram.configs.loss_config import LossConfig
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import EvaluationConfig, LoggingConfig, TrainerConfig
from labram.data.bundles import CLASSIFICATION as TASK_CLASSIFICATION
from labram.data.bundles import REGRESSION as TASK_REGRESSION
from labram.losses import CodebookRegularizedCriterion, build_downstream_criterion
from labram.losses.outputs import LossBreakdown
from labram.models.outputs import PredictorOutput
from labram.optim_factory import (
    apply_lr_wd_schedule,
    compute_component_grad_norms,
    log_lr_wd_grad_metrics,
    optimizer_update,
)
from labram.utils import plots

logger = utils.get_logger(__name__)


def train_class_batch(
    model: torch.nn.Module,
    samples: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    channel_indices: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[LossBreakdown]]:
    """Run the model + criterion. Returns ``(loss, logits, breakdown)``.

    For the codebook-regularized criterion the model emits a ``PredictorOutput``
    and the criterion a ``LossBreakdown`` (component losses for logging); for the
    plain path ``outputs`` is already the logits tensor and ``breakdown`` is None.
    """
    outputs = model(samples, channel_indices)
    if isinstance(criterion, CodebookRegularizedCriterion):
        breakdown = criterion(outputs, target)
        return breakdown.total, outputs.logits, breakdown
    loss = criterion(outputs, target)
    return loss, outputs, None


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
    eval_cfg: Optional[EvaluationConfig] = None,
    nb_classes: Optional[int] = None,
    task: str = TASK_CLASSIFICATION,
    target_stats: Optional[Tuple[float, float]] = None,
    logging_cfg: Optional[LoggingConfig] = None,
) -> Dict[str, float]:
    update_freq = trainer_cfg.update_freq
    if nb_classes is None:
        nb_classes = 1 if is_binary else 2
    is_regression = task == TASK_REGRESSION
    # A regression target is a single float per window, so it is shaped like a
    # binary target (B, 1) -- but it must never be passed through a sigmoid or
    # scored as a probability.
    scalar_target = is_binary or is_regression
    detailed = eval_cfg is not None and eval_cfg.detailed_metrics
    log_grad_components = eval_cfg is not None and eval_cfg.log_grad_components
    grad_freq = eval_cfg.log_grad_freq if eval_cfg is not None else 0
    train_pred: List[torch.Tensor] = []
    train_true: List[torch.Tensor] = []
    # None => no gradient clipping. (Previously `clip_grad or 0`, which passed 0
    # to the scaler and clipped grad-norm to zero — i.e. zeroed all gradients —
    # whenever clip_grad was unset.)
    max_norm = optim_cfg.clip_grad

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
        if scalar_target:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output, breakdown = train_class_batch(
                model, samples, targets, criterion, channel_indices)
        else:
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                loss, output, breakdown = train_class_batch(
                    model, samples, targets, criterion, channel_indices)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.error("Loss is %s, stopping training", loss_value)
            sys.exit(1)

        # Per-loss-component gradient norms (opt-in, periodic). Computed here —
        # before the full backward frees the graph — via autograd.grad with
        # retain_graph=True so the subsequent optimizer step is unaffected.
        component_grad_values = None
        if (log_grad_components and breakdown is not None and grad_freq > 0
                and global_step % grad_freq == 0):
            component_grad_values = compute_component_grad_norms(
                breakdown.components, model.parameters())

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
            grad_norm = optimizer_update(
                loss, optimizer=optimizer, loss_scaler=loss_scaler,
                parameters=model.parameters(), clip_grad=max_norm,
                update_grad=(data_iter_step + 1) % update_freq == 0,
                create_graph=is_second_order, model_ema=model_ema, model=model)
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ``step_scores`` are what a detailed epoch-level report is built from:
        # probabilities for classification, raw scalar predictions for regression.
        step_target = targets.detach().float().cpu()
        if is_regression:
            step_scores = output.detach().float().cpu()
            # Report the running error in the target's own units (years), which
            # the normalized training loss does not give directly.
            step_metric = utils.regression_metrics_fn(
                utils.denormalize(step_scores.numpy(), target_stats),
                utils.denormalize(step_target.numpy(), target_stats),
                ["mae"],
            )["mae"]
        elif is_binary:
            step_scores = torch.sigmoid(output).detach().cpu()
            step_metric = utils.get_metrics(
                step_scores.numpy(), step_target.numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            step_scores = output.detach().float().cpu()
            step_metric = (output.max(-1)[-1] == targets.squeeze()).float().mean()

        # Accumulate predictions for the epoch-level detailed train report.
        if detailed:
            train_pred.append(step_scores)
            train_true.append(step_target)

        metric_logger.update(loss=loss_value)
        # Keep the meter name stable across tasks so the existing logging and
        # progress output need no branching; for regression it holds MAE.
        metric_logger.update(**{('mae' if is_regression else 'class_acc'): step_metric})
        metric_logger.update(loss_scale=loss_scale_value)

        # Per-component losses (classifier / magnitude / phase / quantize) when
        # the codebook-regularized criterion is in use.
        component_values = None
        if breakdown is not None:
            component_values = {f'{name}_loss': float(v.detach().mean())
                                for name, v in breakdown.components.items()}
            metric_logger.update(**component_values)

        if component_grad_values is not None:
            metric_logger.update(**component_grad_values)

        log_lr_wd_grad_metrics(metric_logger, optimizer, grad_norm, log_writer)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if is_regression:
                # MAE is in years (O(10)), so it would flatten the O(1) loss
                # curve if it shared the "loss" plot.
                log_writer.update(mae=step_metric, head="err")
            else:
                log_writer.update(class_acc=step_metric, head="loss")
            # The AMP loss scale can reach ~65536; on its own "scale" plot so it
            # does not flatten the small-valued lr/wd series on the "opt" plot.
            log_writer.update(loss_scale=loss_scale_value, head="scale")
            # Components go to the writer as shares of their total (scale-free,
            # comparable across runs/loss weights) unless relative logging is
            # off; the console/`stats` meters above keep the raw magnitudes.
            for name, value in (utils.relative_components_if_enabled(
                    component_values, logging_cfg) or {}).items():
                log_writer.update(**{name: value}, head="loss")
            for name, value in (utils.relative_components_if_enabled(
                    component_grad_values, logging_cfg) or {}).items():
                log_writer.update(**{name: value}, head="grad")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Epoch-level detailed train metrics (F1 / sensitivity / specificity /
    # confusion matrix / ROC-PR) over all accumulated train predictions.
    if detailed and train_pred:
        p = torch.cat(train_pred, dim=0).numpy()
        t = torch.cat(train_true, dim=0).numpy()
        if is_regression:
            report = utils.regression_report(
                utils.denormalize(p, target_stats), utils.denormalize(t, target_stats))
        else:
            report = utils.classification_report(p, t, is_binary, nb_classes, 0.5)
        stats.update(report.scalars)
        if log_writer is not None:
            _log_eval_stats(log_writer, report.scalars, head="train", epoch=epoch)
            _log_detailed_report(log_writer, report, "train", epoch, eval_cfg)
    return stats


@torch.no_grad()
def evaluate(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    header: str = 'Test:',
    ch_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    is_binary: bool = True,
    nb_classes: Optional[int] = None,
    eval_cfg: Optional[EvaluationConfig] = None,
    log_writer: Optional[Any] = None,
    head: Optional[str] = None,
    epoch: Optional[int] = None,
    task: str = TASK_CLASSIFICATION,
    target_stats: Optional[Tuple[float, float]] = None,
    loss_cfg: Optional[LossConfig] = None,
) -> Dict[str, float]:
    """Evaluate a split.

    When ``eval_cfg.detailed_metrics`` is set (the default), the returned dict is
    augmented with the detailed classification report (F1, sensitivity,
    specificity, confusion-matrix cells, ...), and — when ``log_writer``/``head``
    are given — the confusion matrix and ROC/PR curves are pushed to the
    writer(s).

    Metrics are always computed per crop window (~10 s). Additionally, when
    ``eval_cfg.agg_windows`` selects an aggregation mode and the loader yields a
    per-window case id (a 3-tuple ``(X, y, case_id)``), the windows of each EEG
    case are pooled into one prediction per case and case-level metrics are
    computed *as well*. In that mode the returned dict's primary keys are the
    case-level metrics and the per-window metrics are mirrored under ``window_*``
    keys (and logged under a ``{head}_window`` plot); otherwise the primary keys
    are the per-window metrics.

    For ``task='regression'`` the scalar output is scored directly (no sigmoid)
    and de-normalized with ``target_stats`` first, so the reported error is in the
    target's original units.

    When the loader shards its dataset across ranks (``distributed.dist_eval``),
    the per-rank predictions are all-gathered back into dataset order first, so
    metrics — per-case pooling in particular — see every window of every case
    exactly once rather than only the local shard.
    """
    if metrics is None:
        metrics = ['acc']
    if nb_classes is None:
        nb_classes = 1 if is_binary else 2
    is_regression = task == TASK_REGRESSION
    scalar_target = is_binary or is_regression
    channel_indices = None
    if ch_names is not None:
        channel_indices = utils.get_channel_indices(ch_names)
    # Dispatched on the task: scoring a regression run with the classification
    # criterion would silently compute cross-entropy on ages. ``loss_cfg`` is only
    # forwarded for regression (to pick mse/l1/huber) so the classification eval
    # criterion stays exactly what it was before this branch existed.
    criterion = build_downstream_criterion(
        task, 1 if is_binary else 2, loss_cfg if is_regression else None)

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    groups: List = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg_batch = batch[0]
        # A 3-element batch carries a per-window case id for aggregation.
        target = batch[1] if len(batch) >= 3 else batch[-1]
        group_batch = batch[2] if len(batch) >= 3 else None
        eeg_batch = eeg_batch.float().to(device, non_blocking=True) / 100
        eeg_batch = rearrange(eeg_batch, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if scalar_target:
            target = target.float().unsqueeze(-1)

        # compute output (classify_only skips the decoder branch on the
        # regularized model; the plain model ignores the kwarg)
        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            output = model(eeg_batch, channel_indices=channel_indices, classify_only=True)
            if isinstance(output, PredictorOutput):
                output = output.logits
            loss = criterion(output, target)

        if is_regression:
            # A scalar prediction, not a probability: no sigmoid.
            output = output.float().cpu()
        elif is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.float().cpu() if is_regression else target.cpu()

        pred.append(output)
        true.append(target)
        if group_batch is not None:
            groups.extend(list(group_batch))

        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    # With a DistributedSampler on the eval loader each rank holds only its
    # ~1/world_size shard; gather the shards back into dataset order before
    # computing any metric. (Without dist_eval every rank iterates the full
    # dataset, so there is nothing to gather.)
    if _is_sharded_loader(data_loader):
        pred, true, groups = utils.gather_sharded_eval(
            pred, true, groups, total=_loader_dataset_len(data_loader))

    if is_regression:
        # Undo the loader's z-scoring once, on the gathered arrays, so every
        # downstream metric and aggregation works in the target's original units
        # (years, for age).
        pred = utils.denormalize(pred, target_stats)
        true = utils.denormalize(true, target_stats)

    agg_mode = eval_cfg.agg_windows if eval_cfg is not None else 'none'
    detailed = eval_cfg is None or eval_cfg.detailed_metrics
    loss_avg = metric_logger.loss.global_avg

    # Per-window ("per-crop") metrics: one prediction per ~10 s EEG window.
    window_ret, window_report = _metrics_and_report(
        pred, true, metrics, is_binary, nb_classes, detailed, task)
    window_ret['loss'] = loss_avg

    if agg_mode != 'none' and groups:
        # Pool the windows of each EEG case (recording/subject) into a single
        # prediction per case, then compute metrics at the case level *in
        # addition to* the per-window metrics above. Case-level metrics are the
        # primary set (a clinical decision is per EEG case, not per crop); the
        # per-window metrics are retained under ``window_*`` keys.
        case_pred, case_true = utils.aggregate_windows(
            pred, true, groups, agg_mode, is_binary, is_regression=is_regression)
        logger.info("Aggregated %d windows into %d cases (mode=%s)",
                    len(groups), len(case_true), agg_mode)
        ret, case_report = _metrics_and_report(
            case_pred, case_true, metrics, is_binary, nb_classes, detailed, task)
        ret['loss'] = loss_avg
        ret.update({f'window_{k}': v for k, v in window_ret.items()})
        primary_report = case_report
        # Case-level scalars are logged by the caller (train_loop) under ``head``;
        # log the per-window scalars + detailed report here under ``{head}_window``.
        if log_writer is not None and head is not None:
            _log_eval_stats(log_writer, window_ret, head=f"{head}_window", epoch=epoch)
            if window_report is not None:
                _log_detailed_report(
                    log_writer, window_report, f"{head}_window", epoch, eval_cfg)
    else:
        ret = window_ret
        primary_report = window_report

    if primary_report is not None and log_writer is not None and head is not None:
        _log_detailed_report(log_writer, primary_report, head, epoch, eval_cfg)

    return ret


def _is_sharded_loader(data_loader: Any) -> bool:
    """True when ``data_loader`` splits its dataset across ranks (i.e. carries a
    :class:`~torch.utils.data.DistributedSampler`), so each rank sees a shard."""
    sampler = getattr(data_loader, 'sampler', None)
    return isinstance(sampler, torch.utils.data.DistributedSampler)


def _loader_dataset_len(data_loader: Any) -> Optional[int]:
    """Length of the loader's underlying dataset — the count to truncate to
    after gathering, dropping DistributedSampler's equalizing duplicates."""
    dataset = getattr(data_loader, 'dataset', None)
    try:
        return len(dataset) if dataset is not None else None
    except TypeError:  # iterable-style dataset with no __len__
        return None


def _metrics_and_report(
    pred: Any,
    true: Any,
    metrics: List[str],
    is_binary: bool,
    nb_classes: int,
    detailed: bool,
    task: str = TASK_CLASSIFICATION,
) -> Tuple[Dict[str, float], Optional[Any]]:
    """Compute the requested ``metrics`` and (when ``detailed``) the full
    report for one set of predictions. Returns ``(scalars, report)``.

    The report is a ``ClassificationReport`` or a ``RegressionReport`` depending on
    the task; both expose ``.scalars``."""
    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5, task=task)
    report = None
    if detailed:
        if task == TASK_REGRESSION:
            report = utils.regression_report(pred, true, metrics)
        else:
            report = utils.classification_report(pred, true, is_binary, nb_classes, 0.5)
        ret.update(report.scalars)
    return ret, report


# Rate/probability metrics all live on a normalized [0, 1] scale (plus ``loss``,
# which stays O(1)); they share the ``head`` plot.
_LOGGED_EVAL_RATE_KEYS = (
    'accuracy', 'balanced_accuracy', 'f1', 'f1_weighted', 'precision', 'recall',
    'sensitivity', 'specificity', 'pr_auc', 'roc_auc', 'cohen_kappa', 'loss',
    # Regression goodness-of-fit measures, also on a roughly [-1, 1] scale.
    'r2', 'pearson_r', 'spearman_r', 'age_bias_slope',
)
# Regression errors are in the target's units (years for age, so O(10)); they get
# their own ``{head}_err`` plot so they do not flatten the [0, 1] metrics above.
_LOGGED_EVAL_ERROR_KEYS = (
    'mae', 'rmse', 'mse', 'mae_corrected',
    'pred_mean', 'pred_std', 'target_mean', 'target_std',
)
# Confusion-matrix cell counts are raw integers that can reach the thousands
# (>> 1); logged on a separate ``{head}_cm`` plot so they do not flatten the
# normalized metrics above onto a single shared axis.
_LOGGED_EVAL_COUNT_KEYS = ('cm_tn', 'cm_fp', 'cm_fn', 'cm_tp')

_LOGGED_EVAL_KEYS = (
    _LOGGED_EVAL_RATE_KEYS + _LOGGED_EVAL_COUNT_KEYS + _LOGGED_EVAL_ERROR_KEYS)


def _epoch_axis_step(log_writer, epoch):
    """Place an epoch-level metric on the writer's x-axis.

    Returns the epoch unchanged on the absolute axis; on the relative axis it
    becomes the same normalized-progress value the per-iteration series use, so
    the train and val/test curves stay aligned."""
    if log_writer is not None and hasattr(log_writer, 'epoch_step'):
        return log_writer.epoch_step(epoch)
    return epoch


def _log_eval_stats(log_writer, stats, head, epoch):
    if log_writer is None:
        return
    step = _epoch_axis_step(log_writer, epoch)
    for key, value in stats.items():
        if key in _LOGGED_EVAL_RATE_KEYS:
            log_writer.update(**{key: value}, head=head, step=step)
        elif key in _LOGGED_EVAL_COUNT_KEYS:
            log_writer.update(**{key: value}, head=f"{head}_cm", step=step)
        elif key in _LOGGED_EVAL_ERROR_KEYS:
            log_writer.update(**{key: value}, head=f"{head}_err", step=step)


def _log_detailed_report(log_writer, report, head, step, eval_cfg):
    """Push a report's figures to the writer(s): a confusion matrix and ROC/PR
    curves for classification, a predicted-vs-true scatter for regression.
    Scalars are logged separately via ``_log_eval_stats``."""
    if log_writer is None or report is None or eval_cfg is None:
        return
    step = _epoch_axis_step(log_writer, step)
    if isinstance(report, utils.RegressionReport):
        if eval_cfg.log_curves and report.predictions is not None:
            fig = plots.prediction_scatter_figure(
                report.predictions, report.targets,
                metrics={'mae': report.scalars.get('mae'),
                         'r2': report.scalars.get('r2')},
                title=f'{head} predicted vs true')
            log_writer.report_figure(head, fig, step=step, series='pred_vs_true')
        return
    if eval_cfg.log_confusion_matrix and report.matrix is not None:
        log_writer.report_confusion_matrix(
            head, report.matrix, step=step, labels=report.labels)
    if eval_cfg.log_curves:
        if report.roc is not None:
            fpr, tpr = report.roc
            fig = plots.roc_curve_figure(fpr, tpr, report.scalars.get('roc_auc'),
                                         title=f'{head} ROC')
            log_writer.report_figure(head, fig, step=step, series='roc_curve')
        if report.pr is not None:
            rec, prec = report.pr
            fig = plots.pr_curve_figure(rec, prec, report.scalars.get('pr_auc'),
                                        title=f'{head} PR')
            log_writer.report_figure(head, fig, step=step, series='pr_curve')


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
) -> dict:
    """Epoch loop extracted from the runner.

    Returns a summary of the best-epoch metrics (selected by validation
    accuracy): ``max_accuracy`` (val), ``max_accuracy_test``, ``best_epoch`` and
    the full ``best_val_stats`` / ``best_test_stats`` dicts. Callers that don't
    need it (the plain fine-tune runner) simply ignore the return value; the
    cross-validation runner uses it to aggregate metrics across folds.
    """
    import time
    import labram.runs.common as runner_common

    lr_schedule_values = runner_common.make_lr_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)
    wd_schedule_values = runner_common.make_wd_schedule(
        config.optimizer, config.trainer, num_training_steps_per_epoch)

    # The writer advances once per micro-batch here, so the run spans
    # epochs * steps_per_epoch * update_freq logging steps.
    runner_common.configure_relative_step_axis(
        log_writer, config, num_training_steps_per_epoch,
        steps_per_logged_step=config.trainer.update_freq)

    nb_classes = config.model.nb_classes
    task = getattr(config.model, 'task', TASK_CLASSIFICATION)
    is_regression = task == TASK_REGRESSION
    # A regression head also has a single output, so nb_classes == 1 alone cannot
    # distinguish the two; the task decides.
    is_binary = nb_classes == 1 and not is_regression
    target_stats = getattr(config.model, 'target_stats', None)
    # Model selection metric and its direction: accuracy (higher is better) for
    # classification, MAE (lower is better) for regression.
    select_metric, select_dir = utils.best_metric_for(task, metrics)
    better = (lambda new, cur: new > cur) if select_dir == 'max' else (lambda new, cur: new < cur)

    logger.info(f"Start training for {config.trainer.epochs} epochs")
    start_time = time.time()
    best_val = float('-inf') if select_dir == 'max' else float('inf')
    best_test = 0.0
    best_epoch = -1
    best_val_stats: dict = {}
    best_test_stats: dict = {}

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
            eval_cfg=config.evaluation,
            nb_classes=nb_classes,
            task=task,
            target_stats=target_stats,
            logging_cfg=config.logging,
        )

        # Periodic/rolling per-epoch checkpoints are skipped when only the final
        # model is wanted (see the post-loop save below).
        if config.output.output_dir and config.output.save_ckpt and not config.output.save_only_final_model:
            utils.save_model(
                output_cfg=config.output, trainer_cfg=config.trainer,
                model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch=epoch, model_ema=model_ema,
                enable_deepspeed=enable_deepspeed)

        if loaders.val is not None:
            val_stats = evaluate(loaders.val, model, device, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=is_binary,
                                 nb_classes=nb_classes, eval_cfg=config.evaluation,
                                 log_writer=log_writer, head='val', epoch=epoch,
                                 task=task, target_stats=target_stats,
                                 loss_cfg=config.loss)
            unit = '' if is_regression else '%'
            logger.info(f"Val EEG {select_metric}: {val_stats[select_metric]:.2f}{unit}")
            test_stats = evaluate(loaders.test, model, device, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=is_binary,
                                  nb_classes=nb_classes, eval_cfg=config.evaluation,
                                  log_writer=log_writer, head='test', epoch=epoch,
                                  task=task, target_stats=target_stats,
                                  loss_cfg=config.loss)
            logger.info(f"Test EEG {select_metric}: {test_stats[select_metric]:.2f}{unit}")

            if better(val_stats[select_metric], best_val):
                best_val = val_stats[select_metric]
                if config.output.output_dir and config.output.save_ckpt and not config.output.save_only_final_model:
                    utils.save_model(
                        output_cfg=config.output, trainer_cfg=config.trainer,
                        model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch="best", model_ema=model_ema,
                        enable_deepspeed=enable_deepspeed)
                best_test = test_stats[select_metric]
                best_epoch = epoch
                best_val_stats = dict(val_stats)
                best_test_stats = dict(test_stats)
            logger.info(f'Best {select_metric} val: {best_val:.2f}{unit}, '
                        f'test: {best_test:.2f}{unit}')

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

    # Save the final trained model. Always done at the end; it is the only
    # checkpoint written when save_only_final_model is set.
    if config.output.output_dir and config.output.save_ckpt:
        utils.save_model(
            output_cfg=config.output, trainer_cfg=config.trainer,
            model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler,
            epoch="final", model_ema=model_ema,
            enable_deepspeed=enable_deepspeed)

    runner_common.print_training_time(start_time)

    # No epoch ever improved (e.g. no val loader), so report 0.0 rather than the
    # +/-inf sentinel the comparison started from.
    if best_epoch < 0:
        best_val = best_test = 0.0

    return {
        # Legacy key names, kept so eval/cv_aggregation.py and the ClearML summary
        # tables keep working unchanged; for regression they hold the selection
        # metric (MAE), not an accuracy.
        "max_accuracy": best_val,
        "max_accuracy_test": best_test,
        # Task-neutral aliases, plus the metric's identity so a reader knows what
        # the numbers above actually are.
        "best_metric": best_val,
        "best_metric_test": best_test,
        "best_metric_name": select_metric,
        "best_metric_direction": select_dir,
        "task": task,
        "best_epoch": best_epoch,
        "best_val_stats": best_val_stats,
        "best_test_stats": best_test_stats,
    }
