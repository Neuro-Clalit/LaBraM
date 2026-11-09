# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Recording-level multiple-instance (bag) training and evaluation (mil.*).
# ---------------------------------------------------------
"""End-to-end MIL fine-tuning (``mil.enabled``; ``docs/b1_attention_mil_plan.md``).

A batch is ``B`` recordings x up to ``K`` windows (``labram/data/bags.py``).
The real windows are encoded by the transformer on their present channels,
pooled per recording by ``model.mil`` (``labram/layers/bag_pool.py``) and the
existing head predicts once per recording; the loss is the run's criterion on
the recording target, plus the attention-entropy term for softmax pools and an
optional feature-retention term (MSE to the initial encoder's window features).

:func:`train_one_epoch_bags` and :func:`evaluate_bags` take the same keyword
arguments as ``train_finetune.train_one_epoch`` / ``evaluate`` (plus
``mil_cfg``) and return the same kind of stats dict, so ``train_loop`` swaps
them in unchanged. Evaluation reports per-recording metrics (overall, per
cohort and per dataset) as the primary set, the per-window head predictions as
``window_*``, and the mean of the window predictions as ``case_mean_*`` -- the
mean-pooling benchmark inside the same model.
"""

import copy
import math
import os
import sys
import types
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import labram.utils as utils
from labram.layers.bag_pool import MILPooling, attention_entropy_gap
from labram.losses.outputs import LossBreakdown
from labram.losses.regression import regression_output, regression_target
from labram.optim_factory import apply_lr_wd_schedule, optimizer_update
from labram.train.augment import augment_batch
from labram.train.channel_drop import ChannelDropper
from labram.utils.logging import get_logger

logger = get_logger(__name__)
COMPETING_POOLS = ("gated", "query")
_N_DESC = None


def window_descriptors(x_uv: torch.Tensor) -> torch.Tensor:
    """Quality + STFT + wavelet descriptors ``[N, Q]`` of raw windows ``[N, C, L]`` (uV)."""
    from labram.mil.cache import all_descriptors
    d = all_descriptors(x_uv.float())
    return torch.stack(list(d.values()), -1)


def n_descriptors() -> int:
    """Descriptor count including the minute column."""
    global _N_DESC
    if _N_DESC is None:
        x = torch.randn(1, 3, 2000)
        _N_DESC = window_descriptors(x).shape[-1] + 1
    return _N_DESC


# ----------------------------------------------------------------- model / optim
def attach_mil(model: torch.nn.Module, mil_cfg) -> torch.nn.Module:
    """Give ``model`` its bag pooling (``model.mil``), sized to the head input."""
    dim = model.head.in_features if hasattr(model.head, "in_features") else model.num_features
    kw = dict(score_input=mil_cfg.score_input, dropout=mil_cfg.dropout,
              time_encoding=mil_cfg.time_encoding)
    if mil_cfg.pool == "gated":
        kw.update(attn_dim=mil_cfg.attn_dim, branches=mil_cfg.branches,
                  gated=getattr(mil_cfg, "gated", True),
                  temperature=mil_cfg.temperature, topk_mask_prob=mil_cfg.topk_mask_prob)
    elif mil_cfg.pool == "sigmoid":
        kw.update(attn_dim=mil_cfg.attn_dim)
    elif mil_cfg.pool == "query":
        kw.update(heads=mil_cfg.heads, self_attention=mil_cfg.self_attention)
    model.mil = MILPooling(dim, mil_cfg.pool, n_descriptors(), **kw)
    return model


def split_head_lr(optimizer: torch.optim.Optimizer, model: torch.nn.Module, scale: float) -> None:
    """Move the pool and head parameters into their own param groups with
    ``lr_scale`` multiplied by ``scale`` (the schedule applies ``lr_scale``)."""
    if scale == 1.0:
        return
    model = getattr(model, "module", model)
    fast = {id(p) for n, p in model.named_parameters() if n.startswith(("mil.", "head."))}
    new_groups = []
    for g in optimizer.param_groups:
        moved = [p for p in g["params"] if id(p) in fast]
        if not moved:
            continue
        g["params"] = [p for p in g["params"] if id(p) not in fast]
        ng = {k: v for k, v in g.items() if k != "params"}
        ng["params"] = moved
        ng["lr_scale"] = g.get("lr_scale", 1.0) * scale
        new_groups.append(ng)
    optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
    for ng in new_groups:
        optimizer.add_param_group(ng)
    logger.info("MIL: pool + head learning rate x%.2f (%d params)", scale, len(fast))


@torch.no_grad()
def fit_descriptor_norm(model, loader, device, n_batches: int) -> None:
    """Fit ``model.mil.desc_norm`` on the descriptors of the first training bags."""
    m = getattr(model, "module", model)
    if m.mil.desc_norm is None or bool(m.mil.desc_fitted):
        return
    rows = []
    for i, (x, _, _, minutes, mask) in enumerate(loader):
        if i >= n_batches:
            break
        x, mask = x.to(device), mask.to(device)
        d = window_descriptors(x[mask])
        rows.append(torch.cat([d, minutes.to(device)[mask][:, None]], -1).cpu())
    m.mil.desc_norm.fit(torch.cat(rows))
    m.mil.desc_fitted.fill_(True)
    logger.info("MIL: descriptor standardization fitted on %d windows", sum(len(r) for r in rows))


# ------------------------------------------------------------------ batches
def prepare(batch, device, ch_names, dropper, mil, augment_cfg=None, train=False):
    """(x4 [N, C, A, T], channel_indices, bag dict, y [B, ...]) of a bag batch."""
    x, y, stems, minutes, mask = batch
    x = x.to(device, non_blocking=True)
    mask = mask.to(device)
    minutes = minutes.to(device)
    xr = x[mask]                                                   # [N, C, L] uV
    q = None
    if mil.needs_descriptors:
        d = window_descriptors(xr)
        q = torch.zeros(*mask.shape, d.shape[-1] + 1, device=device)
        q[mask] = torch.cat([d, minutes[mask][:, None]], -1)
    xr = xr.float() / 100
    if train and augment_cfg is not None:
        xr = augment_batch(xr, augment_cfg, ch_names)
    x4 = rearrange(xr, "N C (A T) -> N C A T", T=200)
    bag = {"mask": mask, "q": q, "t": minutes}
    if dropper is not None:
        bag["present"] = dropper.present(x4)
        bag["ci_all"] = dropper.ci_all
    return x4, utils.get_channel_indices(ch_names), bag, y.float().to(device), stems


def _bag_loss(criterion, out, y):
    target = y.unsqueeze(-1) if y.dim() == 1 else y
    loss = criterion(out, target)
    if isinstance(loss, LossBreakdown):
        loss = loss.total
    return loss, target


_REF = {}


def _reference_encoder(model):
    """A frozen copy of the model as it was when training started (retention)."""
    key = id(model)
    if key not in _REF:
        ref = copy.deepcopy(getattr(model, "module", model)).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        _REF[key] = ref
    return _REF[key]


# ------------------------------------------------------------------- train
def train_one_epoch_bags(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                         trainer_cfg=None, optim_cfg=None, num_training_steps_per_epoch=None,
                         model_ema=None, log_writer=None, start_steps=0, lr_schedule_values=None,
                         wd_schedule_values=None, ch_names=None, target_stats=None, mixup_cfg=None,
                         augment_cfg=None, channel_drop_cfg=None, mil_cfg=None, **_ignored) -> Dict[str, float]:
    model.train(True)
    m = getattr(model, "module", model)
    dropper = ChannelDropper.from_config(channel_drop_cfg, ch_names)
    update_freq = trainer_cfg.update_freq if trainer_cfg is not None else 1
    clip_grad = getattr(optim_cfg, "clip_grad", None)
    frozen = epoch < mil_cfg.freeze_encoder_epochs
    if frozen:
        logger.info("MIL epoch %d: encoder frozen (pool + head only)", epoch)
    retention = mil_cfg.retention_weight > 0 and not frozen
    ref = _reference_encoder(model) if mil_cfg.retention_weight > 0 else None
    mixup_on = (mixup_cfg is not None and getattr(mixup_cfg, "enabled", False))
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    optimizer.zero_grad()
    std = target_stats[1] if target_stats else 1.0
    for step_i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        step = step_i // update_freq
        if num_training_steps_per_epoch is not None and step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if step_i % update_freq == 0:
            apply_lr_wd_schedule(optimizer, it, lr_schedule_values, wd_schedule_values)
        x4, ci, bag, y, _ = prepare(batch, device, ch_names, dropper, m.mil, augment_cfg, train=True)
        bag["no_grad_encoder"] = frozen
        if mixup_on and np.random.rand() < getattr(mixup_cfg, "prob", 1.0):
            lam = float(np.random.beta(mixup_cfg.alpha, mixup_cfg.alpha))
            perm = torch.randperm(y.shape[0], device=device)
            bag["mix"] = (perm, lam)
            y = lam * y + (1 - lam) * y[perm]
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            out, alpha, h = model(x4, ci, bag=bag)
            loss, target = _bag_loss(criterion, out, y)
            ent = attention_entropy_gap(alpha.float(), bag["mask"])
            if mil_cfg.pool in COMPETING_POOLS and mil_cfg.entropy_weight > 0:
                loss = loss + mil_cfg.entropy_weight * ent
            ret_loss = None
            if retention:
                with torch.no_grad():
                    _, _, h_ref = ref(x4, ci, bag={**bag, "mix": None, "no_grad_encoder": True})
                ret_loss = F.mse_loss(h.float(), h_ref.float())
                loss = loss + mil_cfg.retention_weight * ret_loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.error("Loss is %s, stopping training", loss_value)
            sys.exit(1)
        update = (step_i + 1) % update_freq == 0
        grad_norm = optimizer_update(loss / update_freq, optimizer=optimizer, loss_scaler=loss_scaler,
                                     parameters=model.parameters(), clip_grad=clip_grad,
                                     update_grad=update, model_ema=model_ema, model=model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        # against the (possibly mixed) target the loss saw
        pred = regression_output(criterion, out.detach()).float().reshape(len(y), -1)[:, 0]
        tgt = regression_target(criterion, y.reshape(len(y), -1))[:, 0]
        mae = float((pred - tgt).abs().mean()) * std
        metric_logger.update(loss=loss_value, mae=mae, attn_entropy_gap=float(ent),
                             attn_max=float(alpha.max(-1).values.mean()))
        if ret_loss is not None:
            metric_logger.update(retention_loss=float(ret_loss))
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))
        metric_logger.update(lr=max(g["lr"] for g in optimizer.param_groups))
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(mae=mae, attn_entropy_gap=float(ent), head="mil")
            log_writer.set_step()
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["samples_processed"] = len(data_loader.dataset)
    return stats


# -------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_bags(data_loader, model, device, header='Test:', ch_names=None, metrics=None,
                  eval_cfg=None, log_writer=None, head=None, epoch=None, task='regression',
                  target_stats=None, criterion=None, channel_drop_cfg=None, mil_cfg=None,
                  out_dir=None, nb_classes=1, **_ignored) -> Dict[str, float]:
    from labram.train import train_finetune as tf
    model.eval()
    m = getattr(model, "module", model)
    dropper = ChannelDropper.from_config(channel_drop_cfg, ch_names)
    preds, trues, groups, losses = [], [], [], []
    win = {"recording": [], "slot": [], "minute": [], "alpha": [], "window_pred": []}
    metric_logger = utils.MetricLogger(delimiter="  ")
    for batch in metric_logger.log_every(data_loader, 50, header):
        x4, ci, bag, y, stems = prepare(batch, device, ch_names, dropper, m.mil)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            out, alpha, h = model(x4, ci, bag=bag)
            wout = m.head(h.to(m.head_dtype()))
        if criterion is not None:
            loss, _ = _bag_loss(criterion, out.float(), y)
            losses.append(float(loss))
        preds.append(regression_output(criterion, out.float()).reshape(len(y), -1)[:, 0].cpu()
                     if criterion is not None else out.float()[:, 0].cpu())
        trues.append(regression_target(criterion, y.reshape(len(y), -1))[:, 0].cpu()
                     if criterion is not None else y.reshape(len(y), -1)[:, 0].cpu())
        groups += list(stems)
        mask = bag["mask"]
        b_idx, k_idx = mask.nonzero(as_tuple=True)
        wpred = (regression_output(criterion, wout.float()) if criterion is not None else wout.float())
        win["recording"] += [stems[i] for i in b_idx.tolist()]
        win["slot"] += k_idx.tolist()
        win["minute"] += bag["t"][mask].tolist()
        win["alpha"] += alpha.float()[mask].tolist()
        win["window_pred"] += wpred.reshape(len(b_idx), -1)[:, 0].tolist()
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()
    if tf._is_sharded_loader(data_loader):
        pred, true, groups = utils.gather_sharded_eval(pred, true, groups,
                                                       total=tf._loader_dataset_len(data_loader))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts = [None] * utils.get_world_size()
            torch.distributed.all_gather_object(parts, win)
            win = {k: sum((p[k] for p in parts), []) for k in win}
    pred = utils.denormalize(pred, target_stats)
    true = utils.denormalize(true, target_stats)
    wdf = pd.DataFrame(win).drop_duplicates(["recording", "slot"])
    wdf["window_pred"] = utils.denormalize(wdf["window_pred"].to_numpy(), target_stats)
    age = dict(zip(groups, true))
    wdf["age"] = wdf["recording"].map(age)

    ret, _, _, _, _ = tf._window_and_case_metrics(pred, true, groups, "mean", metrics, False,
                                                  nb_classes, False, task)
    base = getattr(data_loader.dataset, "base", data_loader.dataset)
    ret.update(tf._cohort_metrics(types.SimpleNamespace(dataset=base), pred, true, groups, "mean",
                                  metrics, False, nb_classes, task, log_writer, head, epoch,
                                  per_dataset=eval_cfg is None or getattr(eval_cfg, "per_dataset_metrics", True),
                                  by_label=True))
    wtrue = wdf["age"].to_numpy()
    for k, v in utils.regression_metrics_fn(wdf["window_pred"].to_numpy(), wtrue, ["mae", "rmse", "r2"]).items():
        ret[f"window_{k}"] = v
    mean_pool = wdf.groupby("recording")["window_pred"].mean()
    rec_true = pd.Series(age)
    for k, v in utils.regression_metrics_fn(mean_pool.to_numpy(), rec_true.reindex(mean_pool.index).to_numpy(),
                                            ["mae", "rmse", "r2"]).items():
        ret[f"case_mean_{k}"] = v
    g = wdf.groupby("recording")["alpha"]
    n = g.size()
    ent = g.apply(lambda a: float(-(a * np.log(np.clip(a, 1e-12, None))).sum()))
    ret["attn_entropy_gap"] = float((np.log(n.clip(lower=1)) - ent).mean())
    ret["attn_max"] = float(g.max().mean())
    if losses:
        ret["loss"] = float(np.mean(losses))
    ret["samples_processed"] = len(wdf)
    if (out_dir and mil_cfg is not None and mil_cfg.save_weights and utils.is_main_process()
            and head in ("val", "test")):
        d = os.path.join(out_dir, "analysis")
        os.makedirs(d, exist_ok=True)
        wdf.to_parquet(os.path.join(d, f"{head}_bag_weights_epoch{int(epoch or 0):02d}.parquet"))
    logger.info("%s recording MAE %.3f (window mean-pool %.3f, window %.3f), attention max %.3f",
                header, ret.get("mae", float("nan")), ret["case_mean_mae"], ret["window_mae"], ret["attn_max"])
    return ret
