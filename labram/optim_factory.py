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
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def summarize_trainable_parameters(model):
    """Split ``model``'s parameters into trainable vs frozen.

    Returns a dict with ``trainable_layers`` / ``frozen_layers`` (unique
    layer names — the parameter path minus the ``.weight``/``.bias`` leaf, so a
    layer isn't listed twice) and ``n_trainable`` / ``n_frozen`` parameter
    counts. Frozen layers are exactly those excluded from the optimizer by
    :func:`get_parameter_groups`.
    """
    trainable_layers, frozen_layers = [], []
    seen_train, seen_frozen = set(), set()
    n_trainable = n_frozen = 0
    for name, param in model.named_parameters():
        layer = name.rsplit('.', 1)[0] if '.' in name else name
        if param.requires_grad:
            n_trainable += param.numel()
            if layer not in seen_train:
                seen_train.add(layer)
                trainable_layers.append(layer)
        else:
            n_frozen += param.numel()
            if layer not in seen_frozen:
                seen_frozen.add(layer)
                frozen_layers.append(layer)
    return {
        'trainable_layers': trainable_layers,
        'frozen_layers': frozen_layers,
        'n_trainable': n_trainable,
        'n_frozen': n_frozen,
    }


def log_trainable_parameters(model, logger=None):
    """Log the trainable/frozen split and the names of the frozen (non-trainable)
    layers — i.e. exactly the layers that are dropped from the optimizer.

    ``n_last_trainable_layers`` (and ``trainable=False``) freeze parameters by
    setting ``requires_grad=False``; :func:`get_parameter_groups` then omits
    them from the optimizer. This makes that outcome visible in the run log.
    """
    import labram.utils as _utils
    logger = logger or _utils.get_logger(__name__)
    s = summarize_trainable_parameters(model)
    total = s['n_trainable'] + s['n_frozen']
    pct = 100.0 * s['n_trainable'] / total if total else 0.0
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%); frozen: %d params across %d layers",
        s['n_trainable'], total, pct, s['n_frozen'], len(s['frozen_layers']))
    if s['frozen_layers']:
        logger.info("Non-trainable (frozen, excluded from optimizer) layers:")
        for layer in s['frozen_layers']:
            logger.info("  [frozen] %s", layer)
    else:
        logger.info("All layers are trainable (no parameters frozen).")
    return s


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(optim_cfg, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = optim_cfg.opt.lower()
    weight_decay = optim_cfg.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=optim_cfg.lr, weight_decay=weight_decay)
    if optim_cfg.opt_eps is not None:
        opt_args['eps'] = optim_cfg.opt_eps
    if optim_cfg.opt_betas is not None:
        opt_args['betas'] = optim_cfg.opt_betas

    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=optim_cfg.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=optim_cfg.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=optim_cfg.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not optim_cfg.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=optim_cfg.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=optim_cfg.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=optim_cfg.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=optim_cfg.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


# ---------------------------------------------------------------------------
# Per-step optimizer helpers (LR/WD scheduling, grad-step, metric logging)
# ---------------------------------------------------------------------------


def optimizer_update(
    loss: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Any,
    parameters: Iterable[torch.nn.Parameter],
    clip_grad: Optional[float],
    update_grad: bool,
    create_graph: bool = False,
    model_ema: Optional[Any] = None,
    model: Optional[torch.nn.Module] = None,
) -> Optional[float]:
    """Run the AMP-scaled backward + (clipped) optimizer step for one micro-step.

    Mirrors the inline stepping previously embedded in the fine-tune loop:
    scale -> backward -> (clip) -> step via ``loss_scaler``; on an update
    boundary (``update_grad``) zero the gradients and advance the EMA model.
    Returns the grad-norm reported by the scaler (``None`` between boundaries).
    """
    grad_norm = loss_scaler(
        loss, optimizer, clip_grad=clip_grad, parameters=parameters,
        create_graph=create_graph, update_grad=update_grad)
    if update_grad:
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
    return grad_norm


def compute_component_grad_norms(
    components: dict,
    parameters: Iterable[torch.nn.Parameter],
    norm_type: float = 2.0,
) -> dict:
    """Gradient norm each loss *component* contributes, for logging.

    For a composite criterion (e.g. classification + spectral + quantization),
    ``components`` maps a name to its (unweighted) scalar loss tensor. We take
    ``autograd.grad`` of each component w.r.t. ``parameters`` with
    ``retain_graph=True`` so the caller's subsequent full backward still works,
    and return ``{'grad_norm_<name>': float}``. This costs one extra backward
    per component, so callers should invoke it only periodically.
    """
    params = [p for p in parameters if p.requires_grad]
    out = {}
    for name, loss in components.items():
        # A component with no path to the parameters (e.g. a constant/detached
        # term) has no gradient to report — record 0.0 rather than raising.
        if not (torch.is_tensor(loss) and loss.requires_grad):
            out[f'grad_norm_{name}'] = 0.0
            continue
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().norm(norm_type).item()) ** norm_type
        out[f'grad_norm_{name}'] = total ** (1.0 / norm_type)
    return out


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
    if grad_norm is not None and not math.isfinite(grad_norm):
        grad_norm = None
    metric_logger.update(grad_norm=grad_norm)
    if log_writer is not None:
        # lr/min_lr/weight_decay live on a small (<= ~1) scale; keep them together
        # on the "opt" plot. grad_norm can be >> 1 and would flatten the others on
        # a shared normalized axis, so it gets its own "grad" plot.
        log_writer.update(lr=max_lr, head="opt")
        log_writer.update(min_lr=min_lr, head="opt")
        log_writer.update(weight_decay=weight_decay_value, head="opt")
        log_writer.update(grad_norm=grad_norm, head="grad")
