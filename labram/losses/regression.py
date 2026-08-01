# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Downstream regression criterion selection (e.g. brain-age prediction).
# ---------------------------------------------------------

from typing import Optional

import torch.nn as nn

from labram.configs.loss_config import LossConfig
from labram.losses.classification import build_classification_criterion

REGRESSION_LOSSES = ("mse", "l1", "huber")


def build_regression_criterion(cfg: Optional[LossConfig] = None) -> nn.Module:
    """Select the downstream regression criterion.

      * ``"mse"``   -> ``nn.MSELoss``
      * ``"l1"``    -> ``nn.L1Loss``
      * ``"huber"`` -> ``nn.HuberLoss`` (default; robust to the long tails of a
        clinical age distribution)
    """
    cfg = cfg or LossConfig()
    name = (cfg.regression_loss or "huber").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta)
    raise ValueError(
        f"Unknown regression_loss {cfg.regression_loss!r} (expected one of {REGRESSION_LOSSES})")


def build_downstream_criterion(
    task: str,
    nb_classes: int,
    cfg: Optional[LossConfig] = None,
) -> nn.Module:
    """Criterion for the downstream head, dispatched on the task.

    Single entry point for both the training and evaluation paths: ``evaluate``
    rebuilds its own criterion, and routing both through here is what stops a
    regression run from silently scoring ages with binary cross-entropy.
    """
    if task == "regression":
        return build_regression_criterion(cfg)
    return build_classification_criterion(nb_classes, cfg)
