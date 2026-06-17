# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Downstream classification criterion selection.
# ---------------------------------------------------------

from typing import Optional

import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy

from labram.losses.config import LossConfig


def build_classification_criterion(nb_classes: int, cfg: Optional[LossConfig] = None) -> nn.Module:
    """Select the downstream classification criterion.

    Reproduces the original runner logic:
      * binary (``nb_classes == 1``)        -> ``BCEWithLogitsLoss``
      * label smoothing > 0                 -> timm ``LabelSmoothingCrossEntropy``
      * otherwise                           -> ``CrossEntropyLoss``
    """
    cfg = cfg or LossConfig()
    if nb_classes == 1:
        return nn.BCEWithLogitsLoss()
    if cfg.classification_label_smoothing > 0.:
        return LabelSmoothingCrossEntropy(smoothing=cfg.classification_label_smoothing)
    return nn.CrossEntropyLoss()
