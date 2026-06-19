# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Composite criterion for codebook-regularized fine-tuning: classification loss
# + spectral reconstruction (amplitude/phase) + quantization commitment loss.
# ---------------------------------------------------------

from typing import Optional

import torch.nn as nn

from labram.configs.loss_config import LossConfig
from labram.losses.outputs import LossBreakdown
from labram.losses.spectral import SpectralReconstructionLoss


class CodebookRegularizedCriterion(nn.Module):
    """Combine a classification loss with VQNSP regularization losses.

    The model produces a :class:`~labram.models.outputs.PredictorOutput`; this
    module owns *all* loss math and weighting. The weighted total is

        ``classifier_weight * L_cls
          + amplitude_weight * L_magnitude
          + phase_weight     * L_phase
          + embedding_weight * L_quantize``

    The reconstruction and quantization terms are only added when present on the
    output (so a classification-only forward yields just the classifier term).
    """

    def __init__(self, classification_criterion: nn.Module, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.classification_criterion = classification_criterion
        self.spectral = SpectralReconstructionLoss(self.cfg)

    def forward(self, output, target) -> LossBreakdown:
        cfg = self.cfg
        cls_loss = self.classification_criterion(output.logits, target)
        components = {'classifier': cls_loss}
        total = cfg.classifier_weight * cls_loss

        if output.recon_magnitude is not None and output.x_patched is not None:
            amp_target, phase_target = self.spectral.spectrum_targets(output.x_patched)
            magnitude_loss, phase_loss = self.spectral(
                output.recon_magnitude, output.recon_phase, amp_target, phase_target)
            components['magnitude'] = magnitude_loss
            components['phase'] = phase_loss
            total = total + cfg.amplitude_weight * magnitude_loss + cfg.phase_weight * phase_loss

        if output.quantize_loss is not None:
            components['quantize'] = output.quantize_loss
            total = total + cfg.embedding_weight * output.quantize_loss

        return LossBreakdown(total=total, components=components)
