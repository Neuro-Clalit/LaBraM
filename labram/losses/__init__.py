# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training losses and their configuration.
# ---------------------------------------------------------

from labram.losses.classification import build_classification_criterion
from labram.configs.loss_config import LossConfig
from labram.losses.codebook_regularized import CodebookRegularizedCriterion
from labram.losses.outputs import LossBreakdown
from labram.losses.spectral import SpectralReconstructionLoss
from labram.losses.vqnsp import get_vqnsp_losses


__all__ = [
    'CodebookRegularizedCriterion',
    'LossBreakdown',
    'LossConfig',
    'SpectralReconstructionLoss',
    'build_classification_criterion',
    'get_vqnsp_losses',
]
