# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training losses and their configuration.
# ---------------------------------------------------------

from labram.losses.classification import build_classification_criterion
from labram.configs.loss_config import LossConfig
from labram.losses.codebook_regularized import CodebookRegularizedCriterion
from labram.losses.outputs import LossBreakdown
from labram.losses.regression import (
    AbnormalConditionedAgeCriterion,
    SessionFeatureAgeCriterion,
    BalancedRegressionLoss,
    MultiTaskAgeCriterion,
    SoftLabelRegressionLoss,
    age_balance_weights,
    build_downstream_criterion,
    build_regression_criterion,
    model_condition,
    regression_output,
    regression_target,
    soft_label_n_bins,
)
from labram.losses.spectral import SpectralReconstructionLoss
from labram.losses.vqnsp import get_vqnsp_losses


__all__ = [
    'AbnormalConditionedAgeCriterion',
    'SessionFeatureAgeCriterion',
    'BalancedRegressionLoss',
    'CodebookRegularizedCriterion',
    'LossBreakdown',
    'LossConfig',
    'MultiTaskAgeCriterion',
    'SoftLabelRegressionLoss',
    'SpectralReconstructionLoss',
    'age_balance_weights',
    'build_classification_criterion',
    'build_downstream_criterion',
    'build_regression_criterion',
    'get_vqnsp_losses',
    'model_condition',
    'regression_output',
    'regression_target',
    'soft_label_n_bins',
]
