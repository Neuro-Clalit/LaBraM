# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# VQNSP loss aggregation: combine component losses into a weighted total.
# ---------------------------------------------------------

from typing import Dict, Optional

import torch

from labram.configs.loss_config import LossConfig


def get_vqnsp_losses(
    embedding_loss: torch.Tensor,
    amplitude_loss: torch.Tensor,
    angle_loss: torch.Tensor,
    cfg: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Combine the VQNSP component losses into a weighted dict.

    With the default :class:`LossConfig` (all weights 1.0) the total equals
    ``embedding_loss + amplitude_loss + angle_loss`` exactly, matching the
    original inline computation.
    """
    cfg = cfg or LossConfig()
    total = (
        cfg.embedding_weight * embedding_loss
        + cfg.amplitude_weight * amplitude_loss
        + cfg.phase_weight * angle_loss
    )
    return {
        'embedding': embedding_loss,
        'amplitude': amplitude_loss,
        'phase': angle_loss,
        'total': total,
    }
