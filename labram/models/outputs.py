# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Structured prediction container for the codebook-regularized classifier.
# Models predict; loss modules consume this. No loss math lives here.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PredictorOutput:
    """Predictions emitted by ``CodebookRegularizedClassifier``.

    ``logits`` is always present. The reconstruction fields are populated only
    when the regularization path runs (training); they are ``None`` on a
    classification-only forward (evaluation). ``x_patched`` carries the patched
    input ``[B, N, A, T]`` so the loss module can build the FFT spectral targets
    without recomputing the rearrange.
    """

    logits: torch.Tensor
    recon_magnitude: Optional[torch.Tensor] = None
    recon_phase: Optional[torch.Tensor] = None
    quantize_loss: Optional[torch.Tensor] = None
    x_patched: Optional[torch.Tensor] = None
