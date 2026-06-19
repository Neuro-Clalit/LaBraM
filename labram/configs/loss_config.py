# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Configurable weights and options for the training losses.
# ---------------------------------------------------------

from dataclasses import dataclass


@dataclass
class LossConfig:
    """Weights and options for LaBraM training losses.

    The defaults reproduce the original hard-coded behaviour exactly:
      * VQNSP total loss = embedding + amplitude + phase (equal weight 1.0),
      * MSE reconstruction (``use_smooth_l1=False``),
      * VQ commitment beta = 1.0,
      * no label smoothing on the downstream classification criterion.
    """

    # VQNSP tokenizer reconstruction (spectral) weights.
    amplitude_weight: float = 1.0
    phase_weight: float = 1.0
    embedding_weight: float = 1.0
    use_smooth_l1: bool = False

    # Vector-quantizer commitment loss.
    vq_commitment_beta: float = 1.0

    # Downstream classification criterion.
    classification_label_smoothing: float = 0.0
