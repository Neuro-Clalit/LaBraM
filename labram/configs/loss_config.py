# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Configurable weights and options for the training losses.
# ---------------------------------------------------------

from dataclasses import dataclass

from labram.configs.base_configs import ConfigBase


@dataclass
class LossConfig(ConfigBase):
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

    # Fraction of FFT frequency bins used by SpectralReconstructionLoss.
    # 1.0 = full spectrum; 0.5 = low half only.  Must be in (0, 1].
    freq_fraction: float = 1.0

    # VQNSP phase reconstruction mode (SpectralReconstructionLoss):
    #   "angle"  -- original LaBraM loss on the std-normalised raw angle,
    #   "sincos" -- LaBraM++ circular loss on (sin phi, cos phi), which removes
    #               the +/-pi wrap-around discontinuity of the raw-angle loss.
    phase_loss: str = "angle"

    # Downstream classification criterion.
    classification_label_smoothing: float = 0.0

    # Codebook-regularized fine-tuning: weight on the classification term when
    # the spectral (amplitude/phase) and quantization losses regularize the
    # downstream task. Reuses amplitude_weight / phase_weight / embedding_weight
    # for the auxiliary terms.
    classifier_weight: float = 1.0
