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

    # Downstream regression criterion (used when the task is regression, e.g.
    # brain-age prediction): "mse", "l1" or "huber". Huber is the default because
    # clinical age distributions are long-tailed at both ends, and it is less
    # dominated by the extremes than a plain squared error.
    regression_loss: str = "huber"
    huber_delta: float = 1.0

    # ``regression_loss = "soft_label"`` (SFCN-style, Peng et al. 2021): the head
    # predicts a distribution over age bins of ``soft_label_bin_width`` years
    # from ``soft_label_min`` to ``soft_label_max``; the target is a Gaussian of
    # width ``soft_label_sigma`` (years) over the bins and the loss is their KL
    # divergence. The scalar prediction is the expectation over the bins.
    soft_label_sigma: float = 2.5
    soft_label_min: float = 1.0
    soft_label_max: float = 89.0
    soft_label_bin_width: float = 1.0

    # Age-balanced regression loss (counters regression to the mean): each
    # window's mse/l1/huber term is weighted by the inverse ("inverse") or the
    # inverse square root ("sqrt_inverse") of the train-split target density,
    # estimated over ``balance_bin_width``-year bins and smoothed with a
    # Gaussian kernel of ``balance_lds_sigma`` years (LDS, Yang et al. 2021;
    # 0 = raw histogram). Weights are clipped at ``balance_max_weight`` times
    # the smallest and renormalized to mean 1 over the train windows, so the
    # loss scale matches the unweighted run. "none" disables it. Train-only:
    # val/test losses stay unweighted.
    balance: str = "none"
    balance_bin_width: float = 1.0
    balance_lds_sigma: float = 2.0
    balance_max_weight: float = 10.0

    # Codebook-regularized fine-tuning: weight on the classification term when
    # the spectral (amplitude/phase) and quantization losses regularize the
    # downstream task. Reuses amplitude_weight / phase_weight / embedding_weight
    # for the auxiliary terms.
    classifier_weight: float = 1.0
