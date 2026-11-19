# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# LaBraM++ training option: the signal-processing + loss improvements from
# "Advancing Brainwave Modeling with a Codebook-Based Foundation Model"
# (arXiv:2505.16724).
# ---------------------------------------------------------

from dataclasses import dataclass, field
from typing import List

from labram.configs.base_configs import ConfigBase

# Allowed values for LaBraMPlusConfig.amplitude_norm.
AMPLITUDE_NORM_CHOICES = ("", "window")

# Allowed values for LaBraMPlusConfig.phase_loss.
PHASE_LOSS_ANGLE = "angle"
PHASE_LOSS_SINCOS = "sincos"
PHASE_LOSS_CHOICES = (PHASE_LOSS_ANGLE, PHASE_LOSS_SINCOS)


@dataclass
class LaBraMPlusConfig(ConfigBase):
    """Opt-in bundle of the LaBraM++ improvements over the original LaBraM.

    The single master switch is :attr:`enabled`. When it is ``False`` (the
    default) every LaBraM++ behaviour is off and the model / losses reproduce
    the original LaBraM exactly, so existing checkpoints and configs are
    unaffected. When ``enabled`` is ``True`` the individual sub-features (each
    defaulting to the LaBraM++ setting) take effect; any one can be turned off
    independently to ablate it.

    The three improvements, all grounded in EEG signal processing:

    * **Common Average Reference (CAR)** -- per patch, subtract the across-channel
      mean to suppress noise shared by every electrode.
    * **Per-patch z-scoring** -- standardise each (channel, patch) window over
      time so amplitude differences across channels do not dominate.
    * **Sine/cosine phase reconstruction** -- the VQNSP tokenizer's phase loss is
      computed on ``(sin phi, cos phi)`` instead of the raw angle, removing the
      +/-pi wrap-around discontinuity of the original ``||phi_hat - phi||^2``
      loss and giving a smooth, circular optimisation target.

    Two device-robustness options (off by default; ``docs/chbp_domain_shift.md``):

    * ``band_hz = [lo, hi]`` -- zero-phase band-limit of every input window (the
      squared magnitude of an order-8 Butterworth high-pass ``lo`` and low-pass
      ``hi``, i.e. a forward-backward filter), so the model only sees a band that
      every recording chain delivers (e.g. ``[1, 45]``).
    * ``amplitude_norm = "window"`` -- rescale every input window so that the
      median over channels of the channel std equals ``amplitude_target`` (model
      units, µV / 100; 0.1 = 10 µV): removes a global gain (amplifier, export
      scale, electrodes) while keeping topography and spectral shape.

    Order: CAR, band-limit, amplitude normalization, z-scoring.

    The ``use_*`` / ``resolved_phase_loss`` helpers fold the master switch into
    each sub-feature so callers only ever consult a single effective value.
    """

    enabled: bool = False
    common_average_reference: bool = True
    z_score_patches: bool = True
    z_score_eps: float = 1e-5
    phase_loss: str = PHASE_LOSS_SINCOS  # "sincos" (LaBraM++) | "angle" (original)
    band_hz: List[float] = field(default_factory=list)  # [] = off, else [lo, hi] Hz
    amplitude_norm: str = ""             # "" (off) | "window"
    amplitude_target: float = 0.1        # model units (µV / 100): 0.1 = 10 µV

    def __post_init__(self) -> None:
        if self.phase_loss not in PHASE_LOSS_CHOICES:
            raise ValueError(
                f"phase_loss must be one of {PHASE_LOSS_CHOICES}; got {self.phase_loss!r}")
        if self.amplitude_norm not in AMPLITUDE_NORM_CHOICES:
            raise ValueError(
                f"amplitude_norm must be one of {AMPLITUDE_NORM_CHOICES}; got {self.amplitude_norm!r}")
        if self.band_hz and (len(self.band_hz) != 2 or not 0 <= self.band_hz[0] < self.band_hz[1]):
            raise ValueError(f"band_hz must be [] or [lo, hi] with 0 <= lo < hi; got {self.band_hz!r}")

    @property
    def use_car(self) -> bool:
        """Whether Common Average Reference is applied to model inputs."""
        return self.enabled and self.common_average_reference

    @property
    def use_z_score(self) -> bool:
        """Whether per-patch z-scoring is applied to model inputs."""
        return self.enabled and self.z_score_patches

    @property
    def use_band(self) -> bool:
        """Whether model inputs are band-limited to ``band_hz``."""
        return self.enabled and bool(self.band_hz)

    @property
    def use_amplitude_norm(self) -> bool:
        """Whether every input window is rescaled to ``amplitude_target``."""
        return self.enabled and bool(self.amplitude_norm)

    @property
    def resolved_phase_loss(self) -> str:
        """Effective VQNSP phase-loss mode ('angle' unless LaBraM++ is enabled)."""
        return self.phase_loss if self.enabled else PHASE_LOSS_ANGLE

    @property
    def preprocesses_input(self) -> bool:
        """True when any input preprocessing (CAR, band, amplitude or z-score) is active."""
        return self.use_car or self.use_z_score or self.use_band or self.use_amplitude_norm
