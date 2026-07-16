# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# LaBraM++ training option: the signal-processing + loss improvements from
# "Advancing Brainwave Modeling with a Codebook-Based Foundation Model"
# (arXiv:2505.16724).
# ---------------------------------------------------------

from dataclasses import dataclass

from labram.configs.base_configs import ConfigBase

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

    The ``use_*`` / ``resolved_phase_loss`` helpers fold the master switch into
    each sub-feature so callers only ever consult a single effective value.
    """

    enabled: bool = False
    common_average_reference: bool = True
    z_score_patches: bool = True
    z_score_eps: float = 1e-5
    phase_loss: str = PHASE_LOSS_SINCOS  # "sincos" (LaBraM++) | "angle" (original)

    def __post_init__(self) -> None:
        if self.phase_loss not in PHASE_LOSS_CHOICES:
            raise ValueError(
                f"phase_loss must be one of {PHASE_LOSS_CHOICES}; got {self.phase_loss!r}")

    @property
    def use_car(self) -> bool:
        """Whether Common Average Reference is applied to model inputs."""
        return self.enabled and self.common_average_reference

    @property
    def use_z_score(self) -> bool:
        """Whether per-patch z-scoring is applied to model inputs."""
        return self.enabled and self.z_score_patches

    @property
    def resolved_phase_loss(self) -> str:
        """Effective VQNSP phase-loss mode ('angle' unless LaBraM++ is enabled)."""
        return self.phase_loss if self.enabled else PHASE_LOSS_ANGLE

    @property
    def preprocesses_input(self) -> bool:
        """True when any input preprocessing (CAR or z-score) is active."""
        return self.use_car or self.use_z_score
