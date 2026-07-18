# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# FFT amplitude + phase reconstruction loss for the VQNSP tokenizer.
# ---------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from labram.configs.loss_config import LossConfig


class SpectralReconstructionLoss(nn.Module):
    """FFT amplitude + phase reconstruction loss for the VQNSP tokenizer.

    Mirrors the original inline VQNSP loss: take the FFT of the raw signal,
    std-normalise the amplitude and phase spectra, and compare them with the
    decoder reconstructions using MSE (or smooth-L1 when ``cfg.use_smooth_l1``).

    Parameter-free, so adding it as a submodule contributes no entries to the
    owning module's ``state_dict``.
    """

    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self._loss_fn = F.smooth_l1_loss if self.cfg.use_smooth_l1 else F.mse_loss

    @staticmethod
    def std_norm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        return (x - mean) / std

    def spectrum_targets(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: [B, N, A, T] -> (amplitude, phase) targets.

        Amplitude is always std-normalised. The phase target depends on
        ``cfg.phase_loss``: ``"angle"`` (original) std-normalises the raw angle,
        while ``"sincos"`` (LaBraM++) returns the raw angle in radians -- the
        circular ``sin``/``cos`` transform is applied inside the loss so the
        target keeps its natural +/-pi range.

        When ``cfg.freq_fraction < 1``, only the first ``ceil(T * freq_fraction)``
        FFT bins are kept, focusing the loss on low frequencies.
        """
        x_fft = torch.fft.fft(x, dim=-1)
        if self.cfg.freq_fraction < 1.0:
            n_freq = max(1, round(x_fft.shape[-1] * self.cfg.freq_fraction))
            x_fft = x_fft[..., :n_freq]
        amplitude = self.std_norm(torch.abs(x_fft))
        angle = torch.angle(x_fft)
        phase = angle if self.cfg.phase_loss == "sincos" else self.std_norm(angle)
        return amplitude, phase

    def reconstruction_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = rearrange(target, 'b n a c -> b (n a) c')
        return self._loss_fn(reconstructed[..., :target.shape[-1]], target)

    def phase_reconstruction_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Phase reconstruction term.

        In ``"angle"`` mode this is a plain MSE/smooth-L1 on the (std-normalised)
        angle, identical to :meth:`reconstruction_loss`. In ``"sincos"`` mode the
        loss is ``||sin(phi_hat) - sin(phi)||^2 + ||cos(phi_hat) - cos(phi)||^2``
        (LaBraM++), a smooth circular objective with no +/-pi discontinuity.
        ``reconstructed`` is the decoder's predicted phase ``phi_hat`` in radians.
        """
        if self.cfg.phase_loss != "sincos":
            return self.reconstruction_loss(reconstructed, target)
        target = rearrange(target, 'b n a c -> b (n a) c')
        reconstructed = reconstructed[..., :target.shape[-1]]
        return (self._loss_fn(torch.sin(reconstructed), torch.sin(target))
                + self._loss_fn(torch.cos(reconstructed), torch.cos(target)))

    def forward(self, reconstructed_amplitude, reconstructed_angle, amplitude_target, angle_target):
        amplitude_loss = self.reconstruction_loss(reconstructed_amplitude, amplitude_target)
        angle_loss = self.phase_reconstruction_loss(reconstructed_angle, angle_target)
        return amplitude_loss, angle_loss
