# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# FFT amplitude + phase reconstruction loss for the VQNSP tokenizer.
# ---------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from labram.losses.config import LossConfig


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
        """``x``: [B, N, A, T] -> (amplitude, phase), each std-normalised."""
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = self.std_norm(torch.abs(x_fft))
        phase = self.std_norm(torch.angle(x_fft))
        return amplitude, phase

    def reconstruction_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = rearrange(target, 'b n a c -> b (n a) c')
        return self._loss_fn(reconstructed, target)

    def forward(self, reconstructed_amplitude, reconstructed_angle, amplitude_target, angle_target):
        amplitude_loss = self.reconstruction_loss(reconstructed_amplitude, amplitude_target)
        angle_loss = self.reconstruction_loss(reconstructed_angle, angle_target)
        return amplitude_loss, angle_loss
