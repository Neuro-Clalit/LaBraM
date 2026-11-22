# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Auxiliary codebook losses for a plain fine-tune model: the encoder's patch
# tokens are sent through a frozen VQNSP projection layer, codebook and decoder,
# and a quantization and / or an amplitude-reconstruction term is added to the
# task loss (``codebook_aux.*``).
# ---------------------------------------------------------
"""Auxiliary codebook losses (``codebook_aux``).

For the patch tokens :math:`t_j` of a window (after the encoder's final norm),
the frozen projection layer :math:`p`, the frozen codebook :math:`\\{e_k\\}` and
the frozen decoder :math:`d`:

.. math::

    z_j = \\frac{p(t_j)}{\\lVert p(t_j) \\rVert}, \\qquad
    k_j = \\arg\\min_k \\lVert z_j - e_k \\rVert^2, \\qquad
    L_{quant} = \\beta \\, \\operatorname{mean}_{j, c} (z_{j,c} - e_{k_j, c})^2

.. math::

    \\hat a = d(z + \\operatorname{sg}(e_{k} - z)), \\qquad
    L_{recon} = \\operatorname{mean} (\\hat a - a)^2

with :math:`a` the FFT amplitude of the (model-preprocessed) input patch,
standardised per window over channels, patches and the kept bins
(``SpectralReconstructionLoss``), and ``sg`` the stop-gradient. With
``freq_band_hz = [lo, hi]`` the mean is taken over the bins of that band only
(1 Hz per bin for 1 s patches at 200 Hz); the target is still standardised over
all kept bins, so it stays on the scale the frozen decoder was trained for. The
added loss is ``amplitude_weight * L_recon + embedding_weight * L_quant``. Nothing in this
module is trained: the gradient only passes *through* it into the encoder.

The module is held next to the model, not registered in it
(:func:`attach_codebook_aux`), so the model's ``state_dict``, checkpoints and
optimizer groups are exactly those of a run without it.
"""
import math
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model

import labram.utils as utils
from labram.configs.loss_config import LossConfig
from labram.losses.spectral import SpectralReconstructionLoss

logger = utils.get_logger(__name__)

# Rows of the token-to-codebook distance matrix computed at once (x 8192 codes).
_CHUNK = 16384
# LaBraM inputs are resampled to 200 Hz; a patch of T samples has FFT bins
# SAMPLE_RATE_HZ / T apart (1 Hz for the usual 200-sample patch).
SAMPLE_RATE_HZ = 200.0


class CodebookAuxiliary(nn.Module):
    """Frozen projection layer + codebook + decoder of a VQNSP tokenizer, used
    as a loss on encoder patch tokens. Always in eval mode.

    A training step calls :meth:`start`, runs the model (every encoder forward
    with gradient reports its tokens through :meth:`accumulate`), then
    :meth:`finish` for the weighted loss and its logged terms.
    """

    def __init__(self, tokenizer: nn.Module, amplitude_weight: float = 0.0,
                 embedding_weight: float = 0.0, freq_fraction: float = 0.5,
                 monitor_every: int = 20, freq_band_hz: Optional[Sequence[float]] = None):
        super().__init__()
        self.encode_task_layer = tokenizer.encode_task_layer
        self.decoder = tokenizer.decoder
        self.decode_task_layer_magnitude = tokenizer.decode_task_layer_magnitude
        # A plain buffer: no EMA update, no usage statistics, no DDP reduce.
        self.register_buffer("codebook", tokenizer.quantizer.embedding.weight.detach().clone())
        self.beta = float(tokenizer.quantizer.beta)
        self.amplitude_weight = float(amplitude_weight)
        self.embedding_weight = float(embedding_weight)
        self.monitor_every = int(monitor_every)
        self.spectral = SpectralReconstructionLoss(LossConfig(freq_fraction=freq_fraction))
        self.freq_band_hz = tuple(float(f) for f in freq_band_hz) if freq_band_hz else None
        for p in self.parameters():
            p.requires_grad_(False)
        self.active = False
        self._monitor = False
        self._steps = 0
        self._parts = []
        super().train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    # ------------------------------------------------------------------ core
    def quantize(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``patch_tokens`` [B, L, D] -> (straight-through quantized [B, L, Q],
        quantization loss, code indices [B, L]). Same result as
        ``NormEMAVectorQuantizer.forward`` in eval mode, without its one-hot."""
        with torch.amp.autocast(patch_tokens.device.type, enabled=False):
            z = F.normalize(self.encode_task_layer(patch_tokens.float()), p=2, dim=-1)
            flat = z.reshape(-1, z.shape[-1])
            with torch.no_grad():
                code_sq = self.codebook.pow(2).sum(1)
                idx = torch.cat([(code_sq - 2 * flat[i:i + _CHUNK] @ self.codebook.t()).argmin(1)
                                 for i in range(0, flat.shape[0], _CHUNK)])
            z_q = self.codebook[idx].view_as(z)
            loss = self.beta * F.mse_loss(z_q, z)
            quantized = z + (z_q - z).detach()
        return quantized, loss, idx.view(z.shape[:-1])

    def band_bins(self, patch_samples: int, n_bins: int) -> Tuple[int, int]:
        """First and last FFT bin (inclusive) of ``freq_band_hz``."""
        hz_per_bin = SAMPLE_RATE_HZ / patch_samples
        lo = int(math.ceil(self.freq_band_hz[0] / hz_per_bin - 1e-9))
        hi = min(int(math.floor(self.freq_band_hz[1] / hz_per_bin + 1e-9)), n_bins - 1)
        if not 0 <= lo <= hi:
            raise ValueError(f"codebook_aux.freq_band_hz={list(self.freq_band_hz)} selects no bin of "
                             f"the {n_bins} kept ({hz_per_bin:g} Hz per bin)")
        return lo, hi

    def reconstruction_losses(self, quantized: torch.Tensor, x: torch.Tensor,
                              channel_indices) -> Tuple[torch.Tensor, torch.Tensor]:
        """(scored loss, loss over all kept bins) of the decoder's output for
        ``quantized`` [B, L, Q] against the spectrum of ``x`` [B, N, A, T]
        (already preprocessed). The scored loss is the band's when
        ``freq_band_hz`` is set, else the same tensor as the full one."""
        b, n = x.shape[0], x.shape[1]
        grid = quantized.view(b, n, -1, quantized.shape[-1]).permute(0, 3, 1, 2)      # [B, Q, N, A]
        features = self.decoder(grid, channel_indices, return_patch_tokens=True)
        magnitude = self.decode_task_layer_magnitude(features).float()
        target, _ = self.spectral.spectrum_targets(x.float())
        target = rearrange(target, 'b n a c -> b (n a) c')
        squared = (magnitude[..., :target.shape[-1]] - target).pow(2)                  # [B, L, bins]
        full = squared.mean()
        if self.freq_band_hz is None:
            return full, full
        lo, hi = self.band_bins(x.shape[-1], squared.shape[-1])
        return squared[..., lo:hi + 1].mean(), full

    def reconstruction_loss(self, quantized: torch.Tensor, x: torch.Tensor, channel_indices) -> torch.Tensor:
        """The scored amplitude loss (in ``freq_band_hz`` when set)."""
        return self.reconstruction_losses(quantized, x, channel_indices)[0]

    # ------------------------------------------------------- per-step protocol
    def start(self) -> None:
        self._parts = []
        self._monitor = (self.amplitude_weight == 0 and self.monitor_every > 0
                         and self._steps % self.monitor_every == 0)
        self._steps += 1
        self.active = True

    def accumulate(self, patch_tokens: torch.Tensor, x: torch.Tensor, channel_indices) -> None:
        """One encoder forward: ``patch_tokens`` [B, N*A, D] of the preprocessed
        input ``x`` [B, N, A, T]."""
        if self.codebook.device != patch_tokens.device:
            self.to(patch_tokens.device)
        quantized, quant, idx = self.quantize(patch_tokens)
        recon = full = None
        if self.amplitude_weight > 0:
            recon, full = self.reconstruction_losses(quantized, x, channel_indices)
        elif self._monitor:
            with torch.no_grad():
                recon, full = self.reconstruction_losses(quantized, x, channel_indices)
        self._parts.append((idx.numel(), quant, recon, idx.reshape(-1), full))

    def finish(self) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """(weighted loss to add or None, logged terms). Groups of one step
        (channel patterns) are averaged by their number of tokens."""
        self.active = False
        parts, self._parts = self._parts, []
        if not parts:
            return None, {}
        total = sum(p[0] for p in parts)
        quant = sum(p[1] * (p[0] / total) for p in parts)
        stats = {"aux_quant_loss": float(quant.detach()),
                 "aux_codes": float(torch.unique(torch.cat([p[3] for p in parts])).numel())}
        loss = self.embedding_weight * quant if self.embedding_weight > 0 else None
        if parts[0][2] is not None:
            recon = sum(p[2] * (p[0] / total) for p in parts)
            stats["aux_recon_loss"] = float(recon.detach())
            if self.freq_band_hz is not None:
                # all kept bins, for comparison with a run scored on all of them
                stats["aux_recon_full_loss"] = float(sum(p[4].detach() * (p[0] / total) for p in parts))
            if self.amplitude_weight > 0:
                loss = self.amplitude_weight * recon if loss is None else loss + self.amplitude_weight * recon
        if loss is not None:
            stats["aux_loss"] = float(loss.detach())
        return loss, stats


def get_codebook_aux(model: nn.Module) -> Optional[CodebookAuxiliary]:
    """The auxiliary held by ``model`` (DDP-wrapped or not), or None."""
    return getattr(getattr(model, "module", model), "codebook_aux", None)


def attach_codebook_aux(model: nn.Module, cfg) -> Optional[CodebookAuxiliary]:
    """Build the auxiliary of ``cfg`` (a ``CodebookAuxConfig``) and hold it on
    ``model`` as a plain attribute: it is not a submodule, so it stays out of
    ``state_dict()``, ``parameters()`` and ``train()``.

    A missing ``tokenizer_weight`` is only a warning here, because evaluation
    code rebuilds the model from a run config whose weight path existed only in
    the training container; a training run checks for the auxiliary itself.
    """
    from labram.models.vqnsp import load_vqnsp_weights, vqnsp_codebook_shape
    if not (cfg.tokenizer_weight.startswith(("http://", "https://"))
            or os.path.isfile(cfg.tokenizer_weight)):
        logger.warning("codebook_aux: tokenizer weight %r not found; the auxiliary loss is "
                       "not attached (fine for evaluation)", cfg.tokenizer_weight)
        return None
    n_codes, code_dim = vqnsp_codebook_shape(cfg.tokenizer_weight)
    tokenizer = create_model(cfg.tokenizer_model, pretrained=False,
                             num_codebook_tokens=n_codes, quantizer_dim=code_dim)
    load_vqnsp_weights(tokenizer, cfg.tokenizer_weight)
    if tokenizer.encoder.embed_dim != model.embed_dim:
        raise ValueError(f"codebook_aux: the tokenizer's projection layer takes {tokenizer.encoder.embed_dim}-d "
                         f"tokens, the model's are {model.embed_dim}-d")
    aux = CodebookAuxiliary(tokenizer, amplitude_weight=cfg.amplitude_weight,
                            embedding_weight=cfg.embedding_weight,
                            freq_fraction=cfg.freq_fraction, monitor_every=cfg.monitor_every,
                            freq_band_hz=getattr(cfg, "freq_band_hz", None))
    model.__dict__["codebook_aux"] = aux
    logger.info("codebook_aux attached from %s: amplitude_weight=%g embedding_weight=%g freq_fraction=%g "
                "freq_band_hz=%s (%d codes x %d, all frozen)", cfg.tokenizer_weight, cfg.amplitude_weight,
                cfg.embedding_weight, cfg.freq_fraction, list(aux.freq_band_hz or []) or "all kept bins",
                n_codes, code_dim)
    return aux
