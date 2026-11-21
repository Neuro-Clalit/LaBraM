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
(``SpectralReconstructionLoss``), and ``sg`` the stop-gradient. The added loss
is ``amplitude_weight * L_recon + embedding_weight * L_quant``. Nothing in this
module is trained: the gradient only passes *through* it into the encoder.

The module is held next to the model, not registered in it
(:func:`attach_codebook_aux`), so the model's ``state_dict``, checkpoints and
optimizer groups are exactly those of a run without it.
"""
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

import labram.utils as utils
from labram.configs.loss_config import LossConfig
from labram.losses.spectral import SpectralReconstructionLoss

logger = utils.get_logger(__name__)

# Rows of the token-to-codebook distance matrix computed at once (x 8192 codes).
_CHUNK = 16384


class CodebookAuxiliary(nn.Module):
    """Frozen projection layer + codebook + decoder of a VQNSP tokenizer, used
    as a loss on encoder patch tokens. Always in eval mode.

    A training step calls :meth:`start`, runs the model (every encoder forward
    with gradient reports its tokens through :meth:`accumulate`), then
    :meth:`finish` for the weighted loss and its logged terms.
    """

    def __init__(self, tokenizer: nn.Module, amplitude_weight: float = 0.0,
                 embedding_weight: float = 0.0, freq_fraction: float = 0.5,
                 monitor_every: int = 20):
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

    def reconstruction_loss(self, quantized: torch.Tensor, x: torch.Tensor, channel_indices) -> torch.Tensor:
        """Amplitude loss of the decoder's output for ``quantized`` [B, L, Q]
        against the spectrum of ``x`` [B, N, A, T] (already preprocessed)."""
        b, n = x.shape[0], x.shape[1]
        grid = quantized.view(b, n, -1, quantized.shape[-1]).permute(0, 3, 1, 2)      # [B, Q, N, A]
        features = self.decoder(grid, channel_indices, return_patch_tokens=True)
        magnitude = self.decode_task_layer_magnitude(features)
        target, _ = self.spectral.spectrum_targets(x.float())
        return self.spectral.reconstruction_loss(magnitude.float(), target)

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
        recon = None
        if self.amplitude_weight > 0:
            recon = self.reconstruction_loss(quantized, x, channel_indices)
        elif self._monitor:
            with torch.no_grad():
                recon = self.reconstruction_loss(quantized, x, channel_indices)
        self._parts.append((idx.numel(), quant, recon, idx.reshape(-1)))

    def finish(self) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """(weighted loss to add or None, logged terms). Groups of one step
        (channel patterns) are averaged by their number of tokens."""
        self.active = False
        parts, self._parts = self._parts, []
        if not parts:
            return None, {}
        total = sum(n for n, *_ in parts)
        quant = sum(q * (n / total) for n, q, _, _ in parts)
        stats = {"aux_quant_loss": float(quant.detach()),
                 "aux_codes": float(torch.unique(torch.cat([i for *_, i in parts])).numel())}
        loss = self.embedding_weight * quant if self.embedding_weight > 0 else None
        if parts[0][2] is not None:
            recon = sum(r * (n / total) for n, _, r, _ in parts)
            stats["aux_recon_loss"] = float(recon.detach())
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
                            freq_fraction=cfg.freq_fraction, monitor_every=cfg.monitor_every)
    model.__dict__["codebook_aux"] = aux
    logger.info("codebook_aux attached from %s: amplitude_weight=%g embedding_weight=%g freq_fraction=%g "
                "(%d codes x %d, all frozen)", cfg.tokenizer_weight, cfg.amplitude_weight,
                cfg.embedding_weight, cfg.freq_fraction, n_codes, code_dim)
    return aux
