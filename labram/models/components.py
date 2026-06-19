# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Shared encode/decode building blocks for the VQNSP tokenizer and the
# codebook-regularized fine-tuning classifier. Keeping the "patch-tokens ->
# quantize -> decode -> magnitude/phase" path in one place means both models
# share a single source of truth.
# ---------------------------------------------------------

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


def quantize_patch_tokens(
    encode_task_layer: nn.Module,
    quantizer: nn.Module,
    patch_tokens: torch.Tensor,
    num_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project encoder patch tokens into the quantizer space and quantize them.

    ``patch_tokens``: ``[B, N*A, enc_dim]`` (encoder output with the CLS token
    already removed). The quantizer projection runs with autocast disabled to
    match the original VQNSP numerics.

    Returns ``(quantized, quantize_loss, codebook_indices)`` where ``quantized``
    has shape ``[B, quantizer_dim, N, A]``.
    """
    with torch.amp.autocast(patch_tokens.device.type, enabled=False):
        to_quantizer_features = encode_task_layer(
            patch_tokens.type_as(encode_task_layer[-1].weight))

    num_tokens = to_quantizer_features.shape[1]
    h, w = num_channels, num_tokens // num_channels
    to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w)
    quantized, quantize_loss, codebook_indices = quantizer(to_quantizer_features)
    return quantized, quantize_loss, codebook_indices


def decode_spectrum(
    decoder: nn.Module,
    magnitude_head: nn.Module,
    phase_head: nn.Module,
    quantized: torch.Tensor,
    channel_indices=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode quantized features into reconstructed magnitude & phase spectra.

    ``quantized``: ``[B, quantizer_dim, N, A]``. Returns
    ``(reconstructed_magnitude, reconstructed_phase)``.
    """
    decoder_features = decoder(quantized, channel_indices, return_patch_tokens=True)
    reconstructed_magnitude = magnitude_head(decoder_features)
    reconstructed_phase = phase_head(decoder_features)
    return reconstructed_magnitude, reconstructed_phase
