# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Shared encode/decode behaviour for the VQNSP tokenizer and the codebook-
# regularized fine-tuning classifier, factored into a mixin both inherit so the
# "patch-tokens -> quantize -> decode -> magnitude/phase" path has a single
# source of truth.
# ---------------------------------------------------------

from typing import Tuple

import torch
from einops import rearrange


class QuantizeDecodeMixin:
    """Quantize/decode methods shared by :class:`VQNSP` and
    :class:`CodebookRegularizedClassifier`.

    The host ``nn.Module`` must register these submodules under exactly these
    names so their ``state_dict`` keys stay identical across both classes (this
    is what lets a VQNSP checkpoint's quantizer/decoder graft onto the
    classifier, and what the checkpoint key-remap targets):

      * ``encode_task_layer``
      * ``quantizer``
      * ``decoder``
      * ``decode_task_layer_magnitude`` / ``decode_task_layer_phase``
    """

    def quantize_patch_tokens(
        self, patch_tokens: torch.Tensor, num_channels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project encoder patch tokens into the quantizer space and quantize.

        ``patch_tokens``: ``[B, N*A, enc_dim]`` (CLS already removed). The
        projection runs with autocast disabled to match the original VQNSP
        numerics. Returns ``(quantized [B, quantizer_dim, N, A], quantize_loss,
        codebook_indices)``.
        """
        with torch.amp.autocast(patch_tokens.device.type, enabled=False):
            to_quantizer_features = self.encode_task_layer(
                patch_tokens.type_as(self.encode_task_layer[-1].weight))

        num_tokens = to_quantizer_features.shape[1]
        h, w = num_channels, num_tokens // num_channels
        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w)
        return self.quantizer(to_quantizer_features)

    def decode_spectrum(
        self, quantized: torch.Tensor, channel_indices=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode quantized features into reconstructed magnitude & phase spectra.

        ``quantized``: ``[B, quantizer_dim, N, A]``. Returns
        ``(reconstructed_magnitude, reconstructed_phase)``.
        """
        decoder_features = self.decoder(quantized, channel_indices, return_patch_tokens=True)
        return (self.decode_task_layer_magnitude(decoder_features),
                self.decode_task_layer_phase(decoder_features))
