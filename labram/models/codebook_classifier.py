# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Codebook-regularized downstream classifier: a fine-tuning model that runs the
# EEG through a (pre-trained) VQNSP quantizer + decoder alongside the encoder so
# spectral-reconstruction and quantization losses can regularize the task.
# ---------------------------------------------------------

from typing import List, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from labram.layers import CodeBookBagEmbedder, FeatureEmbedder
from labram.models.components import decode_spectrum, quantize_patch_tokens
from labram.models.outputs import PredictorOutput

# Supported classification feature sources.
ENCODER_MEAN = 'encoder_mean'
QUANTIZE_MEAN = 'quantize_mean'
BAG_OF_CODES = 'bag_of_codes'
FEATURE_SOURCES = (ENCODER_MEAN, QUANTIZE_MEAN, BAG_OF_CODES)


class CodebookRegularizedClassifier(nn.Module):
    """Encoder + (frozen-codebook) quantizer + decoder, with a classification
    head over configurable feature sources.

    A single encoder forward (patch tokens, fc_norm applied -- matching the
    VQNSP encode path) feeds both:
      * the classification head, over a concatenation of feature sources
        (``encoder_mean`` / ``quantize_mean`` / ``bag_of_codes``), and
      * the quantize -> decode -> magnitude/phase reconstruction path.

    ``forward`` returns a :class:`PredictorOutput`; with ``classify_only=True``
    the decoder branch is skipped (used at evaluation).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        quantizer: nn.Module,
        decoder: nn.Module,
        encode_task_layer: nn.Module,
        decode_task_layer_magnitude: nn.Module,
        decode_task_layer_phase: nn.Module,
        num_classes: int,
        feature_sources: Sequence[str] = (ENCODER_MEAN,),
        features_emb_dim: int = 128,
        linear_embedding: bool = True,
        norm_embedding: bool = True,
        classifier_type: str = 'linear',
    ):
        super().__init__()
        if not feature_sources:
            raise ValueError("feature_sources must list at least one source")
        unknown = [s for s in feature_sources if s not in FEATURE_SOURCES]
        if unknown:
            raise ValueError(f"unknown feature_sources {unknown}; supported: {FEATURE_SOURCES}")

        self.encoder = encoder
        self.quantize = quantizer
        self.decoder = decoder
        self.encode_task_layer = encode_task_layer
        self.decode_task_layer_magnitude = decode_task_layer_magnitude
        self.decode_task_layer_phase = decode_task_layer_phase
        self.patch_size = encoder.patch_size
        self.feature_sources: List[str] = list(feature_sources)

        embed_dim = encoder.embed_dim
        quantizer_dim = quantizer.quantizer_dim
        num_codebook_tokens = quantizer.num_tokens
        self.embedders = nn.ModuleDict()
        for src in self.feature_sources:
            if src == ENCODER_MEAN:
                self.embedders[src] = FeatureEmbedder(
                    embed_dim, features_emb_dim, reduce_dim=1, is_linear=linear_embedding)
            elif src == QUANTIZE_MEAN:
                self.embedders[src] = FeatureEmbedder(
                    quantizer_dim, features_emb_dim, reduce_dim=1, is_linear=linear_embedding)
            elif src == BAG_OF_CODES:
                self.embedders[src] = CodeBookBagEmbedder(num_codebook_tokens, features_emb_dim)

        features_dim = features_emb_dim * len(self.feature_sources)
        self.norm_emb = nn.LayerNorm(features_dim) if norm_embedding else nn.Identity()
        if num_classes <= 0:
            self.classifier_head = nn.Identity()
        elif classifier_type == 'mlp':
            self.classifier_head = nn.Sequential(
                nn.Linear(features_dim, features_dim), nn.GELU(),
                nn.Linear(features_dim, num_classes))
        else:
            self.classifier_head = nn.Linear(features_dim, num_classes)

    def get_num_layers(self) -> int:
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {f'encoder.{n}' for n in self.encoder.no_weight_decay()}
        names |= {f'decoder.{n}' for n in self.decoder.no_weight_decay()}
        names.add('quantize.embedding.weight')
        return names

    def _classification_features(self, patch_tokens, quantized, codebook_indices, batch_size):
        feats = []
        for src in self.feature_sources:
            if src == ENCODER_MEAN:
                feats.append(self.embedders[src](patch_tokens))
            elif src == QUANTIZE_MEAN:
                feats.append(self.embedders[src](rearrange(quantized, 'b d h w -> b (h w) d')))
            elif src == BAG_OF_CODES:
                feats.append(self.embedders[src](codebook_indices.view(batch_size, -1)))
        return torch.cat(feats, dim=-1)

    def forward(self, x, channel_indices=None, classify_only=False, **kwargs) -> PredictorOutput:
        """``x``: ``[B, N, A, T]`` (already patched)."""
        batch_size, num_channels = x.shape[0], x.shape[1]
        patch_tokens = self.encoder.forward_features(
            x, channel_indices=channel_indices, return_patch_tokens=True)

        quantized, quantize_loss, codebook_indices = quantize_patch_tokens(
            self.encode_task_layer, self.quantize, patch_tokens, num_channels)

        features = self._classification_features(
            patch_tokens, quantized, codebook_indices, batch_size)
        logits = self.classifier_head(self.norm_emb(features))

        if classify_only:
            return PredictorOutput(logits=logits)

        recon_magnitude, recon_phase = decode_spectrum(
            self.decoder, self.decode_task_layer_magnitude, self.decode_task_layer_phase,
            quantized, channel_indices)
        return PredictorOutput(
            logits=logits, recon_magnitude=recon_magnitude, recon_phase=recon_phase,
            quantize_loss=quantize_loss, x_patched=x)
