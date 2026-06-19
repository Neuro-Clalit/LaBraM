# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_

from labram.configs.model_config import VQNSPArchConfig
from labram.losses import LossConfig, SpectralReconstructionLoss, get_vqnsp_losses
from labram.models.neural_transformer import NeuralTransformer
from labram.models.quantizer import NormEMAVectorQuantizer
from labram.utils.checkpoint import load_pretrained_weights


class VQNSP(nn.Module):
    def __init__(self, config: VQNSPArchConfig):
        super().__init__()
        enc_cfg = config.encoder
        q_cfg = config.quantizer

        dec_cfg = config.decoder
        if dec_cfg.in_chans != q_cfg.quantizer_dim:
            from dataclasses import replace as _replace
            dec_cfg = _replace(dec_cfg, in_chans=q_cfg.quantizer_dim)
            print(f"Rewrite decoder in_chans to {q_cfg.quantizer_dim}")

        print('Encoder config', enc_cfg.__dict__)
        self.encoder = NeuralTransformer(enc_cfg)

        print('Decoder config', dec_cfg.__dict__)
        self.decoder = NeuralTransformer(dec_cfg)

        self.quantize = NormEMAVectorQuantizer(q_cfg)

        self.patch_size = enc_cfg.patch_size
        self.token_shape = (62, enc_cfg.eeg_window_size // self.patch_size)
        self.decoder_out_dim = config.decoder_out_dim

        enc_dim = enc_cfg.embed_dim
        dec_dim = dec_cfg.embed_dim
        self.encode_task_layer = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.Tanh(),
            nn.Linear(enc_dim, q_cfg.quantizer_dim),
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, config.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, config.decoder_out_dim),
        )

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_config = LossConfig(use_smooth_l1=config.smooth_l1_loss, vq_commitment_beta=q_cfg.beta)
        self.recon_loss = SpectralReconstructionLoss(self.loss_config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, channel_indices=None, **kwargs):
        quantize, codebook_indices, loss = self.encode(data, channel_indices=channel_indices)
        output = {}
        output['token'] = codebook_indices.view(data.shape[0], -1)
        output['input_eeg'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, channel_indices=None):
        batch_size, num_channels, a, t = x.shape
        encoder_features = self.encoder(x, channel_indices, return_patch_tokens=True)

        with torch.amp.autocast(encoder_features.device.type, enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        num_tokens = to_quantizer_features.shape[1]
        h, w = num_channels, num_tokens // num_channels

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, codebook_indices = self.quantize(to_quantizer_features)

        return quantize, codebook_indices, loss

    def decode(self, quantize, channel_indices=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder(quantize, channel_indices, return_patch_tokens=True)
        reconstructed_amplitude = self.decode_task_layer(decoder_features)
        reconstructed_angle = self.decode_task_layer_angle(decoder_features)
        return reconstructed_amplitude, reconstructed_angle

    def get_codebook_indices(self, x, channel_indices=None, **kwargs):
        # for LaBraM pre-training
        return self.get_tokens(x, channel_indices, **kwargs)['token']

    def forward(self, x, channel_indices=None, **kwargs):
        """
        x: shape [B, N, T]
        """

        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        amplitude_target, angle_target = self.recon_loss.spectrum_targets(x)

        quantize, codebook_indices, embedding_loss = self.encode(x, channel_indices)

        reconstructed_amplitude, reconstructed_angle = self.decode(quantize, channel_indices)
        amplitude_loss, angle_loss = self.recon_loss(
            reconstructed_amplitude, reconstructed_angle, amplitude_target, angle_target)

        losses = get_vqnsp_losses(embedding_loss, amplitude_loss, angle_loss, self.loss_config)
        loss = losses['total']

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = losses['embedding'].detach().mean()
        log[f'{split}/rec_loss'] = losses['amplitude'].detach().mean()
        log[f'{split}/rec_angle_loss'] = losses['phase'].detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log


def load_vqnsp_weights(model: nn.Module, pretrained_weight: str) -> None:
    """Load and filter a VQNSP checkpoint into *model* in-place."""
    weights = load_pretrained_weights(pretrained_weight)
    keys = list(weights.keys())
    for k in keys:
        if k.startswith(("loss", "teacher", "scaling")):
            del weights[k]
    model.load_state_dict(weights)
