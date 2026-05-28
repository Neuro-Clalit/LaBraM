# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# ---------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

from labram.models.neural_transformer import NeuralTransformer
from labram.models.quantizer import NormEMAVectorQuantizer
from labram.utils.checkpoint import load_pretrained_weights


class VQNSP(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 num_codebook_tokens=8192,
                 quantizer_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss=False,
                 ):
        super().__init__()
        if decoder_config['in_chans'] != quantizer_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {quantizer_dim}")
            decoder_config['in_chans'] = quantizer_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = NeuralTransformer(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_codebook_tokens=num_codebook_tokens, quantizer_dim=quantizer_dim, beta=1.0,
            kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['eeg_window_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], quantizer_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

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

    def calculate_reconstruction_loss(self, reconstructed, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        return self.loss_fn(reconstructed, target)

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, channel_indices=None, **kwargs):
        """
        x: shape [B, N, T]
        """

        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        quantize, codebook_indices, embedding_loss = self.encode(x, channel_indices)

        reconstructed_amplitude, reconstructed_angle = self.decode(quantize, channel_indices)
        amplitude_loss = self.calculate_reconstruction_loss(reconstructed_amplitude, amplitude)
        angle_loss = self.calculate_reconstruction_loss(reconstructed_angle, angle)
        loss = embedding_loss + amplitude_loss + angle_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = embedding_loss.detach().mean()
        log[f'{split}/rec_loss'] = amplitude_loss.detach().mean()
        log[f'{split}/rec_angle_loss'] = angle_loss.detach().mean()
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
