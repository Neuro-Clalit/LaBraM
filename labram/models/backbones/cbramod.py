# --------------------------------------------------------
# CBraMod encoder (Wang et al., ICLR 2025, arXiv:2412.07236) as a LaBraM-pipeline backbone.
#
# The classes CBraMod, PatchEmbedding, TransformerEncoder and TransformerEncoderLayer
# are vendored from https://github.com/wjq-learning/CBraMod (MIT License, Copyright (c)
# 2025 Jiquan Wang), commit b9e961003214326972c567eff390e75b0287e32a, files
# models/cbramod.py and models/criss_cross_transformer.py. Changes to the upstream
# code: the two files are merged, the `__main__` demos, the unused causal-mask helpers
# and the `torch._C._log_api_usage_once` call are dropped. Module and parameter names
# are upstream's, so the released weights load strictly.
# ---------------------------------------------------------
"""CBraMod: criss-cross transformer over ``[channels, patches]`` tokens.

Input convention of the released weights (pretraining script
``preprocessing_tueg_for_pretraining.py`` and ``pretrain_trainer.py``):
200 Hz, one-second patches of 200 samples, amplitude in µV / 100, and the 19
channels of :data:`CBRAMOD_CHANNELS` in that order. The order matters: the
only position information is a depthwise ``19 x 7`` convolution over the
``channels x patches`` grid, so the channel axis is a fixed layout, not a set.
"""
import copy
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from labram.models.backbones.base import EncoderBackbone, load_state_strict, read_weight_file

# Channel order of the pretraining data (TUEG 01_tcp_ar, `selected_channels`).
CBRAMOD_CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")


# ----------------------------------------------------------------- vendored
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz*ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------ adapter
class CBraModBackbone(EncoderBackbone):
    """CBraMod with our model-input convention.

    ``x`` ``[B, C, A, 200]`` in µV / 100 is already CBraMod's input; the adapter
    only fixes the channel layout: the given channels are written into the 19
    pretraining slots (a slot without a channel stays zero, a flat channel) and
    the tokens of the given channels are returned in the caller's order. A
    channel that has no slot is an error (drop it with ``data.drop_channels``).

    ``keep_recon_head`` keeps the pretraining output layer (``proj_out``) for
    the masked-reconstruction input check; a fine-tune model does not have it.
    """

    name = "cbramod"

    def __init__(self, d_model: int = 200, dim_feedforward: int = 800, n_layer: int = 12,
                 nhead: int = 8, keep_recon_head: bool = False):
        super().__init__()
        self.net = CBraMod(in_dim=200, out_dim=200, d_model=d_model, dim_feedforward=dim_feedforward,
                           seq_len=30, n_layer=n_layer, nhead=nhead)
        self.keep_recon_head = keep_recon_head
        if not keep_recon_head:
            self.net.proj_out = nn.Identity()
        self.embed_dim = d_model
        self.num_layers = n_layer
        self._slot = {name: i for i, name in enumerate(CBRAMOD_CHANNELS)}

    def slots(self, ch_names: Sequence[str]) -> List[int]:
        names = [str(c).upper() for c in ch_names]
        unknown = [c for c in names if c not in self._slot]
        if unknown:
            raise ValueError(f"CBraMod has no slot for channel(s) {unknown}; its layout is "
                             f"{list(CBRAMOD_CHANNELS)} (set data.drop_channels)")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate channel names: {names}")
        return [self._slot[c] for c in names]

    def _layout_tokens(self, x, ch_names, mask=None):
        """(tokens of all 19 slots in layout order, the given channels' slots, all present in order)."""
        slots = self.slots(ch_names)
        x = x.float()                     # the spectral branch takes an FFT of the input
        full = slots == list(range(len(CBRAMOD_CHANNELS)))
        if not full:
            wide = x.new_zeros(x.shape[0], len(CBRAMOD_CHANNELS), *x.shape[2:])
            wide[:, slots] = x
            x = wide
            if mask is not None:
                wide_mask = mask.new_zeros(mask.shape[0], len(CBRAMOD_CHANNELS), *mask.shape[2:])
                wide_mask[:, slots] = mask
                mask = wide_mask
        return self.net(x, mask), slots, full

    def encode(self, x: torch.Tensor, ch_names: Sequence[str], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens, slots, full = self._layout_tokens(x, ch_names, mask)
        return tokens if full else tokens[:, slots]

    def token_grid(self, window_patches: int):
        return len(CBRAMOD_CHANNELS), int(window_patches)

    def encode_grid(self, x: torch.Tensor, ch_names: Sequence[str]) -> torch.Tensor:
        return self._layout_tokens(x, ch_names)[0]

    def layer_id(self, name: str) -> int:
        if name.startswith("net.patch_embedding."):
            return 0
        if name.startswith("net.encoder.layers."):
            return int(name.split(".")[3]) + 1
        return self.num_layers + 1

    def load_pretrained(self, source) -> None:
        state = read_weight_file(source) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
        load_state_strict(self.net, state, "CBraMod",
                          drop_prefixes=() if self.keep_recon_head else ("proj_out.",))
