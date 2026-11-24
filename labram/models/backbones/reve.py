# --------------------------------------------------------
# REVE encoder (El Ouahidi et al., NeurIPS 2025, arXiv:2510.21585) as a LaBraM-pipeline backbone.
#
# GEGLU, RMSNorm, FeedForward, ClassicalAttention, Attention, TransformerBackbone,
# FourierEmb4D, patch_embedding, mlp_pos_embedding and the encoder forward are
# vendored from https://github.com/elouayas/reve_eeg (MIT License, Copyright (c) 2026
# BRAIN - BRoader Artificial INtelligence), commit
# 06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7, files src/models/backbone.py and
# src/models/encoder.py, in the form released with the weights
# (huggingface.co/brain-bzh/reve-base, modeling_reve.py). Changes to the upstream
# code: no `transformers` / `flash_attn` dependency (PyTorch's scaled dot-product
# attention only), no position noise (pretraining only), the learnable-position
# variant and the attention-map helpers are dropped. Module and parameter names are
# upstream's, so the released weights load strictly.
#
# The released WEIGHTS are under the "REVE Responsible Use License v1.0" (see the
# model card): models derived from them must carry that licence.
# ---------------------------------------------------------
"""REVE: a transformer over ``channels x patches`` tokens with a 4-D Fourier
position code (electrode x, y, z and patch index), so any montage is accepted.

Input convention: 200 Hz; patches of 200 samples with 20 samples overlap (a 10 s
window gives 11 patches); electrode positions looked up by name. The authors'
fine-tuning loaders feed µV / 100 (our model input), their pretraining a
per-channel, per-session z-score clipped at 15 SD (``model.encoder.input_norm``).
"""
import json
import math
import os
from typing import Dict, Sequence, Set

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from labram.models.backbones.base import EncoderBackbone, load_state_strict, read_weight_file

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "reve_positions.json")


# ----------------------------------------------------------------- vendored
class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalAttention(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = ClassicalAttention(self.heads)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


class TransformerBackbone(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self.dim, heads=heads, head_dim=head_dim),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.
    """

    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        input_shape = positions.shape
        batch_dims = list(input_shape[:-1])

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        )
        batch_dims.append(-1)
        loc = loc.view(batch_dims)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Input dimension ({current_dim}) is too small for target "
                    f"embedding dimension ({self.dimension}). Expected at least {half_dim}."
                )

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """(B, C, 3) positions -> (B, C * num_patches, 4): every position repeated
        with each patch index as the fourth coordinate."""
        batch, nchans, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()
        time_values = time_values.view(1, 1, num_patches, 1).expand(batch, nchans, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)
        pos_with_time = pos_with_time.view(batch, nchans * num_patches, 4)
        return pos_with_time


def patch_embedding(embed_dim, patch_size):
    to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
    return to_patch_embedding


def mlp_pos_embedding(embed_dim):
    mlp_pos_embedding = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
    return mlp_pos_embedding


class Reve(nn.Module):
    def __init__(self, embed_dim=512, depth=22, heads=8, head_dim=64, mlp_dim_ratio=2.66,
                 use_geglu=True, freqs=4, patch_size=200, patch_overlap=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.freqs = freqs
        self.patch_size = patch_size
        self.overlap_size = patch_overlap

        self.transformer = TransformerBackbone(
            dim=embed_dim, depth=depth, heads=heads, head_dim=head_dim,
            mlp_dim=int(embed_dim * mlp_dim_ratio), geglu=use_geglu,
        )

        self.to_patch_embedding = patch_embedding(self.embed_dim, self.patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``eeg`` (B, C, T), ``pos`` (B, C, 3) -> tokens (B, C, patches, embed_dim)."""
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        _b, c, h, _p = patches.shape

        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        x = rearrange(self.to_patch_embedding(patches), "b c h e -> b (c h) e", c=c, h=h, e=self.embed_dim) + pos_embed
        x = self.transformer(x)
        x = rearrange(x, "b (c h) e -> b c h e", b=_b, c=c, h=h, e=self.embed_dim)
        return x

    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Attention pooling of tokens (B, C, S, E) with the pretrained query -> (B, E)."""
        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")  # (B, C*S, E)
        query_output = self.cls_query_token.expand(b, -1, -1)  # (B, 1, E)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.embed_dim**0.5)  # (B, 1, C*S)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, C*S)
        out = torch.matmul(attention_weights, x).squeeze(1)  # (B, E)
        return out


# ------------------------------------------------------------------ adapter
def load_positions(path: str = POSITIONS_FILE) -> Dict[str, list]:
    """Electrode name (upper case, the names of ``standard_1020``) -> [x, y, z],
    copied from the REVE position bank (huggingface.co/brain-bzh/reve-positions,
    ``positions.json``). ``T3 / T4 / T5 / T6`` share the coordinates of
    ``T7 / T8 / P7 / P8``."""
    with open(path) as fh:
        return json.load(fh)


class REVEBackbone(EncoderBackbone):
    """REVE with our model-input convention.

    ``x`` ``[B, C, A, 200]`` is flattened to the continuous window and cut into
    REVE's overlapping patches. Every channel name must have a position: an
    unknown name is an error (the upstream position bank drops it silently,
    which would shift the positions of the channels after it).
    """

    name = "reve"

    def __init__(self, embed_dim: int = 512, depth: int = 22, heads: int = 8, head_dim: int = 64,
                 mlp_dim_ratio: float = 2.66, freqs: int = 4):
        super().__init__()
        self.net = Reve(embed_dim=embed_dim, depth=depth, heads=heads, head_dim=head_dim,
                        mlp_dim_ratio=mlp_dim_ratio, freqs=freqs)
        self.embed_dim = embed_dim
        self.num_layers = depth
        self._bank = load_positions()
        self._pos_cache: Dict[tuple, torch.Tensor] = {}      # CPU tensors, not part of the state dict

    def positions(self, ch_names: Sequence[str]) -> torch.Tensor:
        key = tuple(str(c).upper() for c in ch_names)
        if key not in self._pos_cache:
            unknown = [c for c in key if c not in self._bank]
            if unknown:
                raise ValueError(f"REVE: no electrode position for channel(s) {unknown} "
                                 f"(set data.drop_channels)")
            self._pos_cache[key] = torch.tensor([self._bank[c] for c in key], dtype=torch.float32)
        return self._pos_cache[key]

    def encode(self, x: torch.Tensor, ch_names: Sequence[str]) -> torch.Tensor:
        b, c, a, t = x.shape
        eeg = x.reshape(b, c, a * t).float()
        pos = self.positions(ch_names).to(eeg.device).unsqueeze(0).expand(b, -1, -1)
        return self.net(eeg, pos)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net.attention_pooling(tokens)

    def pool_parameters(self):
        return [self.net.cls_query_token]

    def layer_id(self, name: str) -> int:
        if name.startswith("net.transformer.layers."):
            return int(name.split(".")[3]) + 1
        if name.startswith(("net.to_patch_embedding.", "net.mlp4d.", "net.ln.")):
            return 0
        return self.num_layers + 1

    def no_weight_decay(self) -> Set[str]:
        return {"net.cls_query_token"}

    def load_pretrained(self, source) -> None:
        state = read_weight_file(source) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
        load_state_strict(self.net, state, "REVE")
