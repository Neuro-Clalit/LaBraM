# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# CLS-attention prediction head: one small global multi-head attention layer
# over the encoder's [cls, patch] tokens, then an MLP on the attended cls token.
# ---------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CLSAttentionHead(nn.Module):
    """Prediction head that lets the class token re-read every patch token.

    Input: the encoder's final tokens ``(B, 1 + N, D)``, class token at index 0.
    One multi-head attention layer with the class token as the query and all
    ``1 + N`` tokens as keys/values (global: every channel x time patch is
    visible). Its output is added back onto the encoder class token (residual),
    normalized, and passed through ``Linear -> GELU -> Dropout -> Linear``.

    Only the class-token row of the attention is computed: a single layer's
    class-token output does not depend on the other query rows, so this equals
    the class-token row of full self-attention at ``O(N)`` instead of ``O(N^2)``.

    ``forward(..., return_attention=True)`` also returns the class-token
    attention weights ``(B, num_heads, 1 + N)`` (index 0 = attention to itself),
    for inspecting which channels / time patches drive the prediction.
    """

    def __init__(self, embed_dim: int, num_outputs: int, num_heads: int = 4,
                 attn_dim: Optional[int] = None, mlp_hidden_dim: int = 256,
                 drop: float = 0.0, attn_drop: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        attn_dim = attn_dim or embed_dim
        if attn_dim % num_heads:
            raise ValueError(
                f"attention head width {attn_dim} is not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.scale = (attn_dim // num_heads) ** -0.5
        self.q = nn.Linear(embed_dim, attn_dim)
        self.kv = nn.Linear(embed_dim, 2 * attn_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(drop))
        self.fc = nn.Linear(mlp_hidden_dim, num_outputs)

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        out, attn = self._attend(tokens)
        out = self.fc(self.mlp(self.norm(out)))
        return (out, attn) if return_attention else out

    def _attend(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = tokens.shape
        h = self.num_heads
        q = self.q(tokens[:, :1]).reshape(b, 1, h, -1).transpose(1, 2)        # B,h,1,d
        k, v = self.kv(tokens).reshape(b, n, 2, h, -1).permute(2, 0, 3, 1, 4)  # B,h,n,d
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)       # B,h,1,n
        cls = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, -1)       # B,h*d
        return tokens[:, 0] + self.proj(cls), attn.squeeze(2)
