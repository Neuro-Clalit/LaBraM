# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Multiple-instance (bag) pooling of window embeddings into a recording embedding.
# ---------------------------------------------------------
"""Bag pooling layers for recording-level MIL (``docs/b1_attention_mil_plan.md``).

Every pool maps the window embeddings of a bag to one recording embedding of
the same width, so the run's existing head applies unchanged:

    forward(h [B, K, D], mask [B, K] (True = real window),
            q [B, K, Q] | None (standardized descriptors),
            t [B, K] | None (window minute)) -> (z [B, D], alpha [B, K], extras)

``alpha`` are the per-window pooling weights (they sum to 1 over the real
windows), so with a linear head the recording prediction is exactly the
alpha-weighted sum of the window predictions.

* ``mean``    (M0) -- masked mean of ``h``; the benchmark.
* ``gated``   (M1) -- gated attention (Ilse et al. 2018):
  ``a = w^T (tanh(V s) * sigmoid(U s))``, masked softmax over the bag.
  ``branches > 1`` (M2) averages several independent attention branches.
* ``sigmoid`` (M3) -- non-competing gates ``g = sigmoid(w^T tanh(V s) + b)``
  (BridgeMIL style); ``z = sum g h / sum g``; ``gate`` is returned in ``extras``.
* ``query``   (M4) -- a learned query cross-attends over the windows
  (multi-head; Set Transformer PMA), optionally after one self-attention block.

``score_input`` chooses what the attention *score* sees: ``"h"`` (encoder
features), ``"q"`` (descriptors only) or ``"hq"`` (both); the pooled value is
always ``h`` (plus a sinusoidal minute encoding when ``time_encoding``).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_KINDS = ("mean", "gated", "sigmoid", "query")
SCORE_INPUTS = ("h", "q", "hq")


def time_encoding(t: torch.Tensor, dim: int, max_minutes: float = 60.0) -> torch.Tensor:
    """Sinusoidal encoding ``[B, K, dim]`` of window minutes ``t [B, K]``."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                      * (-math.log(max_minutes * 10.0) / max(half - 1, 1)))
    ang = t.float()[..., None] * freqs * 2 * math.pi / 10.0
    enc = torch.cat([ang.sin(), ang.cos()], -1)
    return F.pad(enc, (0, dim - enc.shape[-1]))


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=dim).masked_fill(~mask, 0.0)


def attention_entropy_gap(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over bags of ``log n - H(alpha)`` (0 = uniform; Eq. (2.2) of the plan)."""
    n = mask.sum(-1).clamp_min(1).float()
    h = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1)
    return (n.log() - h).mean()


class BagPool(nn.Module):
    """Shared input handling: dropout, time encoding, score-input assembly."""

    def __init__(self, dim: int, n_desc: int = 0, score_input: str = "h", dropout: float = 0.0,
                 time_encoding: bool = False):
        super().__init__()
        if score_input not in SCORE_INPUTS:
            raise ValueError(f"score_input {score_input!r} not in {SCORE_INPUTS}")
        if "q" in score_input and n_desc <= 0:
            raise ValueError(f"score_input={score_input!r} needs descriptors (n_desc > 0)")
        self.dim, self.n_desc, self.score_input = dim, n_desc, score_input
        self.drop = nn.Dropout(dropout)
        self.use_time = time_encoding

    @property
    def score_dim(self) -> int:
        return {"h": self.dim, "q": self.n_desc, "hq": self.dim + self.n_desc}[self.score_input]

    def values(self, h, t):
        h = self.drop(h)
        if self.use_time and t is not None:
            h = h + time_encoding(t, h.shape[-1]).to(h.dtype)
        return h

    def score_inputs(self, v, q):
        if self.score_input == "h":
            return v
        if q is None:
            raise ValueError("this pool scores on descriptors; pass q")
        q = q.to(v.dtype)
        return q if self.score_input == "q" else torch.cat([v, q], -1)


class MeanPool(BagPool):
    """M0: masked mean of the window embeddings."""

    def forward(self, h, mask, q=None, t=None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        v = self.values(h, t)
        alpha = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        return (alpha[..., None] * v).sum(1), alpha, {}


class GatedAttentionPool(BagPool):
    """M1 (``branches=1``) / M2 (``branches>1``): gated attention MIL."""

    def __init__(self, dim, n_desc=0, score_input="h", attn_dim=128, branches=1, gated=True,
                 temperature=1.0, topk_mask_prob=0.0, dropout=0.0, time_encoding=False):
        super().__init__(dim, n_desc, score_input, dropout, time_encoding)
        s = self.score_dim
        self.V = nn.Linear(s, attn_dim * branches)
        self.U = nn.Linear(s, attn_dim * branches) if gated else None
        self.w = nn.Parameter(torch.randn(branches, attn_dim) * 0.02)
        self.branches, self.attn_dim = branches, attn_dim
        self.temperature, self.topk_mask_prob = temperature, topk_mask_prob

    def forward(self, h, mask, q=None, t=None):
        v = self.values(h, t)
        s = self.score_inputs(v, q)
        b, k = mask.shape
        a = torch.tanh(self.V(s))
        if self.U is not None:
            a = a * torch.sigmoid(self.U(s))
        a = a.view(b, k, self.branches, self.attn_dim)
        scores = torch.einsum("bkhl,hl->bhk", a, self.w) / self.temperature    # [B, H, K]
        m = mask[:, None, :].expand(-1, self.branches, -1)
        if self.training and self.topk_mask_prob > 0:
            # ACMIL-style: sometimes hide each branch's top window so attention spreads
            hide = torch.rand(b, self.branches, 1, device=h.device) < self.topk_mask_prob
            top = scores.masked_fill(~m, float("-inf")).argmax(-1, keepdim=True)
            drop = torch.zeros_like(m).scatter(-1, top, True) & hide & (m.sum(-1, keepdim=True) > 1)
            m = m & ~drop
        alpha_b = masked_softmax(scores, m)                                     # [B, H, K]
        z = torch.einsum("bhk,bkd->bd", alpha_b, v) / self.branches
        return z, alpha_b.mean(1), {"alpha_branches": alpha_b}


class SigmoidGatePool(BagPool):
    """M3: independent sigmoid gates per window; weighted mean of the gated windows."""

    def __init__(self, dim, n_desc=0, score_input="h", attn_dim=64, dropout=0.0,
                 time_encoding=False, eps=1e-6):
        super().__init__(dim, n_desc, score_input, dropout, time_encoding)
        self.V = nn.Linear(self.score_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.eps = eps

    def forward(self, h, mask, q=None, t=None):
        v = self.values(h, t)
        g = torch.sigmoid(self.w(torch.tanh(self.V(self.score_inputs(v, q)))).squeeze(-1))
        g = g * mask.float()
        alpha = g / (g.sum(-1, keepdim=True) + self.eps)
        return (alpha[..., None] * v).sum(1), alpha, {"gate": g}


class QueryPool(BagPool):
    """M4: a learned query cross-attends over the windows (pre-norm, multi-head),
    optionally after one self-attention block. Descriptors enter the keys
    through a linear projection (``score_input`` ``q`` / ``hq``)."""

    def __init__(self, dim, n_desc=0, score_input="h", heads=4, self_attention=False,
                 dropout=0.0, time_encoding=False):
        super().__init__(dim, n_desc, score_input, dropout, time_encoding)
        self.block = (nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                                 batch_first=True, norm_first=True)
                      if self_attention else None)
        self.norm = nn.LayerNorm(dim)
        self.qproj = nn.Linear(n_desc, dim) if "q" in score_input else None
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, h, mask, q=None, t=None):
        v = self.values(h, t)
        if self.block is not None:
            v = self.block(v, src_key_padding_mask=~mask)
        key = self.norm(v)
        if self.qproj is not None:
            if q is None:
                raise ValueError("this pool scores on descriptors; pass q")
            pq = self.qproj(q.to(v.dtype))
            key = pq if self.score_input == "q" else key + pq
        z, w = self.attn(self.query.expand(v.shape[0], -1, -1), key, v, key_padding_mask=~mask,
                         need_weights=True, average_attn_weights=True)
        alpha = w[:, 0].masked_fill(~mask, 0.0)
        return z[:, 0], alpha, {}


def build_bag_pool(kind: str, dim: int, n_desc: int = 0, **kw) -> BagPool:
    """``kind`` in :data:`POOL_KINDS`; ``kw`` are the pool's own options."""
    if kind == "mean":
        return MeanPool(dim, n_desc, kw.get("score_input", "h"), kw.get("dropout", 0.0),
                        kw.get("time_encoding", False))
    if kind == "gated":
        return GatedAttentionPool(dim, n_desc, **kw)
    if kind == "sigmoid":
        return SigmoidGatePool(dim, n_desc, **{k: v for k, v in kw.items()
                                               if k in ("score_input", "attn_dim", "dropout", "time_encoding")})
    if kind == "query":
        return QueryPool(dim, n_desc, **{k: v for k, v in kw.items()
                                         if k in ("score_input", "heads", "self_attention", "dropout",
                                                  "time_encoding")})
    raise ValueError(f"unknown bag pool {kind!r} (expected one of {POOL_KINDS})")


class DescriptorNorm(nn.Module):
    """Frozen standardization of window descriptors: ``log1p`` on the
    non-negative heavy-tailed columns, z-score with fitted statistics,
    missing -> 0, clipped at +-6. Statistics are buffers, so they travel with
    the checkpoint."""

    def __init__(self, n_desc: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_desc))
        self.register_buffer("std", torch.ones(n_desc))
        self.register_buffer("log_cols", torch.zeros(n_desc, dtype=torch.bool))

    def _transform(self, q):
        q = torch.where(self.log_cols, torch.log1p(q.clamp_min(0)), q)
        return torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def fit(self, q: torch.Tensor, log_cols=None):
        """``q`` ``[N, Q]`` (any leading shape flattened)."""
        q = q.reshape(-1, q.shape[-1]).float()
        if log_cols is None:
            lo = torch.nan_to_num(q, nan=0.0).amin(0)
            med = torch.nan_to_num(q, nan=0.0).median(0).values
            hi = torch.nan_to_num(q, nan=0.0).quantile(0.99, dim=0) if q.shape[0] <= 2 ** 24 else q.amax(0)
            log_cols = (lo >= 0) & (hi > 20 * med.clamp_min(1e-6))
        self.log_cols.copy_(torch.as_tensor(log_cols, dtype=torch.bool))
        x = self._transform(q)
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))
        return self

    def forward(self, q):
        return ((self._transform(q.float()) - self.mean) / self.std).clamp(-6, 6)


class MILPooling(nn.Module):
    """The bag pooling attached to a model as ``model.mil`` (``mil.*`` config):
    the pool plus the descriptor standardization (when the scores read
    descriptors). ``n_desc`` counts the descriptors including the minute."""

    def __init__(self, dim: int, kind: str = "gated", n_desc: int = 0, **kw):
        super().__init__()
        uses_q = "q" in kw.get("score_input", "h")
        self.n_desc = n_desc if uses_q else 0
        self.pool = build_bag_pool(kind, dim, self.n_desc, **kw)
        self.desc_norm = DescriptorNorm(n_desc) if uses_q else None
        self.register_buffer("desc_fitted", torch.zeros((), dtype=torch.bool))

    @property
    def needs_descriptors(self) -> bool:
        return self.desc_norm is not None

    def forward(self, h, mask, q=None, t=None):
        qn = self.desc_norm(q) if (self.desc_norm is not None and q is not None) else None
        return self.pool(h, mask, qn, t)
