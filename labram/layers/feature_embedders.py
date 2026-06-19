# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Embedders that turn encoder/quantizer features into a fixed-size vector for
# the downstream classification head. Used by CodebookRegularizedClassifier to
# build a concatenation of configurable feature sources.
# ---------------------------------------------------------

from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor


class FeatureEmbedder(nn.Module):
    """Embed a token feature into ``out_dim``, optionally mean-reducing first.

    ``reduce_dim`` (e.g. ``1``) averages over a token axis -- this is the
    "mean over chunks" pooling used for patch tokens and quantized codes.
    ``is_linear`` selects a single Linear (or Identity when dims already match)
    instead of the 2-layer Tanh MLP.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act_layer: Callable = nn.Tanh,
                 reduce_dim: Optional[int] = None,
                 is_linear: bool = False):
        super().__init__()
        if is_linear:
            self.embedder = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        else:
            self.embedder = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                act_layer(),
                nn.Linear(in_dim, out_dim),
            )
        self.reduce_dim = reduce_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.reduce_dim is not None:
            x = x.mean(dim=self.reduce_dim)
        return self.embedder(x)


class CodeBookBagEmbedder(nn.Module):
    """Bag-of-codes: normalized statistics over codebook indices.

    Codebook indices ``[B, N*A]`` (long) are summarized by an ``EmbeddingBag``
    (mean over the bag), giving a fixed ``out_dim`` vector regardless of the
    number of patches/channels. The lookup table is a fresh trainable embedding
    over the (frozen) code IDs -- it learns a downstream representation of the
    code vocabulary and is independent of the frozen quantizer codebook.
    """

    def __init__(self,
                 n_codes: int = 8192,
                 out_dim: int = 128,
                 act_layer: Callable = nn.Tanh):
        super().__init__()
        # sparse=True does not support the dense backward used here.
        self.embedder = nn.Sequential(
            nn.EmbeddingBag(n_codes, out_dim, sparse=False),
            act_layer(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.embedder(x.flatten(1))
