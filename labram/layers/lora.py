# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# LoRA: low-rank adaptation of the frozen transformer linears (Hu et al. 2021).
# ---------------------------------------------------------

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """``nn.Linear`` plus a trainable low-rank update ``B A`` scaled by ``alpha / rank``.

    A subclass (not a wrapper) so ``.weight`` / ``.bias`` keep their names and
    the pretrained checkpoint loads unchanged; the extra ``lora_A`` / ``lora_B``
    parameters are the only new state-dict keys. ``B`` starts at zero, so the
    layer computes exactly the frozen linear until training moves it.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float, dropout: float) -> "LoRALinear":
        """Build around an existing linear, sharing its (frozen) parameters."""
        m = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                rank=rank, alpha=alpha, dropout=dropout)
        m.weight = linear.weight
        if linear.bias is not None:
            m.bias = linear.bias
        m.lora_A.data = m.lora_A.data.to(device=linear.weight.device, dtype=linear.weight.dtype)
        m.lora_B.data = m.lora_B.data.to(device=linear.weight.device, dtype=linear.weight.dtype)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + update * self.scaling


def is_lora_param(name: str) -> bool:
    return name.rsplit('.', 1)[-1] in ('lora_A', 'lora_B')


def inject_lora(model: nn.Module, targets: Sequence[str], rank: int, alpha: float,
                dropout: float = 0.0) -> int:
    """Replace every ``nn.Linear`` named in ``targets`` (``qkv``, ``proj``,
    ``fc1``, ``fc2``) inside the transformer blocks with a :class:`LoRALinear`.
    Returns the number of layers converted; raises when nothing matched."""
    targets = set(targets)
    n = 0
    for name, module in list(model.named_modules()):
        if not (name.startswith('blocks.') or '.blocks.' in name):
            continue
        for child_name, child in list(module.named_children()):
            if (child_name in targets and isinstance(child, nn.Linear)
                    and not isinstance(child, LoRALinear)):
                setattr(module, child_name,
                        LoRALinear.from_linear(child, rank, alpha, dropout))
                n += 1
    if n == 0:
        raise ValueError(f"model.lora.targets={sorted(targets)} matched no block linear")
    return n


def mark_only_lora_trainable(model: nn.Module, train_prefixes: Iterable[str] = ()) -> int:
    """Freeze everything except the LoRA factors and the parameters whose name
    starts with one of ``train_prefixes`` (head / norms). Returns the number of
    trainable parameters."""
    prefixes = tuple(train_prefixes)
    trainable = 0
    for name, p in model.named_parameters():
        keep = is_lora_param(name) or (bool(prefixes) and name.startswith(prefixes))
        p.requires_grad_(keep)
        trainable += p.numel() if keep else 0
    return trainable
