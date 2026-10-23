# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# MLP prediction head on the pooled encoder feature.
# ---------------------------------------------------------

from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_features: int, dims: Sequence[int], drop: float):
    layers, width = [], in_features
    for hidden in dims:
        layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(drop)]
        width = hidden
    return nn.Sequential(*layers), width


class MLPHead(nn.Module):
    """``[Linear -> GELU -> Dropout] x len(hidden_dims) -> Linear``.

    Applied to the pooled feature (``fc_norm`` of the mean patch token, or
    ``[mean, cls]`` with ``concat_cls_token``), so it replaces only the single
    ``nn.Linear`` head. ``fc`` is the output layer that ``init_scale`` shrinks.
    """

    def __init__(self, in_features: int, num_outputs: int,
                 hidden_dims: Sequence[int] = (256,), drop: float = 0.0):
        super().__init__()
        self.hidden, width = _mlp(in_features, hidden_dims, drop)
        self.fc = nn.Linear(width, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.hidden(x))


class MultiTaskHead(nn.Module):
    """Age regression + normal/abnormal classification from one pooled feature.

    ``shared`` (``[Linear -> GELU -> Dropout] x len(shared_dims)``) feeds two
    dedicated branches of the same form: ``age`` -> ``age_fc`` (the age
    output(s)) and ``abnormal`` -> ``abnormal_fc`` (one logit, abnormal = 1).
    The output is ``[age, abnormal_logit]`` concatenated on the last dim, so
    ``[..., :num_outputs]`` is the age prediction and ``[..., -1]`` the logit.
    ``age_fc`` and ``abnormal_fc`` are the layers ``init_scale`` shrinks.
    """

    def __init__(self, in_features: int, num_outputs: int,
                 shared_dims: Sequence[int] = (256,), age_dims: Sequence[int] = (128,),
                 abnormal_dims: Sequence[int] = (64,), drop: float = 0.0):
        super().__init__()
        self.shared, width = _mlp(in_features, shared_dims, drop)
        self.age, age_width = _mlp(width, age_dims, drop)
        self.abnormal, abnormal_width = _mlp(width, abnormal_dims, drop)
        self.age_fc = nn.Linear(age_width, num_outputs)
        self.abnormal_fc = nn.Linear(abnormal_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return torch.cat((self.age_fc(self.age(h)), self.abnormal_fc(self.abnormal(h))), dim=-1)
