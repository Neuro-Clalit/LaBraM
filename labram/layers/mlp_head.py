# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# MLP prediction head on the pooled encoder feature.
# ---------------------------------------------------------

from typing import Sequence

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """``[Linear -> GELU -> Dropout] x len(hidden_dims) -> Linear``.

    Applied to the pooled feature (``fc_norm`` of the mean patch token, or
    ``[mean, cls]`` with ``concat_cls_token``), so it replaces only the single
    ``nn.Linear`` head. ``fc`` is the output layer that ``init_scale`` shrinks.
    """

    def __init__(self, in_features: int, num_outputs: int,
                 hidden_dims: Sequence[int] = (256,), drop: float = 0.0):
        super().__init__()
        layers, width = [], in_features
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(drop)]
            width = hidden
        self.hidden = nn.Sequential(*layers)
        self.fc = nn.Linear(width, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.hidden(x))
