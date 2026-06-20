# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Structured loss-breakdown container returned by composite criteria.
# ---------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class LossBreakdown:
    """The weighted total plus each unweighted component, for logging.

    ``components`` holds the individual (unweighted) loss tensors keyed by name
    (e.g. ``classifier``, ``magnitude``, ``phase``, ``quantize``). ``total`` is
    the weighted sum the optimizer steps on.
    """

    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_log_dict(self, split: str = 'train') -> Dict[str, float]:
        """Flatten into ``{split}/{name}_loss`` scalar floats for logging."""
        out = {f'{split}/total_loss': float(self.total.detach().mean())}
        for name, value in self.components.items():
            out[f'{split}/{name}_loss'] = float(value.detach().mean())
        return out
