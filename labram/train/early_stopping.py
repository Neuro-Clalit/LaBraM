# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Early stopping on a validation plateau (trainer.early_stopping).
# ---------------------------------------------------------
"""Stop fine-tuning when the validation metric stops improving.

With history ``v_0, v_1, ...`` of the model-selection metric (val MAE for age,
lower is better), patience ``k`` and warm-up ``W`` epochs, epoch ``e`` is
checked only when ``e >= W + k - 1`` -- never during warm-up, nor before ``k``
post-warm-up epochs exist -- and training stops when the last ``k`` epochs
bring no improvement of at least ``min_delta`` over every earlier epoch:

    lower is better:   min(v_{e-k+1..e}) >= min(v_{0..e-k}) - min_delta
    higher is better:  max(v_{e-k+1..e}) <= max(v_{0..e-k}) + min_delta

i.e. a plateau (a tie counts) or negative progress over the window.
"""
import math
from typing import Sequence


def should_stop(history: Sequence[float], *, patience: int = 3, min_delta: float = 0.0,
                warmup_epochs: int = 0, lower_is_better: bool = True) -> bool:
    """``history[i]`` is the metric after epoch ``i``; checks the last epoch."""
    if patience < 1:
        raise ValueError("trainer.early_stopping_patience must be >= 1")
    e = len(history) - 1
    if e < warmup_epochs + patience - 1 or e - patience + 1 < 1:
        return False
    window = [v for v in history[e - patience + 1:] if v is not None and math.isfinite(v)]
    before = [v for v in history[:e - patience + 1] if v is not None and math.isfinite(v)]
    if not before:
        return False
    if not window:
        return True                     # nothing finite in the whole window
    if lower_is_better:
        return min(window) >= min(before) - min_delta
    return max(window) <= max(before) + min_delta
