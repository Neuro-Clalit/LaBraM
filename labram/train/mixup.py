# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Mixup / C-Mixup for regression fine-tuning.
# ---------------------------------------------------------

from typing import Tuple

import torch


def mixup_partners(targets: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """Index of the partner each sample is mixed with.

    ``sigma <= 0``: a random derangement-free permutation (plain mixup).
    ``sigma > 0``: C-Mixup — partner ``j`` for ``i`` is drawn with probability
    proportional to ``exp(-(y_i - y_j)^2 / (2 sigma^2))`` (never ``i`` itself),
    ``sigma`` in the units of ``targets``.
    """
    n = targets.shape[0]
    if sigma <= 0:
        return torch.randperm(n, device=targets.device)
    t = targets.float().reshape(-1)
    w = torch.exp(-(t[:, None] - t[None, :]) ** 2 / (2 * sigma ** 2))
    w = w + 1e-12          # keep every off-diagonal entry positive (underflow)
    w.fill_diagonal_(0.0)
    return torch.multinomial(w, 1).squeeze(1)


def mixup_batch(samples: torch.Tensor, targets: torch.Tensor, alpha: float,
                sigma: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Mix a batch with a permuted copy of itself.

    ``lam ~ Beta(alpha, alpha)`` is drawn once per batch (``alpha <= 0`` ->
    ``lam = 1``, i.e. no mixing). Returns ``(samples, targets, lam)`` with both
    mixed by the same ``lam`` and partner assignment. A batch of one sample is
    returned unchanged.
    """
    if samples.shape[0] < 2 or alpha <= 0:
        return samples, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = mixup_partners(targets, sigma)
    mixed_x = lam * samples + (1.0 - lam) * samples[perm]
    mixed_y = lam * targets + (1.0 - lam) * targets[perm]
    return mixed_x, mixed_y, lam
