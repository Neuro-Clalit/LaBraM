# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Downstream regression criterion selection (e.g. brain-age prediction).
# ---------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from labram.configs.loss_config import LossConfig
from labram.losses.classification import build_classification_criterion

REGRESSION_LOSSES = ("mse", "l1", "huber", "soft_label")


def downstream_term_name(task: str) -> str:
    """Name of the downstream loss term in logs: ``regression`` or ``classifier``.

    Used on the plain and the codebook-regularized path alike, so a run's
    ``regression_loss`` series compares directly across both; which criterion
    it is (Huber, L1, ...) is in the run config.
    """
    return "regression" if task == "regression" else "classifier"


def soft_label_centers(cfg: LossConfig) -> torch.Tensor:
    """Bin centers in target units (years): ``min, min + width, ..., <= max``."""
    n = int(round((cfg.soft_label_max - cfg.soft_label_min) / cfg.soft_label_bin_width)) + 1
    if n < 2:
        raise ValueError("loss.soft_label_min/max/bin_width give fewer than 2 bins")
    return cfg.soft_label_min + cfg.soft_label_bin_width * torch.arange(n, dtype=torch.float32)


def soft_label_n_bins(cfg: LossConfig) -> int:
    """Head width for ``regression_loss = "soft_label"``."""
    return int(soft_label_centers(cfg).numel())


class SoftLabelRegressionLoss(nn.Module):
    """KL divergence between a Gaussian soft label over age bins and the head's
    softmax (SFCN, Peng et al. 2021).

    The head emits one logit per bin; the target distribution for a window of
    age ``y`` is ``softmax(-(c_k - y)^2 / (2 sigma^2))`` over the bin centers
    ``c_k``. Both are expressed in the loader's normalized target space (the
    centers and sigma are z-scored with ``target_stats``), so the loss sees the
    same ``(B, 1)`` z-scored target as the scalar criteria. ``predict`` turns
    the logits back into a ``(B, 1)`` scalar — the expectation over the bins —
    which the training/eval loops score exactly like a scalar head's output.
    """

    def __init__(self, centers: torch.Tensor, sigma: float):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"soft-label sigma must be > 0, got {sigma}")
        self.register_buffer("centers", centers.float().reshape(-1))
        self.sigma = float(sigma)

    @property
    def n_bins(self) -> int:
        return int(self.centers.numel())

    def _centers(self, like: torch.Tensor) -> torch.Tensor:
        return self.centers.to(like.device)

    def soft_targets(self, target: torch.Tensor) -> torch.Tensor:
        """``(B, 1)`` or ``(B,)`` targets -> ``(B, n_bins)`` Gaussian soft labels."""
        t = target.float().reshape(-1, 1)
        logits = -(self._centers(t)[None, :] - t) ** 2 / (2 * self.sigma ** 2)
        return torch.softmax(logits, dim=-1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != self.n_bins:
            raise ValueError(
                f"soft-label head must emit {self.n_bins} logits, got {logits.shape[-1]}")
        log_p = F.log_softmax(logits.float(), dim=-1)
        q = self.soft_targets(target)
        # KL(q || p), summed over bins, averaged over the batch. Bins whose soft
        # label underflowed to 0 contribute nothing (0 * log 0 := 0).
        kl = torch.where(q > 0, q * (torch.log(q.clamp_min(1e-30)) - log_p), torch.zeros_like(q))
        return kl.sum(-1).mean()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value over the bins: ``(B, n_bins)`` logits -> ``(B, 1)``."""
        p = torch.softmax(logits.float(), dim=-1)
        return (p * self._centers(p)[None, :]).sum(-1, keepdim=True)


BALANCE_MODES = ("none", "inverse", "sqrt_inverse")


def age_balance_weights(train_targets, cfg: LossConfig) -> Tuple[float, float, torch.Tensor]:
    """Per-bin loss weights from the raw (year-valued) train targets.

    Returns ``(low, width, weights)``: bin ``k`` covers
    ``[low + k * width, low + (k + 1) * width)``; targets outside the range
    clamp to the end bins. The density is the window histogram, optionally
    LDS-smoothed; empty bins inherit their smoothed neighbours' density, and
    weights are normalized to mean 1 over the train windows.
    """
    mode = (cfg.balance or "none").lower()
    if mode not in BALANCE_MODES:
        raise ValueError(f"Unknown loss.balance {cfg.balance!r} (expected one of {BALANCE_MODES})")
    if cfg.balance_bin_width <= 0 or cfg.balance_max_weight < 1:
        raise ValueError("loss.balance_bin_width must be > 0 and balance_max_weight >= 1")
    t = torch.as_tensor(train_targets, dtype=torch.float64).reshape(-1)
    if t.numel() == 0:
        raise ValueError("age-balanced loss needs at least one train target")
    width = float(cfg.balance_bin_width)
    low = float(torch.floor(t.min() / width) * width)
    n_bins = int(torch.floor((t.max() - low) / width).item()) + 1
    idx = ((t - low) / width).floor().long().clamp(0, n_bins - 1)
    counts = torch.bincount(idx, minlength=n_bins).double()
    density = counts
    if cfg.balance_lds_sigma > 0:
        s = cfg.balance_lds_sigma / width
        half = int(max(1, round(3 * s)))
        offsets = torch.arange(-half, half + 1, dtype=torch.float64)
        kernel = torch.exp(-offsets ** 2 / (2 * s ** 2))
        kernel = kernel / kernel.sum()
        density = F.conv1d(counts.view(1, 1, -1), kernel.view(1, 1, -1), padding=half).view(-1)
    density = density.clamp_min(float(density[density > 0].min()))
    if mode == "none":
        weights = torch.ones_like(density)
    else:
        weights = 1.0 / density if mode == "inverse" else density.rsqrt()
    weights = weights.clamp(max=float(weights.min()) * cfg.balance_max_weight)
    weights = weights / weights[idx].mean()
    return low, width, weights.float()


class BalancedRegressionLoss(nn.Module):
    """Weights an element-wise mse/l1/huber loss by the target's age bin.

    The criterion sees z-scored targets, so ``target_stats`` maps them back to
    years before the bin lookup. Mixed (mixup) targets are weighted by the bin
    of the mixed age.
    """

    def __init__(self, base: nn.Module, low: float, width: float, weights: torch.Tensor,
                 target_stats: Optional[Tuple[float, float]] = None):
        super().__init__()
        if getattr(base, "reduction", None) != "none":
            raise ValueError("BalancedRegressionLoss needs a base loss with reduction='none'")
        self.base = base
        self.low, self.width = float(low), float(width)
        self.target_stats = target_stats
        self.register_buffer("weights", weights.float().reshape(-1))

    def sample_weights(self, target: torch.Tensor) -> torch.Tensor:
        years = target.float()
        if self.target_stats is not None:
            mean, std = self.target_stats
            years = years * std + mean
        idx = ((years - self.low) / self.width).floor().long().clamp(0, self.weights.numel() - 1)
        return self.weights.to(target.device)[idx]

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_elem = self.base(output, target)
        return (per_elem * self.sample_weights(target).reshape(per_elem.shape)).mean()

    def extra_repr(self) -> str:
        return (f"bins={self.weights.numel()} x {self.width:g}y from {self.low:g}, "
                f"weight range [{self.weights.min():.2f}, {self.weights.max():.2f}]")


def regression_output(criterion: nn.Module, output: torch.Tensor) -> torch.Tensor:
    """The scalar prediction the metrics score: the head output itself, or the
    soft-label criterion's expectation when the head is a bin distribution."""
    predict = getattr(criterion, "predict", None)
    return predict(output) if callable(predict) else output


def build_regression_criterion(cfg: Optional[LossConfig] = None,
                               target_stats: Optional[Tuple[float, float]] = None,
                               train_targets=None) -> nn.Module:
    """Select the downstream regression criterion.

      * ``"mse"``        -> ``nn.MSELoss``
      * ``"l1"``         -> ``nn.L1Loss``
      * ``"huber"``      -> ``nn.HuberLoss`` (default; robust to the long tails of a
        clinical age distribution)
      * ``"soft_label"`` -> :class:`SoftLabelRegressionLoss`; ``target_stats``
        (the loader's z-scoring mean/std) place its year-valued bins in the
        normalized target space.

    With ``cfg.balance`` set and ``train_targets`` (raw train-window ages)
    given, an mse/l1/huber loss is wrapped in :class:`BalancedRegressionLoss`.
    Without ``train_targets`` (the evaluation path) the loss stays unweighted.
    """
    cfg = cfg or LossConfig()
    name = (cfg.regression_loss or "huber").lower()
    balance = (cfg.balance or "none").lower()
    if balance != "none" and train_targets is not None:
        if name == "soft_label":
            raise ValueError("loss.balance is not supported with regression_loss=soft_label")
        base = {"mse": lambda: nn.MSELoss(reduction="none"),
                "l1": lambda: nn.L1Loss(reduction="none"),
                "huber": lambda: nn.HuberLoss(reduction="none", delta=cfg.huber_delta)}
        if name not in base:
            raise ValueError(
                f"Unknown regression_loss {cfg.regression_loss!r} (expected one of {REGRESSION_LOSSES})")
        low, width, weights = age_balance_weights(train_targets, cfg)
        return BalancedRegressionLoss(base[name](), low, width, weights, target_stats)
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta)
    if name == "soft_label":
        centers, sigma = soft_label_centers(cfg), cfg.soft_label_sigma
        if target_stats is not None:
            mean, std = target_stats
            centers, sigma = (centers - mean) / std, sigma / std
        return SoftLabelRegressionLoss(centers, sigma)
    raise ValueError(
        f"Unknown regression_loss {cfg.regression_loss!r} (expected one of {REGRESSION_LOSSES})")


def build_downstream_criterion(
    task: str,
    nb_classes: int,
    cfg: Optional[LossConfig] = None,
    target_stats: Optional[Tuple[float, float]] = None,
    train_targets=None,
) -> nn.Module:
    """Criterion for the downstream head, dispatched on the task.

    Single entry point for both the training and evaluation paths: ``evaluate``
    rebuilds its own criterion, and routing both through here is what stops a
    regression run from silently scoring ages with binary cross-entropy.
    """
    if task == "regression":
        return build_regression_criterion(cfg, target_stats, train_targets)
    return build_classification_criterion(nb_classes, cfg)
