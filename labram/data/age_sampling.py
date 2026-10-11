# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Age-bin-balanced training sampler for brain-age regression: every age bin is
# equally likely, whatever its share of the cohort. See docs/age_regression.md.
# ---------------------------------------------------------

import logging
import math
from collections import Counter
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def age_bin_index(age: float, width: float, origin: float = 0.0,
                  n_bins: Optional[int] = None) -> int:
    """Bin of ``age`` among ``[origin + k*width, origin + (k+1)*width)``.

    With ``n_bins`` the index is clipped to ``[0, n_bins - 1]``, so the inclusive
    maximum of an age range (70 in [20, 70]) joins the last bin instead of
    opening a one-age bin of its own.
    """
    k = int(math.floor((age - origin) / width))
    if n_bins is not None:
        k = min(max(k, 0), n_bins - 1)
    return k


def age_bin_sample_weights(
    ages: Sequence[float],
    cases: Sequence[str],
    width: float,
    age_range: Sequence[float] = (),
) -> Tuple[List[float], Counter]:
    """Per-sample probabilities for a three-stage draw: an age bin uniformly
    among the non-empty bins, a case uniformly within the bin, a sample uniformly
    within the case.

    Returns ``(weights, cases_per_bin)``; the weights sum to 1. ``ages[i]`` and
    ``cases[i]`` describe sample ``i``; every sample of a case shares its age.
    """
    if width <= 0:
        raise ValueError(f"data.age_bin_width must be > 0, got {width}")
    if len(ages) != len(cases) or not ages:
        raise ValueError("age-bin sampling needs one age and one case per sample")
    origin, n_bins = 0.0, None
    if age_range:
        origin = float(age_range[0])
        n_bins = max(1, math.ceil((age_range[1] - age_range[0]) / width))
    bins = [age_bin_index(a, width, origin, n_bins) for a in ages]
    samples_per_case = Counter(cases)
    case_bin = dict(zip(cases, bins))
    cases_per_bin = Counter(case_bin.values())
    n_bins_used = len(cases_per_bin)
    weights = [1.0 / (n_bins_used * cases_per_bin[b] * samples_per_case[c])
               for b, c in zip(bins, cases)]
    return weights, cases_per_bin


def _ages_and_cases(dataset) -> Tuple[List[float], List[str]]:
    """Raw age and case id of every sample of ``dataset``, in index order.

    Follows ``ConcatDataset`` and ``Subset`` wrappers down to the age loaders
    (``TUABAgeLoader.ages`` / ``group_id``).
    """
    if isinstance(dataset, torch.utils.data.Subset):
        ages, cases = _ages_and_cases(dataset.dataset)
        return [ages[i] for i in dataset.indices], [cases[i] for i in dataset.indices]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        ages, cases = [], []
        for d in dataset.datasets:
            a, c = _ages_and_cases(d)
            ages += a
            cases += c
        return ages, cases
    if not hasattr(dataset, "ages"):
        raise ValueError("data.age_bin_sampling needs age loaders (TUAB_AGE); got "
                         f"{type(dataset).__name__}")
    return list(dataset.ages()), [dataset.group_id(f) for f in dataset.files]


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """``WeightedRandomSampler`` (with replacement) sharded across ranks.

    Every rank draws the same ``num_replicas * num_samples`` indices from a
    generator seeded with ``seed + epoch`` and keeps every ``num_replicas``-th, so
    the shards are disjoint draws of one sequence. Call :meth:`set_epoch` each
    epoch for a fresh draw.
    """

    def __init__(self, weights: Sequence[float], num_samples: Optional[int] = None,
                 num_replicas: int = 1, rank: int = 0, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        total = num_samples if num_samples is not None else len(self.weights)
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = math.ceil(total / num_replicas)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples * self.num_replicas,
                                replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_age_bin_sampler(dataset, width: float, age_range: Sequence[float] = (),
                          num_replicas: int = 1, rank: int = 0,
                          seed: int = 0) -> DistributedWeightedSampler:
    """Training sampler that makes every ``width``-year age bin equally likely.

    An epoch draws ``len(dataset)`` samples with replacement, so its length (and
    the LR schedule built on it) matches the default shuffled sampler.
    """
    ages, cases = _ages_and_cases(dataset)
    weights, cases_per_bin = age_bin_sample_weights(ages, cases, width, age_range)
    origin = float(age_range[0]) if age_range else 0.0
    logger.info(
        "Age-bin sampling: %d bins of %g years, cases per bin: %s",
        len(cases_per_bin), width,
        ", ".join(f"{origin + b * width:g}+: {n}" for b, n in sorted(cases_per_bin.items())))
    return DistributedWeightedSampler(weights, num_samples=len(weights),
                                      num_replicas=num_replicas, rank=rank, seed=seed)
