"""Age-bin-balanced training sampler: each age bin equally likely, then each
case within the bin, then each sample within the case."""
from collections import Counter

import pytest
import torch

from labram.data.age_sampling import (
    DistributedWeightedSampler, age_bin_index, age_bin_sample_weights, build_age_bin_sampler,
)


class _AgeLoader(torch.utils.data.Dataset):
    """Minimal stand-in for TUABAgeLoader: ``files`` + ``ages`` + ``group_id``."""

    def __init__(self, cases):  # cases: [(case_id, age, n_samples)]
        self.files = [f"{c}_{i}.pkl" for c, _, n in cases for i in range(n)]
        self._ages = [a for _, a, n in cases for _ in range(n)]

    def __len__(self):
        return len(self.files)

    def ages(self):
        return list(self._ages)

    def group_id(self, filename):
        return filename.rsplit("_", 1)[0]


def test_bin_index_clips_the_inclusive_maximum_into_the_last_bin():
    assert age_bin_index(20, 5, 20, 10) == 0
    assert age_bin_index(24.9, 5, 20, 10) == 0
    assert age_bin_index(65, 5, 20, 10) == 9
    assert age_bin_index(70, 5, 20, 10) == 9
    assert age_bin_index(42, 5) == 8


def test_bins_then_cases_then_samples_are_uniform():
    # Bin 20-24: one case with 4 samples. Bin 25-29: two cases with 1 and 3.
    ages = [21] * 4 + [26] + [27] * 3
    cases = ["a"] * 4 + ["b"] + ["c"] * 3
    w, per_bin = age_bin_sample_weights(ages, cases, 5, (20, 70))
    assert per_bin == Counter({0: 1, 1: 2})
    assert sum(w) == pytest.approx(1.0)
    assert sum(w[:4]) == pytest.approx(0.5)            # bin 20-24
    assert w[4] == pytest.approx(0.25)                 # case b
    assert sum(w[5:]) == pytest.approx(0.25)           # case c
    assert w[5] == pytest.approx(0.25 / 3)


def test_invalid_width_raises():
    with pytest.raises(ValueError):
        age_bin_sample_weights([30], ["a"], 0)


def test_sampler_balances_a_skewed_cohort():
    # 90 old cases, 10 young: unbalanced shuffling would give ~90% old samples.
    cases = [(f"o{i}", 62, 5) for i in range(90)] + [(f"y{i}", 22, 5) for i in range(10)]
    ds = _AgeLoader(cases)
    sampler = build_age_bin_sampler(ds, 5, (20, 70), seed=0)
    assert len(sampler) == len(ds)
    idx = list(sampler)
    young = sum(ds.files[i].startswith("y") for i in idx) / len(idx)
    assert young == pytest.approx(0.5, abs=0.05)


def test_sampler_follows_concat_and_subset():
    a = _AgeLoader([("a", 22, 2)])
    b = _AgeLoader([("b", 62, 6)])
    ds = torch.utils.data.Subset(torch.utils.data.ConcatDataset([a, b]), [0, 2, 3])
    sampler = build_age_bin_sampler(ds, 5)
    assert sampler.weights.tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_sampler_rejects_loaders_without_ages():
    with pytest.raises(ValueError, match="age loaders"):
        build_age_bin_sampler(torch.utils.data.TensorDataset(torch.zeros(3)), 5)


def test_shards_partition_one_draw_and_epochs_redraw():
    w = [1.0] * 100
    full = DistributedWeightedSampler(w, seed=3)
    r0 = DistributedWeightedSampler(w, num_replicas=2, rank=0, seed=3)
    r1 = DistributedWeightedSampler(w, num_replicas=2, rank=1, seed=3)
    assert len(r0) == len(r1) == 50
    merged = [x for pair in zip(list(r0), list(r1)) for x in pair]
    assert merged == list(full)
    first = list(full)
    full.set_epoch(1)
    assert list(full) != first
