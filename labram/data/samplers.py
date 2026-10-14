# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training sampler that draws a fresh per-recording subset of samples each
# epoch (data.train_minutes_per_epoch).
# ---------------------------------------------------------
"""Per-recording subsampling of the training samples, redrawn every epoch.

A recording contributes up to 20 min (120 samples of 10 s) of highly
redundant windows to brain-age training. :class:`RecordingSubsetSampler`
instead yields, each epoch, at most ``k`` samples per recording drawn at random
(seeded by ``seed + epoch``) from that recording's selected samples, all
recordings shuffled together -- so an epoch is shorter while every recording
and patient still appears, and successive epochs see different windows.
Sharded across ``num_replicas`` like ``DistributedSampler`` (padded by
repetition to an equal count per rank).
"""
import math
from collections import OrderedDict
from typing import List

import torch
import torch.utils.data

from labram.data.tuh_metadata import recording_stem


def _files(dataset) -> List[str]:
    if hasattr(dataset, "files"):
        return list(dataset.files)
    if isinstance(dataset, torch.utils.data.Subset):       # e.g. a debug subset
        inner = _files(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return [f for d in dataset.datasets for f in _files(d)]
    raise TypeError(f"data.train_minutes_per_epoch needs window loaders with .files, "
                    f"got {type(dataset).__name__}")


class RecordingSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, per_recording: int, num_replicas: int = 1, rank: int = 0,
                 seed: int = 0):
        if per_recording < 1:
            raise ValueError("per_recording must be >= 1")
        groups = OrderedDict()
        for i, f in enumerate(_files(dataset)):
            groups.setdefault(recording_stem(f), []).append(i)
        self.groups = list(groups.values())
        self.k = int(per_recording)
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.epoch = 0
        self.total = sum(min(len(g), self.k) for g in self.groups)
        self.num_samples = math.ceil(self.total / num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def indices(self) -> List[int]:
        """This epoch's samples, before sharding."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        chosen = []
        for idx in self.groups:
            if len(idx) <= self.k:
                chosen += idx
            else:
                chosen += [idx[int(p)] for p in torch.randperm(len(idx), generator=g)[:self.k]]
        return [chosen[int(p)] for p in torch.randperm(len(chosen), generator=g)]

    def __iter__(self):
        out = self.indices()
        size = self.num_samples * self.num_replicas
        out += out[:size - len(out)]
        return iter(out[self.rank:size:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        return (f"RecordingSubsetSampler({len(self.groups)} recordings, <= {self.k} samples each, "
                f"{self.total} per epoch)")
