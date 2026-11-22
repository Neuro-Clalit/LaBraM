# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Recording bags for multiple-instance (MIL) training.
# ---------------------------------------------------------
"""One dataset item per recording: a bag of its windows (``mil.*``).

:class:`RecordingBagDataset` wraps a window-selected dataset (any of the
window loaders, concatenated or not). Item ``r`` is

    (x [K, C, L], y, stem, minutes [K])

* train: ``K`` windows drawn at random without replacement from the
  recording's selected windows (fresh every epoch: the draw uses the worker's
  torch RNG, which the DataLoader reseeds per epoch); ``k_choices`` draws ``K``
  per item; a recording with fewer windows gives all of them;
* eval: all of the recording's windows in time order (``max_windows`` caps).

``minutes`` is each window's offset from the recording's first selected
window, in minutes (window index / 6 for 10 s windows). ``y`` is the
recording's target (the first window's; window targets are constant within a
recording). :func:`collate_bags` pads a batch to its largest bag and returns
``(x [B, K, C, L], y, stems, minutes [B, K], mask [B, K])``.
"""

import os
from collections import abc, defaultdict
from typing import List, Optional, Sequence

import numpy as np
import torch

from labram.data.tuh_metadata import recording_stem
from labram.data.window_selection import window_index


class CompactStrings(abc.Sequence):
    """A read-only list of strings kept in one numpy bytes array.

    A data-loader worker forked from the main process shares its memory, but merely
    reading a Python object writes its reference count, so the kernel copies every page
    a worker touches. A per-window list of millions of file names (and per-window tuples)
    is copied that way by every worker, about 2 GB each per epoch: 10 workers outgrow a
    32 GB instance. One numpy buffer has no per-element objects to touch.
    """

    def __init__(self, items):
        self._a = np.array([str(s).encode() for s in items], dtype=bytes)

    def __len__(self):
        return len(self._a)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [b.decode() for b in self._a[i]]
        return self._a[i].decode()

    def __iter__(self):
        for b in self._a:
            yield b.decode()


def compact_files(dataset) -> None:
    """Replace a window dataset's ``files`` list by :class:`CompactStrings` (also inside
    ``ConcatDataset`` / ``Subset``); a no-op for anything else."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for d in dataset.datasets:
            compact_files(d)
    elif isinstance(dataset, torch.utils.data.Subset):
        compact_files(dataset.dataset)
    elif isinstance(getattr(dataset, "files", None), list):
        dataset.files = CompactStrings(dataset.files)


def _files(dataset) -> List[str]:
    from labram.data.samplers import _files as sampler_files
    return list(sampler_files(dataset))


class RecordingBagDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, k: int = 8, train: bool = True,
                 k_choices: Sequence[int] = (), max_windows: int = 0, window_minutes: float = 10 / 60,
                 repeats: int = 1):
        self.base = dataset
        # repeats > 1 lists every recording that many times per epoch, each item a
        # fresh random draw of its windows (the phased window phase, mil.window_bag_repeats)
        self.repeats = max(1, int(repeats))
        self.k, self.train = int(k), bool(train)
        self.k_choices = [int(v) for v in k_choices]
        self.max_windows = int(max_windows)
        files = _files(dataset)
        if len(files) != len(dataset):
            raise ValueError(f"bag dataset: {len(files)} file names for {len(dataset)} items")
        by_rec = defaultdict(list)
        for i, f in enumerate(files):
            by_rec[recording_stem(f)].append((window_index(f), i))
        self.stems = list(by_rec)
        # every recording's (window index, item) pairs, sorted, in one array + offsets: no
        # per-window Python objects for the forked loader workers to copy (CompactStrings)
        rows = [sorted(by_rec[s]) for s in self.stems]
        self._off = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
        self._pairs = np.array([p for x in rows for p in x], dtype=np.int64).reshape(-1, 2)
        del rows, by_rec, files
        compact_files(dataset)

    @property
    def windows(self):
        """Per recording, the ``[n, 2]`` array of its (window index, item) pairs."""
        return [self._rows(r) for r in range(len(self.stems))]

    def _rows(self, r: int) -> np.ndarray:
        return self._pairs[self._off[r]:self._off[r + 1]]

    def __len__(self):
        return len(self.stems) * self.repeats

    def bag_indices(self, r: int):
        rows = self._rows(r)
        if self.train:
            k = int(self.k_choices[torch.randint(len(self.k_choices), (1,)).item()]) if self.k_choices else self.k
            if len(rows) > k:
                pick = sorted(torch.randperm(len(rows))[:k].tolist())
                rows = rows[pick]
        elif self.max_windows and len(rows) > self.max_windows:
            rows = rows[:self.max_windows]
        return [(int(w), int(i)) for w, i in rows]

    def __getitem__(self, r: int):
        r = r % len(self.stems)
        rows = self.bag_indices(r)
        first = int(self._pairs[self._off[r], 0])
        xs, y = [], None
        for widx, i in rows:
            item = self.base[i]
            xs.append(torch.as_tensor(item[0]).float())
            if y is None:
                y = torch.as_tensor(item[1]).float()
        minutes = torch.tensor([(w - first) / 6.0 for w, _ in rows], dtype=torch.float32)
        return torch.stack(xs), y, self.stems[r], minutes


def collate_bags(batch):
    k = max(b[0].shape[0] for b in batch)
    c, length = batch[0][0].shape[1:]
    x = torch.zeros(len(batch), k, c, length)
    minutes = torch.zeros(len(batch), k)
    mask = torch.zeros(len(batch), k, dtype=torch.bool)
    for j, (xb, _, _, mb) in enumerate(batch):
        n = xb.shape[0]
        x[j, :n], minutes[j, :n], mask[j, :n] = xb, mb, True
    y = torch.stack([b[1] for b in batch])
    return x, y, [b[2] for b in batch], minutes, mask


def collate_bag_windows(batch):
    """Flatten bags into a window batch ``(x [N, C, L], y [N, ...], recording ids)``:
    every window is its own sample with its recording's target. Used by the window
    phase of ``mil.window_epochs``, which reads recording by recording (``k`` windows
    of each) because random single-window reads stream slowly from S3."""
    xs, ys, ids = [], [], []
    for xb, yb, stem, _ in batch:
        n = xb.shape[0]
        xs.append(xb)
        ys.append(yb.unsqueeze(0).expand(n, *yb.shape))
        ids.extend([stem] * n)
    return torch.cat(xs), torch.cat(ys), ids


def wrap_bags(dataset, mil_cfg, train: bool) -> Optional[RecordingBagDataset]:
    if dataset is None:
        return None
    return RecordingBagDataset(dataset, k=mil_cfg.bag_size, train=train,
                               k_choices=mil_cfg.bag_size_choices,
                               max_windows=0 if train else mil_cfg.eval_max_windows,
                               repeats=getattr(mil_cfg, "bag_repeats", 1) if train else 1)
