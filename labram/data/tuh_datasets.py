# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# TUH-EEG (TUAB / TUEV) torch.utils.data.Dataset wrappers and split assembly.
# ---------------------------------------------------------

import logging
import os
import pickle
from typing import Callable

import numpy as np
import torch
import torch.utils.data
from scipy.signal import resample

from labram.data.age_splits import (
    SPLIT_FILENAME,
    build_age_split,
    find_age_split,
    load_age_split,
)
from labram.data.tuh_metadata import (
    filter_files_with_age,
    load_age_lookup_for,
    recording_stem,
)

logger = logging.getLogger(__name__)


class TUHLoader(torch.utils.data.Dataset):
    """Parameterised loader for TUH-EEG pickle datasets (TUAB, TUEV, etc.).

    Args:
        root: Directory containing the pickle files.
        files: List of file names within *root*.
        sampling_rate: Target sampling rate; data is resampled if it differs from
            the default 200 Hz recorded rate.
        signal_key: Key used to read the EEG array from each pickle dict.
        duration_sec: Recording duration in seconds at the default rate (used to
            compute the resample target length).
        label_fn: Callable ``(sample, filename) -> label`` mapping a pickle dict
            to its target. The filename is passed because some targets (e.g. the
            patient age, which lives in the EDF header rather than the pickle)
            are keyed by recording rather than stored per window.
    """

    def __init__(
        self,
        root: str,
        files: list,
        sampling_rate: int = 200,
        *,
        signal_key: str,
        duration_sec: int,
        label_fn: Callable,
        recording_sep: str = "_",
        return_id: bool = False,
        group_by: str = "recording",
    ):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self._signal_key = signal_key
        self._duration_sec = duration_sec
        self._label_fn = label_fn
        # Windows from one recording share a filename prefix; the trailing
        # ``<recording_sep><window_index>`` distinguishes windows. ``return_id``
        # makes __getitem__ yield a per-window case id (recording or subject) so
        # inference can aggregate window predictions per case.
        self._recording_sep = recording_sep
        self.return_id = return_id
        self.group_by = group_by

    def __len__(self) -> int:
        return len(self.files)

    def group_id(self, filename: str) -> str:
        """Case id for a window file: the recording (default) or subject.

        A recording strips the trailing window index (``..._<i>`` for TUAB,
        ``...-<i>`` for TUEV); a subject is the leading underscore-token. A
        filename may be a path relative to ``root``, so ids come off the
        basename.
        """
        base = os.path.basename(filename)
        base = base[:-4] if base.endswith(".pkl") else base
        if self.group_by == "subject":
            return base.split("_")[0]
        return base.rsplit(self._recording_sep, 1)[0]

    def _load(self, index):
        filename = self.files[index]
        path = os.path.join(self.root, filename)
        with open(path, "rb") as fh:
            sample = pickle.load(fh)
        X = sample[self._signal_key]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self._duration_sec * self.sampling_rate, axis=-1)
        Y = self._label_fn(sample, filename)
        if self.return_id:
            return torch.FloatTensor(X), Y, self.group_id(filename)
        return torch.FloatTensor(X), Y

    def __getitem__(self, index):
        # A corrupt/truncated pickle (e.g. written during an interrupted or
        # out-of-space preprocessing run) must not abort training: skip to the
        # next readable sample, wrapping around, and only fail if none load.
        n = len(self.files)
        for offset in range(n):
            i = (index + offset) % n
            try:
                return self._load(i)
            except (pickle.UnpicklingError, EOFError, OSError, ValueError, KeyError) as exc:
                if offset == 0:
                    logger.warning(
                        "Skipping unreadable sample %s: %s: %s",
                        self.files[i], type(exc).__name__, exc)
                continue
        raise RuntimeError(
            f"No readable samples in {self.root}: all {n} files failed to load")


class TUABLoader(TUHLoader):
    """Loader for the TUH Abnormal EEG Corpus (TUAB) pickle files."""

    def __init__(self, root: str, files: list, sampling_rate: int = 200):
        super().__init__(
            root, files, sampling_rate,
            signal_key="X",
            duration_sec=10,
            label_fn=lambda s, _filename: s["y"],
        )


class TUEVLoader(TUHLoader):
    """Loader for the TUH EEG Events Corpus (TUEV) pickle files."""

    def __init__(self, root: str, files: list, sampling_rate: int = 200):
        super().__init__(
            root, files, sampling_rate,
            signal_key="signal",
            duration_sec=5,
            label_fn=lambda s, _filename: int(s["label"][0] - 1),
            recording_sep="-",
        )


class TUABAgeLoader(TUHLoader):
    """TUAB windows targeting the patient's age in years (brain-age regression).

    The age is not in the window pickles -- it comes from the EDF header, joined
    on the recording stem via an ``age_metadata.json`` sidecar. When
    ``age_lookup`` is not supplied it is resolved by searching *root* and its
    parents for that sidecar; cross-validation rebuilds loaders positionally as
    ``type(src)(root, files, sampling_rate)``, so the lookup must be recoverable
    from the root alone.

    Targets are z-scored with ``target_stats`` (the train split's mean/std) when
    given: the classification head is initialised with ``init_scale=0.001``, so a
    raw target near 50 would start the run with an enormous loss. Metrics
    de-normalise before reporting, so MAE stays in years.
    """

    _lookup_cache = {}

    def __init__(
        self,
        root: str,
        files: list,
        sampling_rate: int = 200,
        *,
        age_lookup=None,
        target_stats=None,
    ):
        if age_lookup is None:
            age_lookup = self._resolve_lookup(root)
        self.age_lookup = age_lookup
        self.target_stats = target_stats

        # Filter up front: __getitem__ swallows KeyError and substitutes another
        # window, so an unresolvable age must never reach _label_fn.
        kept = filter_files_with_age(files, age_lookup)
        dropped = len(files) - len(kept)
        if dropped:
            logger.info(
                "Dropping %d/%d window(s) in %s with no usable age",
                dropped, len(files), root)

        super().__init__(
            root, kept, sampling_rate,
            signal_key="X",
            duration_sec=10,
            label_fn=self._age_target,
        )

    @classmethod
    def _resolve_lookup(cls, root: str):
        key = os.path.abspath(root)
        if key not in cls._lookup_cache:
            cls._lookup_cache[key] = load_age_lookup_for(root)
        return cls._lookup_cache[key]

    def _age_target(self, _sample, filename: str) -> torch.Tensor:
        age = self.age_lookup[recording_stem(filename)]
        if self.target_stats is not None:
            mean, std = self.target_stats
            age = (age - mean) / std
        return torch.tensor(age, dtype=torch.float32)

    def ages(self) -> list:
        """Raw (un-normalised) age of every window, in ``files`` order."""
        return [self.age_lookup[recording_stem(f)] for f in self.files]


def prepare_TUAB_age_dataset(root, *, normalize_targets: bool = True):
    """Build subject-disjoint TUAB age-regression splits.

    Reuses the window pickles produced by ``dataset_maker/make_TUAB.py`` -- only
    the split assignment and the target differ. Reads ``processed/age_split.json``
    when present, otherwise builds the partition on the fly.

    Returns ``(train, test, val, target_stats)``; the tuple order matches
    :func:`prepare_TUAB_dataset`.
    """
    processed = os.path.join(root, "processed") if os.path.isdir(
        os.path.join(root, "processed")) else root
    ages = load_age_lookup_for(processed)

    split_path = find_age_split(processed)
    if split_path is not None:
        split = load_age_split(split_path)
        logger.info("Using age split %s", split_path)
    else:
        logger.info("No %s found; building the age split in memory", SPLIT_FILENAME)
        split = build_age_split(processed, ages)

    def loader(name, stats):
        return TUABAgeLoader(
            processed, split.files[name], age_lookup=ages, target_stats=stats)

    train_ages = [ages[recording_stem(f)] for f in split.files["train"]]
    target_stats = None
    if normalize_targets:
        mean = sum(train_ages) / len(train_ages)
        var = sum((a - mean) ** 2 for a in train_ages) / max(1, len(train_ages) - 1)
        target_stats = (mean, max(var ** 0.5, 1e-6))
        logger.info("Age target normalization: mean=%.2f std=%.2f", *target_stats)

    return (
        loader("train", target_stats),
        loader("test", target_stats),
        loader("val", target_stats),
        target_stats,
    )


def prepare_TUEV_dataset(root):
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files)
    test_dataset = TUEVLoader(os.path.join(root, "processed_test"), test_files)
    val_dataset = TUEVLoader(os.path.join(root, "processed_eval"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root):
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset
