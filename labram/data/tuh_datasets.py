# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# TUH-EEG (TUAB / TUEV) torch.utils.data.Dataset wrappers and split assembly.
# ---------------------------------------------------------

import json
import logging
import os
import pickle
from collections import OrderedDict
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
        # Consecutive windows concatenated into one sample, starting at each
        # listed file (set by labram.data.window_selection for inputs longer
        # than one pickle).
        self.windows_per_item = 1

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

    def _window_files(self, filename: str) -> list:
        """``filename`` plus the ``windows_per_item - 1`` windows after it."""
        if self.windows_per_item <= 1:
            return [filename]
        directory, base = os.path.split(filename)
        stem, index = base[:-4].rsplit(self._recording_sep, 1)
        return [os.path.join(directory, f"{stem}{self._recording_sep}{int(index) + j}.pkl")
                for j in range(self.windows_per_item)]

    def _load(self, index):
        filename = self.files[index]
        samples = []
        for name in self._window_files(filename):
            with open(os.path.join(self.root, name), "rb") as fh:
                samples.append(pickle.load(fh))
        sample = samples[0]
        X = np.concatenate([s[self._signal_key] for s in samples], axis=-1)
        if self.sampling_rate != self.default_rate:
            X = resample(X, self._duration_sec * len(samples) * self.sampling_rate, axis=-1)
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


NPY_MANIFEST = "manifest.json"
WINDOW_SAMPLES = 2000   # one 10 s window at 200 Hz


class TUABAgeNpyLoader(TUABAgeLoader):
    """:class:`TUABAgeLoader` over the per-recording float32 ``.npy`` format
    (``dataset_maker/make_TUAB_npy.py``).

    Items keep the pickle loader's names (``<split>/<stem>_<k>.pkl``), so the
    age split, window selection, cross-validation and split reuse work
    unchanged; only the read differs: window ``k`` is the contiguous slice
    ``[2000 k, 2000 k + L)`` of a memory-mapped ``recordings/<stem>.npy``
    (``[T, 23]``), bit-identical to the pickle cast to float32.

    ``random_crop`` (training only) moves each sample to a uniformly random
    start within half a sample length of its grid position, kept inside
    ``sample_bounds[stem]`` (the trimmed range set by window selection), so
    every epoch sees different crops of the same recordings.
    """

    MMAP_CACHE = 256   # open memory maps per process (each holds a file descriptor)

    def __init__(self, root, files, sampling_rate=200, *, age_lookup=None,
                 target_stats=None):
        super().__init__(root, files, sampling_rate, age_lookup=age_lookup,
                         target_stats=target_stats)
        self.random_crop = False
        self.sample_bounds = {}
        self._mmaps = OrderedDict()
        self._manifest = None

    def manifest(self):
        if self._manifest is None:
            with open(os.path.join(self.root, NPY_MANIFEST)) as fh:
                self._manifest = json.load(fh)
        return self._manifest

    def _n_samples(self, stem):
        if not hasattr(self, "_lengths"):
            self._lengths = {r["stem"]: r["n_samples"] for r in self.manifest()["recordings"]}
        return self._lengths[stem]

    def window_inventory(self):
        """``(n_windows by (dir, stem), available window names)`` from the
        manifest, so window selection never lists 409k names on disk."""
        n_windows, available = {}, set()
        for row in self.manifest()["recordings"]:
            n_windows[(row["source_split"], row["stem"])] = row["n_windows"]
            available.update(f"{row['source_split']}/{row['stem']}_{k}.pkl"
                             for k in range(row["n_windows"]))
        return n_windows, available

    def _recording(self, stem):
        rec = self._mmaps.pop(stem, None)
        if rec is None:
            rec = np.load(os.path.join(self.root, "recordings", f"{stem}.npy"), mmap_mode="r")
            if len(self._mmaps) >= self.MMAP_CACHE:
                self._mmaps.popitem(last=False)
        self._mmaps[stem] = rec
        return rec

    def __getstate__(self):
        # DataLoader workers each open their own memory maps.
        state = dict(self.__dict__)
        state["_mmaps"] = OrderedDict()
        return state

    def _load(self, index):
        filename = self.files[index]
        stem, k = os.path.basename(filename)[:-4].rsplit(self._recording_sep, 1)
        length = WINDOW_SAMPLES * self.windows_per_item
        start = int(k) * WINDOW_SAMPLES
        if self.random_crop:
            lo, hi = self.sample_bounds.get(stem) or (0, self._n_samples(stem))
            lo, hi = max(lo, start - length // 2), min(hi - length, start + length // 2)
            if hi > lo:
                start = int(torch.randint(lo, hi + 1, (1,)))
        X = np.ascontiguousarray(self._recording(stem)[start:start + length].T)
        if X.shape[-1] != length:
            raise ValueError(f"{filename}: crop [{start}, {start + length}) is out of range")
        if self.sampling_rate != self.default_rate:
            X = resample(X, self._duration_sec * self.windows_per_item * self.sampling_rate,
                         axis=-1)
        Y = self._label_fn(None, filename)
        if self.return_id:
            return torch.from_numpy(X), Y, self.group_id(filename)
        return torch.from_numpy(X), Y


def _age_data_dir(root, data_format):
    """The window directory for a format: ``<root>/processed[_npy]`` when it
    exists, else ``root`` itself (a data_path pointing straight at it)."""
    sub = "processed_npy" if data_format == "npy" else "processed"
    return os.path.join(root, sub) if os.path.isdir(os.path.join(root, sub)) else root


def prepare_TUAB_age_dataset(root, *, normalize_targets: bool = True,
                             data_format: str = "pickle"):
    """Build subject-disjoint TUAB age-regression splits.

    Reuses the window pickles produced by ``dataset_maker/make_TUAB.py`` -- only
    the split assignment and the target differ. Reads ``processed/age_split.json``
    when present, otherwise builds the partition on the fly.

    Returns ``(train, test, val, target_stats)``; the tuple order matches
    :func:`prepare_TUAB_dataset`.
    """
    if data_format not in ("pickle", "npy"):
        raise ValueError(f"data.data_format must be 'pickle' or 'npy', got {data_format!r}")
    processed = _age_data_dir(root, data_format)
    if data_format == "npy" and not os.path.isfile(os.path.join(processed, NPY_MANIFEST)):
        raise FileNotFoundError(
            f"No {NPY_MANIFEST} under {processed}; build it with "
            f"dataset_maker/make_TUAB_npy.py convert + merge")
    loader_cls = TUABAgeNpyLoader if data_format == "npy" else TUABAgeLoader
    ages = load_age_lookup_for(processed)

    split_path = find_age_split(processed)
    if split_path is not None:
        split = load_age_split(split_path)
        logger.info("Using age split %s", split_path)
    else:
        logger.info("No %s found; building the age split in memory", SPLIT_FILENAME)
        split = build_age_split(processed, ages)

    def loader(name, stats):
        return loader_cls(
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
