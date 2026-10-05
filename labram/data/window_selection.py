# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Recording/window selection for the TUAB-family fine-tuning loaders: case-type
# filter, start/end trimming, longer model inputs and a per-recording evaluation
# budget. See docs/age_regression.md.
# ---------------------------------------------------------

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch.utils.data

from labram.data.tuh_metadata import CASE_LABELS, load_label_lookup_for, recording_stem

logger = logging.getLogger(__name__)

# Every TUAB window pickle (dataset_maker/make_TUAB.py) holds 10 s; window ``i``
# of a recording covers [10 i, 10 i + 10) s.
PICKLE_WINDOW_SEC = 10
MAX_WINDOW_SEC = 60
CASE_FILTERS = ("all",) + CASE_LABELS


@dataclass(frozen=True)
class WindowSelection:
    """Which windows of each recording a split uses, and how many per sample.

    A sample is ``window_sec / 10`` consecutive pickles. Sample starts lie on a
    fixed grid per recording -- every ``window_sec`` seconds from the end of the
    trimmed start -- so applying a selection to an already-selected file list
    (a reused ``data_split.json``, a cross-validation fold) is a no-op.
    """

    case_filter: str = "all"
    trim_start_sec: float = 0.0
    trim_end_sec: float = 0.0
    window_sec: int = PICKLE_WINDOW_SEC
    eval_minutes: float = 0.0

    def __post_init__(self):
        if self.case_filter not in CASE_FILTERS:
            raise ValueError(f"data.case_filter must be one of {CASE_FILTERS}, "
                             f"got {self.case_filter!r}")
        if self.trim_start_sec < 0 or self.trim_end_sec < 0 or self.eval_minutes < 0:
            raise ValueError("data.trim_start_sec / trim_end_sec / eval_minutes must be >= 0")
        if (self.window_sec % PICKLE_WINDOW_SEC
                or not PICKLE_WINDOW_SEC <= self.window_sec <= MAX_WINDOW_SEC):
            raise ValueError(
                f"data.window_sec must be a multiple of {PICKLE_WINDOW_SEC} in "
                f"[{PICKLE_WINDOW_SEC}, {MAX_WINDOW_SEC}] (the pickles are "
                f"{PICKLE_WINDOW_SEC} s windows), got {self.window_sec}")
        if 0 < self.eval_minutes * 60 < self.window_sec:
            raise ValueError(f"data.eval_minutes={self.eval_minutes} is shorter than "
                             f"one {self.window_sec} s sample")

    @classmethod
    def from_data_config(cls, data_cfg) -> "WindowSelection":
        return cls(
            case_filter=getattr(data_cfg, "case_filter", "all"),
            trim_start_sec=float(getattr(data_cfg, "trim_start_sec", 0.0)),
            trim_end_sec=float(getattr(data_cfg, "trim_end_sec", 0.0)),
            window_sec=int(getattr(data_cfg, "window_sec", PICKLE_WINDOW_SEC)),
            eval_minutes=float(getattr(data_cfg, "eval_minutes", 0.0)),
        )

    @property
    def is_default(self) -> bool:
        return self == WindowSelection()

    @property
    def windows_per_sample(self) -> int:
        return self.window_sec // PICKLE_WINDOW_SEC

    @property
    def trim_start_windows(self) -> int:
        return math.ceil(self.trim_start_sec / PICKLE_WINDOW_SEC)

    @property
    def trim_end_windows(self) -> int:
        return math.ceil(self.trim_end_sec / PICKLE_WINDOW_SEC)

    @property
    def eval_windows(self) -> int:
        """Per-recording evaluation budget in 10 s windows (0 = unlimited)."""
        return int(self.eval_minutes * 60 // PICKLE_WINDOW_SEC)


def window_index(filename: str, sep: str = "_") -> int:
    """Window index of ``<stem><sep><index>.pkl`` (a bare name or a relative path)."""
    base = os.path.basename(filename)
    base = base[:-4] if base.endswith(".pkl") else base
    return int(base.rsplit(sep, 1)[1])


def _window_path(filename: str, index: int, sep: str = "_") -> str:
    """The sibling window ``index`` of ``filename``, keeping its directory."""
    directory = os.path.dirname(filename)
    name = f"{recording_stem(filename, sep=sep)}{sep}{index}.pkl"
    return os.path.join(directory, name) if directory else name


def scan_recordings(root: str, files: Iterable[str],
                    sep: str = "_") -> Tuple[Dict[Tuple[str, str], int], Set[str]]:
    """List the window directories ``files`` live in (relative to ``root``).

    Returns ``(n_windows, available)``: windows per ``(directory, stem)`` --
    one past the highest index on disk, i.e. the recording length in 10 s
    windows -- and the set of window files that exist. Read from disk rather
    than from ``files`` so trimming the end stays correct (and idempotent) on an
    already-trimmed list.
    """
    n_windows: Dict[Tuple[str, str], int] = {}
    available: Set[str] = set()
    for directory in sorted({os.path.dirname(f) for f in files}):
        for name in os.listdir(os.path.join(root, directory)):
            if not name.endswith(".pkl"):
                continue
            rel = os.path.join(directory, name) if directory else name
            available.add(rel)
            key = (directory, recording_stem(name, sep=sep))
            n_windows[key] = max(n_windows.get(key, 0), window_index(name, sep) + 1)
    return n_windows, available


def select_files(
    files: Iterable[str],
    selection: WindowSelection,
    *,
    n_windows: Dict[Tuple[str, str], int],
    available: Set[str],
    labels: Optional[Dict[str, str]] = None,
    is_eval: bool = False,
    sep: str = "_",
) -> List[str]:
    """The sample-start files of ``files`` that ``selection`` keeps.

    A file starts a sample when its recording passes the case filter, it lies on
    the sample grid inside the trimmed recording, the sample's windows all exist
    on disk, and -- for an evaluation split -- the sample ends within the first
    ``eval_minutes`` after the trimmed start.
    """
    k = selection.windows_per_sample
    start = selection.trim_start_windows
    budget = selection.eval_windows if is_eval else 0
    kept = []
    for f in files:
        stem = recording_stem(f, sep=sep)
        if selection.case_filter != "all" and (labels or {}).get(stem) != selection.case_filter:
            continue
        i = window_index(f, sep)
        if i < start or (i - start) % k:
            continue
        end = n_windows.get((os.path.dirname(f), stem), 0) - selection.trim_end_windows
        if i + k > end or (budget and i + k > start + budget):
            continue
        if all(_window_path(f, i + j, sep) in available for j in range(k)):
            kept.append(f)
    return kept


def _select_leaf(leaf, selection: WindowSelection, is_eval: bool, split: str) -> None:
    """Narrow one loader's ``files`` in place and set its samples' length."""
    sep = getattr(leaf, "_recording_sep", "_")
    labels = (load_label_lookup_for(leaf.root)
              if selection.case_filter != "all" else None)
    n_windows, available = scan_recordings(leaf.root, leaf.files, sep)
    before_files = leaf.files
    leaf.files = select_files(before_files, selection, n_windows=n_windows,
                              available=available, labels=labels,
                              is_eval=is_eval, sep=sep)
    leaf.windows_per_item = selection.windows_per_sample
    recordings = lambda fs: len({recording_stem(f, sep=sep) for f in fs})
    logger.info(
        "window selection [%s]: %d -> %d sample(s) of %d s, %d -> %d recording(s)",
        split, len(before_files), len(leaf.files), selection.window_sec,
        recordings(before_files), recordings(leaf.files))
    if not leaf.files:
        raise ValueError(f"window selection left the {split} split empty "
                         f"(selection={selection})")


def _apply_to_dataset(dataset, selection: WindowSelection, is_eval: bool, split: str):
    if dataset is None:
        return None
    if isinstance(dataset, list):
        return [_apply_to_dataset(d, selection, is_eval, split) for d in dataset]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # Rebuilt: ConcatDataset caches its children's lengths at construction.
        return torch.utils.data.ConcatDataset(
            [_apply_to_dataset(d, selection, is_eval, split) for d in dataset.datasets])
    if (not hasattr(dataset, "windows_per_item")
            or getattr(dataset, "_duration_sec", None) != PICKLE_WINDOW_SEC):
        raise TypeError(
            f"window selection needs TUAB {PICKLE_WINDOW_SEC} s window loaders; got "
            f"{type(dataset).__name__} (and apply it before any Subset wrapping)")
    _select_leaf(dataset, selection, is_eval, split)
    return dataset


def _leaves(dataset) -> list:
    if dataset is None:
        return []
    if isinstance(dataset, list):
        return [leaf for d in dataset for leaf in _leaves(d)]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return [leaf for d in dataset.datasets for leaf in _leaves(d)]
    return [dataset]


def _train_target_stats(train) -> Optional[Tuple[float, float]]:
    """(mean, sample std) of the train samples' raw targets, when the loaders
    expose them (``TUABAgeLoader.ages``)."""
    ages = [a for leaf in _leaves(train) if hasattr(leaf, "ages") for a in leaf.ages()]
    if len(ages) < 2:
        return None
    mean = sum(ages) / len(ages)
    var = sum((a - mean) ** 2 for a in ages) / (len(ages) - 1)
    return mean, max(var ** 0.5, 1e-6)


def apply_window_selection(bundle, selection: WindowSelection):
    """Apply ``selection`` to a :class:`DatasetBundle` in place and return it.

    Case filter and trimming apply to every split; ``eval_minutes`` to val and
    test only. A regression bundle's target z-scoring stats are recomputed from
    the selected train samples, since the filter can shift the age distribution
    (e.g. normal-only skews younger).
    """
    if selection.is_default:
        return bundle
    if selection.case_filter != "all" and not bundle.is_regression:
        raise ValueError(
            f"data.case_filter={selection.case_filter!r} would leave a classification "
            f"task with a single class; it is meant for age regression (TUAB_AGE)")
    bundle.train = _apply_to_dataset(bundle.train, selection, False, "train")
    bundle.val = _apply_to_dataset(bundle.val, selection, True, "val")
    bundle.test = _apply_to_dataset(bundle.test, selection, True, "test")
    if bundle.target_stats is not None:
        stats = _train_target_stats(bundle.train)
        if stats is not None:
            bundle.target_stats = stats
            for split in (bundle.train, bundle.val, bundle.test):
                for leaf in _leaves(split):
                    if hasattr(leaf, "target_stats"):
                        leaf.target_stats = stats
            logger.info("Age target normalization (selected train): mean=%.2f std=%.2f",
                        *stats)
    return bundle
