# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Subject-disjoint train/val/test partition of the preprocessed TUH windows for
# age regression. See docs/age_regression.md.
# ---------------------------------------------------------

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from labram.data.tuh_metadata import filter_files_with_age, recording_stem

# Plain stdlib logging, not labram.utils.get_logger: labram.utils imports
# labram.data, so a labram.utils import here would be circular.
logger = logging.getLogger(__name__)

SPLIT_FILENAME = "age_split.json"

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 12345


@dataclass
class AgeSplit:
    """Window files per split, as paths relative to the ``processed/`` root.

    Train and val are re-partitioned out of the original ``train`` + ``val``
    pool, so one split's windows can live under two different subdirectories.
    Keeping the subdirectory in the path (``train/aaaaaaaq_s004_t000_0.pkl``)
    means each split is still a *single* loader over a single root, rather than a
    ConcatDataset -- which matters because ``enable_window_ids`` does not recurse
    into ConcatDataset.
    """

    files: Dict[str, List[str]]
    seed: int
    val_fraction: float

    def subjects(self, split: str) -> set:
        return {_subject(f) for f in self.files[split]}

    def recordings(self, split: str) -> set:
        return {recording_stem(f) for f in self.files[split]}


def _subject(filename: str) -> str:
    # Matches cross_validation.group_id_for(..., split_by='subject') and
    # TUHLoader.group_id: the leading underscore token of the basename.
    return os.path.basename(filename).split("_")[0]


def _list_windows(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Missing preprocessed window directory: {directory}")
    return sorted(f for f in os.listdir(directory) if f.endswith(".pkl"))


def build_age_split(
    processed_root: str,
    ages: Dict[str, float],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    pool_dirs: Optional[List[str]] = None,
    test_dir: str = "test",
) -> AgeSplit:
    """Re-partition the existing windows into subject-disjoint splits.

    ``dataset_maker/make_TUAB.py`` assigns subjects to train/val independently
    within the ``normal/`` and ``abnormal/`` folders, and 54 TUAB train subjects
    appear as both -- so 16 subjects land in *both* train and val. That leaks
    badly for age, whose value is near-constant per subject, so train and val are
    rebuilt here from the pooled windows by subject. The official eval set
    (``processed/test``) is left untouched.

    Windows whose recording has no usable age are dropped.
    """
    pool_dirs = list(pool_dirs if pool_dirs is not None else ("train", "val"))

    pooled = [
        os.path.join(name, f)
        for name in pool_dirs
        for f in filter_files_with_age(
            _list_windows(os.path.join(processed_root, name)), ages
        )
    ]
    test_files = [
        os.path.join(test_dir, f)
        for f in filter_files_with_age(
            _list_windows(os.path.join(processed_root, test_dir)), ages
        )
    ]

    subjects = sorted({_subject(f) for f in pooled})
    if not subjects:
        raise ValueError(f"No age-labelled windows found under {processed_root}")

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val_subjects = set(shuffled[len(shuffled) - n_val:])

    files = {"train": [], "val": [], "test": sorted(test_files)}
    for filename in sorted(pooled):
        files["val" if _subject(filename) in val_subjects else "train"].append(filename)

    split = AgeSplit(files=files, seed=seed, val_fraction=val_fraction)
    _assert_subject_disjoint(split)
    for name in ("train", "val", "test"):
        logger.info(
            "age split %-5s: %d windows, %d recordings, %d subjects",
            name, len(split.files[name]), len(split.recordings(name)),
            len(split.subjects(name)),
        )
    return split


def _assert_subject_disjoint(split: AgeSplit) -> None:
    """Fail loud on subject leakage -- silent leakage is the bug being fixed."""
    names = ("train", "val", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = split.subjects(left) & split.subjects(right)
            if shared:
                raise ValueError(
                    f"{len(shared)} subject(s) appear in both {left} and {right}: "
                    f"{sorted(shared)[:5]}"
                )


def save_age_split(split: AgeSplit, path: str) -> None:
    # Re-check on the way out as well as on the way in: a caller may have mutated
    # ``files`` since the split was built, and a leaking artifact on disk would
    # quietly poison every run that loads it.
    _assert_subject_disjoint(split)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "version": 1,
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "counts": {
            name: {
                "n_windows": len(split.files[name]),
                "n_recordings": len(split.recordings(name)),
                "n_subjects": len(split.subjects(name)),
            }
            for name in ("train", "val", "test")
        },
        "files": split.files,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote age split to %s", path)


def load_age_split(path: str) -> AgeSplit:
    with open(path) as fh:
        payload = json.load(fh)
    split = AgeSplit(
        files=payload["files"],
        seed=payload.get("seed", DEFAULT_SEED),
        val_fraction=payload.get("val_fraction", DEFAULT_VAL_FRACTION),
    )
    _assert_subject_disjoint(split)
    return split


def find_age_split(processed_root: str, *, filename: str = SPLIT_FILENAME) -> Optional[str]:
    candidate = os.path.join(processed_root, filename)
    return candidate if os.path.isfile(candidate) else None
