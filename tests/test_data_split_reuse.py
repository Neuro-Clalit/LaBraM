"""Reusing a recorded data_split.json to pin the train/val/test case assignment
across fine-tune runs (labram.data.data_split_reuse)."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch.utils.data

from labram.data.bundles import get_dataset_bundle
from labram.data.data_split_reuse import (
    apply_data_split,
    bundle_from_data_split,
    load_data_split_json,
)
from labram.utils.data_split import build_data_split, save_data_split


def _make_tuab(root: Path):
    """A synthetic TUAB layout: train/val/test dirs of pickle windows."""
    layout = {
        "train": [f"s{i:02d}_s001_t000_0.pkl" for i in range(6)],
        "val": [f"s{i:02d}_s001_t000_0.pkl" for i in range(6, 8)],
        "test": [f"s{i:02d}_s001_t000_0.pkl" for i in range(8, 10)],
    }
    for split, files in layout.items():
        d = root / split
        d.mkdir(parents=True)
        for name in files:
            with open(d / name, "wb") as f:
                pickle.dump({"X": np.random.randn(23, 200).astype("f4"), "y": 0}, f)
    return layout


def _files(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        out = []
        for d in dataset.datasets:
            out += list(d.files)
        return sorted(out)
    return sorted(dataset.files)


def test_reuse_reproduces_recorded_split(tmp_path):
    root = tmp_path / "TUAB"
    _make_tuab(root)
    bundle = get_dataset_bundle("TUAB", str(root))

    # Record the default split, then rebuild a bundle from that record.
    split = build_data_split(bundle.train, bundle.val, bundle.test, "TUAB")
    path = save_data_split(split, str(tmp_path / "rec"))

    reused = apply_data_split(bundle, load_data_split_json(path))
    assert _files(reused.train) == _files(bundle.train)
    assert _files(reused.val) == _files(bundle.val)
    assert _files(reused.test) == _files(bundle.test)
    assert reused.nb_classes == bundle.nb_classes


def test_reuse_honors_a_moved_file(tmp_path):
    """A file the record moved from train to val lands in val on reuse — even
    though it physically lives in the train dir."""
    root = tmp_path / "TUAB"
    _make_tuab(root)
    bundle = get_dataset_bundle("TUAB", str(root))
    split = build_data_split(bundle.train, bundle.val, bundle.test, "TUAB")

    moved = split["train"]["files"].pop()   # take one training file...
    split["val"]["files"].append(moved)     # ...and record it as validation.

    reused = apply_data_split(bundle, split)
    assert moved in _files(reused.val)
    assert moved not in _files(reused.train)


def test_bundle_from_data_split_end_to_end(tmp_path):
    root = tmp_path / "TUAB"
    _make_tuab(root)
    bundle = get_dataset_bundle("TUAB", str(root))
    path = save_data_split(
        build_data_split(bundle.train, bundle.val, bundle.test, "TUAB"),
        str(tmp_path / "rec"))

    reused = bundle_from_data_split("TUAB", str(root), path)
    assert len(reused.train) == len(bundle.train)
    assert len(reused.test) == len(bundle.test)


def test_missing_files_raise_for_empty_split(tmp_path):
    root = tmp_path / "TUAB"
    _make_tuab(root)
    bundle = get_dataset_bundle("TUAB", str(root))
    bogus = {"train": {"files": ["nope_1.pkl"]},
             "val": {"files": ["nope_2.pkl"]},
             "test": {"files": ["nope_3.pkl"]}}
    with pytest.raises(ValueError):
        apply_data_split(bundle, bogus)
