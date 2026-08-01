"""Tests for per-window case ids on the TUH loaders and the enable_window_ids
helper used by inference-mode aggregation."""

import pickle

import numpy as np
import pytest
import torch

from labram.data.tuh_datasets import TUABLoader, TUEVLoader
from labram.runs.finetune_setup import enable_window_ids


def _write(root, name, payload):
    with open(root / name, "wb") as f:
        pickle.dump(payload, f)


@pytest.fixture
def tuab_root(tmp_path):
    root = tmp_path / "tuab"
    root.mkdir()
    # Two recordings, 2 windows each: <subj>_s001_t000_<win>.pkl
    files = ["aaaa_s001_t000_0.pkl", "aaaa_s001_t000_1.pkl",
             "bbbb_s002_t000_0.pkl", "bbbb_s002_t000_1.pkl"]
    for i, name in enumerate(files):
        _write(root, name, {"X": np.random.randn(23, 200).astype("f4"), "y": i % 2})
    return root, files


def test_default_returns_pair(tuab_root):
    root, files = tuab_root
    ds = TUABLoader(str(root), files)
    item = ds[0]
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor)


def test_recording_group_id(tuab_root):
    root, files = tuab_root
    ds = TUABLoader(str(root), files)
    ds.return_id = True
    ids = [ds[i][2] for i in range(len(files))]
    assert ids == ["aaaa_s001_t000", "aaaa_s001_t000",
                   "bbbb_s002_t000", "bbbb_s002_t000"]


def test_subject_group_id(tuab_root):
    root, files = tuab_root
    ds = TUABLoader(str(root), files)
    ds.return_id = True
    ds.group_by = "subject"
    ids = [ds[i][2] for i in range(len(files))]
    assert ids == ["aaaa", "aaaa", "bbbb", "bbbb"]


def test_tuev_recording_uses_dash_separator(tmp_path):
    root = tmp_path / "tuev"
    root.mkdir()
    files = ["rec001-0.pkl", "rec001-1.pkl"]
    for name in files:
        _write(root, name, {"signal": np.random.randn(23, 200).astype("f4"),
                            "label": np.array([1])})
    ds = TUEVLoader(str(root), files)
    ds.return_id = True
    assert [ds[i][2] for i in range(2)] == ["rec001", "rec001"]


def test_enable_window_ids_on_loader_and_subset(tuab_root):
    root, files = tuab_root
    ds = TUABLoader(str(root), files)
    enable_window_ids(ds, group_by="subject")
    assert ds.return_id is True and ds.group_by == "subject"

    subset = torch.utils.data.Subset(TUABLoader(str(root), files), [0, 1])
    enable_window_ids(subset, group_by="recording")
    assert subset.dataset.return_id is True

    # list form + None passthrough
    enable_window_ids(None)
    lst = [TUABLoader(str(root), files)]
    enable_window_ids(lst)
    assert lst[0].return_id is True


def test_enable_window_ids_recurses_into_concat_dataset(tuab_root):
    """A cross-validation fold concatenates one loader per source (train + val
    roots), so ids must reach the leaves through ConcatDataset — otherwise
    per-case aggregation silently degrades to window-level."""
    root, files = tuab_root
    parts = [TUABLoader(str(root), files[:2]), TUABLoader(str(root), files[2:])]
    concat = torch.utils.data.ConcatDataset(parts)

    assert enable_window_ids(concat, group_by="recording") == 2
    assert all(p.return_id is True and p.group_by == "recording" for p in parts)
    # The concatenated dataset now yields 3-tuples end to end.
    assert all(len(concat[i]) == 3 for i in range(len(concat)))


def test_enable_window_ids_recurses_through_nested_wrappers(tuab_root):
    """Subset(ConcatDataset(...)) — the debug-mode shape — also reaches leaves."""
    root, files = tuab_root
    parts = [TUABLoader(str(root), files[:2]), TUABLoader(str(root), files[2:])]
    nested = torch.utils.data.Subset(
        torch.utils.data.ConcatDataset(parts), [0, 1, 2])

    assert enable_window_ids(nested, group_by="subject") == 2
    assert all(p.return_id is True and p.group_by == "subject" for p in parts)
    assert len(nested[0]) == 3


def test_enable_window_ids_warns_when_no_leaf_supports_ids(caplog):
    """An aggregation mode configured over a dataset that cannot emit case ids
    must say so rather than silently producing window-level metrics."""
    class _NoIds(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return torch.zeros(2, 200), 0

    with caplog.at_level("WARNING"):
        assert enable_window_ids(_NoIds()) == 0
    assert "window-level metrics only" in caplog.text

    # ...and stays quiet when the caller opts out of the warning.
    caplog.clear()
    with caplog.at_level("WARNING"):
        enable_window_ids(_NoIds(), warn=False)
    assert caplog.text == ""
