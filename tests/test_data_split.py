"""Tests for train/val/test data-split recording (labram.utils.data_split) and
its ClearML-artifact wiring in runs.common.log_data_split_artifact."""

import json
import pickle

import numpy as np
import pytest
import torch

from labram.configs.train_config import LoggingConfig, OutputConfig
from labram.data.tuh_datasets import TUABLoader
from labram.utils.data_split import build_data_split, save_data_split


@pytest.fixture
def tuab(tmp_path):
    root = tmp_path / "tuab"
    root.mkdir()
    files = []
    # subjects: aaaa (train), bbbb (train), cccc (val), dddd (test); 2 windows each
    for subj in ("aaaa", "bbbb", "cccc", "dddd"):
        for w in range(2):
            name = f"{subj}_s1_t0_{w}.pkl"
            files.append(name)
            with open(root / name, "wb") as f:
                pickle.dump({"X": np.zeros((1, 200), "f4"), "y": 0}, f)
    ds = TUABLoader(str(root), files)
    # indices 0-3 = aaaa,bbbb train; 4-5 = cccc val; 6-7 = dddd test
    train = torch.utils.data.Subset(ds, [0, 1, 2, 3])
    val = torch.utils.data.Subset(ds, [4, 5])
    test = torch.utils.data.Subset(ds, [6, 7])
    return train, val, test


def test_build_split_counts_and_ids(tuab):
    train, val, test = tuab
    split = build_data_split(train, val, test, "TUAB")
    assert split['dataset'] == "TUAB"
    assert split['train']['n_windows'] == 4
    assert split['train']['n_subjects'] == 2
    assert split['train']['n_recordings'] == 2
    assert split['val']['subjects'] == ["cccc"]
    assert split['test']['subjects'] == ["dddd"]
    # clean split (over EEG items) -> no subject and no recording overlap
    assert split['subject_overlap'] == {}
    assert split['recording_overlap'] == {}


def test_build_split_detects_subject_overlap(tuab):
    train, val, test = tuab
    # Force leakage: put a train window into test as well.
    leaked = torch.utils.data.Subset(train.dataset, [0])  # subject aaaa
    split = build_data_split(train, val, leaked, "TUAB")
    assert "train∩test" in split['subject_overlap']
    assert "aaaa" in split['subject_overlap']["train∩test"]


def test_build_split_detects_recording_overlap(tuab):
    train, val, test = tuab
    # Windows 0 and 1 are both from recording 'aaaa_s1_t0'. Splitting them across
    # train/test leaks that recording even though the leak is at the window level.
    train_one = torch.utils.data.Subset(train.dataset, [0])   # aaaa_s1_t0_0
    test_one = torch.utils.data.Subset(train.dataset, [1])    # aaaa_s1_t0_1
    split = build_data_split(train_one, None, test_one, "TUAB")
    assert "train∩test" in split['recording_overlap']
    assert "aaaa_s1_t0" in split['recording_overlap']["train∩test"]


def test_save_split_writes_json(tuab, tmp_path):
    train, val, test = tuab
    split = build_data_split(train, val, test, "TUAB")
    path = save_data_split(split, str(tmp_path))
    assert path and path.endswith("data_split.json")
    loaded = json.loads((tmp_path / "data_split.json").read_text())
    assert loaded['train']['n_windows'] == 4
    assert loaded['test']['subjects'] == ["dddd"]


def test_none_splits_are_skipped(tuab):
    train, _, _ = tuab
    split = build_data_split(train, None, None, "TUAB")
    assert 'train' in split and 'val' not in split and 'test' not in split


class _FakeTask:
    def __init__(self):
        self.artifacts = {}

    def upload_artifact(self, name, artifact_object):
        self.artifacts[name] = artifact_object


class _FakeWriter:
    def __init__(self, task):
        self.task = task


def test_log_data_split_artifact_uploads(tuab, tmp_path):
    import labram.runs.common as common
    train, val, test = tuab
    task = _FakeTask()
    path = common.log_data_split_artifact(
        LoggingConfig(log_data_split=True),
        OutputConfig(output_dir=str(tmp_path)),
        train, val, test, log_writer=_FakeWriter(task), dataset_name="TUAB")
    assert path is not None
    assert "data_split" in task.artifacts
    assert task.artifacts["data_split"] == path


def test_log_data_split_disabled(tuab, tmp_path):
    import labram.runs.common as common
    train, val, test = tuab
    path = common.log_data_split_artifact(
        LoggingConfig(log_data_split=False),
        OutputConfig(output_dir=str(tmp_path)),
        train, val, test, log_writer=None)
    assert path is None
