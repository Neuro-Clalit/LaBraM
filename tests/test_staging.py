"""Staging npy recordings onto fast local storage, train first."""
import os

import pytest
import torch

from labram.data import get_dataset_bundle
from labram.data import staging
from test_tuab_npy_classification import processed  # noqa: F401  (fixture)


def _bundle(processed):
    return get_dataset_bundle("TUAB", str(processed), data_format="npy")


def _items(b):
    return [(split[i][0].clone(), split[i][1]) for split in (b.train, b.val, b.test)
            for i in range(len(split))]


def test_stages_everything_when_it_fits(processed, tmp_path):
    ref = _items(_bundle(processed))
    b = _bundle(processed)
    report = staging.stage_bundle(b, str(tmp_path / "nvme"), reserve_gb=0)
    assert report.staged == {"train": 2, "val": 1, "test": 1} and not report.linked
    mirror = b.train.root
    assert mirror.startswith(str(tmp_path / "nvme")) and b.test.root == mirror
    assert all(not os.path.islink(os.path.join(mirror, "recordings", f)) for f in os.listdir(
        os.path.join(mirror, "recordings")))
    got = _items(b)
    assert all(torch.equal(a[0], g[0]) and a[1] == g[1] for a, g in zip(ref, got))
    again = staging.stage_bundle(_bundle(processed), str(tmp_path / "nvme"), reserve_gb=0)
    assert again.copied_bytes == 0                     # reused, not copied twice


def test_train_has_priority_when_space_is_short(processed, tmp_path, monkeypatch):
    one_rec = os.path.getsize(next((processed.parent / "processed_npy/recordings").glob("*.npy")))
    real = staging.shutil.disk_usage

    class Usage:
        free = int(2.5 * one_rec)                      # room for the two train recordings
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda p: Usage())
    b = _bundle(processed)
    report = staging.stage_bundle(b, str(tmp_path / "nvme"), reserve_gb=0)
    monkeypatch.setattr(staging.shutil, "disk_usage", real)
    assert report.staged == {"train": 2} and report.linked == {"val": 1, "test": 1}
    rec = os.path.join(b.val.root, "recordings")
    linked = [f for f in os.listdir(rec) if os.path.islink(os.path.join(rec, f))]
    assert sorted(linked) == ["ccc_s001_t000.npy", "ddd_s001_t000.npy"]
    x, y = b.test[0]                                   # read through the symlink
    assert x.shape == (23, 2000)


def test_disabled_and_sagemaker(processed, monkeypatch):
    assert staging.stage_bundle(_bundle(processed), "none") is None
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")
    assert staging.resolve_stage_dir("auto") is None
