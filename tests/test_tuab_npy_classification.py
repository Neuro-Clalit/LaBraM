"""TUAB normal/abnormal classification from the npy format must serve exactly
the windows and labels the pickle format serves (the pickles are no longer
kept locally)."""
import pickle

import numpy as np
import pytest
import torch

from dataset_maker import make_TUAB_npy as mk
from labram.data import get_dataset_bundle
from labram.data.tuh_datasets import TUABNpyLoader
from labram.data.tuh_metadata import RecordingMetadata, save_metadata_sidecar

PLAN = {"train": [("aaa", 3, "abnormal"), ("bbb", 2, "normal")],
        "val": [("ccc", 2, "abnormal")], "test": [("ddd", 3, "normal")]}


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("tuab")
    proc = root / "processed"
    rng = np.random.default_rng(0)
    meta = {}
    for split, recs in PLAN.items():
        (proc / split).mkdir(parents=True)
        for subject, n, label in recs:
            stem = f"{subject}_s001_t000"
            meta[stem] = RecordingMetadata(stem=stem, subject=subject, session="s001",
                                           token="t000", age=50, sex="F", year=2012,
                                           raw_age=50, label=label)
            for k in range(n):
                with open(proc / split / f"{stem}_{k}.pkl", "wb") as fh:
                    pickle.dump({"X": rng.normal(scale=40, size=(23, 2000)),
                                 "y": int(label == "abnormal")}, fh)
    save_metadata_sidecar(meta, str(proc / "age_metadata.json"))
    (proc / "age_split.json").write_text('{"version": 1, "files": {}}')
    mk.main(["convert", "--src", str(proc), "--dst", str(root / "processed_npy")])
    mk.main(["merge", "--dst", str(root / "processed_npy"), "--sidecar-dir", str(proc)])
    return proc


def test_npy_matches_pickle_windows_and_labels(processed):
    pk = get_dataset_bundle("TUAB", str(processed), data_format="pickle")
    nv = get_dataset_bundle("TUAB", str(processed), data_format="npy")   # sibling processed_npy
    for split in ("train", "val", "test"):
        a, b = getattr(pk, split), getattr(nv, split)
        assert isinstance(b, TUABNpyLoader)
        by_name = {f: i for i, f in enumerate(a.files)}
        assert sorted(by_name) == sorted(f.split("/", 1)[1] for f in b.files)
        for i, f in enumerate(b.files):
            xa, ya = a[by_name[f.split("/", 1)[1]]]
            xb, yb = b[i]
            assert torch.equal(xa, xb) and ya == yb


def test_npy_root_resolution(processed):
    npy = processed.parent / "processed_npy"
    for path in (processed.parent, processed, npy):
        b = get_dataset_bundle("TUAB", str(path), data_format="npy")
        assert len(b.train) == 5 and len(b.val) == 2 and len(b.test) == 3


def test_float16_recordings_are_upcast(processed, tmp_path):
    import shutil
    npy = tmp_path / "npy16"
    shutil.copytree(processed.parent / "processed_npy", npy)
    for f in (npy / "recordings").glob("*.npy"):
        np.save(f, np.load(f).astype(np.float16))
    x, y = TUABNpyLoader(str(npy), ["train/aaa_s001_t000_1.pkl"])[0]
    assert x.dtype == torch.float32 and x.shape == (23, 2000) and y == 1
