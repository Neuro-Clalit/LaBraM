"""The npy data format must serve exactly the samples the pickle format serves,
plus optional random crops for training."""
import pickle

import numpy as np
import pytest
import torch

from dataset_maker import make_TUAB_npy as mk
from labram.data.age_splits import build_age_split, save_age_split
from labram.data.tuh_datasets import TUABAgeNpyLoader, prepare_TUAB_age_dataset
from labram.data.tuh_metadata import RecordingMetadata, save_metadata_sidecar
from labram.data.window_selection import (
    WindowSelection, apply_window_selection, enable_random_crop,
)
from labram.data.bundles import DatasetBundle, REGRESSION


def _corpus(tmp_path):
    """processed/ (pickles + sidecars) and processed_npy/ built from it."""
    processed = tmp_path / "processed"
    rng = np.random.default_rng(1)
    meta = {}
    plan = {"train": [("aaa", 24), ("bbb", 18), ("ccc", 20), ("ddd", 22)], "test": [("eee", 16)]}
    for split, recs in plan.items():
        (processed / split).mkdir(parents=True)
        for i, (subject, n) in enumerate(recs):
            stem = f"{subject}_s001_t000"
            meta[stem] = RecordingMetadata(stem=stem, subject=subject, session="s001",
                                           token="t000", age=30 + 7 * i, sex="F", year=2012,
                                           raw_age=30 + 7 * i,
                                           label="normal" if i % 2 else "abnormal")
            for k in range(n):
                with open(processed / split / f"{stem}_{k}.pkl", "wb") as fh:
                    pickle.dump({"X": rng.normal(scale=40, size=(23, 2000)), "y": 0}, fh)
    save_metadata_sidecar(meta, str(processed / "age_metadata.json"))
    ages = {s: float(m.age) for s, m in meta.items()}
    save_age_split(build_age_split(str(processed), ages, val_fraction=0.25, seed=3,
                                   pool_dirs=["train"]), str(processed / "age_split.json"))
    npy = tmp_path / "processed_npy"
    mk.main(["convert", "--src", str(processed), "--dst", str(npy)])
    mk.main(["merge", "--dst", str(npy), "--sidecar-dir", str(processed)])
    return tmp_path


def _bundle(root, data_format):
    train, test, val, stats = prepare_TUAB_age_dataset(str(root), data_format=data_format)
    return DatasetBundle(train=train, val=val, test=test, ch_names=[], nb_classes=1,
                         metrics=["mae"], task=REGRESSION, target_stats=stats)


SELECTION = WindowSelection(trim_start_sec=60, trim_end_sec=60, eval_minutes=1)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return _corpus(tmp_path_factory.mktemp("tuab"))


def test_npy_serves_the_same_items_and_tensors_as_pickle(corpus):
    pkl = apply_window_selection(_bundle(corpus, "pickle"), SELECTION)
    npy = apply_window_selection(_bundle(corpus, "npy"), SELECTION)
    assert isinstance(npy.train, TUABAgeNpyLoader)
    assert npy.target_stats == pkl.target_stats
    for split in ("train", "val", "test"):
        a, b = getattr(pkl, split), getattr(npy, split)
        assert a.files == b.files and len(a) == len(b) > 0
        for i in range(len(a)):
            (xa, ya), (xb, yb) = a[i], b[i]
            assert xb.dtype == torch.float32 and torch.equal(xa, xb) and torch.equal(ya, yb)


def test_longer_windows_match_too(corpus):
    sel = WindowSelection(trim_start_sec=60, window_sec=30)
    pkl = apply_window_selection(_bundle(corpus, "pickle"), sel)
    npy = apply_window_selection(_bundle(corpus, "npy"), sel)
    assert pkl.train.files == npy.train.files
    assert torch.equal(pkl.train[0][0], npy.train[0][0]) and npy.train[0][0].shape == (23, 6000)


def test_random_crop_stays_in_bounds_and_varies(corpus):
    npy = apply_window_selection(_bundle(corpus, "npy"), SELECTION)
    enable_random_crop(npy.train)
    leaf = npy.train
    stem = leaf.files[0].split("/")[-1][:-4].rsplit("_", 1)[0]
    lo, hi = leaf.sample_bounds[stem]
    rec = np.load(f"{leaf.root}/recordings/{stem}.npy")
    starts = set()
    torch.manual_seed(0)
    for _ in range(40):
        x = leaf[0][0].numpy()
        start = next(s for s in range(lo, hi - 2000 + 1) if np.array_equal(rec[s:s + 2000].T, x))
        assert lo <= start and start + 2000 <= hi
        starts.add(start)
    assert len(starts) > 5


def test_random_crop_requires_npy(corpus):
    with pytest.raises(ValueError, match="npy"):
        enable_random_crop(_bundle(corpus, "pickle").train)


def test_evaluation_is_never_cropped(corpus):
    npy = apply_window_selection(_bundle(corpus, "npy"), SELECTION)
    enable_random_crop(npy.train)
    assert not npy.val.random_crop and not npy.test.random_crop


def test_case_filter_and_cohort_labels_work_on_npy(corpus):
    sel = WindowSelection(case_filter="normal")
    bundle = _bundle(corpus, "npy")
    bundle.val = bundle.test = None      # the tiny synthetic test split is abnormal-only
    npy = apply_window_selection(bundle, sel)
    assert {f.split("/")[-1].split("_")[0] for f in npy.train.files} <= {"bbb", "ddd"}


def test_worker_copies_open_their_own_maps(corpus):
    npy = _bundle(corpus, "npy")
    npy.train[0]
    assert len(pickle.loads(pickle.dumps(npy.train))._mmaps) == 0


def test_missing_manifest_is_a_clear_error(tmp_path):
    (tmp_path / "processed_npy").mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        prepare_TUAB_age_dataset(str(tmp_path), data_format="npy")
