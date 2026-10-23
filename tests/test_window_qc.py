"""Signal-quality window rejection: per-window stats (window_stats.npz) and the
amplitude / flat-channel rule applied by window selection."""
import json

import numpy as np
import pytest

from labram.data.bundles import REGRESSION, DatasetBundle
from labram.data.tuh_datasets import prepare_TUAB_age_dataset
from labram.data.window_qc import (
    WINDOW_STATS_FILE, build, load_window_stats, recording_window_stats, window_passes,
)
from labram.data.window_selection import WindowSelection, apply_window_selection

W = 2000


def test_stats_match_a_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(scale=20, size=(5 * W + 300, 23)).astype(np.float16)   # + a partial window
    x[2 * W + 7, 4] = 900
    x[3 * W:4 * W, 9] = 0.0
    mx, sd = recording_window_stats(x)
    assert len(mx) == len(sd) == 5
    for k in range(5):
        w = x[k * W:(k + 1) * W].astype(np.float32)
        assert mx[k] == np.abs(w).max() and np.isclose(sd[k], w.std(0).min())
    assert list(window_passes(mx, sd, 800, 0.5)) == [True, True, False, False, True]
    assert window_passes(mx, sd, 0, 0).all()


def test_masked_channels_are_ignored():
    x = np.random.default_rng(0).normal(scale=20, size=(2 * W, 23)).astype(np.float32)
    x[:, 22] = 0.0                        # a zero-filled missing channel
    mask = [True] * 22 + [False]
    assert (recording_window_stats(x)[1] == 0).all()
    assert (recording_window_stats(x, mask)[1] > 1).all()


def _corpus(tmp_path):
    """An npy age dataset: 4 train/val recordings and 1 test, each 20 windows."""
    root = tmp_path / "npy"
    (root / "recordings").mkdir(parents=True)
    rng = np.random.default_rng(1)
    plan = {"aaa": "train", "bbb": "train", "ccc": "train", "ddd": "val", "eee": "test"}
    rows, meta, files = [], {}, {"train": [], "val": [], "test": []}
    for i, (subject, split) in enumerate(plan.items()):
        stem = f"{subject}_s001_t000"
        x = rng.normal(scale=20, size=(20 * W, 23)).astype(np.float16)
        np.save(root / "recordings" / f"{stem}.npy", x)
        rows.append({"stem": stem, "n_windows": 20, "n_samples": 20 * W, "source_split": split,
                     "channel_mask": [True] * 23})
        meta[stem] = {"stem": stem, "subject": subject, "session": "s001", "token": "t000",
                      "age": 30 + 5 * i, "sex": "F", "year": 2012, "raw_age": 30 + 5 * i,
                      "label": "normal"}
        files[split] += [f"{split}/{stem}_{k}.pkl" for k in range(20)]
    json.dump({"format_version": 2, "dtype": "float16", "recordings": rows},
              open(root / "manifest.json", "w"))
    json.dump({"version": 1, "recordings": meta}, open(root / "age_metadata.json", "w"))
    json.dump({"version": 1, "seed": 0, "val_fraction": 0.25, "files": files},
              open(root / "age_split.json", "w"))
    return root


def _spoil(root, stem, k, how):
    path = root / "recordings" / f"{stem}.npy"
    x = np.load(path)
    if how == "spike":
        x[k * W + 50, 3] = -1200
    else:
        x[k * W:(k + 1) * W, 11] = 0.0
    np.save(path, x)


def _bundle(root):
    train, test, val, stats = prepare_TUAB_age_dataset(str(root), data_format="npy")
    return DatasetBundle(train=train, val=val, test=test, ch_names=[], nb_classes=1,
                         metrics=["mae"], task=REGRESSION, target_stats=stats)


def _windows(split):
    return {(f.split("/")[-1].rsplit("_", 1)[0].split("_")[0], int(f[:-4].rsplit("_", 1)[1]))
            for f in split.files}


@pytest.fixture
def corpus(tmp_path):
    root = _corpus(tmp_path)
    _spoil(root, "aaa_s001_t000", 3, "spike")
    _spoil(root, "bbb_s001_t000", 8, "flat")
    _spoil(root, "ddd_s001_t000", 2, "spike")       # val: inside the eval budget
    _spoil(root, "eee_s001_t000", 4, "flat")
    build(str(root), workers=2)
    return root


QC = dict(max_abs_uv=800, min_std_uv=0.5)


def test_build_writes_stats_for_every_recording(corpus):
    stats = load_window_stats(str(corpus))
    assert set(stats) == {f"{s}_s001_t000" for s in ("aaa", "bbb", "ccc", "ddd", "eee")}
    assert all(len(mx) == len(sd) == 20 for mx, sd in stats.values())
    assert stats["aaa_s001_t000"][0][3] == 1200 and stats["bbb_s001_t000"][1][8] == 0


def test_rejected_windows_leave_every_split(corpus):
    plain = apply_window_selection(_bundle(corpus), WindowSelection(trim_start_sec=10))
    qc = apply_window_selection(_bundle(corpus), WindowSelection(trim_start_sec=10, **QC))
    assert _windows(plain.train) - _windows(qc.train) == {("aaa", 3), ("bbb", 8)}
    assert _windows(plain.val) - _windows(qc.val) == {("ddd", 2)}
    assert _windows(plain.test) - _windows(qc.test) == {("eee", 4)}


def test_eval_budget_counts_passing_windows_only(corpus):
    sel = WindowSelection(trim_start_sec=10, eval_minutes=1, **QC)     # 6 windows
    qc = apply_window_selection(_bundle(corpus), sel)
    assert sorted(k for _, k in _windows(qc.val)) == [1, 3, 4, 5, 6, 7]
    assert sorted(k for _, k in _windows(qc.test)) == [1, 2, 3, 5, 6, 7]
    again = apply_window_selection(qc, sel)            # idempotent on its own output
    assert _windows(again.val) == _windows(qc.val)


def test_a_sample_fails_when_any_of_its_windows_fails(corpus):
    qc = apply_window_selection(_bundle(corpus), WindowSelection(window_sec=20, **QC))
    assert ("aaa", 2) not in _windows(qc.train) and ("aaa", 4) in _windows(qc.train)


def test_one_threshold_alone(corpus):
    amp = apply_window_selection(_bundle(corpus), WindowSelection(max_abs_uv=800))
    assert ("bbb", 8) in _windows(amp.train) and ("aaa", 3) not in _windows(amp.train)


def test_missing_stats_is_a_clear_error(corpus):
    (corpus / WINDOW_STATS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="labram.data.window_qc"):
        apply_window_selection(_bundle(corpus), WindowSelection(**QC))


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        WindowSelection(max_abs_uv=-1)
