"""A compact cohort stores only the windows a run's selection uses, and serves
bit-identical samples (and the same sample lists) as the full cohort."""
import json

import numpy as np
import pytest

from dataset_maker import make_compact_cohort as mcc
from labram.data import get_dataset_bundle
from labram.data.window_selection import WindowSelection, apply_window_selection
from test_window_qc import _corpus


@pytest.mark.parametrize("sel", [
    WindowSelection(trim_start_sec=20, trim_end_sec=30, eval_minutes=1),
    WindowSelection(trim_start_sec=10, window_sec=20, eval_minutes=1),
])
def test_compact_serves_identical_samples(tmp_path, sel):
    src = _corpus(tmp_path / "src")
    dst = tmp_path / "dst"
    total, manifest = mcc.build(str(src), str(dst), sel, threads=2)
    assert mcc.verify(str(src), str(dst), sel, n=10_000) > 0          # every sample
    full = sum(np.load(f).nbytes for f in (src / "recordings").glob("*.npy"))
    assert total < full
    row = manifest["recordings"][0]
    assert row["n_windows"] == 20 and len(row["window_index"]) < 20
    assert (dst / "age_metadata.json").exists()
    split = json.load(open(dst / "age_split.json"))["files"]
    a = apply_window_selection(get_dataset_bundle("TUAB_AGE", str(dst), data_format="npy"), sel)
    assert sorted(split["train"]) == sorted(a.train.files)


def test_unstored_window_and_random_crop_are_errors(tmp_path):
    from labram.data.window_selection import enable_random_crop
    sel = WindowSelection(trim_start_sec=20, eval_minutes=1)
    src = _corpus(tmp_path / "src")
    mcc.build(str(src), str(tmp_path / "dst"), sel, threads=2)
    b = apply_window_selection(get_dataset_bundle("TUAB_AGE", str(tmp_path / "dst"), data_format="npy"), sel)
    leaf = b.train
    with pytest.raises(KeyError, match="not stored"):
        leaf._compact_start(leaf.files[0].split("/")[1].rsplit("_", 1)[0], 0)
    enable_random_crop(leaf)
    with pytest.raises(ValueError):
        leaf._load(0)


def test_cv_compact_reproduces_every_fold(tmp_path):
    """--cv stores what a recording needs as train *or* eval, so each CV fold
    (recordings moved across splits) selects and serves the same samples."""
    import torch
    from types import SimpleNamespace
    from labram.data.cross_validation import build_grouped_folds, materialize_fold
    sel = WindowSelection(trim_start_sec=20, trim_end_sec=10, eval_minutes=1)
    src = _corpus(tmp_path / "src")
    # cap the cohort's training lists, as make_cohort_dataset does
    sp = json.load(open(src / "age_split.json"))
    sp["files"]["train"] = [f for f in sp["files"]["train"] if int(f[:-4].rsplit("_", 1)[1]) % 3 == 0]
    json.dump(sp, open(src / "age_split.json", "w"))
    dst = tmp_path / "dst"
    mcc.build(str(src), str(dst), sel, threads=2, cv=True)
    cv = SimpleNamespace(n_folds=4, split_by="recording", shuffle=True, seed=1, pool="train_val")
    for k in range(4):
        out = []
        for root in (src, dst):
            b = get_dataset_bundle("TUAB_AGE", str(root), data_format="npy")
            out.append(apply_window_selection(materialize_fold(build_grouped_folds(b, cv), k, b), sel))
        from labram.data.window_selection import _leaves
        for split in ("train", "val", "test"):
            fa, fb = getattr(out[0], split), getattr(out[1], split)
            files = lambda d: [f for leaf in _leaves(d) for f in leaf.files]  # noqa: E731
            assert files(fa) == files(fb) and len(fa) == len(fb) > 0
            for i in range(len(fa)):
                assert torch.equal(fa[i][0], fb[i][0]) and torch.equal(fa[i][1], fb[i][1])
