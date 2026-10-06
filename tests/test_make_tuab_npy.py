"""Repacking TUAB window pickles into per-recording float32 .npy files."""
import json
import pickle

import numpy as np
import pytest
import torch

from dataset_maker import make_TUAB_npy as mk


def _corpus(root, spec):
    """spec: {split: {stem: n_windows}} -> processed/ tree of window pickles."""
    rng = np.random.default_rng(0)
    for split, recs in spec.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for stem, n in recs.items():
            for k in range(n):
                x = rng.normal(scale=50, size=(23, 2000))
                with open(root / split / f"{stem}_{k}.pkl", "wb") as fh:
                    pickle.dump({"X": x, "y": 0}, fh)


def _sidecar(d, stems):
    d.mkdir(parents=True, exist_ok=True)
    recs = {s: {"stem": s, "subject": s.split("_")[0], "session": "s001", "token": "t000",
                "age": 40 + i, "raw_age": 40 + i, "sex": "F", "year": 2012,
                "label": "normal" if i % 2 else "abnormal"} for i, s in enumerate(stems)}
    (d / "age_metadata.json").write_text(json.dumps({"version": 1, "recordings": recs}))
    (d / "age_split.json").write_text(json.dumps({"version": 1, "files": {}}))


SPEC = {"train": {"aaa_s001_t000": 3, "bbb_s001_t000": 5}, "test": {"ccc_s001_t000": 2}}


def test_convert_concatenates_windows_in_order_as_float32(tmp_path):
    _corpus(tmp_path / "src", SPEC)
    mk.main(["convert", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst")])
    arr = np.load(tmp_path / "dst/recordings/bbb_s001_t000.npy", mmap_mode="r")
    assert arr.dtype == np.float32 and arr.shape == (5 * 2000, 23)
    for k in range(5):
        with open(tmp_path / f"src/train/bbb_s001_t000_{k}.pkl", "rb") as fh:
            x = pickle.load(fh)["X"]
        # Same tensor the training loader builds from the pickle today.
        assert torch.equal(torch.from_numpy(np.ascontiguousarray(arr[k * 2000:(k + 1) * 2000].T)),
                           torch.FloatTensor(x))


def test_shards_partition_the_recordings(tmp_path):
    _corpus(tmp_path / "src", SPEC)
    for shard in range(2):
        mk.main(["convert", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"),
                 "--shard", str(shard), "--num-shards", "2"])
    parts = [json.loads(p.read_text()) for p in sorted((tmp_path / "dst/_parts").glob("*.json"))]
    stems = [r["stem"] for p in parts for r in p["recordings"]]
    assert sorted(stems) == sorted(s for recs in SPEC.values() for s in recs)
    assert len(stems) == len(set(stems))


def test_rerun_skips_finished_recordings(tmp_path):
    _corpus(tmp_path / "src", SPEC)
    args = ["convert", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst")]
    mk.main(args)
    mk.main(args)
    part = json.loads(next((tmp_path / "dst/_parts").glob("*.json")).read_text())
    assert all(r["skipped"] for r in part["recordings"])


def test_gaps_and_duplicate_stems_are_rejected(tmp_path):
    _corpus(tmp_path / "gap", {"train": {"aaa_s001_t000": 3}})
    (tmp_path / "gap/train/aaa_s001_t000_1.pkl").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        mk.inventory(mk.Store(str(tmp_path / "gap")))
    _corpus(tmp_path / "dup", {"train": {"aaa_s001_t000": 1}, "val": {"aaa_s001_t000": 1}})
    with pytest.raises(ValueError, match="windows in"):
        mk.inventory(mk.Store(str(tmp_path / "dup")))


def test_merge_writes_manifest_with_labels_and_copies_split(tmp_path):
    _corpus(tmp_path / "src", SPEC)
    mk.main(["convert", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst")])
    _sidecar(tmp_path / "side", [s for recs in SPEC.values() for s in recs])
    mk.main(["merge", "--dst", str(tmp_path / "dst"), "--sidecar-dir", str(tmp_path / "side"),
             "--expect", "3"])
    man = json.loads((tmp_path / "dst/manifest.json").read_text())
    assert man["n_recordings"] == 3 and man["n_windows"] == 10
    row = {r["stem"]: r for r in man["recordings"]}["bbb_s001_t000"]
    assert row["source_split"] == "train" and row["n_samples"] == 10000 and row["label"]
    assert (tmp_path / "dst/age_split.json").exists()


def test_shard_assignment_balances_windows():
    recs = {f"r{i}": ("train", n) for i, n in enumerate([100, 90, 50, 40, 30, 10])}
    shard_of = mk.assign_shards(recs, 2)
    load = [sum(recs[s][1] for s, sh in shard_of.items() if sh == i) for i in range(2)]
    assert abs(load[0] - load[1]) <= 20   # greedy: within the smaller items
