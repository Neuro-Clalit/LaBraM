"""Joining corpus cohorts into one (dataset_maker/make_joint_cohort.py)."""
import pandas as pd
import pytest

from dataset_maker.make_joint_cohort import join

CH = ["FP1", "FP2"]


def _part(name, stems, subjects, split="train", **over):
    man = {"channels": CH, "sample_rate": 200, "window_samples": 2000, "units": "uV",
           "recordings": [{"stem": s, "n_windows": 3, "n_samples": 6000} for s in stems], **over}
    meta = {"version": 1, "recordings": {s: {"stem": s, "subject": u, "age": 40} for s, u in zip(stems, subjects)}}
    files = {"train": [], "val": [], "test": []}
    files[split] = [f"{split}/{s}_0.pkl" for s in stems]
    split_ = {"version": 1, "files": files}
    return name, man, meta, split_, pd.DataFrame({"stem": stems})


def test_join_merges_rows_splits_and_tags_corpus():
    man, meta, split, coh = join([_part("tuh", ["aaaa_s001_t000"], ["aaaa"]),
                                  _part("heedb", ["sub-S00011_ses-1_task-EEG"], ["1"], split="test")])
    assert [r["corpus"] for r in man["recordings"]] == ["tuh", "heedb"]
    assert split["files"]["train"] == ["train/aaaa_s001_t000_0.pkl"]
    assert split["files"]["test"] == ["test/sub-S00011_ses-1_task-EEG_0.pkl"]
    assert len(meta["recordings"]) == 2 and set(coh["corpus"]) == {"tuh", "heedb"}
    assert man["joint"]["heedb"]["recordings"] == 1


@pytest.mark.parametrize("second,msg", [
    (("heedb", ["aaaa_s001_t000"], ["1"]), "stem"),                 # same recording twice
    (("heedb", ["b"], ["aaaa"]), "patient ids"),                    # colliding patient id
])
def test_join_refuses_collisions(second, msg):
    with pytest.raises(ValueError, match=msg):
        join([_part("tuh", ["aaaa_s001_t000"], ["aaaa"]), _part(*second)])


def test_join_refuses_different_channel_layouts():
    with pytest.raises(ValueError, match="channels"):
        join([_part("tuh", ["a"], ["x"]), _part("heedb", ["b"], ["y"], channels=["C3"])])


def test_main_local_output_hard_links_recordings(tmp_path, monkeypatch):
    import json
    import os
    from dataset_maker import make_joint_cohort as mj
    monkeypatch.setattr(mj.hx, "s3_client", lambda profile: None)
    specs = []
    for name, stem, subj in (("tuh", "aaaa_s001_t000", "aaaa"),
                             ("heedb", "sub-S00011_ses-1_task-EEG", "1")):
        _, man, meta, split, _ = _part(name, [stem], [subj])
        side = tmp_path / name
        (side / "recordings").mkdir(parents=True)
        for fname, obj in (("manifest.json", man), ("age_metadata.json", meta),
                           ("age_split.json", split)):
            (side / fname).write_text(json.dumps(obj))
        (side / "recordings" / f"{stem}.npy").write_bytes(b"x" * 10)
        specs += ["--part", f"{name}={side},{side / 'recordings'}"]
    out = tmp_path / "joint"
    mj.main(specs + ["--out", str(out)])
    man = json.loads((out / "manifest.json").read_text())
    assert [r["corpus"] for r in man["recordings"]] == ["tuh", "heedb"]
    linked = out / "recordings" / "aaaa_s001_t000.npy"
    assert os.stat(linked).st_ino == os.stat(tmp_path / "tuh" / "recordings" / linked.name).st_ino
    mj.main(specs + ["--out", str(out)])                 # rerun: existing links are kept
