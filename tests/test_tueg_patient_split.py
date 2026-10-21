"""The shared TUEG patient split: anchored patients keep their split, the rest
are hashed (stable, roughly val_fraction), and make_cohort_dataset uses it so a
patient lands in the same split for every cohort."""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_maker import make_cohort_dataset as mc
from dataset_maker.tueg_patient_split import build_patient_split, hash_split, split_subjects


def test_hash_split_is_stable_and_near_the_fraction():
    subjects = [f"s{i:05d}" for i in range(20000)]
    first = [hash_split(s, 7, 0.1) for s in subjects]
    assert first == [hash_split(s, 7, 0.1) for s in subjects]
    assert abs(np.mean([x == "val" for x in first]) - 0.1) < 0.01
    assert first != [hash_split(s, 8, 0.1) for s in subjects]          # the seed matters


def test_anchors_win_and_adding_patients_changes_nobody_else():
    anchors = {"a": "val", "b": "train"}
    one, src = build_patient_split({"a", "b", "c", "d"}, anchors, seed=1, val_fraction=0.5)
    assert one["a"] == "val" and one["b"] == "train" and src == {"anchor": 2, "hash": 2}
    two, _ = build_patient_split({"a", "b", "c", "d", "e"}, anchors, seed=1, val_fraction=0.5)
    assert all(two[s] == one[s] for s in one)


def test_split_subjects_rejects_a_patient_in_two_splits(tmp_path):
    p = tmp_path / "age_split.json"
    json.dump({"files": {"train": ["train/aaa_s001_t000_0.pkl"],
                         "val": ["val/aaa_s002_t000_0.pkl"]}}, open(p, "w"))
    with pytest.raises(ValueError, match="aaa"):
        split_subjects(str(p))


def _index():
    rows = []
    for subj, n in (("tuab1", 2), ("tueg1", 2), ("tueg2", 1), ("tueg3", 3)):
        rows += [{"subject": subj, "source_split": "eval" if subj == "tuab1" and k else None}
                 for k in range(n)]
    return pd.DataFrame(rows)


def test_assign_splits_uses_the_shared_split_for_tueg_patients():
    j = _index()
    pool = pd.Series(True, index=j.index)
    test = j.source_split == "eval"
    shared = {"tueg1": "val", "tueg2": "train", "tueg3": "train"}
    split = mc.assign_splits(j, pool, test, "all_both", {"tuab1": "train"},
                             val_fraction=0.5, seed=0, patient_split=shared)
    assert list(split[j.subject == "tueg1"]) == ["val", "val"]
    assert set(split[j.subject == "tueg3"]) == {"train"}
    assert list(split[j.subject == "tuab1"]) == ["train", "test"]
    with pytest.raises(ValueError, match="not in the patient split"):
        mc.assign_splits(j, pool, test, "all_both", {"tuab1": "train"}, val_fraction=0.5,
                         seed=0, patient_split={"tueg1": "val"})
