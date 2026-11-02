"""HEEDB cohort build (dataset_maker/make_heedb_cohort.py) on a synthetic index."""
import argparse

import pandas as pd
import pytest

from dataset_maker import heedb_index as hx
from dataset_maker import make_heedb_cohort as mc


def _index():
    rows = []
    for pid, site, label, neuro in [(1, "S0001", "normal", False), (1, "S0002", "abnormal", True),
                                    (2, "S0001", "normal", None), (3, "I0002", None, None),
                                    (4, "S0001", "normal", True), (5, "S0002", "abnormal", False)]:
        for ses in ("1", "2"):
            rows.append({"stem": f"sub-{site}{pid}_ses-{ses}_task-EEG", "patient_id": pid,
                         "subject": f"{site}{pid}", "site": site, "session": ses, "age": 40.0 + pid,
                         "sex": "F", "year": 2018, "label": label, "routine": True, "adult": True,
                         "in_sheet": label is not None,
                         "abnormal_report": pd.NA if label is None else label == "abnormal",
                         "any_neuro_dx": pd.NA if neuro is None else neuro,
                         "fg_seizure": label == "abnormal", "icd_seizure_disorders": neuro})
    return pd.DataFrame(rows).astype({"patient_id": "Int64", "abnormal_report": "boolean",
                                      "any_neuro_dx": "boolean"})


def _args(experiment):
    return argparse.Namespace(experiment=experiment, source="s3://b/src", seed=1, max_per_subject=10,
                              train_minutes=1, trim_sec=10, min_minutes=1.0)


@pytest.fixture
def parts():
    idx = _index()
    manifest = {"format_version": 2, "channels": ["FP1"], "recordings": [
        {"stem": s, "n_windows": 12, "n_samples": 24000} for s in idx["stem"]]}
    split = {"patients": {"1": "train", "2": "val", "3": "test", "4": "train", "5": "train"}}
    return idx, manifest, split


def test_routine_cohort_keeps_patient_splits_and_caps_train_windows(parts):
    manifest, meta, files, counts, j = mc.build(*parts, _args("routine"))
    assert counts["train"]["n_recordings"] == 8 and counts["val"]["n_subjects"] == 1
    assert {r["source_split"] for r in manifest["recordings"]} == {"train", "val", "test"}
    # Patient 1's sessions at two sites are both in train.
    assert set(j.loc[j["patient_id"] == 1, "split"]) == {"train"}
    # train: 1 min = 6 windows inside a 1-window trim; val/test: every window.
    stem = "sub-S00011_ses-1_task-EEG"
    assert len([f for f in files["train"] if stem in f]) == 6
    assert len([f for f in files["val"] if "sub-S00012_ses-1" in f]) == 12
    assert meta[stem]["subject"] == "1" and meta[stem]["label"] == "normal"
    labels = mc.label_table(j)
    assert {"fg_seizure", "icd_seizure_disorders"} <= set(labels.columns)


def test_normal_cohort_drops_abnormal_and_known_neuro(parts):
    *_, j = mc.build(*parts, _args("normal"))
    # report-normal, no known neuro dx; patient 2 (no ICD record) passes.
    assert sorted(j["patient_id"].unique()) == [1, 2]
    assert set(j["site"]) == {"S0001"}


def test_short_recordings_are_dropped(parts):
    idx, manifest, split = parts
    manifest["recordings"][0]["n_windows"] = 3                       # 30 s < 1 min
    *_, j = mc.build(idx, manifest, split, _args("routine"))
    assert manifest["recordings"][0]["stem"] not in set(j["stem"])
