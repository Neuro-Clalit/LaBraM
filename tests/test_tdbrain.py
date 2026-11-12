"""TDBRAIN reader of make_external_npy: BIDS BrainVision files + participants.tsv -> test-only cohort."""
import json
import os
import sys

import numpy as np
import pandas as pd

from dataset_maker import make_external_npy as mx

FS = 500
EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
       "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
EXTRA = ["VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"]


def _write_brainvision(path_vhdr, seconds, offset_uv=20000.0, seed=0):
    """Minimal BrainVision triplet (float32, multiplexed, µV) with a DC offset."""
    base = os.path.basename(path_vhdr)[:-5]
    chans = EEG + EXTRA
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = (offset_uv + 10 * np.sin(2 * np.pi * 10 * t)[:, None]
         + rng.normal(0, 5, (len(t), len(chans)))).astype("<f4")
    x.tofile(path_vhdr[:-5] + ".eeg")
    infos = "\n".join(f"Ch{i}={c},,1,µV" for i, c in enumerate(chans, 1))
    with open(path_vhdr, "w", encoding="utf-8") as fh:
        fh.write("Brain Vision Data Exchange Header File Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
                 f"DataFile={base}.eeg\nMarkerFile={base}.vmrk\nDataFormat=BINARY\n"
                 f"DataOrientation=MULTIPLEXED\nNumberOfChannels={len(chans)}\nSamplingInterval={1e6 / FS:g}\n\n"
                 f"[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n[Channel Infos]\n{infos}\n")
    with open(path_vhdr[:-5] + ".vmrk", "w", encoding="utf-8") as fh:
        fh.write("Brain Vision Data Exchange Marker File, Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
                 f"DataFile={base}.eeg\n\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")


def _tdbrain(tmp_path):
    raw = tmp_path / "raw"
    sessions = [("87974621", "1"), ("87974621", "2"), ("88001234", "1"), ("19681349", "1"), ("88005678", "1")]
    for k, (sub, ses) in enumerate(sessions):
        d = raw / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        d.mkdir(parents=True)
        for j, task in enumerate(("restEC", "restEO")):
            _write_brainvision(str(d / f"sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr"), 121, seed=10 * k + j)
    deriv = raw / "derivatives" / "sub-87974621" / "ses-1" / "eeg"
    deriv.mkdir(parents=True)
    (deriv / "sub-87974621_ses-1_task-restEC_eeg.csv").write_text("Fp1\n0\n")
    pd.DataFrame({
        "participants_ID": ["sub-87974621", "sub-87974621", "sub-88001234", "sub-19681349", "sub-88005678"],
        "DISC/REP": ["DISCOVERY", "DISCOVERY", "DISCOVERY", "REPLICATION", "DISCOVERY"],
        "indication": ["MDD/ADHD", "MDD/ADHD", "HEALTHY", "REPLICATION", ""],
        "formal_status": ["DX", "DX", "HEALTHY", "", "UNKNOWN"],
        "sessID": ["1", "2", "1", "1", "1"],
        "age": ["35,52", "36,10", "61,9", "12,3", ""],
        "gender": ["1", "1", "0", "0", "1"],
        "neoFFI_q1": ["2", "3", "1", "", "4"],
    }).to_csv(raw / "participants.tsv", sep="\t", index=False)
    json.dump({"gender": {"Levels": {"0": "female", "1": "male"}}}, open(raw / "participants.json", "w"))
    return raw


def test_indication_flags():
    f = mx.tdbrain_indication_flags("MDD/ADHD")
    assert f["dx_mdd"] and f["dx_adhd"] and not f["healthy"] and f["clinical_group"] == "dx_mdd"
    assert mx.tdbrain_indication_flags("HEALTHY")["clinical_group"] == "healthy"
    assert mx.tdbrain_indication_flags("")["clinical_group"] == "unknown"
    assert mx.tdbrain_indication_flags("TINNITUS")["dx_other"]
    assert mx.tdbrain_indication_flags("Burnout")["clinical_group"] == "dx_other"


def test_clinical_table_decimal_comma_sex_levels_and_blinded_set(tmp_path):
    t = mx.tdbrain_clinical_table(str(_tdbrain(tmp_path) / "participants.tsv"))
    assert list(t["subject"])[:3] == ["87974621", "87974621", "88001234"]
    assert t["age"].iloc[0] == 35.52 and np.isnan(t["age"].iloc[4])
    assert list(t["sex"]) == ["M", "M", "F", "F", "M"]
    assert t["neoFFI_q1"].iloc[1] == "3"                              # original columns kept
    assert t.loc[3, "clinical_group"] == "blinded" and t.loc[3, "blinded"]
    assert t.loc[2, "healthy"] and t.loc[4, "clinical_group"] == "unknown"


def test_recordings_one_per_session_and_task(tmp_path):
    raw = _tdbrain(tmp_path)
    recs = mx.tdbrain_recordings(str(raw), str(raw / "participants.tsv"))
    stems = [r["stem"] for r in recs]
    # 4 sessions with an age x 2 tasks; derivatives skipped, missing age dropped
    assert len(recs) == 8 and "sub-88005678_ses-1_restEC_tdbrain" not in stems
    second = next(r for r in recs if r["stem"] == "sub-87974621_ses-2_restEO_tdbrain")
    assert second["age"] == 36.10 and second["age_session_matched"] and second["task"] == "restEO"
    assert second["label"] is None and second["dx_mdd"] is True
    healthy = next(r for r in recs if r["subject"] == "88001234")
    assert healthy["label"] == "normal" and healthy["sex"] == "F"


def test_convert_writes_cohort_with_19_channels_and_clinical_table(tmp_path, monkeypatch):
    raw = _tdbrain(tmp_path)
    out = tmp_path / "cohort"
    monkeypatch.setattr(sys, "argv", ["make_external_npy", "--dataset", "tdbrain", "--raw", str(raw),
                                      "--meta", str(raw / "participants.tsv"), "--out", str(out), "--jobs", "1",
                                      "--limit", "2"])
    mx.main()
    m = json.load(open(out / "manifest.json"))
    assert m["n_recordings"] == 2 and not m["cohort"]["errors"]
    r = m["recordings"][0]
    assert r["n_windows"] == 12 and sum(r["channel_mask"]) == 19       # A1/A2/T1/T2 absent
    assert r["annotations"] == {"New Segment/": 1} and r["clinical_group"] == "blinded"   # sub-19* sorts first
    x = np.load(out / "recordings" / f"{r['stem']}.npy")
    assert x.shape == (12 * 2000, 23) and x.dtype == np.float16
    # DC offset (20 mV) removed: no edge transient beyond the 10 µV alpha + noise
    assert np.abs(x[:, np.array(r["channel_mask"])].astype(np.float32)).max() < 200
    ages = json.load(open(out / "age_metadata.json"))["recordings"]
    assert ages[r["stem"]]["age"] == 12.3 and ages[r["stem"]]["label"] is None
    assert len(pd.read_parquet(out / "clinical.parquet")) == 5
