"""Harvard EEG Database index (dataset_maker/heedb_index.py) and its analytics
helpers (labram/eval/heedb_analysis.py), on synthetic BIDS metadata and EDFs.

No network: S3 range reads go to a fake client that serves a local file.
"""
import io

import numpy as np
import pandas as pd
import pytest

from dataset_maker import heedb_index as hx
from dataset_maker import tuh_clinical
from labram.data.tuh_metadata import (bids_canonical_channel, channel_map, edf_header_size,
                                      parse_edf_signal_header, read_edf_signal_header)

LABELS = ["Fp1", "Fp2", "C3", "C4", "T7", "P8", "O1", "O2", "A1", "Cz", "ECGL", "LOC", "CHIN1"]


def write_bids_edf(path, labels, signals, sfreq, *, edf_type="EDF+C"):
    """Minimal EDF with bare BIDS labels: 1 s records, 0.1 µV per digital unit."""
    ns, n_rec = len(labels), signals.shape[1] // sfreq

    def fields(values, width):
        return b"".join(str(v).encode("latin-1").ljust(width)[:width] for v in values)

    main = (b"0".ljust(8) + b"X X X X".ljust(80) + b"Startdate X X X X".ljust(80)
            + b"01.01.18" + b"00.00.00" + str(256 * (ns + 1)).encode().ljust(8)
            + edf_type.encode().ljust(44) + str(n_rec).encode().ljust(8) + b"1".ljust(8)
            + str(ns).encode().ljust(4))
    sig = (fields(labels, 16) + fields([""] * ns, 80) + fields(["uV"] * ns, 8)
           + fields(["-3276.8"] * ns, 8) + fields(["3276.7"] * ns, 8)
           + fields([-32768] * ns, 8) + fields([32767] * ns, 8)
           + fields(["HP:0.1Hz"] * ns, 80) + fields([sfreq] * ns, 8) + fields([""] * ns, 32))
    digital = np.clip(np.round(signals * 10), -32768, 32767).astype("<i2")
    records = digital[:, :n_rec * sfreq].reshape(ns, n_rec, sfreq).transpose(1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(main + sig + records.tobytes())
    return path


class FakeS3:
    """``get_object(Bucket, Key, Range)`` over local files keyed by S3 key."""

    def __init__(self, files):
        self.files = files

    def get_object(self, Bucket, Key, Range):
        data = self.files[Key].read_bytes()
        lo, hi = (int(v) for v in Range.split("=")[1].split("-"))
        return {"Body": io.BytesIO(data[lo:hi + 1])}


def _signals(n_ch, seconds, sfreq, seed=0):
    t = np.arange(seconds * sfreq) / sfreq
    rng = np.random.default_rng(seed)
    return np.stack([20 * np.sin(2 * np.pi * (8 + i % 4) * t) + rng.normal(0, 5, t.size)
                     for i in range(n_ch)])


# ------------------------------------------------------------------ channels
@pytest.mark.parametrize("label,name", [
    ("Fp1", "FP1"), ("T7", "T3"), ("T8", "T4"), ("P7", "T5"), ("P8", "T6"), ("T3", "T3"),
    ("EEG C3-Ref", "C3"), ("EEG FP1-REF", "FP1"), ("cz", "CZ"),
    ("ECGL", None), ("LOC", None), ("DC1", None), ("EDF Annotations", None), ("F9", None)])
def test_bids_canonical_channel(label, name):
    assert bids_canonical_channel(label) == name


def test_tuh_channel_map_is_unchanged_by_default():
    """The TUH converter must still ignore bare labels (no silent behaviour change)."""
    assert channel_map(["Fp1", "EEG C3-REF"]) == {"C3": "EEG C3-REF"}
    assert channel_map(["Fp1", "T7"], bids_canonical_channel) == {"FP1": "Fp1", "T3": "T7"}


# -------------------------------------------------------------------- headers
def test_header_from_bytes_equals_header_from_file(tmp_path):
    path = write_bids_edf(tmp_path / "a.edf", LABELS, _signals(len(LABELS), 8, 256), 256)
    data = path.read_bytes()
    size = edf_header_size(data[:256])
    assert size == 256 * (len(LABELS) + 1)
    assert parse_edf_signal_header(data[:256], data[256:size]) == read_edf_signal_header(str(path))


def test_header_row_maps_bids_labels_and_aux_channels(tmp_path):
    path = write_bids_edf(tmp_path / "a.edf", LABELS, _signals(len(LABELS), 8, 256), 256)
    data = path.read_bytes()
    row = hx.header_row(data[:256], data[256:edf_header_size(data[:256])])
    assert row["edf_type"] == "EDF+C"
    assert row["sfreq"] == 256 and not row["mixed_sfreq"]
    assert row["n_tuh_channels"] == 10
    assert "T3" not in row["missing_channels"] and "T6" not in row["missing_channels"]
    assert {"F3", "T1", "T2", "A2"} <= set(row["missing_channels"])
    assert row["has_ecg"] and row["has_eog"] and row["has_emg"] and not row["has_resp"]
    assert row["lsb_uv"] == pytest.approx(0.1, rel=1e-3)
    assert row["header_duration_sec"] == 8


def test_fetch_header_reads_large_headers_in_two_requests(tmp_path):
    labels = [f"X{i}" for i in range(70)] + ["C3"]
    path = write_bids_edf(tmp_path / "big.edf", labels, _signals(len(labels), 2, 128), 128)
    head, block = hx.fetch_header(FakeS3({"k": path}), "k")
    assert len(block) == 256 * len(labels)
    assert parse_edf_signal_header(head, block).labels[-1] == "C3"


# ------------------------------------------------------------------- segments
def test_segment_is_a_valid_edf_with_the_requested_span(tmp_path):
    from labram.eval.heedb_analysis import fetch_segment_edf
    sig = _signals(len(LABELS), 30, 256)
    src = write_bids_edf(tmp_path / "src.edf", LABELS, sig, 256)
    out = fetch_segment_edf(FakeS3({"k": src}), "k", str(tmp_path / "seg.edf"), 10, 5)
    hdr = read_edf_signal_header(out)
    assert hdr.n_records == 5
    import mne
    raw = mne.io.read_raw_edf(out, preload=True, verbose="error")
    got = raw.get_data(picks=["C3"], units="uV")[0]
    np.testing.assert_allclose(got, np.round(sig[2, 10 * 256:15 * 256] * 10) / 10, atol=0.06)


def test_segment_keep_good_drops_flat_records(tmp_path):
    from labram.eval.heedb_analysis import fetch_segment_edf
    sig = _signals(len(LABELS), 20, 256)
    sig[:, 5 * 256:15 * 256] = 0.0                     # 10 s of zeros (flat)
    src = write_bids_edf(tmp_path / "src.edf", LABELS, sig, 256)
    out = fetch_segment_edf(FakeS3({"k": src}), "k", str(tmp_path / "seg.edf"), 0, 20, keep_good=8)
    assert read_edf_signal_header(out).n_records == 8   # 5 good + 3 of the last 5 good
    import mne
    x = mne.io.read_raw_edf(out, preload=True, verbose="error").get_data(units="uV")
    assert x[:10].std(axis=1).min() > 1


# ------------------------------------------------------------------- metadata
@pytest.fixture
def meta(tmp_path):
    root = tmp_path / "metadata"
    (root / "eeg-metadata").mkdir(parents=True)
    (root / "HEEDB_Metadata").mkdir()
    pd.DataFrame({
        "SiteID": ["S0001"] * 4, "BDSPPatientID": [101, 101, 102, None],
        "BidsFolder": ["sub-S0001101", "sub-S0001101", "sub-S0001102", "sub-S0001103"],
        "SessionID": [1, 2, 1, 1], "CreationTime": ["2018-01-01 10:00"] * 4,
        "StartTime": ["2018-01-01 10:00", "2018-02-01 10:00", "2019-01-01 10:00", "2020-01-01"],
        "EndTime": [""] * 4, "DurationInSeconds": [3600, 86400, 2400, 1800],
        "EEGFolder": ["rEEG", "cEEG", "rEEG", "rEEG"], "ServiceName": ["Routine", "LTM", "Routine", "Routine"],
        "HasXLTEKAnnotations": ["Y"] * 4, "HasPersystAnnotations": ["N"] * 4,
        "AgeAtVisit": [45, 45, 12, 70], "DateOfDeath": [None, None, None, "2021-01-01"],
        "SexDSC": ["Female", "Female", "Male", "Male"], "BidsFlag": [""] * 4,
    }).to_csv(root / "eeg-metadata" / "S0001_eeg_metadata.csv", index=False)
    pd.DataFrame({   # a site without a service label: routine by duration
        "SiteID": ["I0002", "I0002"], "BDSPPatientID": [201, 202],
        "BidsFolder": ["sub-I0002201", "sub-I0002202"], "SessionID": [1, 1],
        "CreationTime": [""] * 2, "StartTime": ["2015-05-05 08:00"] * 2, "EndTime": [""] * 2,
        "DurationInSeconds": [3000, 40000], "HasXLTEKAnnotations": ["Y"] * 2,
        "AgeAtVisit": [60, 61], "SexDSC": ["M", "F"], "BDSPLastModifiedDTS": [""] * 2,
    }).to_csv(root / "eeg-metadata" / "I0002_eeg_metadata.csv", index=False)
    pd.DataFrame({
        "SiteID": ["S0001", "S0001", "S0001"], "BDSPPatientID": [101, 102, 103], "SessionID": [1, 1, 1],
        "AgeAtVisit": [45, 12, 70], "SexDSC": ["Female", "Male", "Male"],
        "ServiceName(EEG)": ["Routine"] * 3, "DeidentifiedName(Reports)": ["r"] * 3,
        "normal": ["report", None, None], "abnormal": [None, "report", None],
        "spikes": [None, "report annotation", "annotation"], "seizure": [None, "verified", None],
        "gpd": [None, "report", None], "grda": [None, None, None],
        "n1": ["annotation", None, None], "N1": [None, "report", None],
    }).to_csv(root / "HEEDB_Metadata" / "S0001_EEG__reports_findings.csv", index=False)
    pd.DataFrame({
        "SiteID": ["S0001", "S0001", "S0001", "I0002"], "BDSPPatientID": [101, 102, 103, 201],
        "Sex": ["F", "M", "M", "M"], "AgeAtVisitAvg": [45, 12, 70, 60],
        "RaceAndEthnicity": ["White"] * 4, "RaceAndEthnicityDSC": ["White"] * 4,
        "VisitCount": [2, 1, 1, 1], "HasEEG": ["Y"] * 4, "HasReports": ["Y"] * 4,
        "MatchedEEGReports": ["Y"] * 4, "ICD10Count": [1, 2, 0, 0], "MedicationCount": [1, 0, 1, 0],
    }).to_csv(root / "HEEDB_Metadata" / "HEEDB_patients.csv", index=False)
    pd.DataFrame({
        "BDSPPatientID": [101, 102], "SiteID": ["S0001", "S0001"], "SexDSC": ["F", "M"],
        "VisitCount": [2, 1], "AgeAtVisitAvg": [45, 12],
        "Seizure Disorders": [None, "G40.909 R56.9"], "Cerebrovascular Diseases": [None, "I63.9"],
        "Miscellaneous": ["F32.9", None],
    }).to_csv(root / "HEEDB_Metadata" / "HEEDB_ICD10_for_Neurology.csv", index=False)
    pd.DataFrame({
        "BDSPPatientID": [101, 103], "SiteID": ["S0001", "S0001"], "SexDSC": ["F", "M"],
        "VisitCount": [2, 1], "AgeAtVisitAvg": [45, 70],
        "Nervous System Drugs": ["N05BA N06AB", "N03AX"], "Cardiovascular System Drugs": [None, "C07AB"],
    }).to_csv(root / "HEEDB_Metadata" / "HEEDB_Medication_ATC.csv", index=False)
    return str(root)


def test_recordings_align_sites_and_flag_routine_adults(meta):
    rec = hx.load_recordings(meta).set_index(["subject", "session"])
    assert rec.loc[("S0001103", "1"), "patient_id"] == 103          # id from the BIDS folder
    assert rec.loc[("S0001101", "1"), "routine"] and not rec.loc[("S0001101", "2"), "routine"]
    assert rec.loc[("S0001101", "1"), "routine_source"] == "service"
    assert rec.loc[("I0002201", "1"), "routine"] and not rec.loc[("I0002202", "1"), "routine"]
    assert rec.loc[("I0002201", "1"), "routine_source"] == "duration"
    assert not rec.loc[("S0001102", "1"), "adult"] and rec.loc[("S0001101", "1"), "adult"]
    assert rec.loc[("S0001101", "1"), "sex"] == "F" and rec.loc[("S0001101", "1"), "year"] == 2018
    assert rec.loc[("S0001103", "1"), "deceased"]


def test_reports_merge_case_variants_and_keep_provenance(meta):
    rep = hx.load_reports(meta).set_index("patient_id")
    assert rep.loc[102, "rf_n1"] == "report"                       # 'N1' merged into 'n1'
    assert rep.loc[102, "rf_spikes"] == "report annotation"
    assert rep.loc[101, "report_normal"] and not rep.loc[101, "abnormal_report"]
    assert rep.loc[102, "abnormal_report"]
    assert pd.isna(rep.loc[103, "abnormal_report"])                # neither call
    flags = hx.finding_flags(rep.reset_index()).set_index(rep.index)
    assert flags.loc[102, "find_spikes"] and flags.loc[102, "find_seizure"]   # verified counts
    assert not flags.loc[101, "find_n1"]                           # annotation only
    assert flags.loc[102, "fg_epileptiform"] and flags.loc[102, "fg_periodic_discharges"]
    assert pd.isna(flags.loc[103, "find_spikes"])                  # no report -> NA


def test_index_has_tuh_aligned_flags(meta):
    idx = hx.build_index(meta).set_index(["subject", "session"])
    epi = idx.loc[("S0001102", "1")]
    assert epi["dx_epilepsy"] and epi["dx_clinical_seizure"] and epi["dx_stroke"]
    assert epi["dx_electrographic_seizure"] and epi["f_gpd"] and epi["any_neuro_dx"]
    assert epi["label"] == "abnormal" and epi["in_sheet"]
    dep = idx.loc[("S0001101", "1")]
    assert dep["dx_depression"] and not dep["dx_epilepsy"] and not dep["any_neuro_dx"]
    assert dep["med_benzodiazepine"] and dep["sedated"]
    # No ICD record -> unknown, not healthy (as tuh_clinical.join_index).
    assert pd.isna(idx.loc[("S0001103", "1"), "dx_epilepsy"])
    assert pd.isna(idx.loc[("S0001103", "1"), "any_neuro_dx"])
    assert idx.loc[("S0001103", "1"), "med_antiseizure"] and not idx.loc[("S0001103", "1"), "sedated"]
    assert set(tuh_clinical.FLAGS) - {"f_oirda"} <= set(idx.columns)
    # The TUH normative rule runs unchanged: 101 is report-normal but sedated.
    assert not tuh_clinical.normative_mask(idx.reset_index()).any()


def test_edf_table_parses_bids_keys_and_sidecars():
    keys = ["EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_eeg.edf",
            "EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_channels.tsv",
            "EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_annotations.csv",
            "EEG/bids/S0001/sub-S0001101/ses-2/eeg/sub-S0001101_ses-2_task-cEEG_eeg.edf",
            "EEG/bids/S0001/sub-S0001101/ses-2/eeg/sub-S0001101_ses-2_task-_Xltek.csv",
            "EEG/bids/S0001/sub-S0001101/ses-2/sub-S0001101_ses-2_scans.tsv"]
    edf = hx.edf_table(pd.DataFrame({"key": keys, "bytes": range(len(keys))})).set_index("session")
    assert list(edf["stem"]) == ["sub-S0001101_ses-1_task-EEG", "sub-S0001101_ses-2_task-cEEG"]
    assert edf.loc["1", "task"] == "EEG" and edf.loc["2", "task"] == "cEEG"
    assert edf.loc["1", "has_annotations"] and not edf.loc["1", "has_xltek_csv"]
    assert edf.loc["2", "has_xltek_csv"] and edf.loc["1", "subject"] == "S0001101"


def test_code_flags_use_prefixes_and_keep_unknown():
    codes = pd.Series(["G40.909 F32.9", "I63.9", None])
    flags = hx.code_flags(codes, {"dx_epilepsy": ("G40",), "dx_stroke": ("I63",)})
    assert list(flags["dx_epilepsy"].astype(object)) == [True, False, pd.NA]
    assert list(flags["dx_stroke"].astype(object)) == [False, True, pd.NA]


def test_download_list_keeps_adult_routine_session_objects(meta):
    keys = ["EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_eeg.edf",
            "EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_channels.tsv",
            "EEG/bids/S0001/sub-S0001101/ses-1/sub-S0001101_ses-1_scans.tsv",
            "EEG/bids/S0001/sub-S0001101/ses-2/eeg/sub-S0001101_ses-2_task-cEEG_eeg.edf",   # LTM
            "EEG/bids/S0001/sub-S0001102/ses-1/eeg/sub-S0001102_ses-1_task-EEG_eeg.edf"]   # child
    dl = hx.download_list(pd.DataFrame({"key": keys, "bytes": [10, 1, 1, 100, 10]}), meta)
    assert sorted(dl["key"]) == sorted(keys[:3])


def test_small_cells_are_suppressed():
    from labram.eval.heedb_analysis import suppress_small
    df = pd.DataFrame({"n": [0, 3, 9, 10, 250], "rate": [0.0, 1.5, 2.0, 3.0, 4.0]})
    out = suppress_small(df, ["n"])
    assert list(out["n"]) == [0, "<10", "<10", 10, 250]
    assert list(out["rate"]) == list(df["rate"])
