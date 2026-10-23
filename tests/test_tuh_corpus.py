"""Whole-TUH tooling: EDF signal-header scan, download chunking, cross-corpus index.

Synthetic EDF headers only (no signal data is needed: everything here is
header-level), so the suite runs without any TUH data.
"""
import json

import pandas as pd
import pytest

from dataset_maker import download_tuh as dl
from dataset_maker import tuh_index as ti
from labram.data.tuh_metadata import (
    TUH_CHANNELS,
    canonical_channel,
    corpus_labels,
    read_edf_signal_header,
    recording_row,
    scan_recordings,
)


def write_edf_header(path, labels, *, sfreq=250, n_records=600, age=42, sex="F"):
    """A full EDF header (main block + one block per signal), no data records."""
    stem = path.stem
    subject = stem.split("_")[0]
    ns = len(labels)

    def fields(values, width):
        return b"".join(str(v).encode("latin-1").ljust(width) for v in values)

    main = (b"0".ljust(8)
            + f"{subject} {sex} 01-JAN-0000 {subject} Age:{age}".encode().ljust(80)
            + f"Startdate 01-JAN-2012 {subject}_s001 XXX X".encode().ljust(80)
            + b"01.01.12".ljust(8) + b"00.00.00".ljust(8) + str(256 * (ns + 1)).encode().ljust(8)
            + b"".ljust(44) + str(n_records).encode().ljust(8) + b"1".ljust(8)
            + str(ns).encode().ljust(4))
    assert len(main) == 256
    sig = (fields(labels, 16) + fields([""] * ns, 80) + fields(["uV"] * ns, 8)
           + fields([-1000] * ns, 8) + fields([1000] * ns, 8)
           + fields([-32768] * ns, 8) + fields([32767] * ns, 8)
           + fields([""] * ns, 80) + fields([sfreq] * ns, 8) + fields([""] * ns, 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(main + sig)
    return path


REF = [f"EEG {c}-REF" for c in TUH_CHANNELS] + ["EEG EKG1-REF", "PHOTIC-REF"]


# ------------------------------------------------------------ header parsing
def test_signal_header_gives_labels_sfreq_and_duration(tmp_path):
    p = write_edf_header(tmp_path / "aaaaaaaa_s001_t000.edf", REF, sfreq=256, n_records=1200)
    h = read_edf_signal_header(str(p))
    assert h.labels == tuple(REF)
    assert set(h.sfreqs) == {256.0}
    assert h.duration_sec == 1200


@pytest.mark.parametrize("label,name", [("EEG FP1-REF", "FP1"), ("EEG FP1-LE", "FP1"),
                                        ("EEG T1-REF", "T1"), ("EEG EKG1-REF", None),
                                        ("EEG 26-REF", None), ("PHOTIC-REF", None)])
def test_canonical_channel(label, name):
    assert canonical_channel(label) == name


def test_row_flags_missing_ear_channels_and_montage(tmp_path):
    labels = [f"EEG {c}-LE" for c in TUH_CHANNELS if c not in ("A1", "A2")]
    p = write_edf_header(tmp_path / "edf/000/aaaaaaaa/s002_2015_12_30/04_tcp_le_a/aaaaaaaa_s002_t001.edf",
                         labels)
    row = recording_row(str(p), str(tmp_path), "tueg")
    assert row["montage"] == "04_tcp_le_a"
    assert row["missing_channels"] == ["A1", "A2"]
    assert row["session_date"] == "2015-12-30"
    assert (row["subject"], row["session"], row["token"]) == ("aaaaaaaa", "s002", "t001")
    assert row["abnormal"] is None and row["epilepsy"] is None


def test_corpus_labels_from_path():
    assert corpus_labels("edf/train/abnormal/01_tcp_ar/x.edf") == {"abnormal": 1, "epilepsy": None}
    assert corpus_labels("00_epilepsy/aaa/s001/01_tcp_ar/x.edf") == {"abnormal": None, "epilepsy": 1}
    assert corpus_labels("01_no_epilepsy/aaa/s001/02_tcp_le/x.edf")["epilepsy"] == 0


def test_scan_keeps_unreadable_files_as_error_rows(tmp_path):
    write_edf_header(tmp_path / "a/aaaaaaaa_s001_t000.edf", REF)
    (tmp_path / "a/bbbbbbbb_s001_t000.edf").write_bytes(b"short")
    rows = scan_recordings(str(tmp_path), corpus="tueg")
    assert len(rows) == 2 and sum("error" in r for r in rows) == 1


# ----------------------------------------------------------------- download
LISTING = """\
drwxr-xr-x          4,096 2023/01/01 00:00:00 .
-rw-r--r--         12,345 2023/01/01 00:00:00 AAREADME.txt
drwxr-xr-x          4,096 2023/01/01 00:00:00 edf
drwxr-xr-x          4,096 2023/01/01 00:00:00 edf/000
-rw-r--r--      1,000,000 2023/01/01 00:00:00 edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf
-rw-r--r--      2,000,000 2023/01/01 00:00:00 edf/000/aaaaaaab/s001_2015_12_30/01_tcp_ar/aaaaaaab_s001_t000.edf
-rw-r--r--        500,000 2023/01/01 00:00:00 edf/001/aaaaaaac/s001_2016_01_01/02_tcp_le/aaaaaaac_s001_t000.edf
"""


def test_listing_is_grouped_into_chunks():
    sizes = dl.chunk_sizes(dl.parse_list_only(LISTING), "edf")
    assert sizes == {"000": {"n_files": 2, "bytes": 3_000_000},
                     "001": {"n_files": 1, "bytes": 500_000},
                     "_meta": {"n_files": 1, "bytes": 12_345}}
    assert dl.chunk_sizes(dl.parse_list_only(LISTING), None)["_all"]["n_files"] == 4


def test_rsync_command_targets_one_chunk():
    corpus = dl.CORPORA["tueg"]
    cmd = dl.rsync_cmd(corpus, "007", "/local/tueg/v2.0.2", "/k")
    assert cmd[-2] == f"{dl.NEDC_HOST}:data/tuh_eeg/tuh_eeg/v2.0.2/edf/007/"
    assert cmd[-1] == "/local/tueg/v2.0.2/edf/007/"
    meta = dl.rsync_cmd(corpus, dl.META, "/local/tueg/v2.0.2", "/k")
    assert "--exclude" in meta and meta[-1] == "/local/tueg/v2.0.2/"


def test_state_survives_reload(tmp_path):
    st = dl.State(str(tmp_path), "tueg")
    st.update("000", status="done", bytes=5)
    assert dl.State(str(tmp_path), "tueg").chunks()["000"] == {"status": "done", "bytes": 5}


# --------------------------------------------------------------------- index
def _row(stem, corpus, path, **kw):
    base = {"stem": stem, "subject": stem.split("_")[0], "session": "s001", "token": "t000",
            "age": 40, "raw_age": 40, "sex": "F", "corpus": corpus, "path": path,
            "montage": "01_tcp_ar", "sfreq": 250.0, "duration_sec": 1200.0,
            "missing_channels": [], "abnormal": None, "epilepsy": None}
    base.update(kw)
    return base


def test_index_merges_copies_of_a_recording():
    tueg = ti.rows_to_frame([_row("aaa_s001_t000", "tueg", "edf/000/aaa/x.edf"),
                             _row("bbb_s001_t000", "tueg", "edf/000/bbb/x.edf")])
    tuab = ti.rows_to_frame([_row("aaa_s001_t000", "tuab", "eval/abnormal/01_tcp_ar/x.edf",
                                  abnormal=1)])
    tuep = ti.rows_to_frame([_row("aaa_s001_t000", "tuep", "00_epilepsy/aaa/x.edf", epilepsy=1)])
    idx = ti.build_index([tueg, tuab, tuep]).set_index("stem")
    assert len(idx) == 2
    a = idx.loc["aaa_s001_t000"]
    assert a["source"] == "tuab" and a["corpora"] == "tuab,tueg,tuep"
    assert (a["abnormal"], a["epilepsy"], a["source_split"]) == (1, 1, "eval")
    assert pd.isna(idx.loc["bbb_s001_t000", "abnormal"])


def test_leakage_audit_finds_eval_subjects_elsewhere():
    tueg = ti.rows_to_frame([_row("aaa_s002_t000", "tueg", "edf/000/aaa/y.edf")])
    tuab = ti.rows_to_frame([_row("aaa_s001_t000", "tuab", "eval/normal/01_tcp_ar/x.edf", abnormal=0),
                             _row("ccc_s001_t000", "tuab", "train/normal/01_tcp_ar/z.edf", abnormal=0)])
    audit = ti.leakage_audit(ti.build_index([tueg, tuab]))
    assert audit["tuab_eval_subjects"] == 1
    assert audit["tuab_eval_subjects_other_recordings"] == 1
    assert audit["subject_overlap"]["tuab"]["tueg"] == 1
    summary = ti.summarize(ti.build_index([tueg, tuab]))
    json.dumps(summary)        # must be JSON-able as is
    assert summary["per_corpus"]["tuab"]["recordings"] == 2


def test_chunk_scan_paths_are_relative_to_corpus_root(tmp_path):
    root = tmp_path / "v2.0.2"
    write_edf_header(root / "edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf", REF)
    write_edf_header(root / "edf/001/aaaaaaab/s001_2015_12_30/01_tcp_ar/aaaaaaab_s001_t000.edf", REF)
    out = tmp_path / "000.json"
    assert dl.scan_chunk(str(root), "edf/000", str(out), "tueg") == 1
    assert json.loads(out.read_text())[0]["path"] == \
        "edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf"
