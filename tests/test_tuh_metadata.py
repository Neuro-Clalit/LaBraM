"""Tests for data.tuh_metadata: parsing patient demographics out of EDF headers.

Builds synthetic 256-byte EDF header blocks in tmp_path rather than reading real
recordings, so the suite needs no external dataset. The byte offsets and token
layout asserted here were validated against TUAB v3.0.0, where the parsed ages
reproduce the corpus AAREADME's DEMOGRAPHICS tables exactly.
"""
import json

import pytest

from labram.data.tuh_metadata import (
    EDF_HEADER_BYTES,
    REDACTED_AGE,
    age_lookup,
    filter_files_with_age,
    find_metadata_sidecar,
    load_age_lookup_for,
    load_metadata_sidecar,
    parse_edf_header_metadata,
    recording_stem,
    save_metadata_sidecar,
    scan_corpus_metadata,
    summarize_metadata,
)


def _header(patient: str, recording: str) -> bytes:
    """An EDF header block: version(8) + patient(80) + recording(80) + rest(88)."""
    block = (
        b"0".ljust(8)
        + patient.encode("latin-1").ljust(80)
        + recording.encode("latin-1").ljust(80)
    )
    return block.ljust(EDF_HEADER_BYTES)


def _write_edf(directory, stem, *, age=42, sex="F", year=2012):
    directory.mkdir(parents=True, exist_ok=True)
    subject = stem.split("_")[0]
    session = stem.split("_")[1] if "_" in stem else "s001"
    path = directory / f"{stem}.edf"
    path.write_bytes(_header(
        f"{subject} {sex} 01-JAN-0000 {subject} Age:{age}",
        f"Startdate 01-JAN-{year} {subject}_{session} XXX X",
    ))
    return path


def test_parses_subject_sex_age_and_year(tmp_path):
    path = _write_edf(tmp_path, "aaaaantl_s001_t001", age=42, sex="F", year=2012)
    meta = parse_edf_header_metadata(str(path))

    assert meta.stem == "aaaaantl_s001_t001"
    assert meta.subject == "aaaaantl"
    assert meta.session == "s001"
    assert meta.token == "t001"
    assert meta.age == 42
    assert meta.raw_age == 42
    assert meta.sex == "F"
    assert meta.year == 2012
    assert not meta.is_redacted


def test_redacted_age_999_is_excluded_but_recorded(tmp_path):
    """TUH writes Age:999 for patients 90+ (HIPAA). It must never become a target."""
    path = _write_edf(tmp_path, "aaaaaiae_s001_t000", age=REDACTED_AGE)
    meta = parse_edf_header_metadata(str(path))

    assert meta.age is None, "the sentinel must not be exposed as a usable age"
    assert meta.raw_age == REDACTED_AGE
    assert meta.is_redacted


def test_age_zero_is_excluded_by_default_range(tmp_path):
    path = _write_edf(tmp_path, "aaaaakuy_s002_t002", age=0)
    assert parse_edf_header_metadata(str(path)).age is None


def test_age_range_is_configurable(tmp_path):
    path = _write_edf(tmp_path, "aaaaaaaa_s001_t000", age=95)
    assert parse_edf_header_metadata(str(path)).age is None
    assert parse_edf_header_metadata(str(path), max_age=100).age == 95


def test_missing_age_token_yields_none_rather_than_raising(tmp_path):
    tmp_path.joinpath("aaaaabbb_s001_t000.edf").write_bytes(
        _header("aaaaabbb M 01-JAN-0000 aaaaabbb", "Startdate 01-JAN-2011 aaaaabbb_s001"))
    meta = parse_edf_header_metadata(str(tmp_path / "aaaaabbb_s001_t000.edf"))

    assert meta.raw_age is None
    assert meta.age is None
    assert meta.sex == "M"


def test_malformed_patient_field_does_not_raise(tmp_path):
    tmp_path.joinpath("weird.edf").write_bytes(_header("", ""))
    meta = parse_edf_header_metadata(str(tmp_path / "weird.edf"))

    assert meta.stem == "weird"
    assert meta.subject == "weird"          # falls back to the leading token
    assert (meta.age, meta.sex, meta.year, meta.session) == (None, None, None, None)


def test_truncated_header_raises(tmp_path):
    tmp_path.joinpath("short.edf").write_bytes(b"tooshort")
    with pytest.raises(ValueError, match="truncated"):
        parse_edf_header_metadata(str(tmp_path / "short.edf"))


def test_scan_walks_nested_corpus_layout_and_summarizes(tmp_path):
    _write_edf(tmp_path / "train" / "normal" / "01_tcp_ar", "aaaaaaaa_s001_t000", age=30, sex="F")
    _write_edf(tmp_path / "train" / "abnormal" / "01_tcp_ar", "aaaaabbb_s001_t000", age=50, sex="M")
    _write_edf(tmp_path / "eval" / "normal" / "01_tcp_ar", "aaaaaccc_s001_t000",
               age=REDACTED_AGE, sex="F")

    meta = scan_corpus_metadata(str(tmp_path))
    assert set(meta) == {"aaaaaaaa_s001_t000", "aaaaabbb_s001_t000", "aaaaaccc_s001_t000"}

    summary = summarize_metadata(meta)
    assert summary["recordings"] == 3
    assert summary["with_age_field"] == 3
    assert summary["usable_ages"] == 2
    assert summary["redacted_90_plus"] == 1
    assert summary["age_mean"] == 40.0
    assert summary["sex_counts"] == {"F": 2, "M": 1}


def test_scan_raises_when_no_edf_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_corpus_metadata(str(tmp_path))


def test_sidecar_round_trips(tmp_path):
    _write_edf(tmp_path, "aaaaaaaa_s001_t000", age=30)
    meta = scan_corpus_metadata(str(tmp_path))
    out = tmp_path / "age_metadata.json"

    save_metadata_sidecar(meta, str(out))
    assert json.loads(out.read_text())["version"] == 1
    assert load_metadata_sidecar(str(out)) == meta


def test_age_lookup_drops_sentinels_and_returns_floats(tmp_path):
    _write_edf(tmp_path, "aaaaaaaa_s001_t000", age=30)
    _write_edf(tmp_path, "aaaaabbb_s001_t000", age=REDACTED_AGE)

    ages = age_lookup(scan_corpus_metadata(str(tmp_path)))
    assert ages == {"aaaaaaaa_s001_t000": 30.0}
    assert isinstance(ages["aaaaaaaa_s001_t000"], float)


@pytest.mark.parametrize("filename, expected", [
    ("aaaaaaaq_s004_t000_0.pkl", "aaaaaaaq_s004_t000"),
    ("aaaaaaaq_s004_t000_17.pkl", "aaaaaaaq_s004_t000"),
    # A window file may carry its split subdirectory, so ids come off the basename.
    ("train/aaaaaaaq_s004_t000_3.pkl", "aaaaaaaq_s004_t000"),
    ("aaaaaaaq_s004_t000_3", "aaaaaaaq_s004_t000"),
])
def test_recording_stem_handles_paths_and_window_indices(filename, expected):
    assert recording_stem(filename) == expected


def test_filter_files_with_age_drops_unlabelled_windows():
    ages = {"aaaaaaaa_s001_t000": 30.0}
    files = [
        "aaaaaaaa_s001_t000_0.pkl",
        "aaaaaaaa_s001_t000_1.pkl",
        "aaaaabbb_s001_t000_0.pkl",   # no age -> must be dropped
    ]
    assert filter_files_with_age(files, ages) == files[:2]


def test_sidecar_is_discovered_from_a_child_directory(tmp_path):
    """A loader must recover its labels from its data root alone: cross-validation
    rebuilds loaders positionally, with no chance to pass a lookup through."""
    _write_edf(tmp_path, "aaaaaaaa_s001_t000", age=30)
    meta = scan_corpus_metadata(str(tmp_path))
    save_metadata_sidecar(meta, str(tmp_path / "processed" / "age_metadata.json"))

    nested = tmp_path / "processed" / "train"
    nested.mkdir(parents=True, exist_ok=True)

    assert find_metadata_sidecar(str(nested)) == str(
        tmp_path / "processed" / "age_metadata.json")
    assert load_age_lookup_for(str(nested)) == {"aaaaaaaa_s001_t000": 30.0}


def test_missing_sidecar_reports_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_TUAB_age"):
        load_age_lookup_for(str(tmp_path))
