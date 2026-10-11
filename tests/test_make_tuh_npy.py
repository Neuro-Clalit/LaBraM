"""EDF -> per-recording npy for TUH corpora (dataset_maker/make_TUH_npy.py).

Writes small but real EDF files (header + int16 data records) so MNE reads them
exactly as it reads TUH recordings.
"""
import json

import numpy as np
import pandas as pd
import pytest

from dataset_maker import make_TUH_npy as mk
from dataset_maker.tuh_index import rows_to_frame
from labram.data.tuh_metadata import TUH_CHANNELS, scan_recordings


def write_edf(path, labels, signals, sfreq, *, age=42):
    """Minimal EDF: 1 s records, int16 with a 1 digital unit = 0.1 µV scale."""
    ns, n_rec = len(labels), signals.shape[1] // sfreq
    subject = path.stem.split("_")[0]

    def fields(values, width):
        return b"".join(str(v).encode("latin-1").ljust(width)[:width] for v in values)

    main = (b"0".ljust(8)
            + f"{subject} F 01-JAN-0000 {subject} Age:{age}".encode().ljust(80)
            + f"Startdate 01-JAN-2012 {subject}_s001 XXX X".encode().ljust(80)
            + b"01.01.12" + b"00.00.00" + str(256 * (ns + 1)).encode().ljust(8)
            + b"".ljust(44) + str(n_rec).encode().ljust(8) + b"1".ljust(8) + str(ns).encode().ljust(4))
    sig = (fields(labels, 16) + fields([""] * ns, 80) + fields(["uV"] * ns, 8)
           + fields(["-3276.8"] * ns, 8) + fields(["3276.7"] * ns, 8)
           + fields([-32768] * ns, 8) + fields([32767] * ns, 8)
           + fields([""] * ns, 80) + fields([sfreq] * ns, 8) + fields([""] * ns, 32))
    digital = np.clip(np.round(signals * 10), -32768, 32767).astype("<i2")
    records = digital[:, :n_rec * sfreq].reshape(ns, n_rec, sfreq).transpose(1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(main + sig + records.tobytes())
    return path


def _signals(n_ch, seconds, sfreq, seed=0):
    t = np.arange(seconds * sfreq) / sfreq
    rng = np.random.default_rng(seed)
    return np.stack([30 * np.sin(2 * np.pi * (8 + i % 5) * t) + rng.normal(0, 5, t.size)
                     for i in range(n_ch)])


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "edf"
    full = [f"EEG {c}-REF" for c in TUH_CHANNELS] + ["EEG EKG1-REF"]
    write_edf(root / "000/aaaaaaaa/s001_2015_01_01/01_tcp_ar/aaaaaaaa_s001_t000.edf",
              full, _signals(len(full), 95, 250), 250)
    no_ears = [f"EEG {c}-LE" for c in TUH_CHANNELS if c not in ("A1", "A2")]
    write_edf(root / "000/aaaaaaab/s001_2015_01_01/04_tcp_le_a/aaaaaaab_s001_t000.edf",
              no_ears, _signals(len(no_ears), 70, 256, seed=1), 256)
    write_edf(root / "000/aaaaaaac/s001_2015_01_01/01_tcp_ar/aaaaaaac_s001_t000.edf",
              full, _signals(len(full), 30, 250, seed=2), 250)      # < 60 s: skipped
    scan = rows_to_frame(scan_recordings(str(root), corpus="tueg"))
    scan.to_parquet(tmp_path / "scan.parquet")
    return tmp_path


def _convert(tmp, *extra):
    mk.main(["convert", "--scan", str(tmp / "scan.parquet"), "--edf-root", str(tmp / "edf"),
             "--dst", str(tmp / "npy"), *extra])
    mk.main(["merge", "--scan", str(tmp / "scan.parquet"), "--dst", str(tmp / "npy")])
    return json.loads((tmp / "npy/manifest.json").read_text())


def test_converts_whole_windows_with_channel_mask(corpus):
    manifest = _convert(corpus)
    recs = {r["stem"]: r for r in manifest["recordings"]}
    assert set(recs) == {"aaaaaaaa_s001_t000", "aaaaaaab_s001_t000"}     # 30 s one skipped
    a = np.load(corpus / "npy/recordings/aaaaaaaa_s001_t000.npy")
    assert a.dtype == np.float16 and a.shape == (9 * 2000, 23)          # 95 s -> 9 windows
    assert recs["aaaaaaaa_s001_t000"]["channel_mask"] == [True] * 23
    b = np.load(corpus / "npy/recordings/aaaaaaab_s001_t000.npy")
    assert b.shape == (7 * 2000, 23)
    ears = [TUH_CHANNELS.index("A1"), TUH_CHANNELS.index("A2")]
    assert not b[:, ears].any() and b[:, 0].any()
    assert [i for i, m in enumerate(recs["aaaaaaab_s001_t000"]["channel_mask"]) if not m] == ears
    assert recs["aaaaaaab_s001_t000"]["montage"] == "04_tcp_le_a"
    assert recs["aaaaaaab_s001_t000"]["sfreq_native"] == 256
    assert manifest["format_version"] == 2 and manifest["dtype"] == "float16"


def test_signal_matches_filtered_source(corpus):
    """The 8 Hz sine of channel 0 survives filtering at its amplitude (µV)."""
    _convert(corpus, "--dtype", "float32")
    a = np.load(corpus / "npy/recordings/aaaaaaaa_s001_t000.npy")
    mid = a[4000:14000, 0]
    assert 25 < np.sqrt(2) * mid.std() < 35


def test_skip_stems_and_rerun(corpus):
    pd.DataFrame({"stem": ["aaaaaaaa_s001_t000"]}).to_parquet(corpus / "skip.parquet")
    manifest = _convert(corpus, "--skip-stems", str(corpus / "skip.parquet"))
    assert [r["stem"] for r in manifest["recordings"]] == ["aaaaaaab_s001_t000"]
    again = _convert(corpus, "--skip-stems", str(corpus / "skip.parquet"))
    assert again["recordings"][0]["sha256"] == manifest["recordings"][0]["sha256"]


def test_merge_reuses_split_and_copies_sidecars(corpus):
    _convert(corpus)
    prev = {"recordings": [{"stem": "aaaaaaaa_s001_t000", "source_split": "val"},
                           {"stem": "aaaaaaab_s001_t000", "source_split": "test"}]}
    (corpus / "prev.json").write_text(json.dumps(prev))
    side = corpus / "side"
    side.mkdir()
    for name in mk.SIDECARS:
        (side / name).write_text("{}")
    mk.main(["merge", "--scan", str(corpus / "scan.parquet"), "--dst", str(corpus / "npy"),
             "--split-from", str(corpus / "prev.json"), "--sidecar-dir", str(side)])
    manifest = json.loads((corpus / "npy/manifest.json").read_text())
    assert {r["stem"]: r["source_split"] for r in manifest["recordings"]} == \
        {"aaaaaaaa_s001_t000": "val", "aaaaaaab_s001_t000": "test"}
    assert all((corpus / "npy" / name).exists() for name in mk.SIDECARS)


def test_merge_excludes_recordings_outside_the_reused_split(corpus):
    _convert(corpus)
    prev = {"recordings": [{"stem": "aaaaaaab_s001_t000", "source_split": "train"}]}
    (corpus / "prev1.json").write_text(json.dumps(prev))
    mk.main(["merge", "--scan", str(corpus / "scan.parquet"), "--dst", str(corpus / "npy"),
             "--split-from", str(corpus / "prev1.json")])
    manifest = json.loads((corpus / "npy/manifest.json").read_text())
    assert [r["stem"] for r in manifest["recordings"]] == ["aaaaaaab_s001_t000"]
    assert list(manifest["excluded"]) == ["aaaaaaaa_s001_t000"]


def test_notch_option_and_batch_array_shard(corpus, monkeypatch):
    manifest = _convert(corpus, "--notch", "60")
    assert manifest["notch_hz"] == "60" and all(r["notch"] == "60" for r in manifest["recordings"])
    assert manifest["line_removal"] == "notch:60"
    again = _convert(corpus, "--notch", "50")          # a different notch is not "already done"
    assert again["notch_hz"] == "50"
    assert mk.parse_notch("none") == () and mk.parse_notch("50,60") == (50.0, 60.0)
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "3")

    class A:
        shard, num_shards = None, 8
    assert mk.resolve_shard(A()) == (3, 8)


def test_interp_line_removal_and_recipe_tags(corpus):
    manifest = _convert(corpus, "--notch", "50,60", "--line-removal", "interp", "--dtype", "float32")
    assert manifest["line_removal"] == "interp:50,60"
    assert all(r["line"] == "interp:50,60" for r in manifest["recordings"])
    x = np.load(corpus / "npy/recordings/aaaaaaaa_s001_t000.npy")
    assert np.isfinite(x).all() and 25 < np.sqrt(2) * x[4000:14000, 0].std() < 35   # EEG kept
    again = _convert(corpus, "--notch", "50,60", "--line-removal", "notch")          # other recipe
    assert again["line_removal"] == "notch:50,60"
