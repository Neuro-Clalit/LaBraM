#!/usr/bin/env python
"""Per-recording meta features for a TUH cohort (TUEG + TUAB) -> one parquet.

The TUEG counterpart of ``make_TUAB_meta.py``. TUEG is kept on disk only as the
processed per-recording npy (its 1.75 TB of EDFs live in S3), so the table is
assembled from four read-only sources instead of one EDF sweep:

* the cohort itself (``cohort.parquet``, ``manifest.json``, ``age_metadata.json``):
  split, age, sex, training label, channel mask, window count;
* the recording index (``dataset_maker.tuh_index``): montage, native sampling
  rate, year, session, corpora, TUAB/TUEP labels;
* NEDC's clinical sheet (``dataset_maker.tuh_clinical``): diagnoses, drug
  classes, report findings -- what TUAB lacks;
* TUEG's header dump (``DOCS/headers.tar.gz``): amplifier family, amplifier
  range and the auxiliary-channel inventory (EKG, EMG, photic, sphenoidal,
  burst-suppression trends) without reading an EDF;
* the cohort npy: signal quality + qEEG (``tuh_meta_features.npy_signal_features``).

    python -m dataset_maker.make_cohort_meta \\
        --cohort /data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3 --n_jobs 8
    # -> <cohort>/meta_features.parquet, read by notebooks/tuab_meta_features.ipynb

Signal features are computed in parts of ``--part_size`` recordings under
``<cohort>/_meta_parts/`` and reused on a rerun, so an interrupted sweep resumes.
See docs/meta_features.md.
"""
import argparse
import glob
import json
import logging
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dataset_maker import tuh_clinical as tc
from dataset_maker.make_cohort_dataset import DEFAULT_INDEX, DEFAULT_SHEET
from labram.data.tuh_meta_features import npy_signal_features, parse_header_dump

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = "/data/datasets/EEG-public/TUH_EEG/tueg/v2.0.2/DOCS/headers.tar.gz"
META_FILENAME = "meta_features.parquet"
INDEX_COLUMNS = ["stem", "session", "token", "year", "montage", "sfreq", "mixed_sfreq",
                 "n_signals", "missing_channels", "abnormal", "epilepsy"]


# ------------------------------------------------------------------ sources
def load_cohort(cohort_dir):
    """One row per cohort recording: split, demographics, label, npy layout."""
    cohort = pd.read_parquet(os.path.join(cohort_dir, "cohort.parquet"))
    manifest = json.load(open(os.path.join(cohort_dir, "manifest.json")))
    recs = pd.DataFrame(manifest["recordings"])[
        ["stem", "n_windows", "channel_mask", "sfreq_native", "duration_sec"]]
    meta = json.load(open(os.path.join(cohort_dir, "age_metadata.json")))["recordings"]
    labels = pd.Series({k: v.get("label") for k, v in meta.items()}, name="label")
    out = cohort.merge(recs, on="stem", how="left").merge(
        labels.rename_axis("stem").reset_index(), on="stem", how="left")
    return out, manifest


def load_header_dump(path):
    """Header features of every EDF in an NEDC dump (``.tar.gz`` or plain text)."""
    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as tar:
            member = next(m for m in tar.getmembers() if m.isfile())
            lines = (raw.decode("latin-1") for raw in tar.extractfile(member))
            rows = list(parse_header_dump(lines))
    else:
        with open(path, encoding="latin-1") as fh:
            rows = list(parse_header_dump(fh))
    # A stem can appear twice when NEDC re-exported a file; keep the first.
    return pd.DataFrame(rows).drop_duplicates("stem").drop(columns=["path"])


def build_table(cohort, index, sheet, headers):
    """Join every non-signal source onto the cohort (left joins: never drops a row)."""
    df = cohort.merge(index[INDEX_COLUMNS], on="stem", how="left")
    df = tc.join_index(df, sheet)
    hdr = headers.rename(columns={"sfreq": "sfreq_header", "n_signals": "n_signals_header",
                                  "duration_min": "duration_min_header"})
    df = df.merge(hdr, on="stem", how="left")
    df["in_header_dump"] = df["equipment_family"].notna()
    df["session_number"] = pd.to_numeric(df["session"].str[1:], errors="coerce")
    df["token_number"] = pd.to_numeric(df["token"].str[1:], errors="coerce")
    df["duration_min"] = df["duration_sec"] / 60.0
    df["n_missing_channels"] = df["missing_channels"].map(
        lambda m: len(m) if m is not None else np.nan)
    df["missing_channels"] = df["missing_channels"].map(
        lambda m: ",".join(m) if m is not None else None)
    df["in_tuab"] = df["corpora"].str.contains("tuab")
    df["in_seizure_corpus"] = df["corpora"].str.contains("tusz|tuep")
    return df.drop(columns=["channel_mask"])


# ------------------------------------------------------------- signal sweep
def _signal_row(task):
    stem, path, channels, mask, sfreq, max_minutes = task
    try:
        x = np.load(path, mmap_mode="r")
        row = npy_signal_features(x, channels, mask, sfreq=sfreq, max_minutes=max_minutes)
        row["signal_error"] = ""
    except Exception as exc:  # noqa: BLE001 -- one bad file must not sink a cohort sweep
        row = {"signal_error": f"{type(exc).__name__}: {exc}"}
    row["stem"] = stem
    return row


def signal_features(cohort_dir, manifest, stems, *, n_jobs, max_minutes, part_size):
    """npy quality + qEEG features, computed in resumable parts."""
    parts_dir = os.path.join(cohort_dir, "_meta_parts")
    os.makedirs(parts_dir, exist_ok=True)
    masks = {r["stem"]: r["channel_mask"] for r in manifest["recordings"]}
    channels, sfreq = manifest["channels"], float(manifest["sample_rate"])
    stems = sorted(stems)
    chunks = [stems[i: i + part_size] for i in range(0, len(stems), part_size)]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for k, chunk in enumerate(chunks):
            part = os.path.join(parts_dir, f"part_{k:04d}.parquet")
            if os.path.exists(part) and set(pd.read_parquet(part, columns=["stem"])["stem"]) == set(chunk):
                continue
            tasks = [(s, os.path.join(cohort_dir, "recordings", f"{s}.npy"), channels,
                      masks.get(s), sfreq, max_minutes) for s in chunk]
            t0 = time.time()
            rows = list(pool.map(_signal_row, tasks, chunksize=8))
            pd.DataFrame(rows).to_parquet(part, index=False)
            logger.info("signal part %d/%d (%d recordings) in %.0fs",
                        k + 1, len(chunks), len(chunk), time.time() - t0)
    parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet")))
    out = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return out[out["stem"].isin(stems)].drop_duplicates("stem")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", required=True, help="cohort directory (make_cohort_dataset output)")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--headers", default=DEFAULT_HEADERS, help="NEDC header dump (headers.tar.gz)")
    p.add_argument("--out", default=None, help=f"default <cohort>/{META_FILENAME}")
    p.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--max_minutes", type=float, default=60.0,
                   help="signal features over at most the first N minutes of a recording")
    p.add_argument("--part_size", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="only the first N recordings (smoke test)")
    p.add_argument("--no_signal", action="store_true", help="skip the npy sweep (header/clinical only)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = time.time()
    cohort, manifest = load_cohort(args.cohort)
    if args.limit:
        cohort = cohort.head(args.limit)
    table = build_table(cohort, pd.read_parquet(args.index), tc.load_sheet(args.sheet),
                        load_header_dump(args.headers))
    if not args.no_signal:
        sig = signal_features(args.cohort, manifest, table["stem"], n_jobs=args.n_jobs,
                              max_minutes=args.max_minutes, part_size=args.part_size)
        table = table.merge(sig, on="stem", how="left")
    out = args.out or os.path.join(args.cohort, META_FILENAME)
    table.to_parquet(out, index=False)
    failed = int((table.get("signal_error", pd.Series(dtype=str)).fillna("") != "").sum())
    print(f"Wrote {len(table)} recordings x {table.shape[1]} columns to {out} in "
          f"{time.time() - start:.0f}s ({failed} signal failures, "
          f"{int((~table['in_header_dump']).sum())} not in the header dump, "
          f"{int((~table['in_sheet']).sum())} not in the clinical sheet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
