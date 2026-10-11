#!/usr/bin/env python
"""Convert TUH EDF recordings straight to one ``.npy`` per recording.

Same signal pipeline as ``make_TUAB.py`` (the published LaBraM/BIOT TUAB
preprocessing) -- the 23 TUH channels in model order, 0.1-75 Hz band-pass,
50 Hz notch, resampled to 200 Hz, microvolts, whole 10 s windows only -- but
without the 10 s pickle stage, and tolerant of what TUEG adds:

* ``-LE`` (linked-ear) as well as ``-REF`` channel names;
* recordings that lack some of the 23 channels (``*_a`` montages have no
  A1/A2): the missing rows are zero and ``channel_mask`` in the manifest marks
  them, so training can exclude or mask them explicitly;
* any extra channels, which are never read.

Output, readable by the TUAB npy loader::

    recordings/<stem>.npy   float16|float32 [T, 23], T = n_windows * 2000
    manifest.json           format_version 2, per-recording metadata + labels

The work list is a corpus scan (``dataset_maker.tuh_index scan`` / the
``download_tuh`` chunk scans), whose ``path`` column is relative to
``--edf-root``. EDFs may be local or on S3 (downloaded to a temp file).

    python -m dataset_maker.make_TUH_npy convert --scan tueg.parquet \\
        --edf-root s3://eeg-data-public/TUH_EEG/tueg/v2.0.2 \\
        --dst s3://eeg-data-public/TUH_EEG/tueg/v2.0.2/processed_npy \\
        --skip-stems tuab.parquet [--shard 0 --num-shards 8]
    python -m dataset_maker.make_TUH_npy merge --scan tueg.parquet --dst ...
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

try:
    from dataset_maker.make_TUAB_npy import Store, assign_shards, npy_bytes, resolve_shard
    from labram.data.tuh_metadata import TUH_CHANNELS, channel_map
except ImportError:      # flat source_dir inside a SageMaker processing container
    from make_TUAB_npy import Store, assign_shards, npy_bytes, resolve_shard
    from tuh_metadata import TUH_CHANNELS, channel_map

SAMPLE_RATE = 200
WINDOW_SAMPLES = 2000
FORMAT_VERSION = 2
L_FREQ, H_FREQ, NOTCH = 0.1, 75.0, 50.0
# The 75 Hz low-pass needs a native rate above 150 Hz; TUH's lowest is 250.
MIN_NATIVE_SFREQ = 2 * H_FREQ
MIN_DURATION_SEC = 60
# float16 keeps ~3 significant digits: rms rounding error is ~0.02% of the
# signal on TUAB (0.003-0.06 µV); only values beyond this are clipped.
FLOAT16_MAX = float(np.finfo(np.float16).max)
SCAN_FIELDS = ("corpus", "path", "subject", "session", "token", "age", "raw_age", "sex", "year",
               "label", "montage", "session_date", "abnormal", "epilepsy", "source_split")
SIDECARS = ("age_metadata.json", "age_split.json")


# ---------------------------------------------------------------- preprocess
def preprocess_recording(edf_path):
    """EDF -> (``[T, 23]`` float64 µV at 200 Hz, whole windows only; channel mask)."""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    present = channel_map(raw.ch_names)          # canonical -> raw label
    if not present:
        raise ValueError("none of the 23 TUH channels present")
    order = [c for c in TUH_CHANNELS if c in present]
    raw.pick([present[c] for c in order])
    raw.load_data(verbose="error")
    if raw.info["sfreq"] < MIN_NATIVE_SFREQ:
        raise ValueError(f"sampling rate {raw.info['sfreq']} Hz < {MIN_NATIVE_SFREQ}")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose="error")
    raw.notch_filter(NOTCH, verbose="error")
    raw.resample(SAMPLE_RATE, verbose="error")
    data = raw.get_data(units="uV")              # [len(order), T]
    n = data.shape[1] // WINDOW_SAMPLES * WINDOW_SAMPLES
    out = np.zeros((n, len(TUH_CHANNELS)), dtype=np.float64)
    mask = [c in present for c in TUH_CHANNELS]
    out[:, [TUH_CHANNELS.index(c) for c in order]] = data[:, :n].T
    return out, mask


# --------------------------------------------------------------------- work list
def read_scan(path):
    import pandas as pd
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.DataFrame(json.load(open(path)))


def work_list(scan, skip_stems=(), min_duration=MIN_DURATION_SEC):
    """Rows to convert: readable, long enough, not already covered elsewhere."""
    ok = scan["error"].isna() if "error" in scan else np.ones(len(scan), bool)
    keep = ok & (scan["duration_sec"] >= min_duration) & ~scan["stem"].isin(set(skip_stems))
    rows = scan[keep].drop_duplicates("stem")
    return {r["stem"]: r for r in rows.to_dict("records")}


# ---------------------------------------------------------------------- convert
def cmd_convert(args):
    shard, num_shards = resolve_shard(args)
    src, dst = Store(args.edf_root), Store(args.dst)
    skip = set()
    for p in args.skip_stems or []:
        skip |= set(read_scan(p)["stem"])
    todo = work_list(read_scan(args.scan), skip, args.min_duration)
    weights = {s: (None, float(r["duration_sec"])) for s, r in todo.items()}
    mine = sorted(s for s, sh in assign_shards(weights, num_shards).items() if sh == shard)
    if args.limit:
        mine = mine[:args.limit]
    dtype = np.dtype(args.dtype)
    t0 = time.time()
    print(f"[shard {shard}/{num_shards}] {len(mine)} of {len(todo)} recordings "
          f"({len(skip)} stems skipped as covered elsewhere), dtype {dtype}", flush=True)

    def convert_one(stem):
        row = todo[stem]
        rel = f"recordings/{stem}.npy"
        meta = dst.head(rel)
        if meta and "sha256" in meta and "channel_mask" in meta:
            return {"stem": stem, **{k: json.loads(v) if k == "channel_mask" else v
                                     for k, v in meta.items()}, "skipped": True}
        try:
            if src.s3:
                with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
                    tmp.write(src.read(row["path"]))
                    tmp.flush()
                    x, mask = preprocess_recording(tmp.name)
            else:
                x, mask = preprocess_recording(os.path.join(src.root, row["path"]))
        except Exception as exc:              # record, never abort the shard
            return {"stem": stem, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if x.shape[0] == 0:
            return {"stem": stem, "error": "shorter than one window"}
        if dtype == np.float16:            # beyond ±65504 µV float16 is inf
            x = np.clip(x, -FLOAT16_MAX, FLOAT16_MAX)
        data = npy_bytes(np.ascontiguousarray(x.astype(dtype)))
        out = {"stem": stem, "n_windows": x.shape[0] // WINDOW_SAMPLES, "n_samples": x.shape[0],
               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
               "channel_mask": mask, "dtype": dtype.name}
        dst.write(rel, data, metadata={**{k: out[k] for k in ("n_windows", "n_samples", "sha256",
                                                                "bytes", "dtype")},
                                       "channel_mask": json.dumps(mask)})
        return out

    rows, done = [], 0
    with cf.ThreadPoolExecutor(args.threads) as pool:
        for r in pool.map(convert_one, mine):
            rows.append(r)
            done += 1
            if done % 50 == 0 or done == len(mine):
                print(f"[shard {shard}] {done}/{len(mine)}, {time.time() - t0:.0f}s", flush=True)
    dst.write(f"_parts/part-{shard:03d}-of-{num_shards:03d}.json", json.dumps(
        {"shard": shard, "num_shards": num_shards, "recordings": rows}, indent=1).encode())
    errors = [r for r in rows if "error" in r]
    print(f"[shard {shard}] done: {len(rows) - len(errors)} converted "
          f"({sum(r.get('skipped', False) for r in rows)} already present), {len(errors)} failed, "
          f"{sum(int(r.get('bytes', 0)) for r in rows) / 1e9:.2f} GB in {time.time() - t0:.0f}s",
          flush=True)


# ------------------------------------------------------------------------ merge
def cmd_merge(args):
    """Shard parts + scan metadata -> manifest.json (failures in ``errors``)."""
    dst = Store(args.dst)
    scan = {r["stem"]: r for r in read_scan(args.scan).to_dict("records")}
    split_of = None
    if args.split_from:
        # Reuse another manifest's split assignment (e.g. TUAB's make_TUAB.py
        # train/val/test), so a re-release is a drop-in replacement.
        with open(args.split_from) as fh:
            split_of = {r["stem"]: r["source_split"] for r in json.load(fh)["recordings"]}
    rows, errors, excluded = {}, {}, {}
    for name in sorted(n for n in dst.list("_parts") if n.endswith(".json")):
        for r in json.loads(dst.read(f"_parts/{name}"))["recordings"]:
            r.pop("skipped", None)
            if "error" in r:
                errors[r["stem"]] = r["error"]
                continue
            s = scan[r["stem"]]
            r.update({k: _plain(s.get(k)) for k in SCAN_FIELDS})
            r["sfreq_native"] = _plain(s.get("sfreq"))
            if split_of is not None:
                if r["stem"] not in split_of:
                    # Not part of the reused protocol (e.g. TUAB recordings that
                    # make_TUAB.py rejected for missing channels): keep it out.
                    excluded[r["stem"]] = f"not in {os.path.basename(args.split_from)}"
                    continue
                r["source_split"] = split_of[r["stem"]]
            for k in ("n_windows", "n_samples", "bytes"):
                r[k] = int(r[k])
            r["duration_sec"] = r["n_samples"] / SAMPLE_RATE
            rows[r["stem"]] = r
    dtypes = sorted({r["dtype"] for r in rows.values()})
    manifest = {
        "format_version": FORMAT_VERSION,
        "layout": "recordings/<stem>.npy, [n_samples, n_channels], time-major",
        "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "channels": list(TUH_CHANNELS),
        "units": "uV",
        "provenance": (f"dataset_maker/make_TUH_npy.py: EDF -> {L_FREQ}-{H_FREQ} Hz band-pass, "
                       f"{NOTCH} Hz notch, resampled to {SAMPLE_RATE} Hz (make_TUAB.py pipeline); "
                       "missing channels zero-filled, see channel_mask"),
        "n_recordings": len(rows),
        "n_windows": sum(r["n_windows"] for r in rows.values()),
        "recordings": [rows[s] for s in sorted(rows)],
        "errors": errors,
        "excluded": excluded,
    }
    dst.write("manifest.json", json.dumps(manifest, indent=1).encode())
    for name in SIDECARS if args.sidecar_dir else ():
        with open(os.path.join(args.sidecar_dir, name), "rb") as fh:
            dst.write(name, fh.read())
    print(f"manifest.json: {len(rows)} recordings, {manifest['n_windows']} windows, "
          f"{sum(r['bytes'] for r in rows.values()) / 1e9:.2f} GB, {len(errors)} failed, "
          f"{len(excluded)} excluded")


def _plain(v):
    """pandas/numpy scalars (and NA) -> JSON-able Python values."""
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--scan", required=True, help="corpus scan (.parquet or .json rows)")
    c.add_argument("--edf-root", required=True, help="root the scan's paths are relative to")
    c.add_argument("--dst", required=True)
    c.add_argument("--skip-stems", nargs="*", help="scans whose stems are already converted")
    c.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    c.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC)
    c.add_argument("--shard", type=int, default=None)
    c.add_argument("--num-shards", type=int, default=1)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--limit", type=int, default=0, help="convert at most N (debug)")
    c.set_defaults(func=cmd_convert)
    m = sub.add_parser("merge")
    m.add_argument("--scan", required=True)
    m.add_argument("--dst", required=True)
    m.add_argument("--split-from", default=None,
                   help="manifest.json whose per-recording source_split to reuse")
    m.add_argument("--sidecar-dir", default=None,
                   help=f"dir whose {' and '.join(SIDECARS)} are copied next to the manifest")
    m.set_defaults(func=cmd_merge)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
