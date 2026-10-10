#!/usr/bin/env python
"""Repack the TUAB window pickles into one float32 ``.npy`` per recording.

The ``make_TUAB.py`` output is one pickle per 10 s window (``<stem>_<k>.pkl``,
float64 ``[23, 2000]`` at 200 Hz). That is 409k small files and 150 GB, which
makes streaming slow and full copies large. This tool concatenates each
recording's windows, in order, into a single time-major array

    recordings/<stem>.npy    float32 [T, 23], T = n_windows * 2000

so any crop is one contiguous read (``np.load(path, mmap_mode="r")[s:s + L]``)
and window ``k`` is exactly ``[2000 k, 2000 k + 2000)``. Values are the pickles'
float64 samples cast to float32 -- the same cast the training loader applies,
so the model sees bit-identical input.

Self-contained (numpy + boto3 + stdlib) so it runs unchanged in a stock CPU
container. Sources and destinations may be local directories or ``s3://``.

    # convert one shard (a SageMaker processing host picks its shard itself)
    python make_TUAB_npy.py convert --src s3://.../edf/processed \\
        --dst s3://.../edf/processed_npy [--shard 0 --num-shards 4]
    # after every shard finished: manifest.json + recording-level split
    python make_TUAB_npy.py merge --dst s3://.../edf/processed_npy \\
        --sidecar-dir /data/.../edf/processed
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import pickle
import sys
import time

import numpy as np

SAMPLE_RATE = 200
WINDOW_SAMPLES = 2000
N_CHANNELS = 23
SPLITS = ("train", "val", "test")
FORMAT_VERSION = 1


# --------------------------------------------------------------------- storage
class Store:
    """Minimal local-or-S3 file access."""

    def __init__(self, root):
        self.s3 = root.startswith("s3://")
        if self.s3:
            import boto3
            from botocore.config import Config
            self.client = boto3.client("s3", config=Config(
                max_pool_connections=128, retries={"max_attempts": 10, "mode": "adaptive"}))
            self.bucket, _, self.prefix = root[5:].partition("/")
            self.prefix = self.prefix.rstrip("/")
        else:
            self.root = root

    def _key(self, rel):
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def list(self, rel_dir):
        """File names directly under ``rel_dir``."""
        if not self.s3:
            path = os.path.join(self.root, rel_dir)
            return sorted(os.listdir(path)) if os.path.isdir(path) else []
        names, prefix = [], self._key(rel_dir).rstrip("/") + "/"
        for page in self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=prefix):
            names += [o["Key"][len(prefix):] for o in page.get("Contents", [])
                      if "/" not in o["Key"][len(prefix):]]
        return sorted(names)

    def read(self, rel):
        if not self.s3:
            with open(os.path.join(self.root, rel), "rb") as fh:
                return fh.read()
        return self.client.get_object(Bucket=self.bucket, Key=self._key(rel))["Body"].read()

    def write(self, rel, data, metadata=None):
        if not self.s3:
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)          # never leave a half-written file
            if metadata:
                with open(path + ".meta.json", "w") as fh:
                    json.dump(metadata, fh)
            return
        self.client.put_object(Bucket=self.bucket, Key=self._key(rel), Body=data,
                               Metadata={k: str(v) for k, v in (metadata or {}).items()})

    def head(self, rel):
        """Metadata of an existing file, or None."""
        if not self.s3:
            path = os.path.join(self.root, rel)
            if not os.path.exists(path):
                return None
            meta_path = path + ".meta.json"
            return json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self._key(rel))["Metadata"]
        except self.client.exceptions.ClientError:
            return None


# ------------------------------------------------------------------ inventory
def inventory(src):
    """``{stem: (split_dir, n_windows)}``; fails on gaps or a stem in two dirs."""
    found = {}
    for split in SPLITS:
        indices = {}
        for name in src.list(split):
            if name.endswith(".pkl"):
                stem, k = name[:-4].rsplit("_", 1)
                indices.setdefault(stem, []).append(int(k))
        for stem, ks in indices.items():
            if stem in found:
                raise ValueError(f"{stem} has windows in {found[stem][0]} and {split}")
            if sorted(ks) != list(range(len(ks))):
                raise ValueError(f"{stem}: window indices are not contiguous from 0")
            found[stem] = (split, len(ks))
    return found


def assign_shards(recordings, num_shards):
    """Deterministic, size-balanced: largest recordings first, each to the
    shard with the fewest windows so far (ties -> lowest shard)."""
    load = [0] * num_shards
    shard_of = {}
    for stem, (_, n) in sorted(recordings.items(), key=lambda kv: (-kv[1][1], kv[0])):
        s = min(range(num_shards), key=lambda i: (load[i], i))
        shard_of[stem] = s
        load[s] += n
    return shard_of


# ------------------------------------------------------------------- convert
def build_recording(src, split, stem, n_windows, pool):
    """Concatenate a recording's windows into float32 ``[T, 23]``."""
    names = [f"{split}/{stem}_{k}.pkl" for k in range(n_windows)]
    windows = list(pool.map(lambda rel: pickle.loads(src.read(rel))["X"], names))
    for k, x in enumerate(windows):
        if x.shape != (N_CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"{names[k]}: shape {x.shape}")
    array = np.ascontiguousarray(np.concatenate(windows, axis=1).T.astype(np.float32))
    # Self-check: a window sliced back out equals that pickle's float32 cast.
    k = n_windows // 2
    if not np.array_equal(array[k * WINDOW_SAMPLES:(k + 1) * WINDOW_SAMPLES].T,
                          windows[k].astype(np.float32)):
        raise AssertionError(f"{stem}: round-trip mismatch at window {k}")
    return array


def npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def resolve_shard(args):
    """Explicit --shard, else the SageMaker host index, else 0 of 1."""
    if args.shard is not None:
        return args.shard, args.num_shards
    cfg = "/opt/ml/config/resourceconfig.json"
    if os.path.exists(cfg):
        rc = json.load(open(cfg))
        hosts = sorted(rc["hosts"])
        return hosts.index(rc["current_host"]), len(hosts)
    return 0, 1


def cmd_convert(args):
    shard, num_shards = resolve_shard(args)
    src, dst = Store(args.src), Store(args.dst)
    t0 = time.time()
    recordings = inventory(src)
    mine = sorted(s for s, sh in assign_shards(recordings, num_shards).items() if sh == shard)
    print(f"[shard {shard}/{num_shards}] {len(mine)} of {len(recordings)} recordings, "
          f"{sum(recordings[s][1] for s in mine)} windows (inventory {time.time() - t0:.0f}s)",
          flush=True)

    window_pool = cf.ThreadPoolExecutor(args.read_threads)

    def convert_one(stem):
        split, n = recordings[stem]
        rel = f"recordings/{stem}.npy"
        meta = dst.head(rel)
        if meta and int(meta.get("n_samples", -1)) == n * WINDOW_SAMPLES and "sha256" in meta:
            return {"stem": stem, "source_split": split, "n_windows": n,
                    "n_samples": n * WINDOW_SAMPLES, "sha256": meta["sha256"],
                    "bytes": int(meta.get("bytes", 0)), "skipped": True}
        data = npy_bytes(build_recording(src, split, stem, n, window_pool))
        sha = hashlib.sha256(data).hexdigest()
        row = {"stem": stem, "source_split": split, "n_windows": n,
               "n_samples": n * WINDOW_SAMPLES, "sha256": sha, "bytes": len(data)}
        dst.write(rel, data, metadata={k: row[k] for k in ("n_samples", "sha256", "bytes")})
        return row

    rows, done = [], 0
    with cf.ThreadPoolExecutor(args.recording_threads) as pool:
        for row in pool.map(convert_one, mine):
            rows.append(row)
            done += 1
            if done % 50 == 0 or done == len(mine):
                print(f"[shard {shard}] {done}/{len(mine)} recordings, "
                      f"{time.time() - t0:.0f}s", flush=True)
    dst.write(f"_parts/part-{shard:03d}-of-{num_shards:03d}.json", json.dumps(
        {"shard": shard, "num_shards": num_shards, "recordings": rows}, indent=1).encode())
    print(f"[shard {shard}] done: {len(rows)} recordings "
          f"({sum(r.get('skipped', False) for r in rows)} already present), "
          f"{sum(r['bytes'] for r in rows) / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------- merge
def cmd_merge(args):
    """Combine shard parts with the labelled age sidecar into manifest.json,
    and carry the subject-disjoint age split over unchanged."""
    dst = Store(args.dst)
    parts = [n for n in dst.list("_parts") if n.endswith(".json")]
    rows = {}
    for name in parts:
        part = json.loads(dst.read(f"_parts/{name}"))
        for r in part["recordings"]:
            r.pop("skipped", None)
            rows[r["stem"]] = r
    meta = json.load(open(os.path.join(args.sidecar_dir, "age_metadata.json")))["recordings"]
    missing = [s for s in rows if s not in meta]
    if missing:
        raise ValueError(f"{len(missing)} recordings have no metadata, e.g. {missing[:3]}")
    if args.expect is not None and len(rows) != args.expect:
        raise ValueError(f"expected {args.expect} recordings, found {len(rows)} in {len(parts)} parts")
    for stem, r in rows.items():
        m = meta[stem]
        r.update({k: m.get(k) for k in ("subject", "session", "token", "age", "raw_age", "sex",
                                        "label", "year")})
        r["duration_sec"] = r["n_samples"] / SAMPLE_RATE
    manifest = {
        "format_version": FORMAT_VERSION,
        "layout": "recordings/<stem>.npy, float32 [n_samples, n_channels], time-major",
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "channels": ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                     "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2"],
        "units": "uV",
        "provenance": ("dataset_maker/make_TUAB.py windows (0.1-75 Hz band-pass, 50 Hz notch, "
                       "resampled to 200 Hz), concatenated per recording in window order and "
                       "cast float64 -> float32 by dataset_maker/make_TUAB_npy.py"),
        "n_recordings": len(rows),
        "n_windows": sum(r["n_windows"] for r in rows.values()),
        "recordings": [rows[s] for s in sorted(rows)],
    }
    dst.write("manifest.json", json.dumps(manifest, indent=1).encode())
    for name in ("age_metadata.json", "age_split.json"):
        with open(os.path.join(args.sidecar_dir, name), "rb") as fh:
            dst.write(name, fh.read())
    print(f"manifest.json: {manifest['n_recordings']} recordings, {manifest['n_windows']} windows, "
          f"{sum(r['bytes'] for r in rows.values()) / 1e9:.2f} GB; copied age_metadata.json and "
          f"age_split.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True, help="processed/ dir with train/ val/ test/ pickles")
    c.add_argument("--dst", required=True, help="output dir for recordings/ and _parts/")
    c.add_argument("--shard", type=int, default=None)
    c.add_argument("--num-shards", type=int, default=1)
    c.add_argument("--recording-threads", type=int, default=8)
    c.add_argument("--read-threads", type=int, default=48)
    c.set_defaults(func=cmd_convert)
    m = sub.add_parser("merge")
    m.add_argument("--dst", required=True)
    m.add_argument("--sidecar-dir", required=True,
                   help="dir with the labelled age_metadata.json and age_split.json")
    m.add_argument("--expect", type=int, default=None, help="expected recording count")
    m.set_defaults(func=cmd_merge)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
