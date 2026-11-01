#!/usr/bin/env python
"""Copy the HEEDB onboarding set from the BDSP access point into our private bucket.

Server-side S3 -> S3 copies (``CopyObject``; nothing passes through this
machine, same region so no egress), driven by the list written by
``python -m dataset_maker.heedb_index download-list`` (every object of the adult
routine sessions: EDF + BIDS sidecars + annotation CSVs). Keys keep their BIDS
path under ``HEEDB/v4.1/raw/`` (``EEG/bids/S0001/...`` -> ``HEEDB/v4.1/raw/bids/S0001/...``).

Resumable: ``run`` first lists what already exists in the destination and skips
objects whose size matches, so a rerun after an interruption only copies what
is missing. ``verify`` compares the destination listing with the list (count and
bytes per site). ``sync-local`` uploads the local metadata / scan / index files.

The destination bucket is created by ``infra/heedb_storage`` (private, KMS,
TLS-only). DUA: never point this at ``s3://eeg-data-public``.

    python -m dataset_maker.download_heedb run --dry-run
    python -m dataset_maker.download_heedb run --workers 128
    python -m dataset_maker.download_heedb verify
    python -m dataset_maker.download_heedb sync-local
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from typing import Dict, Iterable, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_maker import heedb_index as hx  # noqa: E402

DEST_BUCKET = os.environ.get("HEEDB_BUCKET", "heedb-private-574441342949")
DEST_ROOT = "HEEDB/v4.1"
PUBLIC_BUCKETS = ("eeg-data-public",)
SINGLE_COPY_LIMIT = 5 * 1024 ** 3          # CopyObject handles up to 5 GiB


def dest_key(src_key: str) -> str:
    """``EEG/bids/<site>/...`` -> ``HEEDB/v4.1/raw/bids/<site>/...``."""
    prefix = f"{hx.EEG_PREFIX}/"
    if not src_key.startswith(prefix):
        raise ValueError(f"not an HEEDB key: {src_key}")
    return f"{DEST_ROOT}/raw/{src_key[len(prefix):]}"


def _check_bucket(bucket: str) -> None:
    if bucket in PUBLIC_BUCKETS:
        raise SystemExit(f"refusing to copy HEEDB data into public bucket {bucket} (DUA)")


def list_dest(client, bucket: str, prefix: str) -> Dict[str, int]:
    """``{key: size}`` of everything under ``prefix`` in the destination."""
    out: Dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    return out


def todo(objects: pd.DataFrame, existing: Dict[str, int]) -> pd.DataFrame:
    """Rows of ``objects`` (``key``, ``bytes``) not yet in the destination with the same size."""
    dst = objects["key"].map(dest_key)
    done = [existing.get(d) == b for d, b in zip(dst, objects["bytes"])]
    return objects[[not x for x in done]].reset_index(drop=True)


def copy_one(client, bucket: str, key: str, size: int) -> int:
    """Server-side copy of one object; multipart (managed) above 5 GiB."""
    source = {"Bucket": hx.ACCESS_POINT, "Key": key}
    if size < SINGLE_COPY_LIMIT:
        client.copy_object(Bucket=bucket, Key=dest_key(key), CopySource=source)
    else:
        client.copy(source, bucket, dest_key(key))
    return size


def run_copy(client, objects: pd.DataFrame, bucket: str, *, workers: int = 128,
             log_every: int = 2000, state_path: Optional[str] = None) -> dict:
    """Copy ``objects`` in parallel; failures are retried once at the end and
    reported (with their error) in the returned summary / ``state_path``."""
    start, done_bytes, failed = time.time(), 0, {}

    def job(row):
        try:
            return row.key, copy_one(client, bucket, row.key, int(row.bytes)), None
        except Exception as exc:            # keep going; retried below
            return row.key, 0, f"{type(exc).__name__}: {exc}"

    rows = list(objects.itertuples(index=False))
    for attempt in (1, 2):
        failed_now = {}
        with cf.ThreadPoolExecutor(workers) as pool:
            for i, (key, n, err) in enumerate(pool.map(job, rows), 1):
                done_bytes += n
                if err:
                    failed_now[key] = err
                if i % log_every == 0:
                    rate = done_bytes / max(time.time() - start, 1e-9) / 1e6
                    print(f"  attempt {attempt}: {i}/{len(rows)} objects, "
                          f"{done_bytes / 1e12:.3f} TB, {rate:.0f} MB/s, {len(failed_now)} failed", flush=True)
        failed = failed_now
        rows = [r for r in rows if r.key in failed]
        if not rows:
            break
    summary = {"copied_bytes": done_bytes, "seconds": round(time.time() - start, 1),
               "failed": len(failed), "failures": dict(list(failed.items())[:50])}
    if state_path:
        with open(state_path, "w") as fh:
            json.dump(summary, fh, indent=1)
    return summary


def verify(objects: pd.DataFrame, existing: Dict[str, int]) -> pd.DataFrame:
    """Per site: objects / bytes expected vs present with the right size."""
    df = objects.assign(dst=objects["key"].map(dest_key))
    df["ok"] = [existing.get(d) == b for d, b in zip(df["dst"], df["bytes"])]
    df["site"] = df["key"].str.split("/").str[2]
    t = df.groupby("site").agg(expected=("key", "size"), present=("ok", "sum"),
                               expected_tb=("bytes", lambda b: b.sum() / 1e12))
    t["missing"] = t["expected"] - t["present"]
    t.loc["total"] = t.sum()
    return t


def sync_local(client, root: str, bucket: str, dirs: Iterable[str] = ("metadata", "scan", "index")) -> int:
    """Upload ``<root>/{metadata,scan,index}`` to ``HEEDB/v4.1/...`` (skip same size)."""
    existing = list_dest(client, bucket, f"{DEST_ROOT}/")
    n = 0
    for sub in dirs:
        for base, _, files in os.walk(os.path.join(root, sub)):
            for name in files:
                path = os.path.join(base, name)
                key = f"{DEST_ROOT}/{os.path.relpath(path, root)}"
                if existing.get(key) != os.path.getsize(path):
                    client.upload_file(path, bucket, key)
                    n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=hx.DEFAULT_ROOT)
    p.add_argument("--bucket", default=DEST_BUCKET)
    p.add_argument("--profile", default=hx.DEFAULT_PROFILE)
    p.add_argument("--list", default=None, help="download list (default <root>/scan/download_list.parquet)")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=128)
    r.add_argument("--limit", type=int, default=None, help="copy only the first N missing objects")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify")
    sub.add_parser("sync-local")
    args = p.parse_args(argv)
    _check_bucket(args.bucket)
    client = hx.s3_client(args.profile, pool=max(getattr(args, "workers", 64), 64))
    objects = pd.read_parquet(args.list or os.path.join(args.root, "scan", "download_list.parquet"))[["key", "bytes"]]

    if args.command == "sync-local":
        print(f"uploaded {sync_local(client, args.root, args.bucket)} files")
        return
    existing = list_dest(client, args.bucket, f"{DEST_ROOT}/raw/")
    if args.command == "verify":
        print(verify(objects, existing).to_string(float_format=lambda v: f"{v:.3f}"))
        return
    missing = todo(objects, existing)
    if args.limit:
        missing = missing.head(args.limit)
    print(f"{len(objects):,} objects / {objects['bytes'].sum() / 1e12:.3f} TB in the list; "
          f"{len(existing):,} already in s3://{args.bucket}/{DEST_ROOT}/raw/; "
          f"{len(missing):,} to copy ({missing['bytes'].sum() / 1e12:.3f} TB)", flush=True)
    if args.dry_run or missing.empty:
        return
    state = os.path.join(args.root, "logs", "copy_state.json")
    os.makedirs(os.path.dirname(state), exist_ok=True)
    print(json.dumps(run_copy(client, missing, args.bucket, workers=args.workers, state_path=state)))


if __name__ == "__main__":
    sys.exit(main())
