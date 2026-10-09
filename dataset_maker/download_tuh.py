#!/usr/bin/env python
"""Download TUH EEG corpora from NEDC to S3, one chunk at a time.

TUEG v2.0.2 is 1,643 GB of EDF -- far more than the local disk -- so each
chunk (one ``edf/<NNN>/`` group of ~100 patients, ~11 GB) is

    rsync from NEDC -> header-scan the EDFs -> s5cmd to S3 -> verify -> delete

and recorded in ``<state_dir>/<corpus>/download_state.json``, so a rerun skips
finished chunks and resumes an interrupted one (rsync and ``--if-size-differ``
both skip what is already there). The header scan (age, sex, sfreq, channels,
duration; ``labram.data.tuh_metadata``) runs on the local copy so the 1.6 TB
never has to be read back from S3; one JSON per chunk lands in the state dir and
next to the EDFs on S3.

NEDC accepts only SSH-key logins (``nedc-tuh-eeg@www.isip.piconepress.com``);
the key must be registered with help@nedcdata.org. Pass it with ``--ssh-key`` or
``NEDC_SSH_KEY`` (default ``~/.ssh/nedc_tuh`` if present, else ``~/.ssh/id_ed25519``).

    python -m dataset_maker.download_tuh plan   --corpus tueg   # sizes per chunk
    python -m dataset_maker.download_tuh run    --corpus tueg   # all pending chunks
    python -m dataset_maker.download_tuh run    --corpus tueg --chunks 000 001
    python -m dataset_maker.download_tuh run    --corpus tuep --keep-local
    python -m dataset_maker.download_tuh status --corpus tueg
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional

NEDC_HOST = "nedc-tuh-eeg@www.isip.piconepress.com"
BUCKET = "eeg-data-public"
DEFAULT_LOCAL = "/data/datasets/EEG-public/TUH_EEG"
DEFAULT_STATE = "/data/datasets/EEG-public/TUH_EEG/_state"
WHOLE = "_all"          # chunk name of a corpus downloaded in one piece
META = "_meta"          # README / DOCS outside the chunked data dir


@dataclass(frozen=True)
class Corpus:
    remote: str                 # rsync path on the NEDC server
    s3_prefix: str              # key prefix under s3://BUCKET/
    chunk_dir: Optional[str]    # subdir split into one chunk per entry; None = one chunk

    @property
    def version(self):
        return self.remote.rstrip("/").rsplit("/", 1)[-1]


# TUAB is deliberately absent: the local + S3 v3.0.0 copy is kept as is.
CORPORA: Dict[str, Corpus] = {
    "tueg": Corpus("data/tuh_eeg/tuh_eeg/v2.0.2", "TUH_EEG/tueg/v2.0.2", "edf"),
    "tuep": Corpus("data/tuh_eeg/tuh_eeg_epilepsy/v3.1.0", "TUH_EEG/tuep/v3.1.0", None),
}

# `rsync --list-only` line: perms, size (may contain commas), date, time, name
_LIST_RE = re.compile(r"^([dl-][rwxsStT-]{9})\s+([\d,]+)\s+\S+\s+\S+\s+(.+)$")


# --------------------------------------------------------------------- parsing
def parse_list_only(text):
    """``rsync --list-only`` output -> ``[(is_dir, size_bytes, path)]``."""
    entries = []
    for line in text.splitlines():
        m = _LIST_RE.match(line.strip())
        if m:
            entries.append((m.group(1)[0] == "d", int(m.group(2).replace(",", "")), m.group(3)))
    return entries


def chunk_sizes(entries, chunk_dir):
    """Recursive listing of the corpus root -> ``{chunk: {n_files, bytes}}``.

    Files under ``<chunk_dir>/<name>/`` belong to chunk ``name``; everything
    else (README, DOCS) to ``_meta``. With no ``chunk_dir`` the data is one
    chunk ``_all``.
    """
    out: Dict[str, Dict[str, int]] = {}
    for is_dir, size, path in entries:
        if is_dir or path in (".", ""):
            continue
        parts = path.split("/")
        if chunk_dir is None:
            name = WHOLE
        elif parts[0] == chunk_dir and len(parts) > 2:
            name = parts[1]
        else:
            name = META
        c = out.setdefault(name, {"n_files": 0, "bytes": 0})
        c["n_files"] += 1
        c["bytes"] += size
    return dict(sorted(out.items()))


def chunk_rel(corpus, chunk):
    """Path of a chunk relative to the corpus root ('' = whole corpus)."""
    if chunk in (WHOLE, META):
        return ""
    return f"{corpus.chunk_dir}/{chunk}"


# ------------------------------------------------------------------- commands
def ssh_cmd(key):
    return f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"


def rsync_cmd(corpus, chunk, dest, key, dry_run=False):
    """rsync command pulling one chunk into ``dest`` (mirrors the remote tree)."""
    rel = chunk_rel(corpus, chunk)
    src = f"{NEDC_HOST}:{corpus.remote}/{rel + '/' if rel else ''}"
    cmd = ["rsync", "-auxL", "--partial", "--stats", "-e", ssh_cmd(key)]
    if dry_run:
        cmd.append("-n")
    if chunk == META and corpus.chunk_dir:
        cmd += ["--exclude", f"/{corpus.chunk_dir}/"]
    return cmd + [src, os.path.join(dest, rel) + "/" if rel else dest.rstrip("/") + "/"]


def default_key():
    for cand in (os.environ.get("NEDC_SSH_KEY"), "~/.ssh/nedc_tuh", "~/.ssh/id_ed25519"):
        if cand and os.path.exists(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    raise SystemExit("no SSH key found: pass --ssh-key or set NEDC_SSH_KEY")


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------------- state
class State:
    """Per-corpus JSON of chunk -> {status, n_files, bytes, ...}; atomic writes."""

    def __init__(self, state_dir, name):
        self.path = os.path.join(state_dir, name, "download_state.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {"chunks": {}}

    def chunks(self):
        return self.data["chunks"]

    def update(self, chunk, **fields):
        self.chunks().setdefault(chunk, {}).update(fields)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ verifying
def local_inventory(path):
    """``{relpath: size}`` of every file under ``path``."""
    out = {}
    for d, _, files in os.walk(path):
        for f in files:
            full = os.path.join(d, f)
            out[os.path.relpath(full, path)] = os.path.getsize(full)
    return out


def s3_inventory(bucket, prefix):
    import boto3
    client = boto3.client("s3")
    out = {}
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            out[o["Key"][len(prefix.rstrip("/")) + 1:]] = (o["Size"], o["ETag"].strip('"'))
    return out


def verify_upload(local_dir, bucket, prefix, sample=10, seed=0):
    """Every local file is on S3 with the same size; a sample also by MD5/ETag."""
    import hashlib
    local = local_inventory(local_dir)
    remote = s3_inventory(bucket, prefix)
    missing = [k for k, size in local.items() if k not in remote or remote[k][0] != size]
    if missing:
        raise RuntimeError(f"{len(missing)} files missing or size-mismatched on S3, "
                           f"e.g. {missing[:3]}")
    checked = 0
    for rel in random.Random(seed).sample(sorted(local), min(sample, len(local))):
        etag = remote[rel][1]
        if "-" in etag:      # multipart upload: the ETag is not an MD5
            continue
        md5 = hashlib.md5(open(os.path.join(local_dir, rel), "rb").read()).hexdigest()
        if md5 != etag:
            raise RuntimeError(f"checksum mismatch for {rel}: {md5} != {etag}")
        checked += 1
    return {"n_files": len(local), "bytes": sum(local.values()), "md5_checked": checked}


# ---------------------------------------------------------------------- steps
def scan_chunk(local_root, rel, out_path, corpus_name):
    """Header-scan every EDF of a chunk into a JSON list of recording rows.

    Paths in the rows are relative to the corpus root (``edf/000/...``), the
    same layout as on S3, so the scan doubles as the conversion work list."""
    from labram.data.tuh_metadata import scan_recordings
    pattern = f"{rel}/**/*.edf" if rel else "**/*.edf"
    rows = scan_recordings(local_root, corpus=corpus_name, pattern=pattern)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh)
    os.replace(tmp, out_path)
    return len(rows)


def process_chunk(name, corpus, chunk, args, state):
    rel = chunk_rel(corpus, chunk)
    local_root = os.path.join(args.local_dir, name, corpus.version)
    local_dir = os.path.join(local_root, rel) if rel else local_root
    s3_prefix = f"{corpus.s3_prefix}/{rel}" if rel else corpus.s3_prefix

    state.update(chunk, status="downloading", started=_now())
    run(rsync_cmd(corpus, chunk, local_root, args.ssh_key))

    if chunk != META and not args.no_scan:
        scan_path = os.path.join(args.state_dir, name, "scans", f"{chunk}.json")
        os.makedirs(os.path.dirname(scan_path), exist_ok=True)
        n = scan_chunk(local_root, rel, scan_path, name)
        state.update(chunk, n_scanned=n)

    state.update(chunk, status="uploading")
    if chunk == META and corpus.chunk_dir:
        # Top level only; the data dir is uploaded chunk by chunk.
        for entry in os.listdir(local_root):
            if entry == corpus.chunk_dir:
                continue
            src = os.path.join(local_root, entry)
            dst = f"s3://{BUCKET}/{corpus.s3_prefix}/{entry}"
            run(["s5cmd", "--numworkers", str(args.workers), "--log", "error", "cp",
                 "--if-size-differ"] + ([src + "/*", dst + "/"] if os.path.isdir(src) else [src, dst]))
        state.update(chunk, status="done", finished=_now())
        return
    run(["s5cmd", "--numworkers", str(args.workers), "--log", "error", "cp",
         "--if-size-differ", local_dir.rstrip("/") + "/*", f"s3://{BUCKET}/{s3_prefix}/"])

    state.update(chunk, status="verifying")
    stats = verify_upload(local_dir, BUCKET, s3_prefix, sample=args.verify_sample)
    state.update(chunk, **stats)

    if not args.keep_local:
        shutil.rmtree(local_dir)
    state.update(chunk, status="done", finished=_now(), local=args.keep_local)
    print(f"[{name}/{chunk}] done: {stats['n_files']} files, {stats['bytes'] / 1e9:.1f} GB",
          flush=True)


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ commands
def cmd_plan(args):
    corpus = CORPORA[args.corpus]
    out = subprocess.run(["rsync", "-r", "--list-only", "-e", ssh_cmd(args.ssh_key),
                          f"{NEDC_HOST}:{corpus.remote}/"],
                         check=True, capture_output=True, text=True).stdout
    sizes = chunk_sizes(parse_list_only(out), corpus.chunk_dir)
    plan = {"corpus": args.corpus, "remote": corpus.remote, "created": _now(),
            "total_files": sum(c["n_files"] for c in sizes.values()),
            "total_bytes": sum(c["bytes"] for c in sizes.values()), "chunks": sizes}
    path = os.path.join(args.state_dir, args.corpus, "download_plan.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(plan, fh, indent=1)
    big = max(sizes.values(), key=lambda c: c["bytes"])["bytes"]
    print(f"{args.corpus}: {plan['total_files']} files, {plan['total_bytes'] / 1e9:.1f} GB "
          f"in {len(sizes)} chunks (largest {big / 1e9:.1f} GB) -> {path}")


def cmd_run(args):
    corpus = CORPORA[args.corpus]
    plan_path = os.path.join(args.state_dir, args.corpus, "download_plan.json")
    if not os.path.exists(plan_path):
        cmd_plan(args)
    planned = list(json.load(open(plan_path))["chunks"])
    chunks = args.chunks or planned
    unknown = sorted(set(chunks) - set(planned))
    if unknown:
        raise SystemExit(f"unknown chunks {unknown}; known: {planned[:5]}...")
    state = State(args.state_dir, args.corpus)
    free = shutil.disk_usage(args.local_dir if os.path.exists(args.local_dir) else "/").free
    for chunk in chunks:
        if state.chunks().get(chunk, {}).get("status") == "done":
            continue
        need = json.load(open(plan_path))["chunks"][chunk]["bytes"]
        if need * 1.1 > free and not args.keep_local:
            raise SystemExit(f"chunk {chunk} needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
        try:
            process_chunk(args.corpus, corpus, chunk, args, state)
        except Exception as exc:   # record and stop; a rerun resumes this chunk
            state.update(chunk, status="failed", error=str(exc)[:500])
            raise


def cmd_status(args):
    plan_path = os.path.join(args.state_dir, args.corpus, "download_plan.json")
    planned = json.load(open(plan_path))["chunks"] if os.path.exists(plan_path) else {}
    chunks = State(args.state_dir, args.corpus).chunks()
    by = {}
    for c in planned or chunks:
        by.setdefault(chunks.get(c, {}).get("status", "pending"), []).append(c)
    done_b = sum(chunks[c].get("bytes", 0) for c in by.get("done", []))
    total_b = sum(c["bytes"] for c in planned.values()) or float("nan")
    print(f"{args.corpus}: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by.items()))
          + f"; {done_b / 1e9:.1f}/{total_b / 1e9:.1f} GB on S3")
    for c in by.get("failed", []):
        print(f"  failed {c}: {chunks[c].get('error')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "run", "status"])
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    p.add_argument("--chunks", nargs="*", help="chunk names (default: all pending)")
    p.add_argument("--ssh-key", default=None)
    p.add_argument("--local-dir", default=DEFAULT_LOCAL)
    p.add_argument("--state-dir", default=DEFAULT_STATE)
    p.add_argument("--keep-local", action="store_true", help="do not delete the EDFs after upload")
    p.add_argument("--no-scan", action="store_true", help="skip the EDF header scan")
    p.add_argument("--workers", type=int, default=64, help="s5cmd parallel requests")
    p.add_argument("--verify-sample", type=int, default=10, help="files MD5-checked per chunk")
    args = p.parse_args(argv)
    if args.command != "status":
        args.ssh_key = args.ssh_key or default_key()
    {"plan": cmd_plan, "run": cmd_run, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
