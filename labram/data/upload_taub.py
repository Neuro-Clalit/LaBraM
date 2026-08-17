#!/usr/bin/env python3
"""
upload_taub.py — resumable, checksum-verified upload:

    ~/Data/EEG-public/TAUB/   ->   dl_neuro_1:/data/datasets/EEG-public/TAUB/

Run it as many times as you like. Each run:
  1. hashes local files (SHA-256, cached by size+mtime so re-runs are fast),
  2. asks the remote for its SHA-256s in a single ssh call,
  3. uploads only files that are MISSING or whose checksum DIFFERS,
  4. re-reads the remote checksums and confirms every file matches.

If a transfer dies half-way, just run it again — rsync --partial resumes the
partial file and the checksum pass catches anything corrupt.

Requires (on the machine you run this from): python3, rsync, ssh, and tqdm
(`pip install tqdm` — optional; a plain fallback bar is used if it's missing).
The remote needs `sha256sum` and `find` (standard on any Linux host).
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path

# ---------------------------------------------------------------- defaults ----
LOCAL_DIR   = Path("/Users/leon/Data/EEG-public/TAUB")
REMOTE_HOST = "dl_neuro_1"
REMOTE_DIR  = "/data/datasets/EEG-public/TAUB"
CACHE_FILE  = Path.home() / ".cache" / "upload_taub_sha256.json"

# files/dirs never uploaded (repo cruft). Matched against basename, full
# relpath, and any path component. Add more with --exclude on the CLI.
DEFAULT_EXCLUDES = [
    ".git", "__pycache__", ".ipynb_checkpoints", ".DS_Store", ".idea",
    "*.pyc", "*.pyo", "*.tmp", "*.swp", "*.part",
    "upload_taub.py", "upload_taub_sha256.json",
]

# ---------------------------------------------------------------- tqdm shim ---
try:
    from tqdm import tqdm
except ImportError:                                    # minimal fallback
    class tqdm:
        def __init__(self, total=0, desc="", unit="", **k):
            self.total, self.n, self.desc = total or 0, 0, desc
        def update(self, n=1):
            self.n += n
            if self.total:
                pct = 100 * self.n / self.total
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:4.0f}%)",
                      end="", file=sys.stderr, flush=True)
        def close(self):
            print("", file=sys.stderr)
        def __enter__(self): return self
        def __exit__(self, *a): self.close()


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded(rel: str, patterns: list[str]) -> bool:
    """True if relpath matches an exclude (basename, full path, or any dir part)."""
    parts = rel.split("/")
    base = parts[-1]
    for pat in patterns:
        p = pat.rstrip("/")
        if fnmatch(base, p) or fnmatch(rel, p) or p in parts:
            return True
    return False


def local_hashes(local_dir: Path, cache: dict, excludes: list[str]) -> dict[str, str]:
    """relpath -> sha256 for every local file, using the mtime/size cache."""
    files = [p for p in local_dir.rglob("*")
             if p.is_file()
             and not is_excluded(p.relative_to(local_dir).as_posix(), excludes)]
    out: dict[str, str] = {}
    with tqdm(total=len(files), desc="hashing local", unit="file") as bar:
        for p in files:
            rel = p.relative_to(local_dir).as_posix()
            st = p.stat()
            c = cache.get(rel)
            if c and c["size"] == st.st_size and c["mtime"] == int(st.st_mtime):
                out[rel] = c["sha256"]
            else:
                sha = sha256_file(p)
                out[rel] = sha
                cache[rel] = {"size": st.st_size,
                              "mtime": int(st.st_mtime),
                              "sha256": sha}
            bar.update(1)
    return out


def remote_hashes(ssh_cmd: list[str], host: str, remote_dir: str) -> dict[str, str]:
    """relpath -> sha256 for every remote file (one ssh round-trip)."""
    rd = shlex.quote(remote_dir)
    script = (f"cd {rd} 2>/dev/null && "
              f"find . -type f -print0 | xargs -0 sha256sum 2>/dev/null || true")
    res = subprocess.run(ssh_cmd + [host, script],
                         capture_output=True, text=True)
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")          # "<sha>  ./sub/file"
        name = name[2:] if name.startswith("./") else name
        if name:
            out[name] = digest
    return out


def remote_disk_space(ssh_cmd: list[str], host: str, remote_dir: str) -> tuple[int, int]:
    """Return (available_bytes, total_bytes) for the remote directory's filesystem."""
    rd = shlex.quote(remote_dir)
    script = f"df -B1 {rd} 2>/dev/null | tail -1"
    res = subprocess.run(ssh_cmd + [host, script],
                         capture_output=True, text=True)
    if res.returncode != 0:
        return 0, 0
    parts = res.stdout.split()
    if len(parts) >= 4:
        try:
            total = int(parts[1])
            available = int(parts[3])
            return available, total
        except (ValueError, IndexError):
            pass
    return 0, 0


def format_bytes(b: int) -> str:
    """Format bytes as human-readable (B, KB, MB, GB, TB)."""
    for unit, factor in [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if b >= factor:
            return f"{b / factor:.1f} {unit}"
    return f"{b} B"


def upload_chunk(chunk: list[str], local_dir: Path, host: str, remote_dir: str,
                 listfile: str, ssh_cmd: list[str], dry_run: bool) -> tuple[int, str]:
    """Upload a chunk of files. Returns (returncode, chunk_info)."""
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as fh:
        fh.write("\n".join(chunk))
        chunk_listfile = fh.name
    try:
        rsync = [
            "rsync", "-a", "--partial", "--progress", "--human-readable",
            f"--files-from={chunk_listfile}",
            "-e", " ".join(shlex.quote(c) for c in ssh_cmd),
            f"{local_dir}/",
            f"{host}:{shlex.quote(remote_dir)}/",
        ]
        if dry_run:
            rsync.insert(1, "--dry-run")
        # Show output in real-time so user sees progress
        res = subprocess.run(rsync, text=True)
        return res.returncode, f"chunk: {len(chunk)} files"
    finally:
        os.unlink(chunk_listfile)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local",  default=str(LOCAL_DIR))
    ap.add_argument("--host",   default=REMOTE_HOST)
    ap.add_argument("--remote", default=REMOTE_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what WOULD upload, transfer nothing")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                    help="extra glob to skip (repeatable), on top of the defaults")
    ap.add_argument("--no-default-excludes", action="store_true",
                    help="upload everything, ignoring the built-in exclude list")
    args = ap.parse_args()

    excludes = list(args.exclude)
    if not args.no_default_excludes:
        excludes += DEFAULT_EXCLUDES

    local_dir = Path(args.local).expanduser().resolve()
    if not local_dir.is_dir():
        print(f"ERROR: local dir not found: {local_dir}", file=sys.stderr)
        return 2

    # persistent ssh connection so the many ssh/rsync calls share one login
    ctl = Path(tempfile.gettempdir()) / f"upload_taub_{os.getpid()}.sock"
    ssh_cmd = ["ssh",
               "-o", f"ControlPath={ctl}",
               "-o", "ControlMaster=auto",
               "-o", "ControlPersist=120"]

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    print(f"Local : {local_dir}")
    print(f"Remote: {args.host}:{args.remote}")
    print(f"Skip  : {', '.join(excludes) if excludes else '(nothing)'}\n")

    # 1) local checksums
    loc = local_hashes(local_dir, cache, excludes)
    CACHE_FILE.write_text(json.dumps(cache))
    if not loc:
        print("Nothing to upload — local dir is empty.")
        return 0

    # 2) remote checksums
    print("reading remote checksums ...", flush=True)
    subprocess.run(ssh_cmd + [args.host, f"mkdir -p {shlex.quote(args.remote)}"],
                   check=True)
    rem = remote_hashes(ssh_cmd, args.host, args.remote)

    # 3) decide what needs sending
    todo = sorted(rel for rel, sha in loc.items() if rem.get(rel) != sha)
    done = len(loc) - len(todo)
    print(f"\n{len(loc)} local files | {done} already correct | "
          f"{len(todo)} to upload\n")

    if not todo:
        print("✓ Everything is already uploaded and checksum-verified.")
        return 0

    # 3b) check disk space
    upload_size = sum(
        (local_dir / rel).stat().st_size
        for rel in todo
    )
    avail, total = remote_disk_space(ssh_cmd, args.host, args.remote)
    if avail > 0:
        print(f"Remote disk space: {format_bytes(avail)} available of "
              f"{format_bytes(total)} total")
        print(f"Upload size: {format_bytes(upload_size)}")
        if upload_size > avail:
            ratio = 100 * upload_size / avail if avail > 0 else 999
            print(f"\n✗ ERROR: not enough disk space! "
                  f"Need {format_bytes(upload_size)}, "
                  f"but only {format_bytes(avail)} available "
                  f"({ratio:.0f}% would exceed available)",
                  file=sys.stderr)
            return 2
        print()

    if args.dry_run:
        print(f"would upload {len(todo)} file(s) ({format_bytes(upload_size)})")
        return 0

    # 4) transfer files in parallel chunks
    num_workers = 4
    chunk_size = max(1, (len(todo) + num_workers - 1) // num_workers)
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]

    print(f"uploading {len(todo)} file(s) in {len(chunks)} parallel chunk(s) ...")
    failed_chunks = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(upload_chunk, chunk, local_dir, args.host, args.remote,
                     "", ssh_cmd, False): i
            for i, chunk in enumerate(chunks)
        }
        with tqdm(total=len(chunks), desc="upload", unit="chunk") as bar:
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    rc, info = future.result()
                    if rc != 0:
                        failed_chunks.append((chunk_idx, rc))
                except Exception as e:
                    failed_chunks.append((chunk_idx, str(e)))
                bar.update(1)

    if failed_chunks:
        print(f"\n✗ {len(failed_chunks)} chunk(s) failed — re-run this script to resume.",
              file=sys.stderr)
        return 1

    # 5) verify by re-reading remote checksums
    print("\nverifying checksums on remote ...", flush=True)
    rem = remote_hashes(ssh_cmd, args.host, args.remote)
    bad = [rel for rel in loc if rem.get(rel) != loc[rel]]
    if bad:
        print(f"\n✗ {len(bad)} file(s) still do NOT match — re-run to retry:")
        for rel in bad[:20]:
            print(f"    {rel}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        return 1

    print(f"\n✓ Done. All {len(loc)} files present and SHA-256 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
