# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-window signal quality statistics for the per-recording npy format, and
# the amplitude / flat-channel rejection rule window selection applies with
# them. See docs/age_regression.md.
# ---------------------------------------------------------
"""Per-window quality stats of an npy dataset (``window_stats.npz``).

For window ``k`` of a recording (samples ``[2000 k, 2000 k + 2000)``, µV), over
the channels the manifest's ``channel_mask`` marks present:

    max_abs[k] = max over channels and samples of |x|
    min_std[k] = min over channels of the population std over samples

A window is rejected when ``max_abs > max_abs_uv`` (an artifact or a saturated
amplifier) or ``min_std < min_std_uv`` (a flat, disconnected channel); a
threshold of 0 disables that test.

The stats live next to ``manifest.json``, so a dataset copied to S3 / a
SageMaker channel carries them. Build them with::

    python -m labram.data.window_qc --root <npy dir> [--s3 s3://.../<npy dir>/]
"""
import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_STATS_FILE = "window_stats.npz"
WINDOW_SAMPLES = 2000
CHUNK_WINDOWS = 180          # 30 min of 23-channel float32 per read: ~33 MB


def recording_window_stats(x, channel_mask=None) -> Tuple[np.ndarray, np.ndarray]:
    """``(max_abs, min_std)`` per whole 10 s window of ``x`` (``[T, C]``, µV)."""
    n = x.shape[0] // WINDOW_SAMPLES
    cols = np.flatnonzero(channel_mask) if channel_mask is not None else slice(None)
    max_abs = np.empty(n, np.float32)
    min_std = np.empty(n, np.float32)
    for a in range(0, n, CHUNK_WINDOWS):
        b = min(n, a + CHUNK_WINDOWS)
        w = np.asarray(x[a * WINDOW_SAMPLES:b * WINDOW_SAMPLES], dtype=np.float32)[:, cols]
        w = w.reshape(b - a, WINDOW_SAMPLES, -1)
        max_abs[a:b] = np.abs(w).max(axis=(1, 2))
        min_std[a:b] = w.std(axis=1).min(axis=1)
    return max_abs, min_std


def save_window_stats(path: str, stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    stems = sorted(stats)
    lengths = np.array([len(stats[s][0]) for s in stems], np.int64)
    np.savez(path, stems=np.array(stems), offsets=np.concatenate([[0], np.cumsum(lengths)]),
             max_abs=np.concatenate([stats[s][0] for s in stems]) if stems else np.zeros(0, np.float32),
             min_std=np.concatenate([stats[s][1] for s in stems]) if stems else np.zeros(0, np.float32))


def load_window_stats(root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``stem -> (max_abs, min_std)`` from ``<root>/window_stats.npz``."""
    path = os.path.join(root, WINDOW_STATS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"window quality rejection needs {path}; build it with "
            f"`python -m labram.data.window_qc --root {root}`")
    with np.load(path) as z:
        stems, off, mx, sd = z["stems"], z["offsets"], z["max_abs"], z["min_std"]
    return {str(s): (mx[off[i]:off[i + 1]], sd[off[i]:off[i + 1]]) for i, s in enumerate(stems)}


def window_passes(max_abs, min_std, max_abs_uv: float, min_std_uv: float) -> np.ndarray:
    """Boolean mask of the windows that pass both tests (0 disables a test)."""
    ok = np.ones(len(max_abs), bool)
    if max_abs_uv > 0:
        ok &= max_abs <= max_abs_uv
    if min_std_uv > 0:
        ok &= min_std >= min_std_uv
    return ok


# ---------------------------------------------------------------- build (CLI)

def _one(job):
    root, stem, mask, s3_prefix, tmp_dir = job
    path = os.path.join(root, "recordings", f"{stem}.npy")
    fetched = None
    if not os.path.exists(path):
        if not s3_prefix:
            return stem, None
        import boto3
        bucket, key = s3_prefix[5:].split("/", 1)
        fd, fetched = tempfile.mkstemp(suffix=".npy", dir=tmp_dir)
        os.close(fd)
        boto3.client("s3").download_file(bucket, f"{key.rstrip('/')}/recordings/{stem}.npy", fetched)
        path = fetched
    try:
        return stem, recording_window_stats(np.load(path, mmap_mode="r"), mask)
    finally:
        if fetched:
            os.remove(fetched)


def build(root: str, s3_prefix: Optional[str] = None, workers: int = 8,
          tmp_dir: Optional[str] = None, out: Optional[str] = None) -> str:
    import json
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    jobs = [(root, r["stem"], r.get("channel_mask"), s3_prefix, tmp_dir)
            for r in manifest["recordings"]]
    stats, missing = {}, []
    with ProcessPoolExecutor(workers) as pool:
        for i, (stem, st) in enumerate(pool.map(_one, jobs, chunksize=4), 1):
            if st is None:
                missing.append(stem)
            else:
                stats[stem] = st
            if i % 1000 == 0:
                logger.info("window stats: %d / %d recordings", i, len(jobs))
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings are neither in {root} nor "
                                f"fetchable (pass --s3), e.g. {missing[:3]}")
    out = out or os.path.join(root, WINDOW_STATS_FILE)
    save_window_stats(out, stats)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="npy dataset dir (manifest.json)")
    p.add_argument("--s3", default=None, help="S3 copy of the dataset, for recordings not on local disk")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--tmp-dir", default=None, help="where fetched recordings are staged")
    p.add_argument("--out", default=None, help="default <root>/window_stats.npz")
    p.add_argument("--max-abs-uv", type=float, default=800.0, help="for the printed summary")
    p.add_argument("--min-std-uv", type=float, default=0.5, help="for the printed summary")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = build(args.root, args.s3, args.workers, args.tmp_dir, args.out)
    with np.load(out) as z:
        mx, sd = z["max_abs"], z["min_std"]
    n = len(mx)
    print(f"{out}: {n} windows; > {args.max_abs_uv:g} uV: {(mx > args.max_abs_uv).mean():.2%}, "
          f"min channel std < {args.min_std_uv:g} uV: {(sd < args.min_std_uv).mean():.2%}, "
          f"rejected: {(~window_passes(mx, sd, args.max_abs_uv, args.min_std_uv)).mean():.2%}")


if __name__ == "__main__":
    sys.exit(main())
