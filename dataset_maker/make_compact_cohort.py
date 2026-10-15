#!/usr/bin/env python
"""Copy an npy cohort keeping only the windows a run's window selection uses.

Training reads at most the cohort's listed training windows (<= 20 min per
recording) and, for val/test, the first ``eval_minutes`` after the trim; the
rest of every recording is never read. This builder applies the run's own
selection (``WindowSelection`` from a fine-tune config plus ``--set``
overrides) to the source cohort, and writes ``<dst>/recordings/<stem>.npy``
holding just the selected windows, in ascending order, with the same dtype.

The manifest row keeps the original ``n_windows`` (so trimming and the eval
budget resolve exactly as on the full data) and lists the stored windows in
``window_index``; ``NpyWindowsMixin`` maps window ``k`` to its stored row. The
age split is restricted to the stored windows; age metadata, cohort table and
per-recording feature files are copied. ``window_stats.npz`` is not (its
per-window rows would need remapping), so quality rejection needs the full
cohort, and so does ``data.random_crop``.

The selection must match the runs that use the compact copy (trims, window
length, eval budget, case filter); a different selection needs its own copy.

    python -m dataset_maker.make_compact_cohort --src .../cohorts/all_both_v3 \\
        --dst .../cohorts/all_both_v3_compact --verify 2000
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.utils_conf import parse_overrides
from labram.data import get_dataset_bundle
from labram.data.tuh_metadata import recording_stem
from labram.data.window_selection import WindowSelection, _leaves, apply_window_selection, window_index

W = 2000
COPY = ("age_metadata.json", "cohort.parquet")


def selected_windows(root, selection):
    """stem -> sorted window indices the selection uses, and split -> files."""
    bundle = apply_window_selection(get_dataset_bundle("TUAB_AGE", root, data_format="npy"),
                                    selection)
    kept, files = {}, {}
    for split in ("train", "val", "test"):
        for leaf in _leaves(getattr(bundle, split)):
            for f in leaf.files:
                k = window_index(f)
                kept.setdefault(recording_stem(f), set()).update(
                    range(k, k + leaf.windows_per_item))
                files.setdefault(split, []).append(f)
    return {s: sorted(v) for s, v in kept.items()}, files


def _write(job):
    src, dst, ks = job
    x = np.load(src, mmap_mode="r")
    out = np.empty((len(ks) * W, x.shape[1]), dtype=x.dtype)
    for i, k in enumerate(ks):
        out[i * W:(i + 1) * W] = x[k * W:(k + 1) * W]
    np.save(dst, out)
    return out.nbytes


def build(src, dst, selection, threads=16):
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    kept, files = selected_windows(src, selection)
    os.makedirs(os.path.join(dst, "recordings"), exist_ok=True)
    rows = [dict(r) for r in manifest["recordings"] if r["stem"] in kept]
    jobs = [(os.path.join(src, "recordings", f"{r['stem']}.npy"),
             os.path.join(dst, "recordings", f"{r['stem']}.npy"), kept[r["stem"]]) for r in rows]
    t0, total, sizes = time.time(), 0, {}
    with ThreadPoolExecutor(threads) as pool:
        for i, (job, nbytes) in enumerate(zip(jobs, pool.map(_write, jobs)), 1):
            total += nbytes
            sizes[job[0]] = nbytes
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} recordings, {total / 1e9:.1f} GB, {time.time() - t0:.0f} s",
                      flush=True)
    for r in rows:
        r["window_index"] = kept[r["stem"]]
        r["bytes"] = sizes[os.path.join(src, "recordings", f"{r['stem']}.npy")]
        r.pop("sha256", None)
    out = {k: v for k, v in manifest.items() if k not in ("recordings", "n_recordings", "n_windows")}
    out.update(n_recordings=len(rows), n_windows=sum(r["n_windows"] for r in rows),
               n_stored_windows=sum(len(kept[r["stem"]]) for r in rows), recordings=rows,
               compact={"source": os.path.abspath(src), "selection": selection.__dict__})
    json.dump(out, open(os.path.join(dst, "manifest.json"), "w"))
    split = json.load(open(os.path.join(src, "age_split.json")))
    # Same order as the source's selected files: sample indices (hence the
    # seeded shuffle and per-epoch draws) must match the full cohort.
    split["files"] = {s: files.get(s, []) for s in split["files"]}
    json.dump(split, open(os.path.join(dst, "age_split.json"), "w"))
    for name in COPY + tuple(f for f in os.listdir(src)
                             if f.endswith(".npz") and f != "window_stats.npz"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    return total, out


def verify(src, dst, selection, n=2000, seed=0):
    """Bit-compare ``n`` random served samples (and targets) per split."""
    a = apply_window_selection(get_dataset_bundle("TUAB_AGE", src, data_format="npy"), selection)
    b = apply_window_selection(get_dataset_bundle("TUAB_AGE", dst, data_format="npy"), selection)
    rng = random.Random(seed)
    checked = 0
    for split in ("train", "val", "test"):
        la, lb = getattr(a, split), getattr(b, split)
        if la is None:
            continue
        assert la.files == lb.files, f"{split}: different samples after selection"
        for i in rng.sample(range(len(la)), min(n, len(la))):
            (xa, ya), (xb, yb) = la[i], lb[i]
            assert torch.equal(xa, xb) and torch.equal(ya, yb), f"{split} sample {la.files[i]} differs"
            checked += 1
    return checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--config", default="labram/configs/defaults/finetune_tuab_age.json",
                   help="fine-tune config whose data.* window selection the runs use")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.eval_minutes=5")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--verify", type=int, default=0, help="bit-compare N random samples per split")
    args = p.parse_args(argv)
    cfg = FinetuneRunConfig.load_config(args.config, **parse_overrides(args.set))
    selection = WindowSelection.from_data_config(cfg.data)
    if selection.quality_filter:
        raise SystemExit("compact copies do not carry window_stats.npz: drop the quality filter")
    print(f"selection: {selection}")
    total, out = build(args.src, args.dst, selection, args.threads)
    print(f"{args.dst}: {out['n_recordings']} recordings, {out['n_stored_windows']} of "
          f"{out['n_windows']} windows stored, {total / 1e9:.1f} GB")
    if args.verify:
        print(f"verified {verify(args.src, args.dst, selection, args.verify)} samples bit-identical")


if __name__ == "__main__":
    sys.exit(main())
