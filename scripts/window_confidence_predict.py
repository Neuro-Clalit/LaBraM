#!/usr/bin/env python
"""Window predictions with test-time augmentation (TTA) and quality descriptors.

Input for the confidence-weighted pooling of ``docs/window_aggregation_plan.md``
(Proposal A, analysed by ``labram/eval/window_confidence.py`` and
``scripts/window_confidence_report.py``). For every val/test window of the
runs' shared split it stores

* ``pred`` -- the plain prediction (the copy training evaluates),
* ``tta_1 .. tta_{M-1}`` -- the predictions of label-preserving perturbed copies:
  the hemisphere-flipped window and ``--crops`` crops shifted uniformly by up to
  ``--shift-sec`` inside the recording's trimmed range (the last crop also
  hemisphere-flipped), Eq. (3.1) of the plan,
* per-window descriptors of the raw (common-average-referenced) signal:
  relative 30-45 Hz (EMG) power, its channel maximum, max |x|, the mean and
  minimum channel standard deviation.

Each window is read once, with ``--shift-sec`` of context on both sides, and
every run in ``--run`` is scored on the same batch, so the I/O is shared. All
runs must have the same val/test windows (same split and window selection);
this is checked.

    python scripts/window_confidence_predict.py \\
        --data-path /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --run E2L=/data/runs/heedb/e2l/finetune_... --run E2=/data/runs/heedb/e2_r2/finetune \\
        --out-dir /data/runs/heedb/window_conf
"""
import argparse
import bisect
import os
import time

import numpy as np
import pandas as pd
import torch
from einops import rearrange

import labram.utils as utils
from labram.data.dataset_groups import UNKNOWN, dataset_lookup
from labram.data.tuh_datasets import WINDOW_SAMPLES
from labram.data.window_selection import _leaves
from labram.eval.age_analysis import load_age_run
from labram.train.augment import hemisphere_permutation
from labram.train.channel_drop import ChannelDropper

FS = 200.0
EMG_BAND, TOTAL_BAND = (30.0, 45.0), (1.0, 45.0)


class TTAWindows(torch.utils.data.Dataset):
    """Every window of a (window-selected npy) dataset as ``n_crops + 1``
    copies: the window itself and ``n_crops`` crops whose start moves by at most
    ``pad`` samples, kept inside the recording's trimmed range. One read per
    window (the window plus ``pad`` on each side). Crop offsets are seeded per
    window, so every run and every re-run sees the same copies."""

    def __init__(self, dataset, n_crops: int, pad: int, seed: int = 0):
        self.leaves = _leaves(dataset)
        self.starts = np.cumsum([0] + [len(leaf.files) for leaf in self.leaves])
        self.n_crops, self.pad, self.seed = n_crops, pad, seed

    def __len__(self):
        return int(self.starts[-1])

    def _locate(self, i):
        li = bisect.bisect_right(self.starts, i) - 1
        leaf = self.leaves[li]
        return leaf, leaf.files[i - self.starts[li]]

    def stem(self, i):
        leaf, filename = self._locate(i)
        return os.path.basename(filename)[:-4].rsplit(leaf._recording_sep, 1)[0]

    def __getitem__(self, i):
        leaf, filename = self._locate(i)
        stem, k = os.path.basename(filename)[:-4].rsplit(leaf._recording_sep, 1)
        length = WINDOW_SAMPLES * leaf.windows_per_item
        start = int(k) * WINDOW_SAMPLES
        lo, hi = leaf.sample_bounds.get(stem) or (0, leaf._n_samples(stem))
        a, b = max(lo, start - self.pad), min(hi, start + length + self.pad)
        a, b = min(a, start), max(b, start + length)
        seg = leaf._read_rows(stem, a, b - a)
        if leaf.keep_channels is not None:
            seg = seg[:, leaf.keep_channels]
        x = np.ascontiguousarray(seg.T, dtype=np.float32)
        rng = np.random.default_rng((self.seed, i))
        offsets = [start] + [int(o) for o in rng.integers(a, b - length + 1, self.n_crops)]
        copies = np.stack([x[:, o - a:o - a + length] for o in offsets])
        y = leaf._label_fn(None, filename)
        shift = (np.asarray(offsets[1:]) - start) / FS
        return torch.from_numpy(copies), torch.as_tensor(y).float(), stem, int(k), torch.from_numpy(shift)


def descriptors(x: torch.Tensor) -> dict:
    """Quality descriptors of a batch of raw windows ``(B, C, L)`` in uV, over
    the channels that carry signal, after common-average reference."""
    present = (x.abs().amax(-1) > 0).float()                     # (B, C)
    n = present.sum(1).clamp_min(1)
    car = (x - (x * present[..., None]).sum(1, keepdim=True) / n[:, None, None]) * present[..., None]
    spec = torch.fft.rfft(car, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / FS).to(x.device)
    band = lambda lo, hi: spec[..., (freqs >= lo) & (freqs < hi)].sum(-1)   # noqa: E731
    rel = band(*EMG_BAND) / band(*TOTAL_BAND).clamp_min(1e-12)   # (B, C)
    std = car.std(-1)
    big = torch.finfo(std.dtype).max
    return {
        "emg": ((rel * present).sum(1) / n),
        "emg_max": (rel * present).amax(1),
        "max_abs": (car.abs().amax(-1) * present).amax(1),
        "amp_uv": (std * present).sum(1) / n,
        "min_std": torch.where(present > 0, std, torch.full_like(std, big)).amin(1),
    }


def score(model, dropper, x, channel_indices):
    """Window outputs (z-scored age) of ``x`` ``(B, C, L)`` in uV."""
    x = rearrange(x / 100, "B N (A T) -> B N A T", T=200)
    with torch.amp.autocast(x.device.type, enabled=x.device.type == "cuda"):
        if dropper is not None:
            out = dropper.forward(model, x, dropper.present(x))
        else:
            out = model(x, channel_indices=channel_indices, classify_only=True)
    out = getattr(out, "logits", out)
    return out.float().reshape(x.shape[0], -1)[:, 0]


def logged_target_stats(run_dir, default):
    """(mean, std) of the training targets as the run logged them
    (``log/run.log``), else ``default``."""
    import re
    path = os.path.join(run_dir, "log", "run.log")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                m = re.search(r"Age target normalization.*mean=([-\d.]+) std=([-\d.]+)", line)
                if m:
                    return float(m.group(1)), float(m.group(2))
    return default


def predict_split(runs, split, args, device):
    ref = runs[0][1]
    data = TTAWindows(ref.split(split), args.crops, int(args.shift_sec * FS), args.seed)
    if args.max_recordings:                                     # quick check
        stems = [data.stem(i) for i in range(len(data))]
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(sorted(set(stems)), min(args.max_recordings, len(set(stems))),
                              replace=False))
        data = torch.utils.data.Subset(data, [i for i, s in enumerate(stems) if s in keep])
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    perm = torch.as_tensor(hemisphere_permutation(ref.ch_names), device=device)
    channel_indices = utils.get_channel_indices(ref.ch_names)
    droppers = [ChannelDropper.from_config(getattr(r.config, "channel_drop", None), ref.ch_names)
                for _, r in runs]
    preds = {name: [] for name, _ in runs}
    rows = {"recording": [], "window": [], "age_z": []}
    desc, shifts = {}, []
    t0, done = time.time(), 0
    with torch.no_grad():
        for copies, y, stem, k, shift in loader:
            copies = copies.to(device, non_blocking=True)         # (B, 1 + crops, C, L)
            b, m = copies.shape[:2]
            base, crops = copies[:, 0], copies[:, 1:]
            # copies scored: base, flip(base), crop_1 .. crop_{n-1}, flip(crop_n)
            batch = [base, base[:, perm]] + [crops[:, c] for c in range(m - 2)] + [crops[:, -1][:, perm]]
            x = torch.cat(batch)
            for (name, run), dropper in zip(runs, droppers):
                out = score(run.model, dropper, x, channel_indices)
                preds[name].append(out.reshape(len(batch), b).T.cpu())
            for key, v in descriptors(base).items():
                desc.setdefault(key, []).append(v.float().cpu())
            rows["recording"] += list(stem)
            rows["window"] += [int(v) for v in k]
            rows["age_z"].append(y.reshape(b, -1)[:, 0])
            shifts.append(shift)
            done += b
            if done // args.batch_size % 50 == 0:
                rate = done / (time.time() - t0)
                print(f"  {split}: {done}/{len(data)} windows, {rate:.0f}/s, "
                      f"eta {(len(data) - done) / rate / 60:.1f} min", flush=True)
    lookup = dataset_lookup(ref.split(split))
    out = {}
    for name, run in runs:
        p = utils.denormalize(torch.cat(preds[name]).numpy(), run.pred_stats)
        df = pd.DataFrame({"split": split, "recording": rows["recording"], "window": rows["window"]})
        df["age"] = utils.denormalize(torch.cat(rows["age_z"]).numpy(), ref.target_stats)
        df["cohort"] = [ref.labels.get(r) if ref.labels.get(r) in ("normal", "abnormal")
                        else "unlabelled" for r in df["recording"]]
        df["dataset"] = [lookup.get(r, UNKNOWN) for r in df["recording"]]
        df["pred"] = p[:, 0]
        for j in range(1, p.shape[1]):
            df[f"tta_{j}"] = p[:, j]
        for key, v in desc.items():
            df[key] = torch.cat(v).numpy()
        sh = torch.cat(shifts).numpy()
        for j in range(sh.shape[1]):
            df[f"shift_{j + 1}"] = sh[:, j]
        out[name] = df
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--checkpoint", default="checkpoint-best.pth")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--crops", type=int, default=3, help="shifted crops per window (>= 1)")
    ap.add_argument("--shift-sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--max-recordings", type=int, default=0, help="score a random subset (smoke test)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for spec in args.run:
        name, run_dir = spec.split("=", 1)
        run = load_age_run(run_dir, args.data_path, args.checkpoint, stage_dir="none")
        run.pred_stats = logged_target_stats(run_dir, run.target_stats)
        print(f"{name}: epoch {run.epoch}, {len(run.ch_names)} channels, "
              f"pred stats {run.pred_stats}", flush=True)
        if runs:
            for split in args.splits:
                if list(run.split(split).files) != list(runs[0][1].split(split).files):
                    raise SystemExit(f"{name}: {split} windows differ from {runs[0][0]}'s")
            run.bundle = runs[0][1].bundle            # score on the first run's datasets
        runs.append((name, run))

    os.makedirs(args.out_dir, exist_ok=True)
    frames = {name: [] for name, _ in runs}
    for split in args.splits:
        for name, df in predict_split(runs, split, args, device).items():
            frames[name].append(df)
    for name, parts in frames.items():
        path = os.path.join(args.out_dir, f"{name}_windows.parquet")
        pd.concat(parts, ignore_index=True).to_parquet(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
