# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Per-window feature cache of trained fine-tune runs, the input of MIL Stage 1.
# ---------------------------------------------------------
"""Cache what a trained run sees of every window, once.

For each split of a run's window-selected data (``train`` restricted to the
first ``n_train_windows`` windows of every recording, i.e. the same budget as
val/test), store under ``<out>/<run>/<split>/``:

* ``h.npy``    float16 ``[N, D]`` -- the head-input feature of each window
  (``model(x, return_features=True)``; the CLS token after the final norm for
  ``use_mean_pooling=false`` runs such as E2L), computed through the run's
  channel dropper when it has one, exactly like training;
* ``pred.npy`` float32 ``[N]`` -- the window's age prediction in years;
* ``desc.npy`` float32 ``[N, Q]`` -- quality + STFT + wavelet descriptors of
  the raw window (``labram/eval/window_descriptors.py``), names in
  ``desc_columns.json``;
* ``meta.parquet`` -- ``recording, window, minute, age, cohort, dataset`` per row.

``<out>/<run>/head.pt`` holds the run's head weights and ``run.json`` its
target statistics, so a Stage 1 head can start from the trained one. Several
runs on the same split share one data pass (:func:`build_cache`).
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange

DESC_FUNCS = ("quality", "stft", "cwt")


def first_windows(dataset, n: int) -> List[int]:
    """Indices of the first ``n`` windows (by window index) of every recording
    of a window-selected dataset, in dataset order."""
    from labram.data.window_selection import window_index
    from labram.eval.window_descriptors import TTAWindows
    data = TTAWindows(dataset, n_crops=0, pad=0)
    by_rec: Dict[str, List[tuple]] = defaultdict(list)
    for i in range(len(data)):
        leaf, filename = data._locate(i)
        stem = os.path.basename(filename)[:-4].rsplit(leaf._recording_sep, 1)[0]
        by_rec[stem].append((window_index(filename, leaf._recording_sep), i))
    keep = [i for rows in by_rec.values() for _, i in sorted(rows)[:n]]
    return sorted(keep)


def all_descriptors(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    from labram.eval.window_descriptors import cwt_descriptors, quality_descriptors, stft_descriptors
    out = {}
    for fn in (quality_descriptors, stft_descriptors, cwt_descriptors):
        out.update(fn(x))
    return out


@torch.no_grad()
def encode(model, dropper, x_uv: torch.Tensor, channel_indices):
    """(window predictions in model units, head-input features) of raw windows ``(B, C, L)``."""
    x = rearrange(x_uv / 100, "B N (A T) -> B N A T", T=200)
    with torch.amp.autocast(x.device.type, enabled=x.device.type == "cuda"):
        if dropper is not None:
            out, feat = dropper.forward(model, x, dropper.present(x), return_features=True)
        else:
            out, feat = model(x, channel_indices=channel_indices, return_features=True)
    out = getattr(out, "logits", out)
    return out.float().reshape(x.shape[0], -1)[:, 0], feat.float()


def logged_target_stats(run_dir, default):
    import re
    path = os.path.join(run_dir, "log", "run.log")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                m = re.search(r"Age target normalization.*mean=([-\d.]+) std=([-\d.]+)", line)
                if m:
                    return float(m.group(1)), float(m.group(2))
    return default


def build_cache(runs: Dict[str, str], data_path: str, out_dir: str, splits=("val", "test", "train"),
                n_train_windows: int = 30, batch_size: int = 128, num_workers: int = 12,
                checkpoint: str = "checkpoint-best.pth", max_windows: int = 0) -> None:
    """Write the cache of every run in ``runs`` (name -> run dir); all runs must
    share the split and window selection (checked)."""
    import labram.utils as utils
    from labram.data.dataset_groups import UNKNOWN, dataset_lookup
    from labram.eval.age_analysis import load_age_run
    from labram.eval.window_descriptors import TTAWindows
    from labram.data.window_selection import window_index
    from labram.train.channel_drop import ChannelDropper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for name, run_dir in runs.items():
        run = load_age_run(run_dir, data_path, checkpoint, stage_dir="none")
        run.pred_stats = logged_target_stats(run_dir, run.target_stats)
        if loaded:
            for split in splits:
                if list(run.split(split).files) != list(loaded[0][1].split(split).files):
                    raise SystemExit(f"{name}: {split} windows differ from {loaded[0][0]}'s")
            run.bundle = loaded[0][1].bundle
        dropper = ChannelDropper.from_config(getattr(run.config, "channel_drop", None), run.ch_names)
        loaded.append((name, run, dropper))
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        torch.save(run.model.head.state_dict(), os.path.join(out_dir, name, "head.pt"))
        with open(os.path.join(out_dir, name, "run.json"), "w") as fh:
            json.dump({"run_dir": run_dir, "checkpoint": checkpoint, "epoch": run.epoch,
                       "pred_stats": run.pred_stats, "target_stats": run.target_stats,
                       "ch_names": run.ch_names, "data_path": data_path,
                       "head_type": run.config.model.head_type}, fh, indent=2)
    ref = loaded[0][1]
    channel_indices = utils.get_channel_indices(ref.ch_names)

    for split in splits:
        dataset = ref.split(split)
        data = TTAWindows(dataset, n_crops=0, pad=0)
        index = first_windows(dataset, n_train_windows) if split == "train" else list(range(len(data)))
        if max_windows:
            index = index[:max_windows]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, index), batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
        feats = {name: [] for name, _, _ in loaded}
        preds = {name: [] for name, _, _ in loaded}
        desc, recs, wins, ages = defaultdict(list), [], [], []
        t0, done = time.time(), 0
        for copies, y, stem, k, _ in loader:
            x = copies[:, 0].to(device, non_blocking=True)
            for name, run, dropper in loaded:
                p, h = encode(run.model, dropper, x, channel_indices)
                preds[name].append(p.cpu())
                feats[name].append(h.half().cpu())
            for key, v in all_descriptors(x).items():
                desc[key].append(v.float().cpu())
            recs += list(stem)
            wins += [int(v) for v in k]
            ages.append(y.reshape(len(stem), -1)[:, 0])
            done += len(stem)
            if done // batch_size % 200 == 0:
                rate = done / (time.time() - t0)
                print(f"  {split}: {done}/{len(index)} windows, {rate:.0f}/s, "
                      f"eta {(len(index) - done) / rate / 60:.1f} min", flush=True)
        lookup = dataset_lookup(dataset)
        meta = pd.DataFrame({"recording": recs, "window": wins})
        meta["minute"] = (meta["window"] - meta.groupby("recording")["window"].transform("min")) / 6.0
        meta["age"] = utils.denormalize(torch.cat(ages).numpy(), ref.target_stats)
        meta["cohort"] = [ref.labels.get(r) if ref.labels.get(r) in ("normal", "abnormal")
                          else "unlabelled" for r in recs]
        meta["dataset"] = [lookup.get(r, UNKNOWN) for r in recs]
        cols = list(desc)
        dmat = np.stack([torch.cat(desc[c]).numpy() for c in cols], 1).astype(np.float32)
        for name, run, _ in loaded:
            d = os.path.join(out_dir, name, split)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "h.npy"), torch.cat(feats[name]).numpy())
            np.save(os.path.join(d, "pred.npy"),
                    utils.denormalize(torch.cat(preds[name]).numpy(), run.pred_stats).astype(np.float32))
            np.save(os.path.join(d, "desc.npy"), dmat)
            meta.to_parquet(os.path.join(d, "meta.parquet"))
            with open(os.path.join(d, "desc_columns.json"), "w") as fh:
                json.dump(cols, fh)
        print(f"wrote {split}: {len(meta)} windows, {meta.recording.nunique()} recordings", flush=True)


class Cache:
    """Read access to one run's cache: ``Cache(root, run).split('val')`` ->
    dict with ``h`` (float32), ``pred``, ``desc``, ``meta``, ``desc_columns``."""

    def __init__(self, root: str, run: str):
        self.dir = os.path.join(root, run)
        with open(os.path.join(self.dir, "run.json")) as fh:
            self.info = json.load(fh)

    def head_state(self):
        return torch.load(os.path.join(self.dir, "head.pt"), map_location="cpu", weights_only=True)

    def split(self, name: str, mmap: bool = True) -> dict:
        d = os.path.join(self.dir, name)
        mode = "r" if mmap else None
        out = {"h": np.load(os.path.join(d, "h.npy"), mmap_mode=mode),
               "pred": np.load(os.path.join(d, "pred.npy")),
               "desc": np.load(os.path.join(d, "desc.npy"), mmap_mode=mode),
               "meta": pd.read_parquet(os.path.join(d, "meta.parquet"))}
        with open(os.path.join(d, "desc_columns.json")) as fh:
            out["desc_columns"] = json.load(fh)
        return out


def sync_to_s3(local_dir: str, s3_uri: str, profile=None) -> None:
    """Upload ``local_dir`` to ``s3_uri`` after confirming the bucket cannot be
    read publicly (the cache holds HEEDB-derived data; HEEDB DUA)."""
    import subprocess
    import boto3
    from dataset_maker.download_heedb import check_private
    bucket = s3_uri.split("/")[2]
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    check_private(session.client("s3"), bucket)
    cmd = ["aws", "s3", "sync", local_dir, s3_uri, "--only-show-errors"]
    if profile:
        cmd += ["--profile", profile]
    subprocess.run(cmd, check=True)
