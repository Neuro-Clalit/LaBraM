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

A *pretrained, not fine-tuned* encoder (LaBraM-base, CBraMod, REVE) is cached
the same way from a run config and its released weights
(:func:`load_pretrained_run`): ``h.npy`` is then the frozen window embedding a
ridge probe reads (``scripts/fm_probe.py``); it has no trained head, so
``pred.npy`` is NaN and there is no ``head.pt``.
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


@torch.no_grad()
def encode_channel_means(model, x_uv: torch.Tensor, channel_indices) -> torch.Tensor:
    """Per-channel mean of the encoder's patch tokens, concatenated over
    channels: ``(B, C, L)`` raw windows -> ``(B, C * D)``. Unlike the pooled
    head input this keeps the topography, as the flattened-token probes of the
    encoders' own papers do. The feature has one block per channel, so every
    window is encoded on all its channels: a flat (all-zero) channel is fed as
    it is, not left out as the channel dropper would."""
    x = rearrange(x_uv / 100, "B N (A T) -> B N A T", T=200)
    with torch.amp.autocast(x.device.type, enabled=x.device.type == "cuda"):
        tokens = model.forward_features(x, channel_indices=channel_indices, return_patch_tokens=True)
    b, c = x.shape[:2]
    return tokens.float().reshape(b, c, -1, tokens.shape[-1]).mean(2).flatten(1)


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


def load_pretrained_run(config: str, data_path: str, overrides: Sequence[str] = (),
                        stage_dir: str = "none", shared=None):
    """A pretrained, not fine-tuned encoder as an ``AgeRun``: the model of the
    run config ``config`` (+ ``key=value`` overrides) with the weights of
    ``finetune_checkpoint.finetune`` and an untrained head. The data, window
    selection and channels are the config's, read from ``data_path``.
    ``shared`` (an ``AgeRun``) lends its dataset bundle when its data section
    is the same, so several encoders on one cohort list its windows once."""
    import labram.models.registry  # noqa: F401
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.configs.utils_conf import parse_overrides
    from labram.data.tuh_metadata import load_label_lookup_for
    from labram.eval.age_analysis import AgeRun, age_bundle_for_config
    from labram.runs.finetune_setup import load_finetune_checkpoint
    from labram.runs.model_factory import build_finetune_model

    cfg = FinetuneRunConfig.load_config(config, raise_exception=False, **parse_overrides(list(overrides)))
    cfg.mil.enabled = False                    # window features: no bag pool
    cfg.codebook_aux.enabled = False
    cfg.data.data_path, cfg.data.stage_dir = data_path, stage_dir
    if shared is not None and str(shared.config.data) == str(cfg.data):
        bundle = shared.bundle
    else:
        bundle = age_bundle_for_config(cfg, data_path, stage_dir)
    cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
    model = build_finetune_model(cfg)
    if not cfg.finetune_checkpoint.finetune:
        raise ValueError(f"{config}: a pretrained encoder needs finetune_checkpoint.finetune")
    load_finetune_checkpoint(model, cfg.finetune_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    labels = load_label_lookup_for(getattr(bundle.test, "root", data_path))
    return AgeRun(config, cfg, bundle, model, labels, device, None, None)


def recording_channel_stats(loader, device) -> Dict[str, tuple]:
    """Per-recording ``(mean [C], std [C])`` over the windows ``loader`` yields."""
    acc: Dict[str, list] = {}
    for copies, _, stems, _, _ in loader:
        x = copies[:, 0].to(device, non_blocking=True).double()          # [B, C, L]
        s1, s2, n = x.sum(-1), x.pow(2).sum(-1), x.shape[-1]
        for i, stem in enumerate(stems):
            a = acc.setdefault(stem, [torch.zeros_like(s1[0]), torch.zeros_like(s2[0]), 0])
            a[0] += s1[i]
            a[1] += s2[i]
            a[2] += n
    out = {}
    for stem, (s1, s2, n) in acc.items():
        mean = s1 / n
        out[stem] = (mean.float(), (s2 / n - mean.pow(2)).clamp_min(0).sqrt().float())
    return out


def zscore_by_recording(x_uv: torch.Tensor, stems, stats, clip_sd: float = 15.0) -> torch.Tensor:
    """Z-score every channel with its recording's statistics and clip at
    ``+-clip_sd`` (REVE's pretraining normalisation). Returned times 100, so
    the ``/ 100`` of :func:`encode` hands the model the z-scores themselves.
    A channel without signal in the recording stays zero."""
    mean = torch.stack([stats[s][0] for s in stems])[:, :, None]
    std = torch.stack([stats[s][1] for s in stems])[:, :, None]
    z = torch.where(std > 1e-6, (x_uv - mean) / std.clamp_min(1e-6), torch.zeros_like(x_uv))
    return z.clamp(-clip_sd, clip_sd) * 100


def build_cache(runs: Dict[str, str], data_path: str, out_dir: str, splits=("val", "test", "train"),
                n_train_windows: int = 30, batch_size: int = 128, num_workers: int = 12,
                checkpoint: str = "checkpoint-best.pth", max_windows: int = 0,
                pretrained: Dict[str, tuple] = None, recordings=None,
                recording_zscore: Sequence[str] = (), descriptors: bool = True,
                readout: str = "pooled") -> None:
    """Write the cache of every run in ``runs`` (name -> run dir) and of every
    pretrained encoder in ``pretrained`` (name -> ``(config, overrides)``, see
    :func:`load_pretrained_run`); all must share the split and window selection
    (checked). A pretrained encoder has no trained head: its ``pred.npy`` is
    NaN and it writes no ``head.pt``.

    ``recordings``: cache only these recording stems (a list, or ``{split:
    stems}``; a split the dict does not name is cached whole). ``recording_zscore``:
    names whose input is z-scored per recording and channel first (a second
    pass over the split). ``descriptors=False`` skips the window descriptors
    (``desc.npy`` then has no columns). ``readout="channel_mean"`` stores
    :func:`encode_channel_means` as ``h.npy`` instead of the pooled head input
    (pretrained encoders only; ``pred.npy`` is NaN)."""
    import labram.utils as utils
    from labram.data.dataset_groups import UNKNOWN, dataset_lookup
    from labram.eval.age_analysis import load_age_run
    from labram.eval.window_descriptors import TTAWindows
    from labram.train.channel_drop import ChannelDropper

    if readout not in ("pooled", "channel_mean"):
        raise SystemExit(f"readout must be pooled or channel_mean, got {readout!r}")
    if readout == "channel_mean" and runs:
        raise SystemExit("readout=channel_mean is for --pretrained encoders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [(name, run_dir, None) for name, run_dir in runs.items()]
    specs += [(name, cfg, list(over)) for name, (cfg, over) in (pretrained or {}).items()]
    unknown = set(recording_zscore) - {name for name, _, _ in specs}
    if unknown:
        raise SystemExit(f"recording_zscore names {sorted(unknown)} are not among the cached runs")
    loaded = []
    for name, source, overrides in specs:
        frozen = overrides is not None
        if frozen:
            run = load_pretrained_run(source, data_path, overrides,
                                      shared=loaded[0][1] if loaded else None)
            run.pred_stats = run.target_stats
        else:
            run = load_age_run(source, data_path, checkpoint, stage_dir="none")
            run.pred_stats = logged_target_stats(source, run.target_stats)
        if loaded and run.bundle is not loaded[0][1].bundle:
            for split in splits:
                if list(run.split(split).files) != list(loaded[0][1].split(split).files):
                    raise SystemExit(f"{name}: {split} windows differ from {loaded[0][0]}'s")
            run.bundle = loaded[0][1].bundle
        dropper = ChannelDropper.from_config(getattr(run.config, "channel_drop", None), run.ch_names)
        if name in recording_zscore:
            lp = getattr(run.config, "labram_plus", None)
            if dropper is not None or (lp is not None and lp.preprocesses_input):
                raise SystemExit(f"{name}: recording z-scoring needs labram_plus and channel_drop off "
                                 "(the z-score would otherwise come before them)")
        loaded.append((name, run, dropper, frozen))
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        if not frozen:
            torch.save(run.model.head.state_dict(), os.path.join(out_dir, name, "head.pt"))
        info = {"run_dir": source, "checkpoint": None if frozen else checkpoint, "epoch": run.epoch,
                "pred_stats": run.pred_stats, "target_stats": run.target_stats,
                "ch_names": run.ch_names, "data_path": data_path,
                "head_type": run.config.model.head_type, "pretrained": frozen,
                "model": run.config.model.model, "recording_zscore": name in recording_zscore,
                "readout": readout}
        if frozen:
            info["overrides"] = overrides
            info["weights"] = run.config.finetune_checkpoint.finetune
        with open(os.path.join(out_dir, name, "run.json"), "w") as fh:
            json.dump(info, fh, indent=2)
    ref = loaded[0][1]
    channel_indices = utils.get_channel_indices(ref.ch_names)

    for split in splits:
        dataset = ref.split(split)
        data = TTAWindows(dataset, n_crops=0, pad=0)
        index = first_windows(dataset, n_train_windows) if split == "train" else list(range(len(data)))
        keep = recordings.get(split) if isinstance(recordings, dict) else recordings
        if keep is not None:
            keep = set(keep)
            index = [i for i in index if data.stem(i) in keep]
        if max_windows:
            index = index[:max_windows]
        if not index:
            print(f"{split}: no windows to cache", flush=True)
            continue

        def make_loader():
            return torch.utils.data.DataLoader(torch.utils.data.Subset(data, index), batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers, pin_memory=True)
        stats = recording_channel_stats(make_loader(), device) if recording_zscore else None
        # Features go straight to disk (a per-channel readout of the training
        # split is tens of GB): one preallocated float16 file per run, renamed to
        # h.npy when the split is complete, so a partial file is never reused.
        feats = {}
        preds = {name: [] for name, *_ in loaded}
        desc, recs, wins, ages = defaultdict(list), [], [], []
        t0, done = time.time(), 0
        for copies, y, stem, k, _ in make_loader():
            x = copies[:, 0].to(device, non_blocking=True)
            for name, run, dropper, _ in loaded:
                xin = zscore_by_recording(x, stem, stats) if name in recording_zscore else x
                if readout == "channel_mean":
                    h = encode_channel_means(run.model, xin, channel_indices)
                    p = torch.full((len(stem),), float("nan"))
                else:
                    p, h = encode(run.model, dropper, xin, channel_indices)
                preds[name].append(p.cpu())
                if name not in feats:
                    d = os.path.join(out_dir, name, split)
                    os.makedirs(d, exist_ok=True)
                    feats[name] = np.lib.format.open_memmap(
                        os.path.join(d, "h.npy.partial"), mode="w+", dtype=np.float16,
                        shape=(len(index), h.shape[1]))
                feats[name][done:done + len(stem)] = h.half().cpu().numpy()
            if descriptors:
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
        dmat = (np.stack([torch.cat(desc[c]).numpy() for c in cols], 1).astype(np.float32)
                if cols else np.zeros((len(meta), 0), np.float32))
        for name, run, _, frozen in loaded:
            d = os.path.join(out_dir, name, split)
            if done != len(index):
                raise RuntimeError(f"{name} {split}: {done} windows encoded, {len(index)} expected")
            feats[name].flush()
            del feats[name]
            os.replace(os.path.join(d, "h.npy.partial"), os.path.join(d, "h.npy"))
            pred = (np.full(len(meta), np.nan, np.float32) if frozen else
                    utils.denormalize(torch.cat(preds[name]).numpy(), run.pred_stats).astype(np.float32))
            np.save(os.path.join(d, "pred.npy"), pred)
            np.save(os.path.join(d, "desc.npy"), dmat)
            meta.to_parquet(os.path.join(d, "meta.parquet"))
            with open(os.path.join(d, "desc_columns.json"), "w") as fh:
                json.dump(cols, fh)
        print(f"wrote {split}: {len(meta)} windows, {meta.recording.nunique()} recordings, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


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
