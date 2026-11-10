#!/usr/bin/env python
"""Score trained brain-age models per cohort and per source dataset.

Groups: all / normal / abnormal / unlabelled recordings, and -- for cohorts whose
manifest rows carry a corpus -- the same per dataset (``harvard`` = HEEDB,
``temple`` = TUH; ``labram/data/dataset_groups.py``), e.g. ``harvard_normal``.

For each run directory (holding ``run_config.yaml`` and a checkpoint), rebuild
the model and the exact val/test samples from the run's own config (window
selection, input length, preprocessing), predict every window, average the
predictions per recording (a MIL run, ``mil.enabled``, pools each recording's
windows with its own attention instead), and report MAE / RMSE / R^2 / Pearson r for all
recordings and for TUAB's normal and abnormal recordings separately. The
normal/abnormal label comes from the age metadata sidecar (``label`` field).

    python scripts/eval_age_by_cohort.py \\
        --data-path /data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf \\
        --run D=path/to/extracted/D --run C2-head=checkpoints/age_C2_headonly_... \\
        --out cohort_metrics.json

A SageMaker run directory is the extracted ``model.tar.gz`` (``finetune/``).
``--no-window-qc`` scores a run trained with signal-quality window rejection on
every window instead (the unfiltered benchmark). A run with session features
reads them from its ``data.session_features`` inside ``--data-path``, or from
``--session-features``.
"""
import argparse
import dataclasses
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from einops import rearrange

import labram.models.registry  # noqa: F401
import labram.utils as utils
from labram.data import get_dataset_bundle
from labram.data.tuh_metadata import load_label_lookup_for
from labram.data.staging import stage_bundle
from labram.data.window_selection import (
    WindowSelection, apply_window_selection, enable_session_features)
from labram.eval.loading import load_run_config
from labram.losses import build_downstream_criterion, regression_output, soft_label_n_bins
from labram.runs.finetune_setup import enable_window_ids
from labram.data.dataset_groups import UNKNOWN, dataset_lookup
from labram.runs.run_finetune import get_model
from labram.train.channel_drop import ChannelDropper

METRICS = ["mae", "rmse", "r2", "pearson_r"]
COHORTS = ("normal", "abnormal", "unlabelled")


def _predict(model, dataset, ch_names, device, batch_size, to_scalar=None, session=False,
             channel_drop=None, num_workers=8, logits_out=None, age_input=False):
    """Window predictions. ``logits_out`` (a list), when given, also receives
    each batch's full model output before ``to_scalar`` (e.g. every multi-task
    column)."""
    enable_window_ids(dataset, "recording")
    # Under DDP (--nproc-per-node) each rank predicts every world_size-th
    # window; the shards are gathered back in dataset order below.
    sharded = utils.get_world_size() > 1
    sampler = (torch.utils.data.DistributedSampler(dataset, shuffle=False) if sharded
               else torch.utils.data.SequentialSampler(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                         num_workers=num_workers, pin_memory=True)
    channel_indices = utils.get_channel_indices(ch_names)
    # With channel_drop, every sample runs on its present (non-zero) channels.
    dropper = ChannelDropper.from_config(channel_drop, ch_names)
    preds, targets, recs = [], [], []
    with torch.no_grad():
        for x, y, rec in loader:
            x = rearrange(x.float().to(device) / 100, "B N (A T) -> B N A T", T=200)
            extra = {}
            if session:                        # target = [age, *session features]
                extra = {"condition": y[:, 1:].float().to(device)}
                y = y[:, 0]
            elif age_input:                    # model.age_input: the z-scored age is an input
                extra = {"condition": y.float().reshape(-1, 1).to(device)}
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                if dropper is not None:
                    out = dropper.forward(model, x, dropper.present(x),
                                          condition=extra.get("condition"))
                else:
                    out = model(x, channel_indices=channel_indices, classify_only=True, **extra)
            out = getattr(out, "logits", out)
            if logits_out is not None:
                logits_out.append(out.float().cpu())
            if to_scalar is not None:          # soft-label head: bin logits -> expectation
                out = to_scalar(out)
            preds.append(out.float().squeeze(-1).cpu())
            targets.append(y.float())
            recs.extend(rec)
    pred, true = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    if sharded:
        if logits_out:                     # gathered in the same (dataset) order
            local = torch.cat(logits_out).numpy()
            full, _, _ = utils.gather_sharded_eval(local, true, recs, total=len(dataset))
            logits_out[:] = [torch.from_numpy(full)]
        pred, true, recs = utils.gather_sharded_eval(pred, true, recs, total=len(dataset))
    return pred, true, recs


def _predict_bags(model, dataset, ch_names, device, mil_cfg, batch_size=12, channel_drop=None,
                  num_workers=8):
    """Recording predictions of a MIL (bag) model: every recording's selected
    windows form one bag, pooled by the model's attention as in training's
    evaluation (``labram/train/train_mil.py::evaluate_bags``). Returns one
    (prediction, target, recording) per recording."""
    from labram.data.bags import RecordingBagDataset, collate_bags
    from labram.train.train_mil import prepare
    if utils.get_world_size() > 1:
        raise ValueError("bag models are scored on one process (--nproc-per-node 1)")
    bags = RecordingBagDataset(dataset, train=False, max_windows=mil_cfg.eval_max_windows)
    loader = torch.utils.data.DataLoader(bags, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, collate_fn=collate_bags)
    dropper = ChannelDropper.from_config(channel_drop, ch_names)
    m = getattr(model, "module", model)
    preds, targets, recs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x4, ci, bag, y, stems = prepare(batch, device, ch_names, dropper, m.mil, train=False)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                out, _, _ = model(x4, ci, bag=bag)
            preds.append(out.float().reshape(len(y), -1)[:, 0].cpu())
            targets.append(y.reshape(len(y), -1)[:, 0].float().cpu())
            recs.extend(stems)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), recs


def _per_recording(pred, true, recs, target_stats, pred_stats=None):
    """Mean prediction per recording, in years. Targets are de-normalized with
    the evaluated data's stats; predictions with the stats the model was trained
    on (``pred_stats``, default the same), which differ when a model is scored
    on another cohort than its own."""
    pred = utils.denormalize(pred, pred_stats or target_stats)
    true = utils.denormalize(true, target_stats)
    sums = OrderedDict()
    for p, t, r in zip(pred, true, recs):
        s = sums.setdefault(r, [0.0, 0, t])
        s[0] += float(p)
        s[1] += 1
    ids = list(sums)
    return ids, np.array([sums[r][0] / sums[r][1] for r in ids]), np.array([sums[r][2] for r in ids])


def _mean_by_recording(values, recs):
    """(recording ids in first-seen order, mean row of ``values`` per recording)."""
    idx = OrderedDict()
    for i, r in enumerate(recs):
        idx.setdefault(r, []).append(i)
    return list(idx), [values[i].mean(axis=0) for i in idx.values()]


def train_target_stats(cfg, train_data_path):
    """(mean, std) of the training targets of the run, rebuilt from its own
    training data with its own window selection (older run configs do not
    store them)."""
    bundle = get_dataset_bundle(cfg.data.dataset, train_data_path, data_format=cfg.data.data_format)
    return apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data)).target_stats


def logged_target_stats(run_dir):
    """(mean, std) of the training targets as the run logged them
    (``Age target normalization (selected train): mean=.. std=..`` in
    ``log/run.log``); exact for CV folds, whose config stores none."""
    import re
    path = os.path.join(run_dir, "log", "run.log")
    with open(path) as fh:
        for line in fh:
            m = re.search(r"Age target normalization.*mean=([-\d.]+) std=([-\d.]+)", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise ValueError(f"{path}: no 'Age target normalization' line")


def _window_names(dataset):
    """Window file names of ``dataset`` in loader order (leaves concatenated)."""
    from labram.data.window_selection import _leaves
    return [os.path.basename(f) for leaf in _leaves(dataset) for f in getattr(leaf, "files", [])]


def evaluate_run(run_dir, data_path, checkpoint, splits, device, batch_size, data_format=None,
                 train_data_path=None, window_qc=True, session_features=None, return_windows=False,
                 tokenizer_weight="./checkpoints/vqnsp.pth", stage_dir=None, num_workers=8,
                 class_probs=False, pred_stats=None, eval_minutes=None):
    """Score one run on ``splits``. With ``return_windows`` each split also
    carries its window-level predictions (``windows``: names, recording ids,
    predicted / true age in years) and the result the run's window settings
    (``data``), so callers can re-aggregate on a subset of windows. With
    ``class_probs`` a multi-task run's splits also carry ``class_probs``:
    recording -> the mean sigmoid probability of its windows per class column
    (status, diagnosis groups; as training's per-recording metrics), and the
    result its ``class_names``. ``pred_stats`` (mean, std) de-normalizes the
    predictions (default: from ``train_data_path``, else the scored data's own)."""
    cfg = load_run_config(os.path.join(run_dir, "run_config.yaml"))
    cfg.data.data_path = data_path
    if stage_dir is not None:                 # e.g. "none": read in place, no NVMe copy
        cfg.data.stage_dir = stage_dir
    if eval_minutes is not None:              # val/test budget per recording (0 = every window)
        cfg.data.eval_minutes = eval_minutes
    if data_format:
        cfg.data.data_format = data_format
    if cfg.model.codebook_reg.enabled:
        cfg.model.codebook_reg.tokenizer_weight = tokenizer_weight
    bundle = get_dataset_bundle(cfg.data.dataset, data_path, data_format=cfg.data.data_format)
    selection = WindowSelection.from_data_config(cfg.data)
    if not window_qc:
        selection = dataclasses.replace(selection, max_abs_uv=0.0, min_std_uv=0.0)
    bundle = apply_window_selection(bundle, selection)
    if cfg.data.drop_channels:
        from labram.data.channel_selection import drop_channels
        drop_channels(bundle, cfg.data.drop_channels)
    session = cfg.model.session_features != "off"
    if session:
        for split in splits:
            # (the run's saved config predates the groups, set from the file)
            cfg.model.session_feature_groups = enable_session_features(
                getattr(bundle, split), session_features or cfg.data.session_features,
                skip_signal=cfg.model.session_features == "only")
    if cfg.model.session_features != "only":          # a features-only model reads no EEG
        stage_bundle(bundle, cfg.data.stage_dir, reserve_gb=cfg.data.stage_reserve_gb,
                     splits=tuple(s for s in ("val", "test", "train") if s in splits))
    cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
    from labram.data.window_selection import resolve_multitask_class_names
    resolve_multitask_class_names(cfg, train_data_path or data_path)   # the training cohort holds the targets
    if pred_stats is None and train_data_path:
        pred_stats = train_target_stats(cfg, train_data_path)
    to_scalar = None
    if bundle.task == "regression" and cfg.loss.regression_loss == "soft_label":
        cfg.model.nb_classes = soft_label_n_bins(cfg.loss)
        crit = build_downstream_criterion("regression", cfg.model.nb_classes, cfg.loss,
                                          target_stats=bundle.target_stats)
        to_scalar = lambda out: regression_output(crit, out)   # noqa: E731
    elif cfg.model.head_type == "multitask":
        # The head emits [age, abnormal logit]; age metrics score the age column
        # (losses/regression.py::MultiTaskAgeCriterion.predict).
        to_scalar = lambda out: out[..., :1]   # noqa: E731
    model = get_model(cfg)
    state = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    # A run that selected its best epoch on the EMA weights is scored with them.
    key = "model_ema" if (cfg.evaluation.use_ema and "model_ema" in state) else "model"
    model.load_state_dict(state[key])
    model.to(device).eval()

    root = getattr(bundle.test, "root", data_path)
    labels = load_label_lookup_for(root)
    result = {"epoch": state.get("epoch"), "splits": {},
              "pred_target_stats": pred_stats or bundle.target_stats,
              "data": {"eval_minutes": cfg.data.eval_minutes, "window_sec": cfg.data.window_sec,
                       "trim_start_sec": cfg.data.trim_start_sec, "trim_end_sec": cfg.data.trim_end_sec}}
    keep_logits = class_probs and cfg.model.head_type == "multitask"
    if keep_logits:
        result["class_names"] = list(cfg.model.multitask_head.class_names or [])
    for split in splits:
        logits = [] if keep_logits else None
        if getattr(cfg, "mil", None) is not None and cfg.mil.enabled:
            if return_windows or keep_logits or session:
                raise ValueError("bag (mil.enabled) runs are scored per recording only")
            pred, true, recs = _predict_bags(model, getattr(bundle, split), bundle.ch_names, device,
                                             cfg.mil, channel_drop=getattr(cfg, "channel_drop", None),
                                             num_workers=num_workers)
        else:
            pred, true, recs = _predict(model, getattr(bundle, split), bundle.ch_names, device,
                                        batch_size, to_scalar, session=session,
                                        channel_drop=getattr(cfg, "channel_drop", None),
                                        num_workers=num_workers, logits_out=logits,
                                        age_input=getattr(cfg.model, "age_input", "off") != "off")
        ids, p, t = _per_recording(pred, true, recs, bundle.target_stats, pred_stats)
        # group ids are recording stems; no label = unlabelled (e.g. HEEDB without a report)
        cohort = np.array([labels.get(r) if labels.get(r) in ("normal", "abnormal")
                           else "unlabelled" for r in ids])
        source = dataset_lookup(getattr(bundle, split))
        family = np.array([source.get(r, UNKNOWN) for r in ids])
        groups = [("all", np.ones(len(ids), bool))] + [(c, cohort == c) for c in COHORTS]
        if source:                         # per dataset (harvard / temple) x cohort
            for ds in sorted(set(family)):
                groups += [(ds, family == ds)] + [(f"{ds}_{c}", (family == ds) & (cohort == c))
                                                  for c in COHORTS]
        out = {}
        for name, mask in groups:
            if mask.sum() < 2:
                continue
            m = utils.regression_metrics_fn(p[mask], t[mask], METRICS)
            out[name] = {"n": int(mask.sum()), **{k: float(v) for k, v in m.items()}}
        out["predictions"] = {r: float(v) for r, v in zip(ids, p)}
        if keep_logits:                        # columns 1.. are the class logits
            prob = torch.sigmoid(torch.cat(logits)[:, 1:]).numpy()
            out["class_probs"] = dict(zip(*_mean_by_recording(prob, recs)))
        if return_windows:
            names = _window_names(getattr(bundle, split))
            if len(names) != len(recs):
                raise RuntimeError(f"{split}: {len(names)} window names for {len(recs)} predictions")
            out["windows"] = {"names": names, "recs": list(recs),
                              "pred": utils.denormalize(pred, pred_stats or bundle.target_stats),
                              "true": utils.denormalize(true, bundle.target_stats),
                              "labels": [labels.get(r) for r in recs]}
        result["splits"][split] = out
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--checkpoint", default="checkpoint-best.pth")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-format", choices=["pickle", "npy"], default=None,
                    help="override the run's data.data_format (e.g. to compare formats)")
    ap.add_argument("--train-data-path", default=None,
                    help="the run's own training data, when --data-path is another cohort: "
                         "predictions are de-normalized with its training-target stats")
    ap.add_argument("--pred-stats-from-log", action="store_true",
                    help="de-normalize predictions with the training-target stats each run logged "
                         "(log/run.log; exact for CV folds) -- for scoring on another cohort")
    ap.add_argument("--pred-stats", nargs=2, type=float, default=None, metavar=("MEAN", "STD"),
                    help="de-normalize predictions with these training-target stats (years)")
    ap.add_argument("--stage-dir", default=None, help="override data.stage_dir (e.g. 'none')")
    ap.add_argument("--no-window-qc", action="store_true",
                    help="ignore the run's signal-quality window rejection: score every window")
    ap.add_argument("--session-features", default=None,
                    help="session-feature file for --data-path (default: the run's own)")
    ap.add_argument("--eval-minutes", type=float, default=None,
                    help="override data.eval_minutes, the per-recording val/test budget (0 = every window)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nproc-per-node", type=int, default=1,
                    help="GPUs to predict on (one DDP process each; -1 = all visible GPUs). "
                         "Windows are sharded across processes and gathered on rank 0.")
    args = ap.parse_args()
    from labram.utils.launch import relaunch_if_needed
    relaunch_if_needed(args.nproc_per_node, script=os.path.abspath(__file__))
    ns = argparse.Namespace(dist_on_itp=False, dist_url="env://", world_size=1, local_rank=-1,
                            backend="auto", device="auto")
    utils.init_distributed_mode(ns)
    if getattr(ns, "distributed", False) and torch.cuda.is_available():
        device = torch.device("cuda", ns.gpu)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for item in args.run:
        name, run_dir = item.split("=", 1)
        results[name] = evaluate_run(run_dir, args.data_path, args.checkpoint, args.splits,
                                     device, args.batch_size, args.data_format,
                                     train_data_path=args.train_data_path,
                                     window_qc=not args.no_window_qc,
                                     session_features=args.session_features,
                                     pred_stats=(tuple(args.pred_stats) if args.pred_stats else
                                                 logged_target_stats(run_dir) if args.pred_stats_from_log else None),
                                     stage_dir=args.stage_dir, eval_minutes=args.eval_minutes)
        if not utils.is_main_process():
            continue
        for split, cohorts in results[name]["splits"].items():
            print(f"{name:14s} {split:4s} " + "  ".join(
                f"{c}: n={v['n']} MAE={v['mae']:.2f} R2={v['r2']:.2f}" for c, v in cohorts.items()
                if c != "predictions"))
    if args.out and utils.is_main_process():
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    utils.cleanup_distributed()


if __name__ == "__main__":
    main()
