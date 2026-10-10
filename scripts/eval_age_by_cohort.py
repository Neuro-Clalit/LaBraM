#!/usr/bin/env python
"""Score trained brain-age models per cohort: all / normal / abnormal recordings.

For each run directory (holding ``run_config.yaml`` and a checkpoint), rebuild
the model and the exact val/test samples from the run's own config (window
selection, input length, preprocessing), predict every window, average the
predictions per recording, and report MAE / RMSE / R^2 / Pearson r for all
recordings and for TUAB's normal and abnormal recordings separately. The
normal/abnormal label comes from the age metadata sidecar (``label`` field).

    python scripts/eval_age_by_cohort.py \\
        --data-path /data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf \\
        --run D=path/to/extracted/D --run C2-head=checkpoints/age_C2_headonly_... \\
        --out cohort_metrics.json

A SageMaker run directory is the extracted ``model.tar.gz`` (``finetune/``).
"""
import argparse
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
from labram.data.window_selection import WindowSelection, apply_window_selection
from labram.eval.loading import load_run_config
from labram.losses import build_downstream_criterion, regression_output, soft_label_n_bins
from labram.runs.finetune_setup import enable_window_ids
from labram.runs.run_finetune import get_model

METRICS = ["mae", "rmse", "r2", "pearson_r"]


def _predict(model, dataset, ch_names, device, batch_size, to_scalar=None):
    enable_window_ids(dataset, "recording")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=8, pin_memory=True)
    channel_indices = utils.get_channel_indices(ch_names)
    preds, targets, recs = [], [], []
    with torch.no_grad():
        for x, y, rec in loader:
            x = rearrange(x.float().to(device) / 100, "B N (A T) -> B N A T", T=200)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x, channel_indices=channel_indices, classify_only=True)
            out = getattr(out, "logits", out)
            if to_scalar is not None:          # soft-label head: bin logits -> expectation
                out = to_scalar(out)
            preds.append(out.float().squeeze(-1).cpu())
            targets.append(y.float())
            recs.extend(rec)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), recs


def _per_recording(pred, true, recs, target_stats):
    """Mean prediction per recording, in years."""
    pred, true = utils.denormalize(pred, target_stats), utils.denormalize(true, target_stats)
    sums = OrderedDict()
    for p, t, r in zip(pred, true, recs):
        s = sums.setdefault(r, [0.0, 0, t])
        s[0] += float(p)
        s[1] += 1
    ids = list(sums)
    return ids, np.array([sums[r][0] / sums[r][1] for r in ids]), np.array([sums[r][2] for r in ids])


def evaluate_run(run_dir, data_path, checkpoint, splits, device, batch_size, data_format=None):
    cfg = load_run_config(os.path.join(run_dir, "run_config.yaml"))
    cfg.data.data_path = data_path
    if data_format:
        cfg.data.data_format = data_format
    if cfg.model.codebook_reg.enabled:
        cfg.model.codebook_reg.tokenizer_weight = "./checkpoints/vqnsp.pth"
    bundle = get_dataset_bundle(cfg.data.dataset, data_path, data_format=cfg.data.data_format)
    bundle = apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))
    cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
    to_scalar = None
    if bundle.task == "regression" and cfg.loss.regression_loss == "soft_label":
        cfg.model.nb_classes = soft_label_n_bins(cfg.loss)
        crit = build_downstream_criterion("regression", cfg.model.nb_classes, cfg.loss,
                                          target_stats=bundle.target_stats)
        to_scalar = lambda out: regression_output(crit, out)   # noqa: E731
    model = get_model(cfg)
    state = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    # A run that selected its best epoch on the EMA weights is scored with them.
    key = "model_ema" if (cfg.evaluation.use_ema and "model_ema" in state) else "model"
    model.load_state_dict(state[key])
    model.to(device).eval()

    root = getattr(bundle.test, "root", data_path)
    labels = load_label_lookup_for(root)
    result = {"epoch": state.get("epoch"), "splits": {}}
    for split in splits:
        pred, true, recs = _predict(model, getattr(bundle, split), bundle.ch_names, device,
                                    batch_size, to_scalar)
        ids, p, t = _per_recording(pred, true, recs, bundle.target_stats)
        cohort = np.array([labels.get(r) for r in ids])   # group ids are recording stems
        out = {}
        for name, mask in (("all", np.ones(len(ids), bool)), ("normal", cohort == "normal"),
                           ("abnormal", cohort == "abnormal")):
            m = utils.regression_metrics_fn(p[mask], t[mask], METRICS)
            out[name] = {"n": int(mask.sum()), **{k: float(v) for k, v in m.items()}}
        out["predictions"] = {r: float(v) for r, v in zip(ids, p)}
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for item in args.run:
        name, run_dir = item.split("=", 1)
        results[name] = evaluate_run(run_dir, args.data_path, args.checkpoint, args.splits,
                                     device, args.batch_size, args.data_format)
        for split, cohorts in results[name]["splits"].items():
            print(f"{name:14s} {split:4s} " + "  ".join(
                f"{c}: n={v['n']} MAE={v['mae']:.2f} R2={v['r2']:.2f}" for c, v in cohorts.items()
                if c != "predictions"))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
