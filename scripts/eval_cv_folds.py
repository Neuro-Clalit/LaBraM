#!/usr/bin/env python
"""Score cross-validation fold models on their own val/test folds, per source.

For each fold run directory (``.../cv/fold_<k>`` with ``run_config.yaml``,
``cv_split.json`` and a checkpoint) rebuild the fold exactly as training did
-- ``apply_cv_split`` + ``materialize_fold`` on ``--data-path``, the run's
window selection (whose selected training targets give the age z-scoring
stats), session features and channel handling -- predict every window of the
val and test folds, average per recording, and report n / MAE / RMSE / R² per
fold x split x source (TUAB / TUEG, from the TUH index) x cohort
(all / normal / abnormal); then mean ± std over folds. Split ``heldout`` is the
cohort's own test set, which CV with pool=train_val leaves out (the official
TUAB test set): every fold model is scored on the same recordings, with that
fold's training-target stats. A bag (MIL, ``mil.enabled``) fold model pools each
recording's windows with its own attention, as in training's evaluation;
``--save-windows DIR`` then also writes every window's attention weight and
own prediction (``<DIR>/<name>_f<k>_<split>.parquet``).

    python scripts/eval_cv_folds.py --data-path .../cohorts/all_both_v3_cvcompact \\
        --run cvbase=/tmp/cv/cvbase_{0..4}/cv/fold_{0..4} ... --out cv_eval.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_age_by_cohort import _per_recording, _predict, _predict_bags  # noqa: E402

import labram.models.registry  # noqa: F401,E402
from labram.data import get_dataset_bundle  # noqa: E402
from labram.data.cross_validation import apply_cv_split, materialize_fold  # noqa: E402
from labram.data.tuh_metadata import load_label_lookup_for  # noqa: E402
from labram.data.window_selection import (  # noqa: E402
    WindowSelection, apply_window_selection, enable_session_features)
from labram.eval.loading import load_run_config  # noqa: E402
from labram.runs.run_finetune import get_model  # noqa: E402

INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"


def score(pred, true, recs, labels, tuab, target_stats):
    ids, p, t = _per_recording(pred, true, recs, target_stats)
    src = np.array(["TUAB" if r in tuab else "TUEG" for r in ids])
    lab = np.array([labels.get(r) for r in ids])
    rows = []
    for source in ("all", "TUAB", "TUEG"):
        for cohort in ("all", "normal", "abnormal"):
            m = np.ones(len(ids), bool)
            if source != "all":
                m &= src == source
            if cohort != "all":
                m &= lab == cohort
            if m.sum() < 2:
                continue
            e = p[m] - t[m]
            rows.append({"source": source, "cohort": cohort, "n": int(m.sum()),
                         "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                         "R2": float(1 - (e ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum())})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=FOLD_DIR")
    p.add_argument("--checkpoint", default="checkpoint-best.pth")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None)
    p.add_argument("--splits", nargs="+", default=["val", "test", "heldout"])
    p.add_argument("--preds-out", default=None,
                   help="per-recording predictions (years) per exp/fold/split, JSON")
    p.add_argument("--save-windows", default=None, metavar="DIR",
                   help="bag models: per-window attention weight and prediction, one parquet per fold and split")
    p.add_argument("--full-windows", action="store_true",
                   help="score the val fold on every window of its recordings (a training split may list a "
                        "subset), so val is scored exactly like the held-out test split")
    p.add_argument("--bag-batch-size", type=int, default=12, help="bag models: recordings per batch")
    p.add_argument("--num-workers", type=int, default=8, help="bag models: data-loader workers")
    args = p.parse_args(argv)
    preds = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = pd.read_parquet(INDEX, columns=["stem", "corpora"])
    tuab = set(idx.loc[idx.corpora.str.contains("tuab"), "stem"])
    labels = load_label_lookup_for(args.data_path)
    rows = []
    for item in args.run:
        name, fold_dir = item.split("=", 1)
        cfg = load_run_config(os.path.join(fold_dir, "run_config.yaml"))
        k = int(cfg.cross_validation.fold)
        split = json.load(open(os.path.join(fold_dir, "cv_split.json")))
        base = get_dataset_bundle(cfg.data.dataset, args.data_path, data_format="npy")
        bundle = materialize_fold(apply_cv_split(base, split), k, base)
        if args.full_windows:                   # every window of the val recordings, as the test split lists them
            from labram.data.tuh_metadata import recording_stem
            from labram.data.window_selection import _leaves
            for leaf in _leaves(bundle.val):
                n_windows, _ = leaf.window_inventory()
                recs = sorted({(os.path.dirname(f), recording_stem(f)) for f in leaf.files})
                leaf.files = [f"{d}/{stem}_{i}.pkl" for d, stem in recs for i in range(n_windows[(d, stem)])]
        bundle = apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))
        held = apply_window_selection(
            get_dataset_bundle(cfg.data.dataset, args.data_path, data_format="npy"),
            WindowSelection.from_data_config(cfg.data)).test
        for leaf in getattr(held, "datasets", [held]):
            if hasattr(leaf, "target_stats"):
                leaf.target_stats = bundle.target_stats        # the fold's z-scoring
        if cfg.data.drop_channels:              # as training served them (run_finetune.main)
            from labram.data.channel_selection import drop_channels
            from labram.data.window_selection import _leaves
            dropped = {n.upper() for n in cfg.data.drop_channels}
            keep = np.asarray([i for i, c in enumerate(bundle.ch_names) if c.upper() not in dropped])
            drop_channels(bundle, cfg.data.drop_channels)
            for leaf in _leaves(held):          # the separately built held-out split too
                leaf.keep_channels = keep
        bundle.heldout = held
        cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
        from labram.data.window_selection import resolve_multitask_class_names
        resolve_multitask_class_names(cfg, args.data_path)
        session = cfg.model.session_features != "off"
        if session:
            for s in args.splits:
                cfg.model.session_feature_groups = enable_session_features(
                    getattr(bundle, s), cfg.data.session_features,
                    skip_signal=cfg.model.session_features == "only")
        model = get_model(cfg)
        state = torch.load(os.path.join(fold_dir, args.checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        # checkpoint-best.pth stores epoch="best"; the epoch index is in train_state.json
        ts = os.path.join(fold_dir, "train_state.json")
        best_epoch = json.load(open(ts)).get("best_epoch") if os.path.exists(ts) else state.get("epoch")
        mil = getattr(cfg, "mil", None) is not None and cfg.mil.enabled
        for s in args.splits:
            if mil:                             # one attention-pooled prediction per recording
                win = {} if args.save_windows else None
                pred, true, recs = _predict_bags(model, getattr(bundle, s), bundle.ch_names, device, cfg.mil,
                                                 batch_size=args.bag_batch_size,
                                                 channel_drop=getattr(cfg, "channel_drop", None),
                                                 num_workers=args.num_workers, windows_out=win)
                if win:
                    import labram.utils as utils
                    wdf = pd.DataFrame(win)
                    wdf["window_pred"] = utils.denormalize(wdf["window_pred"].to_numpy(), bundle.target_stats)
                    wdf["age"] = wdf["recording"].map(dict(zip(recs, utils.denormalize(true, bundle.target_stats))))
                    os.makedirs(args.save_windows, exist_ok=True)
                    wdf.to_parquet(os.path.join(args.save_windows, f"{name}_f{k}_{s}.parquet"))
            else:
                pred, true, recs = _predict(model, getattr(bundle, s), bundle.ch_names, device,
                                            args.batch_size, session=session,
                                            channel_drop=getattr(cfg, "channel_drop", None))
            ids, py, ty = _per_recording(pred, true, recs, bundle.target_stats)
            preds.setdefault(name, {}).setdefault(str(k), {})[s] = {
                r: [float(a), float(b)] for r, a, b in zip(ids, py, ty)}
            for r in score(pred, true, recs, labels, tuab, bundle.target_stats):
                rows.append({"exp": name, "fold": k, "epoch": best_epoch, "split": s, **r})
        print(f"{name} fold {k}: " + "  ".join(
            f"{r['split']} {r['source']} {r['MAE']:.2f}" for r in rows
            if r["exp"] == name and r["fold"] == k and r["cohort"] == "all"), flush=True)
    d = pd.DataFrame(rows)
    if args.preds_out:
        json.dump(preds, open(args.preds_out, "w"))
    if args.out:
        d.to_csv(args.out, index=False)
    agg = d.groupby(["exp", "split", "source", "cohort"]).agg(
        folds=("fold", "nunique"), n=("n", "mean"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
