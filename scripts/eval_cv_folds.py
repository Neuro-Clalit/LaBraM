#!/usr/bin/env python
"""Score cross-validation fold models on their own val/test folds, per source.

For each fold run directory (``.../cv/fold_<k>`` with ``run_config.yaml``,
``cv_split.json`` and a checkpoint) rebuild the fold exactly as training did
-- ``apply_cv_split`` + ``materialize_fold`` on ``--data-path``, the run's
window selection (whose selected training targets give the age z-scoring
stats), session features and channel handling -- predict every window of the
val and test folds, average per recording, and report n / MAE / RMSE / R² per
fold x split x source (TUAB / TUEG, from the TUH index) x cohort
(all / normal / abnormal); then mean ± std over folds.

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
from eval_age_by_cohort import _per_recording, _predict  # noqa: E402

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
    args = p.parse_args(argv)
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
        bundle = apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))
        cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
        session = cfg.model.session_features != "off"
        if session:
            for s in ("val", "test"):
                cfg.model.session_feature_groups = enable_session_features(
                    getattr(bundle, s), cfg.data.session_features,
                    skip_signal=cfg.model.session_features == "only")
        model = get_model(cfg)
        state = torch.load(os.path.join(fold_dir, args.checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        for s in ("val", "test"):
            pred, true, recs = _predict(model, getattr(bundle, s), bundle.ch_names, device,
                                        args.batch_size, session=session,
                                        channel_drop=getattr(cfg, "channel_drop", None))
            for r in score(pred, true, recs, labels, tuab, bundle.target_stats):
                rows.append({"exp": name, "fold": k, "epoch": state.get("epoch"), "split": s, **r})
        print(f"{name} fold {k}: " + "  ".join(
            f"{r['split']} {r['source']} {r['MAE']:.2f}" for r in rows[-18:] if r["cohort"] == "all"), flush=True)
    d = pd.DataFrame(rows)
    if args.out:
        d.to_csv(args.out, index=False)
    agg = d.groupby(["exp", "split", "source", "cohort"]).agg(
        folds=("fold", "nunique"), n=("n", "mean"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
