#!/usr/bin/env python
"""Re-score trained brain-age runs with artifact-aware window pooling.

Muscle (EMG) artifact reads as "young": the worst under-predictions are
recordings full of high-frequency muscle activity. This script drops or
down-weights EMG-heavy windows before a recording's windows are averaged, and
reports case MAE over a grid of thresholds. The model is not re-run: window
predictions come from each run's ``analysis/window_predictions.parquet`` cache
(written by ``notebooks/age_error_analysis.ipynb``; ``--predict`` computes it
when missing, which needs a GPU). Per-window EMG features are computed from the
run's own val/test windows once and cached as
``analysis/window_artifacts.parquet``.

The threshold is a quantile of the EMG index over val windows and is chosen on
val; read its test MAE as the honest estimate.

    python scripts/age_artifact_pooling.py \\
        --data-path /data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf \\
        --run M1=checkpoints/age_M1_... --run M2=path/to/extracted/M2 \\
        --out artifact_pooling.json
"""
import argparse
import json
import os

import pandas as pd

from labram.eval.age_analysis import (
    POOLING_METHODS, artifact_pooling_sweep, load_age_bundle, select_pooling,
    window_artifact_features)

SPLITS = ("val", "test")


def window_predictions(run_dir, data_path, predict):
    path = os.path.join(run_dir, "analysis", "window_predictions.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if set(SPLITS) <= set(df["split"]):
            return df[df["split"].isin(SPLITS)].copy()
    if not predict:
        raise SystemExit(f"{path}: no val/test window predictions; run the analysis "
                         "notebook on this run first, or pass --predict")
    from labram.eval.age_plots import cached_window_predictions
    return cached_window_predictions(run_dir, data_path, splits=SPLITS)


def artifact_features(run_dir, data_path, windows):
    path = os.path.join(run_dir, "analysis", "window_artifacts.parquet")
    if os.path.exists(path):
        feats = pd.read_parquet(path)
        if len(windows[["split", "idx"]].merge(feats, on=["split", "idx"])) == len(windows):
            return feats
    _, bundle = load_age_bundle(run_dir, data_path)
    feats = pd.concat([window_artifact_features(getattr(bundle, s), windows[windows.split == s])
                       for s in SPLITS], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feats.to_parquet(path)
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--score", default="emg", choices=("emg", "emg_max"),
                    help="per-window artifact index (channel mean or max of 30-45 Hz power)")
    ap.add_argument("--methods", nargs="+", default=list(POOLING_METHODS), choices=POOLING_METHODS)
    ap.add_argument("--min-keep", type=int, default=1,
                    help="cleanest windows every recording keeps whatever their score")
    ap.add_argument("--predict", action="store_true",
                    help="score the model when the window-prediction cache is missing")
    ap.add_argument("--out", help="write the full sweep and the selections as JSON")
    args = ap.parse_args()

    report, sweeps = {}, []
    for spec in args.run:
        name, run_dir = spec.split("=", 1)
        win = window_predictions(run_dir, args.data_path, args.predict)
        win = win.merge(artifact_features(run_dir, args.data_path, win), on=["split", "idx"])
        sweep = artifact_pooling_sweep(win, score=args.score, methods=args.methods,
                                       min_keep=args.min_keep).assign(run=name)
        sweeps.append(sweep)
        base = sweep[(sweep["quantile"] == 1.0) & (sweep.method == args.methods[0])
                     & (sweep.cohort == "all")].set_index("split")["mae"]
        best = select_pooling(sweep)
        report[name] = {"baseline": base.to_dict(), "selected": best.to_dict()}
        print(f"{name}: mean pooling val {base['val']:.2f} / test {base['test']:.2f}  ->  "
              f"{best['method']} q={best['quantile']:.2f} "
              f"val {best['mae_val']:.2f} / test {best['mae_test']:.2f}")

    sweep = pd.concat(sweeps, ignore_index=True)
    table = sweep[sweep.cohort == "all"].pivot_table(
        index=["run", "method", "quantile"], columns="split", values=["mae", "frac_kept"])
    with pd.option_context("display.width", 160, "display.max_rows", 500):
        print(table.round(3))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"score": args.score, "min_keep": args.min_keep, "runs": report,
                       "sweep": sweep.to_dict(orient="records")}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
