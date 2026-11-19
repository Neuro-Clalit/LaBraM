#!/usr/bin/env python
"""External-test brain-age metrics for chosen models (and their median ensemble).

Reads ``scripts/eval_age_by_cohort.py`` outputs on an external cohort (every
run's per-recording predictions) and the cohort's ``age_metadata.json``, and
reports per model: n, MAE with a 95 % bootstrap CI, RMSE, R^2, Pearson r, bias,
the slope of the error over age, the two baselines (the cohort's own mean age
and the training mean age), MAE per age bin, and the bias-corrected MAE.

The bias correction is the linear age trend of the error,
``pred - age = a * age + b``, fitted on the model's **own validation
predictions** (``--val name=json``, normal-report recordings of
``--val-meta``), never on the external cohort; the corrected prediction is
``pred - (a * age + b)``. ``ensemble`` is the per-recording median over the
chosen models (and over its fits for the correction).

``--cv-split`` (a ``cv_split.json`` of ``labram.runs.finetune_cv``) adds a
within-cohort comparison on exactly those folds: for each test fold *k* the
linear recalibration ``age = c * pred + d`` is fitted (OLS) on the recordings of
every other fold and applied to fold *k*; reported are the pooled out-of-fold
MAE / R^2 / r and the MAE mean +- std over folds. It uses the cohort's ages, so
it is a within-cohort CV result, comparable with published within-cohort CV.

    python scripts/external_age_metrics.py \\
        --eval /data/runs/heedb/external_eval/chbp_5min.json \\
        --meta /data/datasets/EEG-public/CHBP/cohorts/chbp_v1/age_metadata.json \\
        --models E2L L1 W4 W7 --train-mean 53.52 \\
        --val E2L=/data/runs/heedb/external_eval/own_all_val_E2L.json ... \\
        --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \\
        --bins 18 30 40 69 --out metrics.csv
"""
import argparse
import json

import numpy as np
import pandas as pd


def predictions(path, models, split="test"):
    """``DataFrame`` stem x model of per-recording predicted ages."""
    runs = json.load(open(path))
    return pd.DataFrame({m: pd.Series(runs[m]["splits"][split]["predictions"]) for m in models})


def age_metrics(y, yhat, train_mean=None, n_boot=2000, seed=0):
    err = yhat - y
    boot = np.abs(err[np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))]).mean(axis=1)
    out = {"n": len(y), "mae": np.abs(err).mean(),
           "mae_ci_lo": np.percentile(boot, 2.5), "mae_ci_hi": np.percentile(boot, 97.5),
           "rmse": np.sqrt((err ** 2).mean()),
           "r2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
           "r": np.corrcoef(y, yhat)[0, 1], "bias": err.mean(),
           "err_slope": np.polyfit(y, err, 1)[0],
           "mae_base_cohort_mean": np.abs(y - y.mean()).mean()}
    if train_mean is not None:
        out["mae_base_train_mean"] = np.abs(y - train_mean).mean()
    return out


def cv_folds(path, stems):
    """Fold index per stem from a ``cv_split.json`` (groups are subjects = stem prefix)."""
    split = json.load(open(path))
    of = {g: f["fold"] for f in split["folds"] for g in f["groups"]}
    return np.array([of[s.split("_")[0]] for s in stems])


def fold_r2(y, yhat):
    """R^2 within one test fold (as Engemann et al. 2022 report it, then averaged)."""
    return 1 - ((yhat - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cv_recalibration(y, yhat, fold):
    """Out-of-fold ``c * pred + d`` (OLS of age on pred over the other folds)."""
    out = np.empty_like(yhat)
    for k in np.unique(fold):
        c, d = np.polyfit(yhat[fold != k], y[fold != k], 1)
        out[fold == k] = c * yhat[fold == k] + d
    per_fold = [np.abs(out[fold == k] - y[fold == k]).mean() for k in np.unique(fold)]
    r2_fold = [fold_r2(y[fold == k], out[fold == k]) for k in np.unique(fold)]
    err = out - y
    return {"cv_mae": np.abs(err).mean(), "cv_mae_fold_mean": np.mean(per_fold),
            "cv_mae_fold_std": np.std(per_fold, ddof=1),
            "cv_r2_fold_mean": np.mean(r2_fold), "cv_r2_fold_std": np.std(r2_fold, ddof=1),
            "cv_r2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "cv_r": np.corrcoef(y, out)[0, 1]}


def gap_trend(val_json, name, val_ages):
    """(a, b) of ``pred - age = a * age + b`` on the model's own normal val recordings."""
    p = pd.Series(json.load(open(val_json))[name]["splits"]["val"]["predictions"])
    p = p[p.index.isin(val_ages.index)]
    age = val_ages.loc[p.index].to_numpy(float)
    return np.polyfit(age, p.to_numpy() - age, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--train-mean", type=float, default=None)
    ap.add_argument("--val", action="append", default=[], metavar="NAME=JSON")
    ap.add_argument("--val-meta", default=None)
    ap.add_argument("--cv-split", default=None, help="cv_split.json for within-cohort CV recalibration")
    ap.add_argument("--bins", nargs="+", type=float, default=[18, 30, 40, 69])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(args.meta))["recordings"]
    pred = predictions(args.eval, args.models)
    age = pd.Series({s: meta[s]["age"] for s in pred.index}, dtype=float)
    pred["ensemble"] = pred[args.models].median(axis=1)

    trend = {}
    if args.val:
        vm = json.load(open(args.val_meta))["recordings"]
        val_ages = pd.Series({s: r["age"] for s, r in vm.items() if r.get("label") == "normal"}, dtype=float)
        for item in args.val:
            name, path = item.split("=", 1)
            trend[name] = gap_trend(path, name, val_ages)
        if all(m in trend for m in args.models):
            trend["ensemble"] = np.median([trend[m] for m in args.models], axis=0)

    fold = cv_folds(args.cv_split, pred.index) if args.cv_split else None
    bins = pd.cut(age, args.bins, right=False)
    rows = []
    y = age.to_numpy()
    for model in pred.columns:
        yhat = pred[model].to_numpy()
        row = {"model": model, **age_metrics(y, yhat, args.train_mean)}
        if model in trend:
            a, b = trend[model]
            corr = yhat - (a * y + b)
            row.update({"trend_a": a, "trend_b": b, "mae_corrected": np.abs(corr - y).mean(),
                        "bias_corrected": (corr - y).mean()})
        if fold is not None:
            row.update(cv_recalibration(y, yhat, fold))
        for interval, mask in bins.groupby(bins, observed=True).groups.items():
            sel = age.index.get_indexer(mask)
            e = yhat[sel] - y[sel]
            label = f"{int(interval.left)}-{int(interval.right) - 1}"
            row[f"mae_{label}"] = np.abs(e).mean()
            row[f"bias_{label}"] = e.mean()
            row[f"n_{label}"] = len(sel)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.round(2).T.to_string())
    if args.out:
        df.to_csv(args.out)


if __name__ == "__main__":
    main()
