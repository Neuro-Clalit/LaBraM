#!/usr/bin/env python
"""Within-cohort CV ridge probe on frozen brain-age embeddings.

Reads the window embeddings of trained runs (``scripts/mil_cache.py --splits
test`` on the cohort: ``<cache>/<run>/test/{h.npy,meta.parquet}``), averages
them per recording, and runs ridge regression of age on the mean embedding
over the folds of a ``cv_split.json`` (groups = subjects = stem prefix). For
each test fold the scaler and ``RidgeCV`` (alpha chosen by efficient LOO on the
training folds only) are fitted on every other fold. A ``dummy`` row predicts
the training folds' mean age. Reports, as Engemann et al. 2022 do, the MAE and
R^2 mean +- std over folds (R^2 within each test fold), and the pooled
out-of-fold MAE / R^2 / r and the MAE mean +- std over folds; writes the
out-of-fold predictions.

    python scripts/external_embedding_probe.py --cache /data/runs/chbp_cv/emb_cache \\
        --runs E2 E2L --cv-split .../cv_split.json \\
        --meta /data/datasets/EEG-public/CHBP/cohorts/chbp_v1/age_metadata.json --out probe.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 5, 25)


def recording_embeddings(cache, run):
    h = np.load(os.path.join(cache, run, "test", "h.npy"))
    meta = pd.read_parquet(os.path.join(cache, run, "test", "meta.parquet"))
    rec = meta[[c for c in ("recording", "rec", "stem") if c in meta.columns][0]].to_numpy()
    df = pd.DataFrame(h.astype(np.float64))
    df["rec"] = rec
    return df.groupby("rec").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cv-split", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = json.load(open(args.meta))["recordings"]
    split = json.load(open(args.cv_split))
    fold_of = {g: f["fold"] for f in split["folds"] for g in f["groups"]}
    rows, oof = [], {}
    for run in ["dummy"] + args.runs:
        X = recording_embeddings(args.cache, args.runs[0] if run == "dummy" else run)
        stems = X.index.to_numpy()
        y = np.array([meta[s]["age"] for s in stems], float)
        fold = np.array([fold_of[s.split("_")[0]] for s in stems])
        pred, alphas = np.empty_like(y), []
        for k in np.unique(fold):
            if run == "dummy":
                pred[fold == k] = y[fold != k].mean()
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(X.to_numpy()[fold != k], y[fold != k])
            pred[fold == k] = model.predict(X.to_numpy()[fold == k])
            alphas.append(model[-1].alpha_)
        err = pred - y
        per_fold = [np.abs(err[fold == k]).mean() for k in np.unique(fold)]
        r2_fold = [1 - (err[fold == k] ** 2).sum() / ((y[fold == k] - y[fold == k].mean()) ** 2).sum()
                   for k in np.unique(fold)]
        rows.append({"run": run if run == "dummy" else f"{run} ridge probe", "n": len(y), "dim": X.shape[1],
                     "cv_mae": np.abs(err).mean(), "cv_mae_fold_mean": np.mean(per_fold),
                     "cv_mae_fold_std": np.std(per_fold, ddof=1),
                     "cv_r2_fold_mean": np.mean(r2_fold), "cv_r2_fold_std": np.std(r2_fold, ddof=1),
                     "cv_r2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "cv_r": np.corrcoef(y, pred)[0, 1] if run != "dummy" else np.nan,
                     "alpha_median": np.median(alphas) if alphas else np.nan})
        oof[run] = pd.Series(pred, index=stems)
    df = pd.DataFrame(rows).set_index("run")
    print(df.round(3).to_string())
    if args.out:
        df.to_csv(args.out)
        pd.DataFrame(oof).to_csv(os.path.splitext(args.out)[0] + "_oof.csv")


if __name__ == "__main__":
    main()
