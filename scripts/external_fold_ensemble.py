#!/usr/bin/env python
"""Cross-validation fold models and their ensemble on an external cohort.

Reads ``eval_age_by_cohort.py`` outputs holding the fold models of one or more
CV studies (run names ``<study>_f<k>``, e.g. ``P3_L1_f0`` ... ``P3_L1_f4``) and
writes, per study: every fold's metrics, their mean +- SD over folds, and the
fold ensemble, whose prediction for recording ``i`` is the median over the
``K`` fold models, ``p_i = median_k p_i^(k)`` (``--agg mean`` for the mean).
``--also`` adds single models (e.g. E2L) as rows for comparison, and
``--ensemble-of`` an ensemble over several studies' folds together.

Metrics as ``external_eval_report.metrics`` (years): MAE, bias (mean of
p - y), Pearson r, R^2 about the cohort's own mean age; plus the mean and SD of
the predictions and the predict-the-cohort-mean MAE. ``--min-age`` /
``--max-age`` restrict the recordings (adults: ``--min-age 18``).

    python scripts/external_fold_ensemble.py \\
        --cohort /data/datasets/EEG-public/HBN/cohorts/hbn_adults_v1 \\
        --eval hbn_adults_v1_all_p34_cv.json hbn_adults_v1_all_p34_cv_f3.json hbn_adults_v1_all.json \\
        --studies P3_L1 P4_L1 --also E2L --min-age 18 --out docs/data/external_eval/hbn/p34_cv_all.csv
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_cohort_comparison import load  # noqa: E402
from external_eval_report import metrics  # noqa: E402


def fold_runs(preds, study):
    """``{k: predictions}`` of the runs named ``<study>_f<k>``."""
    pat = re.compile(re.escape(study) + r"_f(\d+)$")
    return {int(m[1]): p for name, p in preds.items() if (m := pat.match(name))}


def ensemble(fold_preds, agg="median"):
    """Per-recording median (or mean) over fold models, on the recordings every fold scored."""
    wide = pd.concat(list(fold_preds), axis=1, join="inner")
    return wide.median(axis=1) if agg == "median" else wide.mean(axis=1)


def row(meta, pred):
    d = meta[["age"]].join(pred.rename("pred"), how="inner").dropna()
    age, p = d["age"].to_numpy(float), d["pred"].to_numpy(float)
    return {"n": len(d), "age_mean": age.mean(), **metrics(p, age), "pred_mean": p.mean(), "pred_sd": p.std(),
            "mae_predict_mean": np.abs(age - age.mean()).mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--ensemble-of", nargs="*", default=None, help="studies pooled into one ensemble (default: all)")
    ap.add_argument("--agg", choices=("median", "mean"), default="median")
    ap.add_argument("--min-age", type=float, default=0.0)
    ap.add_argument("--max-age", type=float, default=np.inf)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta, preds, _ = load(args.cohort, args.eval, None)
    meta = meta[(meta.age >= args.min_age) & (meta.age <= args.max_age)]
    rows, pooled = [], []
    for name in args.also:
        rows.append({"model": name, "kind": "single model", "folds": 1, **row(meta, preds[name])})
    for study in args.studies:
        folds = fold_runs(preds, study)
        if not folds:
            raise SystemExit(f"no runs named {study}_f<k> in {args.eval}")
        per = [{"model": f"{study}_f{k}", "kind": "fold", "folds": 1, **row(meta, p)} for k, p in sorted(folds.items())]
        rows += per
        num = pd.DataFrame(per).drop(columns=["model", "kind", "folds"])
        rows.append({"model": study, "kind": "mean over folds", "folds": len(per), **num.mean().to_dict()})
        rows.append({"model": study, "kind": "SD over folds", "folds": len(per), **num.std(ddof=1).to_dict()})
        rows.append({"model": study, "kind": f"{args.agg} ensemble", "folds": len(per),
                     **row(meta, ensemble(folds.values(), args.agg))})
        if args.ensemble_of is None or study in args.ensemble_of:
            pooled += list(folds.values())
    if len(args.studies) > 1 and pooled:
        rows.append({"model": "+".join(args.ensemble_of or args.studies), "kind": f"{args.agg} ensemble",
                     "folds": len(pooled), **row(meta, ensemble(pooled, args.agg))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
