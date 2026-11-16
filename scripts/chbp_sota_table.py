#!/usr/bin/env python
"""CHBP brain age: our zero-shot models vs the published within-CHBP benchmark.

Engemann et al. 2022 (NeuroImage 262:119521) report CHBP age prediction as
``sklearn`` 10-fold ``KFold(shuffle=True, random_state=42)`` over the
participants (sorted by id), as the mean ± SD over the 10 test folds of the
fold MAE and of the fold R² (R² against the fold's own mean). Their per-fold
results are public (``results/benchmark-<name>_dataset-chbp.csv`` of
github.com/meeg-ml-benchmarks/brain-age-benchmark-paper, BSD-3).

This script scores our per-recording predictions (``eval_age_by_cohort.py``
outputs) on the same fold scheme, so both sides are aggregated identically:
fold-mean MAE ± SD, fold-mean R² ± SD, plus the pooled MAE with its 95 %
bootstrap CI, bias, Pearson r and pooled R². A frozen model's predictions do
not depend on the folds; the folds only change how the metrics are averaged.

Rows of kind ``uses CHBP labels`` fit two parameters (``a * pred + b``) on the
other nine folds; they are a reference, not zero-shot.

    python scripts/chbp_sota_table.py --meta .../chbp_v1/age_metadata.json \\
        --published-dir docs/data/external_eval/chbp/published_engemann2022 --pred "E2L|zero-shot|chbp_5min.json:E2L" ... \\
        --out docs/data/external_eval/chbp/sota_comparison.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PUBLISHED = {"handcrafted": "handcrafted features + random forest",
             "shallow": "ShallowNet (braindecode)",
             "deep": "Deep4Net (braindecode)",
             "filterbank-riemann": "filterbank Riemann + ridge",
             "filterbank-source": "filterbank source + ridge",
             "dummy": "predict the training-fold mean"}


def fold_index(stems, n_splits=10, seed=42):
    """Fold per stem, ``KFold(shuffle, seed)`` over the subjects sorted by id (as the benchmark)."""
    subjects = sorted({s.split("_")[0] for s in stems})
    fold_of = {}
    for k, (_, test) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(subjects)):
        fold_of.update({subjects[i]: k for i in test})
    return np.array([fold_of[s.split("_")[0]] for s in stems])


def fold_r2(y, p):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def score(y, p, fold, n_boot=2000, seed=0):
    err = p - y
    boot = np.abs(err[np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))]).mean(axis=1)
    ks = np.unique(fold)
    mae_f = [np.abs(err[fold == k]).mean() for k in ks]
    r2_f = [fold_r2(y[fold == k], p[fold == k]) for k in ks]
    return {"n": len(y), "mae_fold_mean": np.mean(mae_f), "mae_fold_sd": np.std(mae_f, ddof=1),
            "r2_fold_mean": np.mean(r2_f), "r2_fold_sd": np.std(r2_f, ddof=1),
            "mae": np.abs(err).mean(), "mae_ci_lo": np.percentile(boot, 2.5),
            "mae_ci_hi": np.percentile(boot, 97.5), "bias": err.mean(),
            "r": np.corrcoef(y, p)[0, 1], "r2": fold_r2(y, p)}


def recalibrate(y, p, fold):
    """Out-of-fold ``a * pred + b`` fitted (OLS of age on pred) on the other folds."""
    out = np.empty_like(p)
    for k in np.unique(fold):
        a, b = np.polyfit(p[fold != k], y[fold != k], 1)
        out[fold == k] = a * p[fold == k] + b
    return out


def load_pred(spec):
    """``a.json:KEY[+b.json:KEY...]`` -> Series stem -> prediction (parts are disjoint subsets)."""
    parts = []
    for item in spec.split("+"):
        path, key = item.rsplit(":", 1)
        parts.append(pd.Series(json.load(open(path))[key]["splits"]["test"]["predictions"]))
    s = pd.concat(parts)
    assert not s.index.duplicated().any(), spec
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--published-dir", required=True,
                    help="folder with the benchmark's benchmark-<name>_dataset-chbp.csv files")
    ap.add_argument("--pred", action="append", required=True, metavar="LABEL|KIND|SPEC[|recal]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for name, label in PUBLISHED.items():
        d = pd.read_csv(os.path.join(args.published_dir, f"benchmark-{name}_dataset-chbp.csv"))
        rows.append({"model": label, "kind": "published, trained on CHBP (10-fold CV)",
                     "mae_fold_mean": d.MAE.mean(), "mae_fold_sd": d.MAE.std(),
                     "r2_fold_mean": d.r2.mean(), "r2_fold_sd": d.r2.std()})

    ages = {s: r["age"] for s, r in json.load(open(args.meta))["recordings"].items()}
    for item in args.pred:
        label, kind, spec, *opt = item.split("|")
        p = load_pred(spec)
        p = p[[s in ages for s in p.index]].sort_index()
        y = np.array([ages[s] for s in p.index], float)
        fold = fold_index(list(p.index))
        pred = p.to_numpy(float)
        if opt and opt[0] == "recal":
            pred = recalibrate(y, pred, fold)
        rows.append({"model": label, "kind": kind, **score(y, pred, fold)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
