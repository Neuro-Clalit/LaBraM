#!/usr/bin/env python
"""Stage B of docs/fm_encoder_onboarding.md: one fine-tuned run against P4_L1, paired per cohort.

Cohorts: the 7,612 held-out test recordings (first 5 minutes; per-recording predictions from the
runs' test bag-weight files) and LEMON, Dortmund, TDBRAIN, SRM, CHBP as scored by
``scripts/eval_age_by_cohort.py`` (``scripts/external_runs/fm_external.sh score``), adults only.

Per cohort: the MAE of P4 fold 0 and of the run with the paired difference and its bootstrap
interval (``labram.eval.cv_compare.bootstrap_delta_mae``), and the same for averages of models:

* ``pair``     mean of P4 fold 0 and the run, against P4 fold 0;
* ``control``  mean of P4 fold 0 and another P4 fold (averaged over the other folds), i.e. what a
               second LaBraM model gives;
* ``ens``      the mean of all available P4 folds, and that ensemble with the run as one more
               member of equal weight, against the ensemble.

    python scripts/fm_external_compare.py --tag rv4dp --name RV4DP_L1_f0 \\
        --test-parquet <run>/analysis/test_bag_weights_epoch12.parquet

``--by-report`` adds one row per report status of a cohort (normal / abnormal / unlabelled, the
``label`` of the cohort's ``age_metadata.json``): ``test_normal``, ``test_harvard_abnormal``,
``tdbrain_unlabelled``, ... -> ``<tag>_vs_p4_by_report.csv``.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bag_mae import recording_predictions  # noqa: E402

from labram.eval.cv_compare import bootstrap_delta_mae  # noqa: E402

COHORTS = {"lemon": "LEMON", "dortmund": "DORTMUND", "tdbrain": "TDBRAIN", "srm": "SRM", "chbp": "CHBP"}


def _predictions(path, prefix):
    if not os.path.exists(path):
        return {}
    return {k: pd.Series(v["splits"]["test"]["predictions"], dtype=float)
            for k, v in json.load(open(path)).items() if k.startswith(prefix)}


def _p4_test_file(fold_dir):
    """The test bag-weight file of a P4 fold at its best validation epoch (``fold_metrics.json``,
    else ``train_state.json``). A fold whose file of that epoch was not saved (fold 3: its last
    epoch) falls back to the latest saved epoch, with a note."""
    best = None
    for name in ("fold_metrics.json", "train_state.json"):
        try:
            best = json.load(open(os.path.join(fold_dir, name)))["best_epoch"]
            break
        except (OSError, KeyError):
            continue
    files = sorted(glob.glob(os.path.join(fold_dir, "analysis", "test_bag_weights_epoch*.parquet")))
    if best is None or not files:
        return None
    exact = os.path.join(fold_dir, "analysis", f"test_bag_weights_epoch{best:02d}.parquet")
    if exact in files:
        return exact
    print(f"note: {fold_dir}: no test predictions of its best epoch {best}; using {os.path.basename(files[-1])}")
    return files[-1]


def _report_labels(cohort_dir):
    """recording -> normal / abnormal / unlabelled (no verdict in the source data)."""
    meta = json.load(open(os.path.join(cohort_dir, "age_metadata.json")))["recordings"]
    return pd.Series({r: (v.get("label") if v.get("label") in ("normal", "abnormal") else "unlabelled")
                      for r, v in meta.items()})


def _delta(pred, ref, age):
    b = bootstrap_delta_mae(pred - age, ref - age)
    return float((pred - age).abs().mean()), b["dMAE"], b["lo"], b["hi"]


def compare(name, age, p4_folds, run):
    """One row: ``p4_folds`` a frame of P4 predictions per fold (``P4_L1_f0`` first), ``run`` a series."""
    f0 = p4_folds["P4_L1_f0"]
    others = [c for c in p4_folds.columns if c != "P4_L1_f0"]
    ens = p4_folds.mean(axis=1)
    k = p4_folds.shape[1]
    row = dict(cohort=name, n=len(age), p4_folds=k, p4=float((f0 - age).abs().mean()))
    row.update(zip(("run", "d", "lo", "hi"), _delta(run, f0, age)))
    row.update(zip(("pair", "pair_d", "pair_lo", "pair_hi"), _delta((f0 + run) / 2, f0, age)))
    row["control"] = float(np.mean([((f0 + p4_folds[c]) / 2 - age).abs().mean() for c in others])) if others else np.nan
    row["ens"] = float((ens - age).abs().mean())
    row.update(zip(("ens_run", "ens_d", "ens_lo", "ens_hi"), _delta((p4_folds.sum(axis=1) + run) / (k + 1), ens, age)))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="file tag of the run's scores: <cohort>_all_<tag>.json")
    ap.add_argument("--name", required=True, help="the run's name inside those files, e.g. RV4DP_L1_f0")
    ap.add_argument("--test-parquet", default=None, help="the run's test bag-weight file at its best epoch")
    ap.add_argument("--eval-dir", default="/data/runs/heedb/fm_encoders/eval")
    ap.add_argument("--p4-dir", default="/data/runs/heedb/external_eval")
    ap.add_argument("--p4-test", default="/data/runs/fm_probe/p4_f0_rescore.json")
    ap.add_argument("--p4-cv", default="/data/runs/heedb/mil_phased_cv/P4_L1")
    ap.add_argument("--cohorts-root", default="/data/datasets/EEG-public")
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--by-report", action="store_true", help="add one row per report status of a cohort")
    ap.add_argument("--train-cohort", default="/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1",
                    help="the cohort of the test split (its age_metadata.json has the report labels)")
    ap.add_argument("--out", default=None, help="CSV (default <eval-dir>/<tag>_vs_p4.csv)")
    args = ap.parse_args()

    rows = []
    if args.test_parquet:
        run = recording_predictions(args.test_parquet)
        folds = {"P4_L1_f0": _predictions(args.p4_test, "P4_L1_f0")["P4_L1_f0"]}
        for d in sorted(glob.glob(os.path.join(args.p4_cv, "fold_[1-9]", "cv", "fold_*"))):
            path = _p4_test_file(d)
            if path:                          # else: a fold whose test predictions are not on this machine
                folds[f"P4_L1_f{d[-1]}"] = recording_predictions(path).pred
        p4 = pd.DataFrame(folds).reindex(run.index)
        if p4.isna().any().any():
            raise ValueError("the run and P4 were not scored on the same test recordings")
        temple = run.index.to_series().str.match(r"^[a-z]{8}_s\d+_t\d+$").to_numpy()
        groups = [("test", np.ones(len(run), bool)), ("test_harvard", ~temple), ("test_temple", temple)]
        if args.by_report:
            label = _report_labels(args.train_cohort).reindex(run.index).fillna("unlabelled").to_numpy()
            for base, m in list(groups):
                groups += [(f"{base}_{g}", m & (label == g)) for g in ("normal", "abnormal", "unlabelled")
                           if (m & (label == g)).any()]
        for name, m in groups:
            rows.append(compare(name, run.age[m], p4[m], run.pred[m]))
    for cohort, folder in COHORTS.items():
        path = os.path.join(args.eval_dir, f"{cohort}_all_{args.tag}.json")
        if not os.path.exists(path):
            continue
        run = _predictions(path, args.name)[args.name]
        folds = _predictions(os.path.join(args.p4_dir, f"{cohort}_all_p34.json"), "P4_L1_f")
        folds.update(_predictions(os.path.join(args.p4_dir, f"{cohort}_all_p4f3.json"), "P4_L1_f"))
        p4 = pd.DataFrame(folds)[sorted(folds)]
        meta = json.load(open(os.path.join(args.cohorts_root, folder, "cohorts", f"{cohort}_v1",
                                           "age_metadata.json")))["recordings"]
        age = pd.Series({r: float(v["age"]) for r, v in meta.items() if v.get("age") is not None})
        age = age.reindex(p4.index)
        keep = (age >= args.min_age).to_numpy()
        run = run.reindex(p4.index)
        if run.isna().any() or p4.isna().any().any():
            raise ValueError(f"{cohort}: the run and P4 were not scored on the same recordings")
        rows.append(compare(cohort, age[keep], p4[keep], run[keep]))
        if args.by_report:
            label = _report_labels(os.path.join(args.cohorts_root, folder, "cohorts", f"{cohort}_v1"))
            label = label.reindex(p4.index).fillna("unlabelled").to_numpy()
            for g in ("normal", "abnormal", "unlabelled"):
                m = keep & (label == g)
                if m.any():
                    rows.append(compare(f"{cohort}_{g}", age[m], p4[m], run[m]))
    table = pd.DataFrame(rows)
    out = args.out or os.path.join(args.eval_dir, f"{args.tag}_vs_p4{'_by_report' if args.by_report else ''}.csv")
    table.to_csv(out, index=False)
    print(table.round(2).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
