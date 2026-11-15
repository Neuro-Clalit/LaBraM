#!/usr/bin/env python
"""One table of zero-shot brain-age metrics per cohort and model, from saved
``eval_age_by_cohort.py`` outputs (per-recording predictions) and the cohorts'
manifests (true ages).

For recording ``i`` of a cohort with true age ``y_i`` and prediction ``p_i``
(years): MAE, bias (mean of p - y), Pearson r, R^2 = 1 - SSE / SST about the
cohort's own mean, and the predict-the-cohort-mean MAE as the floor to beat
(``external_eval_report.metrics``). A cohort still carrying placeholder ages
(``manifest.cohort.age_placeholder``, e.g. TDBRAIN release 3.1 before its
participants table) gets only label-free rows: the predicted-age distribution
and, when the manifest has ``task`` / ``session``, the eyes-closed vs eyes-open
and session-to-session agreement of the predictions.

    python scripts/external_cohort_comparison.py \\
        --set lemon:/data/datasets/EEG-public/LEMON/cohorts/lemon_v1:/data/runs/heedb/external_eval/lemon_all.json \\
        --set tdbrain:/data/datasets/EEG-public/TDBRAIN/cohorts/tdbrain_v1:/data/runs/heedb/external_eval/tdbrain_all.json \\
        --models E2L E2Lrob E2LrobF --out docs/data/external_eval/tdbrain/cohort_comparison.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_eval_report import metrics  # noqa: E402


def load(cohort_dir, eval_jsons, models):
    man = json.load(open(os.path.join(cohort_dir, "manifest.json")))
    meta = pd.DataFrame(man["recordings"]).set_index("stem")
    if "age_session_matched" in meta:              # TDBRAIN: drop first-session fallback ages
        meta = meta[meta["age_session_matched"].fillna(True).astype(bool)]
    if "age" not in meta or meta["age"].isna().all():     # training cohorts keep ages in the sidecar
        ages = json.load(open(os.path.join(cohort_dir, "age_metadata.json")))["recordings"]
        meta["age"] = pd.Series({k: v.get("age") for k, v in ages.items()}, dtype=float)
        if "subject" not in meta:
            meta["subject"] = pd.Series({k: v.get("subject") for k, v in ages.items()})
    preds = {}
    for path in eval_jsons:
        for model, res in json.load(open(path)).items():
            if models and model not in models:
                continue
            split = next(iter(res["splits"].values()))
            preds[model] = pd.Series(split["predictions"], dtype=float)
    return meta, preds, bool(man.get("cohort", {}).get("age_placeholder"))


def label_free(meta, pred):
    d = meta.join(pred.rename("pred"), how="inner")
    out = {"n": len(d), "pred_mean": d.pred.mean(), "pred_sd": d.pred.std()}
    if "task" in d and d["task"].notna().any():
        wide = d.pivot_table(index=["subject", "session"], columns="task", values="pred")
        if {"restEC", "restEO"} <= set(wide.columns):
            w = wide.dropna(subset=["restEC", "restEO"])
            out.update(ec_eo_r=w.restEC.corr(w.restEO), ec_minus_eo=(w.restEC - w.restEO).mean())
    if "session" in d:
        ses = d.groupby(["subject", "session"]).pred.mean().unstack("session")
        if {"1", "2"} <= set(ses.columns):
            s = ses.dropna(subset=["1", "2"])
            out.update(n_two_sessions=len(s), ses1_ses2_r=s["1"].corr(s["2"]), ses2_minus_ses1=(s["2"] - s["1"]).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT_DIR:EVAL.json[,EVAL2.json]")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--min-age", type=float, default=0.0,
                    help="only recordings aged >= this (18 = adults only, as for the TDBRAIN analysis)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.set:
        name, cdir, evals = spec.split(":", 2)
        meta, preds, placeholder = load(cdir, evals.split(","), args.models)
        for model, pred in preds.items():
            if placeholder:
                rows.append({"cohort": name, "model": model, "subset": "all (ages pending)", **label_free(meta, pred)})
                continue
            d = meta.join(pred.rename("pred"), how="inner").dropna(subset=["age"])
            subsets = [(f"age >= {args.min_age:g}" if args.min_age else "all", d[d.age >= args.min_age])]
            for sub, s in subsets:
                if len(s) < 3:
                    continue
                age, p = s["age"].to_numpy(float), s["pred"].to_numpy(float)
                rows.append({"cohort": name, "model": model, "subset": sub, "n": len(s),
                             "age_mean": age.mean(), "age_sd": age.std(), **metrics(p, age),
                             "mae_predict_mean": np.abs(age - age.mean()).mean(),
                             **label_free(s.drop(columns="pred"), s["pred"])})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
