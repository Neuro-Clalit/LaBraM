#!/usr/bin/env python
"""Fixed evaluation protocol for the technical report (Section 10.4 of
``docs/tech_report/``): one evaluation window for every cohort, and rejection
thresholds fixed on validation data instead of each cohort's own quantile.

**Evaluation window.** From the saved windows of ``eval_age_by_cohort.py
--save-windows`` every recording is re-pooled over the windows that start in
its first ``m`` minutes (``m`` = 2, 5, and every window). For a fold model with
window predictions ``p_i`` and attention weights ``a_i`` the pooled prediction
is ``sum_i a_i p_i / sum_i a_i`` over the kept windows: exact for the softmax
attention pool followed by the linear head (E2L: the plain mean). Ensembles are
the mean of their fold models; TDBRAIN and Dortmund sessions the mean of their
files. The in-domain sets were saved for the first 5 minutes only (the training
protocol); an extra set such as ``temple_every`` with E2L's every-window
predictions can be passed.

**Validation-fixed thresholds.** The two scores that exist for a single fold
model are fixed on the validation folds of the training cohort (each fold model
on its own validation fold at its best epoch, adults, first 5 minutes):

* ``spread``: threshold = the 80 % quantile of the validation spread of the
  fold models, pooled; a unit is rejected when the fold mean of its spread is
  above it;
* ``learned``: the gradient-boosted |error| regressor of
  ``rejection_scores.fit_val_model``; threshold = the 80 % quantile of its
  validation predictions, each fold model's recordings predicted by a regressor
  fitted on the other fold models' validation folds (the CV folds are
  patient-disjoint); a unit is rejected when the mean predicted |error| of its
  fold models is above it.

The ``total`` score needs the disagreement of five fold models on the same
recording, which validation recordings do not have (each was trained on by the
other four), so it keeps the cohort's own quantile. Both rules are evaluated on
the first 5 minutes of every cohort, the protocol of the validation windows.

    python scripts/report_protocol.py --set temple:<cohort>:<eval.json>:<windows>:temple ... \\
        --models E2L P3ens P4ens --val-windows P4ens=<parquet,...> --out-dir docs/data/external_eval/tech_report
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rejection_scores as rs  # noqa: E402
from external_cohort_table import load_meta  # noqa: E402
from tdbrain_brain_age import metrics  # noqa: E402

WINDOWS = [("first 2 min", 2), ("first 5 min", 5), ("every window", None)]
QUANTILE = 0.8


def members_of(ev, model):
    """Fold models of an ensemble (or the model itself) whose windows were saved."""
    if model not in ev:
        return []
    return ev[model].get("members", [model])


def val_scores(paths, min_age=18.0):
    """Per validation recording of each fold model: spread, cross-fitted learned
    score (regressor fitted on the other fold models' validation folds)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    frames = []
    for k, path in enumerate(paths):
        f = rs.fold_scores(path)
        f["age"] = pd.read_parquet(path, columns=["recording", "age"]).groupby("recording")["age"].first()
        f["pred_age"] = f["yhat"]
        f["abs_err"] = (f["yhat"] - f["age"]).abs()
        frames.append(f[f["age"] >= min_age].assign(fold=k))
    d = pd.concat(frames)
    d["learned"] = np.nan
    for k in d["fold"].unique():
        tr, te = d["fold"] != k, d["fold"] == k
        gbm = HistGradientBoostingRegressor(**rs.GBM).fit(d.loc[tr, rs.VAL_FEATURES], d.loc[tr, "abs_err"])
        d.loc[te, "learned"] = gbm.predict(d.loc[te, rs.VAL_FEATURES])
    return d


def kept_metrics(d, keep):
    m = metrics(d[keep].reset_index(drop=True), n_boot=1000)
    return {"kept_share": keep.mean(), "n_kept": int(keep.sum()), "mae": m["mae"], "mae_lo": m["mae_lo"],
            "mae_hi": m["mae_hi"], "bias": m["bias"], "age_mean_kept": m["age_mean"],
            "age_mean_rejected": d.loc[~keep, "age"].mean() if (~keep).any() else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--models", nargs="+", default=["E2L", "P3ens", "P4ens"])
    ap.add_argument("--val-windows", action="append", default=[], metavar="MODEL=PARQUET[,PARQUET...]")
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    val_paths = {m: p.split(",") for m, p in (v.split("=", 1) for v in args.val_windows)}

    # thresholds fixed on validation
    thr, val_rows = {}, []
    for model, paths in val_paths.items():
        v = val_scores(paths, args.min_age)
        thr[model] = {"spread": v["spread"].quantile(QUANTILE), "learned": v["learned"].quantile(QUANTILE)}
        val_model, n = rs.fit_val_model(paths, args.min_age)
        thr[model]["val_model"] = val_model
        val_rows.append({"model": model, "val_recordings": len(v), "spread_threshold": thr[model]["spread"],
                         "learned_threshold": thr[model]["learned"],
                         "val_mae_all": v["abs_err"].mean(),
                         "val_mae_kept_spread": v.loc[v["spread"] <= thr[model]["spread"], "abs_err"].mean(),
                         "val_mae_kept_learned": v.loc[v["learned"] <= thr[model]["learned"], "abs_err"].mean()})
        print(f"{model}: thresholds on {len(v):,} validation recordings: spread {thr[model]['spread']:.2f} y, "
              f"learned {thr[model]['learned']:.2f} y", flush=True)

    win_rows, rej_rows = [], []
    for spec in args.set:
        name, cdir, evp, wdir, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        ev = json.load(open(evp))
        for model in args.models:
            mem = members_of(ev, model)
            if not mem or not all(os.path.exists(os.path.join(wdir, f"{m}_test.parquet")) for m in mem):
                continue
            for label, minutes in WINDOWS:
                d = rs.cohort_units(meta, wdir, mem, max_minute=minutes)
                d = d.rename(columns={"pred": "pred"})
                m = metrics(d.reset_index(drop=True), n_boot=1000)
                win_rows.append({"set": name, "model": model, "window": label, **m})
            if model not in thr:
                continue
            d = rs.cohort_units(meta, wdir, mem, val_model=thr[model]["val_model"], max_minute=5)
            rows = [("none", "", np.nan, np.ones(len(d), bool))]
            for score, col in (("spread", "spread"), ("learned", "learned_val")):
                rows.append((score, "cohort 80 % quantile", d[col].quantile(QUANTILE),
                             (d[col] <= d[col].quantile(QUANTILE)).to_numpy()))
                rows.append((score, "validation threshold", thr[model][score], (d[col] <= thr[model][score]).to_numpy()))
            for score, rule, t, keep in rows:
                rej_rows.append({"set": name, "model": model, "score": score, "rule": rule, "threshold": t,
                                 "n": len(d), **kept_metrics(d, keep)})
            print(f"{name} {model}: done", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"protocol_windows.csv": pd.DataFrame(win_rows), "protocol_rejection.csv": pd.DataFrame(rej_rows),
           "protocol_val_thresholds.csv": pd.DataFrame(val_rows)}
    for fname, df in out.items():
        df.to_csv(os.path.join(args.out_dir, fname), index=False)
        print(os.path.join(args.out_dir, fname))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        w = out["protocol_windows.csv"]
        print(w.pivot_table(index=["set", "model"], columns="window", values="mae").round(2).to_string())
        r = out["protocol_rejection.csv"]
        print(r[["set", "model", "score", "rule", "threshold", "kept_share", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
