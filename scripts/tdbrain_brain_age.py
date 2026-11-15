#!/usr/bin/env python
"""Zero-shot brain age on TDBRAIN: accuracy per scoring unit, age band and eye
condition, a within-cohort recalibration for comparison with published
within-TDBRAIN results, and the brain-age gap per psychiatric indication.

Inputs: an ``eval_age_by_cohort.py`` output on the ``tdbrain_v1`` cohort (one
prediction per 2-min file) and the cohort manifest (ages, sessions, clinical
flags written by ``make_external_npy --dataset tdbrain [--relabel]``).

Scoring units: ``file`` (each eyes-closed / eyes-open file on its own),
``session`` (the mean of the session's EC and EO predictions, the unit of the
published TDBRAIN results), ``subject`` (the first session with a matched age).
Only files whose age comes from their own session's participants row
(``age_session_matched``) are used, and only **adults** (``--min-age``, default
18): the 5-17-year-olds lie outside the training ages (2 TUH recordings under 18,
none in HEEDB), and MIPDB showed the models overestimate children by ~10 y.

Metrics for predictions ``p`` and ages ``y`` (years): MAE with a 95 %
subject-level bootstrap interval, bias = mean(p - y), Pearson r, R^2 = 1 -
SSE / SST about the group's own mean. The within-cohort recalibration fits
``y = c * p + d`` on 9 of 10 subject-grouped folds and applies it to the 10th
(it uses TDBRAIN ages, so it is not zero-shot).

Confidence rejection (``--windows`` + ``--reject``, label-free, the ``keep80r``
rule of ``eval_strategies_table.py``): each file's confidence is the
attention-weighted spread of its window predictions, ``s = sqrt(sum_i a_i
(p_i - p)^2)`` (``confidence_thresholds.recordings``; uniform weights for E2L);
an ensemble's spread is the mean of its members' spreads, a session's the mean
of its files'. The ``--reject`` share of sessions with the largest spread get no
prediction; the rest are scored.

Brain-age gap ``g = p - y`` per session. The zero-shot correction removes the
model's age trend fitted on its **own** validation predictions (normal-report
recordings of the training cohort): ``g_c = g - (a * y + b)`` with
``p - y = a * y + b`` on validation (``--val``). Groups are compared with the
healthy controls on first sessions: mean gap difference, Cohen's d,
Welch t-test, and an OLS ``g_c ~ group + age + age^2 + sex`` whose group
coefficient and t statistic are reported.

    python scripts/tdbrain_brain_age.py \\
        --eval /data/runs/heedb/external_eval/tdbrain_all.json \\
        --cohort /data/datasets/EEG-public/TDBRAIN/cohorts/tdbrain_v1 \\
        --models E2L E2Lrob E2LrobF \\
        --val E2L=/data/runs/heedb/external_eval/own_all_val_E2L.json \\
        --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \\
        --out-dir docs/data/external_eval/tdbrain
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

BANDS = [("18-29", 18, 30), ("30-44", 30, 45), ("45-64", 45, 65), ("65+", 65, 200)]
GROUPS = ["healthy", "dx_mdd", "dx_adhd", "dx_ocd", "dx_smc", "dx_other", "unknown"]


def metrics(d, n_boot=2000, seed=0):
    y, p = d["age"].to_numpy(float), d["pred"].to_numpy(float)
    e = p - y
    subs = d["subject"].to_numpy()
    uniq, inv = np.unique(subs, return_inverse=True)
    rng = np.random.default_rng(seed)
    per_sub = np.bincount(inv, np.abs(e)), np.bincount(inv)
    boot = []
    for _ in range(n_boot):                          # resample subjects, not files
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boot.append((w * per_sub[0]).sum() / (w * per_sub[1]).sum())
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": len(d), "n_subjects": len(uniq), "age_mean": y.mean(), "age_sd": y.std(),
            "mae": np.abs(e).mean(), "mae_lo": np.percentile(boot, 2.5), "mae_hi": np.percentile(boot, 97.5),
            "bias": e.mean(), "r": np.corrcoef(p, y)[0, 1] if len(d) > 2 else np.nan,
            "r2": 1 - (e ** 2).sum() / sst if sst > 0 else np.nan,
            "mae_predict_mean": np.abs(y - y.mean()).mean()}


def units(files):
    """file / session / subject tables of (subject, age, pred, ...)."""
    keep = ["subject", "session", "age", "sex", "clinical_group", *GROUPS]
    ses = files.groupby(["subject", "session"]).agg(
        pred=("pred", "mean"), n_files=("pred", "size"), **{k: (k, "first") for k in keep if k not in ("subject", "session")}
    ).reset_index()
    first = ses.sort_values("session", key=lambda s: s.astype(int)).groupby("subject").head(1)
    return {"file": files, "session": ses, "subject": first}


def recalibrated(d, folds=10, seed=0):
    subs = np.array(sorted(d["subject"].unique()))
    fold_of = dict(zip(subs, np.random.default_rng(seed).permutation(len(subs)) % folds))
    f = d["subject"].map(fold_of).to_numpy()
    out = np.empty(len(d))
    for k in range(folds):
        c, b = np.polyfit(d["pred"].to_numpy()[f != k], d["age"].to_numpy()[f != k], 1)
        out[f == k] = c * d["pred"].to_numpy()[f == k] + b
    return d.assign(pred=out)


def file_spread(windows_dir, members):
    """Per file: mean over ``members`` of the attention-weighted window spread."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import confidence_thresholds as ct
    out = [ct.recordings(ct.load(os.path.join(windows_dir, f"{m}_test.parquet")), slice(None))["spread"]
           for m in members]
    return pd.concat(out, axis=1).mean(axis=1)


def gap_trend(val_json, name, val_meta):
    lab = json.load(open(val_meta))["recordings"]
    p = pd.Series(json.load(open(val_json))[name]["splits"]["val"]["predictions"], dtype=float)
    p = p[[s for s in p.index if lab.get(s, {}).get("label") == "normal" and lab[s].get("age") is not None]]
    age = np.array([lab[s]["age"] for s in p.index], float)
    return np.polyfit(age, p.to_numpy() - age, 1)


def ols_group_effect(d, group):
    """Coefficient and t of the group indicator in gap ~ 1 + group + age + age^2 + male (vs healthy)."""
    x = d[(d[group]) | (d["healthy"])].dropna(subset=["sex"])
    if x[group].sum() < 5 or x["healthy"].sum() < 5:
        return np.nan, np.nan
    a = (x["age"] - x["age"].mean()).to_numpy()
    X = np.column_stack([np.ones(len(x)), x[group].astype(float), a, a ** 2, (x["sex"] == "M").astype(float)])
    yv = x["gap_c"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = len(yv) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    return beta[1], beta[1] / np.sqrt(cov[1, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val", action="append", default=[], metavar="NAME=JSON")
    ap.add_argument("--val-meta", default=None)
    ap.add_argument("--min-age", type=float, default=18.0,
                    help="analyse recordings aged >= this (default 18: adults only; 0 = everyone)")
    ap.add_argument("--windows", default=None,
                    help="dir of eval_age_by_cohort.py --save-windows parquets (<model>_test.parquet)")
    ap.add_argument("--reject", type=float, default=0.2,
                    help="share of sessions set aside by spread (with --windows)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.cohort, "manifest.json")))
    if man["cohort"].get("age_placeholder"):
        raise SystemExit("cohort still has placeholder ages: run make_external_npy --relabel first")
    meta = pd.DataFrame(man["recordings"]).set_index("stem")
    meta = meta[meta["age_session_matched"].astype(bool) & (meta["age"] >= args.min_age)]
    print(f"{len(meta)} files aged >= {args.min_age:g} with a session-matched age")
    for g in GROUPS:
        meta[g] = meta[g].fillna(False).astype(bool)
    res = json.load(open(args.eval))
    vals = dict(v.split("=", 1) for v in args.val)
    acc, gaps = [], []
    for model in args.models:
        files = meta.join(pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float).rename("pred"), how="inner")
        u = units(files)
        for unit, d in u.items():
            subsets = [("all", d)] + [(f"age {n}", d[(d.age >= lo) & (d.age < hi)]) for n, lo, hi in BANDS]
            if unit == "file":
                subsets += [(t, d[d.task == t]) for t in ("restEC", "restEO")]
            subsets += [("healthy", d[d.healthy])]
            for name, s in subsets:
                if len(s) >= 10:
                    acc.append({"model": model, "unit": unit, "subset": name, **metrics(s)})
            if unit == "session":
                acc.append({"model": model, "unit": "session, recalibrated (10-fold CV, uses ages)",
                            "subset": "all", **metrics(recalibrated(d))})
                if args.windows:
                    spread = file_spread(args.windows, res[model].get("members", [model]))
                    sp = files.join(spread.rename("spread"), how="inner").groupby(["subject", "session"])["spread"].mean()
                    s = d.set_index(["subject", "session"]).join(sp, how="inner")
                    keep = s[s["spread"] <= s["spread"].quantile(1 - args.reject)].reset_index()
                    acc.append({"model": model, "unit": f"session, reject {args.reject:.0%} by spread (label-free)",
                                "subset": "all", "kept_share": len(keep) / len(s), **metrics(keep)})
                    acc.append({"model": model, "unit": f"session, reject {args.reject:.0%} by spread (label-free)",
                                "subset": "healthy", "kept_share": keep.healthy.sum() / s.healthy.sum(),
                                **metrics(keep[keep.healthy])})
        # brain-age gap per indication: first session of each subject
        s = u["subject"].copy()
        s["gap"] = s["pred"] - s["age"]
        if model in vals and args.val_meta:
            a, b = gap_trend(vals[model], model, args.val_meta)
            s["gap_c"] = s["gap"] - (a * s["age"] + b)
            corr = f"val trend a={a:.3f} b={b:.2f}"
        else:
            s["gap_c"] = np.nan
            corr = "none (no validation predictions)"
        h = s[s.healthy]
        for g in GROUPS:
            x = s[s[g]]
            row = {"model": model, "group": g, "n": len(x), "age_mean": x.age.mean(), "gap_raw": x.gap.mean(),
                   "gap_corrected": x.gap_c.mean(), "correction": corr}
            if g != "healthy" and len(x) >= 5:
                col = "gap_c" if s["gap_c"].notna().any() else "gap"
                pooled = np.sqrt(((len(x) - 1) * x[col].var() + (len(h) - 1) * h[col].var()) / (len(x) + len(h) - 2))
                t = stats.ttest_ind(x[col], h[col], equal_var=False)
                coef, tstat = ols_group_effect(s if col == "gap_c" else s.assign(gap_c=s.gap), g)
                row.update(diff_vs_healthy=x[col].mean() - h[col].mean(), cohens_d=(x[col].mean() - h[col].mean()) / pooled,
                           welch_p=t.pvalue, ols_coef_adj_age_sex=coef, ols_t=tstat)
            gaps.append(row)
    os.makedirs(args.out_dir, exist_ok=True)
    acc, gaps = pd.DataFrame(acc), pd.DataFrame(gaps)
    acc.to_csv(os.path.join(args.out_dir, "tdbrain_accuracy.csv"), index=False)
    gaps.to_csv(os.path.join(args.out_dir, "tdbrain_gap_by_indication.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(acc.round(3).to_string(index=False))
        print(gaps.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
