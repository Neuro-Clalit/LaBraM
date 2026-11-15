#!/usr/bin/env python
"""5-fold CV summary of phased one-run MIL models (docs/mil_stage1_results.md section 13).

For every model directory ``<root>/<MODEL>/fold_<k>/cv/fold_<k>/`` (downloaded
``model.tar.gz``): the best-val epoch and its val / test / Test[temple] /
Test[lemon] from ``log.txt``, the gap val - train-eval at the last epoch, the
per-cohort test MAE of each fold from ``analysis/test_bag_weights_epoch<best>.parquet``
(attention-pooled recording predictions), the fold mean ± SD (sample SD) and the
k-fold ensemble (mean of the fold models' recording predictions), and on LEMON the
per-fold and ensemble MAE with young / old MAE and bias and the slope of
prediction on age, from ``scripts/eval_age_by_cohort.py`` JSONs.

``--bootstrap B`` adds a row ``kind=ensemble_sd``: the SD of every ensemble
statistic over B subject-cluster bootstrap resamples of the test set (whole
subjects drawn with replacement within each cohort row; LEMON has one
recording per subject). It is the test-sampling uncertainty of the one
ensemble, not a spread over trained models like the fold SD.

    PYTHONPATH=.:scripts python scripts/mil_runs/phased_cv_summary.py \\
        --root /data/runs/heedb/mil_phased_cv --models P3_L1 P4_L1 \\
        --lemon /data/runs/heedb/external_eval/lemon_5min_p3.json ... \\
        --out docs/data/mil_phased_cv/p3_p4_cv5
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

LEMON_AGES = "/data/datasets/EEG-public/LEMON/cohorts/lemon_v1/age_metadata.json"
MANIFEST = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/manifest.json"
COHORTS = [("HEEDB", "normal", "Harvard normal"), ("HEEDB", "abnormal", "Harvard abnormal"),
           ("HEEDB", "unlabelled", "Harvard unlabelled"), ("TUAB", "normal", "TUAB normal"),
           ("TUAB", "abnormal", "TUAB abnormal")]


def recording_predictions(path):
    w = pd.read_parquet(path, columns=["recording", "alpha", "window_pred", "age"])
    g = w.groupby("recording")
    return (w["alpha"] * w["window_pred"]).groupby(w["recording"]).sum() / g["alpha"].sum(), g["age"].first()


def fold_log(path):
    d = pd.DataFrame([json.loads(line) for line in open(path)]).set_index("epoch").sort_index()
    best = d["val_mae"].idxmin()
    te = d["train_eval_mae"].dropna()
    row = d.loc[best]
    return {"best_epoch": int(best), "val": row["val_mae"], "test": row["test_mae"],
            "temple": row.get("test_temple_mae", np.nan), "lemon_log": row.get("test_lemon_mae", np.nan),
            "gap": d.loc[te.index[-1], "val_mae"] - te.iloc[-1]}


def lemon_table(files):
    ages = json.load(open(LEMON_AGES))["recordings"]
    preds = {}
    for f in files:
        for name, r in json.load(open(f)).items():
            preds[name] = pd.Series(r["splits"]["test"]["predictions"])
    return preds, ages


def lemon_stats(p, ages, y=None):
    if y is None:
        y = pd.Series({r: ages[r]["age"] for r in p.index})
    e = p - y
    yo, ol = y < 40, y >= 55
    return {"lemon": np.abs(e).mean(), "lemon_young": np.abs(e[yo]).mean(), "lemon_young_bias": e[yo].mean(),
            "lemon_old": np.abs(e[ol]).mean(), "lemon_old_bias": e[ol].mean(), "lemon_slope": np.polyfit(y, p, 1)[0]}


def subjects():
    m = json.load(open(MANIFEST))["recordings"]
    return pd.Series({r["stem"]: r["subject"] for r in m})


def cluster_boot_sd(err, groups, stat, n_boot, seed=0):
    """SD of ``stat(err[idx])`` over subject-cluster bootstrap resamples."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == u) for u in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        vals.append(stat(np.concatenate([members[i] for i in pick])))
    return float(np.std(vals, ddof=1))


def ensemble_boot(ens, age, m, subj, lemon_ens, ages, n_boot):
    e = np.abs((ens - age).to_numpy())
    g = subj.reindex(ens.index).to_numpy()
    sd = {"test": cluster_boot_sd(e, g, lambda i: e[i].mean(), n_boot)}
    for src, coh, label in COHORTS:
        mk = ((m.source == src) & (m.cohort == coh)).to_numpy()
        ek = e[mk]
        sd[label] = cluster_boot_sd(ek, g[mk], lambda i: ek[i].mean(), n_boot)
    if lemon_ens is not None:
        y = pd.Series({r: ages[r]["age"] for r in lemon_ens.index})
        idx = np.arange(len(y))
        stats = {}
        rng = np.random.default_rng(0)
        for _ in range(n_boot):
            i = rng.choice(idx, len(idx))
            for k, v in lemon_stats(lemon_ens.iloc[i].reset_index(drop=True), None, y.iloc[i].reset_index(drop=True)).items():
                stats.setdefault(k, []).append(v)
        sd.update({k: float(np.nanstd(v, ddof=1)) for k, v in stats.items()})
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lemon", nargs="+", default=[], help="eval_age_by_cohort JSONs with <MODEL>_f<k> runs")
    ap.add_argument("--fallback", default=None,
                    help="dir with <MODEL>_f<k>_test.parquet (eval_age_by_cohort --save-windows) for folds "
                         "whose best-epoch analysis parquet is missing")
    ap.add_argument("--lemon-alias", nargs="*", default=[], metavar="MODEL=PREFIX",
                    help="LEMON JSON run prefix of a model directory (e.g. L1=P_L1 for P1)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="subject-cluster bootstrap resamples for the ensemble SD row (0 = off)")
    ap.add_argument("--out", required=True, help="prefix of the output CSVs")
    args = ap.parse_args()
    sys.path.insert(0, "scripts")
    import mil_cohort_summary as mcs
    meta = mcs.recording_table()
    lemon_preds, ages = lemon_table(args.lemon)
    alias = dict(a.split("=", 1) for a in args.lemon_alias)
    subj = subjects() if args.bootstrap else None
    folds, summary = [], []
    for model in args.models:
        preds, age = {}, None
        for d in sorted(glob.glob(f"{args.root}/{model}/fold_*/cv/fold_*")):
            k = int(re.search(r"fold_(\d)$", d).group(1))
            r = {"model": model, "fold": k, **fold_log(f"{d}/log.txt")}
            src = f"{d}/analysis/test_bag_weights_epoch{r['best_epoch']:02d}.parquet"
            if not os.path.exists(src):
                # lost to a spot interruption: the best checkpoint re-scored with
                # eval_age_by_cohort.py --save-windows (same 5-minute evaluation)
                src = os.path.join(args.fallback or "", f"{model}_f{k}_test.parquet")
                r["preds_from"] = "re-scored checkpoint"
            preds[k], age = recording_predictions(src)
            m = meta.reindex(preds[k].index)
            for src, coh, label in COHORTS:
                mk = ((m.source == src) & (m.cohort == coh)).to_numpy()
                r[label] = np.abs(preds[k][mk] - age[mk]).mean()
            key = f"{alias.get(model, model)}_f{k}"
            if key in lemon_preds:
                r.update(lemon_stats(lemon_preds[key], ages))
            folds.append(r)
        fd = pd.DataFrame([f for f in folds if f["model"] == model])
        P = pd.DataFrame(preds)
        ens = P.mean(1)
        m = meta.reindex(P.index)
        e = {"model": model, "kind": "ensemble", "n_folds": P.shape[1], "test": np.abs(ens - age).mean()}
        for src, coh, label in COHORTS:
            mk = ((m.source == src) & (m.cohort == coh)).to_numpy()
            e[label] = np.abs(ens[mk] - age[mk]).mean()
        lp = alias.get(model, model)
        lk = [f"{lp}_f{k}" for k in sorted(preds) if f"{lp}_f{k}" in lemon_preds]
        lemon_ens = pd.concat([lemon_preds[x] for x in lk], axis=1).mean(1) if lk else None
        if lk:
            e.update(lemon_stats(lemon_ens, ages))
        cols = [c for c in fd.columns if c not in ("model", "fold", "best_epoch", "preds_from")]
        summary.append({"model": model, "kind": "mean", "n_folds": len(fd), **fd[cols].mean().to_dict()})
        summary.append({"model": model, "kind": "sd", "n_folds": len(fd), **fd[cols].std(ddof=1).to_dict()})
        summary.append(e)
        if args.bootstrap:
            summary.append({"model": model, "kind": "ensemble_sd", "n_folds": P.shape[1],
                            **ensemble_boot(ens, age, m, subj, lemon_ens, ages, args.bootstrap)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(folds).to_csv(f"{args.out}_folds.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{args.out}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(folds).round(3).to_string(index=False))
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
