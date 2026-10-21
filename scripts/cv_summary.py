#!/usr/bin/env python
"""Summarize cross-validation fold models on the test cohorts.

From ``scripts/eval_cv_folds.py`` outputs (``--out`` CSV, ``--preds-out``
JSON), per experiment:

* **over fold models**: median, mean ± sd, min and max of the per-fold MAE on
  the *official TUAB test* (split ``heldout``: the same 274 recordings for
  every fold model) and on the *TUEG part of each fold's held-out test fold*;
* **median over fold models (ensemble)** on the official TUAB test: each
  recording's prediction is the median (and, for reference, the mean) of the
  fold models' predictions, then scored;
* **TUEG out-of-fold**: every TUEG recording is held out by exactly one fold,
  so its one prediction is pooled into a single score over all TUEG
  recordings of the CV pool.

All for all / normal / abnormal recordings; MAE / RMSE in years, R².

    python scripts/cv_summary.py --csv cv_eval.csv --preds cv_preds.json \\
        --labels .../cohorts/all_both_v3_cvcompact
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from labram.data.tuh_metadata import load_label_lookup_for

INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"


def metrics(p, t):
    e = p - t
    return {"n": len(e), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum())}


def by_cohort(stems, p, t, labels):
    lab = np.array([labels.get(s) for s in stems])
    out = {}
    for c in ("all", "normal", "abnormal"):
        m = np.ones(len(stems), bool) if c == "all" else lab == c
        out[c] = metrics(p[m], t[m])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True, help="dataset dir with age_metadata.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    d = pd.read_csv(args.csv)
    preds = json.load(open(args.preds))
    labels = load_label_lookup_for(args.labels)
    idx = pd.read_parquet(INDEX, columns=["stem", "corpora"])
    tuab = set(idx.loc[idx.corpora.str.contains("tuab"), "stem"])
    rows = []
    # 1) statistics of the per-fold MAE over fold models
    sets = {"TUAB test (official)": (d.split == "heldout") & (d.source == "TUAB"),
            "TUEG test (held-out folds)": (d.split == "test") & (d.source == "TUEG")}
    for name, mask in sets.items():
        g = d[mask].groupby(["exp", "cohort"])["MAE"]
        for (exp, cohort), s in g:
            rows.append({"exp": exp, "set": name, "cohort": cohort, "kind": "over fold models",
                         "folds": len(s), "median": s.median(), "mean": s.mean(),
                         "sd": s.std(), "min": s.min(), "max": s.max()})
    # 2) median / mean ensemble on the official TUAB test, 3) TUEG out-of-fold
    for exp, folds in preds.items():
        held = [f["heldout"] for f in folds.values() if "heldout" in f]
        if held:
            stems = sorted(set.intersection(*(set(h) for h in held)))
            P = np.array([[h[s][0] for s in stems] for h in held])
            t = np.array([held[0][s][1] for s in stems])
            for agg, p in (("median", np.median(P, 0)), ("mean", P.mean(0))):
                for cohort, m in by_cohort(stems, p, t, labels).items():
                    rows.append({"exp": exp, "set": "TUAB test (official)", "cohort": cohort,
                                 "kind": f"{agg} over fold models (ensemble)", "folds": len(held), **m})
        oof = {}
        for f in folds.values():
            for s, (p, t) in f.get("test", {}).items():
                if s not in tuab:
                    oof[s] = (p, t)
        if oof:
            stems = sorted(oof)
            p = np.array([oof[s][0] for s in stems])
            t = np.array([oof[s][1] for s in stems])
            for cohort, m in by_cohort(stems, p, t, labels).items():
                rows.append({"exp": exp, "set": "TUEG test (held-out folds)", "cohort": cohort,
                             "kind": "pooled out-of-fold", "folds": len(folds), **m})
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
