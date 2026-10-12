#!/usr/bin/env python
"""Break brain-age predictions down by data source (TUAB / TUEG) and cohort.

Reads the per-recording predictions written by ``scripts/eval_age_by_cohort.py``
(``--out``) and the true age and normal/abnormal label from the evaluated
data's ``age_metadata.json``; a recording counts as TUAB when it belongs to the
TUAB corpus (TUH index ``corpora``), else as TUEG. Per (split, source, cohort)
it reports n, MAE and RMSE (years) and R^2 over recordings:

    MAE = mean |pred - age|,   RMSE = sqrt(mean (pred - age)^2)
    R^2 = 1 - sum (pred - age)^2 / sum (age - mean age)^2

R^2 is computed within each group, so it depends on that group's age spread
(a narrow-age cohort gets a lower R^2 at the same MAE).

    python scripts/age_breakdown.py \\
        --eval exp1=/tmp/evals/exp1_own.json:/data/.../cohorts/normal_both_v1 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"


def breakdown(eval_json, data_path, tuab_stems, model=None):
    res = json.load(open(eval_json))
    model = model or next(iter(res))
    meta = json.load(open(os.path.join(data_path, "age_metadata.json")))["recordings"]
    rows = []
    for split, out in res[model]["splits"].items():
        for stem, pred in out["predictions"].items():
            m = meta[stem]
            rows.append({"split": split, "source": "TUAB" if stem in tuab_stems else "TUEG",
                         "cohort": m.get("label") or "unlabelled", "age": m["age"], "pred": pred})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (split, source), g in df.groupby(["split", "source"]):
        for cohort, sub in [("all", g)] + [(c, g[g.cohort == c]) for c in ("normal", "abnormal")]:
            if len(sub):
                err = sub.pred - sub.age
                out.append({"split": split, "source": source, "cohort": cohort, "n": len(sub),
                            "MAE": err.abs().mean(), "RMSE": float(np.sqrt((err ** 2).mean())),
                            "R2": 1 - float((err ** 2).sum() / ((sub.age - sub.age.mean()) ** 2).sum())})
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", required=True, metavar="NAME=EVAL_JSON:DATA_PATH")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--out", default=None, help="CSV of all rows")
    args = p.parse_args(argv)
    idx = pd.read_parquet(args.index, columns=["stem", "corpora"])
    tuab = set(idx.loc[idx.corpora.str.contains("tuab"), "stem"])
    tables = []
    for item in args.eval:
        name, rest = item.split("=", 1)
        path, data = rest.rsplit(":", 1)
        t = summarize(breakdown(path, data, tuab))
        t.insert(0, "model", name)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    sys.exit(main())
