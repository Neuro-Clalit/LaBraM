#!/usr/bin/env python
"""Brain-age MAE / bias / r of several models, split by data cohort and report.

Rows: Temple (TUH) and Harvard (HEEDB) val and test, each overall and by report
(normal / abnormal / unlabelled), then every external test cohort (CHBP also by
cap). Each row is given for all ages and for adults (age >= 18). Predictions
are ``scripts/eval_age_by_cohort.py`` outputs (one prediction per recording).

    python scripts/cohort_breakdown.py --own-cohort .../tuh_heedb_v1 \\
        --external chbp=.../chbp_v1 --external lemon=... \\
        --model "E2L|own=a.json:E2L|chbp=b.json:E2L|lemon=c.json:E2L" ... --out breakdown.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HARVARD = {"heedb"}


def load(spec):
    path, key = spec.rsplit(":", 1)
    return json.load(open(path))[key]["splits"]


def own_meta(cohort):
    ages = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
    man = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    rows = []
    for stem, r in ages.items():
        corpus = man.get(stem, {}).get("corpus", "")
        rows.append({"stem": stem, "age": r["age"], "report": r.get("label") or "unlabelled",
                     "dataset": "Harvard" if corpus in HARVARD else "Temple"})
    return pd.DataFrame(rows).set_index("stem")


def ext_meta(cohort, name, caps=None):
    ages = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
    df = pd.DataFrame({"age": {s: r["age"] for s, r in ages.items()}})
    df["dataset"] = name
    if caps is not None:
        df["cap"] = df.index.map(caps).map({"12-bit": "64-channel cap", "16-bit": "120-channel cap"})
    return df


def metrics(y, p):
    e = p - y
    return {"n": len(y), "mae": np.abs(e).mean(), "bias": e.mean(),
            "r": np.corrcoef(y, p)[0, 1] if len(y) > 2 else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--own-cohort", required=True)
    ap.add_argument("--external", action="append", default=[], metavar="NAME=COHORT")
    ap.add_argument("--chbp-encoding", default=None, help="stem,enc csv: CHBP rows also by cap")
    ap.add_argument("--model", action="append", required=True,
                    metavar="LABEL|own=JSON:KEY|NAME=JSON:KEY...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    own = own_meta(args.own_cohort)
    caps = (pd.read_csv(args.chbp_encoding).set_index("stem")["enc"].to_dict()
            if args.chbp_encoding else None)
    ext = {n: ext_meta(c, n, caps if n == "chbp" else None)
           for n, c in (e.split("=", 1) for e in args.external)}

    rows = []
    for item in args.model:
        label, *parts = item.split("|")
        for part in parts:
            name, spec = part.split("=", 1)
            splits = load(spec)
            if name == "own":
                for split in ("val", "test"):
                    pred = pd.Series(splits[split]["predictions"])
                    d = own.loc[own.index.intersection(pred.index)].copy()
                    d["pred"] = pred[d.index]
                    groups = [(f"{ds} {split}", d[d.dataset == ds]) for ds in ("Temple", "Harvard")]
                    groups += [(f"{ds} {split}, {rep}", d[(d.dataset == ds) & (d.report == rep)])
                               for ds in ("Temple", "Harvard") for rep in ("normal", "abnormal", "unlabelled")]
                    for g, sub in groups:
                        rows += [{"model": label, "set": g, "ages": ages,
                                  **metrics(s.age.to_numpy(float), s.pred.to_numpy(float))}
                                 for ages, s in (("all", sub), ("adults", sub[sub.age >= 18])) if len(s)]
            else:
                pred = pd.Series(splits["test"]["predictions"])
                d = ext[name].loc[ext[name].index.intersection(pred.index)].copy()
                d["pred"] = pred[d.index]
                groups = [(name, d)]
                if "cap" in d:
                    groups += [(f"{name}, {c}", d[d.cap == c]) for c in sorted(d.cap.dropna().unique())]
                for g, sub in groups:
                    rows += [{"model": label, "set": g, "ages": ages,
                              **metrics(s.age.to_numpy(float), s.pred.to_numpy(float))}
                             for ages, s in (("all", sub), ("adults", sub[sub.age >= 18])) if len(s)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    for ages in ("all", "adults"):
        t = df[df.ages == ages]
        cell = t.assign(c=t.mae.map("{:.2f}".format) + " / " + t.bias.map("{:+.1f}".format)
                        + " / " + t.r.map("{:.2f}".format))
        piv = cell.pivot_table(index="set", columns="model", values="c", aggfunc="first", sort=False)
        piv.insert(0, "n", t.groupby("set", sort=False).n.first())
        print(f"\n== {ages} ==\n" + piv.to_string())


if __name__ == "__main__":
    main()
