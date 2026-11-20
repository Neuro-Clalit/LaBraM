#!/usr/bin/env python
"""Accuracy of any set of models per cohort and subset, no rejection (Table 9 of
``docs/tech_report/``). The same units, subsets and metrics as
``external_cohort_table.py`` (sessions for TDBRAIN / Dortmund; healthy /
pathological / unlabelled; MAE with a 95 % subject-bootstrap interval, bias, r,
R^2), but a set may take its predictions from several evaluation JSONs, so models
scored in separate runs share one table.

    python scripts/report_model_table.py --models E2L L1 W4 P3ens P4ens \\
        --set temple:<cohort>:<a.json>,<b.json>:temple ... --out <csv>
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_cohort_table import load_meta, units  # noqa: E402
from tdbrain_brain_age import metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json[,EVAL.json...][:FAMILY]")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.set:
        name, cdir, evs, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        res = {}
        for ev in evs.split(","):
            res.update(json.load(open(ev)))
        for model in args.models:
            if model not in res:
                print(f"{name}: {model} not scored", flush=True)
                continue
            d = units(meta, pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float), None)
            unl = ~(d.healthy | d.pathological)
            groups = [("all", d), ("healthy", d[d.healthy]), ("pathological", d[d.pathological])]
            if unl.any():
                groups.append(("unlabelled", d[unl]))
            for group, g in groups:
                if len(g) >= 10:
                    rows.append({"cohort": name, "model": model, "group": group, **metrics(g)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.pivot_table(index=["cohort", "group"], columns="model", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
