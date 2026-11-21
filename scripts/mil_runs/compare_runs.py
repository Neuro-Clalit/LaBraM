#!/usr/bin/env python
"""Compare two models per test cohort from ``scripts/eval_age_by_cohort.py`` outputs.

Each ``--pair KEY:NEW.json:REF.json:COHORT`` names a cohort, the evaluation files holding the new model
(``--name``) and the reference model (``--ref``), scored with the same settings, and the cohort folder
(true ages from its ``age_metadata.json``). Per cohort and per
group of the new file (all / normal / abnormal / per dataset) the table gives n, both MAEs in years
(one prediction per recording, as the evaluation files hold them) and the paired change with a 95 %
interval from resampling recordings (2,000 draws).

    python scripts/mil_runs/compare_runs.py --name P4F_L1_f0 --ref P4_L1_f0 --out comparison.csv \\
        --pair own:run/eval/own.json:/data/runs/heedb/external_eval/own_5min_models.json:<cohort folder>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def paired(new, ref, age, n_boot=2000, seed=0):
    d = np.abs(new - age) - np.abs(ref - age)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pair", action="append", required=True, metavar="KEY:NEW.json:REF.json:COHORT")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for spec in args.pair:
        key, new_path, ref_path, cohort = spec.split(":")
        new, ref = json.load(open(new_path))[args.name], json.load(open(ref_path))[args.ref]
        if new["data"] != ref["data"]:
            raise SystemExit(f"{key}: evaluation settings differ: {new['data']} vs {ref['data']}")
        pn, pr = new["splits"]["test"]["predictions"], ref["splits"]["test"]["predictions"]
        stems = sorted(set(pn) & set(pr))
        a = np.array([pn[s] for s in stems])
        b = np.array([pr[s] for s in stems])
        meta = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
        age = np.array([float(meta[s]["age"]) for s in stems])
        d, lo, hi = paired(a, b, age)
        rows.append({"cohort": key, "n": len(stems), f"mae {args.ref}": float(np.abs(b - age).mean()),
                     f"mae {args.name}": float(np.abs(a - age).mean()), "change": d, "lo": lo, "hi": hi,
                     f"bias {args.ref}": float((b - age).mean()), f"bias {args.name}": float((a - age).mean()),
                     f"slope {args.ref}": float(np.polyfit(age, b, 1)[0]), f"slope {args.name}": float(np.polyfit(age, a, 1)[0])})
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
