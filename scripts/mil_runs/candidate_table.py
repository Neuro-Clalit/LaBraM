#!/usr/bin/env python
"""Per-cohort comparison of feature-candidate runs with a baseline run (all scored by
``scripts/mil_runs/eval_candidates.sh``): recording MAE of each run and the paired change against
the baseline with a 95 % interval from resampling people (patients for Temple / Harvard, subjects
for the external cohorts; 2,000 draws). Temple / Harvard test rows are split by report status,
TDBRAIN is restricted to adults and split healthy / with an indication. Aggregates only.

    python scripts/mil_runs/candidate_table.py --root /data/runs/heedb/feature_candidates/runs \\
        --base P4F_L1 --runs P4F_SLOW_L1 P4F_GREEN_L1 --out table.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

E = "/data/datasets/EEG-public"
OWN = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
COHORTS = {"own": OWN, "dortmund": f"{E}/DORTMUND/cohorts/dortmund_v1", "tdbrain": f"{E}/TDBRAIN/cohorts/tdbrain_v1",
           "srm": f"{E}/SRM/cohorts/srm_v1", "lemon": f"{E}/LEMON/cohorts/lemon_v1", "chbp": f"{E}/CHBP/cohorts/chbp_v1"}


def boot(d, groups, n=2000, seed=0):
    idx = pd.factorize(groups)[0]
    s, c = np.bincount(idx, weights=d), np.bincount(idx).astype(float)
    draw = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    m = s[draw].sum(1) / c[draw].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def predictions(root, run, key):
    path = os.path.join(root, run, "eval", f"{key}.json")
    d = json.load(open(path))
    return d[next(iter(d))]["splits"]["test"]["predictions"]


def cells(key, stems):
    """``[(cohort label, subset label, mask)]`` and the person id of every stem."""
    meta = json.load(open(os.path.join(COHORTS[key], "age_metadata.json")))["recordings"]
    lab = np.array([meta[s].get("label") or "no verdict" for s in stems])
    age = np.array([float(meta[s]["age"]) for s in stems])
    if key == "own":
        co = pd.read_parquet(os.path.join(OWN, "cohort.parquet")).set_index("stem").loc[stems]
        ds = np.where(co.corpus == "heedb", "Harvard", "Temple")
        person = np.where(co.corpus == "heedb", co.patient_id.astype(str), co.subject.astype(str))
        out = [("Temple + Harvard", "all", np.ones(len(stems), bool))]
        for d in ("Harvard", "Temple"):
            out.append((d, "all", ds == d))
            out += [(d, s.replace("unlabelled", "no verdict"), (ds == d) & (lab == s))
                    for s in ("normal", "abnormal", "unlabelled", "no verdict")]
        return [c for c in out if c[2].sum() > 0], person, age
    person = np.array([meta[s].get("subject") or s for s in stems])
    name = {"dortmund": "Dortmund", "tdbrain": "TDBRAIN adults", "srm": "SRM", "lemon": "LEMON", "chbp": "CHBP"}[key]
    if key == "tdbrain":
        adult = age >= 18
        return [(name, "all", adult), (name, "healthy", adult & (lab == "normal")),
                (name, "with an indication", adult & (lab != "normal"))], person, age
    return [(name, "healthy", np.ones(len(stems), bool))], person, age


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for key in COHORTS:
        base = predictions(args.root, args.base, key)
        runs = {r: predictions(args.root, r, key) for r in args.runs
                if os.path.exists(os.path.join(args.root, r, "eval", f"{key}.json"))}
        stems = sorted(set(base).intersection(*[set(p) for p in runs.values()]))
        sel, person, age = cells(key, stems)
        b = np.array([base[s] for s in stems])
        for cohort, subset, m in sel:
            eb = np.abs(b[m] - age[m])
            row = {"cohort": cohort, "subset": subset, "recordings": int(m.sum()), "people": len(set(person[m])),
                   args.base: float(eb.mean())}
            for r, p in runs.items():
                er = np.abs(np.array([p[s] for s in stems])[m] - age[m])
                lo, hi = boot(er - eb, person[m])
                row[r] = float(er.mean())
                row[f"{r} change"] = float((er - eb).mean())
                row[f"{r} lo"], row[f"{r} hi"] = lo, hi
            rows.append(row)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
