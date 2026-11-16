#!/usr/bin/env python
"""Per-recording choice between an in-domain and a device-robust age model (label-free).

A recording looks like the training data when its 45-70 Hz power share and its
amplitude (``scripts/recording_amplitude_features.py stats``) lie inside the
[q, 1-q] quantiles of the *training split* of the training cohort; no age and no
external cohort enters the rule.

* ``hard``: in-domain -> model A (e.g. E2L), else model B (device-robust);
* ``soft``: w = sigmoid(m_hf / s) * sigmoid(m_amp / s) with m = the signed margin
  (log10 units) inside the in-domain range and s = ``--soft-scale``; the
  prediction is w * A + (1 - w) * B;
* ``mean``: (A + B) / 2.

Prediction files are ``scripts/eval_age_by_cohort.py`` outputs.

    python scripts/gated_ensemble.py --stats rec_stats.parquet --train-cohort .../tuh_heedb_v1 \\
        --set "Temple+Harvard val=val:A.json:E2L:B.json:robust" ... --out gated.csv
"""
import argparse
import json

import numpy as np
import pandas as pd


def load_pred(path, key, split):
    return pd.Series(json.load(open(path))[key]["splits"][split]["predictions"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", required=True)
    ap.add_argument("--train-cohort", required=True)
    ap.add_argument("--set", action="append", required=True,
                    metavar="NAME=SPLIT:A_JSON:A_KEY:B_JSON:B_KEY[:AGES_JSON]")
    ap.add_argument("--ages", action="append", default=[], help="age_metadata.json files")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--soft-scale", type=float, default=0.05)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    st = pd.read_parquet(args.stats).drop_duplicates("stem").set_index("stem")
    st["log_hf"] = np.log10(st["hf_ratio"].clip(lower=1e-6))
    st["log_amp"] = np.log10(st["amp_uv"].clip(lower=1e-3))
    tr = st[(st.cohort == args.train_cohort) & (st.split == "train")]
    lo = {c: tr[c].quantile(args.q) for c in ("log_hf", "log_amp")}
    hi = {c: tr[c].quantile(1 - args.q) for c in ("log_hf", "log_amp")}
    print("in-domain range (training split):",
          {c: (round(lo[c], 3), round(hi[c], 3)) for c in lo})
    ages = {}
    for path in args.ages:
        ages.update({s: r["age"] for s, r in json.load(open(path))["recordings"].items()})

    rows = []
    for item in args.set:
        name, spec = item.split("=", 1)
        split, a_json, a_key, b_json, b_key = spec.split(":")[:5]
        a, b = load_pred(a_json, a_key, split), load_pred(b_json, b_key, split)
        stems = a.index.intersection(b.index).intersection(st.index)
        y = np.array([ages[s] for s in stems], float)
        keep = y >= args.min_age
        stems, y = stems[keep], y[keep]
        pa, pb, s = a[stems].to_numpy(), b[stems].to_numpy(), st.loc[stems]
        inside = np.ones(len(stems), bool)
        margin = np.full(len(stems), np.inf)
        for c in ("log_hf", "log_amp"):
            inside &= (s[c] >= lo[c]).to_numpy() & (s[c] <= hi[c]).to_numpy()
            margin = np.minimum(margin, np.minimum(s[c] - lo[c], hi[c] - s[c]).to_numpy())
        w = 1 / (1 + np.exp(-margin / args.soft_scale))
        preds = {"A": pa, "B": pb, "mean": (pa + pb) / 2, "hard": np.where(inside, pa, pb),
                 "soft": w * pa + (1 - w) * pb}
        for method, p in preds.items():
            e = p - y
            rows.append({"set": name, "method": method, "n": len(y), "share_to_B": float((~inside).mean()),
                         "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                         "r": float(np.corrcoef(p, y)[0, 1])})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["set", "n", "share_to_B"], columns="method", values="mae", sort=False)
          .round(2)[["A", "B", "mean", "hard", "soft"]].to_string())
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
