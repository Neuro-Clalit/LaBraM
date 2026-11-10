#!/usr/bin/env python
"""Per-cohort MAE of the best brain-age models under four evaluation strategies.

For each model and cohort (HEEDB normal / abnormal / unlabelled, TUAB normal /
abnormal of the own test split, and LEMON young / old / all):

* ``5 min``   -- the default: first 5 minutes after the 60 s trim;
* ``all``     -- every window of the recording (``--eval-minutes 0``);
* ``trim20w`` -- all windows, minus the windows whose relative attention weight
  is below the 20th percentile of the split (label-free), re-pooled
  (``scripts/attention_trimming.py``, ``trim20_glob``);
* ``keep80r`` -- all windows, MAE of the 80 % of recordings kept after setting
  aside the 20 % with the largest attention-weighted spread of the window
  predictions (selective prediction; the other 20 % get no prediction).

Window-level models (E2, E2L) have no attention: their trim / reject cells are
empty. Inputs are ``scripts/eval_age_by_cohort.py`` JSONs (per-recording
predictions) and ``--save-windows`` parquets.

    python scripts/eval_strategies_table.py --root /data/runs/heedb/external_eval \\
        --out docs/data/external_eval/strategies_by_cohort.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import attention_trimming as at  # noqa: E402

MODELS = ["E2", "E2L", "W4", "L1", "W7", "X1", "X3"]
# cohort -> (5-min JSONs, all-window JSONs, all-window windows dir)
SOURCES = {
    "own": (["own_5min_w.json", "own_5min_e2.json"], ["own_all_w.json", "own_all_e2.json"], "own_all_w"),
    "lemon": (["lemon_5min_w.json", "lemon_5min.json"], ["lemon_all_w.json", "lemon_all.json"], "lemon_all_w"),
}


def predictions(root, files):
    out = {}
    for f in files:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        for model, r in json.load(open(p)).items():
            out.setdefault(model, pd.Series(r["splits"]["test"]["predictions"]))
    return out


def group_masks(cohort, idx, ages, meta):
    if cohort == "lemon":
        a = ages.reindex(idx)
        return [("LEMON all", np.ones(len(idx), bool)), ("LEMON young", (a < 40).to_numpy()),
                ("LEMON old", (a >= 55).to_numpy())]
    m = meta.reindex(idx)
    out = []
    for src, label in (("HEEDB", "Harvard"), ("TUAB", "Temple (TUAB)")):
        for c in ("normal", "abnormal", "unlabelled"):
            mk = ((m["source"] == src) & (m["cohort"] == c)).to_numpy()
            if mk.sum() >= 10:
                out.append((f"{label} {c}", mk))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import mil_cohort_summary as mcs
    meta = mcs.recording_table()
    rows = []
    for cohort, (f5, fall, wdir) in SOURCES.items():
        p5, pall = predictions(args.root, f5), predictions(args.root, fall)
        recs = {}
        for model in MODELS:
            wp = os.path.join(args.root, "windows", wdir, f"{model}_test.parquet")
            if os.path.exists(wp):
                w = pd.read_parquet(wp)
                rf = at.recording_frame(w)
                rf["trim20_glob"] = at.trims(w)["trim20_glob"]
                recs[model] = rf
        ref = next((r for r in recs.values()), None)
        if ref is None:
            continue
        ages = ref["age"]
        for model in MODELS:
            for gname, mask in group_masks(cohort, ages.index, ages, meta):
                idx = ages.index[mask]
                row = {"group": gname, "model": model, "n": int(mask.sum())}
                for key, src in (("5 min", p5), ("all", pall)):
                    if model in src:
                        p = src[model].reindex(idx)
                        row[key] = float(np.abs(p - ages.reindex(idx)).mean())
                if model in recs:
                    r = recs[model].reindex(idx)
                    row["trim20w"] = float(np.abs(r["trim20_glob"] - r["age"]).mean())
                    err = (r["attention"] - r["age"]).to_numpy()
                    row["keep80r"] = at.mae_kept(err, -r["spread"].to_numpy())
                rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    for g, d in out.groupby("group", sort=False):
        print(f"\n{g} (n={d['n'].iloc[0]})")
        print(d.set_index("model").reindex(MODELS)[[c for c in ("5 min", "all", "trim20w", "keep80r")
                                                     if c in d]].round(2).to_string())


if __name__ == "__main__":
    main()
