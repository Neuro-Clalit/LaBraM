#!/usr/bin/env python
"""Per-cohort brain-age table: all / healthy / pathological, with and without the
label-free 20 % rejection by window-prediction spread (Table 12.1 of
``docs/tdbrain_onboarding.md``, for any external cohort).

Inputs per cohort: the cohort directory (manifest with ages), an
``eval_age_by_cohort.py`` output holding the models and their ensembles
(``ensemble_predictions.py``) and the ``--save-windows`` directory of the
member models. The scoring unit is the session (mean of its files, TDBRAIN's
eyes-closed and eyes-open) when the manifest has a ``task`` column, else the
recording. Only recordings aged >= ``--min-age`` (default 18) with an age of
their own session (``age_session_matched`` where present) are scored.

Healthy: the manifest's ``healthy`` flag (TDBRAIN), else ``dx_status == 0``
(MIPDB controls), else ``label == "normal"`` (LEMON, CHBP: healthy-volunteer
corpora). Pathological: every other scored recording.

Metrics and rejection as in ``tdbrain_brain_age.py``: MAE with a 95 %
subject-bootstrap interval, R^2 about the group's own mean, Pearson r; the
rejection sets aside the ``--reject`` share of units with the largest
attention-weighted spread of window predictions (ensemble: mean of its
members' spreads), the threshold taken over all units, then split by group.

    python scripts/external_cohort_table.py \\
        --set lemon:/data/datasets/EEG-public/LEMON/cohorts/lemon_v1:/data/runs/heedb/external_eval/lemon_all_models.json:/data/runs/heedb/external_eval/windows/lemon_all_models \\
        --models E2L P3ens P4ens --out docs/data/external_eval/tdbrain/external_cohort_table.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdbrain_brain_age import file_spread, metrics  # noqa: E402


def healthy_flag(meta):
    if "healthy" in meta and meta["healthy"].notna().any():
        return meta["healthy"].fillna(False).astype(bool)
    if "dx_status" in meta and meta["dx_status"].notna().any():
        return meta["dx_status"] == 0
    return meta["label"] == "normal"


def units(meta, pred, spread):
    d = meta[["subject", "age", "healthy"] + (["session", "task"] if "task" in meta else [])].join(
        pred.rename("pred"), how="inner")
    if spread is not None:
        d = d.join(spread.rename("spread"), how="left")
    if "task" not in meta:
        return d.reset_index(drop=True)
    agg = {"age": "first", "healthy": "first", "pred": "mean"}
    if spread is not None:
        agg["spread"] = "mean"
    return d.groupby(["subject", "session"]).agg(agg).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--reject", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.set:
        name, cdir, ev, wdir = spec.split(":", 3)
        meta = pd.DataFrame(json.load(open(os.path.join(cdir, "manifest.json")))["recordings"]).set_index("stem")
        if "age_session_matched" in meta:
            meta = meta[meta["age_session_matched"].fillna(True).astype(bool)]
        meta = meta[meta["age"] >= args.min_age].copy()
        meta["healthy"] = healthy_flag(meta)
        res = json.load(open(ev))
        for model in args.models:
            pred = pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float)
            spread = file_spread(wdir, res[model].get("members", [model]))
            d = units(meta, pred, spread)
            variants = [("none", d)]
            if spread is not None:
                variants.append((f"reject {args.reject:.0%}", d[d["spread"] <= d["spread"].quantile(1 - args.reject)]))
            for rej, v in variants:
                for group, g in (("all", v), ("healthy", v[v.healthy]), ("pathological", v[~v.healthy])):
                    full = {"all": d, "healthy": d[d.healthy], "pathological": d[~d.healthy]}[group]
                    row = {"cohort": name, "model": model, "rejection": rej, "group": group, "n": len(g),
                           "n_subjects": g["subject"].nunique(), "kept_share": len(g) / len(full) if len(full) else np.nan}
                    if len(g) >= 10:
                        row.update(metrics(g))
                    rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["cohort", "model", "rejection", "group", "n", "n_subjects", "age_mean", "mae", "mae_lo", "mae_hi", "bias", "r2", "r"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df[[c for c in cols if c in df]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
